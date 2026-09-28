package host

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	res "github.com/openabstractions/abstraction-resource/go/client"
)

// The words a refusal takes. They are the whole vocabulary: a call that is not
// answered by the engine reads one of these and nothing else. Each of
// NoEngine, NotPermitted, Insufficient and HoldersRefused carries the reason
// family research/vocabulary/DECISION.md D87 decided, so an inference caller
// reading this provider's reason can tell a lease outcome from an engine
// outcome from an unknown object without parsing detail text
// (CONTRACT.md MODELHOST-L3, MODELHOST-H2).
const (
	// NoEngine: no llama.cpp is configured, or the configured path is not on
	// this machine. Nothing is downloaded and nothing is looked for.
	NoEngine = "engine:none"
	// NotPermitted: no rule lets this provider's program hold the resource.
	NotPermitted = "lease:not_permitted"
	// Insufficient: available capacity is below the file-size admission
	// estimate and no holder could make up the difference.
	Insufficient = "lease:insufficient"
	// HoldersRefused: the holders were asked and did not free enough.
	HoldersRefused = "lease:holders_refused"
	// Unavailable: the object, the lease service or the engine could not be
	// reached, or the engine did not become ready.
	Unavailable = "unavailable"
)

// unknownObject is the reason word for an allowlist entry naming a store and
// an id no source resolves, the one seam abstraction.storage/lend@1 also
// resolves through (D87; CONTRACT.md MODELHOST-L1).
func unknownObject(store, object string) string {
	return fmt.Sprintf("object:unknown:%s/%s", store, object)
}

// A Refusal is one of those words with the detail that produced it.
type Refusal struct {
	Word   string
	Detail string
}

func (r *Refusal) Error() string {
	if r.Detail == "" {
		return r.Word
	}
	return r.Word + ": " + r.Detail
}

func refuse(word, format string, args ...any) *Refusal {
	return &Refusal{Word: word, Detail: fmt.Sprintf(format, args...)}
}

// WordOf is the refusal word an error carries, or the empty string.
func WordOf(err error) string {
	var r *Refusal
	if errors.As(err, &r) {
		return r.Word
	}
	return ""
}

// A Model is one entry of the allowlist: the name callers ask for, and the
// object the storage inventory holds it as.
type Model struct {
	Name   string
	Store  string
	Object string
}

// Leases is the part of abstraction.resource/leases@1 this provider uses. The
// runtime serves it; this provider is an ordinary lease holder that observes
// its yield requests and answers them (CONTRACT.md RES-L3).
type Leases interface {
	Acquire(resource string, amount int64, wait time.Duration) (res.AcquireResult, error)
	BindProcess(lease string, pid int) (res.ProcessBindResult, error)
	Renew(lease string) (res.LeaseChange, error)
	Release(lease string) (res.LeaseChange, error)
	Observe(ctx context.Context, cursor string, wait time.Duration) (res.YieldPage, error)
	Answer(ctx context.Context, request string, answer res.YieldAnswer, reason string) (res.AnswerResult, error)
}

// Options compose a Host.
type Options struct {
	// Models is the explicit allowlist. A name outside it is never loaded.
	Models []Model
	// Engine is the llama.cpp this machine already has.
	Engine Engine
	// Source resolves a model's store and id to a file.
	Source Source
	// Leases is the lease service; without one nothing is ever loaded.
	Leases Leases
	// Resource is the resource each resident's lease is held on; default
	// card:0. Every resident shares the one resource kind and takes its own
	// lease of it.
	Resource string
	// Port is where the engine answers when the allowlist names exactly one
	// model. Zero takes a free one. It is ignored, and a free port is taken
	// for every model, when the allowlist names more than one: two engine
	// processes never share a port.
	Port int
	// Idle unloads a resident nothing has asked for; default five minutes.
	// Each resident keeps its own idle clock.
	Idle time.Duration
	// Active reports whether an operation for the named model is still active
	// or retained for observation. Nil means the idle clock only ever asks
	// this host's own last-used time. Set it to the inference provider's own
	// tracking (Provider.ActiveForModel) so a call already admitted keeps its
	// model loaded for as long as its operation is open, without this host
	// touching anything on every read (CONTRACT.md MODELHOST-H6). It is asked with
	// each resident's own name, so an operation open for one object never
	// keeps a different, idle one loaded.
	Active func(model string) bool
	// Renew is how often a live lease is renewed; default eight seconds.
	Renew time.Duration
	// Wait is how long holders have to yield for one load; default 60s.
	Wait time.Duration
	// HTTP reaches the engine; nil uses a client with no total timeout.
	HTTP *http.Client
	// Now is the clock; nil uses time.Now.
	Now func() time.Time
	// OnError hears what the background loops could not do.
	OnError func(error)
}

// DefaultResource is the accelerator memory a model's residency is held on.
const DefaultResource = "card:0"

// Defaults for the bounds Options leaves at zero.
const (
	DefaultIdle  = 5 * time.Minute
	DefaultRenew = 8 * time.Second
	DefaultWait  = 60 * time.Second
	// stopWithin bounds waiting for a stopped engine to go.
	stopWithin = 20 * time.Second
	// observeWait is one Observe call's wait, inside the contract's bound.
	observeWait = 25 * time.Second
)

// resident is one object this host currently holds: its own engine, its own
// lease, and its own last-used clock.
type resident struct {
	model   string
	process *Process
	lease   string
	amount  int64
	last    time.Time
}

// A Host is one engine process per resident object, each held under its own
// lease and shared by every caller the runtime forwards. An object on the
// allowlist loads on the first call that asks for it and stays resident
// beside whatever else is already loaded (CONTRACT.md MODELHOST-H5): asking for a
// second model does not displace the first.
type Host struct {
	options Options
	allow   map[string]Model
	names   []string
	client  *http.Client
	bases   map[string]string
	ports   map[string]int

	mu        sync.Mutex
	residents map[string]*resident

	closeOnce sync.Once
	closed    chan struct{}
}

// New composes a Host. It starts no engine and takes no lease: the first call
// for each model does both. Every name on the allowlist gets its own reserved
// loopback port before any engine runs, so Base never changes once a caller
// has read it.
func New(options Options) (*Host, error) {
	if options.Source == nil {
		return nil, errors.New("modelhost: a source is required")
	}
	if options.Leases == nil {
		return nil, errors.New("modelhost: the lease service is required")
	}
	if len(options.Models) == 0 {
		return nil, errors.New("modelhost: an explicit model allowlist is required")
	}
	if options.Resource == "" {
		options.Resource = DefaultResource
	}
	if options.Idle <= 0 {
		options.Idle = DefaultIdle
	}
	if options.Renew <= 0 {
		options.Renew = DefaultRenew
	}
	if options.Wait <= 0 {
		options.Wait = DefaultWait
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	h := &Host{options: options, allow: map[string]Model{}, client: options.HTTP,
		bases: map[string]string{}, ports: map[string]int{}, residents: map[string]*resident{},
		closed: make(chan struct{})}
	if h.client == nil {
		h.client = &http.Client{}
	}
	for _, m := range options.Models {
		if m.Name == "" || m.Store == "" || m.Object == "" {
			return nil, fmt.Errorf("modelhost: model %q needs a name, a store and an object", m.Name)
		}
		h.allow[m.Name] = m
		h.names = append(h.names, m.Name)
	}
	sort.Strings(h.names)
	for _, name := range h.names {
		port := options.Port
		if port == 0 || len(h.names) > 1 {
			p, err := FreePort()
			if err != nil {
				return nil, err
			}
			port = p
		}
		h.ports[name] = port
		h.bases[name] = fmt.Sprintf("http://127.0.0.1:%d", port)
	}
	return h, nil
}

// Models is the allowlist, sorted. It is what the declaration names and what
// the router lists; the engine is never asked what it serves.
func (h *Host) Models() []string { return append([]string(nil), h.names...) }

// Base is where a mediated call reaches name's own engine. It is reserved
// when the host is composed and never changes with a load, so no call is
// ever sent to a port a different model's engine answers on. It reads empty
// for a name outside the allowlist.
func (h *Host) Base(name string) string { return h.bases[name] }

// Serves reports whether name is on the allowlist.
func (h *Host) Serves(name string) bool { _, ok := h.allow[name]; return ok }

// Residents is the model names this host holds right now, sorted.
func (h *Host) Residents() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	names := make([]string, 0, len(h.residents))
	for name := range h.residents {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Loaded reports whether name is resident right now.
func (h *Host) Loaded(name string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	_, ok := h.residents[name]
	return ok
}

// Lease is the lease name's residency is held under, or empty when name is
// not resident.
func (h *Host) Lease(name string) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	if r, ok := h.residents[name]; ok {
		return r.lease
	}
	return ""
}

// Ensure makes name resident and reports where its engine answers. The first
// call for an object takes its lease, starts its own engine and waits for it;
// every call after that finds it resident and only marks it used. Two callers
// arriving together load it once, because this is the one place a load
// happens. A call for a different, already-resident object neither waits for
// nor displaces it: the second object gets its own engine on its own port
// beside the first (CONTRACT.md MODELHOST-H5).
func (h *Host) Ensure(ctx context.Context, name string) (string, error) {
	model, ok := h.allow[name]
	if !ok {
		return "", refuse(Unavailable, "no model called %q is hosted here", name)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	select {
	case <-h.closed:
		return "", refuse(Unavailable, "this provider is stopping")
	default:
	}
	if r, ok := h.residents[name]; ok {
		if h.aliveLocked(r) {
			r.last = h.options.Now()
			return h.bases[name], nil
		}
		h.unloadLocked(name, "the engine exited")
	}

	object, found := h.options.Source.Resolve(model.Store, model.Object)
	if !found {
		return "", refuse(unknownObject(model.Store, model.Object), "no store called %s holds %s", model.Store, model.Object)
	}
	if err := h.options.Engine.present(); err != nil {
		return "", refuse(NoEngine, "%v", err)
	}
	amount := object.Size
	if amount <= 0 {
		return "", refuse(Unavailable, "the source reports no size for %s/%s", model.Store, model.Object)
	}
	result, err := h.options.Leases.Acquire(h.options.Resource, amount, h.options.Wait)
	if err != nil {
		return "", refuse(Unavailable, "the lease service did not answer: %v", err)
	}
	switch result.Outcome {
	case res.AcquireOutcomeAcquired:
	case res.AcquireOutcomeNotPermitted:
		return "", refuse(NotPermitted, "no rule lets this provider hold %s", h.options.Resource)
	case res.AcquireOutcomeInsufficient:
		return "", refuse(Insufficient, "%s has less than %s's %d-byte file-size admission estimate", h.options.Resource, name, amount)
	case res.AcquireOutcomeHoldersRefused:
		return "", refuse(HoldersRefused, "%s", asked(result))
	default:
		return "", refuse(Unavailable, "the lease service answered %s", result.Outcome)
	}
	lease := ""
	if result.Lease != nil {
		lease = result.Lease.ID
	}
	release := func() {
		if lease != "" {
			//unchecked: best-effort lease release on a cleanup path; release has no return value to report a failure through
			h.options.Leases.Release(lease)
		}
	}
	process, err := h.options.Engine.start(object.Path, name, h.ports[name])
	if err != nil {
		release()
		return "", refuse(Unavailable, "the engine would not start: %v", err)
	}
	if lease != "" {
		binding, err := h.options.Leases.BindProcess(lease, process.PID())
		if err != nil {
			h.report(fmt.Errorf("modelhost: binding %s's engine to %s: %w", name, lease, err))
		} else if binding.Outcome != res.ProcessBindOutcomeBound {
			h.report(fmt.Errorf("modelhost: engine for %s was not attributed to %s: %s", name, lease, binding.Outcome))
		}
	}
	if err := process.wait(ctx, h.options.Engine.Ready, h.client); err != nil {
		process.stop(stopWithin)
		release()
		return "", refuse(Unavailable, "%v", err)
	}
	h.residents[name] = &resident{model: name, process: process, lease: lease, amount: amount, last: h.options.Now()}
	return h.bases[name], nil
}

// aliveLocked reports whether name's engine is still running. The caller
// holds the lock.
func (h *Host) aliveLocked(r *resident) bool {
	select {
	case <-r.process.exited:
		return false
	default:
		return true
	}
}

// Unload stops name's engine and releases its lease, naming the object that
// lets go. It is what a yield request and the idle timer both do; unloading a
// name that is not resident does nothing.
func (h *Host) Unload(name, why string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.unloadLocked(name, why)
}

// unloadLocked gives name's model and lease back, in that order: the engine
// must exit before the service is told the grant is over. It returns whether
// the resident stopped; a failed release is reported and the book can end its
// still-live grant after confirming the yield.
func (h *Host) unloadLocked(name, why string) bool {
	r, ok := h.residents[name]
	if !ok {
		return false
	}
	r.process.stop(stopWithin)
	if h.aliveLocked(r) {
		h.report(fmt.Errorf("modelhost: %s's engine did not stop after %s", name, why))
		return false
	}
	delete(h.residents, name)
	if r.lease != "" {
		if _, err := h.options.Leases.Release(r.lease); err != nil {
			h.report(fmt.Errorf("modelhost: releasing %s's %s after %s: %w", name, r.lease, why, err))
		}
	}
	return true
}

// Run keeps every resident's lease alive, gives residents back when they are
// asked for, and unloads one nothing has used for its own idle time. It
// returns when ctx ends.
func (h *Host) Run(ctx context.Context) {
	var group sync.WaitGroup
	group.Add(2)
	go func() { defer group.Done(); h.keep(ctx) }()
	go func() { defer group.Done(); h.observe(ctx) }()
	group.Wait()
}

// keep renews every resident's live lease and unloads whichever ones its own
// idle clock has run out on.
func (h *Host) keep(ctx context.Context) {
	ticker := time.NewTicker(h.options.Renew)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-h.closed:
			return
		case <-ticker.C:
		}
		for _, name := range h.Residents() {
			h.tick(name)
		}
	}
}

// tick renews or unloads one resident, on its own idle clock.
func (h *Host) tick(name string) {
	h.mu.Lock()
	r, ok := h.residents[name]
	if !ok {
		h.mu.Unlock()
		return
	}
	if !h.aliveLocked(r) {
		h.unloadLocked(name, "idle")
		h.mu.Unlock()
		return
	}
	if h.options.Now().Sub(r.last) >= h.options.Idle {
		// An operation still open or retained for that model is a model
		// still in use, whatever this host's own last-used time reads.
		if h.options.Active != nil && h.options.Active(r.model) {
			r.last = h.options.Now()
		} else {
			h.unloadLocked(name, "idle")
			h.mu.Unlock()
			return
		}
	}
	lease := r.lease
	h.mu.Unlock()
	if lease == "" {
		return
	}
	change, err := h.options.Leases.Renew(lease)
	if err != nil {
		h.report(fmt.Errorf("modelhost: renewing %s's %s: %w", name, lease, err))
		return
	}
	if !change.Applied {
		// A lease the book no longer holds is a residency nobody granted.
		h.Unload(name, "the lease ended")
	}
}

// observe answers the yield requests addressed to this program. Each request
// asks for an amount of one resource, not a lease by name, so it is answered
// by unloading residents oldest-last-first — the one nothing has touched
// longest goes first — until at least that amount has been freed or nothing
// is left to give back (CONTRACT.md MODELHOST-H5, RES-L2's own idleness order for
// an attached holder). A resident already gone by the time it is asked for is
// freed already, so the answer is yielded either way.
func (h *Host) observe(ctx context.Context) {
	cursor := ""
	for {
		select {
		case <-ctx.Done():
			return
		case <-h.closed:
			return
		default:
		}
		page, err := h.options.Leases.Observe(ctx, cursor, observeWait)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			h.report(fmt.Errorf("modelhost: observing yield requests: %w", err))
			select {
			case <-ctx.Done():
				return
			case <-h.closed:
				return
			case <-time.After(time.Second):
			}
			continue
		}
		if page.Next != "" {
			cursor = page.Next
		}
		for _, request := range page.Requests {
			answer, reason := h.yield(request)
			if _, err := h.options.Leases.Answer(ctx, request.ID, answer, reason); err != nil {
				h.report(fmt.Errorf("modelhost: answering %s: %w", request.ID, err))
			}
		}
	}
}

// yield unloads exactly the resident whose lease the book asked for. A live
// operation keeps its engine until it ends, so an ask for that lease is refused.
// The book checks the instrument after a yielded answer.
func (h *Host) yield(request res.YieldRequest) (res.YieldAnswer, string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if request.Resource != h.options.Resource || request.Lease == "" {
		return res.YieldAnswerRefused, "resource or lease does not match this provider"
	}
	for name, r := range h.residents {
		if r.lease != request.Lease {
			continue
		}
		if r.amount != request.Amount {
			return res.YieldAnswerRefused, "amount does not match the lease"
		}
		if h.options.Active != nil && h.options.Active(r.model) {
			return res.YieldAnswerRefused, "model has an active request"
		}
		if !h.unloadLocked(name, "asked for "+request.Resource+" by "+request.Reason) {
			return res.YieldAnswerRefused, "engine did not stop"
		}
		return res.YieldAnswerYielded, ""
	}
	return res.YieldAnswerRefused, "lease is not resident"
}

// Close unloads every resident and stops the loops.
func (h *Host) Close() {
	h.closeOnce.Do(func() { close(h.closed) })
	for _, name := range h.Residents() {
		h.Unload(name, "the provider is stopping")
	}
}

func (h *Host) report(err error) {
	if h.options.OnError != nil && err != nil {
		h.options.OnError(err)
	}
}

// asked is the holders a refused acquire asked, for the refusal's detail.
func asked(result res.AcquireResult) string {
	if len(result.Asked) == 0 {
		return "no holder was asked"
	}
	parts := make([]string, 0, len(result.Asked))
	for _, row := range result.Asked {
		parts = append(parts, row.Holder+" "+row.Answer)
	}
	return strings.Join(parts, "; ")
}
