package host

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
	"github.com/openabstractions/abstraction-provider-kit/go/sdcpp"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	"github.com/openabstractions/abstraction-resource/go/holder"
)

const (
	NoEngine = refusal.NoEngine
	// EngineUnreadable is an engine folder with a file the provider could not
	// open or read, such as one a scanner holds; it passes once the file is
	// free (MODELHOST-L3).
	EngineUnreadable = "engine:unreadable"
	// The lease refusal words are the resource kit's (holder.Failure.Word).
	NotPermitted   = holder.WordNotPermitted
	Insufficient   = holder.WordInsufficient
	HoldersRefused = holder.WordHoldersRefused
	// LeaseEnded is a load whose lease the resource service ended while the
	// engine was starting, on the holder's one reload too (RES-L4).
	LeaseEnded  = holder.WordEnded
	Unavailable = refusal.Unavailable
)

func unknownObject(store, object string) string {
	return fmt.Sprintf("object:unknown:%s/%s", store, object)
}

// pickleObject is an object the source refuses to bind because it is a
// PyTorch pickle, which OA never loads (MODELHOST-L1).
func pickleObject(store, object string) string {
	return fmt.Sprintf("object:pickle:%s/%s", store, object)
}

// unboundObject is an object that exists in no binding the engine reads:
// a member of a zip, which llama-server cannot open (MODELHOST-L1).
func unboundObject(store, object string) string {
	return fmt.Sprintf("object:unbound:%s/%s", store, object)
}

type Model struct {
	Name, Store, Object string
	// Content is the explicit expected file pin carried by this allowlist entry.
	// Path is unused; the source resolves the object binding for every load.
	Content *programfiles.File
}
type Leases = holder.Leases

type Options struct {
	Models   []Model
	Engine   Engine
	Source   Source
	Leases   Leases
	Resource string
	Idle     time.Duration
	// Active includes pending admissions and retained operations. The
	// inference provider reserves activity before it calls Ensure.
	Active func(model string) bool
	// InFlight counts a model's admissions in progress and its calls still
	// running, without replies kept only for observation. With it, a yield
	// waits for the model's next request boundary and new admissions wait
	// behind it (resource RES-L3); without it, a yield waits for Active.
	InFlight    func(model string) int
	Renew, Wait time.Duration
	Now         func() time.Time
	OnError     func(error)
	// OnEngineExit, when set, receives every engine's exit record, of
	// every kind, once its process tree has ended (MODELHOST-E1).
	OnEngineExit func(EngineExit)
	// Video, when set, adds the video engine kind: one sd-server serving
	// video@1 for Video.Model (CONTRACT.md MODELHOST-V1..V9).
	Video *VideoOptions
	// GPU, when set, is the provider's GPU fault handling: it is asked
	// before every load, stops every engine on a driver reset, and hears
	// each engine's own device-lost line (resource CONTRACT.md RES-G1..G7,
	// MODELHOST-G1).
	GPU *gpufault.Guard
	// Capacity, when set, answers the bytes this machine's memory pool holds
	// for a model's context plan, and zero when unknown. It is read at each
	// load of a curated model.
	Capacity func(context.Context) int64
	// ContextChoice, when set, answers the person's per-conversation context
	// for a model family, and zero for Automatic. It is read at each load.
	ContextChoice func(context.Context, string) int64
}

const DefaultResource = "card:0"
const (
	DefaultIdle  = 5 * time.Minute
	DefaultRenew = 8 * time.Second
	DefaultWait  = 60 * time.Second
)

// Host resolves model objects and owns llama.cpp processes. holder.Holder
// owns lease lifecycle and serializes each model's Ensure with its yield.
type Host struct {
	loadMu       sync.Mutex
	loads        map[string][2]*modelLoad
	loadFailures map[string]modelLoadFailure
	loadLogger   *slog.Logger
	loadLogClose func()
	loadWG       sync.WaitGroup
	loadCtx      context.Context
	loadCancel   context.CancelFunc
	loadClosed   bool
	options      Options
	allow        map[string]Model
	names        []string
	bases        map[string]string
	hosts        map[string]string
	resources    *holder.Holder
	// engines holds each running engine by the host name of its base
	// (MODELHOST-L7, MODELHOST-L8).
	engineMu sync.Mutex
	engines  map[string]*Process
	// video is the video engine kind, or nil.
	video *videoHost
	// context holds each loaded model's context plan, the families whose
	// choice changed while loaded, and the fit fallback's limits
	// (contextreload.go).
	context contextState
	// idle holds each loaded model's own idle time (idle.go).
	idle idleState
	// warm holds, per engine, whether a request used it since it started
	// and the one preload in flight (MODELHOST-W1..W3); warmClient reaches
	// the engines for the preload.
	warm       *warmer
	warmClient *http.Client
	// watch is how often a running engine's sockets are checked again.
	watch time.Duration
	// afterFailedCheck, set by tests only, runs between a failed running
	// socket check and the stop it leads to.
	afterFailedCheck func()
	// recheckEvery and userIdle, set by tests only, replace the background
	// re-hash's period and the session's input idle time (recheck.go).
	recheckEvery time.Duration
	userIdle     func() time.Duration
	// facts caches each chat model's capabilities (facts.go).
	facts factsState
}

// engineHost is the host of name's base URL. It names no network address:
// EngineClient's transport maps it to the running engine's socket, and the
// reserved ".invalid" suffix (RFC 2606) resolves nowhere else.
func engineHost(index int) string { return fmt.Sprintf("engine%d.modelhost.invalid", index) }

// EngineKey is the per-launch key of the engine answering at host, the host
// of its base URL, for this provider's own engine client. A stopped engine
// has none.
func (h *Host) EngineKey(host string) (string, bool) {
	h.engineMu.Lock()
	defer h.engineMu.Unlock()
	p, ok := h.engines[host]
	if !ok {
		return "", false
	}
	return p.key, true
}

// EngineEndpoint is the socket of the engine answering at host and the
// check its server process must pass on every connection (MODELHOST-L7). A
// stopped engine has none.
func (h *Host) EngineEndpoint(host string) (enginehttp.UnixEndpoint, bool) {
	h.engineMu.Lock()
	defer h.engineMu.Unlock()
	p, ok := h.engines[host]
	if !ok {
		return enginehttp.UnixEndpoint{}, false
	}
	return enginehttp.UnixEndpoint{Path: p.Socket(), Verify: p.verify}, true
}

// setEngine records p as name's engine. A new engine's slot holds nothing.
func (h *Host) setEngine(name string, p *Process) {
	h.engineMu.Lock()
	defer h.engineMu.Unlock()
	h.engines[h.hosts[name]] = p
	h.warm.engineChanged(h.hosts[name])
}

// dropEngine forgets p once p is confirmed gone; a later launch for the same
// model has already replaced it.
func (h *Host) dropEngine(p *Process) {
	h.engineMu.Lock()
	defer h.engineMu.Unlock()
	for host, running := range h.engines {
		if running == p {
			delete(h.engines, host)
			h.warm.engineChanged(host)
		}
	}
}

// observeRequest hears every request that runs work on its way to the
// engine at host and answers what it must wait for (MODELHOST-W3).
func (h *Host) observeRequest(host, path string, body []byte) <-chan struct{} {
	return h.warm.observe(host, path, body)
}

// resourcesHeld is the refusal word of a preload's load that the resource
// could not take without asking another holder to yield (MODELHOST-W2,
// inference INF-PL2, resource RES-L6).
const resourcesHeld = "resources:held"

// EnsureSpeculative is Ensure for a preload, which holds the lowest standing
// (MODELHOST-W2): the holder's speculative ensure (resource RES-L6). A
// resident model is left as it is, its idle clock untouched; a model not
// loaded loads only through the no-yield acquire; any failure refuses and
// nothing falls back to Ensure. A refused acquire, a lease service without
// the no-yield acquire, and a load a yield cancelled read resourcesHeld.
func (h *Host) EnsureSpeculative(ctx context.Context, name string) (string, error) {
	return h.ensureShared(ctx, name, true)
}

func New(options Options) (*Host, error) {
	return newHost(options, resolvedDiagnostics)
}

func newHost(options Options, diagnostics func(string) (slog.Handler, func())) (*Host, error) {
	if options.Source == nil {
		return nil, errors.New("modelhost: a source is required")
	}
	if options.Leases == nil {
		return nil, errors.New("modelhost: the lease service is required")
	}
	if len(options.Models) == 0 && options.Video == nil {
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
	if options.Engine.Pin.Release == "" {
		//unchecked: a platform with no pin leaves Pin zero, and every call is refused engine:none
		options.Engine.Pin, _ = PinnedEngine(runtime.GOOS, runtime.GOARCH)
	}
	arguments, err := EngineArguments(options.Engine.Arguments)
	if err != nil {
		return nil, err
	}
	options.Engine.Arguments = arguments
	options.Engine.report = options.OnError
	options.Engine.exited = options.OnEngineExit
	if options.Engine.Files == nil {
		// Records in this provider's state folder prove placed model files
		// for later provider processes too (INF-F7, Decision 2).
		var state string
		switch {
		case options.Engine.Record != "":
			state = filepath.Dir(options.Engine.Record)
		case options.Video != nil && options.Video.Engine.ModelRecord != "":
			state = filepath.Dir(options.Video.Engine.ModelRecord)
		}
		options.Engine.Files = programfiles.NewVerifier(programfiles.Options{StateRoot: state})
		if state != "" && options.OnError != nil {
			logStateOnce(state, options.Engine.Files, options.OnError)
		}
	}
	// A configured engine's socket root must give paths the engine can
	// bind; launch directories a stopped provider left behind are removed
	// unless their socket still answers (MODELHOST-L7). The root itself is
	// made, or refused for its owner or access, at the first launch.
	if options.Engine.Program != "" {
		sockets, err := options.Engine.socketParent()
		if err != nil {
			return nil, fmt.Errorf("modelhost: %w", err)
		}
		if err := enginelaunch.SocketPathFits(sockets); err != nil {
			return nil, fmt.Errorf("modelhost: %w", err)
		}
		for _, err := range enginelaunch.SweepSocketDirs(sockets) {
			if options.OnError != nil {
				options.OnError(fmt.Errorf("modelhost: a stale engine socket directory: %w", err))
			}
		}
	}
	if v := options.Video; v != nil {
		if v.Engine.Pin.Release == "" {
			//unchecked: a platform with no pin leaves Pin zero, and every video is refused engine:none
			v.Engine.Pin, _ = PinnedVideoEngine(runtime.GOOS, runtime.GOARCH)
		}
		if v.Engine.Set.Name == "" {
			v.Engine.Set = FastWan22TI2V5B
		}
		if v.Model == "" {
			v.Model = v.Engine.Set.Name
		}
		if v.Engine.Files == nil {
			v.Engine.Files = options.Engine.Files
		}
		if _, err := sdcpp.ParseArgs(v.Engine.Arguments); err != nil {
			return nil, err
		}
		v.Engine.report = options.OnError
		v.Engine.exited = options.OnEngineExit
		for _, m := range options.Models {
			if m.Name == v.Model {
				return nil, fmt.Errorf("modelhost: %q names both a chat model and the video model", v.Model)
			}
		}
	}
	var admit func(context.Context, string) error
	if gpu := options.GPU; gpu != nil {
		admit = func(ctx context.Context, _ string) error { return gpuRefusal(gpu.AdmitLoad(ctx)) }
		// An engine's device-lost line counts only once that engine exits or
		// fails its own readiness check twice; an engine that keeps
		// answering printed caller text, and nothing is stopped or
		// quarantined (MODELHOST-G1, resource RES-G1). It runs on its own
		// goroutine (engineproc.LostLine), never the engine's output
		// goroutine (KIT-P4).
		lost := func(engine string) func(*Process, string) {
			hear := engineproc.DeviceLost(gpu, "modelhost", engine, options.OnError)
			return func(p *Process, marker string) {
				hear(p.exited, func(ctx context.Context) bool {
					ok, err := p.ready(ctx)
					return err == nil && ok
				}, marker)
			}
		}
		options.Engine.deviceLost = lost("llama-server")
		if options.Video != nil {
			options.Video.Engine.deviceLost = lost("sd-server")
		}
	}
	var h *Host
	r, err := holder.New(holder.Options{
		Admit:  admit,
		Leases: options.Leases, Resource: options.Resource, Idle: options.Idle,
		Renew: options.Renew, Wait: options.Wait, Now: options.Now,
		// Each lease is labelled with the model it holds for the person's
		// Memory view (resource RES-L10). The runtime forwards the work, so
		// the app that asked is not known here and the label is the model
		// alone.
		Label: func(name string) string { return name },
		// A resident's idle time is its family's engine-profile value, else
		// the declared --idle (MODELHOST-H8).
		IdleFor: func(name string) time.Duration {
			if h == nil {
				return 0
			}
			return h.residentIdle(name)
		},
		// The video engine is in use while video work is queued or running
		// (video addendum condition 6); a chat model while the inference
		// provider holds a request for it (MODELHOST-H6).
		Active: func(name string) bool {
			if h != nil && h.video != nil && name == h.video.opts.Model {
				return h.video.active()
			}
			return options.Active != nil && options.Active(name)
		},
		InFlight: inFlight(options, func() *Host { return h }),
		OnError:  options.OnError,
	})
	if err != nil {
		return nil, err
	}
	handler, closeDiagnostics := diagnostics("modelhostd")
	h = &Host{options: options, allow: map[string]Model{},
		bases: map[string]string{}, hosts: map[string]string{}, resources: r, engines: map[string]*Process{},
		watch: socketWatch, warm: newWarmer(), loadLogger: slog.New(handler), loadLogClose: closeDiagnostics}
	constructed := false
	defer func() {
		if !constructed {
			h.Close()
		}
	}()
	h.warmClient = EngineClient(h, options.OnError)
	if options.GPU != nil {
		// A reset stops every engine through the holder (resource RES-G2).
		options.GPU.Attach(r)
	}
	for _, m := range options.Models {
		if m.Name == "" || m.Store == "" || m.Object == "" {
			return nil, fmt.Errorf("modelhost: model %q needs a name, a store and an object", m.Name)
		}
		if m.Content != nil {
			pin := *m.Content
			m.Content = &pin
		}
		h.allow[m.Name] = m
		h.names = append(h.names, m.Name)
	}
	sort.Strings(h.names)
	for index, name := range h.names {
		h.hosts[name] = engineHost(index)
		h.bases[name] = "http://" + engineHost(index)
	}
	if options.Video != nil {
		if h.video, err = newVideoHost(h, *options.Video); err != nil {
			return nil, err
		}
	}
	constructed = true
	return h, nil
}

// inFlight is the holder's InFlight for options: nil without the
// provider's count, so a yield keeps waiting for Active. The video engine's
// queued or running work counts as one request; its admissions do not pass
// the yield gate.
func inFlight(options Options, host func() *Host) func(string) int {
	if options.InFlight == nil {
		return nil
	}
	return func(name string) int {
		if h := host(); h != nil && h.video != nil && name == h.video.opts.Model {
			if h.video.active() {
				return 1
			}
			return 0
		}
		return options.InFlight(name)
	}
}

func (h *Host) Models() []string { return append([]string(nil), h.names...) }

// Base is the URL the inference provider reaches name's engine at through
// EngineClient. Its host names the model's engine slot; no port exists.
func (h *Host) Base(name string) string  { return h.bases[name] }
func (h *Host) Serves(name string) bool  { _, ok := h.allow[name]; return ok }
func (h *Host) Residents() []string      { return h.resources.Residents() }
func (h *Host) Loaded(name string) bool  { _, ok := h.resources.Snapshot(name); return ok }
func (h *Host) Lease(name string) string { lease, _ := h.resources.Snapshot(name); return lease }

// CheckModel validates the immutable model and engine capture before chat
// acceptance. Object resolution, file proof, leasing and loading stay in Ensure.
func (h *Host) CheckModel(name string) error {
	if !h.Serves(name) || h.Base(name) == "" {
		return refuse(Unavailable, "no model called %q is hosted here", name)
	}
	if err := h.options.Engine.configured(); err != nil {
		return refuse(NoEngine, "%v", err)
	}
	return nil
}

// Ensure loads an allowlisted model under a lease. Every mediated inference
// call reaches this hook after Provider reserves model activity. A call racing
// a yield waits on the same component lock and reloads once yield completes.
func (h *Host) Ensure(ctx context.Context, name string) (string, error) {
	return h.ensureShared(ctx, name, false)
}

// component is name's load as the holder runs it: the object resolved and
// its engine proven before any lease is asked for, then the engine started
// under the lease. amount receives the object's size.
func (h *Host) component(ctx context.Context, name string, model Model, amount, charged *int64, leased *atomic.Bool) func() (holder.Component, error) {
	return func() (holder.Component, error) {
		object, err := h.options.Source.Resolve(model.Store, model.Object, h.options.Engine.accepts())
		if errors.Is(err, bridge.ErrPickle) {
			return holder.Component{}, refuse(pickleObject(model.Store, model.Object), "%v", err)
		}
		if errors.Is(err, bridge.ErrUnaccepted) {
			return holder.Component{}, refuse(unboundObject(model.Store, model.Object), "%v", err)
		}
		if err != nil {
			return holder.Component{}, refuse(unknownObject(model.Store, model.Object), "no store called %s holds %s", model.Store, model.Object)
		}
		if model.Content != nil && object.Size != model.Content.Size {
			return holder.Component{}, refuse(Unavailable, "the model size differs from its explicit content pin")
		}
		if err := h.options.Engine.present(ctx); err != nil {
			var refusal *Refusal
			if errors.As(err, &refusal) {
				return holder.Component{}, refusal
			}
			return holder.Component{}, refuse(NoEngine, "%v", err)
		}
		*amount = object.Size
		if *amount <= 0 {
			return holder.Component{}, refuse(Unavailable, "the source reports no size for %s/%s", model.Store, model.Object)
		}
		// The holder runs Start only after Acquire granted Amount on the
		// declared resource (holder.Component), so the grant covers the
		// file's whole size on that resource (MODELHOST-H1, MODELHOST-L9).
		// A curated model's launches each lease what they use: its weights
		// and the KV of its context (resource RES-L9). The holder starts the
		// largest that fits what is free.
		grant := leaseGrant{resource: h.options.Resource, size: object.Size}
		engines := h.modelEngines(model, object)
		// A vision projector beside the model loads with it (--mmproj), and
		// its bytes are part of every launch's lease (MODELHOST-L12).
		projector, hasProjector := h.projector(object)
		h.noteResolved(name, model, object, projector, hasProjector)
		var loads []holder.Component
		for i := range engines {
			if hasProjector && !engines[i].Embeddings {
				engines[i].projector = projector.Path
			}
		}
		for _, engine := range engines {
			if err := engine.cacheRecipe().checkArguments(engine.Arguments); err != nil {
				return holder.Component{}, refuse(Unavailable, "%v", err)
			}
			// A curated plan's need covers weights, KV and the engine's own
			// buffers; the prompt cache and checkpoints come on top of it.
			leaseBytes, err := engine.cacheRecipe().leaseBytes(object.Size)
			if plan := engine.contextLoad; err == nil && plan != nil && plan.need > object.Size {
				leaseBytes, err = engine.cacheRecipe().addTo(plan.need)
			}
			if err != nil {
				return holder.Component{}, refuse(Unavailable, "%v", err)
			}
			if engine.projector != "" {
				if projector.Size > math.MaxInt64-leaseBytes {
					return holder.Component{}, refuse(Unavailable, "the model and its projector overflow a lease")
				}
				leaseBytes += projector.Size
			}
			loads = append(loads, h.engineLoad(ctx, name, engine, object, grant, leaseBytes, charged, leased))
		}
		// A load that does not start records the largest plan, so a fit
		// fallback after it lowers from there.
		h.recordContextLoad(name, engines[0].contextLoad)
		h.recordIdle(name, engines[0].cacheRecipe().idle)
		*charged = loads[0].Amount
		main := loads[0]
		main.Smaller = loads[1:]
		return main, nil
	}
}

// engineLoad is one launch of name's engine as a holder load: started only
// under its own grant of amount, recording the plan and the charge it runs.
func (h *Host) engineLoad(ctx context.Context, name string, engine Engine, object Object, grant leaseGrant, amount int64, charged *int64, leased *atomic.Bool) holder.Component {
	return holder.Component{Amount: amount, Start: func() (holder.Owned, error) {
		leased.Store(true)
		h.recordContextLoad(name, engine.contextLoad)
		*charged = amount
		load, _ := ctx.Value(modelLoadKey{}).(*modelLoad)
		if load != nil {
			load.progress.begin()
			engine.loadPhase = load.progress.publish
		}
		process, err := engine.start(object.Path, name, grant)
		if err != nil {
			return nil, err
		}
		if load != nil {
			load.started = true
			load.readyAt = time.Time{}
		}
		recipe := engine.cacheRecipe()
		process.requestSettings = recipe.requestSettings
		process.answerReserve = recipe.answerReserve
		if recipe.pinsSlots() {
			process.cacheSlots = newCacheSlots(max(recipe.slots, recipe.residentSlots))
		}
		h.setEngine(name, process)
		return &ownedProcess{name: name, process: process, ready: h.options.Engine.Ready, host: h, load: load}, nil
	}}
}

type ownedProcess struct {
	name    string
	process *Process
	ready   time.Duration
	host    *Host
	load    *modelLoad
}

// kit is the provider kit's holder.Owned for this engine (KIT-P5): its
// readiness through the engine's own probe, then the socket rule at
// readiness and every watch while it runs (MODELHOST-H3, MODELHOST-L7).
func (o *ownedProcess) kit() *engineproc.Owned {
	ready := o.ready
	if ready <= 0 {
		ready = DefaultReady
	}
	h, p := o.host, o.process
	var progress func() uint64
	if p.loadProgress != nil {
		progress = p.loadProgress.count
	}
	return &engineproc.Owned{Name: o.name, Process: p.owned, ReadyWithin: ready, Probe: p.ready,
		Progress: progress,
		Sockets:  func() error { return engineSockets(p) }, SocketWord: networkWord(p), Watch: h.watch,
		Resident: func() bool { return h.resources.PID(o.name) == p.PID() }, Report: h.options.OnError,
		OnStop: func() { h.dropEngine(p) }, AfterFailedCheck: h.afterFailedCheck}
}

func (o *ownedProcess) PID() int { return o.process.PID() }

// Ready waits for the engine's health over its socket, each connection bound
// to the engine's process, then confirms the engine's tree holds no network
// socket. A refusal here is a failed readiness: the holder stops the engine
// and releases the lease before the caller is told (MODELHOST-H3,
// MODELHOST-L7).
// An engine that printed an allocation failure and did not become ready
// reads EngineDoesNotFit with its remedy (MODELHOST-L9).
func (o *ownedProcess) Ready(ctx context.Context) error {
	started := time.Now()
	if o.process.log != nil {
		_, logErr := fmt.Fprintf(o.process.log, "%s phase=readiness_begin\n", started.UTC().Format(time.RFC3339Nano))
		if logErr != nil && o.host.options.OnError != nil {
			o.host.options.OnError(logErr)
		}
	}
	err := o.kit().Ready(ctx)
	if o.process.log != nil {
		_, logErr := fmt.Fprintf(o.process.log, "%s phase=readiness elapsed_ms=%d error=%q\n", time.Now().UTC().Format(time.RFC3339Nano), time.Since(started).Milliseconds(), fmt.Sprint(err))
		if logErr != nil && o.host.options.OnError != nil {
			o.host.options.OnError(logErr)
		}
	}
	if err == nil {
		if o.load != nil {
			// Ready runs under the holder lock for this exact start. Publish
			// its first successful readiness only to the load that started it.
			o.load.readyAt = time.Now()
			o.load = nil
		}
		return nil
	}
	if line := o.process.fit.failure(); line != "" {
		return refuse(EngineDoesNotFit, "%s did not fit the card (%s): %s", o.name, line, doesNotFitRemedy)
	}
	return err
}

func (o *ownedProcess) Alive() bool { return o.kit().Alive() }

func (o *ownedProcess) Stop() bool {
	deadline := time.Now().Add(stopWithin)
	if !o.kit().Stop() {
		return false
	}
	if err := o.process.waitCleanup(deadline); err != nil {
		if errors.Is(err, errCleanupPending) {
			if o.host.options.OnError != nil {
				o.host.options.OnError(fmt.Errorf("modelhost: cleanup of %s after engine exit: %w", o.name, err))
			}
			return false
		}
		o.process.reportCleanup(err, o.host.options.OnError)
		// The engine tree is gone, so a stale file must not keep its GPU
		// lease held after a completed but unsuccessful cleanup.
	}
	return true
}

func (h *Host) mapFailure(name string, amount int64, err error) error {
	return refusal.Load(h.options.Resource, name, amount, err)
}

func (h *Host) Unload(name, why string) { h.resources.Unload(name, why) }
func (h *Host) Run(ctx context.Context) {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() { defer close(done); h.contextLoop(ctx) }()
	rechecked := make(chan struct{})
	go func() { defer close(rechecked); h.recheckLoop(ctx) }()
	h.resources.Run(ctx)
	cancel()
	<-done
	<-rechecked
}

// Close stops the video worker, then every engine, and releases the leases.
func (h *Host) Close() {
	h.loadMu.Lock()
	h.loadClosed = true
	if h.loadCancel != nil {
		h.loadCancel()
	}
	h.loadMu.Unlock()
	h.loadWG.Wait()
	h.closeFacts()
	if h.video != nil {
		h.video.close()
	}
	h.resources.Close()
	if h.loadLogClose != nil {
		h.loadLogClose()
	}
}
func (h *Host) yield(request res.YieldRequest) (res.YieldAnswer, string) {
	return h.resources.Yield(request)
}
