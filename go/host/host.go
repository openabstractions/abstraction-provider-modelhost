package host

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
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

type Model struct{ Name, Store, Object string }
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
	Active      func(model string) bool
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
	options   Options
	allow     map[string]Model
	names     []string
	bases     map[string]string
	hosts     map[string]string
	resources *holder.Holder
	// engines holds each running engine by the host name of its base
	// (MODELHOST-L7, MODELHOST-L8).
	engineMu sync.Mutex
	engines  map[string]*Process
	// video is the video engine kind, or nil.
	video *videoHost
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
	model, ok := h.allow[name]
	if !ok {
		return "", refuse(Unavailable, "no model called %q is hosted here", name)
	}
	var amount int64
	err := h.resources.EnsureSpeculative(ctx, name, h.component(ctx, name, model, &amount))
	var failure *holder.Failure
	if errors.As(err, &failure) && failure.Word() == holder.WordHeld {
		return "", refuse(resourcesHeld, "loading %s would ask a holder of %s to yield", name, h.options.Resource)
	}
	if err != nil {
		return "", h.mapFailure(name, amount, err)
	}
	return h.bases[name], nil
}

func New(options Options) (*Host, error) {
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
		options.Engine.Files = enginefiles.NewVerifier(enginefiles.Options{})
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
		// The video engine is in use while video work is queued or running
		// (video addendum condition 6); a chat model while the inference
		// provider holds a request for it (MODELHOST-H6).
		Active: func(name string) bool {
			if h != nil && h.video != nil && name == h.video.opts.Model {
				return h.video.active()
			}
			return options.Active != nil && options.Active(name)
		},
		OnError: options.OnError,
	})
	if err != nil {
		return nil, err
	}
	h = &Host{options: options, allow: map[string]Model{},
		bases: map[string]string{}, hosts: map[string]string{}, resources: r, engines: map[string]*Process{},
		watch: socketWatch, warm: newWarmer()}
	h.warmClient = EngineClient(h, options.OnError)
	if options.GPU != nil {
		// A reset stops every engine through the holder (resource RES-G2).
		options.GPU.Attach(r)
	}
	for _, m := range options.Models {
		if m.Name == "" || m.Store == "" || m.Object == "" {
			return nil, fmt.Errorf("modelhost: model %q needs a name, a store and an object", m.Name)
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
	return h, nil
}

func (h *Host) Models() []string { return append([]string(nil), h.names...) }

// Base is the URL the inference provider reaches name's engine at through
// EngineClient. Its host names the model's engine slot; no port exists.
func (h *Host) Base(name string) string  { return h.bases[name] }
func (h *Host) Serves(name string) bool  { _, ok := h.allow[name]; return ok }
func (h *Host) Residents() []string      { return h.resources.Residents() }
func (h *Host) Loaded(name string) bool  { _, ok := h.resources.Snapshot(name); return ok }
func (h *Host) Lease(name string) string { lease, _ := h.resources.Snapshot(name); return lease }

// Ensure loads an allowlisted model under a lease. Every mediated inference
// call reaches this hook after Provider reserves model activity. A call racing
// a yield waits on the same component lock and reloads once yield completes.
func (h *Host) Ensure(ctx context.Context, name string) (string, error) {
	model, ok := h.allow[name]
	if !ok {
		return "", refuse(Unavailable, "no model called %q is hosted here", name)
	}
	var amount int64
	err := h.resources.Ensure(ctx, name, h.component(ctx, name, model, &amount))
	if err != nil {
		return "", h.mapFailure(name, amount, err)
	}
	return h.bases[name], nil
}

// component is name's load as the holder runs it: the object resolved and
// its engine proven before any lease is asked for, then the engine started
// under the lease. amount receives the object's size.
func (h *Host) component(ctx context.Context, name string, model Model, amount *int64) func() (holder.Component, error) {
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
		// declared resource (holder.Component), so the grant is the file's
		// whole size on that resource (MODELHOST-H1, MODELHOST-L9).
		grant := leaseGrant{resource: h.options.Resource, size: object.Size}
		return holder.Component{Amount: *amount, Start: func() (holder.Owned, error) {
			process, err := h.options.Engine.start(object.Path, name, grant)
			if err != nil {
				return nil, err
			}
			h.setEngine(name, process)
			return &ownedProcess{name: name, process: process, ready: h.options.Engine.Ready, host: h}, nil
		}}, nil
	}
}

type ownedProcess struct {
	name    string
	process *Process
	ready   time.Duration
	host    *Host
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
	return &engineproc.Owned{Name: o.name, Process: p.owned, ReadyWithin: ready, Probe: p.ready,
		Sockets: func() error { return engineSockets(p) }, SocketWord: networkWord(p), Watch: h.watch,
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
	err := o.kit().Ready(ctx)
	if err == nil {
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
func (h *Host) Run(ctx context.Context) { h.resources.Run(ctx) }

// Close stops the video worker, then every engine, and releases the leases.
func (h *Host) Close() {
	if h.video != nil {
		h.video.close()
	}
	h.resources.Close()
}
func (h *Host) yield(request res.YieldRequest) (res.YieldAnswer, string) {
	return h.resources.Yield(request)
}
