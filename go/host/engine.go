package host

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-provider-kit/go/devicelost"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
)

// An Engine is a llama.cpp server program already installed on this machine.
// The path is configured; this package looks for none, downloads none and
// ships none (research/model-bridge/DECISION.md §4 rules 4 and 6). It must be
// the program of the pinned release, in a folder that holds exactly the
// pinned files (VISION 2026-10-01; CONTRACT.md MODELHOST-L3).
type Engine struct {
	// Program is the absolute path of llama-server on this machine.
	Program string
	// Pin is the llama.cpp release Program belongs to. New reads
	// PinnedEngine for this platform when it is zero.
	Pin EnginePin
	// Record is the engine folder's metadata record in this provider's
	// private state, outside the folder (inference CONTRACT.md INF-F2).
	Record string
	// Files proves the engine folder before each start and holds its files
	// until the engine exits (INF-F1..INF-F4). New makes one when it is nil.
	Files *programfiles.Verifier
	// Arguments are extra arguments a person configured, placed after the
	// model, host and port this provider chooses. Each must be on the
	// reviewed list; New refuses any other (EngineArguments, MODELHOST-L5).
	Arguments []string
	// Embeddings asks the engine to serve the embeddings endpoint as well.
	Embeddings bool
	// Ready is the readiness stall budget; first genuine load phases reset it.
	// Zero reads DefaultReady.
	Ready time.Duration
	// MemoryLimit is the engine job's per-process memory limit in bytes on
	// Windows. Zero leaves the established service default without a cap;
	// the service chooses any nonzero value (MODELHOST-E1).
	MemoryLimit uint64
	// Stderr, when set, receives the engine's own output.
	Stderr io.Writer
	// Temp is the engine's private temporary folder, its TEMP and TMP
	// (MODELHOST-L6). Empty reads engine-temp beside Record.
	Temp string
	// Sockets is the folder each launch's private socket directory is made
	// in (MODELHOST-L7). Empty reads the provider kit's fixed per-user root,
	// enginelaunch.SocketRoot, whatever the state folder; tests name their
	// own. A socket path in it must fit enginelaunch.MaxSocketPath.
	Sockets string
	// cache is the provider's private launch recipe. Person arguments never
	// populate it. A nil recipe preserves the engine defaults.
	cache *engineCache

	// fixture are the stand-in engine's own flags. Only this package's
	// tests set them; they are never checked against the reviewed list
	// because no llama-server receives them.
	fixture []string
	// report receives one diagnostic per profile variable left out of the
	// engine environment; New sets it to Options.OnError.
	report func(error)
	// exited receives each engine's exit record; New sets it to
	// Options.OnEngineExit (MODELHOST-E1).
	exited func(EngineExit)
	// beforeLaunch, set by tests only, runs once the launch's socket
	// directory exists and before the engine starts.
	beforeLaunch func(*enginelaunch.SocketDir)
	// deviceLost hears the engine's own device-lost line with the process
	// that printed it; New sets it when the provider has GPU fault handling
	// (MODELHOST-G1).
	deviceLost func(p *Process, marker string)
	loadPhase  func(uint64)
	// modelPin is supplied by the content-evidence owner for this launch.
	modelPin    *programfiles.Placed
	modelRecord string
	// contextLoad is the context plan this launch runs, nil for a model
	// without one; the host records it for reloads and the fit fallback.
	contextLoad *enginePlan
	// projector is the model's vision projector file the host resolved in
	// the model's store, passed as --mmproj; empty for none. It is never a
	// person's argument (EngineArguments refuses --mmproj).
	projector string
}

// engineOutput is where an engine's output goes: out, read on the way for
// the engine's own device-lost line in form when the provider listens for
// one (devicelost, MODELHOST-G1, KIT-P4). The returned line is bound to the
// process once it runs; it is nil when nothing listens.
func engineOutput(out io.Writer, form devicelost.LineForm, deviceLost func(*Process, string)) (io.Writer, *engineproc.LostLine[*Process]) {
	return engineproc.Output(out, form, deviceLost)
}

// socketParent is the folder launch directories are made in: Sockets, or
// the provider kit's per-user socket root.
func (e Engine) socketParent() (string, error) {
	if e.Sockets != "" {
		if !filepath.IsAbs(e.Sockets) {
			return "", fmt.Errorf("the engine socket folder %s is not absolute", e.Sockets)
		}
		return filepath.Clean(e.Sockets), nil
	}
	return enginelaunch.SocketRoot()
}

// DefaultReady bounds a readiness stall. Genuine first load phases reset it;
// a phase with no observable progress can still reach this bound.
const DefaultReady = 180 * time.Second

// accepts is what the pinned engine reads: llama-server at b10825 takes a
// model only as -m FNAME, a whole file (model-locators DECISION.md §8).
func (e Engine) accepts() bridge.Accepts { return bridge.FileOnly }

// present reports the engine this machine has, or why there is none. It is the
// only check made before a lease is asked for, because asking the card for
// bytes no engine can use would take them from a holder for nothing. It
// proves the engine folder against the pin and releases it at once; start
// proves it again and holds it.
func (e Engine) present(ctx context.Context) error {
	if err := e.configured(); err != nil {
		return err
	}
	if err := e.Files.Check(ctx, e.tree()); err != nil {
		return e.filesRefusal(err)
	}
	return nil
}

// configured reports an engine configuration that names no pinned program.
func (e Engine) configured() error {
	if e.Program == "" {
		return errors.New("no llama.cpp engine is configured; declare one with --engine")
	}
	if !filepath.IsAbs(e.Program) {
		return fmt.Errorf("the engine path %s is not absolute", e.Program)
	}
	info, err := os.Stat(e.Program)
	if err != nil {
		return fmt.Errorf("no engine at %s", e.Program)
	}
	if info.IsDir() {
		return fmt.Errorf("the engine path %s is a directory", e.Program)
	}
	if e.Pin.Release == "" {
		return fmt.Errorf("no reviewed llama.cpp release is pinned for %s/%s; this provider launches no other", runtime.GOOS, runtime.GOARCH)
	}
	if err := e.Pin.valid(); err != nil {
		return err
	}
	if !identity.SamePathName(filepath.Base(e.Program), e.Pin.Program) {
		return fmt.Errorf("the engine %s is not %s of the pinned llama.cpp %s", e.Program, e.Pin.Program, e.Pin.Release)
	}
	if e.Files == nil || !filepath.IsAbs(e.Record) {
		return errors.New("the engine has no verifier or no absolute metadata record")
	}
	return nil
}

// tree is the engine folder as enginefiles proves it.
func (e Engine) tree() programfiles.Tree {
	return programfiles.Tree{Name: "llama.cpp " + e.Pin.Release, Root: filepath.Dir(e.Program),
		Manifest: e.Pin.manifest(), Record: e.Record}
}

// filesRefusal is the refusal of an engine folder the kit did not prove. A
// file the kit could not open or read, such as one a scanner holds, is
// engine:unreadable and passes once the file is free; any other refusal
// means the folder is not the pinned release (engine:none).
func (e Engine) filesRefusal(err error) *Refusal {
	if errors.Is(err, programfiles.ErrUnreadable) {
		return refuse(EngineUnreadable, "a file of the engine folder %s could not be read (%v); try again once the program holding it lets go",
			filepath.Dir(e.Program), err)
	}
	return refuse(NoEngine, "the engine folder %s is not the pinned llama.cpp %s (%v); stage %s with modelhostd engine install",
		filepath.Dir(e.Program), e.Pin.Release, err, e.Pin.Archive)
}

// A Process is one running engine owned by this provider: the provider
// kit's supervised process (engineproc, KIT-P1) with what the model host's
// engine kinds keep beside it. On Windows exited is signaled only after its
// contained process tree is confirmed gone. Other platforms retain the
// direct-child lifecycle.
type Process struct {
	cacheSlots      *cacheSlots
	requestSettings requestSettingsByUse
	answerReserve   answerReserve
	loadProgress    *loadProgress
	log             *engineLog
	owned           *engineproc.Process
	id              int
	exited          <-chan struct{}
	// contains answers whether a process is in this engine's contained
	// tree; the Windows job sets it. Without it no socket counts as the
	// engine's, and the readiness check refuses.
	contains identity.TreeContains
	// key is this launch's LLAMA_API_KEY. It lives in this process's memory
	// and the engine's environment block only.
	key string
	// endpoint is a socket engine's private launch directory; it is removed
	// once the engine's exit is confirmed. Nil for a loopback engine.
	endpoint *enginelaunch.SocketDir
	// cleaned closes after provider-owned file holds, socket directories and
	// logs are released. Stop must wait for it before the holder releases its
	// lease; exited alone proves only that the process tree is gone.
	cleaned    chan struct{}
	cleanupErr error
	// Report a completed cleanup error once, whether the exit callback or a
	// caller waiting in Stop observes it first.
	cleanupReported sync.Once
	// client reaches this engine alone: over its socket, binding the server
	// process before each request (MODELHOST-L7), or over its loopback port.
	client *http.Client
	// port and base are a loopback engine's listener: the video kind's
	// sd-server, which serves TCP only (inference CONTRACT.md INF-EL2,
	// MODELHOST-V5). Zero and empty for a socket engine.
	port int
	base string
	// probe reports readiness through client; nil reads llama.cpp's own
	// /health.
	probe func(context.Context, *http.Client) (bool, error)
	// progress reads sd-server's step counts from its output; nil for an
	// engine that reports none there.
	progress *progressTap
	// fit hears a llama-server's allocation failure (MODELHOST-L9); nil
	// for other engines.
	fit *fitTap
}

// startOwned starts spec through the provider kit (KIT-P1, KIT-P3).
func startOwned(spec ownedSpec) (*Process, error) {
	spec.Provider = "modelhost"
	owned, err := engineproc.Start(spec)
	if errors.Is(err, engineproc.ErrEnvironment) {
		return nil, ErrEngineEnvironment
	}
	if err != nil {
		return nil, err
	}
	return &Process{owned: owned, id: owned.PID(), exited: owned.Exited(), contains: owned.Contains()}, nil
}

// startOwnedProcess starts one engine of the model host.
func startOwnedProcess(spec ownedSpec) (*Process, error) { return startOwned(spec) }

// Socket is the socket path this engine serves on; empty for a loopback
// engine.
func (p *Process) Socket() string {
	if p.endpoint == nil {
		return ""
	}
	return p.endpoint.Socket
}

// verify proves the server end of a connection to this engine's socket is
// the engine (engineproc.Process.Verify).
func (p *Process) verify(server identity.Process) (bool, error) { return p.owned.Verify(server) }

// PID is the process id this provider started.
func (p *Process) PID() int {
	if p == nil {
		return 0
	}
	return p.id
}

// FreePort is a loopback port nothing holds right now, for the video kind's
// sd-server, which serves TCP only (MODELHOST-V5). Chat engines serve a
// socket and take no port (MODELHOST-L7).
func FreePort() (int, error) { return engineproc.FreePort() }

var newSocketDir = enginelaunch.NewSocketDir

// leaseGrant is the lease an engine starts under: the file's size, size
// bytes, admitted on resource (MODELHOST-H1).
type leaseGrant struct {
	resource string
	size     int64
}

// start runs one engine for modelPath under alias, serving on a socket in a
// fresh private launch directory, and returns as soon as the child is
// running. It does not wait for the model to load. grant is the lease the
// engine starts under, from which its default placement follows
// (enginePlacement, MODELHOST-L9). The engine folder is proven against the
// pin and held, and the launch directory kept, until the engine's exit is
// confirmed; an unconfirmed exit keeps both (INF-F4, MODELHOST-L7).
func (e Engine) start(modelPath, alias string, grant leaseGrant) (*Process, error) {
	prepareStarted := time.Now()
	if err := e.configured(); err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	extra, err := EngineArguments(e.Arguments)
	if err != nil {
		return nil, refuse(Unavailable, "%v", err)
	}
	key, err := newLaunchKey()
	if err != nil {
		return nil, refuse(Unavailable, "%v", err)
	}
	env, omitted, err := e.launchEnvironment(key)
	if err != nil {
		return nil, refuse(Unavailable, "the engine environment: %v", err)
	}
	if e.report != nil {
		for _, diagnostic := range omitted {
			e.report(diagnostic)
		}
	}
	parent, err := e.socketParent()
	if err != nil {
		return nil, refuse(Unavailable, "%v", err)
	}
	endpoint, err := newSocketDir(parent)
	if err != nil {
		if errors.Is(err, enginelaunch.ErrSocketTransport) {
			return nil, refuse(EngineUnreachable, "the engine socket: %v", err)
		}
		return nil, refuse(Unavailable, "the engine socket: %v", err)
	}
	removeEndpoint := func() error {
		err := endpoint.Remove()
		return err
	}
	reportEndpoint := func(err error) {
		if err != nil && e.report != nil {
			e.report(fmt.Errorf("modelhost: the engine socket directory: %w", err))
		}
	}
	hold, err := e.Files.Launch(context.Background(), e.tree())
	if err != nil {
		reportEndpoint(removeEndpoint())
		return nil, e.filesRefusal(err)
	}
	var modelHold *programfiles.Hold
	if e.modelPin != nil {
		placed := *e.modelPin
		placed.Path = modelPath
		modelHold, err = e.Files.LaunchPlaced(context.Background(), "GGUF "+alias, []programfiles.Placed{placed}, e.modelRecord)
		if err != nil {
			cleanupErr := errors.Join(hold.Release(), removeEndpoint())
			if plain := placedProblem(err, downloadChatAgain); plain != "" && cleanupErr == nil {
				return nil, refuse(Unavailable, "%s", plain)
			}
			return nil, refuse(Unavailable, "the model file: %v", errors.Join(err, cleanupErr))
		}
		modelPath = modelHold.Paths()[0]
	}
	release := func() error {
		var modelErr error
		if modelHold != nil {
			modelErr = modelHold.Release()
		}
		return errors.Join(hold.Release(), modelErr)
	}
	if e.beforeLaunch != nil {
		e.beforeLaunch(endpoint)
	}
	// The provider names the engine's socket itself; --host and --port are
	// never on the reviewed list, so no configured argument can open a
	// network listener (MODELHOST-L7). llama-server b10825 serves AF_UNIX
	// when --host ends in ".sock" and ignores the port then. --no-webui
	// keeps the bundled web UI off; the per-launch key in the environment
	// closes every other route to callers without it (MODELHOST-L8).
	args := []string{"--model", modelPath, "--host", endpoint.Socket, "--no-webui"}
	if alias != "" {
		args = append(args, "--alias", alias)
	}
	if e.projector != "" {
		args = append(args, "--mmproj", e.projector)
	}
	if e.Embeddings {
		args = append(args, "--embeddings")
	}
	args = append(args, enginePlacement(grant.resource, grant.size, extra)...)
	args = append(args, extra...)
	args = append(args, e.cacheRecipe().arguments()...)
	args = append(args, e.fixture...)
	var started *Process
	client, err := enginehttp.NewUnixClient(endpoint.Socket, func(server identity.Process) (bool, error) {
		if started == nil {
			return false, enginehttp.ErrEnginePeer
		}
		return started.verify(server)
	})
	if err != nil {
		endpointErr := removeEndpoint()
		reportEndpoint(endpointErr)
		cleanupErr := errors.Join(release(), endpointErr)
		return nil, refuse(Unavailable, "the engine client: %v", errors.Join(err, cleanupErr))
	}
	model := alias
	if model == "" {
		model = filepath.Base(modelPath)
	}
	// Each model keeps bounded output in provider state across restarts.
	// The hash is a filename, never a model path supplied by the caller.
	logID := sha256.Sum256([]byte(model))
	log, err := openEngineLog(filepath.Join(filepath.Dir(e.Record), "logs", fmt.Sprintf("llama-%x.log", logID[:8])), engineLogBytes)
	if err != nil {
		cleanupErr := errors.Join(release(), removeEndpoint())
		return nil, refuse(Unavailable, "the engine log: %v", errors.Join(err, cleanupErr))
	}
	// Arguments include effective placement and the pinned model locator;
	// the launch key remains in the environment and is never logged.
	_, logErr := fmt.Fprintf(log, "%s phase=launch prepare_ms=%d release=%q model=%q args=%q\n", time.Now().UTC().Format(time.RFC3339Nano), time.Since(prepareStarted).Milliseconds(), e.Pin.Release, model, args)
	if logErr != nil {
		cleanupErr := errors.Join(log.Close(), release(), removeEndpoint())
		return nil, refuse(Unavailable, "the engine log: %v", errors.Join(logErr, cleanupErr))
	}
	var outputWriter io.Writer = log
	if e.Stderr != nil {
		outputWriter = io.MultiWriter(log, e.Stderr)
	}
	progress := &loadProgress{out: outputWriter, publish: e.loadPhase}
	fit := newFitTap(progress)
	output, lost := engineOutput(fit, devicelost.LlamaServer, e.deviceLost)
	p, err := startOwnedProcess(ownedSpec{Program: e.Program, Args: args, Env: env, Output: output,
		MemoryLimit: e.MemoryLimit,
		Engine:      "llama-server", Model: model, Exited: e.exited})
	if err != nil {
		if closeErr := log.Close(); closeErr != nil && e.report != nil {
			e.report(closeErr)
		}
		//unchecked: the start failure is the error returned; a failed close leaves only read-shared handles
		release()
		reportEndpoint(removeEndpoint())
		return nil, err
	}
	started = p
	// A model file this launch hashed in full is in the file cache already
	// (MODELHOST-L10).
	if modelHold == nil || modelHold.Hashed() == 0 {
		startWarming([]string{modelPath}, p.exited, log)
	}
	p.log = log
	p.fit = fit
	p.loadProgress = progress
	p.endpoint = endpoint
	p.key = key
	p.client = client
	cleaned := make(chan struct{})
	p.cleaned = cleaned
	go func() {
		<-p.exited
		defer close(cleaned)
		_, logErr := fmt.Fprintf(log, "%s phase=exit elapsed_ms=%d\n", time.Now().UTC().Format(time.RFC3339Nano), time.Since(prepareStarted).Milliseconds())
		p.cleanupErr = errors.Join(release(), removeEndpoint(), logErr, log.Close())
		p.reportCleanup(p.cleanupErr, e.report)
	}()
	lost.Bind(p)
	return p, nil
}

// errEnginePeer is a readiness that met a server on the engine's socket
// other than the engine.
var errEnginePeer = engineproc.ErrPeer

// wait polls the engine's own health endpoint until it reports the model
// loaded, the child exits, or the budget passes. A server on the socket that
// is not the engine ends the wait at once with errEnginePeer.
func (p *Process) wait(ctx context.Context, budget time.Duration) error {
	if budget <= 0 {
		budget = DefaultReady
	}
	return engineproc.Wait(ctx, p.owned, budget, p.ready)
}

// ready asks the engine kind's own readiness probe through the engine's
// client.
func (p *Process) ready(ctx context.Context) (bool, error) {
	if p.probe != nil {
		return p.probe(ctx, p.client)
	}
	return p.healthy(ctx)
}

// healthy reads llama.cpp's own /health over the engine's socket, which
// answers 503 while the weights are still being read and 200 once the model
// can answer. A socket nothing serves yet reads not healthy; a server that
// fails the peer check is errEnginePeer.
func (p *Process) healthy(ctx context.Context) (bool, error) {
	call, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(call, http.MethodGet, "http://"+enginehttp.UnixHost+"/health", nil)
	if err != nil {
		return false, nil
	}
	response, err := p.client.Do(request)
	if errors.Is(err, enginehttp.ErrEnginePeer) {
		return false, fmt.Errorf("%w: %v", errEnginePeer, err)
	}
	if err != nil {
		return false, unreachable(err)
	}
	defer response.Body.Close()
	//unchecked: draining the body; the health verdict below depends only on the status code
	io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
	return response.StatusCode == http.StatusOK, nil
}

// unreachable is a failed readiness request's error when its connect can
// never succeed (enginehttp.ErrEngineUnreachable), which ends the wait after
// enginehttp.UnreachableFor; nil for an engine that is not listening yet.
func unreachable(err error) error {
	if errors.Is(err, enginehttp.ErrEngineUnreachable) {
		return err
	}
	return nil
}

// stop ends only the engine this provider started. The kit bounds its
// confirmation; the holder keeps the lease if exited stays open.
func (p *Process) stop(within time.Duration) {
	if p == nil || p.owned == nil {
		return
	}
	p.owned.Stop(within)
}

var errCleanupPending = errors.New("modelhost: engine cleanup did not complete within the stop budget")

func (p *Process) waitCleanup(deadline time.Time) error {
	if p == nil || p.cleaned == nil {
		return nil
	}
	remaining := time.Until(deadline)
	if remaining <= 0 {
		select {
		case <-p.cleaned:
			return p.cleanupErr
		default:
			return errCleanupPending
		}
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case <-p.cleaned:
		return p.cleanupErr
	case <-timer.C:
		select {
		case <-p.cleaned:
			return p.cleanupErr
		default:
			return errCleanupPending
		}
	}
}

func (p *Process) reportCleanup(err error, report func(error)) {
	if err != nil && report != nil {
		p.cleanupReported.Do(func() { report(fmt.Errorf("modelhost: engine cleanup: %w", err)) })
	}
}
