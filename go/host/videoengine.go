package host

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/openabstractions/abstraction-inference/go/providerhost/devicelost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-resource/go/gpufault"

	identity "github.com/openabstractions/abstraction-identity"
)

// A VideoEngine is the sd-server of the pinned stable-diffusion.cpp release
// with the pinned video model set (video addendum; CONTRACT.md
// MODELHOST-V1..MODELHOST-V4). modelhostd stages the engine folder and
// records where each model file lies; both are proved before every start,
// and this package downloads nothing.
type VideoEngine struct {
	// Program is the absolute path of sd-server on this machine.
	Program string
	// Pin is the release Program belongs to; New reads PinnedVideoEngine
	// for this platform when it is zero.
	Pin EnginePin
	// Record is the engine folder's metadata record (INF-F2).
	Record string
	// Set is the model set; New reads FastWan22TI2V5B when it is zero.
	Set VideoModelSet
	// Places is the record of where each pinned model file lies
	// (PlaceVideoModels), and ModelRecord the metadata record of their
	// placed proof (VideoModelPlaces, VideoModelRecord). The files stay
	// where a person or another application keeps them; OA copies none.
	Places, ModelRecord string
	// Files proves the engine folder and the placed model files; New makes
	// one when it is nil.
	Files *enginefiles.Verifier
	// Arguments are the sd-server options a person configured; each must
	// be on the reviewed list (sdcpp.ParseArgs, MODELHOST-V3).
	Arguments []string
	// Ready bounds a load; zero reads DefaultVideoReady.
	Ready time.Duration
	// Log is the absolute file the engine's output is kept in, under the
	// provider's own retention (MODELHOST-V8).
	Log string
	// Temp is the engine's private temporary folder; empty reads
	// engine-temp beside Record.
	Temp string
	// MemoryLimit, when not zero, is the job object's per-process memory
	// limit in bytes (video addendum condition 11).
	MemoryLimit uint64

	// fixture are the stand-in engine's own flags; only tests set them.
	fixture []string
	// beforeLaunch, set by tests only, runs with the chosen port before the
	// engine starts.
	beforeLaunch func(port int)
	report       func(error)
	// deviceLost hears the engine's own device-lost line with the process
	// that printed it (MODELHOST-G1).
	deviceLost func(p *Process, marker string)
	// exited receives the engine's exit record (MODELHOST-E1).
	exited func(EngineExit)
}

// configured reports a video engine configuration that names no pinned
// program or model folder.
func (e VideoEngine) configured() error {
	if e.Program == "" {
		return errors.New("no stable-diffusion.cpp engine is configured; declare one with --video-engine")
	}
	if !filepath.IsAbs(e.Program) {
		return fmt.Errorf("the engine path %s is not absolute", e.Program)
	}
	if info, err := os.Stat(e.Program); err != nil || info.IsDir() {
		return fmt.Errorf("no engine at %s", e.Program)
	}
	if e.Pin.Release == "" {
		return fmt.Errorf("no reviewed stable-diffusion.cpp release is pinned for %s/%s; this provider launches no other", runtime.GOOS, runtime.GOARCH)
	}
	if err := e.Pin.valid(); err != nil {
		return err
	}
	if !identity.SamePathName(filepath.Base(e.Program), e.Pin.Program) {
		return fmt.Errorf("the engine %s is not %s of the pinned stable-diffusion.cpp %s", e.Program, e.Pin.Program, e.Pin.Release)
	}
	if err := e.Set.valid(); err != nil {
		return err
	}
	if !filepath.IsAbs(e.Places) || !filepath.IsAbs(e.ModelRecord) {
		return errors.New("the video model places and their record must be absolute paths")
	}
	if e.Files == nil || !filepath.IsAbs(e.Record) {
		return errors.New("the engine has no verifier or no absolute metadata record")
	}
	return nil
}

func (e VideoEngine) engineTree() enginefiles.Tree {
	return enginefiles.Tree{Name: "stable-diffusion.cpp " + e.Pin.Release, Root: filepath.Dir(e.Program),
		Manifest: e.Pin.manifest(), Record: e.Record}
}

// placed is each pinned model file where the places record says it lies,
// as enginefiles proves it in place (INF-F7).
func (e VideoEngine) placed() ([]enginefiles.Placed, map[string]string, error) {
	where, owners, err := readPlacesOwned(e.Places, e.Set)
	if err != nil {
		return nil, nil, err
	}
	out := make([]enginefiles.Placed, 0, len(e.Set.Files))
	for _, f := range e.Set.Files {
		out = append(out, enginefiles.Placed{Path: where[f.Name], Size: f.Size, SHA256: f.SHA256, Owners: owners[f.Name]})
	}
	return out, where, nil
}

// present proves the engine folder and the placed model files and releases
// them at once; it is the only check before a lease is asked for.
func (e VideoEngine) present(ctx context.Context) error {
	if err := e.configured(); err != nil {
		return err
	}
	if err := e.Files.Check(ctx, e.engineTree()); err != nil {
		return fmt.Errorf("the engine folder %s is not the pinned stable-diffusion.cpp %s (%w); stage %s with modelhostd engine install --kind sdcpp",
			filepath.Dir(e.Program), e.Pin.Release, err, e.Pin.Archive)
	}
	files, _, err := e.placed()
	if err != nil {
		return err
	}
	if err := e.Files.CheckPlaced(ctx, e.Set.Name, files, e.ModelRecord); err != nil {
		return fmt.Errorf("the %s files are not the pinned ones where they were placed (%w); place them again with modelhostd models install --kind sdcpp", e.Set.Name, err)
	}
	return nil
}

// CheckVideoEngine proves program's folder against this platform's pinned
// stable-diffusion.cpp release the way a video does before its lease,
// writes record and releases the folder.
func CheckVideoEngine(ctx context.Context, program, record string) error {
	//unchecked: a platform with no pin leaves pin zero, which configured refuses by name
	pin, _ := PinnedVideoEngine(runtime.GOOS, runtime.GOARCH)
	e := VideoEngine{Program: program, Pin: pin, Record: record, Files: enginefiles.NewVerifier(enginefiles.Options{})}
	if e.Program == "" || !filepath.IsAbs(e.Program) {
		return errors.New("the engine path must be absolute")
	}
	if pin.Release == "" {
		return fmt.Errorf("no reviewed stable-diffusion.cpp release is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	if !identity.SamePathName(filepath.Base(program), pin.Program) {
		return fmt.Errorf("the engine %s is not %s of the pinned stable-diffusion.cpp %s", program, pin.Program, pin.Release)
	}
	return e.Files.Check(ctx, e.engineTree())
}

// CheckVideoModels proves set's files where places says they lie, as a
// video does before its lease.
func CheckVideoModels(ctx context.Context, files *enginefiles.Verifier, set VideoModelSet, places, record string) error {
	e := VideoEngine{Set: set, Places: places, ModelRecord: record}
	placed, _, err := e.placed()
	if err != nil {
		return err
	}
	return files.CheckPlaced(ctx, set.Name, placed, record)
}

// arguments is the checked sd-server command line for port, passing each
// model file at the path where maps its name to: the one start holds. Every
// argument the sd.cpp GPU dispatch row requires is added to the configured
// ones (resource RES-G5, MODELHOST-G1); the set's switches are OR-ed in.
func (e VideoEngine) arguments(port int, where map[string]string) ([]string, error) {
	configured := e.Arguments
	if row, ok := gpufault.DispatchFor(gpufault.EngineSDCpp); ok {
		configured = row.WithArgs(configured)
	}
	a, err := sdcpp.ParseArgs(configured)
	if err != nil {
		return nil, err
	}
	set, err := sdcpp.ParseArgs(e.Set.Options)
	if err != nil {
		return nil, fmt.Errorf("the model set's options: %w", err)
	}
	if a, err = withSwitches(a, set); err != nil {
		return nil, err
	}
	a.ListenPort = port
	for role, path := range map[string]*string{RoleDiffusionModel: &a.DiffusionModel, RoleT5XXL: &a.T5XXL, RoleTAE: &a.TAE} {
		f, ok := e.Set.File(role)
		if !ok {
			return nil, fmt.Errorf("the model set has no file for %s", role)
		}
		if !filepath.IsAbs(where[f.Name]) {
			return nil, fmt.Errorf("no place is recorded for %s", f.Name)
		}
		*path = where[f.Name]
	}
	return a.Build()
}

// withSwitches turns on every switch set carries in a; a person's own
// switches and values stay. A model set passes switches alone: one that
// carries a valued option is refused rather than dropped.
func withSwitches(a, set sdcpp.Args) (sdcpp.Args, error) {
	if set.MaxVRAM != "" || set.Backend != "" || set.Threads != 0 || set.VAETileSize != "" {
		return sdcpp.Args{}, fmt.Errorf("the model set's options carry a valued option; a set passes switches only")
	}
	a.FlashAttention = a.FlashAttention || set.FlashAttention
	a.DiffusionFlashAttention = a.DiffusionFlashAttention || set.DiffusionFlashAttention
	a.DiffusionConvDirect = a.DiffusionConvDirect || set.DiffusionConvDirect
	a.VAEConvDirect = a.VAEConvDirect || set.VAEConvDirect
	a.OffloadToCPU = a.OffloadToCPU || set.OffloadToCPU
	a.AutoFit = a.AutoFit || set.AutoFit
	a.VAETiling = a.VAETiling || set.VAETiling
	a.TemporalTiling = a.TemporalTiling || set.TemporalTiling
	a.Verbose = a.Verbose || set.Verbose
	a.Color = a.Color || set.Color
	return a, nil
}

// videoEnvironmentNames are the engine environment's names: the clean
// environment's, with no key; sd-server has none.
func checkVideoEnvironment(env []string) error {
	seen := map[string]bool{}
	for _, item := range env {
		name, value, ok := strings.Cut(item, "=")
		upper := strings.ToUpper(name)
		if !ok || upper == "LLAMA_API_KEY" || !engineEnvironmentNames[upper] || sdcpp.ForbiddenEnvironment(upper) || seen[upper] || value == "" {
			return fmt.Errorf("%w: %q", ErrEngineEnvironment, name)
		}
		seen[upper] = true
	}
	if len(env) == 0 {
		return fmt.Errorf("%w: empty", ErrEngineEnvironment)
	}
	return nil
}

// start runs sd-server on port with the pinned files, the engine folder
// and the placed model files proved and held until the engine's exit is confirmed (MODELHOST-V1, V2), from
// the clean environment (V4), its output kept in Log (V8).
func (e VideoEngine) start(port int) (*Process, error) {
	if err := e.configured(); err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	placed, _, err := e.placed()
	if err != nil {
		return nil, refuse(NoEngine, "%v", err)
	}
	// A person's arguments are refused before anything is held.
	if _, err := sdcpp.ParseArgs(e.Arguments); err != nil {
		return nil, refuse(Unavailable, "%v", err)
	}
	temp := e.Temp
	if temp == "" {
		temp = filepath.Join(filepath.Dir(e.Record), "engine-temp")
	}
	env, omitted, err := Engine{Temp: temp, Record: e.Record}.cleanEnvironment()
	if err != nil {
		return nil, refuse(Unavailable, "the engine environment: %v", err)
	}
	if err := checkVideoEnvironment(env); err != nil {
		return nil, refuse(Unavailable, "the engine environment: %v", err)
	}
	if e.report != nil {
		for _, diagnostic := range omitted {
			e.report(diagnostic)
		}
	}
	engineHold, err := e.Files.Launch(context.Background(), e.engineTree())
	if err != nil {
		return nil, refuse(NoEngine, "the engine folder: %v", err)
	}
	// The model files are held where they lie, read-shared only, until the
	// engine's exit is confirmed (INF-F2, INF-F7).
	modelHold, err := e.Files.LaunchPlaced(context.Background(), e.Set.Name, placed, e.ModelRecord)
	if err != nil {
		//unchecked: the refusal below is the error returned; a failed close leaves only read-shared handles
		engineHold.Release()
		return nil, refuse(NoEngine, "the %s files: %v", e.Set.Name, err)
	}
	release := func() error {
		return errors.Join(engineHold.Release(), modelHold.Release())
	}
	// sd-server opens each model file by the final path of the handle held
	// for it, every folder on which is held and no reparse point: no
	// junction retargeted after the proof can send it to other bytes.
	where := map[string]string{}
	for i, final := range modelHold.Paths() {
		where[e.Set.Files[i].Name] = final
	}
	args, err := e.arguments(port, where)
	if err != nil {
		release()
		return nil, refuse(Unavailable, "%v", err)
	}
	var output io.Writer
	var log *engineLog
	if e.Log != "" {
		log, err = openEngineLog(e.Log, engineLogBytes)
		if err != nil {
			release()
			return nil, refuse(Unavailable, "the engine log: %v", err)
		}
		output = log
	}
	if e.beforeLaunch != nil {
		e.beforeLaunch(port)
	}
	// The progress tap and the device-lost writer each read every engine
	// line; the device-lost writer sits behind the tap and forwards to the
	// log (MODELHOST-G1).
	lostOutput, lost := engineOutput(output, devicelost.SDServer, e.deviceLost)
	tap := &progressTap{next: lostOutput}
	base := "http://127.0.0.1:" + strconv.Itoa(port)
	var started *Process
	// The client binds each loopback connection to this engine's process
	// tree. Its constructor validates the base before anything is spawned.
	client, err := enginehttp.NewClientWithVerifier(base, func(server identity.Process) (bool, error) {
		if started == nil {
			return false, enginehttp.ErrEnginePeer
		}
		return started.verify(server)
	})
	if err != nil {
		cleanupErr := release()
		if log != nil {
			cleanupErr = errors.Join(cleanupErr, log.Close())
		}
		return nil, refuse(Unavailable, "the engine client: %v", errors.Join(err, cleanupErr))
	}
	p, err := startOwnedProcess(ownedSpec{Program: e.Program, Args: append(args, e.fixture...), Env: env, Output: tap,
		MemoryLimit: e.MemoryLimit, Engine: "sd-server", Model: e.Set.Name, Exited: e.exited})
	if err != nil {
		release()
		if log != nil {
			//unchecked: the start failure is the error returned
			log.Close()
		}
		return nil, err
	}
	started = p
	cleaned := make(chan struct{})
	p.cleaned = cleaned
	go func() {
		<-p.exited
		defer close(cleaned)
		cleanupErr := release()
		if log != nil {
			cleanupErr = errors.Join(cleanupErr, log.Close())
		}
		p.cleanupErr = cleanupErr
		p.reportCleanup(cleanupErr, e.report)
	}()
	p.base = base
	p.port = port
	p.probe = videoReady(p.base)
	p.progress = tap
	p.client = client
	lost.Bind(p)
	return p, nil
}

// videoReady is sd-server's readiness: GET /sdcpp/v1/capabilities answers
// 200 with vid_gen in supported_modes (video addendum condition 6). A model
// that loaded as an image model never becomes ready. A connect that can
// never succeed is enginehttp.ErrEngineUnreachable, as for the chat kind.
func videoReady(base string) func(context.Context, *http.Client) (bool, error) {
	return func(ctx context.Context, client *http.Client) (bool, error) {
		call, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()
		request, err := http.NewRequestWithContext(call, http.MethodGet, base+"/sdcpp/v1/capabilities", nil)
		if err != nil {
			return false, nil
		}
		response, err := client.Do(request)
		if err != nil {
			return false, unreachable(err)
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return false, nil
		}
		var capabilities struct {
			Modes []string `json:"supported_modes"`
		}
		if json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&capabilities) != nil {
			return false, nil
		}
		return slices.Contains(capabilities.Modes, "vid_gen"), nil
	}
}
