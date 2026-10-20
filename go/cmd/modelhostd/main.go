// Command modelhostd is the model host: it loads one model file with the
// pinned llama.cpp release staged on this machine, holds its residency under a lease of
// abstraction.resource/leases@1, serves abstraction.inference/chat@1 on its own
// OA endpoint for as long as calls arrive, and gives the model and the lease
// back when the card is asked for or nothing has used it for the idle time.
//
// It bundles no engine and downloads nothing.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	"github.com/openabstractions/abstraction-inference/go/providerhost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	resource "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-kit/go/runtimeprog"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
)

const usage = `modelhostd loads one model file with the pinned llama.cpp release staged here,
holds its residency under a lease of abstraction.resource/leases@1, and serves
abstraction.inference/chat@1 on its own OA endpoint. One copy is loaded and
every caller the runtime forwards shares it. When the card is asked for, or
when nothing has used the model for the idle time, the engine is stopped and
the lease released.

The daemon downloads nothing. --engine names the llama-server of the one
llama.cpp release this provider launches, staged by 'modelhostd engine install'
from the reduced ZIP in the OA package or from the publisher's reviewed ZIP.
Before each start the engine folder is hashed against the pin and held
unchanged while the engine runs. Without --engine, when the path names no
file, and when the folder differs from the pin in any file, every call is
refused engine:none.

The pinned release on Windows x64 is llama.cpp b10825, Vulkan:
  archive  llama-b10825-bin-win-vulkan-x64.zip, 35227898 bytes
  SHA-256  8907e5f3b74cb57d93fd94e187a87d85be24dc64e3b0ff2bc1b287da17b31446
  from     https://github.com/ggml-org/llama.cpp/releases/download/b10825/
  bundle   reduced ZIP with exactly the reviewed runtime files; see
           modelhostd engine pin --json for its name, size and SHA-256
No other platform has a reviewed release; there every call is refused.

With --video-model it also serves abstraction.inference/video@1 with the
pinned stable-diffusion.cpp sd-server and one reviewed video model set
(research/security-reviews/stable-diffusion.cpp-master-843-462d675-video-addendum.md):
  archive  sd-master-462d675-bin-win-vulkan-x64.zip, 38865230 bytes
  SHA-256  b9248856707de2e125538ede8a7dc71a9d800bba67e469e60881dae2a5a1c0a2
  from     https://github.com/leejet/stable-diffusion.cpp/releases/download/master-843-462d675/
  models   fastwan2.2-ti2v-5b: FastWan2.2-TI2V-5B-q8_0.gguf,
           umt5-xxl-encoder-Q8_0.gguf, taew2_2.safetensors;
           hummingbird-xt-i2v: AMD's Hummingbird-XT i2v transformer where
           ASUS's AICreator store keeps it, with the same encoder and
           decoder, under the AMD Responsible AI Use Policy
           (research/security-reviews/hummingbird-xt-i2v-files.md);
           each read where 'modelhostd models install' found it
sd-server has no authentication: while it runs, any local process that
finds its loopback port can use it (README "Video").

The OA PyTorch engine is its own provider, pytorchd; --video-kind torch,
--torch-python, --torch-source, --video-model-option, --image-model and
--kind torch are refused with its name.

Usage:
  modelhostd serve --endpoint NAME --engine PATH --model NAME=STORE/OBJECT
                   [--model NAME=STORE/OBJECT]... [--resource-endpoint PATH]
                   [--resource NAME] [--engine-arg ARG]... [--embeddings]
                   [--idle DURATION] [--wait DURATION]
                   [--ready DURATION] [--chat-memory-limit BYTES]
                   [--runtime-program PATH]
                   [--config FILE] [--home DIR] [--state DIR] [--verbose]
                   [--video-model SET --video-engine PATH
                    [--video-engine-arg ARG]... [--video-time-limit DURATION]
                    [--video-ready DURATION]
                    [--video-memory-limit BYTES]]
  modelhostd engine pin --json
  modelhostd engine install [--kind KIND] (--archive ZIP | --bundle ZIP) [--dir DIR] [--state DIR]
  modelhostd engine check [--kind KIND] --engine PATH [--engine-arg ARG]... [--state DIR]
  modelhostd engine unlock [--kind KIND] [--dir DIR] [--state DIR]
  modelhostd models install --kind sdcpp [--set SET] [--from DIR]...
                   [--config FILE] [--home DIR] [--state DIR]
  modelhostd models check --kind sdcpp [--set SET] [--state DIR]
  modelhostd models protect [--models-root DIR] [--home DIR] [--state DIR]
                   [--job-store DIR] [--dry-run]
  modelhostd gpu list [--state DIR]
  modelhostd gpu clear (KEY | --all) [--state DIR]
  modelhostd help

Commands:
  serve           answer abstraction.inference/chat@1 (Start, Observe,
                  Cancel), and abstraction.inference/embed@1 with
                  --embeddings, on the identity-bound local endpoint NAME, for
                  the designated runtime program, until interrupted
  engine pin      print the versioned JSON pin for Setup and package derivation
  engine install  check the publisher ZIP (--archive) or the reduced OA package
                  ZIP (--bundle) against its own pinned size and SHA-256, write
                  exactly the pinned files into DIR (absent or empty;
                  default <user cache>\OpenAbstractions\engines\llama.cpp-
                  <release>), make DIR read-only for this account and print
                  the llama-server path to pass as --engine
  engine check    prove PATH's folder against the pin as a call would;
                  exit 0 when serve would start it, 1 with the reason, and
                  2 when an --engine-arg is outside the reviewed list
  engine unlock   give DIR, an engine folder this provider recorded, back
                  to this account so it can be removed; without --dir, the
                  folder the engine record names, and nothing when no
                  record exists (uninstallers run it this way)
  models install  find each pinned file of SET in each --from folder, by its
                  name and known locators, then in every store the inventory
                  reads, and record where the first one whose size and
                  SHA-256 are the pin's lies; a candidate whose digest
                  differs, or behind a junction or link, is refused. No file
                  is copied or moved: sd-server reads each file where it
                  lies, held read-only while it runs. A file that this
                  account or the machine's administrators alone can change
                  (a --from folder) or the machine's alone (a store) is
                  hashed in full once per provider process; any other at
                  every launch
  models check    prove the recorded files where they lie, as a video call
                  would
  models protect  make each recorded set folder under the models root (one
                  a places record in --state uses, or one named as a set)
                  owned by this account with a protected ACL of this
                  account, SYSTEM and Administrators that every file below
                  carries as inherited, replacing entries a move left on
                  them; run once for set folders made before OA protected
                  them. Every other folder under the root is listed and left
                  alone. A root that is a drive root, the home, a known
                  folder or behind a link is refused; a set folder holding a
                  link, or a file with another name outside the models root
                  and the job store, is refused unchanged. --dry-run reports
                  and writes nothing
  gpu list        print the requests paused after a graphics driver reset:
                  key, strikes, until when, and what they dispatch; a
                  damaged quarantine file is named with where it was moved
  gpu clear       let the request KEY (or every paused request with --all)
                  run again; a running serve reads the change at its next
                  check
  help            print this help (also -h, --help, or no arguments)

GPU faults: serve reads the Windows driver-reset and standby records
(WerKernel 1001 WATCHDOG, LiveKernelEvent 141/117, Display 4101, Kernel-Power
506/507) and each engine's own device-lost line. On a reset it stops every
engine it runs and releases their leases, fails the running request gpu:reset
and pauses an identical request (10 minutes, doubling; held after three until
cleared), and reloads only once the adapter is listed again. In Modern
Standby no new load starts. While a video runs it holds the wake lease. It
never changes the Windows TDR settings (resource CONTRACT.md RES-G7).

Options:
  --endpoint NAME          local endpoint name; the platform endpoint is
                           \\.\pipe\openabstractions-NAME on Windows and a
                           socket in a per-account directory elsewhere
  --runtime-program P      absolute runtime executable permitted to call;
                           repeat to name both runtime forms on Windows;
                           default is the installed runtime beside this
                           provider (both forms on Windows)
  --engine PATH            serve, engine check: absolute path of the
                           llama-server that engine install staged
  --engine-arg ARG         one extra llama-server argument; repeatable. Only
                           the reviewed list passes: options that size, place
                           or schedule the model (--ctx-size, --n-gpu-layers,
                           --threads, --batch-size, --parallel, --flash-attn,
                           --cache-type-k/-v, --pooling, --device and the
                           like) and switches that turn a feature off. An
                           option that loads code, reads a named file, opens
                           a listener or reaches the network (--rpc, --lora,
                           --host, --path, -hf, --model-url, --tools and
                           every other) is refused by name. Under a card
                           lease the engine starts with --no-mmap and its
                           own --fit layer count, under a memory lease with
                           --n-gpu-layers 0; a given
                           --n-gpu-layers, --load-mode, --mmap, --no-mmap or
                           --mlock replaces that default
  --embeddings             start the engine with embeddings and publish
                           abstraction.inference/embed@1 beside chat@1
  --model NAME=STORE/OBJECT
                           one entry of the explicit allowlist: the name
                           callers ask for, and the object the storage
                           inventory names. Repeatable; nothing outside the
                           list is ever loaded
  --resource-endpoint PATH the platform endpoint abstraction.resource/leases@1
                           answers on, not a name: a pipe path on Windows and
                           a socket path elsewhere, which is what the runtime
                           serving it is listening on. This flag in the
                           declaration comes first; without it,
                           $ABSTRACTION_RESOURCE_ENDPOINT, which the runtime
                           that starts the provider sets to its own; without
                           that, the installed runtime's
  --resource NAME          the resource the residency is held on; default card:0
  --idle DURATION          unload a model nothing has asked for; default 5m
  --wait DURATION          how long holders have to yield for one load; default 1m
  --ready DURATION         how long a model has to load; default 3m
  --chat-memory-limit N    serve: optional Windows per-process job limit for
                           llama-server, in bytes; 0 leaves it unset
  --config FILE            the store configuration
                           abstraction-storage-over-local-stores reads, JSON;
                           default $INVENTORYD_CONFIG
  --home DIR               read this directory as the user's home
  --state DIR              this provider's private state, where the engine
                           folder's metadata record lives; default
                           <user cache>\OpenAbstractions\providers\modelhost.
                           Engine sockets live under %LOCALAPPDATA%\oa\s on
                           Windows, $XDG_RUNTIME_DIR/oa or /tmp/oa-<uid> on
                           Linux and $TMPDIR/oa on macOS, whatever --state is
  --verbose                write the engine's own output to stderr
  --archive ZIP            engine install: the publisher's reviewed archive
  --bundle ZIP             engine install: OA's reviewed-files-only package ZIP
  --json                   engine pin: machine-readable pin (schema 1)
  --dir DIR                engine install, engine unlock: the engine folder
  --kind KIND              engine and models commands: llamacpp (the default
                           for engine) or sdcpp (the video engine and its
                           model set)
  --set SET                models: the reviewed video model set,
                           fastwan2.2-ti2v-5b (the default) or
                           hummingbird-xt-i2v
  --from DIR               models install: a folder that may hold the set's
                           files, such as the one a person downloaded them to;
                           repeatable. A set folder under the models root
                           that holds a recorded file is protected as models
                           protect does after placing
  --models-root DIR        models: the folder OA keeps its model sets in, one
                           folder per set; default <home>\OA-models
  --job-store DIR          models: the runtime's job store, whose results may
                           share a file with a set folder; default
                           %LOCALAPPDATA%\openabstractions\runtime-v1\jobs
  --dry-run                models protect: report each recorded set folder
                           and write nothing
  --video-model SET        serve: also serve abstraction.inference/video@1 for
                           the reviewed model set SET with sd-server; the
                           engine starts on the first video and stops an idle
                           time after the last
  --video-engine PATH      serve: absolute path of the sd-server.exe that
                           engine install --kind sdcpp staged
  --video-engine-arg ARG   serve: one extra sd-server option; repeatable. Only
                           --fa, --diffusion-fa, --diffusion-conv-direct,
                           --vae-conv-direct, --offload-to-cpu, --auto-fit,
                           --max-vram N, --backend DEVICE, -t N, --vae-tiling,
                           --temporal-tiling, --vae-tile-size WxH, -v and
                           --color pass; every option the review forbids is
                           refused by name
  --video-time-limit D     serve: how long one video may generate before its
                           engine is stopped; default 30m
  --video-ready D          serve: how long sd-server has to load the model
                           set; default 15m, a cold read of 11.5 GB
  --video-memory-limit N   serve: the engine job's per-process memory limit in
                           bytes; default 42949672960 (40 GiB, the measured
                           peak of the largest preset with a margin) on
                           Windows, and none elsewhere; 0 sets none

Rights: this provider's program needs abstraction.resource/hold on the
resource it holds; the runtime writes that rule for a declared native
inference provider. Applications never reach this endpoint: their calls travel
through the runtime, which binds the caller and decides its rules first.
`

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr, os.Getenv))
}

func run(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if len(args) == 0 {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	switch args[0] {
	case "help", "-h", "--help", "-help":
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	case "serve":
		return runServe(args[1:], stdout, stderr, getenv)
	case "engine":
		return runEngine(args[1:], stdout, stderr)
	case "models":
		return runModels(args[1:], stdout, stderr, getenv)
	case "gpu":
		return runGPU(args[1:], stdout, stderr)
	}
	//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
	fmt.Fprintf(stderr, "modelhostd: unknown command %q\n\n%s", args[0], usage)
	return 2
}

func wantsHelp(args []string) bool {
	for _, a := range args {
		if a == "-h" || a == "--help" || a == "-help" {
			return true
		}
	}
	return false
}

// repeated collects one option given more than once.
type repeated []string

func (r *repeated) String() string     { return strings.Join(*r, ",") }
func (r *repeated) Set(v string) error { *r = append(*r, v); return nil }

// parseModel reads one allowlist entry, NAME=STORE/OBJECT. The object is named
// the way the storage inventory names it and never by path.
func parseModel(entry string) (host.Model, error) {
	name, object, ok := strings.Cut(entry, "=")
	if !ok || name == "" || object == "" {
		return host.Model{}, fmt.Errorf("--model takes NAME=STORE/OBJECT, not %q", entry)
	}
	store, id, ok := strings.Cut(object, "/")
	if !ok || store == "" || id == "" {
		return host.Model{}, fmt.Errorf("--model %q names no object as STORE/OBJECT", entry)
	}
	return host.Model{Name: name, Store: store, Object: id}, nil
}

func runServe(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if wantsHelp(args) {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	endpoint := fs.String("endpoint", "", "")
	var runtimePrograms repeated
	fs.Var(&runtimePrograms, "runtime-program", "")
	engine := fs.String("engine", getenv("MODELHOST_ENGINE"), "")
	embeddings := fs.Bool("embeddings", false, "")
	resourceEndpoint := fs.String("resource-endpoint", getenv("ABSTRACTION_RESOURCE_ENDPOINT"), "")
	resourceName := fs.String("resource", host.DefaultResource, "")
	idle := fs.Duration("idle", host.DefaultIdle, "")
	wait := fs.Duration("wait", host.DefaultWait, "")
	ready := fs.Duration("ready", host.DefaultReady, "")
	chatMemory := fs.Uint64("chat-memory-limit", 0, "")
	config := fs.String("config", getenv("INVENTORYD_CONFIG"), "")
	home := fs.String("home", "", "")
	stateDir := fs.String("state", "", "")
	verbose := fs.Bool("verbose", false, "")
	var models, engineArgs, videoArgs repeated
	fs.Var(&models, "model", "")
	fs.Var(&engineArgs, "engine-arg", "")
	videoModel := fs.String("video-model", "", "")
	videoEngine := fs.String("video-engine", getenv("MODELHOST_VIDEO_ENGINE"), "")
	fs.Var(&videoArgs, "video-engine-arg", "")
	videoTimeLimit := fs.Duration("video-time-limit", host.DefaultVideoTimeLimit, "")
	videoReady := fs.Duration("video-ready", host.DefaultVideoReady, "")
	videoMemory := videoMemoryFlag(fs)
	// The PyTorch engine's options are parsed only to refuse them by name
	// (movedTorch).
	videoKind := fs.String("video-kind", kindSdCpp, "")
	fs.String("torch-python", "", "")
	fs.String("torch-source", "", "")
	var videoOptions repeated
	fs.Var(&videoOptions, "video-model-option", "")
	fs.String("image-model", "", "")
	fs.Duration("image-time-limit", 0, "")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	var torch []string
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "torch-python", "torch-source", "video-model-option", "image-model", "image-time-limit":
			torch = append(torch, "--"+f.Name)
		}
	})
	if *videoKind == kindTorch {
		torch = append([]string{"--video-kind torch"}, torch...)
	}
	if len(torch) > 0 {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %s refused: %s; run 'openabstractions models host' once and it moves this declaration's PyTorch sets to that provider\n",
			strings.Join(torch, ", "), movedTorch)
		return 2
	}
	if *videoKind != kindSdCpp {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: --video-kind is sdcpp, not %q\n", *videoKind)
		return 2
	}
	if *endpoint == "" {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprint(stderr, "modelhostd serve: --endpoint is required\n\n"+usage)
		return 2
	}
	if len(models) == 0 && *videoModel == "" {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprint(stderr, "modelhostd serve: at least one --model or --video-model is required\n\n"+usage)
		return 2
	}
	if _, err := sdcpp.ParseArgs(videoArgs); err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 2
	}
	if _, err := host.EngineArguments(engineArgs); err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 2
	}
	programs, err := trustedRuntimePrograms(runtimePrograms)
	if err != nil {
		//unchecked: failed stderr is only a lost diagnostic; exit status 2 still refuses to start the provider
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 2
	}
	allowed := make([]host.Model, 0, len(models))
	for _, entry := range models {
		m, err := parseModel(entry)
		if err != nil {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
			return 2
		}
		allowed = append(allowed, m)
	}

	cfg, err := host.LoadConfig(*config)
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: configuration: %v\n", err)
		return 2
	}
	env := host.Env{GOOS: runtime.GOOS, Getenv: getenv, Config: cfg, Home: *home}
	if env.Home == "" {
		env.Home = userHome(getenv)
	}
	if env.Home == "" {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprint(stderr, "modelhostd serve: no home directory; pass --home\n")
		return 2
	}

	state, err := stateFolder(*stateDir)
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 2
	}
	//unchecked: a platform with no pinned release leaves pin zero, and every call is refused engine:none
	pin, _ := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)

	leaseEndpoint := *resourceEndpoint
	if leaseEndpoint == "" {
		leaseEndpoint = resource.DefaultEndpoint()
	}
	report := func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed model action into success
		fmt.Fprintf(stderr, "modelhostd: %v\n", err)
	}
	leases := resource.New(leaseEndpoint)
	gpu, err := newGPUGuard(state, leases, stderr, report)
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 1
	}
	engineOut := io.Writer(nil)
	if *verbose {
		engineOut = stderr
	}
	// provider is assigned below, after model; Active closes over it so the
	// idle clock can ask the inference provider whether an operation for the
	// loaded model is still open, once calls start arriving (CONTRACT.md
	// MODELHOST-H6). keep only runs from model.Run, called well after provider is
	// assigned, so the nil check below never actually fires in this binary.
	var provider *inference.Provider
	var video *host.VideoOptions
	var content *providerhost.DesignatedContent
	if *videoModel != "" {
		content, err = providerhost.NewDesignatedContent("modelhost-video", filepath.Join(state, "content"),
			filepath.Join(state, "content-uploads.json"), programs, videoInputLimit)
		if err != nil {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
			return 1
		}
	}
	if *videoModel != "" {
		videoPin, _ := host.PinnedVideoEngine(runtime.GOOS, runtime.GOARCH)
		set, ok := host.VideoModelSetNamed(*videoModel)
		if !ok {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stderr, "modelhostd serve: no reviewed sdcpp video model set is named %q\n", *videoModel)
			return 2
		}
		video = &host.VideoOptions{Model: *videoModel, Store: content.Store, TimeLimit: *videoTimeLimit,
			Engine: host.VideoEngine{Program: *videoEngine, Pin: videoPin, Record: host.VideoEngineRecord(state, videoPin),
				Set: set, Places: host.VideoModelPlaces(state, set), ModelRecord: host.VideoModelRecord(state, set),
				Arguments: videoArgs, Ready: *videoReady, Log: filepath.Join(state, "logs", "sd-server.log"), MemoryLimit: *videoMemory}}
	}
	model, err := host.New(host.Options{
		Models: allowed,
		Engine: host.Engine{Program: *engine, Pin: pin, Record: host.EngineRecord(state, pin),
			Arguments: engineArgs, Embeddings: *embeddings, Ready: *ready, MemoryLimit: *chatMemory, Stderr: engineOut},
		Source:   host.NewLocalStores(env),
		Leases:   leases,
		GPU:      gpu,
		Resource: *resourceName,
		Idle:     *idle,
		Wait:     *wait,
		Active: func(name string) bool {
			if provider == nil {
				return false
			}
			return provider.ActiveForModel(name)
		},
		OnError:      report,
		OnEngineExit: engineExitReport(stderr),
		Video:        video,
	})
	if err != nil {
		if content != nil {
			//unchecked: the construction failure is reported below
			content.Component.Close()
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 2
	}
	defer model.Close()

	// Each resident's engine is its own OpenAI-compatible host, reached at the
	// one address this provider reserved for it: asking the engine would mean
	// starting it, so the router is told the address up front, one entry per
	// allowlist name, never a shared one two engine processes could collide
	// on (CONTRACT.md MODELHOST-H5). Embed is only declared when this engine kind
	// was started with embeddings, so a request for it without --embeddings
	// reads no_host instead of an upstream failure.
	profiles := []string{router.ProfileChat}
	if *embeddings {
		profiles = append(profiles, router.ProfileEmbed)
	}
	r := router.New()
	engineHosts := make([]*router.Host, 0, len(model.Models()))
	for _, name := range model.Models() {
		engineHost := router.OpenAICompatible("modelhost-engine-"+name, model.Base(name), "/v1", []string{name})
		engineHost.Profiles = profiles
		engineHosts = append(engineHosts, engineHost)
	}
	r.SetHosts(engineHosts...)
	// Admit loads model's object after preflight and before execution; a load
	// that cannot happen is this provider's own refusal word (CONTRACT.md
	// MODELHOST-H2). Every engine request passes the media guard
	// (MODELHOST-L8).
	providerConfig := service.ProviderConfig(model, r, report)
	provider, err = inference.New(providerConfig)
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 1
	}
	defer provider.Close()

	served, err := service.Listen(listen.Endpoint(*endpoint), provider, programs...)
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 1
	}
	if content != nil {
		// video@1 and the content store the runtime stages first frames in
		// and reads results from share this endpoint (CONTRACT.md
		// MODELHOST-V9).
		served.Content = content.Component
		served.Video = model.VideoServer()
	}
	served.OnError = func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed call into success
		fmt.Fprintf(stderr, "modelhostd serve: call: %v\n", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	done := make(chan struct{})
	go func() { defer close(done); model.Run(ctx) }()
	go gpu.Run(ctx)
	defer gpu.Flush()
	bases := make([]string, 0, len(model.Models())+1)
	for _, name := range model.Models() {
		bases = append(bases, name+"@"+model.Base(name))
	}
	if *videoModel != "" {
		bases = append(bases, *videoModel+"@sd-server (video@1, started on demand)")
	}
	//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
	fmt.Fprintf(stdout, "modelhostd serving abstraction.inference/chat@1 on %s; engines at %s\n",
		listen.Endpoint(*endpoint), strings.Join(bases, ", "))
	err = served.Serve(ctx)
	stop()
	//unchecked: shutdown path; model.Close was already deferred above (idempotent), this just runs it before the wait below rather than at function exit
	model.Close()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
	}
	if err != nil {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd serve: %v\n", err)
		return 1
	}
	return 0
}

func userHome(getenv func(string) string) string {
	if runtime.GOOS == "windows" {
		if h := getenv("USERPROFILE"); h != "" {
			return h
		}
	} else if h := getenv("HOME"); h != "" {
		return h
	}
	//unchecked: a failure just leaves h empty, which the caller already checks and reports as "no home directory"
	h, _ := os.UserHomeDir()
	return h
}

// trustedRuntimePrograms is fixed by this provider's installation, never by
// the process connecting to its endpoint (runtimeprog, KIT-E1).
func trustedRuntimePrograms(configured []string) ([]string, error) {
	return runtimeprog.Trusted(configured)
}

func runtimeProgramsAt(self string, configured []string) ([]string, error) {
	return runtimeprog.At(self, configured)
}

// stateFolder is this provider's private state: dir when it is given, or
// <user cache>/OpenAbstractions/providers/modelhost. It is made when absent.
func stateFolder(dir string) (string, error) {
	dir, err := statePath(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

// statePath is the state folder stateFolder names, without making it.
func statePath(dir string) (string, error) {
	if dir == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return "", fmt.Errorf("no state folder; pass --state: %w", err)
		}
		dir = filepath.Join(cache, "OpenAbstractions", "providers", "modelhost")
	}
	if !filepath.IsAbs(dir) {
		return "", fmt.Errorf("--state %s is not an absolute path", dir)
	}
	return filepath.Clean(dir), nil
}

// runEngine stages and releases the pinned engine folder (VISION 2026-10-01).
func runEngine(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 || wantsHelp(args) || args[0] == "help" {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("engine "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	archive := fs.String("archive", "", "")
	bundle := fs.String("bundle", "", "")
	jsonOutput := fs.Bool("json", false, "")
	dir := fs.String("dir", "", "")
	engine := fs.String("engine", "", "")
	stateDir := fs.String("state", "", "")
	kind := fs.String("kind", kindLlamaCpp, "")
	source := fs.String("source", "", "")
	setName := fs.String("set", "", "")
	var engineArgs repeated
	fs.Var(&engineArgs, "engine-arg", "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd engine %s: %s\n", args[0], fmt.Sprintf(format, a...))
		return code
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	if len(engineArgs) > 0 && args[0] != "check" {
		return fail(2, "--engine-arg belongs to engine check and serve")
	}
	if *bundle != "" && (args[0] != "install" || *kind != kindLlamaCpp) {
		return fail(2, "--bundle belongs to llama.cpp engine install")
	}
	if *jsonOutput && args[0] != "pin" {
		return fail(2, "--json belongs to engine pin")
	}
	if _, err := host.EngineArguments(engineArgs); err != nil {
		return fail(2, "%v", err)
	}
	if *kind == kindTorch {
		return fail(2, "%s; check its interpreter with pytorchd engine check", movedTorch)
	}
	if *source != "" || *setName != "" {
		return fail(2, "--source and --set belonged to the PyTorch engine; %s", movedTorch)
	}
	switch *kind {
	case kindLlamaCpp:
	case kindSdCpp:
		return runVideoEngine(args[0], *archive, *dir, *engine, *stateDir, engineArgs, stdout, fail)
	default:
		return fail(2, "--kind is llamacpp or sdcpp, not %q", *kind)
	}
	switch args[0] {
	case "pin":
		if !*jsonOutput || *archive != "" || *bundle != "" || *dir != "" || *engine != "" || *stateDir != "" {
			return fail(2, "engine pin takes --json alone")
		}
		pin, ok := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)
		if !ok {
			return fail(1, "no reviewed llama.cpp release is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		if err := json.NewEncoder(stdout).Encode(pin.Info()); err != nil {
			return fail(1, "writing engine pin: %v", err)
		}
		return 0
	case "install":
		pin, ok := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)
		if !ok {
			return fail(1, "no reviewed llama.cpp release is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		if (*archive == "") == (*bundle == "") {
			return fail(2, "give one of --archive (publisher ZIP) or --bundle (reduced OA package ZIP)")
		}
		target := *dir
		if target == "" {
			cache, err := os.UserCacheDir()
			if err != nil {
				return fail(2, "no default engine folder; pass --dir: %v", err)
			}
			target = filepath.Join(cache, "OpenAbstractions", "engines", "llama.cpp-"+pin.Release)
		}
		state, err := stateFolder(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		input := *archive
		if *bundle != "" {
			input = *bundle
		}
		source, err := filepath.Abs(input)
		if err != nil {
			return fail(2, "%v", err)
		}
		target, err = filepath.Abs(target)
		if err != nil {
			return fail(2, "%v", err)
		}
		verifier := enginefiles.NewVerifier(enginefiles.Options{})
		var program string
		if *bundle != "" {
			program, err = host.InstallBundledEngine(context.Background(), verifier,
				pin, source, target, host.EngineRecord(state, pin))
		} else {
			program, err = host.InstallEngine(context.Background(), verifier,
				pin, source, target, host.EngineRecord(state, pin))
		}
		if err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "installed llama.cpp %s; pass --engine %s\n", pin.Release, program)
		return 0
	case "check":
		if *engine == "" || !filepath.IsAbs(*engine) {
			return fail(2, "--engine names the absolute llama-server path")
		}
		//unchecked: a platform with no pin leaves pin zero, which CheckEngine refuses by name
		pin, _ := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)
		state, err := stateFolder(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		if err := host.CheckEngine(context.Background(), *engine, host.EngineRecord(state, pin)); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s is llama.cpp %s as pinned\n", *engine, pin.Release)
		return 0
	case "unlock":
		if *dir != "" && !filepath.IsAbs(*dir) {
			return fail(2, "--dir names the absolute engine folder")
		}
		//unchecked: a platform with no pin names a record no install wrote, which RecordedEngine refuses
		pin, _ := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)
		state, err := statePath(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		record := host.EngineRecord(state, pin)
		target := *dir
		if target == "" {
			// An uninstaller names no folder: the one the record names is
			// this provider's, and no record means nothing was installed.
			target, err = host.RecordedRoot(record)
			if errors.Is(err, os.ErrNotExist) {
				//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
				fmt.Fprintf(stdout, "no engine record at %s; nothing to unlock\n", record)
				return 0
			}
			if err != nil {
				return fail(1, "%v", err)
			}
		}
		// Only a folder this provider installed or proved is given back; any
		// other folder's ACL is not this command's to change.
		if err := host.RecordedEngine(record, target); err != nil {
			return fail(1, "%v", err)
		}
		// UnlockWithin also reaches the subfolders a killed unlock left
		// read-only, and a recorded folder already removed has nothing to
		// give back.
		if err := enginefiles.UnlockWithin(target); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s belongs to this account again; modelhost refuses it until it is installed again\n", target)
		return 0
	}
	return fail(2, "unknown command; see modelhostd help")
}
