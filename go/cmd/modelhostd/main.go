// Command modelhostd is the model host: it loads one model file with a
// llama.cpp this machine already has, holds its residency under a lease of
// abstraction.resource/leases@1, serves abstraction.inference/chat@1 on its own
// OA endpoint for as long as calls arrive, and gives the model and the lease
// back when the card is asked for or nothing has used it for the idle time.
//
// It bundles no engine and downloads nothing.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	resource "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
)

const usage = `modelhostd loads one model file with the llama.cpp this machine already has,
holds its residency under a lease of abstraction.resource/leases@1, and serves
abstraction.inference/chat@1 on its own OA endpoint. One copy is loaded and
every caller the runtime forwards shares it. When the card is asked for, or
when nothing has used the model for the idle time, the engine is stopped and
the lease released.

It bundles no engine and downloads nothing. --engine names a llama-server a
person already installed; without it, and when the path names no file, every
call is refused no_engine.

Usage:
  modelhostd serve --endpoint NAME --engine PATH --model NAME=STORE/OBJECT
                   [--model NAME=STORE/OBJECT]... [--resource-endpoint PATH]
                   [--resource NAME] [--engine-arg ARG]... [--engine-port N]
                   [--embeddings] [--idle DURATION] [--wait DURATION]
                   [--ready DURATION] [--runtime-program PATH]
                   [--config FILE] [--home DIR] [--verbose]
  modelhostd help

Commands:
  serve      answer abstraction.inference/chat@1 (Start, Observe, Cancel), and
             abstraction.inference/embed@1 with --embeddings, on the
             identity-bound local endpoint NAME, for the designated runtime
             program, until interrupted
  help       print this help (also -h, --help, or no arguments)

Options:
  --endpoint NAME          local endpoint name; the platform endpoint is
                           \\.\pipe\openabstractions-NAME on Windows and a
                           socket in a per-account directory elsewhere
  --runtime-program P      absolute runtime executable permitted to call;
                           repeat to name both runtime forms on Windows;
                           default is the installed runtime beside this
                           provider (both forms on Windows)
  --engine PATH            absolute path of the llama-server this machine has
  --engine-arg ARG         one extra argument for the engine; repeatable
  --engine-port N          the loopback port the engine is told to answer on;
                           the default is a free one taken at start
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
                           serving it is listening on. The default is the
                           installed runtime's, or
                           $ABSTRACTION_RESOURCE_ENDPOINT
  --resource NAME          the resource the residency is held on; default card:0
  --idle DURATION          unload a model nothing has asked for; default 5m
  --wait DURATION          how long holders have to yield for one load; default 1m
  --ready DURATION         how long a model has to load; default 3m
  --config FILE            the store configuration
                           abstraction-storage-over-local-stores reads, JSON;
                           default $INVENTORYD_CONFIG
  --home DIR               read this directory as the user's home
  --verbose                write the engine's own output to stderr

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
	enginePort := fs.Int("engine-port", 0, "")
	embeddings := fs.Bool("embeddings", false, "")
	resourceEndpoint := fs.String("resource-endpoint", getenv("ABSTRACTION_RESOURCE_ENDPOINT"), "")
	resourceName := fs.String("resource", host.DefaultResource, "")
	idle := fs.Duration("idle", host.DefaultIdle, "")
	wait := fs.Duration("wait", host.DefaultWait, "")
	ready := fs.Duration("ready", host.DefaultReady, "")
	config := fs.String("config", getenv("INVENTORYD_CONFIG"), "")
	home := fs.String("home", "", "")
	verbose := fs.Bool("verbose", false, "")
	var models, engineArgs repeated
	fs.Var(&models, "model", "")
	fs.Var(&engineArgs, "engine-arg", "")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *endpoint == "" {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprint(stderr, "modelhostd serve: --endpoint is required\n\n"+usage)
		return 2
	}
	if len(models) == 0 {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprint(stderr, "modelhostd serve: at least one --model is required\n\n"+usage)
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

	leaseEndpoint := *resourceEndpoint
	if leaseEndpoint == "" {
		leaseEndpoint = resource.DefaultEndpoint()
	}
	report := func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed model action into success
		fmt.Fprintf(stderr, "modelhostd: %v\n", err)
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
	model, err := host.New(host.Options{
		Models:   allowed,
		Engine:   host.Engine{Program: *engine, Arguments: engineArgs, Embeddings: *embeddings, Ready: *ready, Stderr: engineOut},
		Source:   host.NewLocalStores(env),
		Leases:   resource.New(leaseEndpoint),
		Resource: *resourceName,
		Port:     *enginePort,
		Idle:     *idle,
		Wait:     *wait,
		Active: func(name string) bool {
			if provider == nil {
				return false
			}
			return provider.ActiveForModel(name)
		},
		OnError: report,
	})
	if err != nil {
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
	provider, err = inference.New(inference.Config{
		Router: r,
		// The runtime bound the application and decided its rules before it
		// forwarded; this endpoint admits that runtime's account alone.
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		// Admit loads model's object before the route, so a load that cannot
		// happen is this provider's own refusal word (CONTRACT.md MODELHOST-H2),
		// not the generic dispatcher this package used to serve for it.
		Admit:   service.Admit(model),
		HTTP:    &http.Client{},
		OnError: report,
	})
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
	served.OnError = func(err error) {
		//unchecked: callback has no return path; a failed diagnostic cannot turn a failed call into success
		fmt.Fprintf(stderr, "modelhostd serve: call: %v\n", err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	done := make(chan struct{})
	go func() { defer close(done); model.Run(ctx) }()
	bases := make([]string, 0, len(model.Models()))
	for _, name := range model.Models() {
		bases = append(bases, name+"@"+model.Base(name))
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
// the process connecting to its endpoint. An explicit path supports an
// isolated runtime installation and test fixtures.
func trustedRuntimePrograms(configured []string) ([]string, error) {
	if len(configured) > 0 {
		return runtimeProgramsAt("", configured)
	}
	self, err := os.Executable()
	if err != nil {
		return nil, err
	}
	return runtimeProgramsAt(self, nil)
}

func runtimeProgramsAt(self string, configured []string) ([]string, error) {
	var names []string
	if len(configured) > 0 {
		names = configured
	} else {
		dir := filepath.Dir(self)
		name := "openabstractions"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		names = append(names, filepath.Join(dir, name))
		if runtime.GOOS == "windows" {
			names = append(names, filepath.Join(dir, "openabstractionsw.exe"))
		}
	}
	var programs []string
	for _, name := range names {
		if !identity.ValidSubjectProgram(name) || !filepath.IsAbs(name) {
			return nil, fmt.Errorf("--runtime-program must be a clean absolute executable path")
		}
		info, err := os.Stat(name)
		if err != nil {
			if len(configured) == 0 && os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("runtime program %s: %w", name, err)
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("runtime program %s is not a regular file", name)
		}
		programs = append(programs, identity.CanonicalProgramPath(name))
	}
	if len(programs) == 0 {
		return nil, errors.New("no installed runtime program beside modelhostd")
	}
	return programs, nil
}
