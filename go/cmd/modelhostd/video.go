package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"

	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// Engine kinds the engine and models commands take. The OA PyTorch engine
// is its own provider, pytorchd; kindTorch is named only to refuse it.
const (
	kindLlamaCpp = "llamacpp"
	kindSdCpp    = "sdcpp"
	kindTorch    = "torch"
)

// movedTorch is the refusal of every PyTorch engine option: the engine is
// its own provider (VISION 2026-10-02, "OA may supply an engine process").
const movedTorch = "the OA PyTorch engine is its own provider now: pytorchd serves its video and image sets, and 'openabstractions models host' declares it"

// defaultVideoMemoryLimit is the engine job's memory limit modelhostd sets
// by default: host.DefaultVideoMemoryLimit where a Windows job carries it,
// and none elsewhere.
func defaultVideoMemoryLimit() uint64 {
	if runtime.GOOS == "windows" {
		return host.DefaultVideoMemoryLimit
	}
	return 0
}

// videoMemoryFlag defines serve's --video-memory-limit on fs, defaulting to
// defaultVideoMemoryLimit.
func videoMemoryFlag(fs *flag.FlagSet) *uint64 {
	return fs.Uint64("video-memory-limit", defaultVideoMemoryLimit(), "")
}

// videoInputLimit bounds one first frame the runtime stages in the
// provider's content store; sd-server's frame bound is 16 MiB.
const videoInputLimit = 20 << 20

// defaultFolder is <user cache>/OpenAbstractions/<kind>/<name>.
func defaultFolder(kind, name string) (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("no default folder; pass --dir: %w", err)
	}
	return filepath.Join(cache, "OpenAbstractions", kind, name), nil
}

// runVideoEngine stages, checks and releases the pinned sd-server folder
// (video addendum condition 1).
func runVideoEngine(command, archive, dir, engine, stateDir string, engineArgs []string, stdout io.Writer, fail func(int, string, ...any) int) int {
	if len(engineArgs) > 0 {
		return fail(2, "--engine-arg belongs to llama-server; serve takes --video-engine-arg")
	}
	pin, ok := host.PinnedVideoEngine(runtime.GOOS, runtime.GOARCH)
	switch command {
	case "install":
		if !ok {
			return fail(1, "no reviewed stable-diffusion.cpp release is pinned for %s/%s", runtime.GOOS, runtime.GOARCH)
		}
		if archive == "" {
			return fail(2, "--archive names the downloaded %s", pin.Archive)
		}
		target := dir
		if target == "" {
			folder, err := defaultFolder("engines", "sd.cpp-"+pin.Release)
			if err != nil {
				return fail(2, "%v", err)
			}
			target = folder
		}
		state, err := stateFolder(stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		source, err := filepath.Abs(archive)
		if err != nil {
			return fail(2, "%v", err)
		}
		if target, err = filepath.Abs(target); err != nil {
			return fail(2, "%v", err)
		}
		program, err := host.InstallEngine(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}),
			pin, source, target, host.VideoEngineRecord(state, pin))
		if err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "installed stable-diffusion.cpp %s; pass --video-engine %s\n", pin.Release, program)
		return 0
	case "check":
		if engine == "" || !filepath.IsAbs(engine) {
			return fail(2, "--engine names the absolute sd-server path")
		}
		state, err := stateFolder(stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		if err := host.CheckVideoEngine(context.Background(), engine, host.VideoEngineRecord(state, pin)); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s is stable-diffusion.cpp %s as pinned\n", engine, pin.Release)
		return 0
	case "unlock":
		return unlockRecorded(dir, stateDir, func(state string) string { return host.VideoEngineRecord(state, pin) }, stdout, fail)
	}
	return fail(2, "unknown command; see modelhostd help")
}

// unlockRecorded gives back the folder a record names, or dir when it is
// the recorded one; no record means nothing was installed.
func unlockRecorded(dir, stateDir string, record func(string) string, stdout io.Writer, fail func(int, string, ...any) int) int {
	if dir != "" && !filepath.IsAbs(dir) {
		return fail(2, "--dir names the absolute folder")
	}
	state, err := statePath(stateDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	path := record(state)
	target := dir
	if target == "" {
		target, err = host.RecordedRoot(path)
		if errors.Is(err, os.ErrNotExist) {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stdout, "no record at %s; nothing to unlock\n", path)
			return 0
		}
		if err != nil {
			return fail(1, "%v", err)
		}
	}
	if err := host.RecordedEngine(path, target); err != nil {
		return fail(1, "%v", err)
	}
	if err := enginefiles.UnlockWithin(target); err != nil {
		return fail(1, "%v", err)
	}
	//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
	fmt.Fprintf(stdout, "%s belongs to this account again; modelhost refuses it until it is installed again\n", target)
	return 0
}

// runModels records where the pinned files of a video model set lie and
// proves them there (video addendum condition 2). It copies no file: OA
// never copies a model for an engine (VISION 2026-10-01).
func runModels(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	if len(args) == 0 || wantsHelp(args) || args[0] == "help" {
		if _, err := io.WriteString(stdout, usage); err != nil {
			return 1
		}
		return 0
	}
	fs := flag.NewFlagSet("models "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	//unchecked: usage text on a help/error path; a failed write to stderr has nothing further to report
	fs.Usage = func() { fmt.Fprint(stderr, usage) }
	kind := fs.String("kind", "", "")
	setName := fs.String("set", host.FastWan22TI2V5B.Name, "")
	var from []string
	fs.Func("from", "", func(v string) error {
		from = append(from, v)
		return nil
	})
	config := fs.String("config", getenv("INVENTORYD_CONFIG"), "")
	home := fs.String("home", "", "")
	stateDir := fs.String("state", "", "")
	modelsRoot := fs.String("models-root", "", "")
	jobStore := fs.String("job-store", defaultJobStore(getenv), "")
	dryRun := fs.Bool("dry-run", false, "")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	fail := func(code int, format string, a ...any) int {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stderr, "modelhostd models %s: %s\n", args[0], fmt.Sprintf(format, a...))
		return code
	}
	if fs.NArg() != 0 {
		return fail(2, "unexpected arguments %q", fs.Args())
	}
	root := *modelsRoot
	if root == "" {
		h := *home
		if h == "" {
			h = userHome(getenv)
		}
		if h != "" {
			root = host.ModelsRoot(h)
		}
	}
	if root != "" && !filepath.IsAbs(root) {
		return fail(2, "--models-root names an absolute folder")
	}
	var linkRoots []string
	if *jobStore != "" {
		if !filepath.IsAbs(*jobStore) {
			return fail(2, "--job-store names an absolute folder")
		}
		linkRoots = append(linkRoots, filepath.Clean(*jobStore))
	}
	if args[0] == "protect" {
		state, err := statePath(*stateDir)
		if err != nil {
			return fail(2, "%v", err)
		}
		return protectModelFolders(root, state, linkRoots, *dryRun, stdout, fail)
	}
	if *dryRun {
		return fail(2, "--dry-run belongs to models protect")
	}
	if *kind == kindTorch {
		return fail(2, "%s; place its sets with pytorchd models install", movedTorch)
	}
	if *kind != kindSdCpp {
		return fail(2, "--kind sdcpp names a video model set; llama.cpp models are store objects (--model)")
	}
	set, ok := host.VideoModelSetNamed(*setName)
	if !ok {
		return fail(2, "no reviewed video model set is named %q", *setName)
	}
	state, err := stateFolder(*stateDir)
	if err != nil {
		return fail(2, "%v", err)
	}
	places, record := host.VideoModelPlaces(state, set), host.VideoModelRecord(state, set)
	switch args[0] {
	case "install":
		searches, code := modelSearches(from, *config, *home, getenv, fail)
		if code != 0 {
			return code
		}
		found, err := host.PlaceVideoModels(context.Background(), set, searches, places)
		if err != nil {
			return fail(1, "%v", err)
		}
		// Each set folder of OA's own that now holds a recorded file is made
		// a model folder, so its files pass the trusted-in-place check; no
		// other folder's access is changed (identity ID-D3).
		if code := adoptSetFolders(root, linkRoots, set.Files, found, stdout, fail); code != 0 {
			return code
		}
		if err := host.CheckVideoModels(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}), set, places, record); err != nil {
			return fail(1, "%v", err)
		}
		for _, f := range set.Files {
			//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
			fmt.Fprintf(stdout, "%s: %s\n", f.Name, found[f.Name])
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "recorded the %s files where they lie; nothing was copied; pass --video-model %s\n", set.Name, set.Name)
		return 0
	case "check":
		if len(from) > 0 {
			return fail(2, "--from belongs to models install")
		}
		if err := host.CheckVideoModels(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}), set, places, record); err != nil {
			return fail(1, "%v", err)
		}
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "the %s files are the pinned ones where they were recorded\n", set.Name)
		return 0
	}
	return fail(2, "unknown command; see modelhostd help")
}

// adoptSetFolders makes each folder directly under root that holds a file
// found for files a model folder of this account (modelplace, KIT-M3).
func adoptSetFolders(root string, linkRoots []string, files []host.VideoModelFile, found map[string]string, stdout io.Writer, fail func(int, string, ...any) int) int {
	err := modelplace.AdoptSetFolders(root, linkRoots, files, found, func(dir string) {
		//unchecked: CLI output; the process exit code, not a returned error, already reports the outcome
		fmt.Fprintf(stdout, "%s: OA's model folder, protected\n", dir)
	})
	if err != nil {
		return fail(1, "%v", err)
	}
	return 0
}

// modelSearches are the --from folders, then every store the inventory
// reads; code is not zero when none can be searched.
func modelSearches(from []string, config, home string, getenv func(string) string, fail func(int, string, ...any) int) ([]host.ModelSearch, int) {
	cfg, err := host.LoadConfig(config)
	if err != nil {
		return nil, fail(2, "configuration: %v", err)
	}
	env := host.Env{GOOS: runtime.GOOS, Getenv: getenv, Config: cfg, Home: home}
	if env.Home == "" {
		env.Home = userHome(getenv)
	}
	var source host.Source
	if env.Home != "" {
		source = host.NewLocalStores(env)
	}
	searches, err := modelplace.Searches(from, host.StoreFiles(source))
	if err != nil {
		return nil, fail(2, "%v", err)
	}
	return searches, 0
}

// protectModelFolders makes each recorded set folder under root a model
// folder of this account (modelplace.ProtectFolders, KIT-M3): a folder a
// places record in state uses, or one named as a set. Any other folder under
// root is listed and left as it is.
func protectModelFolders(root, state string, linkRoots []string, dryRun bool, stdout io.Writer, fail func(int, string, ...any) int) int {
	if root == "" {
		return fail(2, "no models root; pass --models-root DIR or --home DIR")
	}
	recorded, err := host.RecordedSetFolders(filepath.Clean(root), state)
	if err != nil {
		return fail(1, "%v", err)
	}
	failed, err := modelplace.ProtectFolders(root, recorded, linkRoots, dryRun, stdout)
	if err != nil {
		return fail(1, "%v", err)
	}
	if failed > 0 {
		return fail(1, "%d set folders were not protected", failed)
	}
	return 0
}

// defaultJobStore is the runtime's managed job store (modelplace).
func defaultJobStore(getenv func(string) string) string { return modelplace.DefaultJobStore(getenv) }
