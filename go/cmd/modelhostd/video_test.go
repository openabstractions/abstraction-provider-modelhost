package main

import (
	"bytes"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-storage/go/layout"
)

func TestDefaultVideoEngineFolderSharesAccountDataEngines(t *testing.T) {
	cache := accountDataFixture(t)
	got, err := defaultVideoEngineFolder("fixture-release")
	want := filepath.Join(cache, "OpenAbstractions", "engines", "sd.cpp-fixture-release")
	if err != nil || got != want {
		t.Fatalf("video default changed: %q %v, want %q", got, err, want)
	}
	if err := os.MkdirAll(want, 0700); err != nil {
		t.Fatal(err)
	}
	if same, err := identity.SameFile(filepath.Dir(got), layout.EngineCacheRoot(cache)); err != nil || !same {
		t.Fatalf("video uses another existing engines root: %v %v", same, err)
	}
}

func TestDefaultVideoEngineFolderPreservesNativeError(t *testing.T) {
	unsetAccountData(t)
	_, nativeErr := layout.AccountData()
	got, err := defaultVideoEngineFolder("fixture-release")
	if nativeErr == nil || err == nil || got != "" || err.Error() != "no default folder; pass --dir: "+nativeErr.Error() {
		t.Fatalf("video native error changed: root=%q error=%v native=%v", got, err, nativeErr)
	}
}

func noEnv(string) string { return "" }

// serve's --video-memory-limit, given nothing, is 40 GiB on Windows, where
// the engine's job carries it, and none elsewhere; a given value stands.
func TestVideoMemoryLimitDefault(t *testing.T) {
	want := uint64(0)
	if runtime.GOOS == "windows" {
		want = 40 << 30
	}
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	limit := videoMemoryFlag(fs)
	if err := fs.Parse(nil); err != nil || *limit != want {
		t.Fatalf("default %d, want %d (%v)", *limit, want, err)
	}
	fs = flag.NewFlagSet("serve", flag.ContinueOnError)
	limit = videoMemoryFlag(fs)
	if err := fs.Parse([]string{"--video-memory-limit", "0"}); err != nil || *limit != 0 {
		t.Fatalf("given 0: %d (%v)", *limit, err)
	}
}

// engine install --kind sdcpp refuses an archive other than the pinned
// stable-diffusion.cpp release and writes no folder for it (video addendum
// condition 1).
func TestVideoEngineInstallRefusesAnUnpinnedArchive(t *testing.T) {
	archive := filepath.Join(t.TempDir(), "sd-master-462d675-bin-win-vulkan-x64.zip")
	if err := os.WriteFile(archive, []byte("not the release"), 0o600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "engine")
	var out, errs bytes.Buffer
	status := run([]string{"engine", "install", "--kind", "sdcpp", "--archive", archive, "--dir", dir, "--state", t.TempDir()}, &out, &errs, noEnv)
	want := "is not the pinned sd-master-462d675-bin-win-vulkan-x64.zip"
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" {
		want = "no reviewed stable-diffusion.cpp release is pinned"
	}
	if status != 1 || !strings.Contains(errs.String(), want) {
		t.Fatalf("status %d, stderr %q; wanted 1 and %q", status, errs.String(), want)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("a refused archive left %s: %v", dir, err)
	}
}

// engine check --kind sdcpp refuses a program that is not the pinned
// sd-server.
func TestVideoEngineCheckRefusesAnUnpinnedEngine(t *testing.T) {
	program := filepath.Join(t.TempDir(), "sd-server.exe")
	if err := os.WriteFile(program, []byte("not the release"), 0o700); err != nil {
		t.Fatal(err)
	}
	var out, errs bytes.Buffer
	if status := run([]string{"engine", "check", "--kind", "sdcpp", "--engine", program, "--state", t.TempDir()}, &out, &errs, noEnv); status != 1 {
		t.Fatalf("status %d, stderr %q", status, errs.String())
	}
}

// models install refuses found files that are not the pinned set, records
// no places and copies nothing; models check without a record says how to
// place the set (video addendum condition 2, VISION 2026-10-01).
func TestVideoModelsInstallRefusesUnpinnedFiles(t *testing.T) {
	from, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	names := []string{"FastWan2.2-TI2V-5B-q8_0.gguf", "umt5-xxl-encoder-Q8_0.gguf", "taew2_2.safetensors"}
	for _, name := range names {
		if err := os.WriteFile(filepath.Join(from, name), []byte("not the weights"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	state := t.TempDir()
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--kind", "sdcpp", "--from", from, "--home", t.TempDir(), "--state", state}, &out, &errs, noEnv)
	if status != 1 || !strings.Contains(errs.String(), "not the pinned") {
		t.Fatalf("status %d, stderr %q", status, errs.String())
	}
	if entries, err := os.ReadDir(state); err != nil || len(entries) != 0 {
		t.Fatalf("a refused set left state %v: %v", entries, err)
	}
	if entries, err := os.ReadDir(from); err != nil || len(entries) != len(names) {
		t.Fatalf("the searched folder changed: %v %v", entries, err)
	}
	errs.Reset()
	if status := run([]string{"models", "check", "--kind", "sdcpp", "--state", state}, &out, &errs, noEnv); status != 1 || !strings.Contains(errs.String(), "models install") {
		t.Fatalf("check without a record: status %d, stderr %q", status, errs.String())
	}
	for _, args := range [][]string{{"models", "install", "--from", from}, {"models", "check", "--kind", "llamacpp"}} {
		errs.Reset()
		if status := run(args, &out, &errs, noEnv); status != 2 || !strings.Contains(errs.String(), "--kind sdcpp") {
			t.Fatalf("%q: status %d, stderr %q", args, status, errs.String())
		}
	}
	for _, args := range [][]string{
		{"models", "install", "--kind", "sdcpp", "--set", "unreviewed", "--from", from},
		{"models", "check", "--kind", "sdcpp", "--from", from, "--state", state},
		{"serve", "--endpoint", "modelhost-video-set-test", "--video-model", "unreviewed", "--state", state},
	} {
		errs.Reset()
		if status := run(args, &out, &errs, noEnv); status != 2 {
			t.Fatalf("%q: status %d, stderr %q", args, status, errs.String())
		}
	}
}

// The OA PyTorch engine is its own provider (VISION 2026-10-02): every
// PyTorch option of serve, engine and models is refused before anything
// starts, naming pytorchd and how to redeclare it; an unknown video kind is
// refused as before.
func TestTheTorchKindIsRefusedNamingItsProvider(t *testing.T) {
	state := t.TempDir()
	from := t.TempDir()
	var out, errs bytes.Buffer
	for _, args := range [][]string{
		{"serve", "--endpoint", "modelhost-torch-test", "--video-model", "hummingbird-xt-i2v", "--video-kind", "torch", "--state", state},
		{"serve", "--endpoint", "modelhost-torch-test", "--video-model", "hummingbird-xt-i2v", "--torch-python", from + `\python.exe`, "--state", state},
		{"serve", "--endpoint", "modelhost-torch-test", "--video-model", "hummingbird-xt-i2v", "--torch-source", from, "--state", state},
		{"serve", "--endpoint", "modelhost-torch-test", "--video-model", "fastwan2.2-ti2v-5b", "--video-model-option", "decode-dtype=bf16", "--state", state},
		{"serve", "--endpoint", "modelhost-torch-test", "--image-model", "flux2-klein-4b", "--state", state},
		{"models", "install", "--kind", "torch", "--from", from, "--state", state},
		{"models", "check", "--kind", "torch", "--state", state},
		{"engine", "check", "--kind", "torch", "--engine", from + `\python.exe`, "--state", state},
		{"engine", "check", "--set", "flux2-klein-4b", "--engine", from + `\python.exe`, "--state", state},
	} {
		errs.Reset()
		if status := run(args, &out, &errs, noEnv); status != 2 || !strings.Contains(errs.String(), "pytorchd") ||
			!strings.Contains(errs.String(), "models host") {
			t.Fatalf("%q: status %d, stderr %q", args, status, errs.String())
		}
		// A serve line declared before the PyTorch provider says how to move
		// it there.
		if args[0] == "serve" && !strings.Contains(errs.String(), "moves this declaration's PyTorch sets") {
			t.Fatalf("%q: the refusal does not say how to redeclare: %q", args, errs.String())
		}
	}
	errs.Reset()
	if status := run([]string{"serve", "--endpoint", "modelhost-torch-test", "--video-model", "hummingbird-xt-i2v", "--video-kind", "onnx", "--state", state}, &out, &errs, noEnv); status != 2 {
		t.Fatalf("an unknown video kind: status %d, stderr %q", status, errs.String())
	}
	if entries, err := os.ReadDir(state); err != nil || len(entries) != 0 {
		t.Fatalf("a refused command left state %v: %v", entries, err)
	}
}

// serve refuses a video engine option outside the reviewed list by name,
// before anything starts (video addendum conditions 3 and 4).
func TestVideoEngineArgumentIsRefusedByName(t *testing.T) {
	for _, arg := range []string{"--lora-model-dir=x", "-M", "--listen-ip=0.0.0.0", "--steps"} {
		var out, errs bytes.Buffer
		status := run([]string{"serve", "--endpoint", explicitFixtureEndpoint(t, "modelhost-video-arg-test"), "--video-model", "fastwan2.2-ti2v-5b",
			"--video-engine-arg", arg}, &out, &errs, noEnv)
		name, _, _ := strings.Cut(arg, "=")
		if status != 2 || !strings.Contains(errs.String(), name) {
			t.Fatalf("%s: status %d, stderr %q", arg, status, errs.String())
		}
	}
}

// accountDataFixture points layout.AccountData at a fresh folder and returns
// it: %LOCALAPPDATA% on Windows, the data root under $HOME on macOS and
// $XDG_DATA_HOME on Linux.
func accountDataFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	switch runtime.GOOS {
	case "windows":
		t.Setenv("LOCALAPPDATA", dir)
		return dir
	case "darwin":
		t.Setenv("HOME", dir)
		return filepath.Join(dir, "Library", "Application Support", "openabstractions")
	default:
		t.Setenv("HOME", dir)
		t.Setenv("XDG_DATA_HOME", filepath.Join(dir, "data"))
		return filepath.Join(dir, "data", "openabstractions")
	}
}

// unsetAccountData leaves layout.AccountData no base to resolve.
func unsetAccountData(t *testing.T) {
	t.Helper()
	t.Setenv("LOCALAPPDATA", "")
	t.Setenv("HOME", "")
	t.Setenv("XDG_DATA_HOME", "")
}
