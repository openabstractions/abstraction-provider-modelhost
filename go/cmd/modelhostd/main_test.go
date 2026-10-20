package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

type failedWriter struct{}

func (failedWriter) Write([]byte) (int, error) { return 0, errors.New("output unavailable") }

func TestHelpOutputFailureIsNotSuccess(t *testing.T) {
	if status := run([]string{"help"}, failedWriter{}, &bytes.Buffer{}, func(string) string { return "" }); status == 0 {
		t.Fatal("failed help output reported success")
	}
}

// engine install refuses an archive other than the pinned release and writes
// no engine folder for it.
func TestEngineInstallRefusesAnUnpinnedArchive(t *testing.T) {
	archive := filepath.Join(t.TempDir(), "llama-b10825-bin-win-vulkan-x64.zip")
	if err := os.WriteFile(archive, []byte("not the release"), 0o600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "engine")
	var out, errs bytes.Buffer
	status := run([]string{"engine", "install", "--archive", archive, "--dir", dir, "--state", t.TempDir()},
		&out, &errs, func(string) string { return "" })
	want := "is not the pinned llama-b10825-bin-win-vulkan-x64.zip"
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" {
		want = "no reviewed llama.cpp release is pinned"
	}
	if status != 1 || !bytes.Contains(errs.Bytes(), []byte(want)) {
		t.Fatalf("status %d, stderr %q; wanted 1 and %q", status, errs.String(), want)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("a refused archive left %s: %v", dir, err)
	}
}

func TestEnginePinJSONNamesTheSameReviewedFilesAndBundle(t *testing.T) {
	var out, errs bytes.Buffer
	status := run([]string{"engine", "pin", "--json"}, &out, &errs, func(string) string { return "" })
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" {
		if status != 1 || !bytes.Contains(errs.Bytes(), []byte("no reviewed llama.cpp release")) {
			t.Fatalf("unreviewed platform: status %d, stderr %q", status, errs.String())
		}
		return
	}
	if status != 0 {
		t.Fatalf("status %d, stderr %q", status, errs.String())
	}
	var got host.EnginePinInfo
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	want := host.LlamaCppB10825WindowsVulkan.Info()
	if got.Schema != 1 || got.Release != want.Release || got.Program != want.Program ||
		got.Archive != want.Archive || got.Bundle != want.Bundle || len(got.Files) != len(want.Files) {
		t.Fatalf("unexpected pin JSON: %s", out.String())
	}
	for i := range got.Files {
		if got.Files[i] != want.Files[i] {
			t.Fatalf("file %d differs from compiled pin", i)
		}
	}
}

func TestEngineInstallSelectsOneArchiveSource(t *testing.T) {
	var out, errs bytes.Buffer
	status := run([]string{"engine", "install", "--archive", "original.zip", "--bundle", "reduced.zip"},
		&out, &errs, func(string) string { return "" })
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		if status != 2 || !bytes.Contains(errs.Bytes(), []byte("give one of --archive")) {
			t.Fatalf("status %d, stderr %q", status, errs.String())
		}
	}
}

func TestEngineInstallRefusesAnUnpinnedBundle(t *testing.T) {
	bundle := filepath.Join(t.TempDir(), "llama-b10825-oa-win-vulkan-x64.zip")
	if err := os.WriteFile(bundle, []byte("wrong bundle"), 0o600); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "engine")
	var out, errs bytes.Buffer
	status := run([]string{"engine", "install", "--bundle", bundle, "--dir", dir, "--state", t.TempDir()},
		&out, &errs, func(string) string { return "" })
	want := "reduced engine bundle"
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" {
		want = "no reviewed llama.cpp release is pinned"
	}
	if status != 1 || !bytes.Contains(errs.Bytes(), []byte(want)) {
		t.Fatalf("status %d, stderr %q; wanted 1 and %q", status, errs.String(), want)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("a refused bundle left %s: %v", dir, err)
	}
}

// engine unlock gives back only a folder this provider recorded; any other
// folder's ACL is left alone.
func TestEngineUnlockRefusesAFolderWithNoRecord(t *testing.T) {
	var out, errs bytes.Buffer
	status := run([]string{"engine", "unlock", "--dir", t.TempDir(), "--state", t.TempDir()},
		&out, &errs, func(string) string { return "" })
	if status != 1 || !bytes.Contains(errs.Bytes(), []byte("no engine record")) {
		t.Fatalf("status %d, stderr %q; wanted 1 and no engine record", status, errs.String())
	}
}

// engine unlock without --dir gives back the folder the engine record names,
// so an uninstaller needs no path; with no record it has nothing to unlock
// and creates no state.
func TestEngineUnlockWithoutDirUnlocksTheRecordedFolder(t *testing.T) {
	state := filepath.Join(t.TempDir(), "state")
	var out, errs bytes.Buffer
	if status := run([]string{"engine", "unlock", "--state", state}, &out, &errs, func(string) string { return "" }); status != 0 ||
		!bytes.Contains(out.Bytes(), []byte("nothing to unlock")) {
		t.Fatalf("status %d, stdout %q, stderr %q; wanted 0 and nothing to unlock", status, out.String(), errs.String())
	}
	if _, err := os.Stat(state); !os.IsNotExist(err) {
		t.Fatalf("engine unlock created state: %v", err)
	}
	//unchecked: a platform with no pin names the record by an empty release, which this test writes too
	pin, _ := host.PinnedEngine(runtime.GOOS, runtime.GOARCH)
	dir := filepath.Join(t.TempDir(), "engine")
	if err := os.MkdirAll(filepath.Join(dir, "lib"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "lib", "llama-server.exe"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(state, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(host.EngineRecord(state, pin), []byte("root\t"+dir+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		sid, err := enginefiles.AccountSID()
		if err != nil {
			t.Fatal(err)
		}
		if err := enginefiles.SetFolderACL(dir, sid, enginefiles.ReadOnly); err != nil {
			t.Fatal(err)
		}
		//unchecked: cleanup of a folder the test may already have removed
		t.Cleanup(func() { _ = enginefiles.UnlockWithin(dir) })
	}
	out.Reset()
	errs.Reset()
	if status := run([]string{"engine", "unlock", "--state", state}, &out, &errs, func(string) string { return "" }); status != 0 ||
		!bytes.Contains(out.Bytes(), []byte(dir)) {
		t.Fatalf("status %d, stdout %q, stderr %q; wanted 0 naming %s", status, out.String(), errs.String(), dir)
	}
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("the recorded folder is not removable after engine unlock: %v", err)
	}
}

// engine check is the proof a call makes, at configure time.
func TestEngineCheckRefusesAnUnpinnedEngine(t *testing.T) {
	program := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(program, []byte("not the release"), 0o700); err != nil {
		t.Fatal(err)
	}
	var out, errs bytes.Buffer
	status := run([]string{"engine", "check", "--engine", program, "--state", t.TempDir()},
		&out, &errs, func(string) string { return "" })
	if status != 1 || !bytes.Contains(errs.Bytes(), []byte("pinned")) {
		t.Fatalf("status %d, stderr %q; wanted 1 naming the pin", status, errs.String())
	}
}

// serve and engine check refuse an engine argument outside the reviewed list
// by name, before anything starts (MODELHOST-L5).
func TestEngineArgumentOutsideTheListIsRefusedByName(t *testing.T) {
	commands := map[string][]string{
		"serve": {"serve", "--endpoint", "modelhost-arg-test", "--model", "tiny=fake/tiny",
			"--engine-arg", "--rpc", "--engine-arg", "10.0.0.2:50052"},
		"engine check": {"engine", "check", "--engine", filepath.Join(t.TempDir(), "llama-server.exe"),
			"--state", t.TempDir(), "--engine-arg", "--rpc", "--engine-arg", "10.0.0.2:50052"},
	}
	for name, args := range commands {
		var out, errs bytes.Buffer
		status := run(args, &out, &errs, func(string) string { return "" })
		if status != 2 || !bytes.Contains(errs.Bytes(), []byte("--rpc is not on the reviewed llama-server list")) {
			t.Fatalf("%s: status %d, stderr %q; wanted 2 naming --rpc", name, status, errs.String())
		}
	}
}

func TestRuntimeDesignationComesFromProviderInstallation(t *testing.T) {
	dir := t.TempDir()
	runtimeName := "openabstractions"
	if runtime.GOOS == "windows" {
		runtimeName += ".exe"
	}
	installed := filepath.Join(dir, runtimeName)
	if err := os.WriteFile(installed, []byte("runtime"), 0o600); err != nil {
		t.Fatal(err)
	}
	programs, err := runtimeProgramsAt(filepath.Join(dir, "modelhostd"), nil)
	if err != nil || len(programs) != 1 || programs[0] != identity.CanonicalProgramPath(installed) {
		t.Fatalf("installed programs = %v, %v", programs, err)
	}
	if runtime.GOOS == "windows" {
		windowless := filepath.Join(dir, "openabstractionsw.exe")
		if err := os.WriteFile(windowless, []byte("runtime"), 0o600); err != nil {
			t.Fatal(err)
		}
		programs, err = runtimeProgramsAt(filepath.Join(dir, "modelhostd.exe"), nil)
		if err != nil || len(programs) != 2 || programs[1] != identity.CanonicalProgramPath(windowless) {
			t.Fatalf("windowless runtime = %v, %v", programs, err)
		}
	}
	explicit := filepath.Join(t.TempDir(), "custom-runtime")
	if err := os.WriteFile(explicit, []byte("runtime"), 0o600); err != nil {
		t.Fatal(err)
	}
	programs, err = runtimeProgramsAt(filepath.Join(dir, "modelhostd"), []string{explicit})
	if err != nil || len(programs) != 1 || programs[0] != identity.CanonicalProgramPath(explicit) {
		t.Fatalf("explicit program = %v, %v", programs, err)
	}
}
