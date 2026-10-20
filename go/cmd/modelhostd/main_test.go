package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/openabstractions/abstraction-facade/go-core/bootstrap"
	"github.com/openabstractions/abstraction-identity/listen"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/go/privatedir/privatetest"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
)

func TestServeStartupKeepsUnprovenMissingEngineVisible(t *testing.T) {
	for _, variant := range []string{"missing-record", "foreign-record"} {
		t.Run(variant, func(t *testing.T) {
			parent, state := t.TempDir(), t.TempDir()
			dir := filepath.Join(parent, "managed")
			pin := host.EnginePin{Release: "inert", Program: "llama-server.exe"}
			var recordBytes []byte
			record := host.EngineRecord(state, pin)
			if variant == "foreign-record" {
				recordBytes = []byte("root\t" + filepath.Join(t.TempDir(), "foreign") + "\n")
				if err := os.WriteFile(record, recordBytes, 0600); err != nil {
					t.Fatal(err)
				}
			}
			var diagnostics bytes.Buffer
			check, status := prepareServeStartup(t.Context(), pin, filepath.Join(dir, pin.Program), state, filepath.Join(t.TempDir(), "modelhostd.exe"), dir, programfiles.NewVerifier(programfiles.Options{}), &diagnostics)
			if status != 0 || check.State != service.EngineUnverified || check.Reason != "engine:custom_path" || diagnostics.Len() != 0 {
				t.Fatalf("startup status %d, check %#v, diagnostics %q", status, check, diagnostics.String())
			}
			if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unproven missing target changed: %v", err)
			}
			got, err := os.ReadFile(record)
			if recordBytes == nil {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("missing record changed: %v", err)
				}
			} else if err != nil || !bytes.Equal(got, recordBytes) {
				t.Fatalf("foreign record changed: %v", err)
			}
		})
	}
}

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
	// The reviewed pin carries an app-local Visual C++ runtime the publisher's
	// ZIP lacks, so the publisher-archive route refuses before reading it.
	want := "--archive cannot stage llama.cpp b10825: the publisher's ZIP lacks the Microsoft Visual C++"
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
		sid, err := programfiles.AccountSID()
		if err != nil {
			t.Fatal(err)
		}
		if err := programfiles.SetFolderACL(dir, sid, programfiles.ReadOnly); err != nil {
			t.Fatal(err)
		}
		//unchecked: cleanup of a folder the test may already have removed
		t.Cleanup(func() { _ = programfiles.UnlockWithin(dir) })
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

var fixtureEndpointSequence atomic.Int64

func explicitFixtureEndpoint(t *testing.T, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		return fmt.Sprintf(`\\.\pipe\oa-test-%s-%d-%d`, name, os.Getpid(), fixtureEndpointSequence.Add(1))
	}
	return filepath.Join(privatetest.SocketDir(t), name+".sock")
}

// serve and engine check refuse an engine argument outside the reviewed list
// by name, before anything starts (MODELHOST-L5).
func TestEngineArgumentOutsideTheListIsRefusedByName(t *testing.T) {
	commands := map[string][]string{
		"serve": {"serve", "--endpoint", explicitFixtureEndpoint(t, "modelhost-arg-test"), "--model", "tiny=fake/tiny",
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

// Naming is exercised through a real owned socket/pipe; no provider engine runs.
func TestLeafEndpointNamingAndReceiving(t *testing.T) {
	parent, err := os.MkdirTemp("", "st5-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(parent); err != nil {
			t.Error(err)
		}
	})
	if runtime.GOOS != "windows" {
		t.Setenv("TMPDIR", parent)
	}
	t.Setenv("XDG_RUNTIME_DIR", "")
	name := fmt.Sprintf("st5-leaf-%d", os.Getpid())
	if address, err := serveEndpoint(name); address != "" || !errors.Is(err, bootstrap.ErrNoTrustedInstallation) {
		t.Fatalf("default name escaped test refusal: %q %v", address, err)
	}
	address := filepath.Join(parent, name+".sock")
	if runtime.GOOS == "windows" {
		address = `\\.\pipe\oa-test-` + name
	}

	if actual, err := serveEndpoint(address); err != nil || actual != address {
		t.Fatalf("explicit address changed: %q %v", actual, err)
	}
	for _, explicit := range []string{`/owned/path with spaces/name.sock`, `C:\owned\path with spaces\name.sock`, `\\.\pipe\owned.Name_test`} {
		if actual, err := serveEndpoint(explicit); err != nil || actual != explicit {
			t.Fatalf("explicit %q changed to %q: %v", explicit, actual, err)
		}
	}
	listener, err := listen.Listen(address)
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stopListener := context.AfterFunc(ctx, func() { listener.Close() })
	defer stopListener()
	answered := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			answered <- err
			return
		}
		defer conn.Close()
		stop := context.AfterFunc(ctx, func() { conn.Close() })
		defer stop()
		var request [1]byte
		if _, err = io.ReadFull(conn, request[:]); err == nil {
			if request[0] != 'q' {
				err = fmt.Errorf("unexpected request %q", request[0])
			} else {
				_, err = conn.Write([]byte{'r'})
			}
		}
		answered <- err
	}()
	conn, err := listen.Dial(address)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	stopConn := context.AfterFunc(ctx, func() { conn.Close() })
	defer stopConn()
	if _, err = conn.Write([]byte{'q'}); err != nil {
		t.Fatal(err)
	}
	var reply [1]byte
	if _, err = io.ReadFull(conn, reply[:]); err != nil || reply[0] != 'r' {
		t.Fatalf("reply %q: %v", reply, err)
	}
	if err := <-answered; err != nil {
		t.Fatal(err)
	}
}
func TestLeafEndpointRefusesBeforeProviderWork(t *testing.T) {
	for _, name := range []string{"model.v1", "model_v1", "Model", "model name"} {
		var output, diagnostics bytes.Buffer
		code := run([]string{"serve", "--endpoint", name}, &output, &diagnostics, func(string) string { return "" })
		if code != 2 || !strings.Contains(diagnostics.String(), "[a-z0-9-]+") || !strings.Contains(diagnostics.String(), "explicit absolute endpoint path") || output.Len() != 0 {
			t.Fatalf("%q: code %d, output %q, diagnostics %q", name, code, output.String(), diagnostics.String())
		}
	}
	if runtime.GOOS != "windows" {
		// A short name resolves in the runtime directory below; test binaries
		// need the opt-in to resolve installed endpoints at all.
		t.Setenv("OA_TEST_ALLOW_INSTALLED_RUNTIME", "1")
		dir := t.TempDir()
		if err := os.Chmod(dir, 0755); err != nil {
			t.Fatal(err)
		}
		t.Setenv("XDG_RUNTIME_DIR", dir)
		var output, diagnostics bytes.Buffer
		code := run([]string{"serve", "--endpoint", "leaf-v1"}, &output, &diagnostics, func(string) string { return "" })
		if code != 2 || !strings.Contains(diagnostics.String(), "private") || output.Len() != 0 {
			t.Fatalf("namespace: code %d, output %q, diagnostics %q", code, output.String(), diagnostics.String())
		}
	}
}
