package main

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
	"github.com/openabstractions/abstraction-storage/go/layout"
)

// Exercise the startup preparation with real pinned archive staging and tree
// verification, without starting an engine or changing the user's cache.
func TestServePreparationRestagesEarlier24FilePin(t *testing.T) {
	for _, variant := range []string{"upgrade", "missing", "missing-unrecorded", "missing-bad-bundle", "missing-foreign-record", "missing-linked-declaration", "foreign", "unrecorded", "held", "unreadable", "bad-bundle"} {
		t.Run(variant, func(t *testing.T) {
			if strings.HasPrefix(variant, "missing") && (runtime.GOOS != "windows" || runtime.GOARCH != "amd64") {
				t.Skip("absent-only engine commit is supported on Windows amd64")
			}
			if variant == "held" && runtime.GOOS != "windows" {
				t.Skip("Windows held files prevent replacement")
			}
			cache, state, tools := t.TempDir(), t.TempDir(), t.TempDir()
			dir := filepath.Join(cache, "OpenAbstractions", "engines", "llama.cpp-b10825")
			if variant == "foreign" {
				dir = filepath.Join(t.TempDir(), "foreign")
			}
			pin := host.EnginePin{Release: "b10825", Program: "llama-server.exe"}
			bodies := map[string]string{"llama-server.exe": "inert server"}
			for i := 0; i < 23; i++ {
				bodies[fmt.Sprintf("engine-%02d.dll", i)] = fmt.Sprintf("inert library %d", i)
			}
			pin.Files = fixtureEngineFiles(bodies)
			oldBundle := fixtureEngineBundle(t, &pin, bodies, filepath.Join(t.TempDir(), "old.zip"))
			files := programfiles.NewVerifier(programfiles.Options{})
			record := host.EngineRecord(state, pin)
			program, err := host.InstallBundledEngine(t.Context(), files, pin, oldBundle, dir, record)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = programfiles.Unlock(dir) })
			var held *programfiles.Hold
			if variant == "held" {
				held, err = files.Launch(t.Context(), programfiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: programfiles.Manifest{Files: pin.Files}, Record: record})
				if err != nil {
					t.Fatal(err)
				}
				defer held.Release()
			}
			runtimeBodies := map[string]string{"msvcp140.dll": "inert msvcp", "vcruntime140.dll": "inert vcruntime", "vcruntime140_1.dll": "inert vcruntime1"}
			pin.Runtime = host.EngineRuntime{Name: "VC runtime", Version: "test", Folder: "Microsoft.VC145.CRT", Files: fixtureEngineFiles(runtimeBodies)}
			for name, body := range runtimeBodies {
				bodies[name] = body
			}
			bundle := fixtureEngineBundle(t, &pin, bodies, filepath.Join(tools, "engines", "bundle.zip"))
			if variant == "unreadable" {
				if _, err := host.InstallBundledEngine(t.Context(), files, pin, bundle, dir, record); err != nil {
					t.Fatal(err)
				}
				files = programfiles.NewVerifier(programfiles.Options{ACL: unreadableEngineACL{}})
			}
			if variant == "unrecorded" || variant == "missing-unrecorded" {
				if err := os.Remove(record); err != nil {
					t.Fatal(err)
				}
			}
			if variant == "bad-bundle" || variant == "missing-bad-bundle" {
				if err := os.WriteFile(bundle, []byte("unverified"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if variant == "missing-foreign-record" {
				foreign := filepath.Join(t.TempDir(), "foreign")
				if _, err := host.InstallBundledEngine(t.Context(), files, pin, bundle, foreign, record); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = programfiles.Unlock(foreign) })
			}
			if variant == "missing-linked-declaration" {
				alias := filepath.Join(t.TempDir(), "linked-parent")
				if err := os.Symlink(filepath.Dir(dir), alias); err != nil {
					t.Skipf("directory symlink unavailable: %v", err)
				}
				program = filepath.Join(alias, filepath.Base(dir), pin.Program)
			}
			if strings.HasPrefix(variant, "missing") {
				// Keep the provider's record and valid installed bundle while
				// removing only this fixture's previously managed cache tree.
				if err := programfiles.Unlock(dir); err != nil {
					t.Fatal(err)
				}
				if err := os.RemoveAll(dir); err != nil {
					t.Fatal(err)
				}
			}
			managed := layout.EngineCacheDir(cache, "llama.cpp", pin.Release)
			check, err := prepareServeEngineObserved(context.Background(), pin, program, state, filepath.Join(tools, "modelhostd.exe"), managed, files, io.Discard)
			wantState := service.EngineRefused
			degraded := variant == "foreign" || variant == "missing-unrecorded" || variant == "missing-foreign-record"
			if variant == "upgrade" || variant == "missing" {
				wantState = service.EngineRepaired
			}
			if degraded {
				wantState = service.EngineUnverified
			}
			if check.Kind != service.EngineCheckKind || check.Copy != pin.Release || check.State != wantState {
				t.Fatalf("observation %#v", check)
			}
			if _, err := time.Parse(time.RFC3339Nano, check.Checked); err != nil {
				t.Fatal(err)
			}
			if variant == "unreadable" && !errors.Is(err, programfiles.ErrUnreadable) {
				t.Fatalf("wanted unreadable refusal, got %v", err)
			}
			wantFiles := 24
			if variant == "unreadable" {
				wantFiles = 27
			}
			if variant == "upgrade" || variant == "missing" {
				wantFiles = 27
				if err != nil {
					t.Fatal(err)
				}
				// A valid current tree needs no archive on the next startup.
				if err := os.Remove(bundle); err != nil {
					t.Fatal(err)
				}
				current, err := prepareServeEngineObserved(t.Context(), pin, program, state, filepath.Join(tools, "modelhostd.exe"), managed, programfiles.NewVerifier(programfiles.Options{}), io.Discard)
				if err != nil {
					t.Fatal(err)
				}
				if current.State != service.EngineCurrent || current.Reason != "" {
					t.Fatalf("current observation %#v", current)
				}
			} else if !degraded && err == nil {
				t.Fatal("unsafe preparation succeeded")
			}
			if degraded && (err != nil || check.Reason != "engine:custom_path") {
				t.Fatalf("degraded startup changed: %#v %v", check, err)
			}
			entries, err := os.ReadDir(dir)
			if strings.HasPrefix(variant, "missing-") {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("refused missing target changed: %v", err)
				}
				return
			}
			if err != nil || len(entries) != wantFiles {
				t.Fatalf("files %d, want %d: %v", len(entries), wantFiles, err)
			}
			if variant == "upgrade" || variant == "missing" {
				manifest := fixtureEngineFiles(bodies)
				if err := files.Check(t.Context(), programfiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: programfiles.Manifest{Files: manifest}, Record: record}); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestServePreparationCustomObservation(t *testing.T) {
	for _, program := range []string{"", "custom", filepath.Join(t.TempDir(), "custom.exe")} {
		check, err := prepareServeEngineObserved(t.Context(), host.EnginePin{Release: "release", Program: "llama-server.exe"}, program, "", "", "", nil, io.Discard)
		if err != nil || check.State != service.EngineUnverified || check.Reason != "engine:custom_path" {
			t.Fatalf("%#v %v", check, err)
		}
	}
}

// Inject the verifier's typed refusal at startup; the real held-tree case
// above separately exercises Windows' replacement fence.
type unreadableEngineACL struct{}

func (unreadableEngineACL) Check(string, *programfiles.Handle) error {
	return programfiles.ErrUnreadable
}
func (unreadableEngineACL) Restrict([]string) error {
	return errors.New("unreadable tree must not be installed")
}

func fixtureEngineFiles(bodies map[string]string) []programfiles.File {
	var files []programfiles.File
	for name, body := range bodies {
		sum := sha256.Sum256([]byte(body))
		files = append(files, programfiles.File{Path: name, Size: int64(len(body)), SHA256: hex.EncodeToString(sum[:])})
	}
	return files
}

func fixtureEngineBundle(t *testing.T, pin *host.EnginePin, bodies map[string]string, path string) string {
	t.Helper()
	var data bytes.Buffer
	w := zip.NewWriter(&data)
	for name, body := range bodies {
		f, err := w.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data.Bytes())
	pin.Bundle = host.EngineBundle{Name: filepath.Base(path), Size: int64(data.Len()), SHA256: hex.EncodeToString(sum[:])}
	return path
}
