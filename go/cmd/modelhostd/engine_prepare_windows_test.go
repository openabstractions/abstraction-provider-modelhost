//go:build windows

package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
	"github.com/openabstractions/abstraction-storage/go/layout"
)

// A removal of OpenAbstractions runs `modelhostd engine unlock`, and a later
// install keeps the engine folder with the account's full control. The next
// start sets the read-only ACL again when every file proves, and only a file
// that differs from the pin, with no bundled copy to replace it, refuses.
func TestServePreparationRelocksUnlockedEngine(t *testing.T) {
	for _, variant := range []string{"good-files", "bad-file", "bad-file-bundled"} {
		t.Run(variant, func(t *testing.T) {
			cache, state, tools := t.TempDir(), t.TempDir(), t.TempDir()
			dir := filepath.Join(cache, "OpenAbstractions", "engines", "llama.cpp-b10825")
			pin := host.EnginePin{Release: "b10825", Program: "llama-server.exe"}
			bodies := map[string]string{"llama-server.exe": "inert server"}
			for i := 0; i < 4; i++ {
				bodies[fmt.Sprintf("engine-%02d.dll", i)] = fmt.Sprintf("inert library %d", i)
			}
			pin.Files = fixtureEngineFiles(bodies)
			bundle := fixtureEngineBundle(t, &pin, bodies, filepath.Join(tools, "engines", "bundle.zip"))
			record := host.EngineRecord(state, pin)
			program, err := host.InstallBundledEngine(t.Context(), programfiles.NewVerifier(programfiles.Options{}), pin, bundle, dir, record)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = programfiles.Unlock(dir) })
			if err := programfiles.Unlock(dir); err != nil {
				t.Fatal(err)
			}
			changed := filepath.Join(dir, "engine-01.dll")
			if variant != "good-files" {
				if err := os.WriteFile(changed, []byte("inert library X"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			if variant != "bad-file-bundled" {
				if err := os.Remove(bundle); err != nil {
					t.Fatal(err)
				}
			}
			var diagnostics bytes.Buffer
			managed := layout.EngineCacheDir(cache, "llama.cpp", pin.Release)
			files := programfiles.NewVerifier(programfiles.Options{})
			check, err := prepareServeEngineObserved(t.Context(), pin, program, state, filepath.Join(tools, "modelhostd.exe"), managed, files, &diagnostics)
			locked, lockErr := programfiles.Installed(dir)
			if lockErr != nil {
				t.Fatal(lockErr)
			}
			if variant == "bad-file" {
				if err == nil || check.State != service.EngineRefused || !strings.Contains(err.Error(), "Repair OpenAbstractions") {
					t.Fatalf("a changed file without a bundle was not refused: %#v %v", check, err)
				}
				if locked || diagnostics.Len() != 0 {
					t.Fatalf("a refused tree was locked (%v) or reported repaired: %q", locked, diagnostics.String())
				}
				if got, err := os.ReadFile(changed); err != nil || string(got) != "inert library X" {
					t.Fatalf("a refused tree changed: %q %v", got, err)
				}
				return
			}
			if err != nil || check.State != service.EngineRepaired || !locked {
				t.Fatalf("unlocked engine not repaired: %#v locked=%v %v", check, locked, err)
			}
			lines := strings.Count(diagnostics.String(), "\n")
			if variant == "good-files" && (lines != 1 || !strings.Contains(diagnostics.String(), "set the read-only folder ACL")) {
				t.Fatalf("diagnostics %q", diagnostics.String())
			}
			if variant == "bad-file-bundled" && lines != 0 {
				t.Fatalf("a restaged tree reported an ACL-only repair: %q", diagnostics.String())
			}
			if got, err := os.ReadFile(changed); err != nil || string(got) != bodies["engine-01.dll"] {
				t.Fatalf("engine file %q %v", got, err)
			}
			// The repaired tree serves: a fresh process launches and holds it.
			hold, err := programfiles.NewVerifier(programfiles.Options{}).Launch(t.Context(),
				programfiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: programfiles.Manifest{Files: pin.Files}, Record: record})
			if err != nil {
				t.Fatal(err)
			}
			if err := hold.Release(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
