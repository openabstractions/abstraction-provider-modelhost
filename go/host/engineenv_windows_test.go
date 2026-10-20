//go:build windows

package host

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// A profile folder whose registry value names a variable the provider does
// not compute (a redirected folder under %HOMESHARE%) is left out of the
// engine's environment: the engine starts without it, and one diagnostic
// names the variable without the registry value (MODELHOST-L6).
func TestAProfileFolderNamingAnotherVariableIsLeftOut(t *testing.T) {
	real := registryString
	t.Cleanup(func() { registryString = real })
	registryString = func(hive, path, name string) (string, error) {
		if name == "Local AppData" {
			return `%HOMESHARE%\redirected\AppData\Local`, nil
		}
		return real(hive, path, name)
	}
	e := copiedEngine(t)
	report := filepath.Join(t.TempDir(), "env")
	e.fixture = []string{"--fake-env-file", report}
	var mu sync.Mutex
	var reports []string
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine = e
		o.OnError = func(err error) { mu.Lock(); reports = append(reports, err.Error()); mu.Unlock() }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the engine did not start without LOCALAPPDATA: %v", err)
	}
	raw, err := os.ReadFile(report)
	if err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		name, _, _ := strings.Cut(line, "\t")
		names[strings.ToUpper(name)] = true
	}
	if names["LOCALAPPDATA"] || !names["APPDATA"] || !names["USERPROFILE"] {
		t.Fatalf("LOCALAPPDATA delivered=%v, APPDATA=%v, USERPROFILE=%v; wanted only LOCALAPPDATA left out",
			names["LOCALAPPDATA"], names["APPDATA"], names["USERPROFILE"])
	}
	mu.Lock()
	defer mu.Unlock()
	named := 0
	for _, r := range reports {
		if strings.Contains(r, "HOMESHARE") || strings.Contains(r, "redirected") {
			t.Fatalf("a diagnostic carries the registry value: %q", r)
		}
		if strings.Contains(r, "LOCALAPPDATA") {
			named++
		}
	}
	if named != 1 {
		t.Fatalf("%d diagnostics name LOCALAPPDATA, wanted 1: %q", named, reports)
	}
}
