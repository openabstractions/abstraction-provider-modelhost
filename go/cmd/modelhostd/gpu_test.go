package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

// RES-G3: gpu list names a damaged quarantine file, and gpu clear --all moves
// it aside, after which list names where it went and lists nothing.
func TestGPUListAndClearADamagedQuarantine(t *testing.T) {
	state := t.TempDir()
	path := filepath.Join(state, gpuQuarantineFile)
	if err := os.WriteFile(path, []byte(`{"version": 9, "entries": []}`), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(string) string { return "" }
	var out, errs bytes.Buffer
	if code := run([]string{"gpu", "list", "--state", state}, &out, &errs, env); code != 1 || !strings.Contains(errs.String(), "damaged") {
		t.Fatalf("list of a damaged file exited %d: %q", code, errs.String())
	}
	out.Reset()
	if code := run([]string{"gpu", "clear", "--all", "--state", state}, &out, &errs, env); code != 0 || !strings.Contains(out.String(), "moved aside") {
		t.Fatalf("clear --all on a damaged file exited %d: %q %q", code, out.String(), errs.String())
	}
	out.Reset()
	if code := run([]string{"gpu", "list", "--state", state}, &out, &errs, env); code != 0 || !strings.Contains(out.String(), ".bad-") {
		t.Fatalf("list after clear exited %d and printed %q", code, out.String())
	}
}

// RES-G3: a paused request is listed with its key and cleared by a person.
func TestGPUListAndClear(t *testing.T) {
	state := t.TempDir()
	guard, err := gpufault.New(gpufault.Options{State: filepath.Join(state, gpuQuarantineFile)})
	if err != nil {
		t.Fatal(err)
	}
	fp := gpufault.Fingerprint{Model: "fastwan2.2-ti2v-5b", Engine: "sd-server master-843-462d675", Shape: "1280x704x121 fps24"}
	guard.Failed(fp)
	var out, errs bytes.Buffer
	if code := run([]string{"gpu", "list", "--state", state}, &out, &errs, func(string) string { return "" }); code != 0 {
		t.Fatalf("list exited %d: %s", code, errs.String())
	}
	if !strings.Contains(out.String(), fp.Key()) || !strings.Contains(out.String(), "1280x704x121") {
		t.Fatalf("list printed %q", out.String())
	}
	out.Reset()
	if code := run([]string{"gpu", "clear", "--state", state, "nokey"}, &out, &errs, func(string) string { return "" }); code != 1 {
		t.Fatalf("clearing an unknown key exited %d", code)
	}
	if code := run([]string{"gpu", "clear", "--state", state, fp.Key()}, &out, &errs, func(string) string { return "" }); code != 0 {
		t.Fatalf("clear exited %d: %s", code, errs.String())
	}
	if err := guard.Quarantine(fp); err != nil {
		t.Fatalf("the cleared request is still paused: %v", err)
	}
	out.Reset()
	if code := run([]string{"gpu", "list", "--state", state}, &out, &errs, func(string) string { return "" }); code != 0 || out.Len() != 0 {
		t.Fatalf("list after clear exited %d and printed %q", code, out.String())
	}
}
