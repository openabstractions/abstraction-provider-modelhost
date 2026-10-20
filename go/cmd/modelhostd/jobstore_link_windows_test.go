package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// stageFromJobStore registers a stand-in sd-server set whose files the job
// service staged under root\set: each file's first name lies in the
// runtime's job store under base, as LinkResult leaves it (JOB-A17). With
// extra, each file also gets a name under base\setup-state, outside every OA
// root. args are the models install arguments without --job-store.
func stageFromJobStore(t *testing.T, extra bool) (from string, getenv func(string) string, args []string) {
	t.Helper()
	saved := host.VideoModelSets
	t.Cleanup(func() { host.VideoModelSets = saved })
	base, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	getenv = func(name string) string {
		if name == "LOCALAPPDATA" {
			return base
		}
		return ""
	}
	results := filepath.Join(modelplace.DefaultJobStore(getenv), "results")
	root := filepath.Join(base, "OA", "m")
	from = filepath.Join(root, "set")
	state := filepath.Join(base, "OA", "p", "sd")
	setup := filepath.Join(base, "setup-state")
	for _, dir := range []string{results, from, state, setup} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	set := host.FastWan22TI2V5B
	set.Name = "jobstore-link-test"
	set.Files = append([]host.VideoModelFile(nil), set.Files...)
	for i, f := range set.Files {
		data := []byte("stand-in " + f.Name)
		result := filepath.Join(results, fmt.Sprintf("result-%d", i))
		if err := os.WriteFile(result, data, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Link(result, filepath.Join(from, f.Name)); err != nil {
			t.Skipf("this volume makes no hard links: %v", err)
		}
		if extra {
			if err := os.Link(result, filepath.Join(setup, fmt.Sprintf("queued-%d", i))); err != nil {
				t.Fatal(err)
			}
		}
		sum := sha256.Sum256(data)
		set.Files[i].Size, set.Files[i].SHA256, set.Files[i].GitBlobSHA1 = int64(len(data)), hex.EncodeToString(sum[:]), ""
	}
	host.VideoModelSets = append(append([]host.VideoModelSet(nil), saved...), set)
	args = []string{"models", "install", "--kind", "sdcpp", "--set", set.Name, "--from", from, "--models-root", root,
		"--home", filepath.Join(base, "home"), "--state", state}
	return from, getenv, args
}

// models install admits the runtime's job store as one of OA's own roots
// without the flag (modelplace.OwnedRoots, KIT-M3).
func TestModelsInstallAdmitsTheJobStoreWithoutTheFlag(t *testing.T) {
	from, getenv, args := stageFromJobStore(t, false)
	var out, errs bytes.Buffer
	if status := run(args, &out, &errs, getenv); status != 0 {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if err := privatedir.CheckModelFolder(from); err != nil {
		t.Fatalf("the staged set's folder is not a model folder: %v", err)
	}
}

// A name outside OA's own roots, such as a link under Setup's state, still
// refuses the folder unchanged (identity ID-D3).
func TestModelsInstallRefusesALinkOutsideOwnedRoots(t *testing.T) {
	from, getenv, args := stageFromJobStore(t, true)
	var out, errs bytes.Buffer
	if status := run(args, &out, &errs, getenv); status != 1 || !strings.Contains(errs.String(), "not a model folder") {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if privatedir.CheckModelFolder(from) == nil {
		t.Fatal("a folder with a name outside OA's roots became a model folder")
	}
}

// An empty --job-store once meant "no job store" without complaint; it is
// refused before anything is placed.
func TestModelsRefuseAnEmptyJobStore(t *testing.T) {
	from, getenv, args := stageFromJobStore(t, false)
	var out, errs bytes.Buffer
	if status := run(append(args, "--job-store", ""), &out, &errs, getenv); status != 2 || !strings.Contains(errs.String(), "job-store") {
		t.Fatalf("status %d, stderr %q", status, errs.String())
	}
	if privatedir.CheckModelFolder(from) == nil {
		t.Fatal("a refused command protected the folder")
	}
}
