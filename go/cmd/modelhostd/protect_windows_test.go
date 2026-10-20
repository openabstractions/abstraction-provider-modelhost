package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

func junctionAt(t *testing.T, link, target string) {
	t.Helper()
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Fatalf("mklink /J: %v %s", err, out)
	}
	t.Cleanup(func() {
		//unchecked: test cleanup; the temporary folder's removal takes what is left
		os.Remove(link)
	})
}

// Review of 6607b567d: models protect makes only recorded set folders model
// folders, here the one a places record uses, and leaves an unrecorded
// subfolder and a junction under the root as they are; --dry-run reports
// and writes nothing; a home-like root and a junctioned root are refused
// before anything changes.
func TestModelsProtectReprotectsSetFolders(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	state := filepath.Join(base, "state")
	video := filepath.Join(root, "video")
	unrecorded := filepath.Join(root, "unrecorded")
	outside := filepath.Join(base, "elsewhere")
	for _, dir := range []string{video, unrecorded, outside, state} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	set := host.FastWan22TI2V5B
	var lines strings.Builder
	for _, f := range set.Files {
		if err := os.WriteFile(filepath.Join(video, f.Name), []byte("stand-in"), 0o600); err != nil {
			t.Fatal(err)
		}
		lines.WriteString(f.Name + "\t" + filepath.Join(video, f.Name) + "\tS-1-5-18,S-1-5-32-544;account\n")
	}
	if err := os.WriteFile(host.VideoModelPlaces(state, set), []byte(lines.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	junctionAt(t, filepath.Join(root, "linked"), outside)
	protect := func(args ...string) (int, string, string) {
		var out, errs bytes.Buffer
		status := run(append([]string{"models", "protect", "--state", state, "--job-store", filepath.Join(base, "jobs")}, args...), &out, &errs, noEnv)
		return status, out.String(), errs.String()
	}

	status, out, errs := protect("--models-root", root, "--dry-run")
	if status != 0 || !strings.Contains(out, video+": would be protected") || privatedir.CheckModelFolder(video) == nil {
		t.Fatalf("dry run: status %d, stdout %q, stderr %q", status, out, errs)
	}
	status, out, errs = protect("--models-root", root)
	if status != 0 || !strings.Contains(out, video+": protected") || !strings.Contains(out, unrecorded+": not a recorded set") {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out, errs)
	}
	if err := privatedir.CheckModelFolder(video); err != nil {
		t.Fatal(err)
	}
	for _, dir := range []string{unrecorded, outside} {
		if privatedir.CheckModelFolder(dir) == nil {
			t.Fatalf("%s was protected", dir)
		}
	}

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	linkedRoot := filepath.Join(base, "linked-root")
	junctionAt(t, linkedRoot, root)
	for _, bad := range []string{home, linkedRoot} {
		if status, out, errs := protect("--models-root", bad); status != 1 || strings.Contains(out, ": protected") {
			t.Fatalf("root %s: status %d, stdout %q, stderr %q", bad, status, out, errs)
		}
	}
	// Second review of 6958c169b: a broad job store is refused, not taken
	// as OA's own folder.
	for _, broad := range []string{`C:\`, home} {
		if status, out, errs := protect("--models-root", root, "--job-store", broad); status != 1 || strings.Contains(out, ": protected") {
			t.Fatalf("job store %s: status %d, stdout %q, stderr %q", broad, status, out, errs)
		}
	}
	if status, _, errs := protect("--models-root", "OA-models"); status != 2 {
		t.Fatalf("a relative root: status %d, %q", status, errs)
	}
}
