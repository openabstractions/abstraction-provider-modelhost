package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/sys/windows"

	"github.com/openabstractions/abstraction-identity/go/privatedir"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// standInFiles writes one small file per pinned file of files into dir and
// returns the same files pinned to what was written.
func standInFiles(t *testing.T, dir string, files []host.VideoModelFile) []host.VideoModelFile {
	t.Helper()
	out := make([]host.VideoModelFile, len(files))
	for i, f := range files {
		data := []byte("stand-in " + f.Name)
		if err := os.WriteFile(filepath.Join(dir, f.Name), data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		f.Size, f.SHA256, f.GitBlobSHA1 = int64(len(data)), hex.EncodeToString(sum[:]), ""
		out[i] = f
	}
	return out
}

// describeAccess is the owner and access list of p and of every entry in it,
// as SDDL.
func describeAccess(p string) (string, error) {
	paths := []string{p}
	entries, err := os.ReadDir(p)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		paths = append(paths, filepath.Join(p, e.Name()))
	}
	var b strings.Builder
	for _, q := range paths {
		sd, err := windows.GetNamedSecurityInfo(q, windows.SE_FILE_OBJECT,
			windows.OWNER_SECURITY_INFORMATION|windows.DACL_SECURITY_INFORMATION)
		if err != nil {
			return "", err
		}
		b.WriteString(q + " " + sd.String() + "\n")
	}
	return b.String(), nil
}

// installAdopts runs models install for kind and set with a --from folder
// directly under --models-root, and checks that the folder is then a model
// folder and that its sibling under the root kept its access.
func installAdopts(t *testing.T, kind, set string, files func(dir string) []host.VideoModelFile) string {
	t.Helper()
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	from := filepath.Join(root, "set")
	sibling := filepath.Join(root, "sibling")
	state := filepath.Join(base, "state")
	for _, dir := range []string{from, sibling, state} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(sibling, "kept.bin"), []byte("kept"), 0o600); err != nil {
		t.Fatal(err)
	}
	files(from)
	if privatedir.CheckModelFolder(from) == nil || privatedir.CheckModelFolder(sibling) == nil {
		t.Fatal("a fresh folder already passes the model-folder check")
	}
	before, err := describeAccess(sibling)
	if err != nil {
		t.Fatal(err)
	}
	var out, errs bytes.Buffer
	status := run([]string{"models", "install", "--kind", kind, "--set", set, "--from", from, "--models-root", root,
		"--home", filepath.Join(base, "home"), "--state", state, "--job-store", filepath.Join(base, "jobs")}, &out, &errs, noEnv)
	if status != 0 {
		t.Fatalf("status %d, stdout %q, stderr %q", status, out.String(), errs.String())
	}
	if !strings.Contains(out.String(), from+": OA's model folder, protected") {
		t.Fatalf("stdout %q", out.String())
	}
	if err := privatedir.CheckModelFolder(from); err != nil {
		t.Fatalf("the installed set's folder is not a model folder: %v", err)
	}
	after, err := describeAccess(sibling)
	if err != nil {
		t.Fatal(err)
	}
	if after != before || privatedir.CheckModelFolder(sibling) == nil {
		t.Fatalf("the sibling folder's access changed:\nbefore %s\nafter  %s", before, after)
	}
	return out.String()
}

// Delta review of the PyTorch engine kind (coordinator, 2026-10-02): models
// install adopts the set folder it placed files in under the models root;
// a sibling folder under the root keeps its access
// (MODELHOST-V2, identity ID-D3).
func TestModelsInstallAdoptsTheSetFolder(t *testing.T) {
	t.Run("sdcpp", func(t *testing.T) {
		saved := host.VideoModelSets
		t.Cleanup(func() { host.VideoModelSets = saved })
		installAdopts(t, "sdcpp", "adopt-test-sdcpp", func(dir string) []host.VideoModelFile {
			set := host.FastWan22TI2V5B
			set.Name = "adopt-test-sdcpp"
			set.Files = standInFiles(t, dir, set.Files)
			host.VideoModelSets = append(append([]host.VideoModelSet(nil), saved...), set)
			return set.Files
		})
	})
}
