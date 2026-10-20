package host

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-identity/go/privatedir"
	"golang.org/x/sys/windows"
)

// everyoneMay gives p an explicit entry letting Everyone modify it, as a file
// moved in from %TEMP% carries the sandbox group's, and reports whether p's
// DACL names Everyone.
func everyoneMay(t *testing.T, p string, set bool) bool {
	t.Helper()
	if set {
		sd, err := windows.SecurityDescriptorFromString("D:(A;;0x1301bf;;;WD)")
		if err != nil {
			t.Fatal(err)
		}
		dacl, _, err := sd.DACL()
		if err != nil {
			t.Fatal(err)
		}
		if err := windows.SetNamedSecurityInfo(p, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION, nil, nil, dacl, nil); err != nil {
			t.Fatal(err)
		}
	}
	sd, err := windows.GetNamedSecurityInfo(p, windows.SE_FILE_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatal(err)
	}
	return strings.Contains(sd.String(), ";WD)")
}

// Coordinator follow-up 3 (identity ID-D3): a set folder directly under the
// models root is OA's own and is made a model folder, replacing an entry a
// move left on its file; a folder elsewhere, and the root itself, keep their
// access untouched.
func TestAdoptModelFolderTouchesOnlyOAsSetFolders(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	inside := filepath.Join(root, "video")
	outside := filepath.Join(base, "AICreator")
	for _, dir := range []string{inside, outside} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "weights.gguf"), []byte("w"), 0o600); err != nil {
			t.Fatal(err)
		}
		everyoneMay(t, filepath.Join(dir, "weights.gguf"), true)
	}
	if adopted, err := AdoptModelFolder(root, outside, nil); adopted || err != nil {
		t.Fatalf("a folder outside the root: %v %v", adopted, err)
	}
	if !everyoneMay(t, filepath.Join(outside, "weights.gguf"), false) {
		t.Fatal("a folder outside the root had its access changed")
	}
	if adopted, err := AdoptModelFolder(root, root, nil); adopted || err != nil {
		t.Fatalf("the root itself: %v %v", adopted, err)
	}
	if adopted, err := AdoptModelFolder(root, inside, nil); !adopted || err != nil {
		t.Fatalf("a set folder: %v %v", adopted, err)
	}
	if everyoneMay(t, filepath.Join(inside, "weights.gguf"), false) {
		t.Fatal("the moved file kept its entry")
	}
	if err := privatedir.CheckModelFolder(inside); err != nil {
		t.Fatal(err)
	}
	if _, err := AdoptModelFolder("OA-models", inside, nil); err == nil {
		t.Fatal("a relative root was accepted")
	}
}

// Review of 6607b567d, scope: only recorded sets are OA's set folders, a
// folder a places record uses or one named as a set; any other folder
// under the root is not. A home-like root refuses adoption.
func TestRecordedSetFoldersAreTheOnlySetFolders(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "OA-models")
	state := filepath.Join(base, "state")
	for _, dir := range []string{filepath.Join(root, "video"), filepath.Join(root, "unrecorded"), filepath.Join(root, FastWan22TI2V5B.Name), state} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	var lines strings.Builder
	for _, f := range HummingbirdXTI2V.Files {
		lines.WriteString(f.Name + "\t" + filepath.Join(root, "video", f.Name) + "\t" + formatOwners(FolderSearch("").Owners()) + "\n")
	}
	if err := os.WriteFile(VideoModelPlaces(state, HummingbirdXTI2V), []byte(lines.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := RecordedSetFolders(root, state)
	want := []string{filepath.Join(root, FastWan22TI2V5B.Name), filepath.Join(root, "video")}
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("recorded set folders %q, %v; want %q", got, err, want)
	}
	if SetFolderOf(root, filepath.Join(base, "elsewhere", "x")) != "" || SetFolderOf(root, root) != "" {
		t.Fatal("a path outside the root named a set folder")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if adopted, err := AdoptModelFolder(filepath.Dir(home), home, nil); adopted || err == nil {
		t.Fatalf("a set folder under a home-like root: %v %v", adopted, err)
	}
}
