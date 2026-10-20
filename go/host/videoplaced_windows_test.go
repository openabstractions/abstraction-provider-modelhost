package host

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// junction makes link a directory junction to target inside the test's own
// temporary folder and removes the link, never the target, at cleanup.
func junction(t *testing.T, link, target string) {
	t.Helper()
	if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
		t.Fatalf("mklink /J: %v %s", err, out)
	}
	t.Cleanup(func() {
		//unchecked: test cleanup; the temporary folder's removal takes what is left
		os.Remove(link)
	})
}

// The review of da8e3b548 (INF-F7, MODELHOST-V2): a model file found
// through a junction is not placed, sd-server is given the held files' own
// final paths, and a folder of a placed file swapped for a junction to
// other bytes of the same size fails the next video with engine:none
// before any lease.
func TestVideoModelPathsCannotBeRedirectedByAJunction(t *testing.T) {
	f := newVideoFixture(t)
	downloads := filepath.Join(f.dir, "downloads")
	linked := filepath.Join(f.dir, "linked")
	junction(t, linked, downloads)
	_, err := PlaceVideoModels(context.Background(), f.engine.Set, []ModelSearch{FolderSearch(linked)}, filepath.Join(f.dir, "state", "linked.places"))
	if err == nil || !strings.Contains(err.Error(), "link") {
		t.Fatalf("placing through a junction: %v", err)
	}

	h := f.host(t, newFakeLeases(), nil)
	awaitVideo(t, h, startVideo(t, h, videoRequest("before")), wire.VideoStateCompleted)
	args := strings.Split(readFile(f.dir, "args.txt"), "\n")
	for _, name := range []string{"diffusion.gguf", "encoder.gguf"} {
		want := filepath.Join(downloads, name)
		found := false
		for _, a := range args {
			found = found || strings.EqualFold(a, want)
		}
		if !found {
			t.Fatalf("sd-server was not given %s: %q", want, args)
		}
	}
	h.Close()
	if h.Loaded(testVideoModel) {
		t.Fatal("the engine kept running")
	}

	elsewhere := filepath.Join(f.dir, "elsewhere")
	if err := os.MkdirAll(elsewhere, 0o700); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"diffusion.gguf", "encoder.gguf"} {
		same := strings.Repeat("Z", len("weights of "+name))
		if err := os.WriteFile(filepath.Join(elsewhere, name), []byte(same), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// Close returns only after the engine's held model files are released.
	if err := os.Rename(downloads, downloads+".real"); err != nil {
		t.Fatalf("the folder stayed held after Close: %v", err)
	}
	junction(t, downloads, elsewhere)
	leases := newFakeLeases()
	h = f.host(t, leases, nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("after")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != NoEngine {
		t.Fatalf("observed %+v", observed)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatal("a lease was asked for files behind a junction")
	}
}
