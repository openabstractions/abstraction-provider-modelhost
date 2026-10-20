package host

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// MODELHOST-L10: a llama-server launch warms its model file and logs it.
func TestLaunchWarmsTheModelFile(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	defer h.Unload("tiny:1b", "test")
	logs, err := filepath.Glob(filepath.Join(filepath.Dir(h.options.Engine.Record), "logs", "llama-*.log"))
	if err != nil || len(logs) != 1 {
		t.Fatalf("logs %v %v", logs, err)
	}
	awaitLogLine(t, logs[0], "phase=warm ", " GB in ")
}

// MODELHOST-L10: an sd-server launch warms the set's files after the proof
// that held them, and logs it.
func TestVideoLaunchWarmsTheModelSet(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	// Trusted in place: the launch proves the files by their metadata
	// record and reads none of them, and warming reads them instead.
	f.engine.Files = programfiles.NewVerifier(programfiles.Options{ACL: openACL{}, TrustPlaced: func(string, trustedpath.Owners) error { return nil }})
	if err := CheckVideoModels(context.Background(), f.engine.Files, f.engine.Set, f.engine.Places, f.engine.ModelRecord); err != nil {
		t.Fatal(err)
	}
	oldMemory := availableMemory
	availableMemory = func() int64 { return 1 << 30 }
	defer func() { availableMemory = oldMemory }()
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, videoRequest("job-warm"))
	if observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed); observed.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v", observed)
	}
	awaitLogLine(t, f.engine.Log, "phase=warm ", " GB in ")
}

func awaitLogLine(t *testing.T, path string, words ...string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		body, _ := os.ReadFile(path)
		for _, line := range strings.Split(string(body), "\n") {
			if containsAll(line, words) {
				return
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s has no line with %q: %s", path, words, body)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func containsAll(line string, words []string) bool {
	for _, w := range words {
		if !strings.Contains(line, w) {
			return false
		}
	}
	return true
}

func writeSized(t *testing.T, path string, size int) {
	t.Helper()
	data := make([]byte, size)
	for i := range data {
		data[i] = byte(i * 7)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestWarmFilesReadsEveryByte(t *testing.T) {
	dir := t.TempDir()
	sizes := []int{3*warmChunk + 7, 5, warmChunk}
	var paths []string
	var total int64
	for i, size := range sizes {
		p := filepath.Join(dir, string(rune('a'+i)))
		writeSized(t, p, size)
		paths = append(paths, p)
		total += int64(size)
	}
	defer func(old func() int64) { availableMemory = old }(availableMemory)
	availableMemory = func() int64 { return 2 * total }
	r := warmFiles(context.Background(), paths)
	if r.Skipped != "" || r.Err != nil || r.Bytes != total {
		t.Fatalf("result %+v, want %d bytes", r, total)
	}
}

func TestWarmFilesLeavesFilesLargerThanHalfTheFreeMemory(t *testing.T) {
	p := filepath.Join(t.TempDir(), "model")
	writeSized(t, p, 1000)
	defer func(old func() int64) { availableMemory = old }(availableMemory)
	for _, avail := range []int64{0, 1999} {
		availableMemory = func() int64 { return avail }
		if r := warmFiles(context.Background(), []string{p}); r.Skipped == "" || r.Bytes != 0 {
			t.Fatalf("available %d: %+v", avail, r)
		}
	}
	availableMemory = func() int64 { return 2000 }
	if r := warmFiles(context.Background(), []string{p}); r.Skipped != "" || r.Bytes != 1000 {
		t.Fatalf("available 2000: %+v", r)
	}
}

func TestWarmFilesStopsWhenTheEngineExits(t *testing.T) {
	p := filepath.Join(t.TempDir(), "model")
	const size = 4 * warmChunk
	writeSized(t, p, size)
	defer func(old func() int64) { availableMemory = old }(availableMemory)
	availableMemory = func() int64 { return 2 * size }
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := warmFiles(ctx, []string{p})
	if r.Err == nil || r.Bytes != 0 {
		t.Fatalf("result %+v", r)
	}
	if !strings.Contains(r.String(), "stopped") {
		t.Fatalf("line %q", r)
	}
}

// sd-server master-843 reads LTX-2.5's Gemma encoder first, then the
// transformer, then the decoders.
func TestWarmOrderFollowsTheEngine(t *testing.T) {
	where := map[string]string{}
	for _, f := range LTX25.Files {
		where[f.Name] = `C:\m\` + f.Name
	}
	got := warmOrder(LTX25, where)
	if len(got) != len(LTX25.Files) {
		t.Fatalf("order %q", got)
	}
	first := []string{LTXGemmaQ8.Name, "ltx-2.5-22b-distilled-transformer-Q8_0.gguf", "ltx-2.5-audio-vae-bf16.safetensors", "ltx-2.5-video-vae-conv-bf16.safetensors"}
	for i, name := range first {
		if got[i] != `C:\m\`+name {
			t.Fatalf("position %d is %s in %q", i, got[i], got)
		}
	}
	if slices.Contains(warmOrder(LTX25, map[string]string{}), "") {
		t.Fatal("unplaced file in the order")
	}
}
