package host

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

func bar(fill string, step, steps int) string {
	n := 50 * step / steps
	return "\r  |" + strings.Repeat(fill, n) + strings.Repeat(" ", 50-n) + fmt.Sprintf("| %d/%d - 1.58s/it", step, steps)
}

// sd-server reports progress only as its output's progress bars and stage
// lines; the provider reads them: loading and preparing fill the first 10
// percent, sampling the next 60, the decode the rest. Other text counts for
// nothing.
func TestProgressIsReadFromTheEngineOutput(t *testing.T) {
	tap := &progressTap{}
	if _, ok := tap.percent(); ok {
		t.Fatal("progress before any step")
	}
	tap.Write([]byte("[INFO ] loading a prompt about lighthouses\n"))
	if _, ok := tap.percent(); ok {
		t.Fatal("a log line counted as progress")
	}
	tap.Write([]byte(bar("#", 10, 20) + "\n"))
	if got, ok := tap.percent(); !ok || got != 5 {
		t.Fatalf("half a load read as %d%% (%v), want 5%%", got, ok)
	}
	tap.Write([]byte("[INFO ] stable-diffusion.cpp:6941 - generate_video 832x480x49\n"))
	for i, want := range []struct {
		line    string
		percent int
	}{{bar("=", 1, 3), 30}, {bar("=", 3, 3) + "\n", 70},
		{"[INFO ] stable-diffusion.cpp:7038 - sampling completed, taking 45.16s\n", 70},
		{bar("=", 2, 8), 77}, {bar("=", 8, 8) + "\n", 100}} {
		// A bar split across two writes is read once, at its end.
		half := len(want.line) / 2
		tap.Write([]byte(want.line[:half]))
		tap.Write([]byte(want.line[half:] + "\r"))
		if got, ok := tap.percent(); !ok || got != want.percent {
			t.Fatalf("step %d: %d%% (%v), want %d%%", i, got, ok, want.percent)
		}
	}
	tap.reset()
	if _, ok := tap.percent(); ok {
		t.Fatal("progress survived the reset")
	}
	// An engine that prints no stage lines: the first '=' series before any
	// line is preparation, which holds the job inside its first 10 percent.
	tap.Write([]byte(bar("=", 3, 3) + "\n"))
	if got, ok := tap.percent(); !ok || got != 10 {
		t.Fatalf("an unmarked series read as %d%% (%v), want 10%%", got, ok)
	}
	tap.reset()
	// A bar inside a log line, such as an echoed prompt, is not progress:
	// only a segment that starts as print_progress_line's does counts.
	for _, line := range []string{
		"[INFO ] generate_video: prompt: a cat " + strings.TrimPrefix(bar("=", 59, 60), "\r") + "\n",
		"prompt: a cat |" + strings.Repeat("=", 50) + "| 59/60 - 1.00s/it\n",
	} {
		tap.Write([]byte(line))
		if got, ok := tap.percent(); ok {
			t.Fatalf("a bar inside a log line read as %d%%: %q", got, line)
		}
	}
}

// The engine output of a cold hummingbird-xt-i2v job on 2026-10-02
// (research/live-runs/2026-10-02.md, run 2a), in order: the TAE load, the
// first frame's encode (an '=' bar of one step), the text encoder load, the
// stage line, the transformer load, three sampling steps, the closing line,
// the decoder load and a one-tile decode. Observe reported 60 percent right
// after the encode, before sampling began; load is its own stage now.
func TestLoadAndTheFirstFrameEncodeAreNotSampling(t *testing.T) {
	loadBar := func(step, steps int, rate string) string {
		n := 50 * step / steps
		return "\r  |" + strings.Repeat("#", n) + strings.Repeat(" ", 50-n) + fmt.Sprintf("| %d/%d - %s\x1b[K", step, steps, rate)
	}
	tap := &progressTap{}
	for i, want := range []struct {
		output    string
		atMost    int
		atLeast   int
		whatItWas string
	}{
		{"[INFO ] stable-diffusion.cpp:4398 - sampling using LCM method\n" + loadBar(1, 64, "0.00MB/s") + loadBar(64, 64, "13.91MB/s") + "\n", 10, 0, "the TAE load"},
		{"[INFO ] model_loader.cpp:1243 - loading tensors completed, taking 0.20s\n" + bar("=", 1, 1) + "\n", 10, 0, "the first frame's encode"},
		{"[INFO ] stable-diffusion.cpp:6414 - encode_first_stage completed, taking 666 ms\n" + loadBar(68, 242, "3.50GB/s") + loadBar(242, 242, "3.08GB/s") + "\n", 10, 0, "the text encoder load"},
		{"[INFO ] stable-diffusion.cpp:6576 - get_learned_condition completed, taking 4.39s\n[INFO ] stable-diffusion.cpp:6941 - generate_video 512x512x81\n" + loadBar(141, 825, "2.93GB/s") + loadBar(825, 825, "5.00GB/s") + "\n", 10, 0, "the transformer load"},
		{bar("=", 1, 3), 30, 30, "sampling step 1 of 3"},
		{bar("=", 2, 3), 50, 50, "sampling step 2 of 3"},
		{bar("=", 3, 3) + "\n", 70, 70, "sampling step 3 of 3"},
		{"[INFO ] stable-diffusion.cpp:7038 - sampling completed, taking 45.16s\n[INFO ] stable-diffusion.cpp:7182 - generating latent video completed, taking 45.78s\n" + loadBar(64, 64, "91.00MB/s") + "\n", 70, 70, "the decoder load"},
		{bar("=", 1, 1) + "\n", 100, 100, "the decode"},
	} {
		tap.Write([]byte(want.output))
		got, ok := tap.percent()
		// A load step may carry no progress yet; a later one must.
		if (!ok && want.atLeast > 0) || got > want.atMost || got < want.atLeast {
			t.Fatalf("step %d, after %s: %d%% (%v), want %d..%d%%", i, want.whatItWas, got, ok, want.atLeast, want.atMost)
		}
	}
}

// INF-G6: while the engine generates, Observe carries the percent read from
// its output, below 100 until the result is kept.
func TestVideoObserveReportsGenerationProgress(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=4s")
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, videoRequest("job-1"))
	deadline := time.Now().Add(30 * time.Second)
	var seen []int64
	for time.Now().Before(deadline) {
		observed := h.VideoServer().Observe(context.Background(), testCaller, op)
		if observed.Status == wire.VideoStateGenerating && observed.Steps == 100 && observed.Step > 0 {
			seen = append(seen, observed.Step)
		}
		if observed.Status == wire.VideoStateCompleted || observed.Status == wire.VideoStateFailed {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if len(seen) == 0 || seen[len(seen)-1] <= seen[0] || seen[len(seen)-1] > 99 {
		t.Fatalf("progress seen %v", seen)
	}
}
