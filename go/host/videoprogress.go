package host

import (
	"io"
	"regexp"
	"strconv"
	"sync"
)

// sd-server master-843 reports no progress through its job route: a job
// answer carries its status and queue position alone
// (examples/server/async_jobs.cpp make_async_job_json), and the server sets
// no progress callback. The library prints a progress bar to the engine's
// output instead (src/core/util.cpp print_progress_line): "  |====>   | 2/3 -
// 1.58s/it", once per sampling step (src/stable-diffusion.cpp, the sample
// loop) and once per decoded tile when the decode is tiled
// (src/core/ggml_extend.hpp, the tiling loop). Model loading draws its bar
// with '#'. A job also prints '=' bars before it samples, such as the first
// frame's encode, so a bar alone does not say which stage it belongs to: the
// library's own log lines do. "generate_video WxHxF" opens sampling and
// "sampling completed" (or "generating latent video completed") closes it
// (src/stable-diffusion.cpp). The provider reads those counts and lines from
// the output it keeps (MODELHOST-V8) and forwards none of the text.

// progressBar matches one progress line from its start, as
// print_progress_line writes it after its carriage return: two spaces, the
// 50-character bar ('=' for work, '#' for a load), step/steps and the rate.
// A bar anywhere else in a line, such as inside an echoed prompt, is not one.
var progressBar = regexp.MustCompile(`^  \|([=#> ]{50})\| (\d+)/(\d+) - [0-9.]+(?:s/it|it/s|[KMG]?B/s)`)

// The library's stage lines, as its log writes them: "[INFO ] <file:line> -
// <text>".
var (
	samplingStarts = regexp.MustCompile(`^\[INFO \] \S+ +- generate_video \d+x\d+x\d+\s*$`)
	samplingEnds   = regexp.MustCompile(`^\[INFO \] \S+ +- (?:sampling completed|generating latent video completed), taking `)
)

// Shares of a video's progress, in percent. Loading the weights and preparing
// the request (text conditioning, the first frame's encode) come first, then
// sampling, then the decode. The 512x512x81 runs of
// research/live-runs/2026-10-02.md spent about 7 s preparing a cold job, 41
// to 45 s sampling and 5 to 16 s decoding; the 1280x704x121 run of
// 2026-10-01 sampled for 361 s and decoded in tiles for 272 s.
const (
	loadShare     = 10
	samplingShare = 60
	decodeShare   = 30
)

// Stages of one job, in order.
const (
	stageLoad = iota
	stageSampling
	stageDecode
)

// progressTap reads an engine's output on its way to the log and keeps the
// progress of the job in progress.
type progressTap struct {
	next io.Writer

	mu      sync.Mutex
	pending []byte
	// stage is the job's current stage; seen is whether any bar was read
	// since reset; fraction is the stage's latest bar, in [0, 1]; best is the
	// highest percent reported, so progress never goes back.
	stage    int
	seen     bool
	fraction float64
	// samplingSteps is the step count of the sampling series, so a second
	// series without a closing line, as an engine without the stage lines
	// prints it, is read as the decode.
	samplingSteps int
	samplingDone  bool
	best          int
}

func (t *progressTap) Write(p []byte) (int, error) {
	t.mu.Lock()
	t.pending = append(t.pending, p...)
	for {
		i := indexLineEnd(t.pending)
		if i < 0 {
			break
		}
		t.read(t.pending[:i])
		t.pending = t.pending[i+1:]
	}
	if len(t.pending) > 4096 {
		t.pending = t.pending[len(t.pending)-4096:]
	}
	// The engine leaves its current bar open until the next one starts; a
	// tail that already holds a whole bar is read now. Reading the same bar
	// again when it closes changes nothing.
	if progressBar.Match(t.pending) {
		t.read(t.pending)
	}
	t.mu.Unlock()
	if t.next != nil {
		return t.next.Write(p)
	}
	return len(p), nil
}

func indexLineEnd(b []byte) int {
	for i, c := range b {
		if c == '\r' || c == '\n' {
			return i
		}
	}
	return -1
}

// read takes one output segment; t.mu is held.
func (t *progressTap) read(segment []byte) {
	switch {
	case samplingStarts.Match(segment):
		if t.stage == stageLoad {
			t.stage, t.fraction = stageSampling, 0
		}
		return
	case samplingEnds.Match(segment):
		if t.stage < stageDecode {
			t.stage, t.fraction = stageDecode, 0
		}
		return
	}
	m := progressBar.FindSubmatch(segment)
	if m == nil {
		return
	}
	step, err1 := strconv.Atoi(string(m[2]))
	steps, err2 := strconv.Atoi(string(m[3]))
	if err1 != nil || err2 != nil || steps < 1 || step < 1 || step > steps {
		return
	}
	load := len(m[1]) > 0 && (m[1][0] == '#')
	t.seen = true
	switch t.stage {
	case stageLoad:
		// Every bar before sampling, a load's or the first frame's encode,
		// is preparation.
		t.fraction = float64(step) / float64(steps)
	case stageSampling:
		if load {
			// A weight load inside the stage (the transformer is loaded lazily)
			// is not a step.
			return
		}
		if t.samplingDone && steps != t.samplingSteps || t.samplingDone && step < steps {
			// A second series without the closing line is the decode.
			t.stage, t.fraction = stageDecode, float64(step)/float64(steps)
			return
		}
		t.samplingSteps, t.samplingDone, t.fraction = steps, step == steps, float64(step)/float64(steps)
	case stageDecode:
		if load {
			return
		}
		t.fraction = float64(step) / float64(steps)
	}
}

// reset forgets the progress of the job before.
func (t *progressTap) reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.stage, t.seen, t.fraction, t.samplingSteps, t.samplingDone, t.best = stageLoad, false, 0, 0, false, 0
}

// percent is the job's progress in percent: loading and preparing fill
// loadShare, sampling the next samplingShare, the decode the rest. It never
// goes back. ok is false before the engine has printed a bar or a stage line
// for the job.
func (t *progressTap) percent() (int, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.seen && t.stage == stageLoad {
		return 0, false
	}
	var p int
	switch t.stage {
	case stageLoad:
		p = int(loadShare * t.fraction)
	case stageSampling:
		p = loadShare + int(samplingShare*t.fraction)
	default:
		p = loadShare + samplingShare + int(decodeShare*t.fraction)
	}
	if p > t.best {
		t.best = p
	}
	return t.best, true
}
