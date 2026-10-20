package host

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	"github.com/openabstractions/abstraction-provider-kit/go/videoops"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	"github.com/openabstractions/abstraction-resource/go/holder"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// VideoOptions is the video engine kind: one model name callers ask for,
// served by one sd-server this provider owns (CONTRACT.md MODELHOST-V1..V9).
type VideoOptions struct {
	// Model is the name the router lists and callers ask for; empty reads
	// the name of Engine.Set, and any other name is refused.
	Model string
	// Engine is the engine and its model set; the set carries the preset
	// (INF-G8) and the bounds a request is held to (MODELHOST-V6).
	Engine VideoEngine
	// Store is the provider's content store: the runtime stages first
	// frames in it and reads results from it.
	Store *storage.ContentStore
	// TimeLimit bounds one generation once the engine accepted it; zero
	// reads DefaultVideoTimeLimit.
	TimeLimit time.Duration
	// Poll is how often a job is read; zero reads defaultVideoPoll.
	Poll time.Duration
	// Retention is how long an ended operation and its request_id stay
	// held; zero reads defaultVideoRetention, and less is raised to it.
	Retention time.Duration
}

const (
	defaultVideoPoll = time.Second
	// maxEngineErrors is how many failed job reads in a row end a job.
	maxEngineErrors = 30
	// DefaultVideoMemoryLimit is the engine job's per-process memory limit
	// on Windows: the measured peak commit of 1280x704x121, 32.8 GB, with
	// about 30 percent above it (research/live-runs/2026-10-01.md, video
	// addendum condition 11).
	DefaultVideoMemoryLimit = 40 << 30
)

// videoHost is the model host's video@1: the provider kit's operation table
// (videoops, KIT-V1..KIT-V6) running the set on sd-server, with the running
// sd-server's address and process.
type videoHost struct {
	*videoops.Table
	host *Host
	opts VideoOptions
	// kind is the set's engine: sd-server.
	kind videoops.Kind
	// beforeSubmit, set by tests only, runs with the engine's base after
	// the engine is ready and before the job is submitted.
	beforeSubmit func(base string)

	mu      sync.Mutex
	base    string
	process *Process
}

func newVideoHost(h *Host, opts VideoOptions) (*videoHost, error) {
	setName := opts.Engine.Set.Name
	if opts.Model == "" {
		opts.Model = setName
	}
	if opts.Model == "" || opts.Store == nil {
		return nil, errors.New("modelhost: the video kind needs a model set and a content store")
	}
	if opts.Model != setName {
		return nil, fmt.Errorf("modelhost: the video model %q is not the model set %q", opts.Model, setName)
	}
	if err := opts.Engine.Set.valid(); err != nil {
		return nil, fmt.Errorf("modelhost: %w", err)
	}
	if opts.Poll <= 0 {
		opts.Poll = defaultVideoPoll
	}
	table, err := videoops.New(videoops.Options{Model: opts.Model, Store: opts.Store, TimeLimit: opts.TimeLimit,
		Retention: opts.Retention, Holder: h.resources, GPU: h.options.GPU, Now: h.options.Now})
	if err != nil {
		return nil, fmt.Errorf("modelhost: %w", err)
	}
	v := &videoHost{Table: table, host: h, opts: opts}
	var kind videoops.Kind = sdKind{v: v}
	v.kind = kind
	table.Serve(kind)
	return v, nil
}

func (v *videoHost) close() { v.Table.Close() }

// active reports queued or running video work; the holder keeps the engine
// while it is (video addendum condition 6).
func (v *videoHost) active() bool { return v.Table.Active() }

// setStep records the generation's percent from the engine's output, at
// most 99 until the result is kept.
func (v *videoHost) setStep(op *videoOp) {
	v.mu.Lock()
	p := v.process
	v.mu.Unlock()
	if p == nil || p.progress == nil {
		return
	}
	percent, ok := p.progress.percent()
	if !ok {
		return
	}
	v.SetPercent(op, percent)
}

// resetProgress starts the engine's step counts afresh for the next job.
func (v *videoHost) resetProgress() {
	v.mu.Lock()
	p := v.process
	v.mu.Unlock()
	if p != nil && p.progress != nil {
		p.progress.reset()
	}
}

// sdKind runs a video on the pinned sd-server.
type sdKind struct{ v *videoHost }

func (k sdKind) Bounds() videoBounds {
	s := k.v.opts.Engine.Set
	return videoBounds{Width: s.Preset.Width, Height: s.Preset.Height, FPS: s.Preset.FPS, Sizes: s.Sizes,
		MaxFrames: s.MaxFrames, MaxFPS: s.MaxFPS, FirstFrame: s.FirstFrame}
}

// admit builds the body once, so a request the engine would refuse is
// refused before it is queued.
func (k sdKind) Admit(params sdcpp.VideoParams) (wire.StartOutcome, string) {
	if _, err := sdcpp.BuildVideoRequest(k.v.opts.Engine.Set.Preset, params); err != nil {
		return wire.StartOutcomeUnsupportedFeature, reasonOf(err)
	}
	return wire.StartOutcomeAccepted, ""
}

func (k sdKind) Fingerprint(params sdcpp.VideoParams) gpufault.Fingerprint {
	return k.v.fingerprint(params)
}

func (k sdKind) Run(ctx context.Context, op *videoOp) {
	v := k.v
	base, err := v.host.ensureVideo(ctx)
	if cancelled(op) {
		return
	}
	if err != nil {
		v.Finish(op, wire.VideoStateFailed, loadFailure(err), nil)
		return
	}
	params := v.Params(op)
	body, err := sdcpp.BuildVideoRequest(v.opts.Engine.Set.Preset, params)
	if err != nil {
		v.Finish(op, wire.VideoStateFailed, reasonOf(err), nil)
		return
	}
	if v.beforeSubmit != nil {
		v.beforeSubmit(base)
	}
	v.resetProgress()
	job, err := v.submit(ctx, base, body)
	if err != nil {
		if cancelled(op) {
			// The engine may have taken the job before the cancel cut the
			// call short; only stopping it is sure to end that job.
			v.host.resources.Unload(v.opts.Model, "video cancel")
			return
		}
		v.Finish(op, wire.VideoStateFailed, videoEngineRefused, nil)
		return
	}
	v.follow(ctx, op, params, base, job)
}

// fingerprint is what a video dispatches on the GPU: the model, the pinned
// engine, the size, the length, whether a first frame conditions it, and
// the engine's configured arguments. The prompt and seed are not part of it.
func (v *videoHost) fingerprint(p sdcpp.VideoParams) gpufault.Fingerprint {
	set := v.opts.Engine.Set
	return gpufault.Fingerprint{Model: v.opts.Model + " set=" + set.Name + " options=" + strings.Join(set.Options, " "),
		Engine: "sd-server " + v.opts.Engine.Pin.Release,
		Shape: fmt.Sprintf("%dx%dx%d fps%d image=%t args=%s", p.Width, p.Height, p.Frames, p.FPS,
			len(p.InitImage) > 0, strings.Join(v.opts.Engine.Arguments, " "))}
}

// errVideoEngineGone is a call for an engine that no longer runs at the base
// it was started on.
var errVideoEngineGone = errors.New("the video engine at this address is not running")

// engine is the bound client of the running sd-server at base: each of its
// connections is bound to the engine's contained tree by the TCP owner
// table, so a process that took the port after the engine exited receives
// no byte (inference CONTRACT.md INF-EL2, MODELHOST-L7).
func (v *videoHost) engine(base string) (*http.Client, error) {
	v.mu.Lock()
	p := v.process
	v.mu.Unlock()
	if p == nil || p.base != base || p.client == nil {
		return nil, errVideoEngineGone
	}
	select {
	case <-p.exited:
		return nil, errVideoEngineGone
	default:
	}
	return p.client, nil
}

// submit posts one checked body and returns the engine's job id.
func (v *videoHost) submit(ctx context.Context, base string, body []byte) (string, error) {
	if err := sdcpp.CheckVideoRequest(body); err != nil {
		return "", err
	}
	client, err := v.engine(base)
	if err != nil {
		return "", err
	}
	call, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(call, http.MethodPost, base+"/sdcpp/v1/vid_gen", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	job, err := sdcpp.ReadVideoJob(io.LimitReader(response.Body, sdcpp.MaxJobErrorBytes))
	if response.StatusCode != http.StatusAccepted || err != nil || !validJobID(job.ID) {
		return "", fmt.Errorf("the engine answered %d", response.StatusCode)
	}
	return job.ID, nil
}

// validJobID matches the job id the engine's routes accept.
func validJobID(id string) bool {
	if id == "" || len(id) > 128 {
		return false
	}
	for _, c := range id {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// follow polls the engine's job at a bounded interval until it ends, a
// cancel arrives or the preset's time passes (video addendum conditions 9
// and 10).
func (v *videoHost) follow(ctx context.Context, op *videoOp, params sdcpp.VideoParams, base, job string) {
	deadline := time.NewTimer(v.TimeLimit())
	defer deadline.Stop()
	poll := time.NewTicker(v.opts.Poll)
	defer poll.Stop()
	engineState, failures := sdcpp.JobQueued, 0
	for {
		select {
		case <-op.Cancel():
			v.stopJob(base, job, engineState)
			return
		case <-v.Done():
			v.Finish(op, wire.VideoStateFailed, videoEngineLost, nil)
			return
		case <-deadline.C:
			v.host.resources.Unload(v.opts.Model, "video time limit")
			v.Finish(op, wire.VideoStateFailed, videoEngineTimeout, nil)
			return
		case <-poll.C:
		}
		state, err := v.read(ctx, base, job)
		if err != nil {
			if cancelled(op) {
				continue
			}
			failures++
			if failures >= maxEngineErrors || !v.host.videoAlive() {
				v.host.resources.Unload(v.opts.Model, "video engine lost")
				v.Finish(op, wire.VideoStateFailed, v.Lost(ctx, op), nil)
				return
			}
			continue
		}
		failures = 0
		engineState = state.Status
		switch state.Status {
		case sdcpp.JobQueued:
			v.SetState(op, wire.VideoStateQueued, max(state.QueuePosition, 1))
		case sdcpp.JobGenerating:
			v.SetState(op, wire.VideoStateGenerating, 0)
			v.setStep(op)
		case sdcpp.JobCompleted:
			v.complete(op, params, state.Result)
			return
		case sdcpp.JobFailed:
			v.Finish(op, wire.VideoStateFailed, videoEngineFailed, nil)
			return
		case sdcpp.JobCancelled:
			// Only this provider cancels its jobs; another caller of the
			// port did.
			v.Finish(op, wire.VideoStateFailed, videoEngineFailed, nil)
			return
		}
	}
}

// read is one GET of the job, its answer read with the reviewed cap.
func (v *videoHost) read(ctx context.Context, base, job string) (sdcpp.VideoJob, error) {
	call, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	client, err := v.engine(base)
	if err != nil {
		return sdcpp.VideoJob{}, err
	}
	request, err := http.NewRequestWithContext(call, http.MethodGet, base+"/sdcpp/v1/jobs/"+job, nil)
	if err != nil {
		return sdcpp.VideoJob{}, err
	}
	response, err := client.Do(request)
	if err != nil {
		return sdcpp.VideoJob{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return sdcpp.VideoJob{}, fmt.Errorf("the engine answered %d", response.StatusCode)
	}
	state, err := sdcpp.ReadVideoJob(response.Body)
	if err != nil {
		return sdcpp.VideoJob{}, err
	}
	if state.ID != job {
		return sdcpp.VideoJob{}, errors.New("the engine answered for another job")
	}
	return state, nil
}

// complete checks the result against what was asked and keeps it in the
// provider's store as one local delivery (INF-G3).
func (v *videoHost) complete(op *videoOp, params sdcpp.VideoParams, result *sdcpp.VideoResult) {
	data, err := result.Container(v.opts.Engine.Set.Preset.OutputFormat, params.Frames, params.FPS)
	if err != nil {
		v.Finish(op, wire.VideoStateFailed, videoEngineResult, nil)
		return
	}
	media := http.DetectContentType(data)
	if format := v.opts.Engine.Set.Preset.OutputFormat; media != sdcpp.FormatMIME(format) && !(media == "video/avi" && format == sdcpp.FormatAVI) {
		v.Finish(op, wire.VideoStateFailed, videoEngineResult, nil)
		return
	}
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if err := v.Keep(digest, data); err != nil {
		v.Finish(op, wire.VideoStateFailed, "content:unavailable", nil)
		return
	}
	v.Finish(op, wire.VideoStateCompleted, "", &wire.Delivery{Digest: digest, MediaType: media, Size: int64(len(data)), Location: "local"})
}

// stopJob carries out a cancel: the engine's cancel route while it holds the
// job queued, and stopping its process tree once it generates or the route
// answers otherwise. The next job relaunches the engine.
func (v *videoHost) stopJob(base, job, state string) {
	if state == sdcpp.JobQueued {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		client, err := v.engine(base)
		var request *http.Request
		if err == nil {
			request, err = http.NewRequestWithContext(ctx, http.MethodPost, base+"/sdcpp/v1/jobs/"+job+"/cancel", nil)
		}
		if err == nil {
			response, err := client.Do(request)
			if err == nil {
				//unchecked: draining a bounded body for connection reuse; the status decides
				io.Copy(io.Discard, io.LimitReader(response.Body, sdcpp.MaxJobErrorBytes))
				//unchecked: the status decides; a failed close loses nothing read
				response.Body.Close()
				if response.StatusCode == http.StatusOK {
					return
				}
			}
		}
	}
	v.host.resources.Unload(v.opts.Model, "video cancel")
}

// ensureVideo loads the video engine under its lease, on a free loopback
// port chosen for this launch, and returns where it answers.
func (h *Host) ensureVideo(ctx context.Context) (string, error) {
	v := h.video
	engine := v.opts.Engine
	amount := engine.Set.Size()
	err := h.resources.Ensure(ctx, v.opts.Model, func() (holder.Component, error) {
		if err := engine.present(ctx); err != nil {
			return holder.Component{}, refuse(NoEngine, "%v", err)
		}
		return holder.Component{Amount: amount, Start: func() (holder.Owned, error) {
			port, err := FreePort()
			if err != nil {
				return nil, err
			}
			process, err := engine.start(port)
			if err != nil {
				return nil, err
			}
			v.mu.Lock()
			v.base, v.process = process.base, process
			v.mu.Unlock()
			ready := engine.Ready
			if ready <= 0 {
				ready = DefaultVideoReady
			}
			return &ownedProcess{name: v.opts.Model, process: process, ready: ready, host: h}, nil
		}}, nil
	})
	if err != nil {
		return "", h.mapFailure(v.opts.Model, amount, err)
	}
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.base, nil
}

// videoAlive reports whether the video engine last started still runs.
func (h *Host) videoAlive() bool {
	if h.video == nil {
		return false
	}
	h.video.mu.Lock()
	p := h.video.process
	h.video.mu.Unlock()
	if p == nil {
		return false
	}
	select {
	case <-p.exited:
		return false
	default:
		return true
	}
}

// VideoServer is this provider's video@1, for the service host's Video
// field; nil when the video kind is not configured.
func (h *Host) VideoServer() *videoHost { return h.video }
