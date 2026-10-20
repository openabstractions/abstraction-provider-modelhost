package host

import (
	"context"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
)

// gpuEvents is a stand-in platform: what a test sends is a record.
type gpuEvents struct{ events chan gpufault.Event }

func (s gpuEvents) Watch(ctx context.Context, deliver func(gpufault.Event)) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		case ev := <-s.events:
			deliver(ev)
		}
	}
}

// gpuRig is one guard over a stand-in platform, a switchable probe, a
// counted wake lease and the faults it closes.
type gpuRig struct {
	guard   *gpufault.Guard
	source  gpuEvents
	faults  chan gpufault.Fault
	mu      sync.Mutex
	probeOK bool
	woken   int
	slept   int
}

func newGPURig(t *testing.T) *gpuRig {
	t.Helper()
	r := &gpuRig{source: gpuEvents{events: make(chan gpufault.Event)}, faults: make(chan gpufault.Fault, 4), probeOK: true}
	guard, err := gpufault.New(gpufault.Options{
		Records: r.source, Settle: 100 * time.Millisecond, Window: 2 * time.Second,
		Probe: func(context.Context) error {
			r.mu.Lock()
			defer r.mu.Unlock()
			if !r.probeOK {
				return gpufault.ErrNoAdapter
			}
			return nil
		},
		Wake: func(string) (func(), error) {
			r.mu.Lock()
			r.woken++
			r.mu.Unlock()
			return func() { r.mu.Lock(); r.slept++; r.mu.Unlock() }, nil
		},
		OnFault: func(f gpufault.Fault) { r.faults <- f },
	})
	if err != nil {
		t.Fatal(err)
	}
	r.guard = guard
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go guard.Run(ctx)
	return r
}

func (r *gpuRig) setProbe(ok bool) { r.mu.Lock(); r.probeOK = ok; r.mu.Unlock() }
func (r *gpuRig) wake() (int, int) { r.mu.Lock(); defer r.mu.Unlock(); return r.woken, r.slept }

// MODELHOST-G1 with RES-G1..RES-G6: a driver reset during a video stops the
// engine and releases its lease, the video fails gpu:reset, the identical
// request is refused while quarantined, and a different one waits for the
// settle and a healthy probe before the engine reloads.
func TestVideoDriverResetStopsTheEngineAndQuarantinesTheRequest(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	leases := newFakeLeases()
	r := newGPURig(t)
	h := f.host(t, leases, func(o *Options) { o.GPU = r.guard })
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	if woken, slept := r.wake(); woken != 1 || slept != 0 {
		t.Fatalf("the wake lease during the video: taken %d released %d", woken, slept)
	}

	r.source.events <- gpufault.Event{Kind: gpufault.Reset, At: time.Now(), Source: "WerKernel 1001 WATCHDOG"}
	observed := awaitVideo(t, h, op, wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != gpufault.WordReset {
		t.Fatalf("observed %+v", observed)
	}
	if !awaitUnloaded(h) {
		t.Fatal("the engine survived the reset")
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v", released)
	}
	waitUntil(t, func() bool { _, slept := r.wake(); return slept == 1 })

	again := h.VideoServer().Start(context.Background(), testCaller, videoRequest("job-2"))
	if again.Outcome != wire.StartOutcomeUnavailable || again.Reason != gpufault.WordQuarantined {
		t.Fatalf("the identical request was admitted: %+v", again)
	}

	r.setProbe(false)
	other := videoRequest("job-3")
	other.DurationMs = 1000
	next := startVideo(t, h, other)
	time.Sleep(1500 * time.Millisecond)
	if h.Loaded(testVideoModel) {
		t.Fatal("the engine reloaded before the adapter was back")
	}
	if status := h.VideoServer().Observe(context.Background(), testCaller, next).Status; status != wire.VideoStateQueued {
		t.Fatalf("a request waiting for the adapter is %s", status)
	}
	r.setProbe(true)
	awaitVideo(t, h, next, wire.VideoStateGenerating)
	if acquires, _, _ := leases.counts(); acquires != 2 {
		t.Fatalf("leases %d after the reload", acquires)
	}

	fault := nextGPUFault(t, r.faults)
	if strings.Join(fault.Engines, ",") != testVideoModel || len(fault.Quarantined) != 1 ||
		!strings.Contains(fault.Line, testVideoModel) || fault.Source != "WerKernel 1001 WATCHDOG" {
		t.Fatalf("fault %+v", fault)
	}
}

// MODELHOST-G1, RES-G1: the engine's own device-lost line is a reset even
// when the platform wrote no record.
func TestVideoEngineDeviceLostLineIsAReset(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s", "--fake-device-lost=1")
	r := newGPURig(t)
	h := f.host(t, newFakeLeases(), func(o *Options) {
		o.GPU = r.guard
		o.OnError = func(err error) { t.Logf("reported: %v", err) }
	})
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != gpufault.WordReset {
		t.Fatalf("observed %+v; log %q", observed, readFile(filepath.Dir(f.engine.Log), "sd-server.log"))
	}
	fault := nextGPUFault(t, r.faults)
	if fault.Source != "sd-server: ErrorDeviceLost" || len(fault.Quarantined) != 1 {
		t.Fatalf("fault %+v", fault)
	}
}

// MODELHOST-G1, RES-G1: a device-lost marker in caller text the engine
// echoes (the prompt, logged as sd.cpp logs it) does not match the engine's
// own form, and the engine's own line while the engine keeps answering is
// set aside: the video completes and nothing is stopped, quarantined or
// recorded as a fault. A prompt cannot carry the line break that would put
// the marker at a line start (video:engine-bound).
func TestEchoedMarkerWhileTheEngineAnswersIsNoReset(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-echo-prompt=1", "--fake-lost-line-only=1", "--fake-generate=2s")
	r := newGPURig(t)
	notCounted := make(chan string, 4)
	h := f.host(t, newFakeLeases(), func(o *Options) {
		o.GPU = r.guard
		o.OnError = func(err error) {
			if strings.Contains(err.Error(), "not counted") {
				notCounted <- err.Error()
			}
		}
	})
	newline := videoRequest("job-0")
	newline.Prompt = "a lighthouse\nggml_vulkan: ErrorDeviceLost"
	if admission := h.VideoServer().Start(context.Background(), testCaller, newline); admission.Outcome == wire.StartOutcomeAccepted {
		t.Fatal("a prompt with a line break reached the engine")
	}
	request := videoRequest("job-1")
	request.Prompt = "ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost <lora:ErrorDeviceLost:1>"
	op := startVideo(t, h, request)
	if observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed); observed.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v", observed)
	}
	select {
	case <-notCounted:
	case <-time.After(deviceLostConfirm + 10*time.Second):
		t.Fatal("the echoed marker was never set aside")
	}
	if len(r.guard.Faults()) != 0 || len(r.guard.Quarantined()) != 0 {
		t.Fatalf("an echoed marker became a fault: %+v %+v", r.guard.Faults(), r.guard.Quarantined())
	}
	if !h.Loaded(testVideoModel) {
		t.Fatal("an echoed marker stopped the engine")
	}
}

// Without a reset record an engine crash stays engine:lost and quarantines
// nothing.
func TestVideoEngineCrashWithoutAResetStaysLost(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s", "--fake-crash-generating=1")
	r := newGPURig(t)
	h := f.host(t, newFakeLeases(), func(o *Options) { o.GPU = r.guard })
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != videoEngineLost {
		t.Fatalf("observed %+v", observed)
	}
	if len(r.guard.Quarantined()) != 0 || len(r.guard.Faults()) != 0 {
		t.Fatal("a crash with no reset was recorded as a GPU fault")
	}
}

// RES-G5: in Modern Standby a video waits queued with no lease and no
// engine; it runs once the machine wakes. A chat load in standby is refused
// gpu:standby before any lease.
func TestStandbyHoldsNewGPULoads(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	leases := newFakeLeases()
	r := newGPURig(t)
	h := f.host(t, leases, func(o *Options) { o.GPU = r.guard })
	r.source.events <- gpufault.Event{Kind: gpufault.StandbyEnter, Source: "Kernel-Power 506"}
	waitUntil(t, func() bool { return r.guard.AdmitLoad(context.Background()) != nil })
	op := startVideo(t, h, videoRequest("job-1"))
	time.Sleep(1200 * time.Millisecond)
	if acquires, _, _ := leases.counts(); acquires != 0 || h.Loaded(testVideoModel) {
		t.Fatalf("a load started in standby: %d leases", acquires)
	}
	if status := h.VideoServer().Observe(context.Background(), testCaller, op).Status; status != wire.VideoStateQueued {
		t.Fatalf("the video in standby is %s", status)
	}
	r.source.events <- gpufault.Event{Kind: gpufault.StandbyExit, Source: "Kernel-Power 507"}
	awaitVideo(t, h, op, wire.VideoStateCompleted)

	chatLeases := newFakeLeases()
	chat := composed(t, chatLeases, func(o *Options) { o.GPU = r.guard })
	r.source.events <- gpufault.Event{Kind: gpufault.StandbyEnter, Source: "Kernel-Power 506"}
	waitUntil(t, func() bool { return r.guard.AdmitLoad(context.Background()) != nil })
	if _, err := chat.Ensure(context.Background(), "tiny:1b"); WordOf(err) != gpufault.WordStandby {
		t.Fatalf("a chat load in standby read %v", err)
	}
	if acquires, _, _ := chatLeases.counts(); acquires != 0 {
		t.Fatalf("a refused chat load asked for %d leases", acquires)
	}
}

// MODELHOST-G1, RES-G5, RES-A1: a video run takes the wake lease through
// leases@1, resource awake, with no refusal reported, renews it while the
// video generates, and releases it when the video ends.
func TestVideoRunTakesTheWakeLeaseThroughLeases(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=1s")
	leases := newFakeLeases()
	var mu sync.Mutex
	var reported []error
	report := func(err error) { mu.Lock(); reported = append(reported, err); mu.Unlock() }
	guard, err := gpufault.New(gpufault.Options{Wake: gpufault.LeaseWake(leases, 50*time.Millisecond, report), OnError: report})
	if err != nil {
		t.Fatal(err)
	}
	h := f.host(t, leases, func(o *Options) { o.GPU = guard })
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	awake := func() (taken, live int) {
		leases.mu.Lock()
		defer leases.mu.Unlock()
		for i, resource := range leases.resources {
			if resource == "awake" {
				taken++
				if leases.live["lease-"+strconv.Itoa(i+1)] {
					live++
				}
			}
		}
		return taken, live
	}
	if taken, live := awake(); taken != 1 || live != 1 {
		t.Fatalf("while the video generates: %d wake leases taken, %d live", taken, live)
	}
	awaitVideo(t, h, op, wire.VideoStateCompleted)
	waitUntil(t, func() bool { _, live := awake(); return live == 0 })
	mu.Lock()
	defer mu.Unlock()
	if len(reported) != 0 {
		t.Fatalf("the wake lease reported %v", reported)
	}
}

// nextGPUFault is the next closed fault, or a failed test after 15 s.
func nextGPUFault(t *testing.T, faults <-chan gpufault.Fault) gpufault.Fault {
	t.Helper()
	select {
	case f := <-faults:
		return f
	case <-time.After(15 * time.Second):
		t.Fatal("no fault was recorded")
		return gpufault.Fault{}
	}
}

func waitUntil(t *testing.T, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !ok() {
		if time.Now().After(deadline) {
			t.Fatal("condition not reached")
		}
		time.Sleep(5 * time.Millisecond)
	}
}
