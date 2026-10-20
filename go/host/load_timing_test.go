package host

import (
	"context"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	res "github.com/openabstractions/abstraction-resource/go/client"
)

func TestEnsureMeasuredDistinguishesColdAndResident(t *testing.T) {
	requireEnginePeer(t)
	var ticks atomic.Int64
	base := time.Now()
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Now = func() time.Time { return base.Add(time.Duration(ticks.Add(1)) * time.Millisecond) }
	})
	_, first, err := h.EnsureMeasured(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	if first.Milliseconds <= 0 || first.State != "cold" || first.Bytes == nil || *first.Bytes != 4096 {
		t.Fatalf("first load facts: %+v", first)
	}
	// A resident observation completes while the mutable engine map is
	// unavailable to this waiter.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	h.engineMu.Lock()
	_, resident, err := h.EnsureMeasured(ctx, "tiny:1b")
	h.engineMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if resident.Milliseconds <= 0 || resident.State != "loaded" || resident.Bytes != nil {
		t.Fatalf("resident facts: %+v", resident)
	}
	if acquires, _, _ := leases.counts(); acquires != 1 {
		t.Fatalf("measurement added resource work: %d", acquires)
	}
}

func TestEnsureMeasuredFailureHasNoLoadState(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.fixture = []string{"--fake-fail-line", "failed to allocate model"}
	})
	if _, timing, err := h.EnsureMeasured(context.Background(), "tiny:1b"); WordOf(err) != EngineDoesNotFit || timing.State != "" {
		t.Fatalf("failed readiness: %v, timing %+v", err, timing)
	}
}

// Pause the fake book after the real engine start, before the holder checks
// readiness. Both waits can join deterministically without a timed engine.
type timingLeaseBook struct {
	*freeBook
	binding chan struct{}
	ready   chan struct{}
	once    sync.Once
}

func (b *timingLeaseBook) BindProcess(lease string, pid int) (res.ProcessBindResult, error) {
	b.once.Do(func() { close(b.binding) })
	<-b.ready
	return b.freeBook.BindProcess(lease, pid)
}

func TestEnsureMeasuredJoinsReadiness(t *testing.T) {
	requireEnginePeer(t)
	for _, name := range []string{"foreground", "speculative"} {
		t.Run(name, func(t *testing.T) {
			leases := &timingLeaseBook{
				freeBook: &freeBook{fakeLeases: newFakeLeases()},
				binding:  make(chan struct{}),
				ready:    make(chan struct{}),
			}
			h := freeHost(t, leases.freeBook, func(o *Options) { o.Leases = leases })
			release := sync.OnceFunc(func() { close(leases.ready) })
			defer release()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			type result struct {
				timing LoadTiming
				err    error
			}
			first, joined := make(chan result, 1), make(chan result, 1)
			ensure := h.EnsureMeasured
			if name == "speculative" {
				ensure = h.EnsureSpeculativeMeasured
			}
			go func() {
				_, timing, err := ensure(ctx, "tiny:1b")
				first <- result{timing, err}
			}()
			select {
			case <-leases.binding:
			case <-ctx.Done():
				t.Fatal("engine never reached lease binding")
			}
			go func() {
				_, timing, err := h.EnsureMeasured(ctx, "tiny:1b")
				joined <- result{timing, err}
			}()
			ticker := time.NewTicker(time.Millisecond)
			defer ticker.Stop()
			for {
				h.loadMu.Lock()
				load := h.loads["tiny:1b"][0]
				registered := load != nil && (name == "speculative" || load.waiters.Load() == 2)
				h.loadMu.Unlock()
				if registered {
					break
				}
				select {
				case <-ticker.C:
				case <-ctx.Done():
					t.Fatal("foreground wait never registered")
				}
			}
			release()
			for index, done := range []chan result{first, joined} {
				want := "cold"
				if index == 1 && name == "speculative" {
					// The foreground owner finds the speculative engine resident.
					want = "loaded"
				}
				select {
				case got := <-done:
					if got.err != nil || got.timing.State != want {
						t.Fatalf("joined load: %v, timing %+v, want %s", got.err, got.timing, want)
					}
				case <-ctx.Done():
					t.Fatal("load wait did not finish")
				}
			}
			if acquired, _, _ := leases.counts(); acquired+leases.freeCalls() != 1 {
				t.Fatalf("joined waits acquired separate engines: %d regular, %d speculative", acquired, leases.freeCalls())
			}
			if _, timing, err := h.EnsureMeasured(ctx, "tiny:1b"); err != nil || timing.State != "loaded" {
				t.Fatalf("subsequent resident wait: %v, timing %+v", err, timing)
			}
		})
	}
}

// Gate the completed holder call before the shared load publishes its result.
// The test uses the log as a barrier and reads timing from EnsureMeasured.
type timingResultGate struct {
	slog.Handler
	blocked atomic.Bool
	entered chan struct{}
	resume  chan struct{}
}

func (g *timingResultGate) Handle(_ context.Context, record slog.Record) error {
	record.Attrs(func(attr slog.Attr) bool {
		if attr.Key == "phase" && attr.Value.String() == "ready" && g.blocked.CompareAndSwap(false, true) {
			close(g.entered)
			<-g.resume
			return false
		}
		return true
	})
	return nil
}

func TestEnsureMeasuredKeepsAttemptTimingAcrossUnload(t *testing.T) {
	requireEnginePeer(t)
	for _, replace := range []bool{false, true} {
		t.Run(map[bool]string{false: "unload", true: "replacement"}[replace], func(t *testing.T) {
			book := &freeBook{fakeLeases: newFakeLeases()}
			h := freeHost(t, book, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			want := "cold"
			if replace {
				if _, _, err := h.EnsureMeasured(ctx, "tiny:1b"); err != nil {
					t.Fatal(err)
				}
				h.loadWG.Wait()
				want = "loaded"
			}
			gate := &timingResultGate{Handler: slog.NewTextHandler(io.Discard, nil),
				entered: make(chan struct{}), resume: make(chan struct{})}
			h.loadLogger = slog.New(gate)
			release := sync.OnceFunc(func() { close(gate.resume) })
			defer release()
			type result struct {
				timing LoadTiming
				err    error
			}
			done := make(chan result, 1)
			go func() {
				_, timing, err := h.EnsureMeasured(ctx, "tiny:1b")
				done <- result{timing, err}
			}()
			select {
			case <-gate.entered:
			case <-ctx.Done():
				t.Fatal("holder never finished ensuring the original engine")
			}
			h.Unload("tiny:1b", "test unload before timing is read")
			if h.Loaded("tiny:1b") {
				t.Fatal("original engine remained resident")
			}
			if replace {
				// The foreground result is still waiting at the barrier. The
				// speculative slot can start its own exact replacement.
				if _, timing, err := h.EnsureSpeculativeMeasured(ctx, "tiny:1b"); err != nil || timing.State != "cold" {
					t.Fatalf("replacement load: %v, timing %+v", err, timing)
				}
			}
			release()
			select {
			case got := <-done:
				if got.err != nil || got.timing.State != want {
					t.Fatalf("original result: %v, timing %+v, want %s", got.err, got.timing, want)
				}
			case <-ctx.Done():
				t.Fatal("original result did not finish")
			}
		})
	}
}
