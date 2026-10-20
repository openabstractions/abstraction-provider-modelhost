package host

import (
	"context"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestCacheRecipeBounds(t *testing.T) {
	if got := (Engine{}).cacheRecipe().arguments(); !reflect.DeepEqual(got, []string{"--cache-ram", "1024", "--ctx-checkpoints", "2"}) {
		t.Fatalf("unaccounted engine defaults: %v", got)
	}
	// The 512MiB/4 candidate is a source hypothesis, never selected by the
	// production model recipe. 64MiB is the diagnostic per-checkpoint forecast.
	candidate := engineCache{ramMiB: 512, checkpoints: 4, checkpointMiB: 64, slots: 1}
	if err := candidate.checkArguments([]string{"--parallel", "2"}); err == nil {
		t.Error("unforecast slot count admitted")
	}
	if err := candidate.checkArguments([]string{"--parallel", "1"}); err != nil {
		t.Fatal(err)
	}
	if got, err := candidate.leaseBytes(4096); err != nil || got != 4096+768*(1<<20) {
		t.Fatalf("charge=%d err=%v", got, err)
	}
	for _, c := range []engineCache{{ramMiB: -1}, {checkpoints: 4}, {ramMiB: math.MaxInt}, {checkpoints: math.MaxInt, checkpointMiB: math.MaxInt}} {
		if _, err := c.leaseBytes(4096); err == nil {
			t.Errorf("unbounded recipe admitted: %+v", c)
		}
	}
}

func TestFreeCapacityMeasuredLoadLeavesMainLeaseUntouched(t *testing.T) {
	requireEnginePeer(t)
	book := &freeBook{fakeLeases: newFakeLeases(), held: true}
	h := composedTwo(t, book.fakeLeases, func(o *Options) { o.Leases = book })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	mainLease := h.Lease("tiny:1b")
	if _, _, err := h.EnsureSpeculativeMeasured(context.Background(), "small:2b"); WordOf(err) != resourcesHeld || h.Loaded("small:2b") {
		t.Fatalf("held: %v loaded=%v", err, h.Loaded("small:2b"))
	}
	book.setHeld(false)
	if _, _, err := h.EnsureSpeculativeMeasured(context.Background(), "small:2b"); err != nil {
		t.Fatal(err)
	}
	if h.Lease("tiny:1b") != mainLease || !h.Loaded("tiny:1b") || !h.Loaded("small:2b") {
		t.Fatal("title changed main residency")
	}
	if normal, _, _ := book.counts(); normal != 1 || book.freeCalls() != 2 {
		t.Fatalf("normal=%d free=%d", normal, book.freeCalls())
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := h.EnsureSpeculativeMeasured(ctx, "small:2b"); err == nil {
		t.Fatal("canceled load admitted")
	}
	if book.freeCalls() != 2 {
		t.Fatal("canceled call acquired")
	}
	h.Unload("small:2b", "title finished")
	if h.Lease("tiny:1b") != mainLease {
		t.Fatal("title release changed main lease")
	}
}

func TestFreeCapacityCoalescedCanceledWaiterKeepsNoYieldStanding(t *testing.T) {
	requireEnginePeer(t)
	book := &freeBook{fakeLeases: newFakeLeases()}
	h := freeHost(t, book, func(o *Options) { o.Engine.fixture = []string{"--fake-load", "500ms"} })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first, second := make(chan error, 1), make(chan error, 1)
	go func() { _, _, err := h.EnsureSpeculativeMeasured(ctx, "tiny:1b"); first <- err }()
	go func() { _, _, err := h.EnsureSpeculativeMeasured(context.Background(), "tiny:1b"); second <- err }()
	deadline := time.Now().Add(5 * time.Second)
	for {
		h.loadMu.Lock()
		load := h.loads["tiny:1b"][1]
		joined := load != nil && load.waiters.Load() == 2
		h.loadMu.Unlock()
		if joined {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("free-capacity waiters did not coalesce")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	if err := <-first; err == nil {
		t.Fatal("canceled waiter succeeded")
	}
	if err := <-second; err != nil {
		t.Fatal(err)
	}
	if normal, _, _ := book.counts(); normal != 0 || book.freeCalls() != 1 {
		t.Fatalf("normal=%d free=%d", normal, book.freeCalls())
	}
	h.Close()
	book.mu.Lock()
	defer book.mu.Unlock()
	if len(book.live) != 0 {
		t.Fatalf("close retained leases: %+v", book.live)
	}
}

func TestCacheRecipeLeaseAndWeightTiming(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine.cache = &engineCache{ramMiB: 512, checkpoints: 4, checkpointMiB: 64} })
	_, timing, err := h.EnsureMeasured(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	if timing.Bytes == nil || *timing.Bytes != 4096 {
		t.Fatalf("weight timing=%+v", timing)
	}
	leases.mu.Lock()
	if !reflect.DeepEqual(leases.grants, []int64{4096 + 768*(1<<20)}) || !reflect.DeepEqual(leases.resources, []string{DefaultResource}) {
		t.Errorf("grants=%v resources=%v", leases.grants, leases.resources)
	}
	leases.mu.Unlock()
	h.Close()
	_, released, _ := leases.counts()
	if len(released) != 1 {
		t.Fatalf("released=%v", released)
	}
}

func TestCacheRecipeRefusedBeforeAcquireWithoutForecast(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine.cache = &engineCache{checkpoints: 4} })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err == nil {
		t.Fatal("unforecast checkpoints admitted")
	}
	if n, _, _ := leases.counts(); n != 0 {
		t.Fatalf("acquired %d leases", n)
	}
}

func TestCacheRecipeSharesWeightPoolAndRefusesBeforeLaunch(t *testing.T) {
	leases := newFakeLeases()
	leases.capacity = 4096
	h := composed(t, leases, func(o *Options) { o.Engine.cache = &engineCache{ramMiB: 512, checkpoints: 4, checkpointMiB: 64} })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != Insufficient {
		t.Fatalf("cache overhead bypassed pool: %v", err)
	}
	if h.Loaded("tiny:1b") {
		t.Error("refused cache recipe launched")
	}
	leases.mu.Lock()
	defer leases.mu.Unlock()
	if leases.acquires != 1 || len(leases.live) != 0 || len(leases.grants) != 0 {
		t.Fatalf("acquires=%d live=%v grants=%v", leases.acquires, leases.live, leases.grants)
	}
}
