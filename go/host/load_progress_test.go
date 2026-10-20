package host

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"
)

func TestLoadProgressKeepsGenuineColdLoadAlive(t *testing.T) {
	requireEnginePeer(t)
	// The stand-in emits four phases a second apart. Keep a full second
	// of margin for Windows startup and polling while the total load still
	// exceeds the stall budget.
	const stallBudget = 2 * time.Second
	for _, phases := range []bool{true, false} {
		t.Run(map[bool]string{true: "genuine_phases", false: "health_heartbeat_only"}[phases], func(t *testing.T) {
			h := composed(t, newFakeLeases(), func(o *Options) {
				o.Wait = 50 * time.Millisecond
				o.Engine.Ready = stallBudget
				o.Engine.fixture = []string{"--fake-load", "4s"}
				if phases {
					o.Engine.fixture = append(o.Engine.fixture, "--fake-load-phases")
				}
			})
			_, err := h.Ensure(context.Background(), "tiny:1b")
			if phases && err != nil {
				t.Fatal(err)
			}
			if !phases && (err == nil || !strings.Contains(err.Error(), "engine readiness stalled for "+stallBudget.String())) {
				t.Fatalf("unchanged loading health should hit the stall budget: %v", err)
			}
		})
	}
}

func TestLoadProgressPinnedPhasesOnly(t *testing.T) {
	p := &loadProgress{out: io.Discard}
	for _, line := range []string{"health 503 Loading model", "heartbeat", "load_tensors: layer 0 assigned to device CPU"} {
		p.Write([]byte(line + "\n"))
	}
	if p.count() != 0 {
		t.Fatal("heartbeat counted as progress")
	}
	markers := []string{"llama_model_loader: loaded meta data with 1 key-value pairs", "load_tensors: loading model tensors, this can take a while...", "llama_context: constructing llama_context", "sched_reserve: graph nodes  = 10"}
	for i, line := range markers {
		for range 2 {
			p.Write([]byte(line[:5]))
			p.Write([]byte(line[5:] + "\n"))
		}
		if p.count() != uint64(i+1) {
			t.Fatalf("phase %d: %d", i, p.count())
		}
	}
	p.Write([]byte(markers[0] + "\n"))
	if p.count() != 4 {
		t.Fatal("old phase changed progress")
	}
	p.Write([]byte("unknown_load_phase: complete\n"))
	if p.count() != 4 {
		t.Fatal("unknown phase extended completed phase progress")
	}
	outOfOrder := &loadProgress{out: io.Discard}
	outOfOrder.Write([]byte(markers[2] + "\n"))
	outOfOrder.Write([]byte(markers[1] + "\n"))
	if outOfOrder.count() != 3 {
		t.Fatal("an earlier phase's first occurrence changed progress")
	}
	q := &loadProgress{out: io.Discard}
	q.Write([]byte(strings.Repeat("x", 4097) + markers[0] + "\n"))
	if q.count() != 0 {
		t.Fatal("overlong line counted")
	}
}

func TestLoadProgressFourLogLinesKeepEveryWatchdogAdvance(t *testing.T) {
	var phases []uint64
	p := &loadProgress{publish: func(phase uint64) { phases = append(phases, phase) }}
	p.Write([]byte("llama_model_loader: loaded meta data with 1 key-value pairs\n" +
		"load_tensors: loading model tensors, this can take a while...\n" +
		"llama_context: constructing llama_context\n" +
		"sched_reserve: graph nodes  = 10\n"))
	if p.count() != 4 || len(phases) != 4 {
		t.Fatalf("watchdog phases: %v count=%d", phases, p.count())
	}
	for i, phase := range phases {
		if phase != uint64(i+1) {
			t.Fatal(phases)
		}
	}
}

func TestLoadProgressOutOfOrderTraceAdvancesExactlyOnce(t *testing.T) {
	p := &loadProgress{out: io.Discard}
	lines := []string{
		"sched_reserve: graph nodes  = 10", // phase 4 first
		"llama_model_loader: loaded meta data with 1 key-value pairs",
		"load_tensors: loading model tensors, this can take a while...",
		"llama_context: constructing llama_context",
		"sched_reserve: graph nodes  = 10",
		"unknown_load_phase: complete",
		"heartbeat",
	}
	var last uint64
	advances := 0
	for _, line := range lines {
		p.Write([]byte(line + "\n"))
		current := p.count()
		if current > last {
			advances++
		}
		if current != 4 {
			t.Fatalf("line %q: progress=%d", line, current)
		}
		last = current
	}
	if advances != 1 {
		t.Fatalf("advances=%d", advances)
	}
}

type blockedLoadLog struct{ entered, release chan struct{} }

func (b blockedLoadLog) Write(p []byte) (int, error) {
	close(b.entered)
	<-b.release
	return len(p), nil
}

func TestLoadProgressReadDoesNotWaitForLogOutput(t *testing.T) {
	log := blockedLoadLog{entered: make(chan struct{}), release: make(chan struct{})}
	p := &loadProgress{out: log}
	done := make(chan struct{})
	go func() { p.Write([]byte("llama_context: constructing llama_context\n")); close(done) }()
	<-log.entered
	defer func() { close(log.release); <-done }()
	count := make(chan uint64, 1)
	go func() { count <- p.count() }()
	select {
	case got := <-count:
		if got != 3 {
			t.Fatal(got)
		}
	case <-time.After(time.Second):
		t.Fatal("log output blocked the readiness progress read")
	}
}

func TestLoadProgressLiveSubscriptions(t *testing.T) {
	hub := &sharedLoadProgress{}
	first, leave := hub.subscribe()
	p := &loadProgress{publish: hub.publish}
	p.Write([]byte("llama_model_loader: loaded meta data with 1 key-value pairs\n"))
	if got := <-first; got != 1 {
		t.Fatal(got)
	}
	late, leaveLate := hub.subscribe()
	defer leaveLate()
	if got := <-late; loadPhaseName(got) != "metadata" {
		t.Fatal(got)
	}
	leave()
	p.Write([]byte("llama_context: constructing llama_context\n"))
	p.Write([]byte("heartbeat\nload_tensors: loading model tensors, this can take a while...\n"))
	if got := <-late; got != 3 {
		t.Fatal(got)
	}
	select {
	case got := <-first:
		t.Fatalf("dead waiter retained: %d", got)
	default:
	}
	select {
	case got := <-late:
		t.Fatalf("heartbeat/old phase: %d", got)
	default:
	}
	// A subscriber may reenter registration and publication outside hub locks.
	other, leaveOther := hub.subscribe()
	if got := <-other; got != 3 {
		t.Fatal(got)
	}
	hub.publish(4)
	leaveOther()
	if got := <-late; loadPhaseName(got) != "graph" {
		t.Fatal(got)
	}
	leaveLate()
	if len(hub.waiters) != 0 {
		t.Fatal("subscriptions retained")
	}
}

func TestLoadProgressPublisherDoesNotWaitForSubscriber(t *testing.T) {
	hub := &sharedLoadProgress{}
	_, leave := hub.subscribe()
	defer leave()
	done := make(chan struct{})
	go func() {
		for phase := uint64(1); phase <= 4; phase++ {
			hub.publish(phase)
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("publisher waited for callback")
	}
}

func TestLoadProgressNewEngineBeginsWithoutStalePhase(t *testing.T) {
	hub := &sharedLoadProgress{}
	ch, leave := hub.subscribe()
	defer leave()
	hub.publish(4)
	hub.begin()
	late, leaveLate := hub.subscribe()
	defer leaveLate()
	select {
	case phase := <-late:
		t.Fatalf("stale phase: %d", phase)
	default:
	}
	parser := &loadProgress{publish: hub.publish}
	parser.Write([]byte("llama_model_loader: loaded meta data with 1 key-value pairs\n"))
	if phase := <-ch; phase != 1 {
		t.Fatal(phase)
	}
	if phase := <-late; phase != 1 {
		t.Fatal(phase)
	}
}
