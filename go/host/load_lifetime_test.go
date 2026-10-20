package host

import (
	"context"
	"encoding/json"
	"fmt"
	logging "github.com/openabstractions/abstraction-logging/go"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSharedLoadOutlivesItsWaiter(t *testing.T) {
	requireEnginePeer(t)
	for _, cancelOnly := range []bool{false, true} {
		t.Run(map[bool]string{false: "deadline", true: "cancel"}[cancelOnly], func(t *testing.T) {
			leases := newFakeLeases()
			h := composed(t, leases, func(o *Options) {
				o.Engine.fixture = []string{"--fake-load", "600ms"}
				o.Engine.Ready = 2 * time.Second
			})
			timeout := 100 * time.Millisecond
			if cancelOnly {
				timeout = 5 * time.Second
			}
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			if cancelOnly {
				go func() { time.Sleep(100 * time.Millisecond); cancel() }()
			}
			_, err := h.Ensure(ctx, "tiny:1b")
			if WordOf(err) != "engine:loading" {
				t.Errorf("ended wait: %v; want engine:loading", err)
			}
			if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
				t.Fatal(err)
			}
			acquired, _, _ := leases.counts()
			if acquired != 1 {
				t.Errorf("shared load acquired %d leases, want 1", acquired)
			}
		})
	}
}

func TestSharedLoadProgressWaiterCancelAndLateJoin(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-load", "800ms", "--fake-load-phases"}
		o.Engine.Ready = 2 * time.Second
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var first []string
	var shared *sharedLoadProgress
	ctx = WithLoadProgress(ctx, func(phase string) {
		first = append(first, phase)
		// Reenter both owner locks. Callbacks run on this waiter's goroutine.
		h.loadMu.Lock()
		shared = h.loads["tiny:1b"][0].progress
		h.loadMu.Unlock()
		_, leave := shared.subscribe()
		leave()
		cancel()
	})
	if _, err := h.Ensure(ctx, "tiny:1b"); WordOf(err) != "engine:loading" {
		t.Fatal(err)
	}
	if len(first) != 1 || first[0] != "metadata" {
		t.Fatalf("first: %v", first)
	}
	shared.mu.Lock()
	retained := len(shared.waiters)
	shared.mu.Unlock()
	if retained != 0 {
		t.Fatalf("cancelled subscription retained: %d", retained)
	}
	var late []string
	if _, err := h.Ensure(WithLoadProgress(context.Background(), func(phase string) {
		late = append(late, phase)
	}), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if len(late) == 0 || late[0] != "metadata" || late[len(late)-1] != "graph" {
		t.Fatalf("late phases: %v", late)
	}
	shared.mu.Lock()
	retained = len(shared.waiters)
	shared.mu.Unlock()
	if retained != 0 {
		t.Fatalf("finished subscription retained: %d", retained)
	}
	if len(first) != 1 {
		t.Fatalf("cancelled callback invoked: %v", first)
	}
	if acquired, _, _ := leases.counts(); acquired != 1 {
		t.Fatalf("load restarted: %d", acquired)
	}
}

func TestSharedLoadProgressFailureUnregistersWaiter(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.fixture = []string{"--fake-fail-after", "300ms", "--fake-fail-line", "failed to allocate model"}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var shared *sharedLoadProgress
	// Capture the actual subscription through the existing load owner while
	// the fixture is alive; its error must unregister before Ensure returns.
	done := make(chan error, 1)
	go func() { _, err := h.Ensure(WithLoadProgress(ctx, func(string) {}), "tiny:1b"); done <- err }()
	deadline := time.After(time.Second)
	for shared == nil {
		h.loadMu.Lock()
		if load := h.loads["tiny:1b"][0]; load != nil {
			shared = load.progress
		}
		h.loadMu.Unlock()
		if shared == nil {
			select {
			case <-deadline:
				t.Fatal("load never registered")
			case <-time.After(time.Millisecond):
			}
		}
	}
	if err := <-done; err == nil {
		t.Fatal("load failure accepted")
	}
	shared.mu.Lock()
	retained := len(shared.waiters)
	shared.mu.Unlock()
	if retained != 0 {
		t.Fatalf("failed waiter retained: %d", retained)
	}
}

func TestSharedLoadRetainsFailureAfterEveryWaiterLeaves(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-fail-after", "300ms", "--fake-fail-line", "failed to allocate model"}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := h.Ensure(ctx, "tiny:1b"); WordOf(err) != "engine:loading" {
		t.Fatal(err)
	}
	h.loadWG.Wait()
	for _, ensure := range []func(context.Context, string) (string, error){h.Ensure, h.EnsureSpeculative} {
		if _, err := ensure(context.Background(), "tiny:1b"); WordOf(err) != EngineDoesNotFit {
			t.Errorf("retained failure: %v", err)
		}
	}
	acquired, _, _ := leases.counts()
	if acquired != 1 {
		t.Fatalf("failed model acquired %d leases, want 1", acquired)
	}
	h.options.Now = func() time.Time { return time.Now().Add(loadFailureRetry + time.Second) }
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != EngineDoesNotFit {
		t.Fatal(err)
	}
	if acquired, _, _ := leases.counts(); acquired != 2 {
		t.Fatalf("bounded retry acquired %d leases", acquired)
	}
}

func TestSharedLoadRetriesARejectedLeaseWithoutCooldown(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	leases.outcome = res.AcquireOutcomeNotPermitted
	h := composed(t, leases, nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != NotPermitted {
		t.Fatalf("rejected lease: %v", err)
	}
	leases.mu.Lock()
	leases.outcome = res.AcquireOutcomeAcquired
	leases.mu.Unlock()
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("admitted retry: %v", err)
	}
	if acquired, _, _ := leases.counts(); acquired != 2 || !h.Loaded("tiny:1b") {
		t.Fatalf("acquires %d, loaded %v", acquired, h.Loaded("tiny:1b"))
	}
}

type loadRecords struct {
	mu      sync.Mutex
	records []logging.Record
}

func (s *loadRecords) Write(r logging.Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.records = append(s.records, r)
	return nil
}

func TestSharedLoadReportsStructuredStartAndOutcome(t *testing.T) {
	requireEnginePeer(t)
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			h := composed(t, newFakeLeases(), func(o *Options) {
				if fail {
					o.Engine.fixture = []string{"--fake-fail-line", "failed to allocate model"}
				}
			})
			sink := &loadRecords{}
			h.loadLogger = slog.New(logging.NewHandler(sink, &logging.Options{Program: "modelhostd"}))
			_, err := h.Ensure(context.Background(), "tiny:1b")
			if !fail && err != nil {
				t.Fatal(err)
			}
			h.loadWG.Wait()
			if len(sink.records) != 2 {
				t.Fatalf("records %v", sink.records)
			}
			outcome, word := "ready", "ready"
			if fail {
				outcome = "failed"
				word = EngineDoesNotFit
			}
			for i, phase := range []string{"start", outcome} {
				r := sink.records[i]
				if r.Attrs["family"] != "tiny:1b" || r.Attrs["phase"] != phase || r.Attrs["duration_ms"] == "" {
					t.Fatalf("record %+v", r)
				}
				if i == 1 && r.Attrs["result"] != word {
					t.Fatalf("result %+v", r)
				}
				body, _ := json.Marshal(r)
				for key, value := range r.Attrs {
					if key == "args" || strings.Contains(value, h.options.Engine.Program) {
						t.Fatalf("private path/arguments in attribute %s=%q", key, value)
					}
				}
				if strings.Contains(string(body), "failed to allocate model") {
					t.Fatalf("private detail in structured record %s", body)
				}
			}
		})
	}
}

func TestSharedLoadRetainsBoundedEngineOutputAndPhases(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine.fixture = []string{"--fake-load", "100ms", "--fake-output", "load probe"} })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	key, ok := h.EngineKey(h.hosts["tiny:1b"])
	if !ok {
		t.Fatal("no launch key")
	}
	h.Unload("tiny:1b", "test")
	logs, err := filepath.Glob(filepath.Join(filepath.Dir(h.options.Engine.Record), "logs", "llama-*.log"))
	if err != nil || len(logs) != 1 {
		t.Fatalf("logs %v %v", logs, err)
	}
	body, err := os.ReadFile(logs[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, word := range []string{"phase=launch", "prepare_ms=", "release=", "args=", "phase=readiness", "elapsed_ms=", "stdout: load probe", "stderr: load probe"} {
		if !strings.Contains(string(body), word) {
			t.Errorf("log lacks %q: %s", word, body)
		}
	}
	if strings.Contains(string(body), key) {
		t.Fatal("engine log disclosed launch key")
	}
	if len(body) > engineLogBytes {
		t.Fatal("engine log exceeds existing bound")
	}
}

func TestSharedLoadStillHonorsLeaseRevocation(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-load", "600ms"}
		o.Engine.Ready = 2 * time.Second
	})
	run, cancelRun := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); h.Run(run) }()
	defer func() { cancelRun(); <-done }()
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := h.Ensure(ctx, "tiny:1b"); WordOf(err) != "engine:loading" {
		t.Fatal(err)
	}
	leases.mu.Lock()
	for id := range leases.live {
		leases.live[id] = false
	}
	leases.outcome = res.AcquireOutcomeNotPermitted
	leases.mu.Unlock()
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != NotPermitted {
		t.Fatalf("revoked lease: %v", err)
	}
	if h.Loaded("tiny:1b") {
		t.Fatal("revoked load remained resident")
	}
}

func TestSharedLoadSpeculativeAndNormalWaitersShareOneEngine(t *testing.T) {
	requireEnginePeer(t)
	book := &freeBook{fakeLeases: newFakeLeases()}
	h := freeHost(t, book, func(o *Options) {
		o.Engine.fixture = []string{"--fake-load", "600ms"}
		o.Engine.Ready = 2 * time.Second
	})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := h.EnsureSpeculative(ctx, "tiny:1b"); WordOf(err) != "engine:loading" {
		t.Fatalf("preload wait: %v", err)
	}
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	acquired, _, _ := book.counts()
	if acquired+book.freeCalls() != 1 {
		t.Fatalf("normal %d + speculative %d acquisitions", acquired, book.freeCalls())
	}
}

func TestSharedLoadShutdownJoinsAndReleases(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine.fixture = []string{"--fake-load", "30s"}; o.Engine.Ready = 2 * time.Second })
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := h.Ensure(ctx, "tiny:1b"); WordOf(err) != "engine:loading" {
		t.Fatal(err)
	}
	h.Close()
	if h.Loaded("tiny:1b") {
		t.Fatal("shutdown left model resident")
	}
	acquired, released, _ := leases.counts()
	if acquired != 1 || len(released) != 1 {
		t.Fatalf("shutdown acquired %d released %v", acquired, released)
	}
}
