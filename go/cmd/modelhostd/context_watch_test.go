package main

import (
	"context"
	"errors"
	"net"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	configwire "github.com/openabstractions/abstraction-config/go/abstraction/config"
	configclient "github.com/openabstractions/abstraction-config/go/client"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
)

// slowObserver answers each Observe after delay with the next scripted
// observation, then holds until the test ends.
type slowObserver struct {
	delay   time.Duration
	mu      sync.Mutex
	answers []configwire.ConfigObservationV2
	stop    chan struct{}
}

func (s *slowObserver) Observe(string, int64) (configwire.ConfigObservationV2, error) {
	s.mu.Lock()
	if len(s.answers) == 0 {
		s.mu.Unlock()
		<-s.stop
		return configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Unchanged}, nil
	}
	next := s.answers[0]
	s.answers = s.answers[1:]
	s.mu.Unlock()
	select {
	case <-time.After(s.delay):
	case <-s.stop:
	}
	return next, nil
}

func serveObserver(t *testing.T, handler configwire.ConfigObserverV2) *configclient.Client {
	t.Helper()
	endpoint := explicitFixtureEndpoint(t, "config")
	listener, err := listen.ListenFramed(endpoint, listen.Program)
	dispatch := func(frame []byte) ([]byte, error) {
		return configwire.ServeEndpoint(frame, "fixture-config", "", &configwire.ConfigObserverV2Dispatcher{Handler: handler})
	}
	if runtime.GOOS == "darwin" {
		if listener != nil {
			listener.Close()
		}
		if listener != nil || !errors.Is(err, identity.ErrNotProven) {
			t.Fatal("Darwin Unix socket did not refuse Program proof", err)
		}
		// Exercise the actual Config client budget and wire dispatcher over
		// an in-process fixture stream. Native admission above retains its
		// production proof floor.
		var calls sync.WaitGroup
		t.Cleanup(calls.Wait)
		return configclient.NewWithTransport(listen.FrameClient{Dialer: func(context.Context, string) (net.Conn, error) {
			client, server := net.Pipe()
			calls.Add(1)
			go func() {
				defer calls.Done()
				defer server.Close()
				frame, err := listen.ReadFrameFrom(server, 1<<20)
				if err != nil {
					return
				}
				if reply, err := dispatch(frame); err == nil {
					_ = listen.WriteFrameTo(server, reply, 1<<20)
				}
			}()
			return client, nil
		}})
	}
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	var calls sync.WaitGroup
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			calls.Add(1)
			go func() {
				defer calls.Done()
				defer conn.Close()
				// The config service gives an observation its own longer call budget.
				callCtx, cancelCall := context.WithTimeout(ctx, 30*time.Second)
				defer cancelCall()
				call, err := listen.ReceiveFramed(callCtx, conn, listen.Program, 1<<20)
				if err != nil {
					t.Log("receive:", err)
					return
				}
				defer call.Close()
				reply, err := dispatch(call.Frame)
				if err == nil {
					err = call.Reply(reply)
				}
				if err != nil && ctx.Err() == nil {
					t.Log(err)
				}
			}()
		}
	}()
	t.Cleanup(func() { cancel(); listener.Close(); <-done; calls.Wait() })
	return configclient.New(endpoint)
}

// The watch's client waits out an observation longer than the config
// client's two-second default, and the change it reports reaches the model
// owner's reload. Before 2026-11-10 the watch used that default, every
// observation ended in "context deadline exceeded", and no change arrived.
func TestContextWatchDeliversAChangeSlowerThanTheClientDefault(t *testing.T) {
	saved := contextWatchWait
	contextWatchWait = 4 * time.Second
	t.Cleanup(func() { contextWatchWait = saved })
	delay := configclient.DefaultTimeout + 500*time.Millisecond
	snapshot := func(values map[string]string) *configwire.ReadResult { r := contextRead(values); return &r }
	observer := &slowObserver{delay: delay, stop: make(chan struct{}), answers: []configwire.ConfigObservationV2{
		{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "1", Snapshot: snapshot(map[string]string{})},
		{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "2", Snapshot: snapshot(map[string]string{"qwen3.6-35b-a3b": "131072"})},
	}}
	config := serveObserver(t, observer)
	t.Cleanup(func() { close(observer.stop) })

	// The plain client modelhostd discovers derives its observation limit
	// from the wait (config CFG-W12).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	changed := make(chan string, 4)
	var logged []error
	var mu sync.Mutex
	done := make(chan struct{})
	go func() {
		defer close(done)
		watchModelContext(ctx, config, func(family string) { changed <- family },
			func(err error) { mu.Lock(); logged = append(logged, err); mu.Unlock() })
	}()
	select {
	case family := <-changed:
		if family != "qwen3.6-35b-a3b" {
			t.Fatalf("changed %q", family)
		}
	case <-time.After(20 * time.Second):
		cancel()
		<-done
		mu.Lock()
		defer mu.Unlock()
		t.Fatalf("the context change never reached the model owner; logged %v", logged)
	}
	cancel()
	<-done
	mu.Lock()
	defer mu.Unlock()
	if len(logged) != 0 {
		t.Fatalf("watch logged %v", logged)
	}
	// The next load reads the person's new choice.
	if got := contextChoice(fakeConfig{result: contextRead(map[string]string{"qwen3.6-35b-a3b": "131072"})}, func(err error) { t.Error(err) })(context.Background(), "qwen3.6-35b-a3b"); got != 131072 {
		t.Fatalf("next load reads %d", got)
	}
}

// failingObserver fails every Observe with the same error.
type failingObserver struct {
	mu    sync.Mutex
	calls int
}

func (f *failingObserver) ObserveTypedContext(context.Context, string, int64) (configwire.ConfigObservationV2, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	return configwire.ConfigObservationV2{}, context.DeadlineExceeded
}

// A watch that keeps failing pauses longer each time up to its bound, and
// logs the repeated failure once instead of every half minute.
func TestContextWatchBacksOffAndLogsARepeatedFailureOnce(t *testing.T) {
	savedPause, savedBackoff, savedMax := contextWatchPause, contextWatchBackoff, contextWatchBackoffMax
	t.Cleanup(func() {
		contextWatchPause, contextWatchBackoff, contextWatchBackoffMax = savedPause, savedBackoff, savedMax
	})
	contextWatchBackoff, contextWatchBackoffMax = 30*time.Second, 10*time.Minute
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var pauses []time.Duration
	contextWatchPause = func(_ context.Context, d time.Duration) bool {
		pauses = append(pauses, d)
		if len(pauses) == 12 {
			cancel()
			return false
		}
		return true
	}
	observer := &failingObserver{}
	var logged []string
	watchModelContext(ctx, observer, func(string) { t.Error("a failure reported a change") }, func(err error) { logged = append(logged, err.Error()) })
	if observer.calls != 12 {
		t.Fatalf("observed %d times for 12 pauses", observer.calls)
	}
	want := []time.Duration{30 * time.Second, time.Minute, 2 * time.Minute, 4 * time.Minute, 8 * time.Minute}
	for len(want) < 12 {
		want = append(want, 10*time.Minute)
	}
	if !slices.Equal(pauses, want) {
		t.Fatalf("pauses %v, want %v", pauses, want)
	}
	if len(logged) != 1 || !strings.Contains(logged[0], "context deadline exceeded") {
		t.Fatalf("logged %q, want the failure once", logged)
	}
}

// After failures, a good answer is logged once and resets the pause.
func TestContextWatchResumesAfterFailures(t *testing.T) {
	savedPause := contextWatchPause
	t.Cleanup(func() { contextWatchPause = savedPause })
	var pauses []time.Duration
	contextWatchPause = func(_ context.Context, d time.Duration) bool { pauses = append(pauses, d); return true }
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	script := []error{context.DeadlineExceeded, context.DeadlineExceeded, nil, context.DeadlineExceeded}
	observer := observerFunc(func(context.Context, string, int64) (configwire.ConfigObservationV2, error) {
		if len(script) == 0 {
			cancel()
			return configwire.ConfigObservationV2{}, context.Canceled
		}
		err := script[0]
		script = script[1:]
		if err != nil {
			return configwire.ConfigObservationV2{}, err
		}
		return configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Unchanged, Cursor: "1"}, nil
	})
	var logged []string
	watchModelContext(ctx, observer, func(string) {}, func(err error) { logged = append(logged, err.Error()) })
	if want := []time.Duration{contextWatchBackoff, 2 * contextWatchBackoff, contextWatchBackoff}; !slices.Equal(pauses, want) {
		t.Fatalf("pauses %v, want %v", pauses, want)
	}
	if len(logged) != 3 || !strings.Contains(logged[1], "resumed after 2") {
		t.Fatalf("logged %q", logged)
	}
}

type observerFunc func(context.Context, string, int64) (configwire.ConfigObservationV2, error)

func (f observerFunc) ObserveTypedContext(ctx context.Context, cursor string, wait int64) (configwire.ConfigObservationV2, error) {
	return f(ctx, cursor, wait)
}

// An unchanged answer after a refused snapshot is the refusal standing: the
// watch logs no recovery until a snapshot reads model_context. On the
// owner's machine the log said the watch resumed half a minute after each
// refusal while the key stayed refused, and the refusal read as transient.
func TestContextWatchRefusalStandsThroughUnchanged(t *testing.T) {
	savedPause, savedWait := contextWatchPause, contextWatchWait
	t.Cleanup(func() { contextWatchPause, contextWatchWait = savedPause, savedWait })
	contextWatchPause = func(context.Context, time.Duration) bool { return true }
	contextWatchWait = time.Second
	refused := configwire.ReadResult{Outcome: configwire.ReadOutcomeRead, Entries: []configwire.Entry{},
		Refused: []configwire.KeyRefusal{{Key: modelContextKey, Reason: configwire.KeyRefusalReasonNotGranted}}}
	granted := contextRead(map[string]string{"qwen3.6-35b-a3b": "131072"})
	run := func(answers ...configwire.ConfigObservationV2) []string {
		observer := &slowObserver{stop: make(chan struct{}), answers: answers}
		config := serveObserver(t, observer)
		ctx, cancel := context.WithCancel(context.Background())
		var mu sync.Mutex
		var logged []string
		done := make(chan struct{})
		go func() {
			defer close(done)
			watchModelContext(ctx, config, func(string) {},
				func(err error) { mu.Lock(); logged = append(logged, err.Error()); mu.Unlock() })
		}()
		deadline := time.Now().Add(20 * time.Second)
		for {
			observer.mu.Lock()
			left := len(observer.answers)
			observer.mu.Unlock()
			if left == 0 || time.Now().After(deadline) {
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		// The last scripted answer has been taken; let the watch act on it.
		time.Sleep(300 * time.Millisecond)
		cancel()
		close(observer.stop)
		<-done
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(logged)
	}
	logged := run(
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "1", Snapshot: &refused},
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Unchanged, Cursor: "1"},
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Unchanged, Cursor: "1"})
	if len(logged) != 1 || !strings.Contains(logged[0], "not_granted") {
		t.Fatalf("logged %q, want the refusal alone while it stands", logged)
	}
	logged = run(
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "1", Snapshot: &refused},
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Unchanged, Cursor: "1"},
		configwire.ConfigObservationV2{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "2", Snapshot: &granted})
	if len(logged) != 2 || logged[1] != "model_context watch resumed after 1 failed attempts" {
		t.Fatalf("logged %q, want the refusal, then one recovery once the key reads", logged)
	}
}
