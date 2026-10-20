package host

import (
	"context"
	"testing"
	"time"
)

// Describe states the loaded contexts while a load is still running. On R28
// a 32 s Qwen3.6 load held its component's lock, Describe waited on it past
// the runtime's two-second readiness probe, the runtime read modelhost
// unreachable, and OpenCode's first chat was refused as activating.
func TestLoadedContextsAnswersWhileAModelLoads(t *testing.T) {
	h, _, _ := sizedHost(t, 1<<40, "--fake-load", "3s")
	done := make(chan error, 1)
	go func() { _, err := h.Ensure(context.Background(), "tiny:1b"); done <- err }()
	for deadline := time.Now().Add(10 * time.Second); ; {
		h.context.mu.Lock()
		_, started := h.context.loaded["tiny:1b"]
		h.context.mu.Unlock()
		if started {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the load never started")
		}
		time.Sleep(10 * time.Millisecond)
	}
	read := make(chan map[string]int64, 1)
	go func() { read <- h.LoadedContexts() }()
	select {
	case got := <-read:
		if len(got) != 0 {
			t.Fatalf("a model still loading reported loaded contexts %v", got)
		}
	case <-time.After(time.Second):
		t.Fatal("LoadedContexts waited for the running load")
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if got := h.LoadedContexts()["tiny:1b"]; got != 262144 {
		t.Fatalf("reported context %d after the load, want 262144", got)
	}
}
