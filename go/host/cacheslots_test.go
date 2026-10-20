package host

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestCacheSlotsKeepRefillCostAcrossWarmLongHits(t *testing.T) {
	s := newCacheSlots(2)
	now := time.Unix(1, 0)
	refill := func(scope string, duration time.Duration, miss bool) *cacheSlotLease {
		t.Helper()
		lease, err := s.reserve(t.Context(), scope)
		if err != nil {
			t.Fatal(err)
		}
		start := now
		now = now.Add(duration)
		lease.observe(now.Sub(start), miss)
		lease.release(true)
		return lease
	}
	main := refill("main", 40*time.Second, true)
	title := refill("title", 2*time.Second, true)
	if main.id == title.id {
		t.Fatal("title displaced main while another slot was empty")
	}
	warm := refill("main", time.Millisecond, false)
	if !warm.reused || warm.id != main.id || s.slots[main.id].cost != 40*time.Second {
		t.Fatal("cheap warm hit reset main refill cost", s.slots)
	}
	subagent := refill("subagent", time.Second, true)
	if subagent.id != title.id || s.inflation != float64(2*time.Second) {
		t.Fatal("lowest keep-value victim/inflation", s.slots, s.inflation)
	}
	if s.slots[main.id].scope != "main" {
		t.Fatal("long main lost", s.slots)
	}
}

func TestCacheSlotsUnlabelledProtectsLabelledAndUnknownCost(t *testing.T) {
	s := newCacheSlots(2)
	main, _ := s.reserve(t.Context(), "main")
	main.release(true) // Missing measurement remains unknown, not invented.
	other, _ := s.reserve(t.Context(), "")
	other.observe(10*time.Second, true)
	other.release(true)
	again, _ := s.reserve(t.Context(), "")
	if again.id != other.id || s.slots[main.id].scope != "main" || s.slots[main.id].costKnown {
		t.Fatal(s.slots)
	}
	again.release(false)
	later, _ := s.reserve(t.Context(), "replacement")
	again.observe(time.Hour, true) // A released response cannot change a new assignment.
	if s.slots[later.id].costKnown {
		t.Fatal("late response observation changed new assignment")
	}
	later.release(true)
}

func TestCacheSlotsProgramPartitionControlsPrefixReuse(t *testing.T) {
	h := &Host{engines: map[string]*Process{"engine": {cacheSlots: newCacheSlots(2)}}}
	raw := []byte(`{"messages":[{"role":"user","content":"shared prefix"}],"cache_prompt":true}`)
	request := func(scope, partition string, complete bool) map[string]any {
		t.Helper()
		out, release, err := h.prepareScopedCacheRequest(t.Context(), "engine", "/v1/chat/completions", raw, scope, partition)
		if err != nil {
			t.Fatal(err)
		}
		var body map[string]any
		if err := json.Unmarshal(out, &body); err != nil {
			t.Fatal(err)
		}
		release(complete)
		return body
	}
	for _, scope := range []string{"chat-a", "chat-b"} {
		if body := request(scope, "program-a", true); body["cache_prompt"] != true {
			t.Fatal(body)
		}
	}
	if body := request("chat-c", "program-b", true); body["cache_prompt"] != false {
		t.Fatal("cross-program shared prefix was reusable", body)
	}
	if body := request("", "", true); body["cache_prompt"] != false {
		t.Fatal("missing evidence reused prefix", body)
	}
	request("failed", "program-a", false)
	if body := request("failed", "program-a", true); body["cache_prompt"] != false {
		t.Fatal("failed assignment supplied program evidence", body)
	}
}

func TestCacheSlotsAffinityReplacementAndCancellation(t *testing.T) {
	s := newCacheSlots(2)
	a, reused, release, err := s.acquire(context.Background(), "account/app/chat-a")
	if err != nil || reused {
		t.Fatalf("first: %d %v %v", a, reused, err)
	}
	release(true)
	b, reused, release, err := s.acquire(context.Background(), "account/other-app/chat-a")
	if err != nil || reused || a == b {
		t.Fatalf("cross app: %d %v %v", b, reused, err)
	}
	release(true)
	again, reused, release, err := s.acquire(context.Background(), "account/app/chat-a")
	if err != nil || !reused || again != a {
		t.Fatalf("same T3: %d %v %v", again, reused, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := s.acquire(ctx, "account/app/chat-a"); !errors.Is(err, context.Canceled) {
		t.Errorf("busy cancellation: %v", err)
	}
	release(true)
	_, _, release, _ = s.acquire(context.Background(), "replacement")
	release(false)
	_, reused, release, _ = s.acquire(context.Background(), "replacement")
	if reused {
		t.Error("failed request retained affinity")
	}
	release(true)
	_, reused, release, _ = s.acquire(context.Background(), "")
	if reused {
		t.Error("absent conversation gained affinity")
	}
	release(true)
	_, reused, release, _ = s.acquire(context.Background(), "")
	if reused {
		t.Error("unbound call reused a slot")
	}
	release(true)
}

func TestCacheSlotsResponseOwnsRelease(t *testing.T) {
	s := newCacheSlots(1)
	_, _, release, _ := s.acquire(context.Background(), "bound")
	ctx, cancel := context.WithCancel(context.Background())
	body := &cacheResponseBody{ReadCloser: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), release: release, complete: true, ctx: ctx}
	if !s.slots[0].busy {
		t.Fatal("response did not retain slot")
	}
	cancel()
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	if s.slots[0].busy || s.slots[0].scope != "" {
		t.Fatal("canceled response retained assignment")
	}
	_, reused, release, _ := s.acquire(context.Background(), "bound")
	if reused {
		t.Error("canceled request became a cache hit")
	}
	body = &cacheResponseBody{ReadCloser: io.NopCloser(strings.NewReader("data: [DONE]\n\n")), release: release, complete: true, ctx: context.Background()}
	if _, err := io.ReadAll(body); err != nil {
		t.Fatal(err)
	}
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	_, reused, release, _ = s.acquire(context.Background(), "bound")
	if !reused {
		t.Error("completed request lost affinity")
	}
	release(true)
}

type blockedCacheBody struct {
	started chan struct{}
	closed  chan struct{}
	once    sync.Once
}

func (b *blockedCacheBody) Read([]byte) (int, error) {
	close(b.started)
	<-b.closed
	return 0, io.ErrClosedPipe
}
func (b *blockedCacheBody) Close() error {
	b.once.Do(func() { close(b.closed) })
	return nil
}

func TestCacheSlotsConcurrentReadClose(t *testing.T) {
	s := newCacheSlots(1)
	_, _, release, _ := s.acquire(context.Background(), "bound")
	ctx, cancel := context.WithCancel(context.Background())
	underlying := &blockedCacheBody{started: make(chan struct{}), closed: make(chan struct{})}
	var releases atomic.Int32
	body := &cacheResponseBody{ReadCloser: underlying, complete: true, ctx: ctx, release: func(ok bool) {
		releases.Add(1)
		release(ok)
	}}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = body.Read(make([]byte, 1))
	}()
	<-underlying.started
	cancel()
	if err := body.Close(); err != nil {
		t.Fatal(err)
	}
	<-done
	if releases.Load() != 1 || s.slots[0].busy || s.slots[0].scope != "" {
		t.Fatalf("release=%d slot=%+v", releases.Load(), s.slots[0])
	}
}

func TestCacheSlotsOverrideUntrustedBodyAndLaunchLifetime(t *testing.T) {
	h := &Host{engines: map[string]*Process{"engine": {cacheSlots: newCacheSlots(1)}}}
	raw := []byte(`{"messages":[],"id_slot":999,"cache_prompt":true,"chat_template_kwargs":{"preserve_thinking":false}}`)
	out, release, err := h.prepareCacheRequest(context.Background(), "engine", "/v1/chat/completions", raw)
	if err != nil {
		t.Fatal(err)
	}
	defer release(true)
	var body map[string]any
	if err := json.Unmarshal(out, &body); err != nil {
		t.Fatal(err)
	}
	if body["id_slot"] != float64(0) || body["cache_prompt"] != false || body["chat_template_kwargs"].(map[string]any)["preserve_thinking"] != true {
		t.Fatalf("body=%s", out)
	}
	// An engine replacement starts with no old slot state. A disabled recipe
	// leaves body handling unchanged and never enables cache policy implicitly.
	h.engines["engine"] = &Process{}
	out, cleanup, err := h.prepareCacheRequest(context.Background(), "engine", "/v1/chat/completions", raw)
	if err != nil || cleanup != nil || string(out) != string(raw) {
		t.Fatalf("disabled recipe: %s %v", out, err)
	}
}
