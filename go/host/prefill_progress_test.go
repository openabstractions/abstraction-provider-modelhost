package host

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	router "github.com/openabstractions/abstraction-router/go"
)

type prefillRoundTrip func(*http.Request) (*http.Response, error)

func (f prefillRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func prefillEvent(id string, processed, total, cache int64) string {
	return fmt.Sprintf("data: {\"id\":%q,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null}}],\"prompt_progress\":{\"processed\":%d,\"total\":%d,\"cache\":%d,\"time_ms\":1}}\n\n", id, processed, total, cache)
}

func invalidPrefillEvents() []string {
	valid := prefillEvent("request-a", 2000, 10000, 0)
	return []string{
		valid, prefillEvent("request-a", 1500, 10000, 0), prefillEvent("request-b", 3000, 10000, 0),
		prefillEvent("request-a", 3000, 11000, 0), prefillEvent("request-a", 3000, 10000, 1),
		prefillEvent("request-a", -1, 10000, 0), prefillEvent("request-a", 11000, 10000, 0),
		prefillEvent("request-a", 3000, 10000, 4000),
		string(bytes.ReplaceAll([]byte(prefillEvent("request-a", 3000, 10000, 0)), []byte(`"processed":3000`), []byte(`"processed":"3000"`))),
		string(bytes.ReplaceAll([]byte(prefillEvent("request-a", 3000, 10000, 0)), []byte(`"processed":3000`), []byte(`"processed":3000.5`))),
		string(bytes.ReplaceAll([]byte(prefillEvent("request-a", 3000, 10000, 0)), []byte(`"processed":3000`), []byte(`"processed":3000,"processed":4000`))),
		"data: {\"usage\":{\"completion_tokens\":9000}}\n\n", ": heartbeat\n\n",
		"data: {\"id\":\"request-a\",\"choices\":[],\"prompt_progress\":{\"processed\":3000}}\n\n",
		"data: malformed\n\n",
	}
}

func TestPrefillProgressParserValidatesNativeCounters(t *testing.T) {
	var got [][2]int64
	b := &prefillProgressBody{report: func(p, total int64) bool { got = append(got, [2]int64{p, total}); return true }}
	initial := prefillEvent("request-a", 0, 10000, 0)
	b.observe([]byte(initial[:11]))
	b.observe([]byte(initial[11:]))
	if len(got) != 0 {
		t.Fatal("initial/cache-only event renewed liveness")
	}
	b.observe([]byte(prefillEvent("request-a", 1000, 10000, 0)))
	b.observe([]byte(prefillEvent("request-a", 2000, 10000, 0)))
	for _, event := range invalidPrefillEvents() {
		b.observe([]byte(event))
	}
	if len(got) != 2 {
		t.Fatalf("non-progress authority accepted: %v", got)
	}
	b.observe([]byte(prefillEvent("request-a", 3000, 10000, 0)))
	if len(got) != 3 || got[2] != [2]int64{3000, 10000} {
		t.Fatal("real progress after rejected data was lost")
	}
	b.observe([]byte("data: {\"id\":\"request-a\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n"))
	b.observe([]byte(prefillEvent("request-a", 4000, 10000, 0)))
	if len(got) != 3 {
		t.Fatal("prefill authority survived generated output")
	}
	cacheCalls := 0
	cache := &prefillProgressBody{report: func(p, total int64) bool {
		cacheCalls++
		if p != 500 || total != 1000 {
			t.Error("cached prefix counted as new work")
		}
		return true
	}}
	cache.observe([]byte(prefillEvent("cached", 9000, 10000, 9000)))
	cache.observe([]byte(prefillEvent("cached", 9500, 10000, 9000)))
	if cacheCalls != 1 {
		t.Fatal("cached prefix renewed liveness or real uncached work was lost")
	}
	raw := []byte(prefillEvent("passthrough", 100, 1000, 0) + ": ping\n\ndata: malformed\n\n")
	calls := 0
	stream := &prefillProgressBody{ReadCloser: io.NopCloser(bytes.NewReader(raw)), report: func(int64, int64) bool { calls++; return true }}
	actual, err := io.ReadAll(stream)
	if err != nil || !bytes.Equal(actual, raw) || calls != 1 {
		t.Fatal("progress observation changed the actual response bytes")
	}
	stream.Close()
	stream.observe([]byte(prefillEvent("passthrough", 200, 1000, 0)))
	if calls != 1 {
		t.Fatal("closed response retained liveness authority")
	}
}

type prefillReadAck struct {
	io.ReadCloser
	read chan<- struct{}
}

func (b prefillReadAck) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.read <- struct{}{}
	}
	return n, err
}

func TestPrefillProgressActualProviderKeepsActiveWorkAndStillStalls(t *testing.T) {
	var clockMu sync.Mutex
	now := time.Unix(1000, 0)
	clock := func() time.Time { clockMu.Lock(); defer clockMu.Unlock(); return now }
	advance := func(d time.Duration) { clockMu.Lock(); now = now.Add(d); clockMu.Unlock() }
	reader, writer := io.Pipe()
	defer writer.Close()
	read := make(chan struct{}, 32)
	accepted := make(chan bool, 32)
	requestCtx := make(chan context.Context, 1)
	joined := make(chan struct{})
	h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(), engines: map[string]*Process{"engine": {key: "fixture-key"}}}
	native := prefillRoundTrip(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if json.NewDecoder(r.Body).Decode(&body) != nil || body["return_progress"] != true {
			t.Error("production HTTP owner did not request native progress")
		}
		requestCtx <- r.Context()
		go func() { <-r.Context().Done(); reader.CloseWithError(r.Context().Err()); close(joined) }()
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: reader}, nil
	})
	client := engineClient(h, nil, native)
	ownedTransport := client.Transport
	client.Transport = prefillRoundTrip(func(r *http.Request) (*http.Response, error) {
		response, err := ownedTransport.RoundTrip(r)
		if err != nil {
			return response, err
		}
		body, ok := response.Body.(*prefillProgressBody)
		if !ok {
			t.Error("native progress parser missing")
			return response, nil
		}
		report := body.report
		body.report = func(p, total int64) bool { ok := report(p, total); accepted <- ok; return ok }
		response.Body = prefillReadAck{ReadCloser: body, read: read}
		return response, nil
	})
	engine := router.OpenAICompatible("modelhost", "http://engine", "/v1", []string{"fixture"})
	engine.Profiles = []string{router.ProfileChat}
	p, err := inference.New(inference.Config{ChatLogicalOwner: "modelhost", Router: router.New(engine), HTTP: client, Now: clock, Idle: time.Hour,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: func(context.Context, *router.Host, string) (string, bool) { return "", true }})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	subject := inference.Subject{Account: "synthetic", Program: "prefill-test"}
	window := p.GetChatHistoryWindow(subject)
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "prefill", HistoryEpoch: window.Window.HistoryEpoch}, Target: wire.ChatTarget{Model: "fixture"}, Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "synthetic"}}}}, Options: &wire.ChatOptions{MaxOutput: 16}}}
	start := p.StartV3(inference.WithRoute(t.Context(), inference.RouteNative, "program"), subject, request)
	if start.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("start: %+v", start)
	}
	var active context.Context
	select {
	case active = <-requestCtx:
	case <-time.After(3 * time.Second):
		t.Fatal("request not dispatched")
	}
	write := func(event string) {
		t.Helper()
		if _, err := io.WriteString(writer, event); err != nil {
			t.Fatal(err)
		}
		select {
		case <-read:
		case <-time.After(3 * time.Second):
			t.Fatal("parser did not consume event")
		}
	}
	write(prefillEvent("request-a", 0, 10000, 0))
	for _, processed := range []int64{1000, 2000} {
		advance(179 * time.Second)
		write(prefillEvent("request-a", processed, 10000, 0))
		select {
		case ok := <-accepted:
			if !ok {
				t.Fatal("actual parser→provider rejected active prefill")
			}
		case <-time.After(3 * time.Second):
			t.Fatal("no trusted callback")
		}
	}
	// Let the real watchdog tick while the fake clock is already beyond the
	// original 180-second deadline. Genuine parsed progress must keep it live.
	select {
	case <-active.Done():
		t.Fatal("active prefill expired after more than 180 total seconds")
	case <-time.After(250 * time.Millisecond):
	}
	// A wholly malformed SSE JSON line correctly terminates the codec itself;
	// malformed counters within valid JSON must instead grant no liveness.
	negative := invalidPrefillEvents()
	negative = negative[:len(negative)-1]
	for _, event := range negative {
		advance(time.Second)
		write(event)
	}
	if len(accepted) != 0 {
		t.Fatal("negative corpus reached liveness callback")
	}
	advance((180 - time.Duration(len(negative))) * time.Second)
	deadline := time.Now().Add(3 * time.Second)
	cursor := int64(0)
	ended := false
	for time.Now().Before(deadline) && !ended {
		page := p.ObserveV3(t.Context(), subject, start.Request, cursor, 256, 65536, 10)
		cursor = page.Next
		for _, event := range page.Events {
			if event.Kind == wire.ChatEventKindProgress {
				t.Fatal("prefill fabricated a public progress phase")
			}
			if event.End != nil {
				if event.End.Outcome != wire.ChatResultOutcomeFailed || event.End.Cause != "stalled" {
					t.Fatalf("wrong terminal: %+v", event.End)
				}
				ended = true
			}
		}
	}
	if !ended {
		t.Fatal("180 seconds after last actual advance did not stall")
	}
	if inference.ReportExecutionPrefill(active, 3000, 10000) || inference.ReportExecutionPrefill(context.Background(), 1, 2) {
		t.Fatal("late/foreign context renewed operation")
	}
	select {
	case <-joined:
	case <-time.After(3 * time.Second):
		t.Fatal("owned fake HTTP request did not join")
	}
}
