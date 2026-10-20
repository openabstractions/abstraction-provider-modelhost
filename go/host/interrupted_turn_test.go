package host

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
)

// streamingInProcess answers a request from handler as an engine does: the
// response returns once the handler writes its header, and the body streams
// what it writes after, until it returns.
type streamingInProcess struct{ handler http.Handler }

type streamingWriter struct {
	header  http.Header
	pipe    *io.PipeWriter
	status  int
	started chan struct{}
	once    sync.Once
}

func (w *streamingWriter) Header() http.Header { return w.header }
func (w *streamingWriter) WriteHeader(status int) {
	w.once.Do(func() { w.status = status; close(w.started) })
}
func (w *streamingWriter) Write(p []byte) (int, error) {
	w.WriteHeader(http.StatusOK)
	return w.pipe.Write(p)
}
func (w *streamingWriter) Flush() {}

func (p streamingInProcess) RoundTrip(req *http.Request) (*http.Response, error) {
	body, pipe := io.Pipe()
	w := &streamingWriter{header: http.Header{}, pipe: pipe, started: make(chan struct{})}
	go func() {
		p.handler.ServeHTTP(w, req)
		w.WriteHeader(http.StatusOK)
		pipe.Close()
	}()
	select {
	case <-w.started:
	case <-req.Context().Done():
		pipe.CloseWithError(req.Context().Err())
		return nil, req.Context().Err()
	}
	return &http.Response{StatusCode: w.status, Header: w.header, Body: body, Request: req}, nil
}

// interruptEngine answers every chat request at once, except the prompt
// "interrupt me": it streams one token and then waits until the request is
// canceled, as llama-server does while it generates.
type interruptEngine struct {
	titleSlotEngine
	generating chan struct{}
}

func (e *interruptEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
		Slot        *int  `json:"id_slot"`
		CachePrompt *bool `json:"cache_prompt"`
	}
	if r.URL.Path != "/v1/chat/completions" || json.NewDecoder(r.Body).Decode(&body) != nil || body.Slot == nil || body.CachePrompt == nil || len(body.Messages) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	prompt := body.Messages[len(body.Messages)-1].Content
	e.mu.Lock()
	e.calls = append(e.calls, titleSlotCall{prompt: prompt, slot: *body.Slot, cachePrompt: *body.CachePrompt})
	e.mu.Unlock()
	w.Header().Set("Content-Type", "text/event-stream")
	if prompt != "interrupt me" {
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
		return
	}
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"thinking\"}}]}\n\n")
	close(e.generating)
	<-r.Context().Done()
}

// An interrupted turn keeps its conversation's prompt cache. The person
// stops a reply (OpenCode's Esc cancels the chat@3 request) and sends the
// next message: that turn goes to the slot the conversation held with the
// prompt cache on, and the engine reads only what the history added. Before
// this, the canceled request cleared the slot's conversation and program,
// and the next turn sent cache_prompt false and re-read the whole history
// (21k tokens, 79 s on the owner's machine).
func TestAnInterruptedTurnKeepsTheConversationsPromptCache(t *testing.T) {
	engine := &interruptEngine{generating: make(chan struct{})}
	h, provider, subject, call := chatSlotProvider(t, 1, streamingInProcess{handler: engine})
	if outcome, cause := call("turn one", "session", false); outcome != wire.ChatResultOutcomeCompleted {
		t.Fatalf("turn one: %s %s", outcome, cause)
	}

	window := provider.GetChatHistoryWindow(subject)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	admission := provider.StartV3(ctx, subject, wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "interrupted", HistoryEpoch: window.Window.HistoryEpoch},
		Conversation: "session", Target: wire.ChatTarget{Model: "fixture"}, Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser,
			Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "interrupt me"}}}}}})
	if admission.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("interrupted turn admission: %+v", admission)
	}
	select {
	case <-engine.generating:
	case <-ctx.Done():
		t.Fatal("the interrupted turn never reached the engine")
	}
	if cancelled := provider.CancelV3(subject, admission.Request); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel: %+v", cancelled)
	}
	for cursor, ended := int64(0), false; !ended; {
		page := provider.ObserveV3(ctx, subject, admission.Request, cursor, 256, 65536, 1000)
		if page.Outcome != wire.PageOutcomePage {
			t.Fatalf("interrupted turn observation: %s", page.Outcome)
		}
		for _, event := range page.Events {
			if event.End != nil {
				if event.End.Outcome != wire.ChatResultOutcomeCancelled {
					t.Fatalf("interrupted turn ended %s %s, want cancelled", event.End.Outcome, event.End.Cause)
				}
				ended = true
			}
		}
		cursor = page.Next
	}
	// The slot is released once the canceled response body closes.
	slots := h.engines["engine"].cacheSlots
	for deadline := time.Now().Add(5 * time.Second); ; {
		slots.mu.Lock()
		busy := slots.slots[0].busy
		slots.mu.Unlock()
		if !busy {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the canceled turn kept its slot busy")
		}
		time.Sleep(10 * time.Millisecond)
	}

	if outcome, cause := call("turn three", "session", false); outcome != wire.ChatResultOutcomeCompleted {
		t.Fatalf("turn three: %s %s", outcome, cause)
	}
	calls := engine.taken()
	if len(calls) != 3 || calls[1].prompt != "interrupt me" || calls[2].prompt != "turn three" {
		t.Fatalf("engine saw %+v, want turn one, the interrupted turn and turn three", calls)
	}
	if calls[2].slot != calls[0].slot || !calls[2].cachePrompt {
		t.Fatalf("turn after the interruption ran on slot %d with cache_prompt %v, want slot %d with the prompt cache", calls[2].slot, calls[2].cachePrompt, calls[0].slot)
	}
}
