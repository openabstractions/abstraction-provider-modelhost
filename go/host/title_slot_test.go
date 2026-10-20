package host

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	router "github.com/openabstractions/abstraction-router/go"
)

// titleSlotEngine is a chat engine fixture that records the slot and cache
// setting of every chat request the media guard lets through.
type titleSlotEngine struct {
	mu    sync.Mutex
	calls []titleSlotCall
}

type titleSlotCall struct {
	prompt      string
	slot        int
	cachePrompt bool
}

func (e *titleSlotEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
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
	e.mu.Lock()
	e.calls = append(e.calls, titleSlotCall{prompt: body.Messages[len(body.Messages)-1].Content, slot: *body.Slot, cachePrompt: *body.CachePrompt})
	e.mu.Unlock()
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
}

func (e *titleSlotEngine) taken() []titleSlotCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]titleSlotCall(nil), e.calls...)
}

// titleSlotProvider is the actual inference provider over a modelhost engine
// whose launch admitted slots resident conversation slots. call starts one
// chat@3 request and returns its terminal outcome and cause.
func titleSlotProvider(t *testing.T, slots int) (*Host, *titleSlotEngine, func(label, conversation string, title bool) (wire.ChatResultOutcome, string)) {
	t.Helper()
	engine := &titleSlotEngine{}
	h, _, _, call := chatSlotProvider(t, slots, inProcess{handler: engine})
	return h, engine, call
}

// chatSlotProvider is the actual inference provider over a modelhost engine
// whose launch admitted slots resident conversation slots; engine answers the
// requests the media guard lets through. call runs one chat@3 request to its
// terminal outcome.
func chatSlotProvider(t *testing.T, slots int, engine http.RoundTripper) (*Host, *inference.Provider, inference.Subject, func(label, conversation string, title bool) (wire.ChatResultOutcome, string)) {
	t.Helper()
	h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(),
		engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(slots)}}}
	route := router.OpenAICompatible("modelhost", h.Base("fixture"), "/v1", h.Models())
	route.Profiles = []string{router.ProfileChat}
	provider, err := inference.New(inference.Config{ChatLogicalOwner: "modelhost", Router: router.New(route), HTTP: engineClient(h, nil, engine),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: func(context.Context, *router.Host, string) (string, bool) { return "", true }})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { provider.Close() })
	subject := inference.Subject{Account: "fixture-account", Program: "opencode-fixture"}
	window := provider.GetChatHistoryWindow(subject)
	if window.Window == nil {
		t.Fatal("history window unavailable")
	}
	call := func(label, conversation string, title bool) (wire.ChatResultOutcome, string) {
		t.Helper()
		request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: label, HistoryEpoch: window.Window.HistoryEpoch}, Conversation: conversation,
			Target: wire.ChatTarget{Model: "fixture"}, Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser,
				Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: label}}}}}}
		if title {
			use := wire.ChatUseTitle
			request.Use = &use
		}
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		admission := provider.StartV3(ctx, subject, request)
		if admission.Outcome != wire.ChatStartOutcomeAccepted {
			t.Fatalf("%s admission: %+v", label, admission)
		}
		cursor := int64(0)
		for {
			page := provider.ObserveV3(ctx, subject, admission.Request, cursor, 256, 65536, 1000)
			if page.Outcome != wire.PageOutcomePage {
				t.Fatalf("%s observation: %s", label, page.Outcome)
			}
			for _, event := range page.Events {
				if event.End != nil {
					return event.End.Outcome, event.End.Cause
				}
			}
			cursor = page.Next
			if page.AtEnd {
				t.Fatalf("%s ended without a terminal event", label)
			}
		}
	}
	return h, provider, subject, call
}

// OpenCode's title request carries no conversation (CC-OPENCODE-TITLE-MODEL).
// On an engine with one resident slot it fails with upstream:503 without
// reaching the engine, and the chat's next turn finds its prompt cache in the
// slot it left.
func TestATitleNeverTakesTheOnlyResidentSlot(t *testing.T) {
	h, engine, call := titleSlotProvider(t, 1)
	if outcome, cause := call("turn one", "session", false); outcome != wire.ChatResultOutcomeCompleted {
		t.Fatalf("turn one: %s %s", outcome, cause)
	}
	if outcome, cause := call("title", "", true); outcome != wire.ChatResultOutcomeFailed || cause != "upstream:503" {
		t.Fatalf("title on the only slot: %s %s, want failed upstream:503", outcome, cause)
	}
	if outcome, cause := call("turn two", "session", false); outcome != wire.ChatResultOutcomeCompleted {
		t.Fatalf("turn two: %s %s", outcome, cause)
	}
	calls := engine.taken()
	if len(calls) != 2 || calls[0].prompt != "turn one" || calls[1].prompt != "turn two" || calls[0].slot != 0 || calls[1].slot != 0 || !calls[1].cachePrompt {
		t.Fatalf("engine saw %+v, want both turns on slot 0 with the prompt cache and no title", calls)
	}
	slots := h.engines["engine"].cacheSlots
	slots.mu.Lock()
	defer slots.mu.Unlock()
	if slots.slots[0].scope == "" || slots.slots[0].busy {
		t.Fatalf("the conversation lost its slot: %+v", slots.slots[0])
	}
}

// With two resident slots the title runs in the slot no conversation holds,
// the conversation keeps its own, and a later title reuses the title's slot.
// Once a second conversation holds that slot, a title fails instead of
// evicting either conversation.
func TestATitleRunsBesideResidentConversations(t *testing.T) {
	_, engine, call := titleSlotProvider(t, 2)
	steps := []struct {
		label, conversation string
		title               bool
		want                wire.ChatResultOutcome
	}{
		{"main one", "main", false, wire.ChatResultOutcomeCompleted},
		{"title one", "", true, wire.ChatResultOutcomeCompleted},
		{"main two", "main", false, wire.ChatResultOutcomeCompleted},
		{"title two", "", true, wire.ChatResultOutcomeCompleted},
		{"other one", "other", false, wire.ChatResultOutcomeCompleted},
		{"title three", "", true, wire.ChatResultOutcomeFailed},
		{"main three", "main", false, wire.ChatResultOutcomeCompleted},
	}
	for _, step := range steps {
		if outcome, cause := call(step.label, step.conversation, step.title); outcome != step.want {
			t.Fatalf("%s: %s %s, want %s", step.label, outcome, cause, step.want)
		}
	}
	slot := map[string]int{}
	for _, c := range engine.taken() {
		slot[c.prompt] = c.slot
	}
	if len(slot) != 6 {
		t.Fatalf("engine saw %v, want six requests and no third title", slot)
	}
	main := slot["main one"]
	if slot["main two"] != main || slot["main three"] != main || slot["title one"] == main || slot["title two"] != slot["title one"] || slot["other one"] != slot["title one"] {
		t.Fatalf("slots %v: a title shared or evicted a conversation's slot", slot)
	}
}
