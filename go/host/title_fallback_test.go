package host

import (
	"context"
	"strings"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	router "github.com/openabstractions/abstraction-router/go"
)

// A title for a model whose one resident slot holds a conversation is
// served by the smallest other chat model that loads without asking any
// holder to yield; the main model keeps its lease, its engine and its
// conversation's slot. With the card held, nothing loads, nothing is
// evicted, and the title fails with upstream:503.
func TestATitleMovesToTheSmallestModelThatFits(t *testing.T) {
	requireEnginePeer(t)
	book := &freeBook{fakeLeases: newFakeLeases()}
	h := freeHost(t, book, func(o *Options) {
		o.Models = []Model{
			{Name: "main:30b", Store: "fake", Object: "main"},
			{Name: "small:1b", Store: "fake", Object: "small"},
			{Name: "tiny:135m", Store: "fake", Object: "tiny"},
		}
		o.Source = fakeSource{objects: map[string]Object{
			"fake/main":  {Store: "fake", ID: "main", Path: weightsFile(t, 16384), File: "model.gguf", Family: "main", Size: 16384},
			"fake/small": {Store: "fake", ID: "small", Path: weightsFile(t, 8192), File: "model.gguf", Family: "small", Size: 8192},
			"fake/tiny":  {Store: "fake", ID: "tiny", Path: weightsFile(t, 4096), File: "model.gguf", Family: "tiny", Size: 4096},
		}}
	})
	if _, err := h.Ensure(context.Background(), "main:30b"); err != nil {
		t.Fatal(err)
	}
	mainLease := h.Lease("main:30b")
	// The main model's launch admitted one resident slot, and the chat's
	// conversation holds it.
	h.engineMu.Lock()
	slots := newCacheSlots(1)
	h.engines[h.hosts["main:30b"]].cacheSlots = slots
	h.engineMu.Unlock()
	conversation, err := slots.reserve(context.Background(), "conversation", "partition")
	if err != nil {
		t.Fatal(err)
	}
	conversation.release(true)

	route := router.OpenAICompatible("modelhost", h.Base("main:30b"), "/v1", h.Models())
	route.Profiles = []string{router.ProfileChat}
	provider, err := inference.New(inference.Config{ChatLogicalOwner: "modelhost", Router: router.New(route), HTTP: EngineClient(h, nil),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: modelhostAdmit(h)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { provider.Close() })
	subject := inference.Subject{Account: "fixture-account", Program: "opencode-fixture"}
	window := provider.GetChatHistoryWindow(subject)
	if window.Window == nil {
		t.Fatal("history window unavailable")
	}
	title := func(key string) (wire.ChatResultOutcome, string, string) {
		t.Helper()
		use := wire.ChatUseTitle
		request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: key, HistoryEpoch: window.Window.HistoryEpoch}, Use: &use,
			Target: wire.ChatTarget{Model: "main:30b"}, Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser,
				Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "Generate a title for this conversation."}}}}}}
		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
		defer cancel()
		admission := provider.StartV3(ctx, subject, request)
		if admission.Outcome != wire.ChatStartOutcomeAccepted {
			t.Fatalf("%s admission: %+v", key, admission)
		}
		var text strings.Builder
		cursor := int64(0)
		for {
			page := provider.ObserveV3(ctx, subject, admission.Request, cursor, 256, 65536, 1000)
			if page.Outcome != wire.PageOutcomePage {
				t.Fatalf("%s observation: %s", key, page.Outcome)
			}
			for _, event := range page.Events {
				if event.Part != nil && event.Part.Kind == wire.ChatPartKindText {
					text.WriteString(event.Part.Text)
				}
				if event.End != nil {
					return event.End.Outcome, event.End.Cause, text.String()
				}
			}
			cursor = page.Next
			if page.AtEnd {
				t.Fatalf("%s ended without a terminal event", key)
			}
		}
	}
	unchanged := func(step string) {
		t.Helper()
		if !h.Loaded("main:30b") || h.Lease("main:30b") != mainLease {
			t.Fatalf("%s: the main model lost its residency or lease", step)
		}
		slots.mu.Lock()
		defer slots.mu.Unlock()
		if slots.slots[0].scope != "conversation" {
			t.Fatalf("%s: the conversation lost its slot: %+v", step, slots.slots[0])
		}
	}

	outcome, cause, text := title("title-free")
	if outcome != wire.ChatResultOutcomeCompleted || !strings.Contains(text, "tiny:135m") {
		t.Fatalf("title with room on the card: %s %s %q, want completed by tiny:135m", outcome, cause, text)
	}
	if !h.Loaded("tiny:135m") || !strings.HasPrefix(h.Lease("tiny:135m"), "free-") || h.Loaded("small:1b") {
		t.Fatalf("tiny loaded %v lease %q, small loaded %v; want tiny alone through the no-yield acquire",
			h.Loaded("tiny:135m"), h.Lease("tiny:135m"), h.Loaded("small:1b"))
	}
	unchanged("title with room")
	if acquires, _, _ := book.counts(); acquires != 1 {
		t.Fatalf("%d yielding acquires; want the main model's one", acquires)
	}

	// The card is full: no model loads beside the main one.
	h.Unload("tiny:135m", "test")
	book.setHeld(true)
	outcome, cause, _ = title("title-held")
	if outcome != wire.ChatResultOutcomeFailed || cause != "upstream:503" || h.Loaded("tiny:135m") || h.Loaded("small:1b") {
		t.Fatalf("title on a full card: %s %s, tiny %v small %v; want upstream:503 and nothing loaded", outcome, cause, h.Loaded("tiny:135m"), h.Loaded("small:1b"))
	}
	unchanged("title on a full card")
	if acquires, _, _ := book.counts(); acquires != 1 {
		t.Fatalf("%d yielding acquires; want the main model's one", acquires)
	}
}
