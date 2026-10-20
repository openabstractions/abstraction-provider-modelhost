package host

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Owner, 2026-11-09: a new chat that lands on a slot an old chat used "has to
// trim to position of new chat". Modelhost sends the engine only the new
// chat's own messages, its slot and cache_prompt; llama.cpp then keeps the
// longest common token prefix of that slot and discards everything after it.
// On the owner's machine the old chat that took the new chat's slot reused
// 16110 tokens, the shared system prompt and tools of a 16121-token prompt.
// No saved slot or old conversation is restored: the request carries no
// earlier turn, and the lease reports a miss.
func TestCacheSlotsNewChatSendsOnlyItsOwnPrompt(t *testing.T) {
	h := &Host{engines: map[string]*Process{"engine": {cacheSlots: newCacheSlots(1)}}}
	system := map[string]any{"role": "system", "content": "OpenCode system prompt and tools"}
	send := func(scope string, messages []any) map[string]any {
		t.Helper()
		raw, err := json.Marshal(map[string]any{"messages": messages, "stream": true})
		if err != nil {
			t.Fatal(err)
		}
		out, release, err := h.prepareScopedCacheRequest(t.Context(), "engine", "/v1/chat/completions", raw, scope, "opencode")
		if err != nil {
			t.Fatal(err)
		}
		defer release(true)
		var body map[string]any
		if err := json.Unmarshal(out, &body); err != nil {
			t.Fatal(err)
		}
		return body
	}
	old := []any{system,
		map[string]any{"role": "user", "content": "Generate an image of a red fox in snow"},
		map[string]any{"role": "assistant", "content": "Image generation isn't available right now."}}
	send("chat-old", old)

	// The new chat's first lease is a miss: the only slot holds the old chat,
	// and a lease is a hit only for the same conversation scope.
	s := h.engines["engine"].cacheSlots
	if probe, err := s.reserve(t.Context(), "chat-new", "opencode"); err != nil || probe.reused || probe.cause != "not_resident" || !probe.evicted {
		t.Fatalf("the new chat's first lease should miss and evict the old chat: %+v %v", probe, err)
	} else {
		probe.release(false) // A failed lease on its own program's slot keeps the new chat's scope.
	}
	send("chat-old", old)

	fresh := []any{system, map[string]any{"role": "user", "content": "hi"}}
	body := send("chat-new", fresh)
	if body["id_slot"] != float64(0) || body["cache_prompt"] != true {
		t.Fatalf("the new chat should share the program's prefix on the only slot: %v", body)
	}
	// Normalise through JSON so the comparison sees exactly what the engine gets.
	var want []any
	raw, _ := json.Marshal(fresh)
	_ = json.Unmarshal(raw, &want)
	if !reflect.DeepEqual(body["messages"], want) {
		t.Fatalf("the engine received turns beyond the new chat's own prompt: %v", body["messages"])
	}
	for _, field := range []string{"id_slot", "cache_prompt", "chat_template_kwargs", "messages", "stream"} {
		delete(body, field)
	}
	if len(body) != 0 {
		t.Fatalf("modelhost added request fields that could restore other state: %v", body)
	}
	if got, err := s.reserve(t.Context(), "chat-new", "opencode"); err != nil || !got.reused || got.cause != "resident" {
		t.Fatalf("the new chat's own next turn should find its slot: %+v %v", got, err)
	} else {
		got.release(true)
	}
	if got, err := s.reserve(t.Context(), "chat-old", "opencode"); err != nil || got.reused || got.cause != "not_resident" {
		t.Fatalf("the old chat must not find its state after the new chat took the slot: %+v %v", got, err)
	} else {
		got.release(true)
	}
}
