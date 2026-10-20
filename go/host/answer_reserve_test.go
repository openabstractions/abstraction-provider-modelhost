package host

import (
	"context"
	"encoding/json"
	"testing"
)

// A thinking model given a small output limit spent all of it reasoning and
// the chat ended upstream:reasoning_only with no answer ("Say hi in five
// words." at max_tokens 50 and 600 on qwen3.6-35b-a3b, 2026-11-11). The
// family's engine profile keeps part of each named limit for the answer.
// Reasoning is always bounded: a request without a limit or effort gets
// the family's default effort, medium, at 4096 tokens (architect ruling
// 2026-11-12 §3). A limit below twice the reserve, or reasoning_effort
// none, turns thinking off: at max_tokens 10 the budget message alone ate
// the limit (R34, OpenCode "Answer in one word." ended
// upstream:reasoning_only).
func TestAnswerReserveBoundsReasoningWithinTheOutputLimit(t *testing.T) {
	recipe := profileCache("qwen3.6-35b-a3b", nil, 0)
	if recipe.answerReserve.tokens != 256 {
		t.Fatalf("qwen3.6 answer reserve = %d, want the profile's 256", recipe.answerReserve.tokens)
	}
	for _, resident := range []bool{false, true} {
		for _, tc := range []struct {
			body   string
			budget any
			off    bool
		}{
			{`{"messages":[],"max_tokens":10}`, nil, true},
			{`{"messages":[],"max_tokens":50}`, nil, true},
			{`{"messages":[],"max_tokens":511}`, nil, true},
			{`{"messages":[],"max_tokens":512}`, float64(256), false},
			{`{"messages":[],"max_tokens":600}`, float64(344), false},
			{`{"messages":[],"max_tokens":4096,"reasoning_effort":"none"}`, nil, true},
			{`{"messages":[],"max_tokens":4096}`, float64(3072), false},
			{`{"messages":[],"max_tokens":32000}`, float64(4096), false},
			{`{"messages":[],"max_tokens":32000,"reasoning_effort":"high"}`, float64(16384), false},
			{`{"messages":[],"max_tokens":4096,"reasoning_effort":"low"}`, float64(1024), false},
			{`{"messages":[],"max_tokens":4096,"reasoning_effort":"unknown-word"}`, float64(3072), false},
			{`{"messages":[],"reasoning_effort":"high"}`, float64(16384), false},
			{`{"messages":[]}`, float64(4096), false},
			{`{"messages":[],"max_tokens":600,"reasoning_budget_tokens":-1}`, float64(-1), false},
		} {
			process := &Process{answerReserve: recipe.answerReserve}
			if resident {
				process.cacheSlots = newCacheSlots(2)
			}
			h := &Host{engines: map[string]*Process{"engine": process}}
			out, release, err := h.prepareCacheRequest(context.Background(), "engine", "/v1/chat/completions", []byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			if release != nil {
				release(true)
			}
			var body map[string]any
			if err := json.Unmarshal(out, &body); err != nil {
				t.Fatal(err)
			}
			if got := body["reasoning_budget_tokens"]; got != tc.budget {
				t.Fatalf("resident=%v %s: reasoning_budget_tokens = %v, want %v", resident, tc.body, got, tc.budget)
			}
			kwargs, _ := body["chat_template_kwargs"].(map[string]any)
			if off := kwargs["enable_thinking"] == false; off != tc.off {
				t.Fatalf("resident=%v %s: thinking off = %v, want %v (%s)", resident, tc.body, off, tc.off, out)
			}
			if tc.off && body["reasoning_budget_message"] != nil {
				t.Fatalf("resident=%v %s: thinking off still sends the budget message", resident, tc.body)
			}
		}
	}
	// A family without a reserve keeps the request bytes.
	h := &Host{engines: map[string]*Process{"engine": {}}}
	raw := []byte(`{"messages":[],"max_tokens":50}`)
	if out, _, err := h.prepareCacheRequest(context.Background(), "engine", "/v1/chat/completions", raw); err != nil || string(out) != string(raw) {
		t.Fatalf("no reserve changed the request: %s %v", out, err)
	}
}
