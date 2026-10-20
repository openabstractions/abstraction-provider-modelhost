package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

// This fixture consumes chat@3 output, including its actual terminal event.
// Public reasoning contributes counts only; replay uses whole provider seals.
// Diagnostics contain no content, argument text, seal bytes or raw SSE.
type cacheKnowledgeOutput struct {
	text                             bytes.Buffer
	replay                           []wire.ChatReplay
	end                              *wire.ChatResult
	counts                           map[string]int
	lengths                          map[string]int
	sealed                           map[int64]bool
	closed                           map[int64]bool
	sealBytes, invalidSeals, omitted int
}

func cacheKnowledgeEngineMetadata(raw []byte) (timings map[string]any, finish string, err error) {
	finish = "not_observed"
	for _, line := range strings.Split(string(raw), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var event map[string]any
		if err = json.Unmarshal([]byte(data), &event); err != nil {
			return
		}
		if value, ok := event["timings"].(map[string]any); ok {
			timings = value
		}
		if choices, ok := event["choices"].([]any); ok {
			for _, choice := range choices {
				if value, ok := choice.(map[string]any); ok {
					if raw, ok := value["finish_reason"].(string); ok && raw != "" {
						word := wire.FinishReason(raw)
						finish = "other"
						if word.Known() {
							finish = string(word)
						}
					}
				}
			}
		}
	}
	return
}

func TestCacheKnowledgeStalledTerminalDiagnostic(t *testing.T) {
	for _, cause := range []string{"stalled", "DO_NOT_LOG"} {
		output := &cacheKnowledgeOutput{end: &wire.ChatResult{Outcome: wire.ChatResultOutcomeFailed, Cause: cause}}
		got := output.diagnostic()
		want := "stalled"
		if cause != "stalled" {
			want = "other"
		}
		if got["terminal_outcome"] != "failed" || got["terminal_cause"] != want {
			t.Fatal("typed stall classification lost or arbitrary cause leaked")
		}
	}
}

// The live fixture's prompts are synthetic. Its already-owned SSE tap permits
// this diagnostic suffix of public reasoning; opaque replay fields are ignored.
func cacheKnowledgeSyntheticReasoningTail(raw []byte) string {
	const limit = 768 // Worst JSON escaping leaves room in the existing 8-KiB row.
	tail := ""
	for _, line := range strings.Split(string(raw), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var event struct {
			Choices []struct {
				Delta struct {
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &event) != nil {
			continue
		}
		for _, choice := range event.Choices {
			part := choice.Delta.ReasoningContent
			if part == "" {
				part = choice.Delta.Reasoning
			}
			tail += part
			if len(tail) > limit {
				tail = tail[len(tail)-limit:]
				for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
					tail = tail[1:]
				}
			}
		}
	}
	return tail
}

func (o *cacheKnowledgeOutput) consume(events []wire.ChatEvent) {
	if o.counts == nil {
		o.counts, o.lengths, o.sealed, o.closed = map[string]int{}, map[string]int{}, map[int64]bool{}, map[int64]bool{}
	}
	for _, event := range events {
		if event.End != nil {
			end := *event.End
			o.end = &end
		}
		if event.BlockEnd != nil && event.BlockEnd.SealOmitted != nil {
			o.omitted++
		}
		if event.BlockEnd != nil {
			o.closed[event.BlockEnd.Block] = true
		}
		p := event.Part
		if p == nil {
			continue
		}
		kind := p.Kind.String()
		o.counts[kind]++
		o.lengths[kind] += len(p.Text) + len(p.Arguments)
		if p.Kind == wire.ChatPartKindText {
			o.text.WriteString(p.Text)
		}
		if p.Seal == nil {
			continue
		}
		seal := *p.Seal
		o.sealBytes += len(seal.Bytes)
		// Match this fixture's selected OpenAI-compatible replay wire and
		// its existing whole-seal/combined-field bound. Never reconstruct it.
		if p.Kind != wire.ChatPartKindReasoning || o.sealed[p.Block] || seal.Origin != router.WireOpenAICompatible || len(seal.Bytes) == 0 || !utf8.Valid(seal.Bytes) || o.sealBytes > 512<<10 || len(o.replay) >= 64 {
			o.invalidSeals++
			continue
		}
		o.sealed[p.Block] = true
		seal.Bytes = bytes.Clone(seal.Bytes)
		o.replay = append(o.replay, wire.ChatReplay{Block: p.Block, Seal: seal})
	}
}

func (o *cacheKnowledgeOutput) diagnostic() map[string]any {
	outcome, finish, cause := "not_observed", "not_observed", "not_observed"
	if o.end != nil {
		outcome = o.end.Outcome.String()
		switch o.end.Cause {
		case "", "upstream:reasoning_only", "upstream:stream", "upstream:malformed", "upstream:error", "cancel", "caller", "stalled":
			cause = o.end.Cause
		default:
			cause = "other"
		}
		if o.end.FinishReason != nil {
			// Preserve a known finish word; unknown upstream words carry no
			// fixture action and must not become arbitrary diagnostic text.
			switch *o.end.FinishReason {
			case wire.FinishReasonStop, wire.FinishReasonLength, wire.FinishReasonToolCalls:
				finish = string(*o.end.FinishReason)
			default:
				finish = "other"
			}
		}
	}
	closed := true
	for _, replay := range o.replay {
		closed = closed && o.closed[replay.Block]
	}
	value := map[string]any{"terminal_outcome": outcome, "terminal_cause": cause, "finish_reason": finish, "part_counts": o.counts, "part_bytes": o.lengths, "visible_text_bytes": o.text.Len(), "opaque_replay_count": len(o.replay), "opaque_replay_bytes": o.sealBytes, "opaque_replay_valid": o.invalidSeals == 0 && o.omitted == 0 && closed, "invalid_seals": o.invalidSeals, "seal_omissions": o.omitted}
	if o.end != nil && o.end.Usage.Output != nil {
		value["reported_output_tokens"] = *o.end.Usage.Output
	}
	if o.end != nil {
		value["operation_timing"] = o.end.Timing
	}
	return value
}

func (o *cacheKnowledgeOutput) assistant() (wire.ChatMessage, error) {
	if o.end == nil || o.end.Outcome != wire.ChatResultOutcomeCompleted {
		return wire.ChatMessage{}, errors.New("fixture requires an observed completed terminal")
	}
	if o.invalidSeals != 0 || o.omitted != 0 {
		return wire.ChatMessage{}, errors.New("fixture cannot replay invalid or omitted seal material")
	}
	for _, replay := range o.replay {
		if !o.closed[replay.Block] {
			return wire.ChatMessage{}, errors.New("fixture cannot replay an unclosed seal block")
		}
	}
	if o.counts[wire.ChatPartKindToolCall.String()] != 0 || o.counts[wire.ChatPartKindModelRefusal.String()] != 0 {
		return wire.ChatMessage{}, errors.New("fixture has no tool-result or model-refusal continuation")
	}
	if o.text.Len() == 0 {
		return wire.ChatMessage{}, fmt.Errorf("fixture has no replayable assistant content (public reasoning bytes=%d)", o.lengths[wire.ChatPartKindReasoning.String()])
	}
	reply := wire.ChatMessage{Role: wire.RoleAssistant, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: o.text.String()}}, Reasoning: o.replay}
	if _, err := wire.ToJSON(reply); err != nil {
		return wire.ChatMessage{}, fmt.Errorf("fixture assistant codec: %w", err)
	}
	return reply, nil
}

func TestCacheKnowledgeResidentTerminalRepresentations(t *testing.T) {
	for _, mode := range []string{"visible", "sealed_reasoning", "mixed", "failed", "malformed", "unsealed", "empty", "tool_only", "reasoning_tool", "model_refusal", "functional_partial_length", "functional_unanswered_length", "live_model_default", "live_model_default_reasoning_only"} {
		t.Run(mode, func(t *testing.T) {
			maxOutput := int64(128)
			useModelTemperature := mode == "live_model_default" || mode == "live_model_default_reasoning_only"
			functional := mode == "visible" || mode == "mixed" || mode == "functional_partial_length" || mode == "functional_unanswered_length" || useModelTemperature
			if functional {
				maxOutput = cacheKnowledgeFunctionalOutput
			}
			h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(), engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(2)}}}
			var mu sync.Mutex
			calls, assistantMessages := 0, 0
			historySizes := map[int]int{}
			diagnostics := map[string]map[string]any{}
			transport := inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					MaxTokens   int64           `json:"max_tokens"`
					Temperature json.RawMessage `json:"temperature"`
					Messages    []struct {
						Role      string  `json:"role"`
						Content   *string `json:"content"`
						Reasoning string  `json:"reasoning_content"`
					} `json:"messages"`
					Content string `json:"content"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				switch r.URL.Path {
				case "/v1/chat/completions":
					if body.MaxTokens != maxOutput {
						t.Error("actual Provider changed the explicit total output budget")
					}
					if useModelTemperature {
						if len(body.Temperature) != 0 {
							t.Error("real live request did not omit temperature")
						}
					} else {
						var temperature float64
						if len(body.Temperature) == 0 || string(body.Temperature) == "null" || json.Unmarshal(body.Temperature, &temperature) != nil || temperature != 0 {
							t.Error("inert/comparative temperature control changed")
						}
					}
					mu.Lock()
					calls++
					historySizes[len(body.Messages)]++
					for index, message := range body.Messages {
						wantRole := "user"
						if index%2 == 1 {
							wantRole = "assistant"
						}
						if message.Role != wantRole {
							t.Error("history role/order changed")
						}
						if message.Role != "assistant" {
							continue
						}
						assistantMessages++
						wantText, wantReplay := "", ""
						if mode == "visible" || mode == "mixed" || mode == "functional_partial_length" || mode == "live_model_default" {
							wantText = "synthetic answer"
						}
						if mode == "sealed_reasoning" || mode == "mixed" || mode == "live_model_default" {
							wantReplay = "opaque replay"
						}
						text := ""
						if message.Content != nil {
							text = *message.Content
						}
						if text != wantText || message.Reasoning != wantReplay {
							t.Error("assistant text or opaque replay changed in actual codec")
						}
						if mode == "sealed_reasoning" && text != "" {
							t.Error("reasoning was fabricated into answer text")
						}
					}
					mu.Unlock()
					w.Header().Set("Content-Type", "text/event-stream")
					if mode == "malformed" {
						fmt.Fprint(w, "data: {malformed}\n\n")
						return
					}
					delta := map[string]any{}
					if mode == "visible" || mode == "mixed" || mode == "failed" || mode == "functional_partial_length" || mode == "live_model_default" {
						delta["content"] = "synthetic answer"
					}
					if mode == "sealed_reasoning" || mode == "mixed" || mode == "reasoning_tool" || mode == "functional_unanswered_length" || useModelTemperature {
						delta["reasoning_content"] = "opaque replay"
					}
					if mode == "unsealed" {
						delta["reasoning_content"] = strings.Repeat("r", (512<<10)+1)
					}
					if mode == "tool_only" || mode == "reasoning_tool" {
						delta["tool_calls"] = []any{map[string]any{"index": 0, "id": "synthetic-call", "function": map[string]string{"name": "synthetic-tool", "arguments": "{}"}}}
					}
					if mode == "model_refusal" {
						delta["reasoning_content"] = "opaque replay"
						delta["refusal"] = "synthetic refusal"
					}
					emit := func(v any) { raw, _ := json.Marshal(v); fmt.Fprintf(w, "data: %s\n\n", raw) }
					emit(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": delta}}})
					if mode == "failed" {
						emit(map[string]any{"error": map[string]string{"message": "synthetic failure"}})
						return
					}
					finish := "length"
					if functional && mode != "functional_partial_length" && mode != "functional_unanswered_length" && mode != "live_model_default_reasoning_only" {
						finish = "stop"
					}
					emit(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}, "usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 128}, "timings": map[string]any{"prompt_n": 1, "cache_n": 0, "prompt_ms": 1, "predicted_n": 128, "predicted_ms": 1000}})
					fmt.Fprint(w, "data: [DONE]\n\n")
				case applyTemplatePath:
					var rendered strings.Builder
					for _, message := range body.Messages {
						rendered.WriteString("<" + message.Role + ">")
						rendered.WriteString(message.Reasoning)
						if message.Content != nil {
							rendered.WriteString(*message.Content)
						}
						rendered.WriteString("</" + message.Role + ">\n")
					}
					rendered.WriteString("<assistant>")
					json.NewEncoder(w).Encode(map[string]any{"prompt": rendered.String()})
				case tokenizePath:
					tokens := []int64{}
					for _, c := range body.Content {
						tokens = append(tokens, int64(c))
					}
					json.NewEncoder(w).Encode(map[string]any{"tokens": tokens})
				default:
					t.Error("unexpected fixture endpoint")
					w.WriteHeader(404)
				}
			})}
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			values, err := cacheKnowledgeResidentSequenceBudget(ctx, h, "fixture", transport, func(context.Context, *router.Host, string) (string, bool) { return "", true }, maxOutput, useModelTemperature, func(label string, value map[string]any) error {
				mu.Lock()
				defer mu.Unlock()
				diagnostics[label] = value
				full, e := json.Marshal(value)
				if e != nil {
					return e
				}
				if len(full) > 8192 {
					return errors.New("diagnostic exceeded bound")
				}
				if useModelTemperature {
					if value["synthetic_reasoning_tail"] != "opaque replay" {
						return errors.New("synthetic public reasoning tail changed")
					}
				} else if _, ok := value["synthetic_reasoning_tail"]; ok {
					return errors.New("comparative control gained raw reasoning logging")
				}
				privateSafe := map[string]any{}
				for key, item := range value {
					if key != "synthetic_reasoning_tail" {
						privateSafe[key] = item
					}
				}
				raw, e := json.Marshal(privateSafe)
				if e != nil {
					return e
				}
				if len(raw) > 8192 || bytes.Contains(raw, []byte("opaque replay")) || bytes.Contains(raw, []byte("synthetic answer")) || bytes.Contains(raw, []byte("reasoning_content")) {
					return errors.New("diagnostic exposed content or exceeded bound")
				}
				return nil
			})
			positive := mode == "visible" || mode == "mixed" || mode == "functional_partial_length" || mode == "live_model_default"
			if positive {
				if err != nil {
					t.Fatal(err)
				}
				if len(values) != 8 || calls != 8 || len(diagnostics) != 8 || assistantMessages != 7 {
					t.Fatalf("incomplete eight-step terminal matrix: values=%d calls=%d diagnostics=%d assistant_history=%d", len(values), calls, len(diagnostics), assistantMessages)
				}
				if historySizes[1] != 4 || historySizes[3] != 2 || historySizes[5] != 1 || historySizes[7] != 1 {
					t.Fatal("full eight-step history composition changed")
				}
				for _, value := range diagnostics {
					finish := "stop"
					if mode == "functional_partial_length" {
						finish = "length"
					}
					if value["terminal_outcome"] != "completed" || value["finish_reason"] != finish || value["opaque_replay_valid"] != true || value["requested_max_output"] != int64(4096) {
						t.Fatal("completed terminal evidence missing")
					}
				}
			} else {
				if err == nil || len(diagnostics) != 2 || calls != 2 {
					t.Fatalf("invalid terminal continued: error=%v diagnostics=%d calls=%d", err, len(diagnostics), calls)
				}
				for _, value := range diagnostics {
					if (mode == "functional_unanswered_length" || mode == "live_model_default_reasoning_only") && (value["terminal_outcome"] != "failed" || value["terminal_cause"] != "upstream:reasoning_only" || value["upstream_finish_reason"] != "length" || value["requested_max_output"] != int64(4096)) {
						t.Fatal("unanswered reasoning-only length terminal was accepted")
					}
					if mode == "sealed_reasoning" && (value["terminal_outcome"] != "failed" || value["terminal_cause"] != "upstream:reasoning_only" || value["opaque_replay_count"].(int) == 0 || value["opaque_replay_valid"] != true) {
						t.Fatal("actual sealed-only Q8 refusal or seal evidence lost")
					}
					if mode == "sealed_reasoning" && (value["reported_output_tokens"] != int64(128) || value["upstream_finish_reason"] != "length" || value["requested_max_output"] != int64(128)) {
						t.Fatal("cap-before-answer diagnostic control lost")
					}
					if (mode == "tool_only" || mode == "reasoning_tool" || mode == "model_refusal") && value["terminal_outcome"] != "completed" {
						t.Fatal("tool/refusal-part output was reclassified as reasoning-only")
					}
					if (mode == "failed" || mode == "malformed") && value["terminal_outcome"] != "failed" {
						t.Fatal("failed terminal lost")
					}
					if mode == "unsealed" && (value["seal_omissions"].(int) == 0 || value["opaque_replay_count"].(int) != 0) {
						t.Fatal("omitted seal became replay")
					}
				}
			}
		})
	}
}

func TestCacheKnowledgeOutputRejectsMalformedSeal(t *testing.T) {
	for _, seal := range []wire.ReasoningSeal{{Origin: "wrong", Bytes: []byte("opaque")}, {Origin: router.WireOpenAICompatible, Bytes: []byte{0xff}}, {Origin: router.WireOpenAICompatible}} {
		o := &cacheKnowledgeOutput{}
		o.consume([]wire.ChatEvent{{Part: &wire.ChatPart{Kind: wire.ChatPartKindReasoning, Seal: &seal}}, {End: &wire.ChatResult{Outcome: wire.ChatResultOutcomeCompleted}}})
		if _, err := o.assistant(); err == nil || o.invalidSeals != 1 {
			t.Fatal("malformed seal admitted")
		}
	}
	o := &cacheKnowledgeOutput{}
	o.consume([]wire.ChatEvent{{Part: &wire.ChatPart{Kind: wire.ChatPartKindText, Text: "answer"}}, {Part: &wire.ChatPart{Block: 1, Kind: wire.ChatPartKindReasoning, Seal: &wire.ReasoningSeal{Origin: router.WireOpenAICompatible, Bytes: []byte("opaque")}}}, {End: &wire.ChatResult{Outcome: wire.ChatResultOutcomeCompleted}}})
	if _, err := o.assistant(); err == nil || o.diagnostic()["opaque_replay_valid"] != false {
		t.Fatal("unclosed seal admitted")
	}
}

func TestCacheKnowledgeSyntheticReasoningTailIsBoundedAndPublic(t *testing.T) {
	public := strings.Repeat("界", 600)
	raw, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": public, "reasoning": public, "content": "visible", "seal": map[string]string{"bytes": "opaque-not-for-logging"}}}}})
	tail := cacheKnowledgeSyntheticReasoningTail(append(append([]byte("data: "), raw...), []byte("\n\ndata: [DONE]\n\n")...))
	if len(tail) > 768 || !utf8.ValidString(tail) || !strings.HasSuffix(public, tail) || strings.Contains(tail, "opaque") || len(tail) == 0 {
		t.Fatal("synthetic public tail lost its bound or source")
	}
	raw = []byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"one\",\"reasoning\":\"one\"}}]}\n\n")
	if cacheKnowledgeSyntheticReasoningTail(raw) != "one" {
		t.Fatal("reasoning aliases duplicated")
	}
}
