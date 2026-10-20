package host

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	router "github.com/openabstractions/abstraction-router/go"
)

// budgetEngine stands in for llama.cpp b10825 with a thinking template. It
// streams one reasoning token per thought step until reasoning_budget_tokens
// runs out, then writes reasoning_budget_message into the reasoning and
// forces the end-of-thinking tag. With no message the model goes on with the
// thought it was cut in, after the tag, as content. A CPU probe of b10825 on
// 2026-11-11 showed both: budget 10 alone gave content " looking for a job
// in <place>.", the rest of the cut sentence; with a message the content
// started fresh.
func budgetEngine(t *testing.T, seen chan<- map[string]any, ignoresMessage bool) prefillRoundTrip {
	const answer = "Hi there, how are you?"
	thought := make([]string, 500)
	for i := range thought {
		thought[i] = fmt.Sprintf("Wait, let's count properly (%d). ", i)
	}
	return prefillRoundTrip(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		seen <- body
		limit, _ := body["max_tokens"].(float64)
		budget := len(thought)
		if b, ok := body["reasoning_budget_tokens"].(float64); ok && b >= 0 {
			budget = min(int(b), budget)
		}
		message, _ := body["reasoning_budget_message"].(string)
		var stream strings.Builder
		emit := func(delta map[string]any, finish string) {
			choice := map[string]any{"index": 0, "delta": delta}
			if finish != "" {
				choice["finish_reason"] = finish
			}
			raw, _ := json.Marshal(map[string]any{"choices": []any{choice}})
			fmt.Fprintf(&stream, "data: %s\n\n", raw)
		}
		used := 0
		for _, step := range thought[:budget] {
			emit(map[string]any{"reasoning_content": step}, "")
			used++
		}
		finish := "stop"
		switch {
		case budget == len(thought) || message != "":
			if message != "" {
				emit(map[string]any{"reasoning_content": message}, "")
			}
			if message != "" && ignoresMessage {
				// R28 on the real Qwen3.6: the model went on with the cut
				// thought as content and wrote its own end tag, split here
				// across two chunks, before the answer.
				for _, piece := range []string{"Hi there, how are you today? (6) - wait, let's count properly.\n",
					"Hi(1) there(2) how(3) are(4) you(5).\n</th", "ink>\n\n"} {
					emit(map[string]any{"content": piece}, "")
				}
			}
			emit(map[string]any{"content": answer}, "")
		default:
			for _, step := range thought[budget:] {
				if used >= int(limit) {
					finish = "length"
					break
				}
				emit(map[string]any{"content": step}, "")
				used++
			}
		}
		emit(map[string]any{}, finish)
		stream.WriteString("data: [DONE]\n\n")
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(stream.String())), Request: r}, nil
	})
}

// On R27 "Say hi in five words." at max_output 600 on qwen3.6-35b-a3b
// returned the model's counting ("Wait, let's count properly…") as answer
// text with stop_reason max_output: the answer reserve cut the reasoning at
// 344 tokens and the model finished the cut thought after the forced tag.
// The family's engine profile closes cut reasoning with a message, so the
// answer text holds only the answer and the reasoning stays in its channel.
func TestAnswerReserveKeepsCutReasoningOutOfTheAnswerText(t *testing.T) {
	answerCue(t, false)
}

// On R28 the same request over chat@2 still returned the cut thought, a
// literal </think> and then the answer as text: the budget message was sent
// and written into the reasoning, and Qwen3.6 went on thinking as content
// anyway. Modelhost's engine transport moves the text before the model's own
// end tag into the reasoning channel.
func TestAnswerReserveMovesAThoughtTheModelContinuedPastTheCut(t *testing.T) {
	answerCue(t, true)
}

func answerCue(t *testing.T, ignoresMessage bool) {
	t.Helper()
	recipe := profileCache("qwen3.6-35b-a3b", nil, 0)
	if recipe.answerReserve.tokens != 256 || recipe.answerReserve.message == "" {
		t.Fatalf("qwen3.6 answer reserve = %+v, want 256 tokens and a closing message", recipe.answerReserve)
	}
	seen := make(chan map[string]any, 1)
	h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(),
		engines: map[string]*Process{"engine": {key: "fixture-key", answerReserve: recipe.answerReserve}}}
	client := engineClient(h, nil, budgetEngine(t, seen, ignoresMessage))
	engine := router.OpenAICompatible("modelhost", "http://engine", "/v1", []string{"fixture"})
	engine.Profiles = []string{router.ProfileChat}
	p, err := inference.New(inference.Config{ChatLogicalOwner: "modelhost", Router: router.New(engine), HTTP: client, Idle: time.Hour,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: func(context.Context, *router.Host, string) (string, bool) { return "", true }})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	subject := inference.Subject{Account: "synthetic", Program: "answer-cue-test"}
	window := p.GetChatHistoryWindow(subject)
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "answer-cue", HistoryEpoch: window.Window.HistoryEpoch},
		Target: wire.ChatTarget{Model: "fixture"},
		Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "Say hi in five words."}}}},
			Options: &wire.ChatOptions{MaxOutput: 600}}}
	start := p.StartV3(inference.WithRoute(t.Context(), inference.RouteNative, "program"), subject, request)
	if start.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("start: %+v", start)
	}
	var text, reasoning strings.Builder
	var end *wire.ChatResult
	deadline := time.Now().Add(5 * time.Second)
	for cursor := int64(0); end == nil && time.Now().Before(deadline); {
		page := p.ObserveV3(t.Context(), subject, start.Request, cursor, 256, 65536, 50)
		cursor = page.Next
		for _, event := range page.Events {
			if event.Part != nil {
				switch event.Part.Kind {
				case wire.ChatPartKindText:
					text.WriteString(event.Part.Text)
				case wire.ChatPartKindReasoning:
					reasoning.WriteString(event.Part.Text)
				}
			}
			if event.End != nil {
				end = event.End
			}
		}
	}
	if end == nil || end.Outcome != wire.ChatResultOutcomeCompleted || end.FinishReason == nil || *end.FinishReason != wire.FinishReasonStop {
		t.Fatalf("terminal = %+v, want completed with finish stop", end)
	}
	if text.String() != "Hi there, how are you?" {
		t.Fatalf("answer text = %.120q (%d bytes), want only the answer", text.String(), text.Len())
	}
	cutThought := ""
	if ignoresMessage {
		cutThought = "Hi there, how are you today? (6) - wait, let's count properly.\nHi(1) there(2) how(3) are(4) you(5).\n"
	}
	if !strings.Contains(reasoning.String(), "let's count properly (343)") || !strings.HasSuffix(reasoning.String(), recipe.answerReserve.message+cutThought) {
		t.Fatalf("reasoning channel lost the cut thought or its closing message: %d bytes", reasoning.Len())
	}
	body := <-seen
	if body["reasoning_budget_tokens"] != float64(344) || body["reasoning_budget_message"] != recipe.answerReserve.message {
		t.Fatalf("engine request budget=%v message=%q, want 344 and the profile message", body["reasoning_budget_tokens"], body["reasoning_budget_message"])
	}
}
