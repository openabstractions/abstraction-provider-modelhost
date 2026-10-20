package service

import (
	"context"
	"runtime"
	"strings"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
	router "github.com/openabstractions/abstraction-router/go"
)

// chat3Turn is one chat@3 answer as a caller observes it.
type chat3Turn struct {
	reasoning, text string
	end             *wire.ChatResult
}

func runChat3Turn(t *testing.T, mediator *inference.Provider, subject inference.Subject, key string, content wire.ChatContent) chat3Turn {
	t.Helper()
	window := mediator.GetChatHistoryWindow(subject)
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: key, HistoryEpoch: window.Window.HistoryEpoch}, Target: wire.ChatTarget{Model: "vision:4b"}, Content: content}
	a := mediator.StartV3(context.Background(), subject, request)
	if a.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("%s: admission %+v", key, a)
	}
	var turn chat3Turn
	var reasoning, text strings.Builder
	ctx, done := context.WithTimeout(context.Background(), 30*time.Second)
	defer done()
	for cursor := int64(0); turn.end == nil && ctx.Err() == nil; {
		page := mediator.ObserveV3(ctx, subject, a.Request, cursor, 256, 65536, 1000)
		if page.Outcome != wire.PageOutcomePage {
			t.Fatalf("%s: observe %+v", key, page)
		}
		for _, event := range page.Events {
			if p := event.Part; p != nil {
				switch p.Kind {
				case wire.ChatPartKindReasoning:
					reasoning.WriteString(p.Text)
				case wire.ChatPartKindText:
					text.WriteString(p.Text)
				}
			}
			if event.End != nil {
				turn.end = event.End
			}
		}
		cursor = page.Next
	}
	if turn.end == nil {
		t.Fatalf("%s: no terminal", key)
	}
	turn.reasoning, turn.text = reasoning.String(), text.String()
	return turn
}

// Architect ruling 2026-11-12, reasoning effort: a thinking llama.cpp model
// behind modelhost asked for effort none answers with no reasoning; asked
// for low it thinks within its catalogue budget (the engine profile's 1024
// tokens), the budget message ends the thinking, and it answers; a named
// budget wins over the effort. The runtime relays chat@3 to the leaf, the
// leaf maps the option onto reasoning_effort and reasoning_budget_tokens,
// modelhost's reserveAnswer turns those into the engine's controls, and
// reasoning_applied reports what ran. A required native extension the model
// takes is admitted and sent, and one it does not take refuses naming it.
func TestChat3ReasoningEffortReachesALlamaCppModel(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS != "windows" {
		t.Skip("native engine peer proof requires Windows")
	}
	program, principal := nativeTitleIdentity(t)
	account, err := principal.Account()
	if err != nil {
		t.Fatal(err)
	}
	endpoint := visionLeaf(t, buildFakeEngine(t), program)
	native, err := router.NewNative(declaration.ProviderName, endpoint, listen.ServerExpectation{Principal: principal, Program: program}, []string{"vision:4b"}, []string{router.ProfileChat})
	if err != nil {
		t.Fatal(err)
	}
	routes := router.New(native)
	surveyCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := routes.SurveyContext(surveyCtx); err != nil {
		t.Fatal(err)
	}
	mediator, err := inference.New(inference.Config{ChatLogicalOwner: "fixture-runtime-job-owner", Router: routes,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "not_permitted"
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mediator.Close() })
	subject := inference.Subject{Account: account, Program: program}
	ask := func(reasoning *wire.ChatReasoning) wire.ChatContent {
		return wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "What is the capital of France?"}}}},
			Options: &wire.ChatOptions{MaxOutput: 4096, Reasoning: reasoning}, Guarantees: []wire.ChatGuarantee{wire.ChatGuaranteeLocalOnly}}
	}

	none := runChat3Turn(t, mediator, subject, "effort-none", ask(&wire.ChatReasoning{Effort: wire.ChatReasoningEffortNone}))
	if none.end.Outcome != wire.ChatResultOutcomeCompleted || none.text != "Paris" || none.reasoning != "" {
		t.Fatalf("none: %s/%s text %q reasoning %q", none.end.Outcome, none.end.Cause, none.text, none.reasoning)
	}
	if a := none.end.ReasoningApplied; a == nil || a.Effort == nil || *a.Effort != wire.ChatReasoningEffortNone || a.Support != wire.ChatReasoningSupportApplied {
		t.Fatalf("none applied %+v", none.end.ReasoningApplied)
	}

	low := runChat3Turn(t, mediator, subject, "effort-low", ask(&wire.ChatReasoning{Effort: wire.ChatReasoningEffortLow}))
	if low.end.Outcome != wire.ChatResultOutcomeCompleted || low.text != "Paris" {
		t.Fatalf("low: %s/%s text %q", low.end.Outcome, low.end.Cause, low.text)
	}
	if thought := strings.Count(low.reasoning, "think "); thought != 1024 || !strings.HasSuffix(low.reasoning, "directly now.\n") {
		t.Fatalf("low thought %d tokens, ending %q", thought, low.reasoning[max(0, len(low.reasoning)-40):])
	}
	if a := low.end.ReasoningApplied; a == nil || a.Effort == nil || *a.Effort != wire.ChatReasoningEffortLow || a.Support != wire.ChatReasoningSupportApplied || a.Budget != nil {
		t.Fatalf("low applied %+v", low.end.ReasoningApplied)
	}

	budget := int64(64)
	capped := runChat3Turn(t, mediator, subject, "effort-budget", ask(&wire.ChatReasoning{Effort: wire.ChatReasoningEffortHigh, Budget: &budget}))
	if thought := strings.Count(capped.reasoning, "think "); thought != 64 || capped.text != "Paris" {
		t.Fatalf("budget: thought %d tokens, text %q", thought, capped.text)
	}
	if a := capped.end.ReasoningApplied; a == nil || a.Budget == nil || *a.Budget != 64 || a.BudgetSupport == nil || *a.BudgetSupport != wire.ChatReasoningSupportApplied {
		t.Fatalf("budget applied %+v", capped.end.ReasoningApplied)
	}

	// The model's own switch is a native extension: required, it is
	// admitted and wins over the standard effort, which reads overridden.
	native3 := ask(&wire.ChatReasoning{Effort: wire.ChatReasoningEffortHigh})
	native3.Extensions = map[string]string{router.ExtensionLlamaCppEnableThinking: "false"}
	native3.RequiredExtensions = []string{router.ExtensionLlamaCppEnableThinking}
	off := runChat3Turn(t, mediator, subject, "native-off", native3)
	if off.end.Outcome != wire.ChatResultOutcomeCompleted || off.reasoning != "" {
		t.Fatalf("native off: %s/%s reasoning %q", off.end.Outcome, off.end.Cause, off.reasoning)
	}
	if a := off.end.ReasoningApplied; a == nil || a.Support != wire.ChatReasoningSupportOverridden {
		t.Fatalf("native off applied %+v", off.end.ReasoningApplied)
	}

	// An extension the model does not take: dropped and reported when
	// optional, refused naming it when required.
	foreign := ask(nil)
	foreign.Extensions = map[string]string{"acme/speed": "fast"}
	dropped := runChat3Turn(t, mediator, subject, "foreign-optional", foreign)
	if dropped.end.Outcome != wire.ChatResultOutcomeCompleted || len(dropped.end.DroppedExtensions) != 1 || dropped.end.DroppedExtensions[0] != "acme/speed" {
		t.Fatalf("optional foreign extension: %s/%s dropped %v", dropped.end.Outcome, dropped.end.Cause, dropped.end.DroppedExtensions)
	}
	foreign.RequiredExtensions = []string{"acme/speed"}
	window := mediator.GetChatHistoryWindow(subject)
	refused := mediator.StartV3(context.Background(), subject, wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "foreign-required", HistoryEpoch: window.Window.HistoryEpoch},
		Target: wire.ChatTarget{Model: "vision:4b"}, Content: foreign})
	if refused.Outcome != wire.ChatStartOutcomeUnsupported || refused.Reason != "extension:acme/speed" {
		t.Fatalf("required foreign extension: %+v", refused)
	}
}
