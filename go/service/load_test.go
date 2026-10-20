package service

import (
	"context"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"reflect"
	"testing"
	"time"
)

func TestExecutionProgressProjectsDeclaredPhaseChanges(t *testing.T) {
	var phases []string
	report := projectExecutionProgress(func(phase string) {
		declared := false
		for _, member := range wire.ChatProgressPhaseValues() {
			declared = declared || member == wire.ChatProgressPhase(phase)
		}
		if !declared {
			t.Fatalf("undeclared chat progress: %q", phase)
		}
		event := wire.ChatEvent{Kind: wire.ChatEventKindProgress, Progress: &wire.ChatProgress{Phase: wire.ChatProgressPhase(phase)}}
		raw, err := wire.ToJSON(event)
		if err != nil {
			t.Fatal(err)
		}
		var decoded wire.ChatEvent
		if err := wire.FromJSON(raw, &decoded); err != nil || decoded.Progress == nil || string(decoded.Progress.Phase) != phase {
			t.Fatalf("progress wire: %s %v", raw, err)
		}
		phases = append(phases, phase)
	})
	for _, phase := range []string{"metadata", "metadata", "tensors", "heartbeat", "context", "graph", "queued"} {
		report(phase)
	}
	if !reflect.DeepEqual(phases, []string{"loading", "warming"}) {
		t.Fatalf("chat phases: %v", phases)
	}
	var warm []string
	_ = projectExecutionProgress(func(phase string) { warm = append(warm, phase) })
	if len(warm) != 0 {
		t.Fatal("warm request invented loading", warm)
	}
}

func TestLoadingAdmissionAnswersBeforeTheStartClientBudget(t *testing.T) {
	started := time.Now()
	word, ok := admitLoad(context.Background(), "chosen", func(ctx context.Context, name string) (string, error) {
		if name != "chosen" {
			t.Fatal(name)
		}
		<-ctx.Done()
		return "", &host.Refusal{Word: "engine:loading"}
	})
	if ok || word != "engine:loading" || time.Since(started) > 2*time.Second {
		t.Fatalf("admission %q %v after %s", word, ok, time.Since(started))
	}
}
