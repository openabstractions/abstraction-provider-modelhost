package service

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	res "github.com/openabstractions/abstraction-resource/go/client"
)

// AcquireFree is the no-yield acquire of a card nobody else holds, through
// which a preload loads (MODELHOST-W2, resource RES-L6).
func (g granting) AcquireFree(resource string, amount int64) (res.AcquireFreeResult, error) {
	return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeAcquired,
		Lease: &res.Lease{ID: "lease-1", Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}

// prefixRequest is a preload@1 request: a system message long enough to be
// worth evaluating, and no user message.
func prefixRequest(system string) wire.Request {
	return wire.Request{Model: "tiny:1b", Messages: []wire.Message{{Role: wire.RoleSystem,
		Parts: []wire.Part{{Kind: wire.PartKindText, Text: system}}}}}
}

// preload@1 end to end (CONTRACT.md MODELHOST-W1..W3, inference INF-PL1..PL4):
// the call loads the model under its lease and the engine evaluates the
// caller's own prefix; a chat call then uses the engine, and the next
// preload answers busy without reaching the engine; a prefix that ends with
// the user's message is invalid before anything loads.
func TestPreloadLoadsTheModelAndEvaluatesTheCallersPrefix(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the model host confirms its engine's listener through abstraction-identity's socket owner tables, which only Windows has; elsewhere every load is refused (MODELHOST-L7)")
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	transport, model := serveTransport(t, buildFakeEngine(t), identity.CanonicalProgramPath(filepath.Clean(self)))
	preloader, chatClient := wire.NewPreloaderClient(transport), wire.NewChatClient(transport)

	invalid := prefixRequest("short")
	invalid.Messages = append(invalid.Messages, chat().Messages...)
	if reply, err := preloader.Preload(invalid); err != nil || reply.Outcome != wire.PreloadOutcomeInvalid || model.Loaded("tiny:1b") {
		t.Fatalf("a prefix ending with the user's message: %+v %v; loaded %v", reply, err, model.Loaded("tiny:1b"))
	}

	system := strings.Repeat("You are a careful assistant on this computer. ", 12)
	reply, err := preloader.Preload(prefixRequest(system))
	if err != nil || reply.Outcome != wire.PreloadOutcomeEvaluated || reply.Model != "tiny:1b" || reply.Host == "" {
		t.Fatalf("preload %+v %v", reply, err)
	}
	// The stand-in's template is "[content]" per message and its tokens are
	// bytes, so the shared tokens are the system message and the opening
	// bracket of the user's.
	if want := int64(len("[" + system + "][")); reply.Tokens != want {
		t.Fatalf("the engine evaluated %d tokens, want %d", reply.Tokens, want)
	}
	if !model.Loaded("tiny:1b") {
		t.Fatal("the preload did not leave the model loaded")
	}

	admission, err := chatClient.Start(chat())
	if err != nil || admission.Outcome != wire.StartOutcomeAccepted {
		t.Fatalf("start %+v %v", admission, err)
	}
	for cursor := int64(0); ; {
		page, err := chatClient.Observe(admission.Operation, cursor, 256, 65536, 2000)
		if err != nil || page.Outcome != wire.PageOutcomePage {
			t.Fatalf("observe %+v %v", page, err)
		}
		cursor = page.Next
		if page.AtEnd {
			break
		}
	}
	if reply, err := preloader.Preload(prefixRequest(system)); err != nil || reply.Outcome != wire.PreloadOutcomeBusy || reply.Reason != "engine:used" || reply.Tokens != 0 {
		t.Fatalf("a preload after a turn answered %+v %v", reply, err)
	}
}
