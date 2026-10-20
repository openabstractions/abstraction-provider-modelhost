package service

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	res "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"
)

type heldTitleLeases struct {
	granting
	normal atomic.Int32
	free   atomic.Int32
}

func (b *heldTitleLeases) Acquire(resource string, amount int64, wait time.Duration) (res.AcquireResult, error) {
	b.normal.Add(1)
	return b.granting.Acquire(resource, amount, wait)
}

func (b *heldTitleLeases) AcquireFree(string, int64) (res.AcquireFreeResult, error) {
	b.free.Add(1)
	return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeHeld, Asked: []res.YieldRecord{}}, nil
}

func TestFreeCapacityChat3UsesExistingModelhostAdmissionAndRefusesHeldBeforeEngine(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	book := &heldTitleLeases{granting: granting{released: make(chan string, 4)}}
	_, model := serveTransport(t, buildFakeEngine(t), identity.CanonicalProgramPath(filepath.Clean(self)), func(o *host.Options) { o.Leases = book })
	routes := router.New()
	routes.SetHosts(router.OpenAICompatible("modelhost-engine", model.Base("tiny:1b"), "/v1", model.Models()))
	config := ProviderConfig(model, routes, nil)
	provider, err := inference.New(config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := provider.Close(); err != nil {
			t.Error(err)
		}
	})
	// Exercise the actual ProviderConfig admission and declared-owner composition.
	// The separate native tests cover the real designated leaf and runtime relay.
	subject := inference.Subject{Account: "fixture-account", Program: self}
	window := provider.GetChatHistoryWindow(subject)
	if window.Window == nil {
		t.Fatalf("window=%+v", window)
	}
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "title-held", HistoryEpoch: window.Window.HistoryEpoch},
		Target: wire.ChatTarget{Model: "tiny:1b"}, Content: wire.ChatContent{
			Messages:   []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "title"}}}},
			Guarantees: []wire.ChatGuarantee{wire.ChatGuaranteeLocalOnly, wire.ChatGuaranteeFreeCapacity}}}
	admission := provider.StartV3(context.Background(), subject, request)
	if admission.Outcome != wire.ChatStartOutcomeUnavailable || admission.Reason != "resource:held" {
		t.Fatalf("admission=%+v", admission)
	}
	if book.normal.Load() != 0 || book.free.Load() != 1 || model.Loaded("tiny:1b") {
		t.Fatalf("normal=%d free=%d loaded=%v", book.normal.Load(), book.free.Load(), model.Loaded("tiny:1b"))
	}
}

func TestFreeCapacityColdChat3JoinsLoadingAndPreservesMainLease(t *testing.T) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	book := &nativeTitleBook{granting: granting{released: make(chan string, 8)}}
	engine := buildFakeEngine(t, "-ldflags", "-X main.defaultLoadTime=1500ms")
	_, model := serveTransport(t, engine, identity.CanonicalProgramPath(filepath.Clean(self)), func(o *host.Options) {
		o.Leases = book
		o.Source = titleObjects{o.Source.(oneObject)}
		o.Models = append(o.Models, host.Model{Name: "main:2b", Store: "fake", Object: "main"})
	})
	_, err = model.Ensure(context.Background(), "main:2b")
	if runtime.GOOS == "linux" {
		// The Linux launcher currently provides a direct child, without a
		// contained tree that can prove socket ownership. A load must fail
		// closed and release its acquisition before the title can join it.
		if host.WordOf(err) != host.EngineNetworkSocket || model.Loaded("main:2b") || book.normal.Load() != 1 || book.free.Load() != 0 {
			t.Fatal("unproven engine escaped socket admission", err, model.Loaded("main:2b"), book.normal.Load(), book.free.Load())
		}
		select {
		case released := <-book.released:
			if released == "" {
				t.Fatal("engine refusal released no lease identity")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("engine refusal leaked its lease")
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	mainLease := model.Lease("main:2b")
	if mainLease == "" || book.normal.Load() != 1 {
		t.Fatal("main residency was not established", mainLease, book.normal.Load())
	}
	book.normal.Store(0) // Count only the title request's normal acquisitions.
	routes := router.New()
	routes.SetHosts(router.OpenAICompatible("modelhost-engine", model.Base("tiny:1b"), "/v1", model.Models()))
	provider, err := inference.New(ProviderConfig(model, routes, nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := provider.Close(); err != nil {
			t.Error(err)
		}
	})
	subject := inference.Subject{Account: "fixture-account", Program: self}
	window := provider.GetChatHistoryWindow(subject)
	if window.Window == nil {
		t.Fatalf("window=%+v", window)
	}
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "title-cold", HistoryEpoch: window.Window.HistoryEpoch},
		Target: wire.ChatTarget{Model: "tiny:1b"}, Content: wire.ChatContent{
			Messages:   []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "title"}}}},
			Guarantees: []wire.ChatGuarantee{wire.ChatGuaranteeLocalOnly, wire.ChatGuaranteeFreeCapacity}}}
	admission := provider.StartV3(context.Background(), subject, request)
	if admission.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("cold admission=%+v", admission)
	}
	result := titleResult(t, provider, subject, admission.Request)
	if result.Outcome != wire.ChatResultOutcomeCompleted || result.Cause != "" || result.Refusal != nil {
		t.Fatalf("cold result=%+v", result)
	}
	if book.normal.Load() != 0 || book.free.Load() != 1 || model.Lease("main:2b") != mainLease || !model.Loaded("main:2b") {
		t.Fatalf("normal=%d free=%d main=%q/%q", book.normal.Load(), book.free.Load(), mainLease, model.Lease("main:2b"))
	}
	select {
	case released := <-book.released:
		t.Fatalf("title released a lease: %q", released)
	default:
	}
}
