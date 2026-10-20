package host

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

// contextHost is a host whose one model is the curated Qwen3.6 35B on a
// 128 GB machine, with the person's choice in chosen and the provider's
// request activity in active. It returns a reader of the last launch's
// --ctx-size and --parallel.
func contextHost(t *testing.T, chosen, active *atomic.Int64, fixture ...string) (*Host, func() (string, string)) {
	t.Helper()
	requireEnginePeer(t)
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	argsFile := filepath.Join(t.TempDir(), "arguments")
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.cache = nil
		o.Engine.fixture = append([]string{"--fake-args-file", argsFile}, fixture...)
		original := o.Source.(fakeSource)
		object := original.objects["fake/tiny"]
		object.Size = entry.DownloadBytes
		original.objects["fake/tiny"] = object
		o.Source = defaultProofSource{Source: original, current: true, proof: content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}}
		o.Capacity = func(context.Context) int64 { return 128 << 30 }
		o.ContextChoice = func(context.Context, string) int64 { return chosen.Load() }
		o.Active = func(string) bool { return active.Load() != 0 }
	})
	launched := func() (string, string) {
		raw, err := os.ReadFile(argsFile)
		if err != nil {
			t.Fatal(err)
		}
		args := strings.Split(string(raw), "\n")
		value := func(flag string) string {
			for i := 0; i+1 < len(args); i++ {
				if args[i] == flag {
					return args[i+1]
				}
			}
			return ""
		}
		return value("--ctx-size"), value("--parallel")
	}
	return h, launched
}

const contextFamily = "qwen3.6-35b-a3b"

// A changed model_context unloads an idle model at once, and the next request
// loads it with the new flags.
func TestContextChangeWhileIdleReloadsAtTheNewSize(t *testing.T) {
	var chosen, active atomic.Int64
	h, launched := contextHost(t, &chosen, &active)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if ctx, parallel := launched(); ctx != "524288" || parallel != "2" {
		t.Fatalf("automatic launch --ctx-size %s --parallel %s", ctx, parallel)
	}
	chosen.Store(65536)
	h.ContextChanged("other-family")
	if !h.Loaded("tiny:1b") {
		t.Fatal("another family's change unloaded the model")
	}
	h.ContextChanged(contextFamily)
	if h.Loaded("tiny:1b") {
		t.Fatal("an idle model stayed loaded after its context changed")
	}
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if ctx, parallel := launched(); ctx != "131072" || parallel != "2" {
		t.Fatalf("reload --ctx-size %s --parallel %s, want 65536 in each of two slots", ctx, parallel)
	}
}

// A change during a request unloads only after that request finishes.
func TestContextChangeDuringARequestWaitsForIt(t *testing.T) {
	var chosen, active atomic.Int64
	h, launched := contextHost(t, &chosen, &active)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	active.Store(1)
	chosen.Store(131072)
	h.ContextChanged(contextFamily)
	h.applyContextChanges()
	if !h.Loaded("tiny:1b") {
		t.Fatal("a model with a request in flight was unloaded")
	}
	active.Store(0)
	h.applyContextChanges()
	if h.Loaded("tiny:1b") {
		t.Fatal("the model stayed loaded after its request finished")
	}
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if ctx, parallel := launched(); ctx != "262144" || parallel != "2" {
		t.Fatalf("reload --ctx-size %s --parallel %s", ctx, parallel)
	}
}

// A load at the planned size that fails for lack of memory retries once at
// the next smaller size.
func TestContextFitFailureRetriesAtTheNextSmallerSize(t *testing.T) {
	var chosen, active atomic.Int64
	// The engine fails above 262144 tokens: the automatic 2 x 262144 does
	// not fit, and 2 x 131072 does.
	h, launched := contextHost(t, &chosen, &active, "--fake-fail-above-ctx", "262144")
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the smaller retry did not load: %v", err)
	}
	if ctx, parallel := launched(); ctx != "262144" || parallel != "2" {
		t.Fatalf("retry --ctx-size %s --parallel %s, want 131072 in each of two slots", ctx, parallel)
	}
	if got := h.contextLimit(contextFamily); got != 131072 {
		t.Fatalf("fallback limit %d", got)
	}
	// A new choice ends the fallback.
	h.ContextChanged(contextFamily)
	if got := h.contextLimit(contextFamily); got != 0 {
		t.Fatalf("fallback limit %d after a new choice", got)
	}
}

// A model already at its smallest size has no smaller retry.
func TestContextFitFailureAtTheSmallestSizeFails(t *testing.T) {
	var chosen, active atomic.Int64
	chosen.Store(32768)
	h, _ := contextHost(t, &chosen, &active, "--fake-fail-above-ctx", "1024")
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != EngineDoesNotFit {
		t.Fatalf("smallest size: %v, want %s", err, EngineDoesNotFit)
	}
}
