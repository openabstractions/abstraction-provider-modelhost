package host

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	res "github.com/openabstractions/abstraction-resource/go/client"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

// sizedBook is the fake lease book whose no-yield acquire grants only what
// is free beside another engine's lease (resource RES-L6, RES-L9).
type sizedBook struct {
	*fakeLeases
	free  int64
	asked []int64
}

func (b *sizedBook) AcquireFree(resource string, amount int64) (res.AcquireFreeResult, error) {
	b.mu.Lock()
	b.asked = append(b.asked, amount)
	fits := amount <= b.free
	b.mu.Unlock()
	if !fits {
		return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeHeld, Asked: []res.YieldRecord{}}, nil
	}
	granted, err := b.Acquire(resource, amount, 0)
	return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeAcquired, Lease: granted.Lease, Asked: []res.YieldRecord{}}, err
}

// sizedHost is the curated Qwen3.6 35B on the owner's Strix Halo, whose card
// the resource table reports at 102.7 GiB, with free bytes left beside a
// picture engine. It returns the launch's --ctx-size and --parallel reader.
func sizedHost(t *testing.T, free int64, fixture ...string) (*Host, *sizedBook, func() (string, string)) {
	t.Helper()
	requireEnginePeer(t)
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	argsFile := filepath.Join(t.TempDir(), "arguments")
	book := &sizedBook{fakeLeases: newFakeLeases(), free: free}
	h := composed(t, book.fakeLeases, func(o *Options) {
		o.Leases = book
		o.Engine.cache = nil
		o.Engine.fixture = append([]string{"--fake-args-file", argsFile}, fixture...)
		original := o.Source.(fakeSource)
		object := original.objects["fake/tiny"]
		object.Size = entry.DownloadBytes
		original.objects["fake/tiny"] = object
		o.Source = defaultProofSource{Source: original, current: true, proof: content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}}
		o.Capacity = func(context.Context) int64 { return 110_324_465_664 } // 102.75 GiB, DXGI's figure
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
	return h, book, launched
}

func need(t *testing.T, context int64, slots int) int64 {
	t.Helper()
	bytes, ok := modelservice.ContextNeed(contextFamily, context, slots)
	if !ok {
		t.Fatalf("no need for %d x %d", context, slots)
	}
	return bytes
}

// lease is the plan's need plus its engine cache: ramMiB of RAM prompt
// cache and each slot's checkpoints, 64 MiB plus 2080 bytes per token of the
// slot, at most 16 and within 4 GiB at the slot's full context.
func lease(t *testing.T, context int64, slots int, ramMiB int64) int64 {
	t.Helper()
	each := 64 + (context*2080+(1<<20)-1)/(1<<20)
	count := min(16, 4096/each)
	return need(t, context, slots) + (ramMiB+count*int64(slots)*each)<<20
}

// With the card to itself the model loads its full plan, the trained 262,144
// tokens in each of two slots, and leases that plan's weights and KV plus
// its full prompt cache and checkpoints.
func TestChatLeasesItsPlanWhenAlone(t *testing.T) {
	h, book, launched := sizedHost(t, 1<<40)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if ctx, parallel := launched(); ctx != "524288" || parallel != "2" {
		t.Fatalf("launch --ctx-size %s --parallel %s", ctx, parallel)
	}
	if got, want := book.grants[0], lease(t, 262144, 2, 1536); got != want {
		t.Fatalf("lease %d, want the plan's need and cache %d", got, want)
	}
	if got := h.LoadedContexts()["tiny:1b"]; got != 262144 {
		t.Fatalf("reported context %d", got)
	}
}

// Beside a picture engine the model first keeps its plan with the smaller
// prompt cache, then drops a resident slot; with less room it takes the
// largest size that fits. The context it reports is the one it loaded, never
// the plan it could not get.
func TestChatTakesTheLargestContextThatFits(t *testing.T) {
	for _, tc := range []struct {
		name          string
		free          int64
		ctx, parallel string
		tokens        int64
		slots         int
		ramMiB        int64
	}{
		{"the smaller cache fits", 91 << 29, "524288", "2", 262144, 2, 1024},
		{"one slot fits", 40 << 30, "262144", "1", 262144, 1, 1024},
		{"a smaller size fits", 32 << 30, "65536", "1", 65536, 1, 1024},
		{"nothing fits: asks for the smallest", 20 << 30, "32768", "1", 32768, 1, 1024},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h, book, launched := sizedHost(t, tc.free)
			if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
				t.Fatal(err)
			}
			if ctx, parallel := launched(); ctx != tc.ctx || parallel != tc.parallel {
				t.Fatalf("launch --ctx-size %s --parallel %s, want %s %s", ctx, parallel, tc.ctx, tc.parallel)
			}
			if got, want := book.grants[len(book.grants)-1], lease(t, tc.tokens, tc.slots, tc.ramMiB); got != want {
				t.Fatalf("lease %d, want %d", got, want)
			}
			if got := h.LoadedContexts()["tiny:1b"]; got != tc.tokens {
				t.Fatalf("reported context %d, loaded %d per conversation", got, tc.tokens)
			}
			if len(book.asked) < 2 || book.asked[0] != lease(t, 262144, 2, 1536) {
				t.Fatalf("no-yield asks %v did not begin at the full plan", book.asked)
			}
		})
	}
}

// The owner's demo machine: Qwen beside LTX-2.5 (53 GiB) and the small
// holders the resource table showed on 2026-11-12 (dwm 1.17, Edge 1.41,
// WorkloadsSessionHost 0.68 GiB) on the 102.75-GiB card. Qwen keeps its full
// plan, two slots of 262,144 tokens with the whole cache, and both fit.
func TestChatFitsBesideTheVideoModel(t *testing.T) {
	const capacity = int64(110_324_465_664)
	others := int64(53<<30) + 3_500_405_637 // 3.26 GiB
	h, book, launched := sizedHost(t, capacity-others)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if ctx, parallel := launched(); ctx != "524288" || parallel != "2" {
		t.Fatalf("launch --ctx-size %s --parallel %s", ctx, parallel)
	}
	got := book.grants[len(book.grants)-1]
	if want := lease(t, 262144, 2, 1536); got != want || got+others > capacity {
		t.Fatalf("lease %d (want %d) beside %d exceeds %d", got, want, others, capacity)
	}
	t.Logf("Qwen lease %.2f GiB; with LTX and the small holders %.2f of %.2f GiB", float64(got)/(1<<30), float64(got+others)/(1<<30), float64(capacity)/(1<<30))
}
