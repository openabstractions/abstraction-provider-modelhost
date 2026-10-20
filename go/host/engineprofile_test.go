package host

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/holder"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

func profileCatalog(use modelwire.CatalogUse) modelwire.FitResult {
	return modelservice.FitCatalog(context.Background(), use, modelservice.CatalogSources{Pool: func(context.Context) (int64, error) { return 0, nil }, Served: func(context.Context, string) ([]modelservice.ServedFamily, error) { return nil, nil }})
}

// The production profile keeps 32k per slot in two non-unified resident
// slots. Transport, placement and MTP stay under the existing launch owner.
func TestEngineProfileProductionArgumentsGolden(t *testing.T) {
	testEngineProfileArgumentsGolden(t, true)
}

// A profile without resident slots keeps the engine's automatic slots.
func TestEngineProfileAutomaticSlotsArgumentsGolden(t *testing.T) {
	testEngineProfileArgumentsGolden(t, false)
}

func TestEngineProfileProductionKeepsTwoResidentSlots(t *testing.T) {
	p := engineProfiles["qwen3.6-35b-a3b"]
	if p.SlotContext != slotContextDivided || measuredSlots("qwen3.6-35b-a3b") != 2 {
		t.Fatalf("production resident profile %+v", p)
	}
}

// measuredSlots is the conversation slots family's plan keeps when the
// machine's memory is unknown: the measured startup profile.
func measuredSlots(family string) int {
	plan, _ := modelservice.PlanContext(family, 0, 0)
	return plan.Slots
}

// The adapter maps the model owner's plan onto llama.cpp flags from profile
// data: divided slots multiply --ctx-size by --parallel.
func TestEngineProfileAdapterFollowsContextPlan(t *testing.T) {
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	model := Model{Store: "fixture", Object: "weights"}
	object := Object{Locator: "weights.gguf", Size: entry.DownloadBytes}
	proof := content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}
	value := func(args []string, flag string) string {
		for i := 0; i+1 < len(args); i++ {
			if args[i] == flag {
				return args[i+1]
			}
		}
		return ""
	}
	// One checkpoint is 64 MiB plus 2080 bytes for each token of its slot,
	// and a slot's checkpoints stay within 4 GiB at its full context.
	for _, tc := range []struct {
		name                string
		capacity            int64
		chosen              int64
		ctx, parallel       string
		resident, cache     int
		checkpoints, ramMiB string
	}{
		// A 128 GB gfx1151 machine: Automatic is the trained 262144 tokens in
		// each of two slots. A 584-MiB checkpoint keeps 7 under the cap.
		{"automatic-large", 128 << 30, 0, "524288", "2", 2, 1536 + 7*2*584, "7", "1536"},
		// 262144 fits once and not twice: raising context drops to one slot.
		{"chosen-one-slot", 36 << 30, 262144, "262144", "1", 1, 1536 + 7*584, "7", "1536"},
		{"chosen-two-slots", 36 << 30, 131072, "262144", "2", 2, 1536 + 12*2*324, "12", "1536"},
		// Unknown memory keeps the measured profile.
		{"unknown-memory", 0, 0, "65536", "2", 2, 1536 + 16*2*129, "16", "1536"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var asked string
			h := &Host{options: Options{Source: defaultProofSource{proof: proof, current: true},
				Capacity:      func(context.Context) int64 { return tc.capacity },
				ContextChoice: func(_ context.Context, family string) int64 { asked = family; return tc.chosen }}}
			engine := h.modelEngine(model, object)
			if asked != entry.Family {
				t.Fatalf("choice read for %q", asked)
			}
			if got := value(engine.Arguments, "--ctx-size"); got != tc.ctx {
				t.Fatalf("--ctx-size %s, want %s: %v", got, tc.ctx, engine.Arguments)
			}
			if got := value(engine.Arguments, "--parallel"); got != tc.parallel {
				t.Fatalf("--parallel %s, want %s: %v", got, tc.parallel, engine.Arguments)
			}
			recipe := engine.cacheRecipe()
			if got := recipe.arguments(); value(got, "--ctx-checkpoints") != tc.checkpoints || value(got, "--cache-ram") != tc.ramMiB {
				t.Fatalf("cache arguments %v, want %s checkpoints and %s MiB", got, tc.checkpoints, tc.ramMiB)
			}
			if recipe.residentSlots != tc.resident {
				t.Fatalf("resident slots %d, want %d", recipe.residentSlots, tc.resident)
			}
			amount, err := recipe.leaseBytes(object.Size)
			if err != nil || amount != object.Size+int64(tc.cache)<<20 {
				t.Fatalf("lease %d err=%v, want cache %d MiB", amount, err, tc.cache)
			}
		})
	}
}

func disableResidentFixture(t *testing.T) {
	t.Helper()
	const family = "qwen3.6-35b-a3b"
	original := engineProfiles[family]
	automatic := original
	automatic.SlotContext = ""
	engineProfiles[family] = automatic
	t.Cleanup(func() { engineProfiles[family] = original })
}

func testEngineProfileArgumentsGolden(t *testing.T, resident bool) {
	requireEnginePeer(t)
	if !resident {
		disableResidentFixture(t)
	}
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	argsFile := filepath.Join(t.TempDir(), "arguments")
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.cache = nil
		o.Engine.fixture = []string{"--fake-args-file", argsFile}
		original := o.Source.(fakeSource)
		object := original.objects["fake/tiny"]
		object.Size = entry.DownloadBytes
		original.objects["fake/tiny"] = object
		o.Source = defaultProofSource{Source: original, current: true, proof: content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}}
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(raw), "\n")[1:]
	endpoint, ok := h.EngineEndpoint(h.hosts["tiny:1b"])
	if !ok {
		t.Fatal("no engine endpoint")
	}
	object, err := h.options.Source.Resolve("fake", "tiny", h.options.Engine.accepts())
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--model", object.Path, "--host", endpoint.Path, "--no-webui", "--alias", "tiny:1b", "--no-mmap", "--ctx-size", "32768"}
	if resident {
		want[len(want)-1] = "65536"
		want = append(want, "--parallel", "2")
	}
	want = append(want, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--cache-ram", "1536", "--ctx-checkpoints", "16")
	if resident {
		want = append(want, "--slot-prompt-similarity", "0")
	}
	want = append(want, "--fake-args-file", argsFile)
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("production arguments\n got %q\nwant %q", args, want)
	}
	slots := h.engines[h.hosts["tiny:1b"]].cacheSlots
	if resident && (slots == nil || len(slots.slots) != 2) {
		t.Fatal("resident profile did not activate two provider slots")
	}
	if !resident && slots != nil {
		t.Fatal("automatic-slot profile pinned provider slots")
	}
}

func TestEngineProfileSelectionAndForecast(t *testing.T) {
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	model := Model{Store: "fixture", Object: "weights"}
	object := Object{Locator: "weights.gguf", Size: entry.DownloadBytes}
	proof := content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}
	for _, tc := range []struct {
		name       string
		source     Source
		embeddings bool
		args       []string
		mib        int64
	}{
		// Curated: 1536 MiB cache plus 16 checkpoints of 64 MiB and 2080
		// bytes per token of each slot's window. Unknown: the default
		// profile's 1024 MiB and two flat 128-MiB checkpoints per slot.
		{"catalogue", defaultProofSource{proof: proof, current: true}, false, nil, 1536 + 16*2*129},
		{"unknown", legacyDefaultSource{}, false, nil, 1024 + 2*4*128},
		{"embedding", defaultProofSource{proof: proof, current: true}, true, nil, 0},
		{"parallel", defaultProofSource{proof: proof, current: true}, false, []string{"-np", "8"}, 1536 + 16*8*73},
		{"explicit-one", defaultProofSource{proof: proof, current: true}, false, []string{"--parallel", "1"}, 1536 + 16*129},
		{"unknown-explicit-two", legacyDefaultSource{}, false, []string{"--parallel", "2"}, 1024 + 2*2*128},
		{"automatic", defaultProofSource{proof: proof, current: true}, false, []string{"--parallel", "-1"}, 1536 + 16*4*129},
		{"explicit-two", defaultProofSource{proof: proof, current: true}, false, []string{"--parallel", "2"}, 1536 + 16*2*129},
		{"explicit-context", defaultProofSource{proof: proof, current: true}, false, []string{"--ctx-size", "4096"}, 1536 + 16*4*73},
		{"explicit-unified", defaultProofSource{proof: proof, current: true}, false, []string{"--kv-unified"}, 1536 + 16*4*129},
		{"explicit-pair", defaultProofSource{proof: proof, current: true}, false, []string{"--parallel", "2", "--ctx-size", "65536"}, 1536 + 16*2*129},
		{"different-pair", defaultProofSource{proof: proof, current: true}, false, []string{"--parallel", "2", "--ctx-size", "4096"}, 1536 + 16*2*69},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := &Host{options: Options{Source: tc.source, Engine: Engine{Embeddings: tc.embeddings, Arguments: tc.args}}}
			engine := h.modelEngine(model, object)
			recipe := engine.cacheRecipe()
			amount, err := recipe.leaseBytes(object.Size)
			if err != nil || amount != object.Size+tc.mib*(1<<20) {
				t.Fatalf("lease=%d err=%v recipe=%+v", amount, err, recipe)
			}
			want := []string{"--cache-ram", "1536", "--ctx-checkpoints", "16"}
			if _, unknown := tc.source.(legacyDefaultSource); unknown {
				want = []string{"--cache-ram", "1024", "--ctx-checkpoints", "2"}
			}
			if tc.embeddings {
				want = []string{"--cache-ram", "0", "--ctx-checkpoints", "0"}
			}
			wantResident := 0
			if tc.name == "catalogue" || tc.name == "explicit-two" || tc.name == "explicit-pair" {
				wantResident = 2
				// A pinned slot never loads another partition's parked prompt.
				want = append(want, "--slot-prompt-similarity", "0")
			}
			if !reflect.DeepEqual(recipe.arguments(), want) || recipe.slots != 0 {
				t.Fatalf("profile changed concurrency: %+v args=%v", recipe, recipe.arguments())
			}
			if err := recipe.checkArguments(engine.Arguments); err != nil {
				t.Fatal(err)
			}
			if recipe.residentSlots != wantResident {
				t.Fatalf("resident slots=%d want=%d args=%v", recipe.residentSlots, wantResident, engine.Arguments)
			}
		})
	}
	h := &Host{options: Options{Source: defaultProofSource{proof: proof, current: true}, Engine: Engine{cache: &engineCache{ramMiB: 256, checkpoints: 2, checkpointMiB: 64}}}}
	explicit := h.modelEngine(model, object)
	if strings.Contains(strings.Join(explicit.Arguments, " "), "--parallel") || explicit.cache.residentSlots != 0 {
		t.Fatal("private recipe gained resident policy", explicit.Arguments)
	}
}

// The b10825 checkpoint sizes on the owner's Qwen3.6 MTP slot, 2026-11-12:
// the recurrent snapshot plus the draft layer's KV of the tokens held. The
// forecast for a slot holding that many tokens covers each.
func TestEngineProfileCheckpointForecastCoversMeasuredSizes(t *testing.T) {
	profile := engineProfiles["qwen3.6-35b-a3b"]
	for _, m := range []struct {
		tokens int64
		mib    float64
	}{{388, 63.6}, {12276, 87.0}, {31241, 124.5}, {35800, 133.6}} {
		each, count := checkpointForecast(profile, m.tokens)
		if float64(each) < m.mib || count != profile.Checkpoints {
			t.Errorf("%d tokens: forecast %d MiB x %d, measured %.1f MiB", m.tokens, each, count, m.mib)
		}
	}
	// The cap holds a slot's checkpoints at the trained context.
	each, count := checkpointForecast(profile, 262144)
	if each != 584 || count != 7 || each*count > profile.CheckpointCapMiB {
		t.Fatalf("262144-token slot: %d MiB x %d", each, count)
	}
	// An unknown window forecasts the cap for one checkpoint.
	if each, count := checkpointForecast(profile, 0); each != profile.CheckpointCapMiB || count != 1 {
		t.Fatalf("unknown window: %d MiB x %d", each, count)
	}
}

// Every engine profile names a curated family.
func TestEngineProfileFamiliesAreCatalogued(t *testing.T) {
	seen := map[string]bool{}
	for _, use := range []modelwire.CatalogUse{modelwire.CatalogUseCode, modelwire.CatalogUseChat} {
		for _, entry := range profileCatalog(use).Entries {
			seen[entry.Family] = true
		}
	}
	for family := range engineProfiles {
		if family != "default" && !seen[family] {
			t.Fatalf("profile has no catalogue entry: %s", family)
		}
	}
}

// Independent startup oracle from the b10825 Windows Vulkan 32k-per-slot,
// MTP-n3, two-non-unified-resident-slot run of 2026-11-05 (model digest
// 55983c5a...). Buffer lines print two decimal MiB: add <.01 MiB uncertainty,
// then round positive values up to whole MiB. Repeated draft reserve lines
// describe the same buffer. This sums observed buffers, not the catalogue's
// private Need formula.
func TestEngineProfileStartupWithinFitNeed(t *testing.T) {
	const family = "qwen3.6-35b-a3b"
	if measuredSlots(family) != 2 {
		t.Fatal("startup facts need a new slot measurement")
	}
	weightsMiB := int64(21270 + 516)
	kvMiB := int64(1280 + 128) // Main KV across both slots plus one draft KV.
	recurrentMiB := int64(503)
	computeMiB := int64(114 + 41 + 77 + 41) // Target and draft, once each.
	outputMiB := int64(2 + 2)
	observedMiB := weightsMiB + kvMiB + recurrentMiB + computeMiB + outputMiB
	for _, use := range []modelwire.CatalogUse{modelwire.CatalogUseCode, modelwire.CatalogUseChat} {
		found := false
		for _, entry := range profileCatalog(use).Entries {
			if entry.Family != family {
				continue
			}
			found = true
			if entry.Context != 32768 || engineProfiles[family].AutoSlots != 4 {
				t.Fatal("startup facts need a new context/slot measurement")
			}
			// The prompt cache and checkpoints are leased beside the need.
			need := observedMiB*(1<<20) + (1 << 30)
			if need > entry.NeedBytes {
				t.Errorf("%s observed startup+margin=%d exceeds Fit Need=%d", use, need, entry.NeedBytes)
			}
		}
		if !found {
			t.Fatalf("recorded family absent for %s", use)
		}
	}
}

func TestEngineProfileComponentAmountAndFailure(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine.cache = nil })
	var weight, charged int64
	var leased atomic.Bool
	component, err := h.component(context.Background(), "tiny:1b", h.options.Models[0], &weight, &charged, &leased)()
	want := int64(4096) + 2048*(1<<20)
	if err != nil || component.Amount != want || charged != want || weight != 4096 {
		t.Fatalf("component=%+v weight=%d charged=%d err=%v", component, weight, charged, err)
	}
	if leased.Load() {
		t.Fatal("component preparation started engine")
	}
	failure := &holder.Failure{Stage: holder.StageAcquire, Result: res.AcquireResult{Outcome: res.AcquireOutcomeInsufficient}}
	refusal := h.mapFailure("tiny:1b", charged, failure)
	if WordOf(refusal) != Insufficient || !strings.Contains(refusal.Error(), fmt.Sprint(want)+"-byte") {
		t.Fatalf("charged refusal=%v", refusal)
	}
	if n, _, _ := leases.counts(); n != 0 {
		t.Fatalf("component acquired %d leases", n)
	}
}
