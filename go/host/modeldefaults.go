package host

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

// contentProver is optional on Source. Existing source implementations retain
// their behavior; a proof is current hashed evidence for the exact binding.
type contentProver interface {
	ContentProof(string, string) (content.Object, bool)
}

func (h *Host) modelEngine(model Model, object Object) Engine {
	return h.modelEngines(model, object)[0]
}

// modelEngines is model's launch for object, then the smaller launches the
// same curated model can run instead, largest first: the planned context in
// fewer resident slots, then each smaller per-conversation size in one slot.
// The holder loads the largest that fits what is free (resource RES-L9). A
// model without a context plan, or on a machine whose capacity is unknown,
// has one launch.
func (h *Host) modelEngines(model Model, object Object) []Engine {
	engine := h.options.Engine
	engine.Arguments = slices.Clone(engine.Arguments)
	engine.modelPin = nil
	engine.modelRecord = ""
	// An explicit private provider recipe (including inert test probes) keeps
	// its forecast. Production chat engines always receive a data profile.
	profiled := engine.cache == nil
	if !engine.Embeddings && profiled {
		cache := profileCache("default", engine.Arguments, 0)
		engine.cache = &cache
	}
	if model.Content != nil {
		engine.modelPin = &programfiles.Placed{Size: model.Content.Size, SHA256: model.Content.SHA256,
			Owners: modelplace.FolderSearch(filepath.Dir(object.Path)).Owners()}
		id := sha256.Sum256([]byte(model.Store + "/" + model.Object))
		engine.modelRecord = filepath.Join(filepath.Dir(engine.Record), "models", fmt.Sprintf("gguf-%x.json", id[:]))
	}
	if engine.Embeddings {
		return []Engine{engine}
	}
	entry, curated := h.curatedEntry(model, object)
	if !curated {
		return []Engine{engine}
	}
	plan := h.contextPlan(entry, profiled)
	engines := []Engine{planEngine(engine, entry, plan, profiled)}
	// The plan with the smaller RAM prompt cache comes before any smaller
	// context, and every smaller plan keeps that cache.
	if lower, ok := smallerCache(engines[0], entry.Family, profiled); ok {
		engines = append(engines, lower)
	}
	for _, smaller := range smallerPlans(entry.Family, plan) {
		e := planEngine(engine, entry, smaller, profiled)
		if lower, ok := smallerCache(e, entry.Family, profiled); ok {
			e = lower
		}
		engines = append(engines, e)
	}
	return engines
}

// curatedEntry is the catalogue entry object is, by current hashed content
// evidence: its digest and size. False for a source without evidence, stale
// evidence, or a file no entry names.
func (h *Host) curatedEntry(model Model, object Object) (modelwire.FitEntry, bool) {
	prover, ok := h.options.Source.(contentProver)
	if !ok {
		return modelwire.FitEntry{}, false
	}
	proof, current := prover.ContentProof(model.Store, model.Object)
	if !current || proof.Evidence != content.EvidenceHashed || proof.Partial || proof.Modified == "" || proof.Size != object.Size || proof.Locator != object.Locator {
		return modelwire.FitEntry{}, false
	}
	// The existing model-owned parser returns curated entries even when a
	// zero pool fits none. This is an in-process metadata read, with no IPC.
	fit := modelservice.FitCatalog(context.Background(), modelwire.CatalogUseCode, modelservice.CatalogSources{
		Pool:   func(context.Context) (int64, error) { return 0, nil },
		Served: func(context.Context, string) ([]modelservice.ServedFamily, error) { return nil, nil },
	})
	for _, entry := range fit.Entries {
		if entry.Digest == proof.Digest && entry.DownloadBytes == proof.Size {
			return entry, true
		}
	}
	return modelwire.FitEntry{}, false
}

// planEngine is engine launched at plan: its context arguments, its cache
// profile for the plan's slots and the plan it records.
func planEngine(engine Engine, entry modelwire.FitEntry, plan enginePlan, profiled bool) Engine {
	engine.Arguments = planArguments(slices.Clone(engine.Arguments), entry, plan)
	if plan.family != "" {
		engine.contextLoad = &plan
	}
	if profiled {
		resident := residentPlanSlots(entry.Family, plan, engine.Arguments)
		cache := profileCache(entry.Family, engine.Arguments, resident)
		cache.residentSlots = resident
		engine.cache = &cache
	}
	return engine
}

// smallerCache is engine with its family's smaller RAM prompt cache, for a
// launch whose free memory does not hold the full cache. ok is false when
// the profile names none.
func smallerCache(engine Engine, family string, profiled bool) (Engine, bool) {
	ram := engineProfiles[family].RAMMinMiB
	if !profiled || engine.cache == nil || ram <= 0 || ram >= engine.cache.ramMiB {
		return engine, false
	}
	cache := engine.cache.withRAM(ram)
	engine.cache = &cache
	return engine, true
}

// smallerPlans are the loads below plan that keep the model serving, largest
// first: plan's context in each fewer resident slot, then each smaller size
// the machine fits in one slot. Each carries its own memory need. A plan
// without a family, or without a known need, has none.
func smallerPlans(family string, plan enginePlan) []enginePlan {
	if plan.family == "" || plan.need <= 0 {
		return nil
	}
	var out []enginePlan
	add := func(context int64, slots int) {
		need, ok := modelservice.ContextNeed(family, context, slots)
		if !ok || need >= plan.need {
			return
		}
		smaller := enginePlan{context: context, slots: slots, family: family, need: need}
		for _, size := range plan.sizes {
			if size < context {
				smaller.smaller = size
			}
		}
		smaller.sizes = plan.sizes
		out = append(out, smaller)
	}
	for slots := plan.slots - 1; slots >= 1; slots-- {
		add(plan.context, slots)
	}
	for i := len(plan.sizes) - 1; i >= 0; i-- {
		if size := plan.sizes[i]; size < plan.context {
			add(size, 1)
		}
	}
	return out
}

// enginePlan is a curated model's context plan as the adapter maps it:
// context tokens for each of slots resident conversations. Slots zero keeps
// the engine's automatic slots with context as the whole window.
type enginePlan struct {
	context int64
	slots   int
	// family names the planned family, empty for no plan; smaller is the
	// next smaller per-conversation size that fits, zero for none.
	family  string
	smaller int64
	// need is the model owner's memory for this plan: weights, the KV of
	// context × slots tokens, the startup residual and the margin
	// (modelservice.ContextNeed). The load leases it. Zero when unknown.
	need int64
	// sizes are the per-conversation sizes the machine fits, smallest first.
	sizes []int64
}

// contextPlan resolves entry's plan when its engine profile declares how the
// engine splits context across slots (decision D, 2026-11-08): the model
// owner's PlanContext from this machine's memory and the person's chosen
// size. A private recipe or a profile without slot_context keeps the use's
// context in the engine's automatic slots.
func (h *Host) contextPlan(entry modelwire.FitEntry, profiled bool) enginePlan {
	profile, ok := engineProfiles[entry.Family]
	if !profiled || !ok || profile.SlotContext == "" {
		return enginePlan{context: entry.Context}
	}
	var capacity, chosen int64
	if h.options.Capacity != nil {
		capacity = h.options.Capacity(context.Background())
	}
	if h.options.ContextChoice != nil {
		chosen = h.options.ContextChoice(context.Background(), entry.Family)
	}
	plan, ok := modelservice.PlanContext(entry.Family, capacity, chosen)
	if !ok {
		return enginePlan{context: entry.Context}
	}
	// A load that did not fit lowers this family's context until the
	// person's choice changes (fitFallback).
	if limit := h.contextLimit(entry.Family); limit > 0 && plan.Context > limit {
		if lowered, ok := modelservice.PlanContext(entry.Family, capacity, limit); ok && lowered.Context <= limit {
			plan = lowered
		}
	}
	out := enginePlan{context: plan.Context, slots: plan.Slots, family: entry.Family, need: plan.NeedBytes, sizes: plan.Sizes}
	if need, ok := modelservice.ContextNeed(entry.Family, out.context, max(out.slots, 1)); ok {
		out.need = need
	}
	for _, size := range plan.Sizes {
		if size < plan.Context {
			out.smaller = size
		}
	}
	return out
}

// engineWindow is the engine's whole context for plan under the family's
// slot_context: each divided slot holds an equal share.
func engineWindow(family string, plan enginePlan) (int64, bool) {
	if plan.slots <= 0 || engineProfiles[family].SlotContext != slotContextDivided {
		return plan.context, plan.context > 0
	}
	if plan.context <= 0 || plan.context > math.MaxInt64/int64(plan.slots) {
		return 0, false
	}
	return plan.context * int64(plan.slots), true
}

func curatedModelArguments(configured []string, entry modelwire.FitEntry) []string {
	h := &Host{}
	return planArguments(configured, entry, h.contextPlan(entry, true))
}

// planArguments maps plan onto llama.cpp's --ctx-size and --parallel. An
// operator's explicit --parallel, --ctx-size or --kv-unified that disagrees
// with the plan keeps the operator's whole choice and drops the resident
// slots.
func planArguments(configured []string, entry modelwire.FitEntry, plan enginePlan) []string {
	args, err := EngineArguments(configured)
	if err != nil {
		return slices.Clone(configured)
	}
	flags := map[string]bool{}
	values := map[string]string{}
	for i := 0; i < len(args); i++ {
		flags[args[i]] = true
		if engineFlags[args[i]].value != nil {
			values[args[i]] = args[i+1]
			i++
		}
	}
	window, ok := engineWindow(entry.Family, plan)
	resident := ok && plan.slots > 0
	if flags["--parallel"] && values["--parallel"] != strconv.Itoa(plan.slots) {
		resident = false
	}
	if flags["--kv-unified"] {
		resident = false
	}
	if resident && flags["--ctx-size"] && values["--ctx-size"] != strconv.FormatInt(window, 10) {
		resident = false
	}
	if !resident {
		window = entry.Context
	}
	if window > 0 && !flags["--ctx-size"] {
		args = append(args, "--ctx-size", strconv.FormatInt(window, 10))
	}
	if resident && !flags["--parallel"] {
		args = append(args, "--parallel", strconv.Itoa(plan.slots))
	}
	// Provider-owned optimization for the catalogue's reviewed MTP weights
	// (first-run DECISION section 2, measured 2026-10-06). No store supplies
	// flags. An explicit speculation mode keeps its entire operator choice.
	if entry.Family == "qwen3.6-35b-a3b" && !flags["--spec-type"] {
		args = append(args, "--spec-type", "draft-mtp")
		if !flags["--spec-draft-n-max"] {
			args = append(args, "--spec-draft-n-max", "3")
		}
	}
	return args
}

// residentPlanSlots is plan's slots when the final arguments carry exactly
// its --parallel and --ctx-size pair without unified KV, and zero otherwise.
// Only that pair activates provider slot affinity.
func residentPlanSlots(family string, plan enginePlan, configured []string) int {
	window, ok := engineWindow(family, plan)
	if plan.slots <= 0 || !ok {
		return 0
	}
	args, err := EngineArguments(configured)
	if err != nil {
		return 0
	}
	values := map[string]string{}
	for i := 0; i < len(args); i++ {
		if args[i] == "--kv-unified" {
			return 0
		}
		if engineFlags[args[i]].value != nil {
			values[args[i]] = args[i+1]
			i++
		}
	}
	if values["--parallel"] != strconv.Itoa(plan.slots) || values["--ctx-size"] != strconv.FormatInt(window, 10) {
		return 0
	}
	return plan.slots
}
