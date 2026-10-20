package host

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"time"
)

//go:embed engineprofile.json
var engineProfileJSON []byte

type engineProfile struct {
	// RAMMiB is llama.cpp's --cache-ram prompt cache: conversations parked
	// out of a slot, and the previous version of an edited one. RAMMinMiB is
	// the smaller cache a launch takes when free memory does not hold
	// RAMMiB; zero keeps RAMMiB.
	RAMMiB    int `json:"cache_ram_mib"`
	RAMMinMiB int `json:"cache_ram_min_mib"`
	// Checkpoints is --ctx-checkpoints, the context checkpoints each slot
	// keeps. A hybrid model's recurrent memory rewinds only to one of them,
	// so an edited message re-reads from the newest checkpoint before it.
	Checkpoints int `json:"ctx_checkpoints"`
	// One checkpoint takes CheckpointMiB plus CheckpointBytesPerToken for
	// each token the slot holds (b10825 Qwen3.6 MTP measured 62.8 MiB +
	// 2.02 KiB/token). CheckpointCapMiB bounds one slot's checkpoints at its
	// full context; the launch lowers the count to fit it.
	CheckpointMiB           int `json:"checkpoint_mib"`
	CheckpointBytesPerToken int `json:"checkpoint_bytes_per_token"`
	CheckpointCapMiB        int `json:"checkpoint_cap_mib"`
	AutoSlots               int `json:"auto_slots"`
	// SlotContext says how the engine splits its context across resident
	// conversation slots. "divided": llama.cpp gives each non-unified
	// --parallel slot an equal share of --ctx-size. Empty: the family keeps
	// the engine's automatic slots and no context plan applies.
	SlotContext     string               `json:"slot_context"`
	RequestSettings requestSettingsByUse `json:"request_settings"`
	// IdleMinutes is how long a resident of this family stays loaded after
	// its last use (MODELHOST-H8). Zero keeps the declared --idle.
	IdleMinutes int `json:"idle_minutes"`
	// AnswerReserveTokens is the least of a request's output limit a
	// thinking model keeps for its answer; see reserveAnswer. Zero leaves
	// reasoning unbounded within the limit.
	AnswerReserveTokens int `json:"answer_reserve_tokens"`
	// ReasoningBudgetMessage closes reasoning that the reserve cut short.
	// llama.cpp writes it into the reasoning before the end-of-thinking tag,
	// so the model starts its answer rather than finishing the cut thought
	// in the answer text. A profile with a reserve names one.
	ReasoningBudgetMessage string `json:"reasoning_budget_message"`
	// ReasoningBudgets is the reasoning budget in tokens of each effort
	// word (low, medium, high), and DefaultReasoningEffort the word a
	// request that names none gets; see reserveAnswer. A profile with a
	// reserve names both, and reasoning stays bounded.
	ReasoningBudgets       map[string]int `json:"reasoning_budgets"`
	DefaultReasoningEffort string         `json:"default_reasoning_effort"`
}

// slotContextDivided is the slot_context word for llama.cpp's non-unified
// slots, each holding --ctx-size divided by --parallel tokens.
const slotContextDivided = "divided"

type requestSettings struct {
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`
}

type requestSettingsByUse map[string]requestSettings

var engineProfiles = func() map[string]engineProfile {
	var profiles map[string]engineProfile
	if err := json.Unmarshal(engineProfileJSON, &profiles); err != nil {
		panic(err)
	}
	if _, ok := profiles["default"]; !ok {
		panic("modelhost: no default engine profile")
	}
	for _, p := range profiles {
		if p.RAMMiB <= 0 || p.RAMMiB > 1<<20 || p.RAMMinMiB < 0 || p.RAMMinMiB > p.RAMMiB || p.Checkpoints <= 0 || p.Checkpoints > 256 ||
			p.CheckpointMiB <= 0 || p.CheckpointMiB > 1<<20 || p.CheckpointBytesPerToken < 0 || p.CheckpointBytesPerToken > 1<<20 ||
			p.CheckpointCapMiB < 0 || p.CheckpointCapMiB > 1<<20 || p.CheckpointBytesPerToken > 0 && p.CheckpointCapMiB < p.CheckpointMiB || p.AutoSlots <= 0 || (p.SlotContext != "" && p.SlotContext != slotContextDivided) ||
			p.IdleMinutes < 0 || p.IdleMinutes > 24*60 || p.AnswerReserveTokens < 0 || p.AnswerReserveTokens > 1<<20 ||
			p.AnswerReserveTokens > 0 && p.ReasoningBudgetMessage == "" || len(p.ReasoningBudgetMessage) > 1024 ||
			p.AnswerReserveTokens > 0 && p.ReasoningBudgets[p.DefaultReasoningEffort] <= 0 {
			panic("modelhost: unbounded engine profile")
		}
		for effort, budget := range p.ReasoningBudgets {
			if effort == "" || effort == "none" || budget <= 0 || budget > 1<<20 {
				panic("modelhost: unbounded engine profile")
			}
		}
	}
	return profiles
}()

// profileCache is family's cache recipe for configured arguments. resident
// is the conversation slots its context plan keeps, zero for the engine's
// automatic slots.
func profileCache(family string, configured []string, resident int) engineCache {
	p, ok := engineProfiles[family]
	if !ok {
		p = engineProfiles["default"]
	}
	slots := p.AutoSlots
	if resident > 0 {
		slots = resident
	}
	var window int64
	divided := false
	if args, err := EngineArguments(configured); err == nil {
		hasContext, hasParallel, unified := false, false, slices.Contains(args, "--kv-unified")
		for i := 0; i+1 < len(args); i++ {
			if args[i] == "--ctx-size" {
				hasContext = true
				window, _ = strconv.ParseInt(args[i+1], 10, 64)
			}
			if args[i] == "--parallel" {
				hasParallel = true
				if n, err := strconv.Atoi(args[i+1]); err == nil {
					if n > 0 {
						slots = n
						divided = !unified
					} else {
						slots = p.AutoSlots
					}
				}
			}
		}
		if hasContext && !hasParallel {
			slots = p.AutoSlots
		}
	}
	c := engineCache{ramMiB: p.RAMMiB, checkpoints: p.Checkpoints, requestSettings: p.RequestSettings,
		idle: time.Duration(p.IdleMinutes) * time.Minute, answerReserve: answerReserve{tokens: p.AnswerReserveTokens, message: p.ReasoningBudgetMessage,
			budgets: p.ReasoningBudgets, effort: p.DefaultReasoningEffort}}
	// A divided slot holds its share of --ctx-size; an automatic or unified
	// slot may grow to the whole window.
	if p.SlotContext == slotContextDivided && divided {
		window /= int64(slots)
	}
	each, count := checkpointForecast(p, window)
	c.checkpoints = count
	if slots > math.MaxInt/each {
		c.checkpointMiB = -1
	} else {
		c.checkpointMiB = slots * each
	}
	return c
}

// checkpointForecast is one checkpoint's MiB in a slot of window tokens and
// the count the slot keeps under the profile's byte cap. An unknown window
// forecasts the cap itself for each checkpoint.
func checkpointForecast(p engineProfile, window int64) (each, count int) {
	if p.CheckpointBytesPerToken == 0 {
		return p.CheckpointMiB, p.Checkpoints
	}
	each = p.CheckpointCapMiB
	if window > 0 && window <= int64(p.CheckpointCapMiB)<<20/int64(p.CheckpointBytesPerToken) {
		each = min(p.CheckpointMiB+int((window*int64(p.CheckpointBytesPerToken)+(1<<20)-1)>>20), p.CheckpointCapMiB)
	}
	return each, max(1, min(p.Checkpoints, p.CheckpointCapMiB/each))
}

// withRAM is c with the prompt cache at mib, the smaller launch a
// profile's cache_ram_min_mib names.
func (c engineCache) withRAM(mib int) engineCache {
	c.ramMiB = mib
	return c
}

// engineCache is set only by a provider recipe after its memory is admitted.
// slotPath is used by the disposable live knowledge probe; persisted cache
// storage remains a separate owner.
type engineCache struct {
	ramMiB, checkpoints, reuse int
	// checkpointMiB is the recipe's forecast per checkpoint across all slots.
	// A count without a byte forecast cannot enter a resource lease.
	checkpointMiB int
	slots         int
	// residentSlots activates provider affinity only for the admitted profile
	// argument pair. The explicit private recipe's slots field stays separate.
	residentSlots   int
	requestSettings requestSettingsByUse
	answerReserve   answerReserve
	slotPath        string
	// idle is the family's own idle time, zero for the declared --idle
	// (MODELHOST-H8).
	idle time.Duration
}

func (c engineCache) arguments() []string {
	args := []string{"--cache-ram", strconv.Itoa(c.ramMiB), "--ctx-checkpoints", strconv.Itoa(c.checkpoints)}
	if c.pinsSlots() {
		// Each request names its slot (cacheSlots). llama-server b10825 then
		// loads a prompt from the one RAM cache all programs share whenever
		// slot prompt similarity is on, as it is by default; at 0 a named
		// slot never loads another partition's parked prompt.
		args = append(args, "--slot-prompt-similarity", "0")
	}
	if c.reuse > 0 {
		args = append(args, "--cache-reuse", strconv.Itoa(c.reuse))
	}
	if c.slotPath != "" {
		args = append(args, "--slot-save-path", c.slotPath)
	}
	if c.slots > 0 {
		args = append(args, "--parallel", strconv.Itoa(c.slots))
	}
	return args
}

// pinsSlots reports that the engine gets a cacheSlots table, so every chat
// request names its slot and cache partition.
func (c engineCache) pinsSlots() bool {
	return max(c.slots, c.residentSlots) > 0 && c.ramMiB > 0
}

func (c engineCache) leaseBytes(weights int64) (int64, error) {
	if weights <= 0 {
		return 0, fmt.Errorf("modelhost: cache recipe has no bounded memory forecast")
	}
	return c.addTo(weights)
}

// addTo is base plus the recipe's cache memory: the RAM prompt cache and
// every slot's checkpoints. A curated plan's need covers weights and KV, and
// the lease adds this to it.
func (c engineCache) addTo(base int64) (int64, error) {
	if base < 0 || c.ramMiB < 0 || c.checkpoints < 0 || c.checkpointMiB < 0 || c.slots < 0 || c.checkpoints > 0 && c.checkpointMiB == 0 {
		return 0, fmt.Errorf("modelhost: cache recipe has no bounded memory forecast")
	}
	// Check each operation before multiplication, including on 32-bit hosts.
	ram, count, checkpoint := int64(c.ramMiB), int64(c.checkpoints), int64(c.checkpointMiB)
	if count > 0 && checkpoint > (math.MaxInt64-ram)/count {
		return 0, fmt.Errorf("modelhost: cache recipe memory overflows")
	}
	mib := ram + count*checkpoint
	if mib > (math.MaxInt64-base)/(1<<20) {
		return 0, fmt.Errorf("modelhost: cache recipe memory overflows")
	}
	return base + mib*(1<<20), nil
}

// A nil recipe selects the provider's default chat profile. Embeddings use
// an empty recipe; engine built-in defaults never add unaccounted memory.
func (e Engine) cacheRecipe() engineCache {
	if e.Embeddings {
		return engineCache{}
	}
	if e.cache != nil {
		return *e.cache
	}
	return profileCache("default", e.Arguments, 0)
}

func (c engineCache) checkArguments(args []string) error {
	if c.slots == 0 {
		return nil
	}
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--parallel" && args[i+1] != strconv.Itoa(c.slots) {
			return fmt.Errorf("modelhost: cache recipe does not forecast the configured parallel slots")
		}
	}
	return nil
}
