package host

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"sync"
	"sync/atomic"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

type cacheSlot struct {
	scope     string
	partition string
	busy      bool
	used      uint64
	occupied  bool
	cost      time.Duration
	costKnown bool
	keep      float64
}

// cacheSlots belongs to one engine launch. In-flight requests keep their slot
// until the response body closes. Equal admitted resident-slot sizes reduce
// GreedyDual-Size to measured refill cost plus the inflation value; no per-state
// byte measurement is invented. No slot identity or plaintext is written to disk.
type cacheSlots struct {
	mu        sync.Mutex
	slots     []cacheSlot
	clock     uint64
	inflation float64
	changed   chan struct{}
}

type cacheSlotLease struct {
	id          int
	reused      bool
	release     func(bool)
	observe     func(time.Duration, bool)
	cachePrompt bool
	evicted     bool
	cause       string
}

func newCacheSlots(n int) *cacheSlots {
	return &cacheSlots{slots: make([]cacheSlot, n), changed: make(chan struct{})}
}

func (s *cacheSlots) acquire(ctx context.Context, scope string) (int, bool, func(bool), error) {
	lease, err := s.reserve(ctx, scope)
	if err != nil {
		return 0, false, nil, err
	}
	return lease.id, lease.reused, lease.release, nil
}

func (s *cacheSlots) reserve(ctx context.Context, scope string, partitions ...string) (*cacheSlotLease, error) {
	partition := ""
	if len(partitions) > 0 {
		partition = partitions[0]
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		s.mu.Lock()
		chosen, matching := -1, false
		for i, slot := range s.slots {
			if scope != "" && slot.scope == scope && slot.partition == partition {
				chosen, matching = i, true
				break
			}
			if !slot.busy && (chosen < 0 || cacheSlotLess(slot, s.slots[chosen], scope == "")) {
				chosen = i
			}
		}
		if chosen >= 0 && !s.slots[chosen].busy {
			s.clock++
			previous := s.slots[chosen]
			if !matching && previous.occupied {
				s.inflation = max(s.inflation, previous.keep)
			}
			current := cacheSlot{scope: scope, partition: partition, busy: true, used: s.clock, occupied: true, keep: s.inflation}
			if matching {
				current.cost, current.costKnown = previous.cost, previous.costKnown
				current.keep += float64(current.cost)
			}
			s.slots[chosen] = current
			epoch := s.clock
			s.mu.Unlock()
			var once sync.Once
			lease := &cacheSlotLease{id: chosen, reused: matching, cachePrompt: partition != "" && (!previous.occupied || previous.partition == partition), evicted: previous.scope != "" && !matching, cause: "not_resident"}
			switch {
			case partition == "":
				lease.cause = "missing_partition"
			case matching:
				lease.cause = "resident"
			case previous.occupied && previous.partition != partition:
				lease.cause = "program_change"
			case scope == "":
				lease.cause = "unlabelled"
			}
			lease.observe = func(elapsed time.Duration, miss bool) {
				if elapsed < 0 {
					return
				}
				s.mu.Lock()
				defer s.mu.Unlock()
				slot := &s.slots[chosen]
				if !slot.busy || slot.used != epoch {
					return
				}
				// Warm hits refresh recency with the saved refill cost; their
				// cheap latency never replaces a long conversation's cost.
				if miss {
					slot.cost, slot.costKnown = elapsed, true
				}
				slot.keep = s.inflation + float64(slot.cost)
			}
			lease.release = func(complete bool) {
				once.Do(func() {
					s.mu.Lock()
					defer s.mu.Unlock()
					s.slots[chosen].busy = false
					if !complete {
						s.slots[chosen].scope = ""
						s.slots[chosen].partition = ""
						s.slots[chosen].cost, s.slots[chosen].costKnown = 0, false
					}
					s.slots[chosen].keep = s.inflation + float64(s.slots[chosen].cost)
					close(s.changed)
					s.changed = make(chan struct{})
				})
			}
			return lease, nil
		}
		changed := s.changed
		s.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func cacheSlotLess(a, b cacheSlot, unlabelled bool) bool {
	if a.occupied != b.occupied {
		return !a.occupied
	}
	if unlabelled && (a.scope == "") != (b.scope == "") {
		return a.scope == ""
	}
	if a.keep != b.keep {
		return a.keep < b.keep
	}
	return a.used < b.used
}

func (h *Host) prepareCacheRequest(ctx context.Context, host, path string, raw []byte) ([]byte, func(bool), error) {
	scope, affinity, _ := inference.ExecutionCacheScope(ctx)
	partition := inference.ExecutionCachePartition(ctx)
	if !affinity || partition == "" {
		scope = ""
	}
	use, _ := inference.ExecutionUse(ctx)
	return h.prepareScopedCacheRequest(ctx, host, path, raw, scope, partition, use)
}

func (h *Host) prepareScopedCacheRequest(ctx context.Context, host, path string, raw []byte, scope, partition string, uses ...wire.ChatUse) ([]byte, func(bool), error) {
	if path != "/v1/chat/completions" {
		return raw, nil, nil
	}
	h.engineMu.Lock()
	p := h.engines[host]
	var slots *cacheSlots
	var settings requestSettings
	var reserve answerReserve
	if p != nil {
		slots = p.cacheSlots
		reserve = p.answerReserve
		if len(uses) != 0 {
			settings = p.requestSettings[uses[0].String()]
		}
	}
	h.engineMu.Unlock()
	if slots == nil {
		if len(settings.ChatTemplateKwargs) != 0 || reserve.tokens > 0 {
			var body map[string]any
			if err := json.Unmarshal(raw, &body); err != nil {
				return nil, nil, err
			}
			if len(settings.ChatTemplateKwargs) != 0 {
				mergeRequestSettings(body, settings)
			}
			reserveAnswer(body, reserve)
			out, err := encodeJSON(body)
			return out, nil, err
		}
		return raw, nil, nil
	}
	lease, err := slots.reserve(ctx, scope, partition)
	if err != nil {
		return nil, nil, err
	}
	var once sync.Once
	release := func(complete bool) {
		once.Do(func() {
			observation := inference.ExecutionObservation{Restore: "miss", Cause: lease.cause, Evicted: lease.evicted}
			if lease.reused {
				observation.Restore = "hit"
			}
			if !complete {
				observation.Restore, observation.Cause = "miss", "request_failed"
			}
			if cost := inference.ReportExecutionObservation(ctx, observation); complete && cost != nil && *cost <= int64((1<<63-1)/time.Millisecond) {
				lease.observe(time.Duration(*cost)*time.Millisecond, !lease.reused)
			}
			lease.release(complete)
		})
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		release(false)
		return nil, nil, err
	}
	body["id_slot"], body["cache_prompt"] = lease.id, lease.cachePrompt
	mergeRequestSettings(body, settings)
	reserveAnswer(body, reserve)
	out, err := encodeJSON(body)
	if err != nil {
		release(false)
		return nil, nil, err
	}
	return out, release, nil
}

// answerReserve is a thinking family's reasoning policy from its engine
// profile: the share of each output limit kept for the answer, the message
// that closes reasoning cut at its budget, the reasoning budget of each
// effort word, and the effort a request that names none gets.
type answerReserve struct {
	tokens  int
	message string
	budgets map[string]int
	effort  string
}

// reserveAnswer decides a thinking model's reasoning for one engine request,
// and is the one place that does: thinking off, or a reasoning budget in
// tokens. Its inputs are the request's reasoning_effort, its max_tokens and
// the family's policy; a later chat@3 reasoning option arrives here as
// reasoning_effort.
//
//   - reasoning_effort "none" turns thinking off (thinkingOff).
//   - A request that names its own reasoning_budget_tokens or
//     thinking_budget_tokens keeps it.
//   - Any other request gets the budget of its effort word, or of the
//     family's default effort when it names none or one the profile lacks:
//     reasoning is always bounded (architect ruling 2026-11-12 §3).
//   - A named max_tokens below twice the answer reserve turns thinking off:
//     the budget message alone is about twenty tokens, and at max_tokens 10
//     the forced message spent the whole limit and left a reasoning-only
//     result (R34 reproduction in commit 38fe32b174).
//   - A larger max_tokens keeps a quarter of the limit for the answer, at
//     least the reserve and at most half, and the budget fits in the rest.
//
// llama.cpp b10825 reads reasoning_budget_tokens per request and ends the
// reasoning there. The cut reasoning ends with reasoning_budget_message:
// without it the model finishes its interrupted thought after the
// end-of-thinking tag, and that thought becomes answer text. A model
// without a reasoning section ignores the budget and the switch.
func reserveAnswer(body map[string]any, reserve answerReserve) {
	if reserve.tokens <= 0 {
		return
	}
	effort, _ := body["reasoning_effort"].(string)
	if effort == "none" {
		thinkingOff(body)
		return
	}
	for _, name := range []string{"reasoning_budget_tokens", "thinking_budget_tokens"} {
		if _, set := body[name]; set {
			return
		}
	}
	budget, named := reserve.budgets[effort]
	if !named {
		budget = reserve.budgets[reserve.effort]
	}
	if limit, ok := body["max_tokens"].(float64); ok && limit >= 1 && limit <= 1<<30 && limit == math.Trunc(limit) {
		n := int(limit)
		if n < 2*reserve.tokens {
			thinkingOff(body)
			return
		}
		answer := min(max(n/4, reserve.tokens), n/2)
		if budget <= 0 || budget > n-answer {
			budget = n - answer
		}
	}
	if budget <= 0 {
		return
	}
	body["reasoning_budget_tokens"] = budget
	if _, set := body["reasoning_budget_message"]; !set && reserve.message != "" {
		body["reasoning_budget_message"] = reserve.message
	}
}

// thinkingOff asks the engine for an answer without a reasoning section:
// llama.cpp reads chat_template_kwargs enable_thinking false as its own
// switch, and a thinking template opens and closes an empty section.
func thinkingOff(body map[string]any) {
	kwargs, _ := body["chat_template_kwargs"].(map[string]any)
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	kwargs["enable_thinking"] = false
	body["chat_template_kwargs"] = kwargs
}

func mergeRequestSettings(body map[string]any, settings requestSettings) {
	kwargs, _ := body["chat_template_kwargs"].(map[string]any)
	if kwargs == nil {
		kwargs = map[string]any{}
	}
	for name, value := range settings.ChatTemplateKwargs {
		kwargs[name] = value
	}
	kwargs["preserve_thinking"] = true
	body["chat_template_kwargs"] = kwargs
}

type cacheResponseBody struct {
	io.ReadCloser
	release  func(bool)
	complete bool
	failed   atomic.Bool
	ctx      context.Context
}

func (b *cacheResponseBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil && err != io.EOF {
		b.failed.Store(true)
	}
	return n, err
}
func (b *cacheResponseBody) Close() error {
	err := b.ReadCloser.Close()
	b.release(b.complete && !b.failed.Load() && err == nil && b.ctx.Err() == nil)
	return err
}
