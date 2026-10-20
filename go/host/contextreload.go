package host

import (
	"context"
	"slices"
	"sync"
	"time"
)

// contextState is the model host's side of decision D's reload and fit
// fallback (2026-11-08). loaded is the context plan each loaded name runs;
// stale names a loaded model whose family's chosen context changed, unloaded
// once no request uses it so the next request loads at the new size; limit is
// a family's per-conversation ceiling after a load at the planned size did not
// fit, kept until the person's choice for that family changes.
type contextState struct {
	mu     sync.Mutex
	loaded map[string]enginePlan
	stale  map[string]bool
	limit  map[string]int64
}

// recordContextLoad keeps the plan name's current launch runs.
func (h *Host) recordContextLoad(name string, plan *enginePlan) {
	c := &h.context
	c.mu.Lock()
	defer c.mu.Unlock()
	if plan == nil {
		delete(c.loaded, name)
		return
	}
	if c.loaded == nil {
		c.loaded = map[string]enginePlan{}
	}
	c.loaded[name] = *plan
}

// LoadedContexts are the per-conversation context each loaded curated model
// runs now: the plan its engine started with. The model host states it on
// Describe beside the resident names. The runtime reports that actual context
// to clients (MODELHOST-H10 and ROUTE-D14). Residency comes from the holder's
// resident list, which takes no component lock: a load holds its component
// lock until the engine is ready, and the runtime's two-second readiness probe
// must finish without waiting for the load (R28 incident in commit 6aa7581b2f).
func (h *Host) LoadedContexts() map[string]int64 {
	resident := map[string]bool{}
	for _, name := range h.Residents() {
		resident[name] = true
	}
	c := &h.context
	c.mu.Lock()
	defer c.mu.Unlock()
	out := map[string]int64{}
	for name, plan := range c.loaded {
		if plan.family != "" && plan.context > 0 && resident[name] {
			out[name] = plan.context
		}
	}
	return out
}

func (h *Host) contextLimit(family string) int64 {
	c := &h.context
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.limit[family]
}

// fitFallback lowers name's family to the next smaller size that fits after
// its load at the planned size did not fit, and reports whether one exists.
// It applies once: the lowered load records a plan with its own smaller size,
// and the caller retries a single time.
func (h *Host) fitFallback(name string) bool {
	c := &h.context
	c.mu.Lock()
	plan, ok := c.loaded[name]
	if !ok || plan.family == "" || plan.smaller <= 0 {
		c.mu.Unlock()
		return false
	}
	if c.limit == nil {
		c.limit = map[string]int64{}
	}
	c.limit[plan.family] = plan.smaller
	c.mu.Unlock()
	if h.loadLogger != nil {
		h.loadLogger.Warn("model context does not fit; retrying smaller", "family", plan.family, "model", name,
			"context", plan.context, "slots", plan.slots, "retry_context", plan.smaller)
	}
	return true
}

// ContextChanged tells the host that the person's context for family
// changed. The family's fit fallback ends, and each loaded model of that
// family is unloaded once no request uses it; the next request loads it at
// the new size.
func (h *Host) ContextChanged(family string) {
	c := &h.context
	c.mu.Lock()
	delete(c.limit, family)
	for name, plan := range c.loaded {
		if plan.family == family && h.Loaded(name) {
			if c.stale == nil {
				c.stale = map[string]bool{}
			}
			c.stale[name] = true
		}
	}
	c.mu.Unlock()
	h.applyContextChanges()
}

// applyContextChanges unloads each stale model that no request uses and
// keeps the others for the next pass.
func (h *Host) applyContextChanges() {
	c := &h.context
	c.mu.Lock()
	names := make([]string, 0, len(c.stale))
	for name := range c.stale {
		names = append(names, name)
	}
	c.mu.Unlock()
	slices.Sort(names)
	for _, name := range names {
		unloaded, busy := h.resources.UnloadIdle(name, "context changed")
		if busy {
			continue
		}
		c.mu.Lock()
		delete(c.stale, name)
		c.mu.Unlock()
		if unloaded && h.loadLogger != nil {
			h.loadLogger.Info("model context changed; unloaded for the next load", "model", name)
		}
	}
}

// contextLoop applies pending context changes each renew interval, which is
// how a model in use at the change unloads once its request ends.
func (h *Host) contextLoop(ctx context.Context) {
	every := h.options.Renew
	if every <= 0 {
		every = DefaultRenew
	}
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.applyContextChanges()
		}
	}
}
