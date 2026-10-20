package host

import (
	"context"
	"errors"
	"github.com/openabstractions/abstraction-resource/go/holder"
	"sync/atomic"
	"time"
)

// modelLoad owns one bounded attempt independently of its callers' waits.
// The holder owns all process, lease, readiness and yield supervision.
type modelLoad struct {
	done     chan struct{}
	err      error
	waiters  atomic.Int32
	retry    bool
	progress *sharedLoadProgress
	bytes    int64
	started  bool
	readyAt  time.Time
}

// LoadTiming is the Ensure caller's own wait, measured by the host clock. Load
// classification is present when the holder proved readiness during this wait
// or found an existing residency.
// This owner has no page-cache, adoption/lending or cache-restore observation.
type LoadTiming struct {
	Milliseconds int64
	State        string
	Bytes        *int64
}

type loadTimingKey struct{}

// EnsureMeasured preserves Ensure's loading/lifetime owner and returns its
// observations without another resolution, process or resource operation.
func (h *Host) EnsureMeasured(ctx context.Context, name string) (string, LoadTiming, error) {
	return h.ensureMeasured(ctx, name, false)
}

// EnsureSpeculativeMeasured uses the same no-yield loading owner as
// EnsureSpeculative and preserves its observations through execution preparation.
func (h *Host) EnsureSpeculativeMeasured(ctx context.Context, name string) (string, LoadTiming, error) {
	return h.ensureMeasured(ctx, name, true)
}

func (h *Host) ensureMeasured(ctx context.Context, name string, speculative bool) (string, LoadTiming, error) {
	started := h.options.Now()
	var timing LoadTiming
	base, err := h.ensureShared(context.WithValue(ctx, loadTimingKey{}, &timing), name, speculative)
	if elapsed := h.options.Now().Sub(started); elapsed >= 0 {
		timing.Milliseconds = elapsed.Milliseconds()
	}
	return base, timing, err
}

const loadFailureRetry = 30 * time.Second

type modelLoadFailure struct {
	err   error
	until time.Time
}

func (h *Host) logLoad(name, phase, word string, elapsed time.Duration) {
	if h.loadLogger != nil {
		h.loadLogger.Info("model load", "family", name, "phase", phase, "duration_ms", elapsed.Milliseconds(), "result", word)
	}
}

func (h *Host) ensureShared(ctx context.Context, name string, speculative bool) (string, error) {
	waitStarted := time.Now()
	model, ok := h.allow[name]
	if !ok {
		return "", refuse(Unavailable, "no model called %q is hosted here", name)
	}
	if ctx.Err() != nil {
		return "", refuse(Unavailable, "%v", ctx.Err())
	}
	// A request waits here while the model yields at its request boundary
	// (resource RES-L3); a preload never waits behind a yield.
	if !speculative {
		if err := h.resources.Pass(ctx, name); err != nil {
			return "", refuse(Unavailable, "%v", err)
		}
	}
	standing := 0
	if speculative {
		standing = 1
	}
	h.loadMu.Lock()
	if h.loadClosed {
		h.loadMu.Unlock()
		return "", refuse(Unavailable, "model host closed")
	}
	if h.loadCtx == nil {
		h.loadCtx, h.loadCancel = context.WithCancel(context.Background())
		h.loads = make(map[string][2]*modelLoad)
		h.loadFailures = make(map[string]modelLoadFailure)
	}
	if failed, ok := h.loadFailures[name]; ok {
		if h.options.Now().Before(failed.until) {
			h.loadMu.Unlock()
			return "", failed.err
		}
		delete(h.loadFailures, name)
	}
	pair := h.loads[name]
	load := pair[standing]
	created := load == nil
	if load == nil {
		progress := &sharedLoadProgress{}
		if other := pair[1-standing]; other != nil {
			progress = other.progress
		}
		load = &modelLoad{done: make(chan struct{}), progress: progress}
		load.waiters.Add(1)
		pair[standing] = load
		h.loads[name] = pair
		owner, cancel := context.WithCancel(h.loadCtx)
		owner = context.WithValue(owner, modelLoadKey{}, load)
		h.loadWG.Add(1)
		go func() {
			defer h.loadWG.Done()
			defer cancel()
			var amount, charged int64
			var leased atomic.Bool
			started := time.Now()
			h.logLoad(name, "start", "started", 0)
			ensure := func(ctx context.Context, name string, prepare func() (holder.Component, error)) error {
				return h.resources.EnsureWithWaiter(ctx, name, prepare, func() bool { return load.waiters.Load() > 0 })
			}
			if speculative {
				ensure = h.resources.EnsureSpeculative
			}
			err := ensure(owner, name, h.component(owner, name, model, &amount, &charged, &leased))
			if err != nil && !speculative {
				if mapped := h.mapFailure(name, charged, err); WordOf(mapped) == EngineDoesNotFit && h.fitFallback(name) {
					// One retry at the next smaller context (decision D).
					leased.Store(false)
					err = ensure(owner, name, h.component(owner, name, model, &amount, &charged, &leased))
				}
			}
			retry := errors.Is(err, holder.ErrSpeculativeCancelled)
			var failure *holder.Failure
			if speculative && errors.As(err, &failure) && failure.Word() == holder.WordHeld {
				err = refuse(resourcesHeld, "loading %s would ask a holder of %s to yield", name, h.options.Resource)
			} else if err != nil {
				err = h.mapFailure(name, charged, err)
			}
			phase, word := "ready", "ready"
			if err != nil {
				phase = "failed"
				word = WordOf(err)
				if word == "" {
					word = Unavailable
				}
			}
			h.logLoad(name, phase, word, time.Since(started))
			h.loadMu.Lock()
			load.err = err
			load.bytes = amount
			load.retry = retry
			// The immutable Host is rebuilt for declaration/pin changes. A
			// bounded retry keeps failed engine starts/readiness from repeatedly
			// reacquiring the card. Pre-lease refusals can be retried at once.
			if err != nil && leased.Load() && !retry {
				h.loadFailures[name] = modelLoadFailure{err: err, until: h.options.Now().Add(loadFailureRetry)}
			} else if err == nil {
				delete(h.loadFailures, name)
			}
			pair := h.loads[name]
			pair[standing] = nil
			if pair[0] == nil && pair[1] == nil {
				delete(h.loads, name)
			} else {
				h.loads[name] = pair
			}
			close(load.done)
			h.loadMu.Unlock()
		}()
	}
	if !created {
		load.waiters.Add(1)
	}
	h.loadMu.Unlock()
	defer load.waiters.Add(-1)
	var phases <-chan uint64
	var unsubscribe func()
	callback, _ := ctx.Value(loadProgressCallbackKey{}).(func(string))
	if callback != nil {
		phases, unsubscribe = load.progress.subscribe()
		defer unsubscribe()
	}
	// Separate standing waiters preserve the holder's promotion of a
	// speculative load when real work arrives. Its component lock still
	// permits only one actual engine start across both paths.
	for {
		select {
		case phase := <-phases:
			if ctx.Err() == nil {
				callback(loadPhaseName(phase))
			}
		case <-load.done:
			// Drain genuine phases already observed before readiness completed.
			for phases != nil && ctx.Err() == nil {
				select {
				case phase := <-phases:
					callback(loadPhaseName(phase))
				default:
					phases = nil
				}
			}
			if load.retry && !speculative && ctx.Err() == nil {
				if unsubscribe != nil {
					unsubscribe()
				}
				return h.ensureShared(ctx, name, speculative)
			}
			if timing, ok := ctx.Value(loadTimingKey{}).(*LoadTiming); ok {
				if load.err == nil && ctx.Err() == nil {
					switch {
					case !load.started:
						timing.State = "loaded"
					case load.readyAt.After(waitStarted):
						timing.State = "cold"
					case !load.readyAt.IsZero():
						timing.State = "loaded"
					}
				}
				if load.bytes > 0 {
					value := load.bytes
					timing.Bytes = &value
				}
			}
			if load.err != nil {
				return "", load.err
			}
			return h.bases[name], nil
		case <-ctx.Done():
			return "", refuse("engine:loading", "%s is loading; try again shortly", name)
		}
	}
}
