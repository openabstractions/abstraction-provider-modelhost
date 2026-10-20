package main

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"time"

	configwire "github.com/openabstractions/abstraction-config/go/abstraction/config"
	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
	resource "github.com/openabstractions/abstraction-resource/go/client"
	rights "github.com/openabstractions/abstraction-rights/go/abstraction/rights/api"
)

// modelContextKey is the configuration key holding the person's
// per-conversation context for each local model family, as decimal tokens.
// An absent family is Automatic (decision D, 2026-11-08). The declaration
// owner names the same key on the chat facet.
const modelContextKey = declaration.ContextKey

// contextReadBudget bounds each load's reads of memory and the person's
// choice; a slow or absent service reads as unknown and Automatic.
const contextReadBudget = 2 * time.Second

// capacityReader is the resource table read the context plan needs.
type capacityReader interface {
	HoldersContext(context.Context, string, bool) (resource.ResourceState, error)
}

// machineCapacity answers the bytes the model's resource pool holds: the
// capacity the resource table reports for it, which is the pool the lease
// book admits against (resource CONTRACT.md RES-L2). The model host computes
// no memory figure of its own. Zero means the table did not answer.
func machineCapacity(table capacityReader, name string) func(context.Context) int64 {
	return func(ctx context.Context) int64 {
		ctx, cancel := context.WithTimeout(ctx, contextReadBudget)
		defer cancel()
		state, err := table.HoldersContext(ctx, name, false)
		if err != nil {
			return 0
		}
		return max(state.Capacity, 0)
	}
}

// typedReader is the configuration read the context plan needs.
type typedReader interface {
	ReadTypedContext(context.Context, []string) (configwire.ReadResult, error)
}

// contextChoice answers the person's per-conversation context for a family
// from model_context, and zero for Automatic. The model host is OA's own
// program: its read never asks the person, and a read that fails or is
// refused is logged before the load runs at Automatic.
func contextChoice(config typedReader, logf func(error)) func(context.Context, string) int64 {
	return func(ctx context.Context, family string) int64 {
		ctx, cancel := context.WithTimeout(ctx, contextReadBudget)
		defer cancel()
		result, err := config.ReadTypedContext(ctx, []string{modelContextKey})
		if err != nil {
			logf(fmt.Errorf("model_context read failed; %s loads at Automatic: %w", family, err))
			return 0
		}
		values, err := modelContextValues(result)
		if err != nil {
			logf(fmt.Errorf("%w; %s loads at Automatic", err, family))
			return 0
		}
		if raw, ok := values[family]; ok {
			tokens, err := strconv.ParseInt(raw, 10, 64)
			if err != nil || tokens <= 0 {
				logf(fmt.Errorf("model_context for %s is %q, which is no token count; it loads at Automatic", family, raw))
				return 0
			}
			return tokens
		}
		return 0
	}
}

// modelContextValues is model_context's family-to-tokens map from a typed
// read, or an error naming a refusal or an outcome other than read.
func modelContextValues(result configwire.ReadResult) (map[string]string, error) {
	if result.Outcome != configwire.ReadOutcomeRead {
		return nil, fmt.Errorf("model_context read answered %s%s", result.Outcome, refusalText(result.Refusal))
	}
	for _, refused := range result.Refused {
		if refused.Key == modelContextKey {
			return nil, fmt.Errorf("model_context read refused: %s%s", refused.Reason, refusalText(refused.Refusal))
		}
	}
	for _, entry := range result.Entries {
		if entry.Key == modelContextKey && entry.Value != nil && entry.Value.Kind == configwire.ValueKindTextMap {
			return entry.Value.TextMap, nil
		}
	}
	return map[string]string{}, nil
}

func refusalText(detail *rights.RefusalDetail) string {
	if detail == nil {
		return ""
	}
	return fmt.Sprintf(" (%s %s %s)", detail.Source, detail.Decision, detail.Action)
}

// typedObserver is the configuration observation the reload watch needs.
type typedObserver interface {
	ObserveTypedContext(context.Context, string, int64) (configwire.ConfigObservationV2, error)
}

// contextWatchWait is one observation's wait. A failed or refused
// observation pauses contextWatchBackoff, doubling up to
// contextWatchBackoffMax while failures repeat.
var (
	contextWatchWait       = 25 * time.Second
	contextWatchBackoff    = 30 * time.Second
	contextWatchBackoffMax = 10 * time.Minute
)

// contextWatchPause waits d unless ctx ends first; false means ctx ended.
var contextWatchPause = func(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// watchModelContext tells changed about each family whose model_context
// value differs from the last snapshot, until ctx ends (decision D's reload,
// 2026-11-08). The first snapshot is the baseline. A failed or refused
// observation is asked again after a pause that doubles up to its bound;
// a failure is logged when its text differs from the one before, and the
// first good answer after failures is logged once. An unchanged answer
// after a refused snapshot is that refusal standing, so it recovers nothing.
// Before this, the log said the watch resumed while model_context was still
// refused, and a refusal read as fixed half a minute later.
func watchModelContext(ctx context.Context, config typedObserver, changed func(string), logf func(error)) {
	cursor := ""
	var last map[string]string
	baseline := false
	backoff := contextWatchBackoff
	logged := ""
	failures := 0
	refused := false
	fail := func(err error) bool {
		failures++
		if text := err.Error(); text != logged {
			logged = text
			logf(err)
		}
		pause := backoff
		backoff = min(backoff*2, contextWatchBackoffMax)
		return contextWatchPause(ctx, pause)
	}
	recovered := func() {
		if failures > 0 {
			logf(fmt.Errorf("model_context watch resumed after %d failed attempts", failures))
		}
		backoff, logged, failures = contextWatchBackoff, "", 0
	}
	for ctx.Err() == nil {
		observation, err := config.ObserveTypedContext(ctx, cursor, contextWatchWait.Milliseconds())
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			if !fail(fmt.Errorf("model_context watch failed: %w", err)) {
				return
			}
			continue
		}
		switch observation.Outcome {
		case configwire.ObservationOutcomeV2Unchanged:
			cursor = observation.Cursor
			if !refused {
				recovered()
			}
			continue
		case configwire.ObservationOutcomeV2Changed, configwire.ObservationOutcomeV2Gap:
		default:
			cursor = ""
			if !fail(fmt.Errorf("model_context watch answered %s%s", observation.Outcome, refusalText(observation.Refusal))) {
				return
			}
			continue
		}
		cursor = observation.Cursor
		if observation.Snapshot == nil {
			continue
		}
		values, err := modelContextValues(*observation.Snapshot)
		refused = err != nil
		if err != nil {
			if !fail(fmt.Errorf("model_context watch: %w", err)) {
				return
			}
			continue
		}
		recovered()
		if baseline {
			for _, family := range changedFamilies(last, values) {
				changed(family)
			}
		}
		last, baseline = values, true
	}
}

// changedFamilies are the families whose value differs between two
// model_context maps, sorted.
func changedFamilies(before, after map[string]string) []string {
	var out []string
	for family, value := range after {
		if before[family] != value {
			out = append(out, family)
		}
	}
	for family := range before {
		if _, ok := after[family]; !ok {
			out = append(out, family)
		}
	}
	slices.Sort(out)
	return out
}
