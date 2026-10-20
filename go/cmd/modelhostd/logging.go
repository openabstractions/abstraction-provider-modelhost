package main

import (
	"log/slog"
	"math"
	"os"
	"regexp"

	inference "github.com/openabstractions/abstraction-inference/go"
	logging "github.com/openabstractions/abstraction-logging/go"
)

func resolvedDiagnostics(program string) (slog.Handler, func()) {
	return logging.ResolveDiagnostics(program, os.Stderr, logging.Resolve)
}

var cacheOperationName = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)

// The existing operation record owns these facts. This diagnostic consumer
// deliberately selects cache fields rather than serializing the source record.
//
// slot says where the request ran: "kept" is the slot that last served the
// same conversation and program, "changed" any other slot. It says nothing
// about the engine's reuse; reused_tokens and prefill_tokens do, from the
// engine's own timings, and reuse names their share: "most" when the engine
// reused more of the prompt than it read, "some", or "none".
func cacheObservationRecord(logger *slog.Logger) func(inference.Record) {
	return func(record inference.Record) {
		if logger == nil || !cacheOperationName.MatchString(record.Operation) || record.Timing == nil {
			return
		}
		timing := record.Timing
		switch timing.Restore {
		case "hit", "miss", "refused":
		default:
			return
		}
		switch timing.RestoreCause {
		case "resident", "not_resident", "unlabelled", "program_change", "missing_partition", "request_failed":
		default:
			return
		}
		slot := map[string]string{"hit": "kept", "miss": "changed", "refused": "refused"}[timing.Restore]
		attrs := []any{"operation", record.Operation, "slot", slot, "slot_cause", timing.RestoreCause}
		if read, reused := timing.PrefillTokens, timing.ReusedTokens; read != nil && reused != nil && *read >= 0 && *reused >= 0 {
			reuse := "none"
			if *reused > *read {
				reuse = "most"
			} else if *reused > 0 {
				reuse = "some"
			}
			attrs = append(attrs, "prefill_tokens", *read, "reused_tokens", *reused, "reuse", reuse)
		}
		if n := timing.CacheEvictions; n != nil && *n >= 0 && *n <= 1 {
			attrs = append(attrs, "cache_evictions", *n)
		}
		if n := timing.PrefillMS; n != nil && *n >= 0 {
			attrs = append(attrs, "prefill_ms", *n)
		}
		if n := timing.Engine.PromptMS; n != nil && *n >= 0 && !math.IsNaN(*n) && !math.IsInf(*n, 0) {
			attrs = append(attrs, "engine_prompt_ms", *n)
		}
		logger.Info("cache observation", attrs...)
	}
}
