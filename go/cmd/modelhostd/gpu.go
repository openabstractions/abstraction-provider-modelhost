package main

import (
	"io"
	"log/slog"

	logging "github.com/openabstractions/abstraction-logging/go"
	"github.com/openabstractions/abstraction-provider-kit/go/quarantine"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	"github.com/openabstractions/abstraction-resource/go/holder"
)

// gpuQuarantineFile is the quarantine's file in the provider's state.
const gpuQuarantineFile = gpufault.QuarantineFile

// newGPUGuard is serve's GPU fault handling (resource CONTRACT.md
// RES-G1..RES-G7, MODELHOST-G1): the resource kit's provider composition
// (gpufault.ForProvider) with the wake lease through leases@1 and OA
// logging.
func newGPUGuard(state string, leases holder.Leases, stderr io.Writer, report func(error)) (*gpufault.Guard, error) {
	return gpufault.ForProvider(gpufault.ProviderOptions{Name: "modelhostd", State: state,
		Wake: gpufault.LeaseWake(leases, 0, report), Logger: slog.New(logging.LegacyDefault("modelhostd")),
		Stderr: stderr, Report: report})
}

// runGPU lists or clears the quarantine (the provider kit's quarantine).
func runGPU(args []string, stdout, stderr io.Writer) int {
	return quarantine.Run("modelhostd", args, stateFolder, usage, stdout, stderr)
}
