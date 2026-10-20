package main

import (
	"io"
	"log/slog"

	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// engineExitReport writes each engine's exit, of every kind, as one OA
// logging record and one line on stderr (MODELHOST-E1, KIT-P3).
func engineExitReport(stderr io.Writer, logger *slog.Logger) func(host.EngineExit) {
	return engineproc.Report(logger, stderr, "modelhostd")
}
