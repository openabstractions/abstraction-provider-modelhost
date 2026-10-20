package main

import (
	"io"
	"log/slog"

	logging "github.com/openabstractions/abstraction-logging/go"
	"github.com/openabstractions/abstraction-provider-kit/go/engineproc"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// engineExitReport writes each engine's exit, of every kind, as one OA
// logging record and one line on stderr (MODELHOST-E1, KIT-P3).
func engineExitReport(stderr io.Writer) func(host.EngineExit) {
	return engineproc.Report(slog.New(logging.LegacyDefault("modelhostd")), stderr, "modelhostd")
}
