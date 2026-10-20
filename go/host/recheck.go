package host

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity/go/trustedpath"
)

// The background re-hash of placed model files (MODELHOST-L11; Decision 2,
// condition 5, research/fast-load/PROPOSAL.md). A record whose last full
// hash is older than recheckAge is read in full again while nothing loads
// and nobody has used the machine for userIdleFor. Media that changes with
// no metadata change shows here; the file's next load is then refused.
const (
	recheckAge   = 30 * 24 * time.Hour
	recheckEvery = 10 * time.Minute
	userIdleFor  = 10 * time.Minute
)

// recheckRecord is one placed record and the verifier that proves by it.
type recheckRecord struct {
	files  *programfiles.Verifier
	record string
}

// recheckRecords are the records this host proves by: each engine tree's
// and the placed model files' (the video set's and each pinned chat
// model's). Engine trees are executed code and get the same re-hash
// (Decision 2, amendment 2).
func (h *Host) recheckRecords() []recheckRecord {
	var out []recheckRecord
	if h.video != nil && h.video.opts.Engine.Files != nil {
		for _, record := range []string{h.video.opts.Engine.Record, h.video.opts.Engine.ModelRecord} {
			if record != "" {
				out = append(out, recheckRecord{h.video.opts.Engine.Files, record})
			}
		}
	}
	if h.options.Engine.Record != "" && h.options.Engine.Files != nil {
		out = append(out, recheckRecord{h.options.Engine.Files, h.options.Engine.Record})
		//unchecked: a pattern without meta-character errors only on a bad pattern
		chat, _ := filepath.Glob(filepath.Join(filepath.Dir(h.options.Engine.Record), "models", "gguf-*.json"))
		for _, record := range chat {
			out = append(out, recheckRecord{h.options.Engine.Files, record})
		}
	}
	return out
}

// quiet reports whether a re-hash may run now: no model is loading, no
// video engine runs and the person has left the machine alone.
func (h *Host) quiet() bool {
	h.loadMu.Lock()
	loading := len(h.loads) > 0
	h.loadMu.Unlock()
	if loading {
		return false
	}
	if h.video != nil {
		h.video.mu.Lock()
		running := h.video.process != nil
		h.video.mu.Unlock()
		if running {
			return false
		}
	}
	idle := h.userIdle
	if idle == nil {
		idle = userIdleTime
	}
	return idle() >= userIdleFor
}

// recheckLoop runs recheckOnce every recheckEvery until ctx ends.
func (h *Host) recheckLoop(ctx context.Context) {
	every := h.recheckEvery
	if every <= 0 {
		every = recheckEvery
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			h.recheckOnce(ctx)
		}
	}
}

// recheckOnce re-hashes each due record while the host stays quiet. A
// re-hash stops as soon as the host is no longer quiet; it holds no proof
// lock while it reads, so a load never waits for it. A mismatch writes one
// line through OnError.
func (h *Host) recheckOnce(ctx context.Context) {
	for _, due := range h.recheckRecords() {
		if ctx.Err() != nil || !h.quiet() {
			return
		}
		rctx, cancel := context.WithCancel(ctx)
		done := make(chan struct{})
		go func() {
			t := time.NewTicker(250 * time.Millisecond)
			defer t.Stop()
			for {
				select {
				case <-done:
					return
				case <-t.C:
					if !h.quiet() {
						cancel()
						return
					}
				}
			}
		}()
		r, err := due.files.RecheckRecord(rctx, due.record, recheckAge)
		close(done)
		cancel()
		h.reportRecheck(r, err)
	}
}

func (h *Host) reportRecheck(r programfiles.Recheck, err error) {
	if h.options.OnError == nil {
		return
	}
	switch {
	case r.Broken != "":
		h.options.OnError(fmt.Errorf("modelhost: the file %s no longer matches its pinned bytes; its next load is refused until it is downloaded or installed again", filepath.FromSlash(r.Broken)))
	case err != nil && !errors.Is(err, context.Canceled):
		h.options.OnError(fmt.Errorf("modelhost: the monthly re-check of the model files: %w", err))
	}
}

// logStateOnce writes, once per start, why records in the state folder
// prove nothing across starts, and each AppContainer or capability
// principal that may change a folder above it (Decision 2, amendment 1).
func logStateOnce(state string, files *programfiles.Verifier, report func(error)) {
	if err := files.StateProblem(filepath.Join(state, "engine.files")); err != nil {
		report(fmt.Errorf("modelhost: every start hashes the engine and model files in full: %w", err))
	}
	for _, line := range trustedpath.ForeignCapabilities(state) {
		report(fmt.Errorf("modelhost: an app container may change a folder above the provider's state: %s; the folder's Medium integrity label keeps it from renaming or replacing what OA keeps there", line))
	}
}
