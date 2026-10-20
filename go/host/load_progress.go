package host

import (
	"context"
	"io"
	"strings"
	"sync"
)

// loadProgress projects first occurrences of pinned b10825 phase lines.
// Sources: src/llama-model-loader.cpp (metadata), src/llama-model.cpp
// (load_tensors), src/llama-context.cpp (constructor and graph reserve).
// These are ordinal phase transitions, not bytes, percentages or heartbeats.
type loadProgress struct {
	mu       sync.Mutex
	out      io.Writer
	line     []byte
	phase    uint64
	overflow bool
	publish  func(uint64)
}

func (p *loadProgress) count() uint64 { p.mu.Lock(); defer p.mu.Unlock(); return p.phase }

func (p *loadProgress) Write(b []byte) (int, error) {
	p.mu.Lock()
	var phases [4]uint64
	n := 0
	for _, c := range b {
		if c == '\n' {
			if !p.overflow {
				line := string(p.line)
				for i, marker := range []string{
					"llama_model_loader: loaded meta data with ",
					"load_tensors: loading model tensors, this can take a while...",
					"llama_context: constructing llama_context",
					"sched_reserve: graph nodes  = ",
				} {
					if strings.HasPrefix(line, marker) && uint64(i+1) > p.phase {
						p.phase = uint64(i + 1)
						phases[n] = p.phase
						n++
					}
				}
			}
			p.line = p.line[:0]
			p.overflow = false
		} else if len(p.line) < 4096 && !p.overflow {
			p.line = append(p.line, c)
		} else {
			p.overflow = true
		}
	}
	p.mu.Unlock()
	if p.publish != nil {
		for _, phase := range phases[:n] {
			p.publish(phase)
		}
	}
	if p.out == nil {
		return len(b), nil
	}
	return p.out.Write(b)
}

type loadProgressCallbackKey struct{}
type modelLoadKey struct{}

// WithLoadProgress observes genuine pinned engine phases during this Ensure
// wait. The callback runs on the waiting goroutine outside owner locks and must
// be bounded and nonblocking. A late waiter receives the current observed phase.
// Ending the wait removes its subscription; it leaves accepted loading intact.
// Phase is one of metadata, tensors, context, graph. These name phase beginnings.
func WithLoadProgress(ctx context.Context, callback func(phase string)) context.Context {
	return context.WithValue(ctx, loadProgressCallbackKey{}, callback)
}

// sharedLoadProgress keeps only actual phase ordinals and bounded mailboxes.
// Publishing never waits for a caller or invokes application code.
type sharedLoadProgress struct {
	mu      sync.Mutex
	phase   uint64
	waiters map[chan uint64]struct{}
}

// begin belongs to the holder's actual engine start. A speculative attempt may
// yield and restart under its established policy; its old phase is then stale.
func (p *sharedLoadProgress) begin() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phase = 0
	for ch := range p.waiters {
		draining := true
		for draining {
			select {
			case <-ch:
			default:
				draining = false
			}
		}
	}
}

func (p *sharedLoadProgress) publish(phase uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if phase <= p.phase || phase > 4 {
		return
	}
	p.phase = phase
	for ch := range p.waiters {
		select {
		case ch <- phase:
		default:
		}
	}
}

func (p *sharedLoadProgress) subscribe() (<-chan uint64, func()) {
	ch := make(chan uint64, 4)
	p.mu.Lock()
	if p.waiters == nil {
		p.waiters = make(map[chan uint64]struct{})
	}
	p.waiters[ch] = struct{}{}
	if p.phase != 0 {
		ch <- p.phase
	}
	p.mu.Unlock()
	return ch, func() { p.mu.Lock(); delete(p.waiters, ch); p.mu.Unlock() }
}

func loadPhaseName(phase uint64) string {
	return [...]string{"", "metadata", "tensors", "context", "graph"}[phase]
}
