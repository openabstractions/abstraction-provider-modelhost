package host

import (
	"sync"
	"time"
)

// idleState is each loaded name's own idle time (MODELHOST-H8): its model
// family's engine-profile value, from the same per-family data as its cache
// recipe. A name without one uses the declared --idle.
type idleState struct {
	mu   sync.Mutex
	idle map[string]time.Duration
}

// recordIdle keeps the idle time of name's current launch; zero clears it.
func (h *Host) recordIdle(name string, idle time.Duration) {
	s := &h.idle
	s.mu.Lock()
	defer s.mu.Unlock()
	if idle <= 0 {
		delete(s.idle, name)
		return
	}
	if s.idle == nil {
		s.idle = map[string]time.Duration{}
	}
	s.idle[name] = idle
}

// residentIdle is name's own idle time, zero for the declared --idle.
func (h *Host) residentIdle(name string) time.Duration {
	s := &h.idle
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.idle[name]
}
