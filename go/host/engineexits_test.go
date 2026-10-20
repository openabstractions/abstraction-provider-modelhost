package host

import (
	"sync"
	"testing"
)

// engineExits collects the exit records a host reports (MODELHOST-E1).
type engineExits struct {
	mu  sync.Mutex
	got []EngineExit
}

func (e *engineExits) hook(x EngineExit) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.got = append(e.got, x)
}

// await is the first record, once one arrives.
func (e *engineExits) await(t *testing.T) EngineExit {
	t.Helper()
	waitUntil(t, func() bool {
		e.mu.Lock()
		defer e.mu.Unlock()
		return len(e.got) > 0
	})
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.got[0]
}
