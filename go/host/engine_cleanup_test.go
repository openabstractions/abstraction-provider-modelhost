package host

import (
	"errors"
	"testing"
	"time"
)

func TestEngineCleanupCompletionDistinguishesPendingAndFailed(t *testing.T) {
	finished := make(chan struct{})
	p := &Process{cleaned: finished}
	if err := p.waitCleanup(time.Now()); !errors.Is(err, errCleanupPending) {
		t.Fatalf("unfinished cleanup: %v", err)
	}
	failure := errors.New("the endpoint could not be removed")
	p.cleanupErr = failure
	close(finished)
	if err := p.waitCleanup(time.Now()); !errors.Is(err, failure) {
		t.Fatalf("completed cleanup failure: %v", err)
	}
	p.cleanupErr = nil
	if err := p.waitCleanup(time.Now()); err != nil {
		t.Fatalf("completed cleanup: %v", err)
	}
}
