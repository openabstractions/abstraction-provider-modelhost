package host

import (
	"net/netip"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// Condition 6: an engine that holds a listener beyond loopback is stopped
// after readiness and its lease released. The stand-in listens on 127.0.0.1
// only; the owner table the check reads is the real one plus one row naming
// the engine's own process on 0.0.0.0, so no socket beyond loopback is ever
// opened and no firewall prompt is raised.
func TestVideoEngineListeningBeyondLoopbackIsRefused(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "engine.pid")
	f := newVideoFixture(t, "--fake-pid-file="+pidFile)
	real := ownerSockets
	t.Cleanup(func() { ownerSockets = real })
	reported := false
	ownerSockets = func() ([]identity.Socket, error) {
		sockets, err := real()
		if err != nil {
			return nil, err
		}
		raw, readErr := os.ReadFile(pidFile)
		pid, convErr := strconv.Atoi(strings.TrimSpace(string(raw)))
		if readErr != nil || convErr != nil {
			return sockets, nil
		}
		reported = true
		return append(sockets, identity.Socket{Proto: "tcp", Addr: netip.IPv4Unspecified(), Port: 1, PID: pid}), nil
	}
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != ListenerNotLoopback {
		t.Fatalf("observed %+v", observed)
	}
	if !reported {
		t.Fatal("the owner table never named the engine")
	}
	if _, released, _ := leases.counts(); len(released) != 1 || h.Loaded(testVideoModel) {
		t.Fatal("the engine beyond loopback kept running or kept its lease")
	}
	if len(f.bodies(t)) != 0 {
		t.Fatal("a request reached the engine beyond loopback")
	}
}

// Condition 11: the configured memory limit is set on the engine's job: an
// engine that commits more than it allows ends before it is ready, and the
// same engine without the limit runs.
func TestVideoEngineMemoryLimitBindsTheJob(t *testing.T) {
	f := newVideoFixture(t, "--fake-alloc=268435456")
	f.engine.MemoryLimit = 64 << 20
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != "engine:unavailable" {
		t.Fatalf("observed %+v", observed)
	}
	unlimited := newVideoFixture(t, "--fake-alloc=268435456")
	h = unlimited.host(t, newFakeLeases(), nil)
	if observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted); observed.Status != wire.VideoStateCompleted {
		t.Fatalf("without a limit: %+v", observed)
	}
}
