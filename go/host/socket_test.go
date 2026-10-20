//go:build windows || linux

package host

import (
	"context"
	"errors"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
)

// launchDir is the launch directory of name's running engine.
func launchDir(t *testing.T, h *Host, name string) string {
	t.Helper()
	endpoint, ok := h.EngineEndpoint(strings.TrimPrefix(h.Base(name), "http://"))
	if !ok {
		t.Fatalf("%s has no running engine", name)
	}
	return filepath.Dir(endpoint.Path)
}

// gone waits for path to disappear.
func gone(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%s outlived its engine", path)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// CONTRACT.md MODELHOST-L7: a process other than the engine serving on the
// engine's socket path is refused by its process id at readiness, before
// any request reaches it; the engine is stopped and the lease given back.
func TestAnotherServerOnTheEngineSocketIsRefused(t *testing.T) {
	var reached atomic.Int32
	var impostor net.Listener
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-no-listen"}
		o.Engine.Ready = 5 * time.Second
		o.Engine.beforeLaunch = func(d *enginelaunch.SocketDir) {
			l, err := net.Listen("unix", d.Socket)
			if err != nil {
				t.Errorf("the impostor could not listen: %v", err)
				return
			}
			impostor = l
			go func() {
				_ = http.Serve(l, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					reached.Add(1)
					_, _ = w.Write([]byte(`{"status":"ok"}`))
				}))
			}()
		}
	})
	t.Cleanup(func() {
		if impostor != nil {
			impostor.Close()
		}
	})
	_, err := h.Ensure(context.Background(), "tiny:1b")
	if WordOf(err) != EnginePeer {
		t.Fatalf("an impostor on the engine's socket: %v, wanted %s", err, EnginePeer)
	}
	if reached.Load() != 0 {
		t.Fatalf("the impostor received %d requests", reached.Load())
	}
	if h.Loaded("tiny:1b") {
		t.Fatal("the engine stayed resident")
	}
	if acquires, released, _ := leases.counts(); acquires != 1 || len(released) != 1 {
		t.Fatalf("acquired %d, released %v; wanted the one lease given back", acquires, released)
	}
}

// CONTRACT.md MODELHOST-L7: the launch directory is removed once the engine
// exits, after an unload and after a crash that left its socket file behind.
func TestTheLaunchDirectoryIsRemovedWhenTheEngineExits(t *testing.T) {
	h := composed(t, newFakeLeases(), nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	unloaded := launchDir(t, h, "tiny:1b")
	h.Unload("tiny:1b", "test")
	gone(t, unloaded)

	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	crashed := launchDir(t, h, "tiny:1b")
	if crashed == unloaded {
		t.Fatal("a second launch reused the first launch's directory")
	}
	engine, err := os.FindProcess(h.residentPID("tiny:1b"))
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Kill(); err != nil {
		t.Fatal(err)
	}
	gone(t, crashed)
}

// MODELHOST-L7: Close has finished the provider's post-exit cleanup before it
// returns, including when the engine exited on its own before Close. A caller
// can then end without leaving the final socket and its held file behind.
func TestCloseReturnsAfterTheLaunchDirectoryIsRemoved(t *testing.T) {
	for _, crash := range []bool{false, true} {
		name := "stopped"
		if crash {
			name = "already exited"
		}
		t.Run(name, func(t *testing.T) {
			leases := newFakeLeases()
			h := composed(t, leases, nil)
			if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
				t.Fatal(err)
			}
			dir := launchDir(t, h, "tiny:1b")
			if crash {
				engine, err := os.FindProcess(h.residentPID("tiny:1b"))
				if err != nil {
					t.Fatal(err)
				}
				if err := engine.Kill(); err != nil {
					t.Fatal(err)
				}
			}
			h.Close()
			if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("launch directory survived Close: %v", err)
			}
			if acquired, released, _ := leases.counts(); acquired != 1 || len(released) != 1 {
				t.Fatalf("lease after Close: acquired %d, released %v", acquired, released)
			}
		})
	}
}

// A completed cleanup error is reported, but a proved-dead engine no longer
// reserves GPU memory merely because a filesystem action failed afterward.
func TestCompletedCleanupErrorReleasesTheLease(t *testing.T) {
	leases := newFakeLeases()
	var reported []error
	h := composed(t, leases, func(o *Options) {
		o.OnError = func(err error) { reported = append(reported, err) }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	h.engineMu.Lock()
	p := h.engines[h.hosts["tiny:1b"]]
	h.engineMu.Unlock()
	if p == nil {
		t.Fatal("the engine was not registered")
	}
	p.owned.Stop(stopWithin)
	<-p.cleaned
	failure := errors.New("fixture: cleanup could not remove a file")
	p.cleanupErr = failure
	h.Close()
	if acquired, released, _ := leases.counts(); acquired != 1 || len(released) != 1 {
		t.Fatalf("lease after completed cleanup error: acquired %d, released %v", acquired, released)
	}
	found := false
	for _, err := range reported {
		found = found || errors.Is(err, failure)
	}
	if !found {
		t.Fatalf("cleanup error was not reported: %v", reported)
	}
}

// CONTRACT.md MODELHOST-L7: a launch directory a stopped provider left with
// a dead socket in it is removed when the next provider starts.
func TestNewSweepsAStaleLaunchDirectory(t *testing.T) {
	engine := pinnedEngine(t)
	stale := filepath.Join(engine.Sockets, "deadbeef")
	if err := os.MkdirAll(stale, 0o700); err != nil {
		t.Fatal(err)
	}
	l, err := net.Listen("unix", filepath.Join(stale, enginelaunch.SocketName))
	if err != nil {
		t.Fatal(err)
	}
	l.(*net.UnixListener).SetUnlinkOnClose(false)
	l.Close()
	composed(t, newFakeLeases(), func(o *Options) { o.Engine = engine })
	if _, err := os.Lstat(stale); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the stale launch directory survived New: %v", err)
	}
}

// CONTRACT.md MODELHOST-L7: a socket folder whose launch sockets cannot fit
// an AF_UNIX address is refused when the provider is made.
func TestASocketFolderTooLongIsRefused(t *testing.T) {
	engine := pinnedEngine(t)
	engine.Sockets = filepath.Join(engine.Sockets, strings.Repeat("s", enginelaunch.MaxSocketPath))
	_, err := New(Options{Models: []Model{{Name: "tiny:1b", Store: "fake", Object: "tiny"}}, Engine: engine,
		Source: fakeSource{}, Leases: newFakeLeases()})
	if !errors.Is(err, enginelaunch.ErrSocketPath) {
		t.Fatalf("a socket folder too long: %v", err)
	}
}

// CONTRACT.md MODELHOST-L7: two hosts share one socket root. A second host
// that starts while the first host's engine has its launch directory and no
// socket yet keeps that directory, and the first engine comes up in it.
func TestASecondHostKeepsAStartingEnginesDirectory(t *testing.T) {
	first := pinnedEngine(t)
	sockets := first.Sockets
	var kept error
	var swept []error
	first.beforeLaunch = func(d *enginelaunch.SocketDir) {
		second := pinnedEngine(t)
		second.Sockets = sockets
		composed(t, newFakeLeases(), func(o *Options) {
			o.Engine = second
			o.OnError = func(err error) { swept = append(swept, err) }
		})
		_, kept = os.Lstat(filepath.Join(d.Dir, "held"))
	}
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine = first })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the first host's engine after a second host started: %v", err)
	}
	if kept != nil || len(swept) != 0 {
		t.Fatalf("the second host's sweep took the starting engine's directory: %v %v", kept, swept)
	}
}
