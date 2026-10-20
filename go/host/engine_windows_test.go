//go:build windows

package host

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"golang.org/x/sys/windows"
)

func TestExplicitSocketTransportFailureRefusesBeforeEngineLaunch(t *testing.T) {
	previous := newSocketDir
	t.Cleanup(func() { newSocketDir = previous })
	newSocketDir = func(string) (*enginelaunch.SocketDir, error) {
		return nil, fmt.Errorf("%w: connect: invalid argument", enginelaunch.ErrSocketTransport)
	}
	leases := newFakeLeases()
	h := composed(t, leases, nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != EngineUnreachable {
		t.Fatalf("transport failure: %v, want %s", err, EngineUnreachable)
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("failed load released %v leases", released)
	}
}

// The service supplies a bounded Windows job to each chat engine. A smaller
// configured limit must reach the spawned process, while the default admits
// the same stand-in workload.
func TestChatEngineMemoryLimitBindsTheJob(t *testing.T) {
	limitedLeases := newFakeLeases()
	limited := composed(t, limitedLeases, func(o *Options) {
		o.Engine.MemoryLimit = 32 << 20
		o.Engine.fixture = []string{"--fake-alloc=134217728"}
	})
	if _, err := limited.Ensure(context.Background(), "tiny:1b"); WordOf(err) != Unavailable {
		t.Fatalf("an engine above its process memory limit: %v, want unavailable", err)
	}
	if _, released, _ := limitedLeases.counts(); len(released) != 1 {
		t.Fatalf("a failed load released %v leases", released)
	}
	unlimited := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.fixture = []string{"--fake-alloc=134217728"}
	})
	if unlimited.options.Engine.MemoryLimit != 0 {
		t.Fatalf("unset chat memory limit %d", unlimited.options.Engine.MemoryLimit)
	}
	if _, err := unlimited.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the engine with no configured limit: %v", err)
	}
}

// A provider-owned engine may spawn its own worker. The lease must remain
// until the contained parent and worker are both confirmed gone.
func TestYieldConfirmsContainedDescendantBeforeRelease(t *testing.T) {
	leases := newFakeLeases()
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-child-pid-file", pidFile}
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	var childPID int
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(pidFile)
		if err == nil {
			childPID, err = strconv.Atoi(strings.TrimSpace(string(raw)))
			if err != nil {
				t.Fatal(err)
			}
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if childPID == 0 {
		t.Fatal("fake engine did not spawn its worker")
	}
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(childPID))
	if err != nil {
		t.Fatalf("open child %d: %v", childPID, err)
	}
	defer windows.CloseHandle(handle)
	if status, err := windows.WaitForSingleObject(handle, 0); err != nil || status != uint32(windows.WAIT_TIMEOUT) {
		t.Fatalf("fixture worker was not live before yield: status=%d err=%v", status, err)
	}
	leases.beforeRelease = func() {
		status, waitErr := windows.WaitForSingleObject(handle, 0)
		if waitErr != nil || status != windows.WAIT_OBJECT_0 {
			t.Errorf("lease Release preceded confirmed worker exit: status=%d err=%v", status, waitErr)
		}
	}
	lease := h.Lease("tiny:1b")
	answer, reason := h.yield(res.YieldRequest{ID: "ask-tree", Resource: "card:0", Amount: 4096, Lease: lease})
	if answer != res.YieldAnswerYielded {
		t.Fatalf("yield=%s reason=%s", answer, reason)
	}
	_, released, _ := leases.counts()
	if len(released) != 1 || released[0] != lease {
		t.Fatalf("released=%v wanted %s", released, lease)
	}
	status, err := windows.WaitForSingleObject(handle, 0)
	if err != nil {
		t.Fatal(err)
	}
	if status != windows.WAIT_OBJECT_0 {
		t.Fatalf("worker %d is still running", childPID)
	}
}

// An engine that exits during readiness can leave a worker behind. Cleanup
// cannot release its grant until supervision confirms that worker ended too.
func TestFailedStartupConfirmsDescendantBeforeRelease(t *testing.T) {
	leases := newFakeLeases()
	pidFile := filepath.Join(t.TempDir(), "child.pid")
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-child-pid-file", pidFile, "--fake-exit-after-child"}
	})
	leases.beforeRelease = func() {
		raw, err := os.ReadFile(pidFile)
		if err != nil {
			t.Errorf("child PID missing before release: %v", err)
			return
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
		if err != nil {
			t.Errorf("child PID malformed: %v", err)
			return
		}
		handle, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(pid))
		if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
			return
		}
		if err != nil {
			t.Errorf("inspect worker before release: %v", err)
			return
		}
		defer windows.CloseHandle(handle)
		status, waitErr := windows.WaitForSingleObject(handle, 0)
		if waitErr != nil || status != windows.WAIT_OBJECT_0 {
			t.Errorf("failed startup released while worker alive: status=%d err=%v", status, waitErr)
		}
	}
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err == nil {
		t.Fatal("engine that exited during startup was admitted")
	}
	_, released, _ := leases.counts()
	if len(released) != 1 {
		t.Fatalf("failed startup released %v leases", released)
	}
}

// INF-F4: the pinned engine folder is held with read sharing while the engine
// runs, so its files cannot be swapped under it, and it is free again once
// the engine's exit is confirmed.
func TestThePinnedFolderIsHeldWhileTheEngineRuns(t *testing.T) {
	e := copiedEngine(t)
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine = e })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	moved := e.Program + ".moved"
	if err := os.Rename(e.Program, moved); err == nil {
		t.Fatal("the running engine's program was renamed")
	}
	h.Unload("tiny:1b", "test")
	deadline := time.Now().Add(5 * time.Second)
	for {
		err := os.Rename(e.Program, moved)
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the folder is still held after the engine stopped: %v", err)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
