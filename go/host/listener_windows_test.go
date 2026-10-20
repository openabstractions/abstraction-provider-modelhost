//go:build windows

package host

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/netip"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"golang.org/x/sys/windows"
)

// enginePID reads the process id the stand-in engine wrote at start.
func enginePID(t *testing.T, file string) int {
	t.Helper()
	raw, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return pid
}

// CONTRACT.md MODELHOST-L7: after readiness the provider reads the socket
// owner tables, and an engine whose tree holds any TCP socket in any state
// or bound UDP socket, loopback included, is stopped, its lease released,
// and the call refused. The stand-in serves on its socket only, so the owner
// table this test reads is the real one plus one row naming the engine's own
// process on 127.0.0.1; nothing in this test opens a network listener.
func TestANetworkSocketStopsTheEngineAndRefuses(t *testing.T) {
	real := ownerAllSockets
	t.Cleanup(func() { ownerAllSockets = real })
	pidFile := filepath.Join(t.TempDir(), "pid")
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine.fixture = []string{"--fake-pid-file", pidFile} })
	engine := 0
	ownerAllSockets = func() ([]identity.Socket, error) {
		sockets, err := real()
		if err != nil {
			return nil, err
		}
		raw, err := os.ReadFile(pidFile)
		if err != nil {
			return nil, err
		}
		engine, _ = strconv.Atoi(string(raw))
		return append(sockets, identity.Socket{Proto: "tcp", Addr: netip.AddrFrom4([4]byte{127, 0, 0, 1}), Port: 1, PID: engine}), nil
	}
	_, err := h.Ensure(context.Background(), "tiny:1b")
	if WordOf(err) != EngineNetworkSocket {
		t.Fatalf("an engine holding a loopback listener: %v, wanted %s", err, EngineNetworkSocket)
	}
	if h.Loaded("tiny:1b") {
		t.Fatal("the engine holding a network socket stayed resident")
	}
	acquires, released, _ := leases.counts()
	if acquires != 1 || len(released) != 1 {
		t.Fatalf("acquired %d, released %v; wanted the one lease given back", acquires, released)
	}
	if engine == 0 {
		t.Fatal("the check never read the engine's process id")
	}
	handle, err := windows.OpenProcess(windows.SYNCHRONIZE, false, uint32(engine))
	if errors.Is(err, windows.ERROR_INVALID_PARAMETER) {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	defer windows.CloseHandle(handle)
	if status, err := windows.WaitForSingleObject(handle, 0); err != nil || status != windows.WAIT_OBJECT_0 {
		t.Fatalf("engine %d still runs after the refusal: status=%d err=%v", engine, status, err)
	}
}

// CONTRACT.md MODELHOST-L7: the engine is told a socket path in its launch
// directory and no port, and once it is ready the real owner tables show no
// socket in any state and no UDP socket of its process: a launch opens no
// network socket.
func TestALaunchOpensNoNetworkSocket(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "pid")
	argsFile := filepath.Join(t.TempDir(), "args")
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine.fixture = []string{"--fake-pid-file", pidFile, "--fake-args-file", argsFile}
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the socket engine was refused: %v", err)
	}
	engine := enginePID(t, pidFile)
	sockets, err := identity.AllSockets()
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range sockets {
		if s.PID == engine {
			t.Fatalf("the engine owns %s %s:%d", s.Proto, s.Addr, s.Port)
		}
	}
	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(raw), "\n")
	endpoint, ok := h.EngineEndpoint(strings.TrimPrefix(h.Base("tiny:1b"), "http://"))
	if !ok {
		t.Fatal("no endpoint for the resident engine")
	}
	host := ""
	for i, arg := range args {
		if arg == "--port" || strings.HasPrefix(arg, "--port=") {
			t.Fatalf("the engine was given a port: %q", args)
		}
		if arg == "--host" && i+1 < len(args) {
			host = args[i+1]
		}
	}
	if host != endpoint.Path || filepath.Base(host) != "e.sock" || !strings.HasPrefix(host, h.options.Engine.Sockets) {
		t.Fatalf("--host %q, wanted the launch socket %q", host, endpoint.Path)
	}
	if _, err := os.Lstat(host); err != nil {
		t.Fatalf("the engine's socket: %v", err)
	}
}

// watched arms a socket table that names one process's outbound socket
// once armed holds its id, and shortens the watch interval.
func watched(t *testing.T) *atomic.Int32 {
	t.Helper()
	real, interval := ownerAllSockets, socketWatch
	t.Cleanup(func() { ownerAllSockets, socketWatch = real, interval })
	socketWatch = 50 * time.Millisecond
	var armed atomic.Int32
	ownerAllSockets = func() ([]identity.Socket, error) {
		sockets, err := real()
		if pid := armed.Load(); err == nil && pid != 0 {
			sockets = append(sockets, identity.Socket{Proto: "tcp", Addr: netip.MustParseAddr("192.0.2.1"), Port: 50000, PID: int(pid)})
		}
		return sockets, err
	}
	return &armed
}

// running starts h's holder loop, which notices an exited engine.
func running(t *testing.T, h *Host) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { defer close(done); h.Run(ctx) }()
	t.Cleanup(func() { cancel(); <-done })
}

// CONTRACT.md MODELHOST-L7: the socket check runs again while the engine
// runs. A network socket that appears after readiness stops that engine,
// the holder gives its lease back, and the provider reports
// engine:network-socket.
func TestANetworkSocketAfterReadinessStopsTheEngine(t *testing.T) {
	armed := watched(t)
	var mu sync.Mutex
	var reports []string
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.OnError = func(err error) { mu.Lock(); reports = append(reports, err.Error()); mu.Unlock() }
	})
	running(t, h)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(150 * time.Millisecond)
	if !h.Loaded("tiny:1b") {
		t.Fatal("a clean engine was stopped by the watch")
	}
	armed.Store(int32(h.residentPID("tiny:1b")))
	deadline := time.Now().Add(10 * time.Second)
	for h.Loaded("tiny:1b") {
		if time.Now().After(deadline) {
			t.Fatal("the engine holding a network socket kept running")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v, wanted the lease given back", released)
	}
	mu.Lock()
	defer mu.Unlock()
	found := false
	for _, r := range reports {
		found = found || strings.HasPrefix(r, EngineNetworkSocket)
	}
	if !found {
		t.Fatalf("no %s report: %q", EngineNetworkSocket, reports)
	}
}

// CONTRACT.md MODELHOST-L7: when the model is unloaded and loaded again
// between a failed running check and the stop it leads to, the stop ends
// only the engine that failed. The newer engine stays resident under its
// lease, and nothing is reported for an engine that is no longer resident.
func TestAFailedCheckStopsOnlyItsOwnEngine(t *testing.T) {
	armed := watched(t)
	var mu sync.Mutex
	var reports []string
	leases := newFakeLeases()
	var replaced atomic.Int32
	var h *Host
	h = composed(t, leases, func(o *Options) {
		o.OnError = func(err error) { mu.Lock(); reports = append(reports, err.Error()); mu.Unlock() }
	})
	h.afterFailedCheck = func() {
		if !replaced.CompareAndSwap(0, 1) {
			return
		}
		h.Unload("tiny:1b", "test")
		if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
			t.Errorf("the replacement load: %v", err)
		}
		replaced.Store(2)
	}
	running(t, h)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	first := h.residentPID("tiny:1b")
	armed.Store(int32(first))
	deadline := time.Now().Add(10 * time.Second)
	for replaced.Load() != 2 {
		if time.Now().After(deadline) {
			t.Fatal("the running check never failed")
		}
		time.Sleep(20 * time.Millisecond)
	}
	time.Sleep(300 * time.Millisecond)
	second := h.residentPID("tiny:1b")
	if !h.Loaded("tiny:1b") || second == 0 || second == first {
		t.Fatalf("resident pid %d after replacing %d; the newer engine was stopped", second, first)
	}
	if acquires, released, _ := leases.counts(); acquires != 2 || len(released) != 1 {
		t.Fatalf("acquired %d, released %v; wanted the newer engine's lease held", acquires, released)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, r := range reports {
		if strings.HasPrefix(r, EngineNetworkSocket) {
			t.Fatalf("a report for an engine no longer resident: %q", r)
		}
	}
}

// INF-EL2 through one owner: the chat engine on its socket is judged by the
// socket rule, from the all-sockets table, and the video engine on its port
// by the loopback rule, from the listener table.
func TestEachLaunchPathReadsItsOwnRule(t *testing.T) {
	realListeners, realAll := ownerSockets, ownerAllSockets
	t.Cleanup(func() { ownerSockets, ownerAllSockets = realListeners, realAll })
	var listeners, all atomic.Int32
	ownerSockets = func() ([]identity.Socket, error) { listeners.Add(1); return realListeners() }
	ownerAllSockets = func() ([]identity.Socket, error) { all.Add(1); return realAll() }

	h := composed(t, newFakeLeases(), nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if all.Load() == 0 || listeners.Load() != 0 {
		t.Fatalf("the chat engine read all=%d listeners=%d, wanted the socket rule alone", all.Load(), listeners.Load())
	}
	all.Store(0)
	f := newVideoFixture(t)
	v := f.host(t, newFakeLeases(), nil)
	if observed := awaitVideo(t, v, startVideo(t, v, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted); observed.Status != wire.VideoStateCompleted {
		t.Fatalf("the video engine: %+v", observed)
	}
	if listeners.Load() == 0 || all.Load() != 0 {
		t.Fatalf("the video engine read listeners=%d all=%d, wanted the loopback rule alone", listeners.Load(), all.Load())
	}
}

// MODELHOST-V5 with INF-EL2's loopback rule: a process outside the engine's
// tree listening on the video engine's port is refused by the TCP owner
// table before any request reaches it, and the video fails without a
// readiness from it.
func TestAForeignListenerOnTheVideoPortIsRefused(t *testing.T) {
	var reached atomic.Int32
	var foreign net.Listener
	t.Cleanup(func() {
		if foreign != nil {
			foreign.Close()
		}
	})
	f := newVideoFixture(t, "--fake-no-listen")
	f.engine.Ready = 3 * time.Second
	f.engine.beforeLaunch = func(port int) {
		l, err := net.Listen("tcp4", "127.0.0.1:"+strconv.Itoa(port))
		if err != nil {
			t.Errorf("the foreign listener: %v", err)
			return
		}
		foreign = l
		go func() {
			_ = http.Serve(l, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				reached.Add(1)
				_, _ = w.Write([]byte(`{"supported_modes":["vid_gen"]}`))
			}))
		}()
	}
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed {
		t.Fatalf("observed %+v with a foreign listener on the port", observed)
	}
	if reached.Load() != 0 {
		t.Fatalf("the foreign listener received %d requests", reached.Load())
	}
	if _, released, _ := leases.counts(); len(released) != 1 || h.Loaded(testVideoModel) {
		t.Fatal("the engine kept running or kept its lease")
	}
}

// MODELHOST-V5 with INF-EL2's loopback rule: video submit, poll and cancel
// go through the engine's bound client. When sd-server exits after
// readiness and another process takes its port, the job request never
// reaches that process.
func TestAForeignListenerAfterReadinessReceivesNoJob(t *testing.T) {
	var reached atomic.Int32
	var foreign net.Listener
	t.Cleanup(func() {
		if foreign != nil {
			foreign.Close()
		}
	})
	f := newVideoFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	h.video.beforeSubmit = func(base string) {
		h.video.mu.Lock()
		p := h.video.process
		h.video.mu.Unlock()
		engine, err := os.FindProcess(p.PID())
		if err != nil {
			t.Errorf("the engine: %v", err)
			return
		}
		if err := engine.Kill(); err != nil {
			t.Errorf("stopping the engine: %v", err)
			return
		}
		<-p.exited
		_, port, err := net.SplitHostPort(strings.TrimPrefix(base, "http://"))
		if err != nil {
			t.Errorf("the engine base %s: %v", base, err)
			return
		}
		l, err := net.Listen("tcp4", "127.0.0.1:"+port)
		if err != nil {
			t.Errorf("the foreign listener: %v", err)
			return
		}
		foreign = l
		go func() {
			_ = http.Serve(l, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				reached.Add(1)
				w.WriteHeader(http.StatusAccepted)
				_, _ = w.Write([]byte(`{"id":"stolen","status":"queued"}`))
			}))
		}()
	}
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed {
		t.Fatalf("observed %+v with a foreign listener on the port", observed)
	}
	if foreign == nil {
		t.Fatal("the foreign listener never took the port")
	}
	if reached.Load() != 0 {
		t.Fatalf("the foreign listener received %d requests", reached.Load())
	}
}
