package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	inference "github.com/openabstractions/abstraction-inference/go"
	infwire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	res "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/holder"
	router "github.com/openabstractions/abstraction-router/go"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// buildFakeEngine builds the stand-in engine once per test binary. It stands
// where llama-server stands, so these tests measure this provider and not
// llama.cpp.
var fakeEngine = sync.OnceValues(func() (string, error) {
	dir, err := os.MkdirTemp("", "modelhost-engine-")
	if err != nil {
		return "", err
	}
	name := "fakeengine"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	target := filepath.Join(dir, name)
	build := exec.Command("go", "build", "-o", target, "./testdata/fakeengine")
	if out, err := build.CombinedOutput(); err != nil {
		return "", fmt.Errorf("build fakeengine: %v\n%s", err, out)
	}
	return target, nil
})

// requireEnginePeer skips a test that sends a request to a running stand-in
// engine where the engine's server process cannot be bound. Every engine
// request first proves its peer (inference CONTRACT.md INF-EL2): the chat
// engine on its AF_UNIX socket (identity ID-T8) and the video engine on its
// loopback port. enginePeerProbe asks abstraction-identity for both bindings
// on a listener of this process and skips only on ErrNoBinding, which macOS
// answers for both; the provider refuses every engine there by design
// (README "Platforms"). Any other probe failure fails the test.
func requireEnginePeer(t *testing.T) {
	t.Helper()
	err := enginePeerProbe()
	if errors.Is(err, identity.ErrNoBinding) {
		t.Skipf("this platform cannot bind an engine's server process: %v", err)
	}
	if err != nil {
		t.Fatalf("engine peer probe: %v", err)
	}
}

var enginePeerProbe = sync.OnceValue(func() error {
	dir, err := os.MkdirTemp("", "oa-mhp")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	socket, err := net.Listen("unix", filepath.Join(dir, "s"))
	if err != nil {
		return fmt.Errorf("AF_UNIX: %w", err)
	}
	if err := probeServerBinding(socket, "unix", identity.BindUnixServer); err != nil {
		return fmt.Errorf("AF_UNIX: %w", err)
	}
	loopback, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return fmt.Errorf("loopback: %w", err)
	}
	if err := probeServerBinding(loopback, "tcp4", identity.BindLoopbackServer); err != nil {
		return fmt.Errorf("loopback: %w", err)
	}
	return nil
})

// probeServerBinding dials l, its own listener, binds the dialed
// connection's server end and closes l.
func probeServerBinding(l net.Listener, network string, bind func(net.Conn, *identity.Options) (*identity.Binding, error)) error {
	defer l.Close()
	accepted := make(chan net.Conn, 1)
	go func() {
		c, err := l.Accept()
		if err != nil {
			close(accepted)
			return
		}
		accepted <- c
	}()
	conn, err := net.Dial(network, l.Addr().String())
	if err != nil {
		return err
	}
	defer conn.Close()
	if server, ok := <-accepted; ok {
		defer server.Close()
	}
	binding, err := bind(conn, &identity.Options{ConnectedAt: time.Now(), SkipCodeSignature: true})
	if err != nil {
		return err
	}
	return binding.Close()
}

func engineProgram(t *testing.T) string {
	t.Helper()
	program, err := fakeEngine()
	if err != nil {
		t.Fatal(err)
	}
	return program
}

// openACL stands in for the read-only folder ACL, which tests do not set on
// their temporary folders.
type openACL struct{}

func (openACL) Restrict([]string) error                 { return nil }
func (openACL) Check(string, *enginefiles.Handle) error { return nil }

// fakePin pins the stand-in engine's folder, which holds the program alone.
func fakePin(t *testing.T, program string) EnginePin {
	t.Helper()
	raw, err := os.ReadFile(program)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return EnginePin{Release: "test", Program: filepath.Base(program),
		Files: []enginefiles.File{{Path: filepath.Base(program), Size: int64(len(raw)), SHA256: hex.EncodeToString(sum[:])}}}
}

// pinnedEngine is the stand-in engine under its own pin, proven by a
// verifier that checks no folder ACL. Its sockets go to a short private
// folder: the names t.TempDir makes can pass the 107-byte socket path.
func pinnedEngine(t *testing.T) Engine {
	t.Helper()
	program := engineProgram(t)
	return Engine{Program: program, Pin: fakePin(t, program), Ready: 30 * time.Second,
		Record:  filepath.Join(t.TempDir(), "engine.files"),
		Sockets: socketFolder(t),
		Files:   enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})}
}

// socketFolder is a socket root not yet made, in a short folder of the
// temporary directory, for the engines' launch directories; the first launch
// makes it private. It is removed when the test ends.
func socketFolder(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "oa-mh")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return filepath.Join(dir, "s")
}

// engineHTTP is this provider's own engine client over h's engines, the one
// the inference provider mediates calls with.
func engineHTTP(h *Host) *http.Client {
	client := EngineClient(h, nil)
	client.Timeout = 2 * time.Second
	return client
}

// weightsFile is a file standing in for a model's weights. Nothing reads its
// bytes: the source reports its path and its size, and the engine only stats it.
func weightsFile(t *testing.T, size int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, make([]byte, size), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fakeSource holds objects by store/id; an unbound one exists only as a
// member of a zip, and asked records each Accept the host passed.
type fakeSource struct {
	objects map[string]Object
	unbound map[string]bool
	asked   *[]bridge.Accepts
}

func (f fakeSource) Resolve(store, id string, accept bridge.Accepts) (Object, error) {
	if f.asked != nil {
		*f.asked = append(*f.asked, accept)
	}
	if f.unbound[store+"/"+id] {
		return Object{Store: store, ID: id}, fmt.Errorf("%w: a member of a zip", bridge.ErrUnaccepted)
	}
	o, ok := f.objects[store+"/"+id]
	if !ok {
		return Object{}, errors.New("unknown object")
	}
	return o, nil
}
func (f fakeSource) Stores() []string   { return []string{"fake"} }
func (f fakeSource) Root(string) string { return "" }

// fakeLeases is the lease service as this provider sees it: it grants, it
// records renewals and releases, and it can put one yield request to the
// holder the way the book does.
type fakeLeases struct {
	mu            sync.Mutex
	outcome       res.AcquireOutcome
	err           error
	acquires      int
	renews        int
	released      []string
	answered      []string
	events        []string
	live          map[string]bool
	seq           int
	bindings      map[string]int
	bindOutcome   res.ProcessBindOutcome
	beforeRelease func()
	// resources is the resource of every Acquire, in order.
	resources []string
	// capacity, when not zero, is the resource's size: an Acquire beyond
	// what the live leases leave is insufficient. amounts and grants are
	// each live lease's amount and every granted amount in order.
	capacity int64
	amounts  map[string]int64
	grants   []int64

	pending chan res.YieldRequest
}

func newFakeLeases() *fakeLeases {
	return &fakeLeases{outcome: res.AcquireOutcomeAcquired, bindOutcome: res.ProcessBindOutcomeBound,
		live: map[string]bool{}, bindings: map[string]int{}, pending: make(chan res.YieldRequest, 4)}
}

func (f *fakeLeases) BindProcess(lease string, pid int) (res.ProcessBindResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.bindings[lease] = pid
	return res.ProcessBindResult{Outcome: f.bindOutcome}, nil
}

func TestUnverifiableEngineBindingKeepsResidentAndReports(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	leases.bindOutcome = res.ProcessBindOutcomeUnverifiable
	var reported []error
	h := composed(t, leases, func(options *Options) {
		options.OnError = func(err error) { reported = append(reported, err) }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("conservatively reserved load failed: %v", err)
	}
	if !h.Loaded("tiny:1b") || len(reported) != 1 || !strings.Contains(reported[0].Error(), "unverifiable") {
		t.Fatalf("resident=%v, attribution reports=%v", h.Loaded("tiny:1b"), reported)
	}
}

func (f *fakeLeases) Acquire(resource string, amount int64, wait time.Duration) (res.AcquireResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return res.AcquireResult{}, f.err
	}
	f.acquires++
	f.resources = append(f.resources, resource)
	if f.outcome != res.AcquireOutcomeAcquired {
		return res.AcquireResult{Outcome: f.outcome, Asked: []res.YieldRecord{{Holder: "host:lmstudio", Answer: "refused: no rule"}}}, nil
	}
	if f.capacity > 0 {
		held := int64(0)
		for lease := range f.live {
			held += f.amounts[lease]
		}
		if held+amount > f.capacity {
			return res.AcquireResult{Outcome: res.AcquireOutcomeInsufficient, Asked: []res.YieldRecord{}}, nil
		}
	}
	f.seq++
	id := "lease-" + strconv.Itoa(f.seq)
	f.live[id] = true
	if f.amounts == nil {
		f.amounts = map[string]int64{}
	}
	f.amounts[id] = amount
	f.grants = append(f.grants, amount)
	return res.AcquireResult{Outcome: res.AcquireOutcomeAcquired,
		Lease: &res.Lease{ID: id, Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}

func (f *fakeLeases) Renew(lease string) (res.LeaseChange, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.renews++
	return res.LeaseChange{Applied: f.live[lease]}, nil
}

func (f *fakeLeases) Release(lease string) (res.LeaseChange, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.beforeRelease != nil {
		f.beforeRelease()
	}
	f.released = append(f.released, lease)
	f.events = append(f.events, "release:"+lease)
	applied := f.live[lease]
	delete(f.live, lease)
	return res.LeaseChange{Applied: applied}, nil
}

func (f *fakeLeases) Observe(ctx context.Context, cursor string, wait time.Duration) (res.YieldPage, error) {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case request := <-f.pending:
		return res.YieldPage{Requests: []res.YieldRequest{request}, Next: request.ID}, nil
	case <-timer.C:
		return res.YieldPage{Requests: []res.YieldRequest{}, Next: cursor}, nil
	case <-ctx.Done():
		return res.YieldPage{Requests: []res.YieldRequest{}, Next: cursor}, ctx.Err()
	}
}

func (f *fakeLeases) Answer(ctx context.Context, request string, answer res.YieldAnswer, reason string) (res.AnswerResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.answered = append(f.answered, request+"="+answer.String())
	f.events = append(f.events, "answer:"+request)
	return res.AnswerResult{Recorded: true}, nil
}

func (f *fakeLeases) counts() (acquires int, released []string, answered []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.acquires, append([]string(nil), f.released...), append([]string(nil), f.answered...)
}

// composed is one host over the fake engine and the fake lease service.
func composed(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	path := weightsFile(t, 4096)
	options := Options{
		Models: []Model{{Name: "tiny:1b", Store: "fake", Object: "tiny"}},
		Engine: pinnedEngine(t),
		Source: fakeSource{objects: map[string]Object{
			"fake/tiny": {Store: "fake", ID: "tiny", Path: path, File: "model.gguf", Family: "tiny", Size: 4096}}},
		Leases: leases,
		Idle:   time.Hour,
		Renew:  50 * time.Millisecond,
		Wait:   time.Second,
	}
	if adjust != nil {
		adjust(&options)
	}
	h, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	return h
}

// composedTwo is one host whose allowlist names two objects, each with its
// own weights file and its own size, so a test can hold both resident at
// once and tell their leases apart by amount.
func composedTwo(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	first := weightsFile(t, 4096)
	second := weightsFile(t, 8192)
	options := Options{
		Models: []Model{
			{Name: "tiny:1b", Store: "fake", Object: "tiny"},
			{Name: "small:2b", Store: "fake", Object: "small"},
		},
		Engine: pinnedEngine(t),
		Source: fakeSource{objects: map[string]Object{
			"fake/tiny":  {Store: "fake", ID: "tiny", Path: first, File: "model.gguf", Family: "tiny", Size: 4096},
			"fake/small": {Store: "fake", ID: "small", Path: second, File: "model.gguf", Family: "small", Size: 8192},
		}},
		Leases: leases,
		Idle:   time.Hour,
		Renew:  50 * time.Millisecond,
		Wait:   time.Second,
	}
	if adjust != nil {
		adjust(&options)
	}
	h, err := New(options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.Close)
	return h
}

// answered reads one completion out of the engine the host loaded, so the test
// asserts a model that answers and not only a process that started.
func answered(t *testing.T, h *Host, base string) string {
	t.Helper()
	response, err := engineHTTP(h).Get(base + "/v1/models")
	if err != nil {
		t.Fatalf("the engine at %s did not answer: %v", base, err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(response.Body, 1<<16))
	return string(raw)
}

// The first call takes a lease and starts the engine; the second finds the
// same engine loaded and asks for nothing.
func TestFirstCallLoadsAndTheSecondReuses(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, nil)
	base, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	if !h.Loaded("tiny:1b") {
		t.Fatalf("tiny:1b is not resident after the first call")
	}
	first := h.residentPID("tiny:1b")
	leases.mu.Lock()
	boundPID := leases.bindings[h.Lease("tiny:1b")]
	leases.mu.Unlock()
	if boundPID != first {
		t.Fatalf("engine pid %d was bound as %d", first, boundPID)
	}
	t.Logf("first call: lease %s, engine pid %d at %s, models %s", h.Lease("tiny:1b"), first, base, answered(t, h, base))

	again, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatalf("second call: %v", err)
	}
	if again != base || h.residentPID("tiny:1b") != first {
		t.Fatalf("the second call did not reuse the load: %s pid %d, wanted %s pid %d", again, h.residentPID("tiny:1b"), base, first)
	}
	acquires, released, _ := leases.counts()
	if acquires != 1 || len(released) != 0 {
		t.Fatalf("acquires %d, releases %v; one model is loaded once", acquires, released)
	}
}

// residentPID is the engine process this host holds for name, for a test
// that has to prove a later call started nothing new.
func (h *Host) residentPID(name string) int {
	return h.resources.PID(name)
}

// A model nothing has asked for is given back, with its lease.
func TestIdleUnloadsAndReleasesTheLease(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	exits := &engineExits{}
	h := composed(t, leases, func(o *Options) {
		o.Idle = 120 * time.Millisecond
		o.Renew = 30 * time.Millisecond
		o.OnEngineExit = exits.hook
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	lease := h.Lease("tiny:1b")
	pid := h.resources.PID("tiny:1b")
	// MODELHOST-E1: the idle stop is the chat engine's exit record.
	defer func() {
		x := exits.await(t)
		if x.Engine != "llama-server" || x.Model != "tiny:1b" || x.PID != pid || !x.Stopped || x.Notable() {
			t.Fatalf("exit record %+v, want the stopped llama-server %d", x, pid)
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if !h.Loaded("tiny:1b") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the idle model was never unloaded")
		}
		time.Sleep(20 * time.Millisecond)
	}
	_, released, _ := leases.counts()
	if len(released) != 1 || released[0] != lease {
		t.Fatalf("released %v, wanted the idle model's lease %s", released, lease)
	}
	t.Logf("idle unload released %s", released[0])
}

// An operation an inference provider still counts active or retained keeps an
// idle model loaded past the idle window; once the operation retires, the
// idle unload proceeds the same way TestIdleUnloadsAndReleasesTheLease does.
func TestOpenOperationKeepsAnIdleModelLoaded(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var open atomic.Bool
	open.Store(true)
	h := composed(t, leases, func(o *Options) {
		o.Idle = 120 * time.Millisecond
		o.Renew = 30 * time.Millisecond
		o.Active = func(string) bool { return open.Load() }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)

	// Outlast the idle window several times over while the operation is open;
	// the model must not be taken out from under it.
	time.Sleep(6 * h.options.Idle)
	if !h.Loaded("tiny:1b") {
		t.Fatal("an open operation's model was unloaded before the operation retired")
	}

	// The operation retires; the idle unload proceeds.
	open.Store(false)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if !h.Loaded("tiny:1b") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the model was never unloaded once its operation retired")
		}
		time.Sleep(20 * time.Millisecond)
	}
	_, released, _ := leases.counts()
	if len(released) != 1 {
		t.Fatalf("released %v, wanted exactly one release once the operation retired", released)
	}
}

// A yield request unloads the model, releases the lease and is answered
// yielded — the whole of what this provider owes the resource table.
func TestYieldRequestUnloadsAndAnswersYielded(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, nil)
	base, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	lease := h.Lease("tiny:1b")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	leases.pending <- res.YieldRequest{ID: "ask-1", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: lease}

	deadline := time.Now().Add(10 * time.Second)
	for {
		_, released, answers := leases.counts()
		if !h.Loaded("tiny:1b") && len(released) == 1 && len(answers) == 1 {
			if released[0] != lease || answers[0] != "ask-1=yielded" {
				t.Fatalf("released %v answered %v after a yield request for %s", released, answers, lease)
			}
			t.Logf("yield: unloaded, released %s, answered %s", released[0], answers[0])
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the yield request was not answered: loaded=%v released=%v answers=%v", h.Residents(), released, answers)
		}
		time.Sleep(20 * time.Millisecond)
	}
	if _, err := engineHTTP(h).Get(base + "/v1/models"); err == nil {
		t.Fatal("the engine still answers after the yield")
	}
}

// A second object asked for while the first is resident gets its own engine
// on its own port: it neither waits for the first nor displaces it, and each
// takes its own lease (CONTRACT.md MODELHOST-H5).
func TestTwoResidentsLoadIndependently(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composedTwo(t, leases, nil)
	firstBase, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	secondBase, err := h.Ensure(context.Background(), "small:2b")
	if err != nil {
		t.Fatal(err)
	}
	if firstBase == secondBase {
		t.Fatalf("both residents answer at %s; each resident needs its own engine", firstBase)
	}
	if !h.Loaded("tiny:1b") || !h.Loaded("small:2b") {
		t.Fatalf("residents %v, wanted both tiny:1b and small:2b resident together", h.Residents())
	}
	if got := h.Residents(); len(got) != 2 || got[0] != "small:2b" || got[1] != "tiny:1b" {
		t.Fatalf("Residents() = %v, wanted [small:2b tiny:1b]", got)
	}
	firstLease, secondLease := h.Lease("tiny:1b"), h.Lease("small:2b")
	if firstLease == "" || secondLease == "" || firstLease == secondLease {
		t.Fatalf("leases %q and %q, wanted two distinct leases", firstLease, secondLease)
	}
	if acquires, _, _ := leases.counts(); acquires != 2 {
		t.Fatalf("acquires %d, wanted one per resident", acquires)
	}
	t.Logf("tiny:1b at %s (lease %s), small:2b at %s (lease %s)", firstBase, firstLease, secondBase, secondLease)
	if _, err := engineHTTP(h).Get(firstBase + "/v1/models"); err != nil {
		t.Fatalf("tiny:1b's own engine did not answer: %v", err)
	}
	if _, err := engineHTTP(h).Get(secondBase + "/v1/models"); err != nil {
		t.Fatalf("small:2b's own engine did not answer: %v", err)
	}
}

// Each resident keeps its own idle clock: the one loaded first goes idle and
// is unloaded first, while a later resident that has not reached its own
// idle time yet stays loaded (CONTRACT.md MODELHOST-H5, MODELHOST-H8).
func TestEachResidentHasItsOwnIdleClock(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composedTwo(t, leases, func(o *Options) {
		o.Idle = 300 * time.Millisecond
		o.Renew = 20 * time.Millisecond
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(150 * time.Millisecond)
	if _, err := h.Ensure(context.Background(), "small:2b"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)

	deadline := time.Now().Add(10 * time.Second)
	for h.Loaded("tiny:1b") {
		if time.Now().After(deadline) {
			t.Fatal("tiny:1b, loaded first, was never unloaded on its own idle clock")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !h.Loaded("small:2b") {
		t.Fatal("small:2b was unloaded on tiny:1b's idle clock instead of its own")
	}

	deadline = time.Now().Add(10 * time.Second)
	for h.Loaded("small:2b") {
		if time.Now().After(deadline) {
			t.Fatal("small:2b was never unloaded on its own idle clock")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A yield request names one lease. Another resident stays loaded even when
// its engine shares the same resource.
func TestYieldUnloadsOnlyRequestedResident(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composedTwo(t, leases, nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if _, err := h.Ensure(context.Background(), "small:2b"); err != nil {
		t.Fatal(err)
	}
	tinyLease := h.Lease("tiny:1b")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)

	leases.pending <- res.YieldRequest{ID: "ask-1", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: tinyLease}

	deadline := time.Now().Add(10 * time.Second)
	for h.Loaded("tiny:1b") {
		if time.Now().After(deadline) {
			t.Fatal("tiny:1b, the requested resident, was never yielded")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !h.Loaded("small:2b") {
		t.Fatal("small:2b was yielded despite its distinct lease")
	}
	_, released, answers := leases.counts()
	if len(released) != 1 || released[0] != tinyLease {
		t.Fatalf("released %v, wanted only tiny:1b's lease %s", released, tinyLease)
	}
	if len(answers) != 1 || answers[0] != "ask-1=yielded" {
		t.Fatalf("answers %v, wanted ask-1=yielded", answers)
	}
}

// The book asks its oldest lease, while this provider's usage clock can put
// another resident first. Releasing that other resident before Answer makes
// the book retire the asked lease even though its engine is still running.
func TestYieldReleasesTheLeaseTheBookAskedFor(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	clock := time.Now()
	h := composedTwo(t, leases, func(o *Options) { o.Now = func() time.Time { return clock } })
	for _, name := range []string{"tiny:1b", "small:2b", "tiny:1b"} {
		clock = clock.Add(time.Minute)
		if _, err := h.Ensure(context.Background(), name); err != nil {
			t.Fatal(err)
		}
	}
	tinyLease := h.Lease("tiny:1b")
	smallLease := h.Lease("small:2b")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	leases.pending <- res.YieldRequest{ID: "ask-older-lease", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: tinyLease}
	deadline := time.Now().Add(10 * time.Second)
	for {
		_, _, answered := leases.counts()
		if len(answered) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("yield request was not answered")
		}
		time.Sleep(20 * time.Millisecond)
	}
	if h.Loaded("tiny:1b") || !h.Loaded("small:2b") {
		t.Fatalf("asked lease %s stayed loaded while %s was unloaded: residents %v", tinyLease, smallLease, h.Residents())
	}
	_, released, answered := leases.counts()
	if len(released) != 1 || released[0] != tinyLease || len(answered) != 1 || answered[0] != "ask-older-lease=yielded" {
		t.Fatalf("release must precede Answer for asked lease %s: released %v answered %v", tinyLease, released, answered)
	}
	leases.mu.Lock()
	events := append([]string(nil), leases.events...)
	leases.mu.Unlock()
	if len(events) != 2 || events[0] != "release:"+tinyLease || events[1] != "answer:ask-older-lease" {
		t.Fatalf("release and answer order: %v", events)
	}
}

func TestYieldRefusesAnActiveModelUntilItsCallEnds(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var active atomic.Bool
	active.Store(true)
	h := composed(t, leases, func(o *Options) { o.Active = func(string) bool { return active.Load() } })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	lease := h.Lease("tiny:1b")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	request := res.YieldRequest{ID: "ask-active", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: lease}
	leases.pending <- request
	deadline := time.Now().Add(10 * time.Second)
	for {
		_, released, answered := leases.counts()
		if len(answered) > 0 {
			if answered[0] != "ask-active=refused" || len(released) != 0 || !h.Loaded("tiny:1b") {
				t.Fatalf("active resident was yielded: released %v answered %v loaded %v", released, answered, h.Loaded("tiny:1b"))
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("active yield request was not answered")
		}
		time.Sleep(20 * time.Millisecond)
	}
	active.Store(false)
	request.ID = "ask-after-call"
	leases.pending <- request
	for {
		_, released, answered := leases.counts()
		if len(answered) == 2 {
			if answered[1] != "ask-after-call=yielded" || len(released) != 1 || released[0] != lease || h.Loaded("tiny:1b") {
				t.Fatalf("idle resident was not yielded: released %v answered %v loaded %v", released, answered, h.Loaded("tiny:1b"))
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("post-call yield request was not answered")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func modelhostAdmit(h *Host) inference.Admitter {
	return func(ctx context.Context, _ *router.Host, model string) (string, bool) {
		if _, err := h.Ensure(ctx, model); err != nil {
			word := WordOf(err)
			if word == "" {
				word = Unavailable
			}
			return word, false
		}
		return "", true
	}
}

func modelhostInference(t *testing.T, h *Host, admit inference.Admitter, decide inference.Decider, client *http.Client, profiles ...string) *inference.Provider {
	t.Helper()
	r := router.New()
	engine := router.OpenAICompatible("modelhost", h.Base("tiny:1b"), "/v1", h.Models())
	engine.Profiles = append([]string(nil), profiles...)
	r.SetHosts(engine)
	p, err := inference.New(inference.Config{
		Router: r,
		Admit:  admit,
		Decide: decide,
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		HTTP: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

type heldDecision struct{ release chan struct{} }

type decisionBarrier struct{ calls chan *heldDecision }

func (b *decisionBarrier) admit(model *Host) inference.Admitter {
	base := modelhostAdmit(model)
	return func(ctx context.Context, host *router.Host, name string) (string, bool) {
		if reason, ok := base(ctx, host, name); !ok {
			return reason, false
		}
		call := &heldDecision{release: make(chan struct{})}
		select {
		case b.calls <- call:
		case <-ctx.Done():
			return "caller", false
		}
		select {
		case <-call.release:
			return "", true
		case <-ctx.Done():
			return "caller", false
		}
	}
}

func receiveHeldDecision(t *testing.T, calls <-chan *heldDecision) *heldDecision {
	t.Helper()
	select {
	case call := <-calls:
		return call
	case <-time.After(5 * time.Second):
		t.Fatal("inference Start did not reach the admission barrier")
		return nil
	}
}

func receiveStart(t *testing.T, starts <-chan infwire.Admission) infwire.Admission {
	t.Helper()
	select {
	case admission := <-starts:
		return admission
	case <-time.After(5 * time.Second):
		t.Fatal("inference Start did not return")
		return infwire.Admission{}
	}
}

// A Start can load its model in Admit before it enters the inference
// provider's active-operation map. A yield in that interval must preserve
// the resident that the pending Start is about to use.
func TestPendingInferenceStartBlocksModelYield(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var provider *inference.Provider
	h := composed(t, leases, func(o *Options) {
		o.Active = func(model string) bool {
			return provider != nil && provider.ActiveForModel(model)
		}
	})

	barrier := &decisionBarrier{calls: make(chan *heldDecision, 1)}
	provider = modelhostInference(t, h, barrier.admit(h), func(context.Context, inference.Subject, string, string) (string, error) {
		return "permitted", nil
	}, engineHTTP(h), router.ProfileChat)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	started := make(chan infwire.Admission, 1)
	go func() {
		started <- provider.Start(ctx, inference.Subject{Account: "test-account", Program: "test-caller"}, infwire.Request{
			Model:    "tiny:1b",
			Messages: []infwire.Message{{Role: infwire.RoleUser, Parts: []infwire.Part{{Kind: infwire.PartKindText, Text: "hello"}}}},
		})
	}()
	held := receiveHeldDecision(t, barrier.calls)
	if !provider.ActiveForModel("tiny:1b") {
		t.Fatal("the pending Start was not visible as model activity")
	}
	lease := h.Lease("tiny:1b")
	answer, reason := h.yield(res.YieldRequest{ID: "ask-pending-start", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	close(held.release)
	var admission infwire.Admission
	select {
	case admission = <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("Start did not finish after the decision barrier opened")
	}
	if admission.Outcome != infwire.StartOutcomeAccepted {
		t.Fatalf("pending Start was not accepted after its reservation protected the model: %+v", admission)
	}
	if answer != res.YieldAnswerRefused || !h.Loaded("tiny:1b") {
		t.Fatalf("pending Start must keep its resident; yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
	if !provider.ActiveForModel("tiny:1b") {
		t.Fatal("accepted operation was not active after its reservation transferred")
	}
	answer, reason = h.yield(res.YieldRequest{ID: "ask-registered-operation", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerRefused || !h.Loaded("tiny:1b") {
		t.Fatalf("registered operation must keep its resident; yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
}

// Overlapping admissions each own one activity reservation. Cancelling one
// request leaves its sibling protected; after both refusals, the count is
// cleared and the resident can yield.
func TestPendingInferenceAdmissionsCountAndCleanUpIndependently(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var provider *inference.Provider
	h := composed(t, leases, func(o *Options) {
		o.Active = func(model string) bool {
			return provider != nil && provider.ActiveForModel(model)
		}
	})
	barrier := &decisionBarrier{calls: make(chan *heldDecision, 2)}
	provider = modelhostInference(t, h, barrier.admit(h), func(context.Context, inference.Subject, string, string) (string, error) {
		return "permitted", nil
	}, engineHTTP(h), router.ProfileChat)

	firstCtx, cancelFirst := context.WithCancel(context.Background())
	defer cancelFirst()
	first := make(chan infwire.Admission, 1)
	go func() {
		first <- provider.Start(firstCtx, inference.Subject{Account: "first-account", Program: "first-caller"}, infwire.Request{
			Model: "tiny:1b", Messages: []infwire.Message{{Role: infwire.RoleUser, Parts: []infwire.Part{{Kind: infwire.PartKindText, Text: "first"}}}},
		})
	}()
	firstDecision := receiveHeldDecision(t, barrier.calls)

	secondCtx, cancelSecond := context.WithCancel(context.Background())
	defer cancelSecond()
	second := make(chan infwire.Admission, 1)
	go func() {
		second <- provider.Start(secondCtx, inference.Subject{Account: "second-account", Program: "second-caller"}, infwire.Request{
			Model: "tiny:1b", Messages: []infwire.Message{{Role: infwire.RoleUser, Parts: []infwire.Part{{Kind: infwire.PartKindText, Text: "second"}}}},
		})
	}()
	secondDecision := receiveHeldDecision(t, barrier.calls)
	defer close(firstDecision.release)
	defer close(secondDecision.release)

	cancelFirst()
	if admission := receiveStart(t, first); admission.Outcome == infwire.StartOutcomeAccepted {
		t.Fatalf("cancelled first Start was accepted: %+v", admission)
	}
	if !provider.ActiveForModel("tiny:1b") {
		t.Fatal("cancelling one of two reservations cleared the remaining model activity")
	}
	lease := h.Lease("tiny:1b")
	answer, reason := h.yield(res.YieldRequest{ID: "ask-one-cancelled", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerRefused || !h.Loaded("tiny:1b") {
		t.Fatalf("the second pending Start lost protection: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}

	cancelSecond()
	if admission := receiveStart(t, second); admission.Outcome == infwire.StartOutcomeAccepted {
		t.Fatalf("cancelled second Start was accepted: %+v", admission)
	}
	if provider.ActiveForModel("tiny:1b") {
		t.Fatal("cancelled admissions left model activity behind")
	}
	answer, reason = h.yield(res.YieldRequest{ID: "ask-no-pending-starts", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerYielded || h.Loaded("tiny:1b") {
		t.Fatalf("resident stayed protected after both starts ended: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
}

func TestAdmitRefusalReleasesModelActivity(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var provider *inference.Provider
	h := composed(t, leases, func(o *Options) {
		o.Active = func(model string) bool {
			return provider != nil && provider.ActiveForModel(model)
		}
	})
	admit := func(ctx context.Context, host *router.Host, model string) (string, bool) {
		if _, ok := modelhostAdmit(h)(ctx, host, model); !ok {
			return Unavailable, false
		}
		return "fixture-refusal", false
	}
	provider = modelhostInference(t, h, admit,
		func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		engineHTTP(h), router.ProfileChat)

	admission := provider.Start(context.Background(), inference.Subject{Account: "test-account", Program: "test-caller"}, infwire.Request{
		Model: "tiny:1b", Messages: []infwire.Message{{Role: infwire.RoleUser, Parts: []infwire.Part{{Kind: infwire.PartKindText, Text: "hello"}}}},
	})
	if admission.Outcome != infwire.StartOutcomeUnavailable || admission.Reason != "fixture-refusal" {
		t.Fatalf("Admit refusal changed: %+v", admission)
	}
	if provider.ActiveForModel("tiny:1b") {
		t.Fatal("refused Admit left a model activity reservation")
	}
	answer, reason := h.yield(res.YieldRequest{ID: "ask-after-admit-refusal", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: h.Lease("tiny:1b")})
	if answer != res.YieldAnswerYielded || h.Loaded("tiny:1b") {
		t.Fatalf("refused Admit kept its resident: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
}

type embeddingBarrier struct {
	entered     chan struct{}
	release     chan struct{}
	releaseOnce sync.Once
}

func (b *embeddingBarrier) open() { b.releaseOnce.Do(func() { close(b.release) }) }

func (b *embeddingBarrier) RoundTrip(request *http.Request) (*http.Response, error) {
	b.entered <- struct{}{}
	select {
	case <-b.release:
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(`{"data":[{"index":0,"embedding":[0.25,-0.5]}],"usage":{"prompt_tokens":1}}`)), Request: request}, nil
	case <-request.Context().Done():
		return nil, request.Context().Err()
	}
}

// Embed has no operation row, so its admission activity must stay registered
// through the synchronous upstream call and clear on return.
func TestEmbedProtectsModelThroughSynchronousCall(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var provider *inference.Provider
	h := composed(t, leases, func(o *Options) {
		o.Active = func(model string) bool {
			return provider != nil && provider.ActiveForModel(model)
		}
	})
	upstream := &embeddingBarrier{entered: make(chan struct{}, 1), release: make(chan struct{})}
	defer upstream.open()
	provider = modelhostInference(t, h, modelhostAdmit(h),
		func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		&http.Client{Transport: upstream}, router.ProfileEmbed)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	embedded := make(chan infwire.Embeddings, 1)
	go func() {
		embedded <- provider.Embed(ctx, inference.Subject{Account: "embed-account", Program: "embed-caller"}, infwire.EmbedRequest{
			Model: "tiny:1b", Inputs: []string{"hello"},
		})
	}()
	select {
	case <-upstream.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("Embed did not reach its controlled upstream call")
	}
	lease := h.Lease("tiny:1b")
	if !provider.ActiveForModel("tiny:1b") {
		t.Fatal("synchronous Embed was not visible as model activity")
	}
	answer, reason := h.yield(res.YieldRequest{ID: "ask-during-embed", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerRefused || !h.Loaded("tiny:1b") {
		t.Fatalf("active Embed did not protect its resident: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
	upstream.open()
	select {
	case result := <-embedded:
		if result.Outcome != infwire.EmbedOutcomeCompleted || len(result.Vectors) != 1 {
			t.Fatalf("controlled Embed result %+v", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Embed did not return after the upstream barrier opened")
	}
	if provider.ActiveForModel("tiny:1b") {
		t.Fatal("completed synchronous Embed left a model activity reservation")
	}
	answer, reason = h.yield(res.YieldRequest{ID: "ask-after-embed", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerYielded || h.Loaded("tiny:1b") {
		t.Fatalf("completed Embed kept its resident: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
}

func TestEmbedCancellationReleasesModelActivity(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	var provider *inference.Provider
	h := composed(t, leases, func(o *Options) {
		o.Active = func(model string) bool {
			return provider != nil && provider.ActiveForModel(model)
		}
	})
	upstream := &embeddingBarrier{entered: make(chan struct{}, 1), release: make(chan struct{})}
	defer upstream.open()
	provider = modelhostInference(t, h, modelhostAdmit(h),
		func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		&http.Client{Transport: upstream}, router.ProfileEmbed)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	embedded := make(chan infwire.Embeddings, 1)
	go func() {
		embedded <- provider.Embed(ctx, inference.Subject{Account: "embed-account", Program: "embed-caller"}, infwire.EmbedRequest{
			Model: "tiny:1b", Inputs: []string{"hello"},
		})
	}()
	select {
	case <-upstream.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("Embed did not reach its controlled upstream call")
	}
	lease := h.Lease("tiny:1b")
	if !provider.ActiveForModel("tiny:1b") {
		t.Fatal("synchronous Embed was not visible as model activity")
	}
	cancel()
	select {
	case result := <-embedded:
		if result.Outcome == infwire.EmbedOutcomeCompleted {
			t.Fatalf("cancelled Embed completed: %+v", result)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Embed did not return after cancellation")
	}
	if provider.ActiveForModel("tiny:1b") {
		t.Fatal("cancelled synchronous Embed left a model activity reservation")
	}
	answer, reason := h.yield(res.YieldRequest{ID: "ask-after-cancelled-embed", Resource: DefaultResource,
		Amount: 4096, Reason: "competing caller", Lease: lease})
	if answer != res.YieldAnswerYielded || h.Loaded("tiny:1b") {
		t.Fatalf("cancelled Embed kept its resident: yield answered %s (%s), loaded=%v", answer, reason, h.Loaded("tiny:1b"))
	}
}

// Every refusal this provider can give, in its own words.
func TestRefusals(t *testing.T) {
	t.Run("no_engine when none is configured", func(t *testing.T) {
		h := composed(t, newFakeLeases(), func(o *Options) { o.Engine.Program = "" })
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if WordOf(err) != NoEngine {
			t.Fatalf("refusal %v, wanted %s", err, NoEngine)
		}
	})
	t.Run("no_engine when the configured path is not there", func(t *testing.T) {
		missing := filepath.Join(t.TempDir(), "llama-server-that-is-not-here")
		h := composed(t, newFakeLeases(), func(o *Options) { o.Engine.Program = missing })
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if WordOf(err) != NoEngine {
			t.Fatalf("refusal %v, wanted %s", err, NoEngine)
		}
	})
	t.Run("the engine is not started before the lease is refused", func(t *testing.T) {
		leases := newFakeLeases()
		leases.outcome = res.AcquireOutcomeNotPermitted
		h := composed(t, leases, nil)
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if WordOf(err) != NotPermitted {
			t.Fatalf("refusal %v, wanted %s", err, NotPermitted)
		}
		if h.Loaded("tiny:1b") {
			t.Fatal("a refused lease still loaded a model")
		}
	})
	for _, c := range []struct {
		outcome res.AcquireOutcome
		word    string
	}{
		{res.AcquireOutcomeInsufficient, Insufficient},
		{res.AcquireOutcomeHoldersRefused, HoldersRefused},
		{res.AcquireOutcomeInvalid, Unavailable},
	} {
		t.Run(c.outcome.String(), func(t *testing.T) {
			leases := newFakeLeases()
			leases.outcome = c.outcome
			h := composed(t, leases, nil)
			_, err := h.Ensure(context.Background(), "tiny:1b")
			if WordOf(err) != c.word {
				t.Fatalf("refusal %v, wanted %s", err, c.word)
			}
		})
	}
	t.Run("unavailable when the lease service does not answer", func(t *testing.T) {
		leases := newFakeLeases()
		leases.err = errors.New("no lease service")
		h := composed(t, leases, nil)
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if WordOf(err) != Unavailable {
			t.Fatalf("refusal %v, wanted %s", err, Unavailable)
		}
	})
	t.Run("lease ended when the service ends the lease during the engine's start", func(t *testing.T) {
		// RES-L4: the holder reloads once; a second ending is an acquire
		// failure wrapping holder.ErrLeaseEnded, and the service answered.
		h := composed(t, newFakeLeases(), nil)
		ended := fmt.Errorf("ensure: %w", &holder.Failure{Stage: holder.StageAcquire, Err: holder.ErrLeaseEnded})
		err := h.mapFailure("tiny:1b", 4096, ended)
		if WordOf(err) != LeaseEnded {
			t.Fatalf("refusal %v, wanted %s", err, LeaseEnded)
		}
		if !strings.Contains(err.Error(), "ended the lease while the engine was starting") || strings.Contains(err.Error(), "did not answer") {
			t.Fatalf("refusal detail %q", err.Error())
		}
		// Any other acquire failure without an outcome still reads unavailable.
		other := &holder.Failure{Stage: holder.StageAcquire, Err: errors.New("no lease service")}
		if word := WordOf(h.mapFailure("tiny:1b", 4096, other)); word != Unavailable {
			t.Fatalf("a silent lease service read %s", word)
		}
	})
	t.Run("unknown object when no store holds the object", func(t *testing.T) {
		// CONTRACT.md MODELHOST-L1, D87 (providers review F12): the allowlist
		// names a store and an id no source resolves, the reason travels as
		// object:unknown:<store>/<id>, the same family the lend seam uses.
		h := composed(t, newFakeLeases(), func(o *Options) { o.Source = fakeSource{objects: map[string]Object{}} })
		_, err := h.Ensure(context.Background(), "tiny:1b")
		want := unknownObject("fake", "tiny")
		if WordOf(err) != want {
			t.Fatalf("refusal %v, wanted %s", err, want)
		}
	})
	t.Run("pickle object when the allowlist names a pickle by locator", func(t *testing.T) {
		// CONTRACT.md MODELHOST-L1, modelbridge MODELBRIDGE-L20, storage LS-1:
		// OA never loads a pickle; the shared source refuses to bind one, and
		// the host refuses before any lease.
		home := t.TempDir()
		root := filepath.Join(home, "packages")
		if err := os.MkdirAll(filepath.Join(root, "esrgan"), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "esrgan", "4x.pth"), make([]byte, 4096), 0o600); err != nil {
			t.Fatal(err)
		}
		env := Env{GOOS: runtime.GOOS, Home: home, Getenv: func(string) string { return "" }}
		env.Config.ExtraRoots = []source.ExtraRoot{{Store: "packages", Path: root}}
		leases := newFakeLeases()
		h := composed(t, leases, func(o *Options) {
			o.Models = []Model{{Name: "tiny:1b", Store: "packages", Object: "esrgan/4x.pth"}}
			o.Source = NewLocalStores(env)
		})
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if want := pickleObject("packages", "esrgan/4x.pth"); WordOf(err) != want {
			t.Fatalf("refusal %v, wanted %s", err, want)
		}
		if leases.acquires != 0 {
			t.Fatalf("a lease was taken for a pickle: %d", leases.acquires)
		}
	})
	t.Run("unbound object when it lies inside a zip", func(t *testing.T) {
		// CONTRACT.md MODELHOST-L1, modelbridge MODELBRIDGE-L19: llama-server
		// reads a whole file, so a zip member is refused before any lease.
		leases := newFakeLeases()
		var asked []bridge.Accepts
		h := composed(t, leases, func(o *Options) {
			o.Source = fakeSource{unbound: map[string]bool{"fake/tiny": true}, asked: &asked}
		})
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if want := unboundObject("fake", "tiny"); WordOf(err) != want {
			t.Fatalf("refusal %v, wanted %s", err, want)
		}
		if leases.acquires != 0 {
			t.Fatalf("a lease was taken for an object no engine can read: %d", leases.acquires)
		}
		if len(asked) != 1 || !reflect.DeepEqual(asked[0], bridge.FileOnly) {
			t.Fatalf("the host asked for %+v; llama-server reads a file only", asked)
		}
	})
	t.Run("unavailable for a model outside the allowlist", func(t *testing.T) {
		h := composed(t, newFakeLeases(), nil)
		_, err := h.Ensure(context.Background(), "something-else")
		if WordOf(err) != Unavailable {
			t.Fatalf("refusal %v, wanted %s", err, Unavailable)
		}
	})
}

// A model that is still loading is waited for, and one that never loads is
// unavailable rather than a call that hangs.
func TestTheEngineIsWaitedForAndBounded(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-load", "400ms"}
	})
	began := time.Now()
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("a loading engine: %v", err)
	}
	if took := time.Since(began); took < 400*time.Millisecond {
		t.Fatalf("the call returned after %s, before the engine was ready", took)
	}
	h.Unload("tiny:1b", "test")

	stubborn := newFakeLeases()
	slow := composed(t, stubborn, func(o *Options) {
		o.Engine.fixture = []string{"--fake-load", "30s"}
		o.Engine.Ready = 600 * time.Millisecond
	})
	_, err := slow.Ensure(context.Background(), "tiny:1b")
	if WordOf(err) != Unavailable {
		t.Fatalf("a model that never loads: %v, wanted %s", err, Unavailable)
	}
	if slow.Loaded("tiny:1b") {
		t.Fatal("a model that never loaded is held")
	}
	// A load that did not finish gives the card back; a grant nobody occupied
	// is not left standing between the next asker and the bytes.
	if _, released, _ := stubborn.counts(); len(released) != 1 {
		t.Fatalf("a failed load released %v, wanted the lease it took", released)
	}
}
