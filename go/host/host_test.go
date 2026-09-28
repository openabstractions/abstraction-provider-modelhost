package host

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	res "github.com/openabstractions/abstraction-resource/go/client"
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

func engineProgram(t *testing.T) string {
	t.Helper()
	program, err := fakeEngine()
	if err != nil {
		t.Fatal(err)
	}
	return program
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

type fakeSource struct{ objects map[string]Object }

func (f fakeSource) Resolve(store, id string) (Object, bool) {
	o, ok := f.objects[store+"/"+id]
	return o, ok
}
func (f fakeSource) Stores() []string   { return []string{"fake"} }
func (f fakeSource) Root(string) string { return "" }

// fakeLeases is the lease service as this provider sees it: it grants, it
// records renewals and releases, and it can put one yield request to the
// holder the way the book does.
type fakeLeases struct {
	mu          sync.Mutex
	outcome     res.AcquireOutcome
	err         error
	acquires    int
	renews      int
	released    []string
	answered    []string
	events      []string
	live        map[string]bool
	seq         int
	bindings    map[string]int
	bindOutcome res.ProcessBindOutcome

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
	if f.outcome != res.AcquireOutcomeAcquired {
		return res.AcquireResult{Outcome: f.outcome, Asked: []res.YieldRecord{{Holder: "host:lmstudio", Answer: "refused: no rule"}}}, nil
	}
	f.seq++
	id := "lease-" + strconv.Itoa(f.seq)
	f.live[id] = true
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
		Engine: Engine{Program: engineProgram(t), Ready: 30 * time.Second},
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
		Engine: Engine{Program: engineProgram(t), Ready: 30 * time.Second},
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
func answered(t *testing.T, base string) string {
	t.Helper()
	response, err := http.Get(base + "/v1/models")
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
	t.Logf("first call: lease %s, engine pid %d at %s, models %s", h.Lease("tiny:1b"), first, base, answered(t, base))

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
	h.mu.Lock()
	defer h.mu.Unlock()
	r, ok := h.residents[name]
	if !ok {
		return 0
	}
	return r.process.PID()
}

// A model nothing has asked for is given back, with its lease.
func TestIdleUnloadsAndReleasesTheLease(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Idle = 120 * time.Millisecond; o.Renew = 30 * time.Millisecond })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	lease := h.Lease("tiny:1b")
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
	if _, err := http.Get(base + "/v1/models"); err == nil {
		t.Fatal("the engine still answers after the yield")
	}
}

// A second object asked for while the first is resident gets its own engine
// on its own port: it neither waits for the first nor displaces it, and each
// takes its own lease (CONTRACT.md MODELHOST-H5).
func TestTwoResidentsLoadIndependently(t *testing.T) {
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
		t.Fatalf("both residents answer at %s; each resident needs its own port", firstBase)
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
	if _, err := http.Get(firstBase + "/v1/models"); err != nil {
		t.Fatalf("tiny:1b's own engine did not answer: %v", err)
	}
	if _, err := http.Get(secondBase + "/v1/models"); err != nil {
		t.Fatalf("small:2b's own engine did not answer: %v", err)
	}
}

// Each resident keeps its own idle clock: the one loaded first goes idle and
// is unloaded first, while a later resident that has not reached its own
// idle time yet stays loaded (CONTRACT.md MODELHOST-H5, MODELHOST-H8).
func TestEachResidentHasItsOwnIdleClock(t *testing.T) {
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

func TestYieldKeepsLeaseWhenEngineExitIsUnconfirmed(t *testing.T) {
	leases := newFakeLeases()
	h := &Host{options: Options{Resource: "card:0", Leases: leases}, residents: map[string]*resident{
		"tiny:1b": {model: "tiny:1b", lease: "lease-1", amount: 4096, process: &Process{exited: make(chan struct{})}},
	}}
	answer, reason := h.yield(res.YieldRequest{ID: "ask-1", Resource: "card:0", Amount: 4096, Lease: "lease-1"})
	_, released, _ := leases.counts()
	if answer != res.YieldAnswerRefused || reason != "engine did not stop" || !h.Loaded("tiny:1b") || len(released) != 0 {
		t.Fatalf("unconfirmed engine exit: answer %s (%s), released %v, loaded %v", answer, reason, released, h.Loaded("tiny:1b"))
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
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.Arguments = []string{"--fake-load", "400ms"}
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
		o.Engine.Arguments = []string{"--fake-load", "30s"}
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
