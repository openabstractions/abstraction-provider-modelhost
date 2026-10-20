package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/user"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	res "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"
)

type nativeTitleBook struct {
	granting
	held         bool
	normal, free atomic.Int32
}

func (b *nativeTitleBook) Acquire(resource string, amount int64, _ time.Duration) (res.AcquireResult, error) {
	b.normal.Add(1)
	return res.AcquireResult{Outcome: res.AcquireOutcomeAcquired, Lease: &res.Lease{ID: "main-lease", Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}
func (b *nativeTitleBook) AcquireFree(resource string, amount int64) (res.AcquireFreeResult, error) {
	b.free.Add(1)
	if b.held {
		return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeHeld, Asked: []res.YieldRecord{}}, nil
	}
	return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeAcquired, Lease: &res.Lease{ID: "free-title", Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}

type titleObjects struct{ oneObject }

func (s titleObjects) Resolve(store, id string, a bridge.Accepts) (host.Object, error) {
	if store == s.object.Store && id == "main" {
		object := s.object
		object.ID = id
		return object, nil
	}
	return s.oneObject.Resolve(store, id, a)
}

func nativeTitleIdentity(t *testing.T) (string, identity.User) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	current, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	principal := identity.User{Kind: "posix", UID: os.Getuid()}
	if runtime.GOOS == "windows" {
		principal = identity.User{Kind: "windows", SID: current.Uid}
	}
	return identity.NormalizeSubjectProgram(self), principal
}

func titleRuntime(t *testing.T, endpoint, program string, principal identity.User) *inference.Provider {
	t.Helper()
	native, err := router.NewNative(declaration.ProviderName, endpoint, listen.ServerExpectation{Principal: principal, Program: program}, []string{"tiny:1b"}, []string{router.ProfileChat})
	if err != nil {
		t.Fatal(err)
	}
	p, err := inference.New(inference.Config{ChatLogicalOwner: "fixture-runtime-job-owner", Router: router.New(native),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			t.Error("local title applied credential")
			return nil, "not_permitted"
		},
		ResolveSelectionV2: func(_ context.Context, _ inference.Subject, name string) (wire.SelectionPolicyV2, bool, error) {
			return wire.SelectionPolicyV2{Name: name, Candidates: []wire.SelectionCandidateV2{{Model: "tiny:1b", Placement: wire.CandidatePlacementLocal}}}, name == "title", nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := p.Close(); err != nil {
			t.Error(err)
		}
	})
	return p
}

func titleRequest(p *inference.Provider, subject inference.Subject, key string) wire.RequestV3 {
	window := p.GetChatHistoryWindow(subject)
	return wire.RequestV3{Identity: acceptance.RequestIdentity{Key: key, HistoryEpoch: window.Window.HistoryEpoch}, Target: wire.ChatTarget{SelectionPolicy: "title"},
		Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "write a title"}}}}, Guarantees: []wire.ChatGuarantee{wire.ChatGuaranteeLocalOnly, wire.ChatGuaranteeFreeCapacity}}}
}

func titleResult(t *testing.T, p *inference.Provider, subject inference.Subject, request string) wire.ChatResult {
	t.Helper()
	cursor := int64(0)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	for ctx.Err() == nil {
		page := p.ObserveV3(ctx, subject, request, cursor, 256, 65536, 1000)
		if page.Outcome != wire.PageOutcomePage {
			t.Fatalf("observe=%+v", page)
		}
		for _, event := range page.Events {
			if event.End != nil {
				return *event.End
			}
		}
		cursor = page.Next
	}
	t.Fatal("title did not finish")
	return wire.ChatResult{}
}

func TestChat3NativeTitleRelayPreservesMainLeaseAndNoYield(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS != "windows" {
		t.Skip("native engine peer proof requires Windows")
	}
	program, principal := nativeTitleIdentity(t)
	account, err := principal.Account()
	if err != nil {
		t.Fatal(err)
	}
	engine := buildFakeEngine(t)
	for _, held := range []bool{true, false} {
		t.Run(fmt.Sprintf("held-%v", held), func(t *testing.T) {
			book := &nativeTitleBook{granting: granting{released: make(chan string, 8)}, held: held}
			transport, model := serveTransport(t, engine, program, func(o *host.Options) {
				o.Leases = book
				o.Source = titleObjects{o.Source.(oneObject)}
				o.Models = append(o.Models, host.Model{Name: "main:2b", Store: "fake", Object: "main"})
			})
			window, err := wire.NewChatV3Client(transport).GetHistoryWindow()
			if err != nil || window.Window == nil || window.Window.LogicalOwner != declaration.ProviderName {
				t.Fatalf("leaf window=%+v %v", window, err)
			}
			if _, err := model.Ensure(context.Background(), "main:2b"); err != nil {
				t.Fatal(err)
			}
			mainLease := model.Lease("main:2b")
			mediator := titleRuntime(t, transport.Endpoint, program, principal)
			subject := inference.Subject{Account: account, Program: program}
			r := titleRequest(mediator, subject, "title")
			a := mediator.StartV3(context.Background(), subject, r)
			if a.Outcome != wire.ChatStartOutcomeAccepted {
				t.Fatalf("runtime admission=%+v", a)
			}
			result := titleResult(t, mediator, subject, a.Request)
			if held {
				if result.Outcome != wire.ChatResultOutcomeFailed || result.Cause != "remote:unavailable" || result.Refusal != nil || model.Loaded("tiny:1b") {
					t.Fatalf("held result=%+v loaded=%v", result, model.Loaded("tiny:1b"))
				}
			} else if result.Outcome != wire.ChatResultOutcomeCompleted || result.Cause != "" || result.Refusal != nil {
				t.Fatalf("granting result=%+v", result)
			}
			if book.normal.Load() != 1 || book.free.Load() != 1 || model.Lease("main:2b") != mainLease || !model.Loaded("main:2b") {
				t.Fatalf("normal=%d free=%d main=%q/%q", book.normal.Load(), book.free.Load(), mainLease, model.Lease("main:2b"))
			}
			select {
			case released := <-book.released:
				t.Fatalf("request released main lease: %q", released)
			default:
			}
		})
	}
}

type restartedLeaf struct {
	mu                 sync.Mutex
	client             *wire.ChatV3Client
	stop               func()
	start              func() (*wire.ChatV3Client, func(), *router.Router, error)
	routes             *router.Router
	starts, unknown    atomic.Int32
	oldEpoch, newEpoch string
}

func (g *restartedLeaf) GetHistoryWindow() (wire.ChatHistoryRead, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	old, err := g.client.GetHistoryWindow()
	if err != nil {
		return old, err
	}
	g.stop()
	client, stop, routes, err := g.start()
	if err != nil {
		return wire.ChatHistoryRead{}, err
	}
	g.client, g.stop, g.routes = client, stop, routes
	fresh, err := client.GetHistoryWindow()
	if err != nil {
		return fresh, err
	}
	if old.Window == nil || fresh.Window == nil || old.Window.LogicalOwner != fresh.Window.LogicalOwner || old.Window.HistoryEpoch == fresh.Window.HistoryEpoch {
		return wire.ChatHistoryRead{}, fmt.Errorf("restart failed to replace the volatile epoch")
	}
	g.oldEpoch, g.newEpoch = old.Window.HistoryEpoch, fresh.Window.HistoryEpoch
	return old, nil
}
func (g *restartedLeaf) Start(r wire.RequestV3) (wire.ChatAdmission, error) {
	g.mu.Lock()
	client := g.client
	g.mu.Unlock()
	g.starts.Add(1)
	a, err := client.Start(r)
	if a.Outcome == wire.ChatStartOutcomeUnknown {
		g.unknown.Add(1)
	}
	return a, err
}
func (g *restartedLeaf) Observe(id string, cursor, maxEvents, maxBytes, wait int64) (wire.ChatPage, error) {
	g.mu.Lock()
	client := g.client
	g.mu.Unlock()
	return client.Observe(id, cursor, maxEvents, maxBytes, wait)
}
func (g *restartedLeaf) Cancel(id string) (wire.ChatCancellation, error) {
	g.mu.Lock()
	client := g.client
	g.mu.Unlock()
	return client.Cancel(id)
}

// The gate models a leaf process restart between the relay's history read and
// its Start. Both generations are the real modelhost Listen/Serve owner.
func TestChat3NativeLeafRestartIsUnknownWithoutSecondStart(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS == "darwin" {
		t.Skip("Program proof unavailable")
	}
	program, principal := nativeTitleIdentity(t)
	account, err := principal.Account()
	if err != nil {
		t.Fatal(err)
	}
	leafEndpoint := endpointFor(t)
	start := func() (*wire.ChatV3Client, func(), *router.Router, error) {
		routes := router.New()
		p, err := inference.New(ProviderConfig(nil, routes, nil))
		if err != nil {
			return nil, nil, nil, err
		}
		h, err := Listen(leafEndpoint, p, program)
		if err != nil {
			p.Close()
			return nil, nil, nil, err
		}
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() { done <- h.Serve(ctx) }()
		var once sync.Once
		stop := func() { once.Do(func() { cancel(); <-done; p.Close() }) }
		return wire.NewChatV3Client(listen.FrameClient{Endpoint: leafEndpoint}.WithDefaults(5*time.Second, MaxFrameBytes)), stop, routes, nil
	}
	client, stop, routes, err := start()
	if err != nil {
		t.Fatal(err)
	}
	gate := &restartedLeaf{client: client, stop: stop, start: start, routes: routes}
	t.Cleanup(func() { gate.mu.Lock(); stop := gate.stop; gate.mu.Unlock(); stop() })
	endpoint := leafEndpoint + "-restart-gate"
	listener, err := listen.ListenFramed(endpoint, listen.Program)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := listener.Accept()
			if err != nil {
				return
			}
			func() {
				defer conn.Close()
				call, err := listen.ReceiveFramed(ctx, conn, listen.Program, MaxFrameBytes)
				if call != nil {
					defer call.Close()
				}
				if err != nil {
					if ctx.Err() == nil {
						t.Error(err)
					}
					return
				}
				reply, err := wire.ServeEndpoint(call.Frame, "fixture-restart-gate", "", &wire.ChatV3Dispatcher{Handler: gate})
				if err == nil {
					err = call.Reply(reply)
				}
				if err != nil && !errors.Is(err, context.Canceled) {
					t.Error(err)
				}
			}()
		}
	}()
	t.Cleanup(func() { cancel(); listener.Close(); <-done })
	mediator := titleRuntime(t, endpoint, program, principal)
	subject := inference.Subject{Account: account, Program: program}
	r := titleRequest(mediator, subject, "restart")
	a := mediator.StartV3(context.Background(), subject, r)
	if a.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("admission=%+v", a)
	}
	result := titleResult(t, mediator, subject, a.Request)
	if result.Outcome != wire.ChatResultOutcomeFailed || result.Cause != "remote:uncertain" || result.Refusal != nil {
		t.Fatalf("result=%+v", result)
	}
	again := mediator.StartV3(context.Background(), subject, r)
	if again.Request != a.Request || again.Outcome != wire.ChatStartOutcomeAccepted || gate.starts.Load() != 1 || gate.unknown.Load() != 1 {
		t.Fatalf("again=%+v starts=%d unknown=%d", again, gate.starts.Load(), gate.unknown.Load())
	}
	gate.mu.Lock()
	defer gate.mu.Unlock()
	if gate.oldEpoch == "" || gate.oldEpoch == gate.newEpoch || len(gate.routes.Asked()) != 0 {
		t.Fatalf("epochs=%q/%q routed=%v", gate.oldEpoch, gate.newEpoch, gate.routes.Asked())
	}
}
