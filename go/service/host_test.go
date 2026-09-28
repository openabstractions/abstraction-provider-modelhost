package service

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	res "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// granting is a lease service that grants whatever is asked for. What a
// refused acquire does is the host package's own test; this one measures the
// contract boundary.
type granting struct{ released chan string }

func (g granting) Acquire(resource string, amount int64, _ time.Duration) (res.AcquireResult, error) {
	return res.AcquireResult{Outcome: res.AcquireOutcomeAcquired,
		Lease: &res.Lease{ID: "lease-1", Resource: resource, Amount: amount}, Asked: []res.YieldRecord{}}, nil
}
func (g granting) Renew(string) (res.LeaseChange, error) { return res.LeaseChange{Applied: true}, nil }
func (g granting) BindProcess(string, int) (res.ProcessBindResult, error) {
	return res.ProcessBindResult{Outcome: res.ProcessBindOutcomeBound}, nil
}
func (g granting) Release(lease string) (res.LeaseChange, error) {
	select {
	case g.released <- lease:
	default:
	}
	return res.LeaseChange{Applied: true}, nil
}
func (g granting) Observe(ctx context.Context, cursor string, wait time.Duration) (res.YieldPage, error) {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
	return res.YieldPage{Requests: []res.YieldRequest{}, Next: cursor}, nil
}
func (g granting) Answer(context.Context, string, res.YieldAnswer, string) (res.AnswerResult, error) {
	return res.AnswerResult{Recorded: true}, nil
}

type oneObject struct{ object host.Object }

func (o oneObject) Resolve(store, id string) (host.Object, bool) {
	if store == o.object.Store && id == o.object.ID {
		return o.object, true
	}
	return host.Object{}, false
}
func (o oneObject) Stores() []string   { return []string{o.object.Store} }
func (o oneObject) Root(string) string { return "" }

func endpointFor(t *testing.T) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		return fmt.Sprintf(`\\.\pipe\modelhost-test-%d-%d`, os.Getpid(), time.Now().UnixNano())
	}
	dir, err := os.MkdirTemp("", "modelhost-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return filepath.Join(dir, "s")
}

func buildFakeEngine(t *testing.T) string {
	t.Helper()
	name := "fakeengine"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	target := filepath.Join(t.TempDir(), name)
	build := exec.Command("go", "build", "-o", target, "../host/testdata/fakeengine")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fakeengine: %v\n%s", err, out)
	}
	return target
}

// serve composes the whole provider — the model host, the inference provider
// over the engine's address, and the endpoint — and returns a chat@1 client.
func serve(t *testing.T, engine string) (*wire.ChatClient, *host.Host) {
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return serveForProgram(t, engine, identity.CanonicalProgramPath(filepath.Clean(self)))
}

func serveForProgram(t *testing.T, engine, runtimeProgram string) (*wire.ChatClient, *host.Host) {
	t.Helper()
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS == "darwin" {
		t.Skip("Program proof unavailable")
	}
	weights := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(weights, make([]byte, 2048), 0o600); err != nil {
		t.Fatal(err)
	}
	var provider *inference.Provider
	model, err := host.New(host.Options{
		Models: []host.Model{{Name: "tiny:1b", Store: "fake", Object: "tiny"}},
		Engine: host.Engine{Program: engine, Ready: 30 * time.Second},
		Source: oneObject{object: host.Object{Store: "fake", ID: "tiny", Path: weights, File: "model.gguf", Size: 2048}},
		Leases: granting{released: make(chan string, 4)},
		Idle:   time.Hour,
		Active: func(name string) bool {
			if provider == nil {
				return false
			}
			return provider.ActiveForModel(name)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	r := router.New()
	r.SetHosts(router.OpenAICompatible("modelhost-engine", model.Base("tiny:1b"), "/v1", model.Models()))
	provider, err = inference.New(inference.Config{Router: r,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: Admit(model),
		HTTP:  &http.Client{}, Idle: 10 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	endpoint := endpointFor(t)
	served, err := Listen(endpoint, provider, runtimeProgram)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- served.Serve(ctx) }()
	t.Cleanup(func() { cancel(); <-done; provider.Close() })
	transport := listen.FrameClient{Endpoint: endpoint}.WithDefaults(35*time.Second, MaxFrameBytes)
	return wire.NewChatClient(transport), model
}

func TestSameAccountOtherProgramCannotCallModelHost(t *testing.T) {
	client, model := serveForProgram(t, buildFakeEngine(t), filepath.Join(t.TempDir(), "another-runtime"))
	admission, err := client.Start(chat())
	if err != nil || admission.Outcome != wire.StartOutcomeForbidden {
		t.Fatalf("unapproved Start = %+v, %v", admission, err)
	}
	if model.Loaded("tiny:1b") {
		t.Fatal("unapproved caller loaded the model")
	}
}

func TestListenRequiresRuntimeProgram(t *testing.T) {
	if h, err := Listen(endpointFor(t), nil); h != nil || err == nil {
		t.Fatalf("Listen without designation = %v, %v", h, err)
	}
}

func chat() wire.Request {
	return wire.Request{Model: "tiny:1b",
		Messages: []wire.Message{{Role: wire.RoleUser, Parts: []wire.Part{{Kind: wire.PartKindText, Text: "hi"}}}}}
}

// One chat@1 call loads the model and reads the engine's answer back through
// the contract. The load happened inside the call that needed it.
func TestChatLoadsTheModelAndMediatesToTheEngine(t *testing.T) {
	client, model := serve(t, buildFakeEngine(t))
	if model.Loaded("tiny:1b") {
		t.Fatal("a model was loaded before any call arrived")
	}
	admission, err := client.Start(chat())
	if err != nil || admission.Outcome != wire.StartOutcomeAccepted {
		t.Fatalf("start %+v %v", admission, err)
	}
	if !model.Loaded("tiny:1b") {
		t.Fatalf("after the call the host does not hold %q", "tiny:1b")
	}
	var text strings.Builder
	cursor := int64(0)
	for i := 0; i < 50; i++ {
		page, err := client.Observe(admission.Operation, cursor, 256, 65536, 2000)
		if err != nil || page.Outcome != wire.PageOutcomePage {
			t.Fatalf("observe %+v %v", page, err)
		}
		for _, d := range page.Deltas {
			if d.Kind == wire.DeltaKindPart {
				text.WriteString(d.Part.Text)
			}
		}
		cursor = page.Next
		if page.AtEnd {
			break
		}
	}
	if !strings.Contains(text.String(), "hosted by tiny:1b") {
		t.Fatalf("the engine's answer read %q", text.String())
	}
	t.Logf("chat@1 answered %q with lease %s", text.String(), model.Lease("tiny:1b"))
}

// A refusal reaches the caller in the host's own word, not as a transport
// failure: no engine is configured, so the call reads no_engine.
func TestChatRefusesWithTheHostsOwnWord(t *testing.T) {
	client, _ := serve(t, "")
	admission, err := client.Start(chat())
	if err != nil {
		t.Fatal(err)
	}
	if admission.Outcome != wire.StartOutcomeUnavailable || admission.Reason != host.NoEngine {
		t.Fatalf("start %+v, wanted unavailable with reason %s", admission, host.NoEngine)
	}
}
