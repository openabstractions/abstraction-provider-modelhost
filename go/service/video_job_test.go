package service

import (
	"bytes"
	"context"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	jobwire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/job"
	"github.com/openabstractions/abstraction-inference/go/providerhost"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	"github.com/openabstractions/abstraction-job/go/acceptanceprovider"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// A video job the runtime accepts reaches the model host through the
// runtime's job executor: a native router host with profile video and a
// video:<model> entry, as the runtime builds it from a declaration, is
// selected, the provider's video@1 runs the stand-in sd-server, and the
// executor commits the provider's WebM to the caller (INF-G3, INF-G5).
func TestVideoJobReachesTheModelHostThroughTheRuntime(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS == "darwin" {
		t.Skip("Program proof unavailable")
	}
	requireSocketOwnerTables(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	program := identity.CanonicalProgramPath(filepath.Clean(self))
	root := t.TempDir()
	content, err := providerhost.NewDesignatedContent("modelhost-video", filepath.Join(root, "content"), filepath.Join(root, "uploads.json"), []string{program}, 20<<20)
	if err != nil {
		t.Fatal(err)
	}
	const model = "fastwan2.2-ti2v-5b"
	modelHost, err := host.New(host.Options{Leases: granting{released: make(chan string, 4)}, Source: oneObject{}, Idle: time.Hour,
		Video: &host.VideoOptions{Model: model, Engine: stagedVideoEngine(t), Store: content.Store, Poll: 20 * time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(modelHost.Close)
	leaf, err := inference.New(ProviderConfig(modelHost, router.New(), nil))
	if err != nil {
		t.Fatal(err)
	}
	endpoint := endpointFor(t)
	served, err := Listen(endpoint, leaf, program)
	if err != nil {
		t.Fatal(err)
	}
	served.Content = content.Component
	served.Video = modelHost.VideoServer()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- served.Serve(ctx) }()
	t.Cleanup(func() { cancel(); <-done; leaf.Close() })

	// The runtime side: the router host the runtime builds from a
	// declaration listing profile video and the entry video:<model>.
	current, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	principal := identity.User{Kind: "posix", UID: os.Getuid()}
	if runtime.GOOS == "windows" {
		principal = identity.User{Kind: "windows", SID: current.Uid}
	}
	native, err := router.NewNative("modelhost", endpoint, listen.ServerExpectation{Principal: principal, Program: program},
		[]string{"video:" + model}, []string{router.ProfileVideo})
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var committed [][]byte
	mediator, err := inference.New(inference.Config{Router: router.New(native),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "not_permitted"
		},
		PrepareContentWrite: func(context.Context, inference.Subject, string, string, int64) (inference.ContentCommitter, inference.ContentOutcome) {
			return func(_ context.Context, _ string, data []byte) inference.ContentOutcome {
				mu.Lock()
				committed = append(committed, bytes.Clone(data))
				mu.Unlock()
				return inference.ContentResolved
			}, inference.ContentResolved
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = mediator.Close() })
	caller := inference.Subject{Account: "caller", Program: `C:\Apps\comfyui.exe`}
	execution := inference.NewJobExecution(mediator, func(_ context.Context, scope, _ string) (inference.Subject, inference.ContentOutcome) {
		if scope != "caller-scope" {
			return inference.Subject{}, inference.ContentForbidden
		}
		return caller, inference.ContentResolved
	}, nil)
	jobs, err := acceptanceprovider.OpenWithExecutor(filepath.Join(root, "jobs"), "owner", execution)
	if err != nil {
		t.Fatal(err)
	}
	jobCtx, stopJobs := context.WithCancel(context.Background())
	jobsDone := make(chan struct{})
	go func() { _ = jobs.Execute(jobCtx); close(jobsDone) }()
	t.Cleanup(func() { stopJobs(); <-jobsDone })

	history, err := jobs.Bind("caller-scope").GetHistoryWindow()
	if err != nil {
		t.Fatal(err)
	}
	image := jobwire.ImageRequest{Mode: jobwire.ModeGenerate, Extensions: map[string]string{}, Model: model, Prompt: "a lighthouse at dusk", Size: "auto", Count: 1,
		Guarantees: []jobwire.RequestGuarantee{jobwire.RequestGuaranteeLocalOnly}}
	document := &jobwire.Document{Request: &jobwire.Request{Profile: jobwire.ProfileVideo, DurationMs: 1000, Image: image}}
	submission := acceptance.Submission{Identity: acceptance.RequestIdentity{Key: "video-1", HistoryEpoch: history.HistoryEpoch},
		Kind: inference.InferenceJobKind, Spec: jobwire.Encode(document), RequiredGuarantees: []string{inference.RecoverableUpstreamGuarantee}}
	if _, err := jobs.Bind("caller-scope").Submit(submission); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(90 * time.Second)
	var snapshot *acceptance.OperationSnapshot
	for time.Now().Before(deadline) {
		observed, err := jobs.BindOperations("caller-scope").ObserveWork(submission.Identity)
		if err == nil && observed.Snapshot != nil && (observed.Snapshot.State == acceptance.WorkStateComplete ||
			observed.Snapshot.State == acceptance.WorkStateFailed || observed.Snapshot.State == acceptance.WorkStateCancelled) {
			snapshot = observed.Snapshot
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if snapshot == nil || snapshot.State != acceptance.WorkStateComplete {
		t.Fatalf("the video job did not complete: %+v", snapshot)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(committed) != 1 || http.DetectContentType(committed[0]) != "video/webm" {
		t.Fatalf("committed %d results", len(committed))
	}
}
