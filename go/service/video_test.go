package service

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/mediationcontent"
	"github.com/openabstractions/abstraction-inference/go/providerhost"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// requireSocketOwnerTables skips a test that runs the stand-in sd-server
// where its loopback listener cannot be proved. The provider reads the
// engine's sockets through abstraction-identity's socket-owner tables, which
// exist on Windows alone, and refuses every engine elsewhere with
// engine:listener-not-loopback (CONTRACT.md MODELHOST-V5, MODELHOST-L7). The
// host package's tests replace that check off Windows; this package runs the
// provider as it ships.
func requireSocketOwnerTables(t *testing.T) {
	t.Helper()
	if _, err := identity.Sockets(); err != nil {
		t.Skipf("the provider refuses every engine here: %v", err)
	}
}

// stagedVideoEngine builds the stand-in sd-server under a pin of its own
// folder and places a three-file model set where it was downloaded.
func stagedVideoEngine(t *testing.T) host.VideoEngine {
	t.Helper()
	// Placed model paths follow no symbolic link (INF-F7); macOS's
	// temporary folder is under one.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "engine")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	program := filepath.Join(dir, "sd-server")
	if runtime.GOOS == "windows" {
		program += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", program, "../host/testdata/fakesdserver").CombinedOutput(); err != nil {
		t.Fatalf("build fakesdserver: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(program)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	pin := host.EnginePin{Release: "test", Program: filepath.Base(program),
		Files: []enginefiles.File{{Path: filepath.Base(program), Size: int64(len(raw)), SHA256: hex.EncodeToString(sum[:])}}}
	source := filepath.Join(root, "downloads")
	if err := os.MkdirAll(source, 0o700); err != nil {
		t.Fatal(err)
	}
	set := host.VideoModelSet{Name: "fastwan2.2-ti2v-5b", Preset: host.FastWanPreset, Sizes: host.FastWan22TI2V5B.Sizes,
		MaxFrames: 121, MaxFPS: 30, FirstFrame: host.FirstFrameOptional}
	for _, f := range []struct{ name, role string }{{"d.gguf", host.RoleDiffusionModel}, {"e.gguf", host.RoleT5XXL}, {"t.safetensors", host.RoleTAE}} {
		data := []byte("weights " + f.name)
		if err := os.WriteFile(filepath.Join(source, f.name), data, 0o600); err != nil {
			t.Fatal(err)
		}
		s := sha256.Sum256(data)
		file := host.VideoModelFile{Name: f.name, Role: f.role, Size: int64(len(data)), SHA256: hex.EncodeToString(s[:])}
		if f.role == host.RoleTAE {
			blob := sha1.New()
			fmt.Fprintf(blob, "blob %d\x00%s", len(data), data)
			file.GitBlobSHA1 = hex.EncodeToString(blob.Sum(nil))
		}
		set.Files = append(set.Files, file)
	}
	state := filepath.Join(root, "state")
	files := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	places := host.VideoModelPlaces(state, set)
	if _, err := host.PlaceVideoModels(context.Background(), set, []host.ModelSearch{host.FolderSearch(source)}, places); err != nil {
		t.Fatal(err)
	}
	return host.VideoEngine{Program: program, Pin: pin, Record: host.VideoEngineRecord(state, pin), Set: set,
		Places: places, ModelRecord: host.VideoModelRecord(state, set), Files: files,
		Ready: 30 * time.Second, Log: filepath.Join(state, "sd-server.log")}
}

// The provider endpoint serves video@1 and its content store to the
// designated runtime as the runtime's native video backend uses them: the
// first frame is staged through content-writer@1, Start and Observe run on
// video@1, and the result is read through content-reader@1 (INF-G3,
// INF-G5, MODELHOST-V9).
func TestVideoOnTheProviderEndpoint(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS == "darwin" {
		t.Skip("Program proof unavailable")
	}
	requireSocketOwnerTables(t)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	programs := []string{identity.CanonicalProgramPath(filepath.Clean(self))}
	root := t.TempDir()
	content, err := providerhost.NewDesignatedContent("modelhost-video", filepath.Join(root, "content"), filepath.Join(root, "uploads.json"), programs, 20<<20)
	if err != nil {
		t.Fatal(err)
	}
	model, err := host.New(host.Options{Leases: granting{released: make(chan string, 4)}, Source: oneObject{}, Idle: time.Hour,
		Video: &host.VideoOptions{Model: "fastwan2.2-ti2v-5b", Engine: stagedVideoEngine(t), Store: content.Store, Poll: 20 * time.Millisecond}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	provider, err := inference.New(ProviderConfig(model, router.New(), nil))
	if err != nil {
		t.Fatal(err)
	}
	endpoint := endpointFor(t)
	served, err := Listen(endpoint, provider, programs...)
	if err != nil {
		t.Fatal(err)
	}
	served.Content = content.Component
	served.Video = model.VideoServer()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- served.Serve(ctx) }()
	t.Cleanup(func() { cancel(); <-done; provider.Close() })
	transport := listen.FrameClient{Endpoint: endpoint}.WithDefaults(35*time.Second, MaxFrameBytes)

	var frame bytes.Buffer
	if err := png.Encode(&frame, image.NewNRGBA(image.Rect(0, 0, 64, 64))); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(frame.Bytes())
	digest := "sha256:" + hex.EncodeToString(sum[:])
	if err := mediationcontent.Upload(ctx, transport, "job-0001-first-frame", digest, "image/png", frame.Bytes(), 20<<20); err != nil {
		t.Fatal(err)
	}
	client := wire.NewVideoClient(transport)
	admission, err := client.Start(wire.VideoRequest{RequestID: "job-1", Model: "fastwan2.2-ti2v-5b", Prompt: "a lighthouse", Size: "auto",
		DurationMs: 1000, ImageDigest: digest, ImageMediaType: "image/png"})
	if err != nil || admission.Outcome != wire.StartOutcomeAccepted {
		t.Fatalf("start %+v %v", admission, err)
	}
	deadline := time.Now().Add(60 * time.Second)
	var observed wire.VideoObservation
	for {
		observed, err = client.Observe(admission.Operation)
		if err != nil {
			t.Fatal(err)
		}
		if observed.Status != wire.VideoStateQueued && observed.Status != wire.VideoStateGenerating || time.Now().After(deadline) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if observed.Status != wire.VideoStateCompleted || observed.Delivery == nil {
		t.Fatalf("observed %+v", observed)
	}
	d := observed.Delivery
	data, err := mediationcontent.Download(ctx, transport, d.Digest, mediationcontent.SniffedMedia(d.MediaType), d.Size, 64<<20)
	if err != nil || len(data) != int(d.Size) {
		t.Fatalf("download: %v", err)
	}
}
