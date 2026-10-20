package host

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	storage "github.com/openabstractions/abstraction-storage/go"
)

// fakeSDServer builds the stand-in sd-server once per test binary.
var fakeSDServer = sync.OnceValues(func() (string, error) {
	dir, err := os.MkdirTemp("", "modelhost-sdserver-")
	if err != nil {
		return "", err
	}
	target := filepath.Join(dir, "fakesdserver")
	if runtime.GOOS == "windows" {
		target += ".exe"
	}
	build := exec.Command("go", "build", "-o", target, "./testdata/fakesdserver")
	if out, err := build.CombinedOutput(); err != nil {
		return "", fmt.Errorf("build fakesdserver: %v\n%s", err, out)
	}
	return target, nil
})

const testVideoModel = "fastwan2.2-ti2v-5b"

var testCaller = inference.Subject{Account: "S-1-5-21-test", Program: `C:\OA\openabstractions.exe`}

// videoFixture is one staged sd-server folder, a model set whose files lie
// in two stores, where the fixture placed them, and the provider's content
// store.
type videoFixture struct {
	engine   VideoEngine
	store    *storage.ContentStore
	dir      string
	requests string
	files    *enginefiles.Verifier
	// placed is where each model file lies, by name.
	placed map[string]string
}

// otherAppLocator is where the stand-in other application's store keeps
// the decoder.
const otherAppLocator = "vendor/taew/decoder.safetensors"

// fileSource is an inventory source of one store, other-app, that binds
// whole files by locator, as bridge.LocalStores binds a stored file.
type fileSource map[string]string

func (s fileSource) Resolve(store, id string, accept bridge.Accepts) (Object, error) {
	p, ok := s[id]
	if store != "other-app" || !ok {
		return Object{}, fmt.Errorf("%w: %s", bridge.ErrUnknownObject, id)
	}
	return Object{Store: store, Locator: id, Path: p, Binding: bridge.ObjectBinding{Kind: bridge.BindFile, Path: p}}, nil
}
func (s fileSource) Stores() []string   { return []string{"other-app"} }
func (s fileSource) Root(string) string { return "" }

// testModelSet is a set of three small files pinned the way a real set is,
// each by SHA-256 and the decoder also by its Git blob SHA-1. The diffusion
// model and encoder lie in root/downloads, a folder a person downloaded
// them to; the decoder lies in another application's store, which an
// inventory source binds. searches finds each where it lies.
func testModelSet(t *testing.T, root string) (set VideoModelSet, searches []ModelSearch) {
	t.Helper()
	set = VideoModelSet{Name: testVideoModel, Review: videoReview, Preset: FastWanPreset,
		Sizes: FastWan22TI2V5B.Sizes, MaxFrames: 121, MaxFPS: 30, FirstFrame: FirstFrameOptional}
	downloads := filepath.Join(root, "downloads")
	other := fileSource{}
	for _, f := range []struct{ name, role string }{{"diffusion.gguf", RoleDiffusionModel}, {"encoder.gguf", RoleT5XXL}, {"decoder.safetensors", RoleTAE}} {
		data := []byte("weights of " + f.name)
		target := filepath.Join(downloads, f.name)
		if f.role == RoleTAE {
			target = filepath.Join(root, "other-app", filepath.FromSlash(otherAppLocator))
			other[otherAppLocator] = target
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		file := VideoModelFile{Name: f.name, Role: f.role, Size: int64(len(data)), SHA256: hex.EncodeToString(sum[:]), License: "test"}
		if f.role == RoleTAE {
			blob := sha1.New()
			fmt.Fprintf(blob, "blob %d\x00", len(data))
			blob.Write(data)
			file.GitBlobSHA1 = hex.EncodeToString(blob.Sum(nil))
			file.Locators = []string{otherAppLocator}
		}
		set.Files = append(set.Files, file)
	}
	return set, []ModelSearch{FolderSearch(downloads), StoreSearch{Files: StoreFiles(other)}}
}

func newVideoFixture(t *testing.T, fixture ...string) *videoFixture {
	t.Helper()
	built, err := fakeSDServer()
	if err != nil {
		t.Fatal(err)
	}
	// Placed model paths follow no symbolic link (INF-F7); macOS's
	// temporary folder is under one.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	engineDir := filepath.Join(root, "engine")
	if err := os.MkdirAll(engineDir, 0o700); err != nil {
		t.Fatal(err)
	}
	program := filepath.Join(engineDir, "sd-server")
	if runtime.GOOS == "windows" {
		program += ".exe"
	}
	raw, err := os.ReadFile(built)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(program, raw, 0o700); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	pin := EnginePin{Release: "test", Program: filepath.Base(program),
		Files: []enginefiles.File{{Path: filepath.Base(program), Size: int64(len(raw)), SHA256: hex.EncodeToString(sum[:])}}}
	set, searches := testModelSet(t, root)
	state := filepath.Join(root, "state")
	files := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	placed, err := PlaceVideoModels(context.Background(), set, searches, VideoModelPlaces(state, set))
	if err != nil {
		t.Fatal(err)
	}
	store, err := storage.NewContentStore("video", filepath.Join(root, "content"))
	if err != nil {
		t.Fatal(err)
	}
	requests := filepath.Join(root, "requests")
	if err := os.MkdirAll(requests, 0o700); err != nil {
		t.Fatal(err)
	}
	return &videoFixture{dir: root, store: store, requests: requests, files: files, placed: placed, engine: VideoEngine{
		Program: program, Pin: pin, Record: VideoEngineRecord(state, pin), Set: set,
		Places: VideoModelPlaces(state, set), ModelRecord: VideoModelRecord(state, set), Files: files,
		Ready: 30 * time.Second, Log: filepath.Join(state, "logs", "sd-server.log"),
		fixture: append([]string{"--fake-requests-dir=" + requests, "--fake-args-file=" + filepath.Join(root, "args.txt"),
			"--fake-env-file=" + filepath.Join(root, "env.txt"), "--fake-cancels-file=" + filepath.Join(root, "cancels.txt")}, fixture...),
	}}
}

func (f *videoFixture) host(t *testing.T, leases *fakeLeases, adjust func(*Options)) *Host {
	t.Helper()
	options := Options{Leases: leases, Source: fakeSource{}, Idle: time.Hour, Renew: 50 * time.Millisecond, Wait: time.Second,
		Video: &VideoOptions{Model: testVideoModel, Engine: f.engine, Store: f.store, Poll: 20 * time.Millisecond}}
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

func (f *videoFixture) bodies(t *testing.T) [][]byte {
	t.Helper()
	entries, err := os.ReadDir(f.requests)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]byte
	for _, e := range entries {
		raw, err := os.ReadFile(filepath.Join(f.requests, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, raw)
	}
	return out
}

func videoRequest(id string) wire.VideoRequest {
	return wire.VideoRequest{RequestID: id, Model: testVideoModel, Prompt: "a lighthouse at dusk", Size: "auto", DurationMs: 2000}
}

// awaitVideo observes op until its status is one of want.
func awaitVideo(t *testing.T, h *Host, op string, want ...wire.VideoState) wire.VideoObservation {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for {
		observed := h.VideoServer().Observe(context.Background(), testCaller, op)
		if slices.Contains(want, observed.Status) {
			return observed
		}
		if time.Now().After(deadline) {
			t.Fatalf("operation %s is %s, waiting for %v", op, observed.Status, want)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func startVideo(t *testing.T, h *Host, request wire.VideoRequest) string {
	t.Helper()
	admission := h.VideoServer().Start(context.Background(), testCaller, request)
	if admission.Outcome != wire.StartOutcomeAccepted || admission.Operation == "" {
		t.Fatalf("admission %+v", admission)
	}
	return admission.Operation
}

// Conditions 3, 6, 7 and 9: a video runs on the stand-in sd-server, which
// listens on loopback at the provider's port with the pinned paths and gets
// a body of allow-listed keys only; the result is one checked local delivery.
func TestVideoGeneratesThroughTheEngine(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, videoRequest("job-1"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted || observed.Delivery == nil {
		t.Fatalf("observed %+v", observed)
	}
	d := observed.Delivery
	ref, ok := f.store.Find(d.Digest)
	if !ok || d.Location != "local" || d.MediaType != "video/webm" || d.Size != ref.Size {
		t.Fatalf("delivery %+v", d)
	}
	data, err := os.ReadFile(f.store.Path(ref))
	if err != nil || http.DetectContentType(data) != "video/webm" {
		t.Fatalf("stored result: %v", err)
	}
	args, err := os.ReadFile(filepath.Join(f.dir, "args.txt"))
	if err != nil {
		t.Fatal(err)
	}
	words := strings.Split(string(args), "\n")
	h.video.mu.Lock()
	port := strings.TrimPrefix(h.video.base, "http://127.0.0.1:")
	h.video.mu.Unlock()
	for _, pair := range [][2]string{{"--listen-ip", "127.0.0.1"}, {"--listen-port", port},
		{"--diffusion-model", filepath.Join(f.dir, "downloads", "diffusion.gguf")},
		{"--t5xxl", filepath.Join(f.dir, "downloads", "encoder.gguf")},
		{"--tae", filepath.Join(f.dir, "other-app", filepath.FromSlash(otherAppLocator))}} {
		i := slices.Index(words, pair[0])
		if i < 0 || i+1 >= len(words) || words[i+1] != pair[1] {
			t.Fatalf("%s is not %s in %q", pair[0], pair[1], words)
		}
	}
	for _, word := range words {
		if !strings.HasPrefix(word, "--fake-") && sdcpp.CheckFlag(word) != nil || word == "-M" {
			t.Fatalf("forbidden word %q", word)
		}
	}
	bodies := f.bodies(t)
	if len(bodies) != 1 {
		t.Fatalf("%d engine requests", len(bodies))
	}
	if err := sdcpp.CheckVideoRequest(bodies[0]); err != nil {
		t.Fatal(err)
	}
	var body map[string]any
	if err := json.Unmarshal(bodies[0], &body); err != nil {
		t.Fatal(err)
	}
	// 2000 ms at the preset's 24 fps is 48 frames, aligned to 49; size auto
	// is the preset's 832×480.
	if body["video_frames"] != float64(49) || body["fps"] != float64(24) || body["width"] != float64(832) || body["height"] != float64(480) {
		t.Fatalf("body %s", bodies[0])
	}
	if _, present := body["init_image"]; present {
		t.Fatal("a text video carried an init image")
	}
	if acquires, _, _ := leases.counts(); acquires != 1 {
		t.Fatalf("%d leases", acquires)
	}
}

// INF-G5: a Start repeating a request_id the provider holds names the same
// operation and starts nothing; another caller's operation is unknown to it.
func TestVideoRepeatedRequestIDReturnsTheAdmittedOperation(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	h := f.host(t, newFakeLeases(), nil)
	first := startVideo(t, h, videoRequest("job-1"))
	repeat := videoRequest("job-1")
	repeat.ImageDigest, repeat.ImageMediaType = "sha256:"+strings.Repeat("0", 64), "image/png"
	if again := startVideo(t, h, repeat); again != first {
		t.Fatalf("repeat named %s, not %s", again, first)
	}
	awaitVideo(t, h, first, wire.VideoStateCompleted)
	if again := startVideo(t, h, videoRequest("job-1")); again != first {
		t.Fatalf("a repeat after the end named %s", again)
	}
	if n := len(f.bodies(t)); n != 1 {
		t.Fatalf("%d engine requests", n)
	}
	other := inference.Subject{Account: testCaller.Account, Program: `C:\Other\app.exe`}
	if observed := h.VideoServer().Observe(context.Background(), other, first); observed.Status != wire.VideoStateUnknown {
		t.Fatalf("another program observed %+v", observed)
	}
	if cancelled := h.VideoServer().Cancel(context.Background(), other, first); cancelled.Outcome != wire.CancelOutcomeUnknown {
		t.Fatalf("another program cancelled: %+v", cancelled)
	}
	if mine := startVideo(t, h, wire.VideoRequest{RequestID: "job-1", Model: testVideoModel, Prompt: "p", Size: "auto", DurationMs: 500}); mine != first {
		// The same request_id from the same caller is one operation.
		t.Fatal("request_id was not held")
	}
}

// INF-G5: request ids are held for at least 24 hours after their operation
// ends, then retired with their results.
func TestVideoRequestIDsAreHeldForTheRetention(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	var mu sync.Mutex
	offset := time.Duration(0)
	now := func() time.Time { mu.Lock(); defer mu.Unlock(); return time.Now().Add(offset) }
	h := f.host(t, newFakeLeases(), func(o *Options) { o.Now = now; o.Video.Retention = time.Hour })
	op := startVideo(t, h, videoRequest("job-1"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted)
	if retention := h.VideoServer().Start(context.Background(), testCaller, videoRequest("job-1")).RetentionMs; retention != (24 * time.Hour).Milliseconds() {
		t.Fatalf("retention %d ms; less than 24 hours is raised to it", retention)
	}
	mu.Lock()
	offset = 23 * time.Hour
	mu.Unlock()
	if again := startVideo(t, h, videoRequest("job-1")); again != op {
		t.Fatal("request_id dropped before 24 hours")
	}
	mu.Lock()
	offset = 25 * time.Hour
	mu.Unlock()
	if gone := h.VideoServer().Observe(context.Background(), testCaller, op); gone.Status != wire.VideoStateUnknown {
		t.Fatalf("retired operation %+v", gone)
	}
	if _, kept := f.store.Find(observed.Delivery.Digest); kept {
		t.Fatal("a retired result stayed in the store")
	}
	if fresh := startVideo(t, h, videoRequest("job-1")); fresh == op {
		t.Fatal("a retired request_id named the old operation")
	}
}

// INF-G5: a first frame the store does not hold is content:unknown, and no
// engine starts.
func TestVideoMissingFirstFrameIsContentUnknown(t *testing.T) {
	f := newVideoFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	request := videoRequest("job-1")
	request.ImageDigest, request.ImageMediaType = "sha256:"+strings.Repeat("ab", 32), "image/png"
	admission := h.VideoServer().Start(context.Background(), testCaller, request)
	if admission.Outcome != wire.StartOutcomeInvalid || admission.Reason != "content:unknown" || admission.Operation != "" {
		t.Fatalf("admission %+v", admission)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 || h.Loaded(testVideoModel) {
		t.Fatal("an engine started for a refused video")
	}
}

func (f *videoFixture) stage(t *testing.T, data []byte) string {
	t.Helper()
	sum := sha256.Sum256(data)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	ref, err := f.store.Place(digest, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f.store.Path(ref), data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := f.store.Commit(ref); err != nil {
		t.Fatal(err)
	}
	return digest
}

// Condition 8: the engine receives the provider's own PNG of the request's
// exact size; WebP and other frames are refused before the engine.
func TestVideoFirstFrameIsTheProvidersOwnPNG(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	h := f.host(t, newFakeLeases(), nil)
	src := image.NewRGBA(image.Rect(0, 0, 300, 300))
	for i := range src.Pix {
		src.Pix[i] = byte(i * 7)
	}
	src.Set(0, 0, color.RGBA{255, 0, 0, 255})
	var jpg bytes.Buffer
	if err := jpeg.Encode(&jpg, src, nil); err != nil {
		t.Fatal(err)
	}
	request := videoRequest("job-1")
	request.Size = "480x832"
	request.ImageDigest, request.ImageMediaType = f.stage(t, jpg.Bytes()), "image/jpeg"
	awaitVideo(t, h, startVideo(t, h, request), wire.VideoStateCompleted)
	var body struct {
		InitImage string `json:"init_image"`
	}
	if err := json.Unmarshal(f.bodies(t)[0], &body); err != nil {
		t.Fatal(err)
	}
	frame, err := base64.StdEncoding.DecodeString(body.InitImage)
	if err != nil || bytes.Equal(frame, jpg.Bytes()) {
		t.Fatalf("the caller's bytes reached the engine: %v", err)
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(frame))
	if err != nil || format != "png" || config.Width != 480 || config.Height != 832 {
		t.Fatalf("frame %s %dx%d %v", format, config.Width, config.Height, err)
	}
	webp := []byte("RIFF\x1a\x00\x00\x00WEBPVP8 \x0e\x00\x00\x00\x30\x01\x00\x9d\x01\x2a\x01\x00\x01\x00")
	request = videoRequest("job-2")
	request.ImageDigest, request.ImageMediaType = f.stage(t, webp), "image/webp"
	if admission := h.VideoServer().Start(context.Background(), testCaller, request); admission.Outcome != wire.StartOutcomeUnsupportedFeature || admission.Reason != sdcpp.ReasonFirstFrame {
		t.Fatalf("WebP frame: %+v", admission)
	}
}

// Condition 7 and INF-G8: values beyond the engine's reviewed bounds are
// refused before the engine; an optional extension naming a preset value is
// ignored and a required one refused.
func TestVideoRequestBoundsAndPreset(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	for _, tc := range []struct {
		change  func(*wire.VideoRequest)
		outcome wire.StartOutcome
		reason  string
	}{
		{func(r *wire.VideoRequest) { r.Size = "1024x1024" }, wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonImageSize},
		{func(r *wire.VideoRequest) { r.Size = "65536x21846" }, wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonImageSize},
		{func(r *wire.VideoRequest) { r.Prompt = strings.Repeat("a", sdcpp.MaxPromptBytes+1) }, wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonVideoBound},
		{func(r *wire.VideoRequest) { r.DurationMs = 10000 }, wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonVideoBound},
		{func(r *wire.VideoRequest) { r.Fps = 60 }, wire.StartOutcomeUnsupportedFeature, sdcpp.ReasonVideoBound},
		{func(r *wire.VideoRequest) {
			r.Extensions = map[string]string{"vendor/steps": "50"}
			r.RequiredExtensions = []string{"vendor/steps"}
		}, wire.StartOutcomeUnsupportedFeature, "extension:vendor/steps"},
		{func(r *wire.VideoRequest) { r.Prompt = "" }, wire.StartOutcomeInvalid, "prompt"},
		{func(r *wire.VideoRequest) { r.DurationMs = 0 }, wire.StartOutcomeInvalid, "duration_ms"},
		{func(r *wire.VideoRequest) { r.Model = "other" }, wire.StartOutcomeNoHost, "model:unknown"},
		{func(r *wire.VideoRequest) { r.RequestID = "" }, wire.StartOutcomeInvalid, "request_id"},
	} {
		request := videoRequest("bound")
		tc.change(&request)
		admission := h.VideoServer().Start(context.Background(), testCaller, request)
		if admission.Outcome != tc.outcome || admission.Reason != tc.reason {
			t.Fatalf("%+v: %+v", request, admission)
		}
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatal("a refused request reached the engine")
	}
	request := videoRequest("preset")
	request.Size, request.DurationMs, request.Fps = "1280x704", 5000, 0
	request.Extensions = map[string]string{"sample_steps": "50", "lora": "x", inference.ClaimExtension: "app"}
	request.RequiredExtensions = []string{inference.ClaimExtension}
	awaitVideo(t, h, startVideo(t, h, request), wire.VideoStateCompleted)
	var body struct {
		Frames int `json:"video_frames"`
		Sample struct {
			Steps int `json:"sample_steps"`
		} `json:"sample_params"`
	}
	raw := f.bodies(t)[0]
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body.Frames != 121 || body.Sample.Steps != FastWanPreset.Steps || bytes.Contains(raw, []byte("lora")) {
		t.Fatalf("body %s", raw)
	}
}

// Condition 9: a result whose frames, rate, MIME type, magic bytes or size
// differ from what was asked fails without a delivery.
func TestVideoResultIsCheckedAgainstTheRequest(t *testing.T) {
	requireEnginePeer(t)
	for _, mode := range []string{"frames", "fps", "mime", "magic", "huge", "fail"} {
		t.Run(mode, func(t *testing.T) {
			f := newVideoFixture(t, "--fake-result="+mode)
			h := f.host(t, newFakeLeases(), nil)
			observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateCompleted, wire.VideoStateFailed)
			want := videoEngineResult
			if mode == "fail" {
				want = videoEngineFailed
			}
			if observed.Status != wire.VideoStateFailed || observed.Reason != want || observed.Delivery != nil {
				t.Fatalf("observed %+v", observed)
			}
			if strings.Contains(observed.Reason, "prompt") {
				t.Fatal("engine text reached the caller")
			}
		})
	}
}

// Condition 10: a job the engine still holds queued is cancelled through
// its route, and the engine keeps running.
func TestVideoCancelQueuedUsesTheRoute(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-queue=30s")
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateQueued)
	deadline := time.Now().Add(30 * time.Second)
	for len(f.bodies(t)) == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	time.Sleep(100 * time.Millisecond)
	pid := h.resources.PID(testVideoModel)
	if cancelled := h.VideoServer().Cancel(context.Background(), testCaller, op); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", cancelled)
	}
	for !strings.HasPrefix(readFile(f.dir, "cancels.txt"), "queued ") && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !strings.HasPrefix(readFile(f.dir, "cancels.txt"), "queued ") {
		t.Fatal("the engine's cancel route was not called")
	}
	time.Sleep(200 * time.Millisecond)
	if !h.Loaded(testVideoModel) || h.resources.PID(testVideoModel) != pid {
		t.Fatal("cancelling a queued job stopped the engine")
	}
	if observed := h.VideoServer().Observe(context.Background(), testCaller, op); observed.Status != wire.VideoStateCancelled {
		t.Fatalf("observed %+v", observed)
	}
	if again := h.VideoServer().Cancel(context.Background(), testCaller, op); again.Outcome != wire.CancelOutcomeEnded {
		t.Fatalf("second cancel %+v", again)
	}
}

func readFile(dir, name string) string {
	raw, _ := os.ReadFile(filepath.Join(dir, name))
	return string(raw)
}

// Condition 10: a generating job is cancelled by stopping the engine's
// process tree, and the lease is released; the next job relaunches it.
func TestVideoCancelGeneratingStopsTheEngine(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	if cancelled := h.VideoServer().Cancel(context.Background(), testCaller, op); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", cancelled)
	}
	deadline := time.Now().Add(30 * time.Second)
	for h.Loaded(testVideoModel) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if h.Loaded(testVideoModel) {
		t.Fatal("the generating engine kept running")
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v", released)
	}
	if readFile(f.dir, "cancels.txt") != "" {
		t.Fatal("a generating job was sent to the cancel route")
	}
	startVideo(t, h, videoRequest("job-2"))
	for !h.Loaded(testVideoModel) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if acquires, _, _ := leases.counts(); acquires != 2 || !h.Loaded(testVideoModel) {
		t.Fatalf("the next job did not relaunch the engine: %d leases", acquires)
	}
}

// awaitUnloaded waits for the video engine to stop and returns whether it
// did within 30 seconds.
func awaitUnloaded(h *Host) bool { return awaitUnloadedWithin(h, 30*time.Second) }

// awaitUnloadedWithin waits up to within for the video engine to stop.
func awaitUnloadedWithin(h *Host, within time.Duration) bool {
	deadline := time.Now().Add(within)
	for h.Loaded(testVideoModel) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	return !h.Loaded(testVideoModel)
}

// Condition 10: a cancel that cuts the job submission short stops the
// engine, which took the job before it answered; the lease is released.
func TestVideoCancelDuringSubmitStopsTheEngine(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-stall-submit=60s")
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, videoRequest("job-1"))
	deadline := time.Now().Add(30 * time.Second)
	for len(f.bodies(t)) == 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if len(f.bodies(t)) != 1 {
		t.Fatal("the job never reached the engine")
	}
	if cancelled := h.VideoServer().Cancel(context.Background(), testCaller, op); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", cancelled)
	}
	if !awaitUnloaded(h) {
		t.Fatal("the engine holding the unanswered job kept running")
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v", released)
	}
}

// Condition 10: a cancel while the engine loads stops the load and gives the
// lease back; no request reaches the engine.
func TestVideoCancelDuringLoadStopsTheEngine(t *testing.T) {
	f := newVideoFixture(t, "--fake-load=60s")
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	op := startVideo(t, h, videoRequest("job-1"))
	deadline := time.Now().Add(30 * time.Second)
	for readFile(f.dir, "args.txt") == "" && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if readFile(f.dir, "args.txt") == "" {
		t.Fatal("the engine never started")
	}
	if cancelled := h.VideoServer().Cancel(context.Background(), testCaller, op); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", cancelled)
	}
	// The load's own budget is 30 seconds; the cancel ends it well before.
	// Loaded waits while the load holds the component, so the elapsed time
	// is what shows the cancel cut the load short.
	cancelledAt := time.Now()
	if !awaitUnloaded(h) || time.Since(cancelledAt) > 10*time.Second {
		t.Fatalf("the loading engine ran %s after the cancel", time.Since(cancelledAt))
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v", released)
	}
	if len(f.bodies(t)) != 0 {
		t.Fatal("a cancelled video reached the engine")
	}
}

// An engine that ends while it generates fails the video as engine:lost and
// is unloaded.
func TestVideoEngineCrashMidGenerationIsLost(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s", "--fake-crash-generating=1")
	leases := newFakeLeases()
	exits := &engineExits{}
	h := f.host(t, leases, func(o *Options) { o.OnEngineExit = exits.hook })
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != videoEngineLost {
		t.Fatalf("observed %+v", observed)
	}
	if !awaitUnloaded(h) {
		t.Fatal("the crashed engine stayed resident")
	}
	// MODELHOST-E1: the engine's own exit, with its code, is recorded.
	if x := exits.await(t); x.Engine != "sd-server" || x.Model == "" || x.Code != 3 || x.Stopped || !x.Notable() {
		t.Fatalf("exit record %+v", x)
	}
}

// Closing the provider while a video generates fails it as engine:lost and
// stops the engine.
func TestVideoHostCloseDuringPollingEndsTheVideo(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	h.Close()
	observed := h.VideoServer().Observe(context.Background(), testCaller, op)
	if observed.Status != wire.VideoStateFailed || observed.Reason != videoEngineLost {
		t.Fatalf("observed %+v", observed)
	}
	if h.Loaded(testVideoModel) {
		t.Fatal("the engine outlived the provider")
	}
}

// A Start beyond the waiting queue's bound is exhausted.
func TestVideoQueueIsBounded(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	h := f.host(t, newFakeLeases(), nil)
	// The queue holds MaxVideoQueue behind the running video. The first job
	// leaves the queue only when the worker takes it, so the test waits for it
	// to generate before filling the queue; filling at once raced the worker
	// and refused the last job queue:full on a loaded machine (2026-10-07).
	running := startVideo(t, h, videoRequest("job-0"))
	if observed := awaitVideo(t, h, running, wire.VideoStateGenerating, wire.VideoStateFailed, wire.VideoStateCompleted); observed.Status != wire.VideoStateGenerating {
		t.Fatalf("the first video %+v", observed)
	}
	for i := 1; i <= MaxVideoQueue; i++ {
		startVideo(t, h, videoRequest(fmt.Sprintf("job-%d", i)))
	}
	admission := h.VideoServer().Start(context.Background(), testCaller, videoRequest("one-too-many"))
	if admission.Outcome != wire.StartOutcomeExhausted || admission.Reason != videoQueueFull {
		t.Fatalf("admission beyond the queue %+v", admission)
	}
}

// Condition 10: a job past the preset's time limit fails and its engine is
// stopped.
func TestVideoTimeLimitStopsTheEngine(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	leases := newFakeLeases()
	h := f.host(t, leases, func(o *Options) { o.Video.TimeLimit = 300 * time.Millisecond })
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != videoEngineTimeout {
		t.Fatalf("observed %+v", observed)
	}
	if h.Loaded(testVideoModel) {
		t.Fatal("the engine outlived its time limit")
	}
}

// Condition 6: the engine stays while video work is queued and stops an
// idle window after the last job.
func TestVideoEngineStopsAfterTheIdleWindow(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=400ms")
	leases := newFakeLeases()
	h := f.host(t, leases, func(o *Options) { o.Idle = 150 * time.Millisecond })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	first := startVideo(t, h, videoRequest("job-1"))
	second := startVideo(t, h, videoRequest("job-2"))
	awaitVideo(t, h, first, wire.VideoStateCompleted)
	awaitVideo(t, h, second, wire.VideoStateCompleted)
	if acquires, _, _ := leases.counts(); acquires != 1 {
		t.Fatalf("the engine was reloaded between queued jobs: %d leases", acquires)
	}
	deadline := time.Now().Add(10 * time.Second)
	for h.Loaded(testVideoModel) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if h.Loaded(testVideoModel) {
		t.Fatal("the idle engine kept running")
	}
}

// Condition 6: an engine that does not report vid_gen is never ready, and
// the load is refused with its lease released.
func TestVideoEngineWithoutVideoModeIsNotReady(t *testing.T) {
	f := newVideoFixture(t, "--fake-mode=img_gen")
	f.engine.Ready = 500 * time.Millisecond
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
	if observed.Status != wire.VideoStateFailed || observed.Reason != "engine:unavailable" {
		t.Fatalf("observed %+v", observed)
	}
	if _, released, _ := leases.counts(); len(released) != 1 || h.Loaded(testVideoModel) {
		t.Fatal("the unready engine kept its lease")
	}
}

// Condition 5: the engine starts from the clean environment; GGML_, SD_,
// VK_, HF_ and the proxies never reach it.
func TestVideoEngineEnvironmentIsClean(t *testing.T) {
	requireEnginePeer(t)
	for _, name := range []string{"GGML_BACKEND_PATH", "SD_VK_DEVICE", "VK_ICD_FILENAMES", "HF_TOKEN", "HTTP_PROXY", "OA_VIDEO_SECRET"} {
		t.Setenv(name, "planted")
	}
	f := newVideoFixture(t)
	h := f.host(t, newFakeLeases(), nil)
	awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateCompleted)
	for _, name := range strings.Split(readFile(f.dir, "env.txt"), "\n") {
		upper := strings.ToUpper(name)
		if sdcpp.ForbiddenEnvironment(upper) || upper == "OA_VIDEO_SECRET" || upper == "LLAMA_API_KEY" || !engineEnvironmentNames[upper] {
			t.Fatalf("%s reached the engine", name)
		}
	}
	// The forbidden prefixes hold even for a name added to the allowlist.
	for _, name := range []string{"GGML_BACKEND_PATH", "VK_LAYER_PATH", "SD_MMAP_FLAGS", "HF_ENDPOINT", "HTTPS_PROXY"} {
		engineEnvironmentNames[name] = true
	}
	t.Cleanup(func() {
		for _, name := range []string{"GGML_BACKEND_PATH", "VK_LAYER_PATH", "SD_MMAP_FLAGS", "HF_ENDPOINT", "HTTPS_PROXY"} {
			delete(engineEnvironmentNames, name)
		}
	})
	for _, env := range [][]string{{"PATH=x", "GGML_BACKEND_PATH=y"}, {"PATH=x", "VK_LAYER_PATH=y"}, {"PATH=x", "sd_mmap_flags=1"},
		{"PATH=x", "HF_ENDPOINT=h"}, {"PATH=x", "https_proxy=p"}, {"PATH=x", "LLAMA_API_KEY=k"}} {
		if err := checkVideoEnvironment(env); !errors.Is(err, ErrEngineEnvironment) {
			t.Fatalf("%v admitted", env)
		}
	}
}

// Conditions 3 and 4: a configured option outside the reviewed list is
// refused by name before anything starts.
func TestVideoEngineArgumentsAreTheReviewedList(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t)
	for _, args := range [][]string{{"--lora-model-dir", "x"}, {"-M", "vid_gen"}, {"--listen-ip", "0.0.0.0"}, {"--rpc-servers=h:1"}, {"--steps", "50"}} {
		_, err := New(Options{Leases: newFakeLeases(), Source: fakeSource{},
			Video: &VideoOptions{Model: testVideoModel, Engine: VideoEngine{Arguments: args}, Store: f.store}})
		if !errors.Is(err, sdcpp.ErrForbiddenFlag) && !errors.Is(err, sdcpp.ErrArgument) {
			t.Fatalf("%q: %v", args, err)
		}
	}
	f.engine.Arguments = []string{"--diffusion-fa", "--offload-to-cpu"}
	h := f.host(t, newFakeLeases(), nil)
	awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateCompleted)
	args := readFile(f.dir, "args.txt")
	if !strings.Contains(args, "--diffusion-fa\n--offload-to-cpu") {
		t.Fatalf("args %q", args)
	}
}

// Conditions 1 and 2: an engine folder that differs from its pin in any
// file, an extra file included, and a placed model file that changed or
// went away where it lies, are refused before a lease.
func TestVideoFoldersAreProvedBeforeEveryStart(t *testing.T) {
	for _, tamper := range []func(f *videoFixture) error{
		func(f *videoFixture) error {
			return os.WriteFile(f.placed["decoder.safetensors"], []byte("weights of decoder.safetensorZ"), 0o600)
		},
		func(f *videoFixture) error {
			return os.Remove(f.placed["encoder.gguf"])
		},
		func(f *videoFixture) error {
			return os.WriteFile(filepath.Join(filepath.Dir(f.engine.Program), "ggml-extra.dll"), []byte("x"), 0o600)
		},
	} {
		f := newVideoFixture(t)
		if err := tamper(f); err != nil {
			t.Fatal(err)
		}
		leases := newFakeLeases()
		h := f.host(t, leases, nil)
		observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed, wire.VideoStateCompleted)
		if observed.Status != wire.VideoStateFailed || observed.Reason != NoEngine {
			t.Fatalf("observed %+v", observed)
		}
		if acquires, _, _ := leases.counts(); acquires != 0 {
			t.Fatal("a lease was asked for an unproven folder")
		}
	}
}

// Condition 2 and VISION 2026-10-01: placing a set copies nothing. Each
// pinned file is bound where it lies, from a folder and from another
// application's store; a candidate whose digest differs is refused and the
// next is proved; a file no search proves leaves no record.
func TestPlaceVideoModelsBindsInPlace(t *testing.T) {
	// Placed model paths follow no symbolic link (INF-F7); macOS's
	// temporary folder is under one.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	set, searches := testModelSet(t, root)
	// A decoy of the pinned name and size, and other bytes, lies first in
	// the order the searches are tried.
	decoy := filepath.Join(root, "downloads", "decoder.safetensors")
	if err := os.WriteFile(decoy, []byte("weights of decoder.safetensorZ"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "downloads", "unrelated.bin"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	before := treeFiles(t, root)
	places := filepath.Join(root, "state", "models.places")
	found, err := PlaceVideoModels(context.Background(), set, searches, places)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"diffusion.gguf": filepath.Join(root, "downloads", "diffusion.gguf"),
		"encoder.gguf": filepath.Join(root, "downloads", "encoder.gguf"), "decoder.safetensors": filepath.Join(root, "other-app", filepath.FromSlash(otherAppLocator))}
	for name, path := range want {
		if found[name] != path {
			t.Fatalf("%s found at %q, want %q", name, found[name], path)
		}
	}
	recorded, err := readPlaces(places, set)
	if err != nil || len(recorded) != 3 || recorded["decoder.safetensors"] != want["decoder.safetensors"] {
		t.Fatalf("recorded %v %v", recorded, err)
	}
	// No copy: the only new file is the places record, and every file that
	// was there is unchanged.
	after := treeFiles(t, root)
	delete(after, places)
	delete(after, places+".lock") // the record's own write lock
	if len(after) != len(before) {
		t.Fatalf("placing wrote files: before %v after %v", before, after)
	}
	for path, content := range before {
		if after[path] != content {
			t.Fatalf("%s changed", path)
		}
	}
	held := VideoEngine{Set: set, Places: places, ModelRecord: filepath.Join(root, "state", "models.files")}
	if err := CheckVideoModels(context.Background(), enginefiles.NewVerifier(enginefiles.Options{}), set, places, held.ModelRecord); err != nil {
		t.Fatal(err)
	}
	args, err := held.arguments(41000, recorded)
	if err != nil || !slices.Contains(args, want["decoder.safetensors"]) || !slices.Contains(args, want["diffusion.gguf"]) {
		t.Fatalf("arguments %q, %v", args, err)
	}

	for i, bad := range []func(*VideoModelSet){
		func(s *VideoModelSet) { s.Files[0].SHA256 = strings.Repeat("0", 64) },
		func(s *VideoModelSet) { s.Files[1].Size++ },
		func(s *VideoModelSet) { s.Files[2].GitBlobSHA1 = strings.Repeat("0", 40) },
		func(s *VideoModelSet) { s.Files[0].Name = "missing.gguf" },
		func(s *VideoModelSet) { s.Files[2].Locators = []string{"../escape.safetensors"} },
		func(s *VideoModelSet) { s.Files[2].SHA256 = "" },
		func(s *VideoModelSet) { s.UsePolicy = UsePolicy{Name: "a policy"} },
	} {
		changed := set
		changed.Files = slices.Clone(set.Files)
		bad(&changed)
		refused := filepath.Join(root, "state", fmt.Sprintf("refused-%d.places", i))
		if _, err := PlaceVideoModels(context.Background(), changed, searches, refused); err == nil {
			t.Fatalf("case %d was placed", i)
		}
		if _, err := os.Stat(refused); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("case %d left a record", i)
		}
	}
	// With the decoy as the only candidate, its digest is the refusal.
	only := []ModelSearch{FolderSearch(filepath.Join(root, "downloads"))}
	if _, err := PlaceVideoModels(context.Background(), set, only, filepath.Join(root, "state", "decoy.places")); err == nil || !strings.Contains(err.Error(), "decoder.safetensors") {
		t.Fatalf("the decoy: %v", err)
	}
	if err := FastWan22TI2V5B.valid(); err != nil {
		t.Fatal(err)
	}
	for _, s := range VideoModelSets {
		if err := s.valid(); err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
		if named, ok := VideoModelSetNamed(s.Name); !ok || named.Name != s.Name {
			t.Fatalf("%s is not found by name", s.Name)
		}
	}
	if err := SdCppMaster843WindowsVulkan.valid(); err != nil || len(SdCppMaster843WindowsVulkan.Files) != 21 {
		t.Fatalf("the reviewed pin: %v", err)
	}
}

// treeFiles is every regular file under root with its content.
func treeFiles(t *testing.T, root string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		data, err := os.ReadFile(path)
		out[path] = string(data)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// INF-F2 and INF-F7: while the engine runs, each model file is held where
// it lies; on Windows it cannot be opened for writing, renamed or deleted,
// and it is writable again once the engine is gone.
func TestVideoModelFilesAreHeldWhileTheEngineRuns(t *testing.T) {
	requireEnginePeer(t)
	f := newVideoFixture(t, "--fake-generate=60s")
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, videoRequest("job-1"))
	awaitVideo(t, h, op, wire.VideoStateGenerating)
	for name, path := range f.placed {
		if runtime.GOOS == "windows" {
			if w, err := os.OpenFile(path, os.O_WRONLY, 0); err == nil {
				w.Close()
				t.Fatalf("%s opened for writing while the engine ran", name)
			}
			if err := os.Rename(path, path+".moved"); err == nil {
				t.Fatalf("%s was renamed while the engine ran", name)
			}
		}
	}
	if cancelled := h.VideoServer().Cancel(context.Background(), testCaller, op); cancelled.Outcome != wire.CancelOutcomeCancelled {
		t.Fatalf("cancel %+v", cancelled)
	}
	if !awaitUnloaded(h) {
		t.Fatal("the engine kept running")
	}
	// The hold ends when the engine's exit is confirmed, which follows the
	// unload by the time the process wait returns.
	deadline := time.Now().Add(10 * time.Second)
	for name, path := range f.placed {
		for {
			w, err := os.OpenFile(path, os.O_WRONLY, 0)
			if err == nil {
				w.Close()
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("%s is still held after the engine stopped: %v", name, err)
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
}

// Condition 12: the engine's output is kept in the provider's own log and
// never reaches a caller.
func TestVideoEngineOutputStaysInTheProvidersLog(t *testing.T) {
	f := newVideoFixture(t, "--fake-result=fail")
	h := f.host(t, newFakeLeases(), nil)
	observed := awaitVideo(t, h, startVideo(t, h, videoRequest("job-1")), wire.VideoStateFailed)
	if strings.Contains(observed.Reason, "engine text") {
		t.Fatal("engine text reached the caller")
	}
	deadline := time.Now().Add(10 * time.Second)
	for !strings.Contains(readFile(filepath.Dir(f.engine.Log), filepath.Base(f.engine.Log)), "prompt text") && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if !strings.Contains(readFile(filepath.Dir(f.engine.Log), filepath.Base(f.engine.Log)), "prompt text") {
		t.Fatal("the engine's output was not kept")
	}
}

func TestEngineLogRotatesAtItsBound(t *testing.T) {
	path := filepath.Join(t.TempDir(), "engine.log")
	l, err := openEngineLog(path, 64)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if _, err := l.Write([]byte(strings.Repeat("x", 20))); err != nil {
			t.Fatal(err)
		}
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{path, path + ".1"} {
		info, err := os.Stat(name)
		if err != nil || info.Size() > 64 {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

// Video addendum condition 3 and the live run of 2026-10-01: the FastWan
// set always runs with flash attention in the diffusion model and a tiled
// decode, each passed once beside a person's own options.
func TestFastWanLaunchesWithFlashAttentionAndTiledDecode(t *testing.T) {
	models := t.TempDir()
	where := map[string]string{}
	for _, file := range FastWan22TI2V5B.Files {
		where[file.Name] = filepath.Join(models, file.Name)
	}
	e := VideoEngine{Set: FastWan22TI2V5B, Arguments: []string{"--diffusion-fa", "-v"}}
	args, err := e.arguments(41000, where)
	if err != nil {
		t.Fatal(err)
	}
	count := func(word string) int {
		n := 0
		for _, a := range args {
			if a == word {
				n++
			}
		}
		return n
	}
	if count("--diffusion-fa") != 1 || count("--vae-tiling") != 1 || count("-v") != 1 {
		t.Fatalf("args %q", args)
	}
	e.Arguments = nil
	if args, err = e.arguments(41000, where); err != nil || count("--diffusion-fa") != 1 || count("--vae-tiling") != 1 {
		t.Fatalf("args %q, %v", args, err)
	}
	// A set passes switches only; a valued option is refused, never dropped.
	for _, options := range [][]string{{"--max-vram", "12"}, {"--backend", "vulkan0"}, {"-t", "4"}, {"--vae-tile-size", "64x64"}} {
		valued := e
		valued.Set.Options = options
		if _, err := valued.arguments(41000, where); err == nil {
			t.Fatalf("a set with %q was accepted", options)
		}
	}
	if DefaultVideoMemoryLimit != 40<<30 {
		t.Fatalf("memory limit %d", DefaultVideoMemoryLimit)
	}
}

// MODELHOST-V6: a request is held to its model set's sizes, frames, rate
// and first-frame use, which are data in the set's row; a video model name
// other than the set's is refused at construction.
func TestVideoRequestsAreHeldToTheModelSet(t *testing.T) {
	f := newVideoFixture(t)
	set := f.engine.Set
	set.Sizes = [][2]int{{832, 480}}
	set.MaxFrames = 49
	set.MaxFPS = 24
	set.FirstFrame = FirstFrameRequired
	f.engine.Set = set
	leases := newFakeLeases()
	h := f.host(t, leases, nil)
	for _, c := range []struct {
		change func(*wire.VideoRequest)
		reason string
	}{
		{func(r *wire.VideoRequest) {}, videoFrameRequired},
		{func(r *wire.VideoRequest) { r.Size = "1280x704" }, videoFrameRequired},
	} {
		r := videoRequest("bounds")
		c.change(&r)
		if a := h.VideoServer().Start(context.Background(), testCaller, r); a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != c.reason {
			t.Fatalf("admission %+v, want %s", a, c.reason)
		}
	}
	set.FirstFrame = FirstFrameOptional
	f.engine.Set = set
	h = f.host(t, leases, nil)
	for _, c := range []struct {
		change func(*wire.VideoRequest)
		reason string
	}{
		{func(r *wire.VideoRequest) { r.Size = "1280x704" }, sdcpp.ReasonImageSize},
		{func(r *wire.VideoRequest) { r.DurationMs = 5000 }, sdcpp.ReasonVideoBound},
		{func(r *wire.VideoRequest) { r.Fps = 30 }, sdcpp.ReasonVideoBound},
	} {
		r := videoRequest("bounds")
		c.change(&r)
		if a := h.VideoServer().Start(context.Background(), testCaller, r); a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != c.reason {
			t.Fatalf("admission %+v, want %s", a, c.reason)
		}
	}
	set.FirstFrame = FirstFrameNone
	f.engine.Set = set
	h = f.host(t, leases, nil)
	r := videoRequest("frame")
	r.ImageDigest, r.ImageMediaType = "sha256:"+strings.Repeat("0", 64), "image/png"
	if a := h.VideoServer().Start(context.Background(), testCaller, r); a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != sdcpp.ReasonFirstFrame {
		t.Fatalf("a first frame to a text-only set: %+v", a)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatal("a refused request reached the engine")
	}
	if _, err := New(Options{Leases: newFakeLeases(), Source: fakeSource{},
		Video: &VideoOptions{Model: "another-model", Engine: f.engine, Store: f.store}}); err == nil {
		t.Fatal("a video model other than its set's was accepted")
	}
}

// Video addendum amendment 1 and the Hummingbird-XT file review: the
// hummingbird-xt-i2v row takes a first frame only, at 512x512, 81 frames
// and 16 fps, runs AMD's three lcm steps on AMD's sigma table with flash
// attention, binds the i2v file by its digest where ASUS's store keeps it,
// and names AMD's use policy. Stand-in files carry the row's settings.
func TestHummingbirdRowRunsAsData(t *testing.T) {
	requireEnginePeer(t)
	row := HummingbirdXTI2V
	if err := row.valid(); err != nil {
		t.Fatal(err)
	}
	transformer, _ := row.File(RoleDiffusionModel)
	if transformer.SHA256 != "6937e10a78a130d5f3a4d44705db476d98af5932b3cad1e011180294ca2b8291" || transformer.Size != 9999659672 ||
		!slices.Contains(transformer.Locators, "hummingbird/HM_2.2_5B/transformer_20251218/diffusion_pytorch_model_i2v.safetensors") {
		t.Fatalf("the i2v file %+v", transformer)
	}
	if row.UsePolicy.Name != "AMD Responsible AI Use Policy" || row.UsePolicy.License != "Open RAIL-MS" || !strings.Contains(row.UsePolicy.URL, "AMD-AGI/HummingbirdXT") {
		t.Fatalf("use policy %+v", row.UsePolicy)
	}
	f := newVideoFixture(t)
	set := f.engine.Set
	set.Options, set.Preset, set.Sizes, set.MaxFrames, set.MaxFPS, set.FirstFrame, set.UsePolicy =
		row.Options, row.Preset, row.Sizes, row.MaxFrames, row.MaxFPS, row.FirstFrame, row.UsePolicy
	f.engine.Set = set
	h := f.host(t, newFakeLeases(), nil)
	text := videoRequest("text-only")
	if a := h.VideoServer().Start(context.Background(), testCaller, text); a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != videoFrameRequired {
		t.Fatalf("a text-only request: %+v", a)
	}
	src := image.NewRGBA(image.Rect(0, 0, 300, 300))
	var frame bytes.Buffer
	if err := jpeg.Encode(&frame, src, nil); err != nil {
		t.Fatal(err)
	}
	wide := videoRequest("wide")
	wide.Size = "832x480"
	wide.ImageDigest, wide.ImageMediaType = f.stage(t, frame.Bytes()), "image/jpeg"
	if a := h.VideoServer().Start(context.Background(), testCaller, wide); a.Outcome != wire.StartOutcomeUnsupportedFeature || a.Reason != sdcpp.ReasonImageSize {
		t.Fatalf("832x480: %+v", a)
	}
	request := videoRequest("clip")
	request.DurationMs = 5000
	request.ImageDigest, request.ImageMediaType = wide.ImageDigest, "image/jpeg"
	observed := awaitVideo(t, h, startVideo(t, h, request), wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted {
		t.Fatalf("observed %+v", observed)
	}
	var body struct {
		Width, Height int
		Frames        int `json:"video_frames"`
		FPS           int `json:"fps"`
		Sample        struct {
			Steps  int       `json:"sample_steps"`
			Method string    `json:"sample_method"`
			Sigmas []float64 `json:"custom_sigmas"`
		} `json:"sample_params"`
	}
	raw := f.bodies(t)[0]
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatal(err)
	}
	if body.Width != 512 || body.Height != 512 || body.Frames != 81 || body.FPS != 16 || body.Sample.Steps != 3 ||
		body.Sample.Method != "lcm" || !slices.Equal(body.Sample.Sigmas, []float64{1, 0.967742, 0.909091, 0}) {
		t.Fatalf("body %s", raw)
	}
	args := readFile(f.dir, "args.txt")
	if !strings.Contains(args, "--diffusion-fa") || strings.Contains(args, "--vae-tiling") {
		t.Fatalf("args %q", args)
	}
}

// Review of da8e3b548 items 4 and 5 (MODELHOST-V2, INF-F7): placing records
// the owners each file may have to be trusted in place, the machine's and
// this account for a folder a person names and the machine's alone for
// another application's store, and every proof asks the trusted-in-place
// check with those owners before the metadata shortcut.
func TestPlacedModelOwnersReachTheTrustCheck(t *testing.T) {
	f := newVideoFixture(t)
	var mu sync.Mutex
	asked := map[string]trustedpath.Owners{}
	files := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}, TrustPlaced: func(p string, o trustedpath.Owners) error {
		mu.Lock()
		defer mu.Unlock()
		asked[p] = o
		return trustedpath.ErrUntrusted
	}})
	if err := CheckVideoModels(context.Background(), files, f.engine.Set, f.engine.Places, f.engine.ModelRecord); err != nil {
		t.Fatal(err)
	}
	machine := trustedpath.MachineOwners()
	for name, want := range map[string]bool{"diffusion.gguf": true, "encoder.gguf": true, "decoder.safetensors": false} {
		o, ok := asked[f.placed[name]]
		if !ok || o.Account != want || !slices.Equal(o.SIDs, machine.SIDs) {
			t.Fatalf("%s was checked with %+v (%v), want account %v", name, o, ok, want)
		}
	}
	_, owners, err := readPlacesOwned(f.engine.Places, f.engine.Set)
	if err != nil || !owners["diffusion.gguf"].Account || owners["decoder.safetensors"].Account {
		t.Fatalf("recorded owners %+v, %v", owners, err)
	}
	for _, bad := range []string{"S-1-5-18;admin", "not-a-sid", "S-1-5-18,S 1",
		// Coordinator follow-up 2: a third principal beside the machine's
		// owners, here the Codex sandbox group's SID, is refused, as is a
		// SID named twice.
		formatOwners(machine) + ",S-1-5-21-3833002460-2194338898-3510754826-1002;account",
		"S-1-5-21-3833002460-2194338898-3510754826-1002",
		"S-1-5-18,S-1-5-18",
	} {
		if _, ok := parseOwners(bad); ok {
			t.Fatalf("owners %q were read", bad)
		}
	}
	both := machine
	both.Account = true
	for _, good := range []trustedpath.Owners{machine, both, {Account: true}} {
		if got, ok := parseOwners(formatOwners(good)); !ok || got.Account != good.Account || !slices.Equal(got.SIDs, good.SIDs) {
			t.Fatalf("owners %+v read back as %+v, %v", good, got, ok)
		}
	}
	if err := os.WriteFile(f.engine.Places, []byte("diffusion.gguf\t"+f.placed["diffusion.gguf"]+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readPlaces(f.engine.Places, f.engine.Set); err == nil {
		t.Fatal("a record without owners was read")
	}
}
