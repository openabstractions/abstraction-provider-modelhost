package host

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

// TestLiveEmbedBGE exercises modelhost's real embed@1 route with a cached,
// pre-verified GGUF. The test owns its model declaration, fake lease book,
// socket root, and engine process; the external runner owns the memory and
// no-output limits. No result vector or source text is logged.
func TestLiveEmbedBGE(t *testing.T) {
	if os.Getenv("OA_LIVE_EMBED_BGE") != "1" {
		t.Skip("set OA_LIVE_EMBED_BGE=1 and MODELHOST_LIVE_* for an isolated cached-model embed check")
	}
	need := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is not set", name)
		}
		return value
	}
	program := need("MODELHOST_LIVE_ENGINE")
	state := need("MODELHOST_LIVE_STATE")
	model := need("MODELHOST_LIVE_MODEL")
	out := need("MODELHOST_LIVE_OUT")
	sockets := need("MODELHOST_LIVE_SOCKETS")
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	log, err := os.Create(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()

	const name = "live:bge-m3"
	pin, ok := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	if !ok {
		t.Fatal("no pinned engine for this platform")
	}
	var provider *inference.Provider
	h, err := New(Options{
		Models: []Model{{Name: name, Store: "live", Object: "bge"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin),
			Arguments:  []string{"--ctx-size", "8192", "--n-gpu-layers", "0", "--parallel", "1", "--ubatch-size", "256", "--no-mmap"},
			Embeddings: true, Sockets: sockets, MemoryLimit: 4 << 30, Ready: 3 * time.Minute, Stderr: log},
		Source: fakeSource{objects: map[string]Object{"live/bge": {Store: "live", ID: "bge", Path: model,
			File: filepath.Base(model), Family: "bge-m3", Size: info.Size()}}},
		Leases: newFakeLeases(), Idle: time.Hour,
		Active:  func(model string) bool { return provider != nil && provider.ActiveForModel(model) },
		OnError: func(err error) { t.Logf("provider: %v", err) },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		h.Close()
		entries, err := os.ReadDir(sockets)
		if err != nil && !os.IsNotExist(err) {
			t.Errorf("inspect socket root after Close: %v", err)
		} else if len(entries) != 0 {
			t.Errorf("Close left %d entries in the isolated socket root", len(entries))
		}
	}()

	r := router.New()
	engine := router.OpenAICompatible("modelhost", h.Base(name), "/v1", h.Models())
	engine.Profiles = []string{router.ProfileEmbed}
	r.SetHosts(engine)
	provider, err = inference.New(inference.Config{
		Router: r,
		Admit:  modelhostAdmit(h),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		HTTP: EngineClient(h, func(err error) { t.Logf("engine client: %v", err) }),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	result := provider.Embed(ctx, inference.Subject{Account: "live", Program: "embed-test"}, wire.EmbedRequest{
		Model: name, Inputs: []string{"A short test sentence."},
	})
	if result.Outcome != wire.EmbedOutcomeCompleted || len(result.Vectors) != 1 || result.Dimensions != 1024 {
		t.Fatalf("embed outcome=%s reason=%q vectors=%d dimensions=%d", result.Outcome, result.Reason, len(result.Vectors), result.Dimensions)
	}
	vector, err := base64.StdEncoding.DecodeString(result.Vectors[0])
	if err != nil || len(vector) != 1024*4 {
		t.Fatalf("embed vector bytes=%d decode error=%v", len(vector), err)
	}
	nonzero := false
	for index := 0; index < 1024; index++ {
		value := float64(math.Float32frombits(binary.LittleEndian.Uint32(vector[index*4:])))
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("embed vector contains a non-finite value at index %d", index)
		}
		nonzero = nonzero || value != 0
	}
	if !nonzero {
		t.Fatal("embed vector contains only zero values")
	}
	t.Logf("embed@1 completed: one %d-dimensional vector, %d input tokens", result.Dimensions, result.Usage.Input)
}
