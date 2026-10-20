package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	"github.com/openabstractions/abstraction-provider-kit/go/providerhost"
	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	router "github.com/openabstractions/abstraction-router/go"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
	contentdigest "github.com/openabstractions/abstraction-storage/go/digest"
)

// visionGGUF is a GGUF v3 header of string, uint32 and bool values, then
// filler standing in for tensors.
func visionGGUF(strs map[string]string, nums map[string]uint32, bools map[string]bool) []byte {
	out := append([]byte(nil), "GGUF"...)
	out = binary.LittleEndian.AppendUint32(out, 3)
	out = binary.LittleEndian.AppendUint64(out, 1)
	out = binary.LittleEndian.AppendUint64(out, uint64(len(strs)+len(nums)+len(bools)))
	str := func(s string) {
		out = binary.LittleEndian.AppendUint64(out, uint64(len(s)))
		out = append(out, s...)
	}
	for k, v := range nums {
		str(k)
		out = binary.LittleEndian.AppendUint32(out, 4)
		out = binary.LittleEndian.AppendUint32(out, v)
	}
	for k, v := range bools {
		str(k)
		out = binary.LittleEndian.AppendUint32(out, 7)
		if v {
			out = append(out, 1)
		} else {
			out = append(out, 0)
		}
	}
	for k, v := range strs {
		str(k)
		out = binary.LittleEndian.AppendUint32(out, 8)
		str(v)
	}
	return append(out, make([]byte, 8192)...)
}

// visionHub is a Hugging Face hub whose one snapshot holds a thinking,
// tool-calling model and its F16 projector, read by the model host's own
// local-stores source; it answers the source and the model's locator.
func visionHub(t *testing.T) (host.Source, string) {
	t.Helper()
	home := t.TempDir()
	rev := strings.Repeat("b", 40)
	repo := filepath.Join(home, ".cache", "huggingface", "hub", "models--acme--Vision-GGUF")
	snap := filepath.Join(repo, "snapshots", rev)
	files := map[string][]byte{
		filepath.Join(repo, "refs", "main"): []byte(rev),
		filepath.Join(snap, "Vision-4B-Q4_K_M.gguf"): visionGGUF(map[string]string{"general.architecture": "qwen3", "general.type": "model",
			"tokenizer.chat_template": "{%- if tools %}<tools>{{ tools }}</tools>{%- endif %}{%- if enable_thinking %}<think>{%- endif %}"},
			map[string]uint32{"qwen3.embedding_length": 2048}, nil),
		filepath.Join(snap, "mmproj-F16.gguf"): visionGGUF(map[string]string{"general.architecture": "clip", "general.type": "mmproj"},
			map[string]uint32{"clip.vision.projection_dim": 2048}, map[string]bool{"clip.has_vision_encoder": true}),
	}
	for path, body := range files {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	src := host.NewLocalStores(source.Env{GOOS: runtime.GOOS, Home: home, Getenv: func(string) string { return "" }})
	return src, "models--acme--Vision-GGUF/snapshots/" + rev + "/Vision-4B-Q4_K_M.gguf"
}

// visionLeaf composes modelhostd's chat provider as serve does: the model
// host over the hub, its facts read before Routes, the content store the
// runtime stages chat images in, and the endpoint stating capabilities.
func visionLeaf(t *testing.T, engine, runtimeProgram string) string {
	t.Helper()
	src, locator := visionHub(t)
	var provider *inference.Provider
	model, err := host.New(host.Options{
		Models: []host.Model{{Name: "vision:4b", Store: "huggingface", Object: locator}},
		Engine: pinnedEngine(t, engine),
		Source: src,
		Leases: granting{released: make(chan string, 4)},
		Idle:   time.Hour,
		Active: func(name string) bool { return provider != nil && provider.ActiveForModel(name) },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	model.RefreshFacts(ctx)
	cancel()
	root := t.TempDir()
	content, err := providerhost.NewDesignatedContent("modelhost-content", filepath.Join(root, "content"), filepath.Join(root, "uploads.json"), []string{runtimeProgram}, 20<<20)
	if err != nil {
		t.Fatal(err)
	}
	config := ProviderConfig(model, Routes(model, false), nil)
	config.ResolveContent = content.ResolveContent
	provider, err = inference.New(config)
	if err != nil {
		t.Fatal(err)
	}
	endpoint := endpointFor(t)
	served, err := Listen(endpoint, provider, runtimeProgram)
	if err != nil {
		t.Fatal(err)
	}
	served.Content = content.Component
	served.ModelCapabilities = ModelCapabilities(model)
	serveCtx, stop := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- served.Serve(serveCtx) }()
	t.Cleanup(func() { stop(); <-done; provider.Close() })
	return endpoint
}

func pngImage(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	img.Set(1, 1, color.RGBA{R: 200, A: 255})
	var b bytes.Buffer
	if err := png.Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// A runtime surveying modelhost reads each model's capabilities from the
// endpoint's description, and a chat@3 turn with an image part reaches the
// engine as image_url data: the runtime stages the image in modelhost's
// content store and modelhost's provider resolves it (MODELHOST-L12).
func TestChat3ImageReachesTheEngineAndCapabilitiesReachTheRouter(t *testing.T) {
	if err := identity.CanEver(listen.Program); err != nil || runtime.GOOS != "windows" {
		t.Skip("native engine peer proof requires Windows")
	}
	program, principal := nativeTitleIdentity(t)
	account, err := principal.Account()
	if err != nil {
		t.Fatal(err)
	}
	endpoint := visionLeaf(t, buildFakeEngine(t), program)

	native, err := router.NewNative(declaration.ProviderName, endpoint, listen.ServerExpectation{Principal: principal, Program: program}, []string{"vision:4b"}, []string{router.ProfileChat})
	if err != nil {
		t.Fatal(err)
	}
	routes := router.New(native)
	surveyCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := routes.SurveyContext(surveyCtx); err != nil {
		t.Fatal(err)
	}
	caps := routes.AliasCapabilities(declaration.ProviderName, "vision:4b")
	if caps == nil || !slices.Equal(caps.Input, []string{"text", "image"}) || !slices.Equal(caps.Output, []string{"text"}) || caps.Tools == nil || !*caps.Tools {
		t.Fatalf("alias capabilities %+v", caps)
	}
	if r := caps.Reasoning; r == nil || !r.Budget || len(r.Efforts) != 4 || r.DefaultEffort == nil || *r.DefaultEffort != "medium" ||
		!slices.Equal(caps.Extensions, []string{router.ExtensionLlamaCppEnableThinking, router.ExtensionLlamaCppReasoningBudget}) {
		t.Fatalf("alias reasoning %+v extensions %v", caps.Reasoning, caps.Extensions)
	}

	picture := pngImage(t)
	sum := sha256.Sum256(picture)
	digest := contentdigest.Hex(sum[:])
	mediator, err := inference.New(inference.Config{ChatLogicalOwner: "fixture-runtime-job-owner", Router: routes,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "not_permitted"
		},
		ResolveContent: func(_ context.Context, _ inference.Subject, d string, max int64) ([]byte, inference.ContentOutcome) {
			if d != digest || int64(len(picture)) > max {
				return nil, inference.ContentUnknown
			}
			return picture, inference.ContentResolved
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mediator.Close() })
	subject := inference.Subject{Account: account, Program: program}
	window := mediator.GetChatHistoryWindow(subject)
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: "vision", HistoryEpoch: window.Window.HistoryEpoch}, Target: wire.ChatTarget{Model: "vision:4b"},
		Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{
			{Kind: wire.PartKindText, Text: "describe this picture"}, {Kind: wire.PartKindImage, Digest: digest, MediaType: "image/png"}}}},
			Guarantees: []wire.ChatGuarantee{wire.ChatGuaranteeLocalOnly}}}
	a := mediator.StartV3(context.Background(), subject, request)
	if a.Outcome != wire.ChatStartOutcomeAccepted {
		t.Fatalf("runtime admission=%+v", a)
	}
	var text strings.Builder
	var end *wire.ChatResult
	observeCtx, done := context.WithTimeout(context.Background(), 30*time.Second)
	defer done()
	for cursor := int64(0); end == nil && observeCtx.Err() == nil; {
		page := mediator.ObserveV3(observeCtx, subject, a.Request, cursor, 256, 65536, 1000)
		if page.Outcome != wire.PageOutcomePage {
			t.Fatalf("observe=%+v", page)
		}
		for _, event := range page.Events {
			if event.Part != nil {
				text.WriteString(event.Part.Text)
			}
			if event.End != nil {
				end = event.End
			}
		}
		cursor = page.Next
	}
	if end == nil || end.Outcome != wire.ChatResultOutcomeCompleted {
		t.Fatalf("result=%+v text=%q", end, text.String())
	}
	if !strings.Contains(text.String(), "image:"+digest) {
		t.Fatalf("the engine answered %q without the image %s", text.String(), digest)
	}
}
