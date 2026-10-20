package host

import (
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-identity/placetest"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// ggufFile is a GGUF v3 header of string, uint32 and bool values in key
// order, then filler standing in for tensors.
func ggufFile(strs map[string]string, nums map[string]uint32, bools map[string]bool) []byte {
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

const visionRev = "0123456789abcdef0123456789abcdef01234567"

// visionStore is a Hugging Face hub with one snapshot holding a thinking,
// tool-calling model and its F16 projector with an audio encoder, and the
// model host's own local-stores source over it.
func visionStore(t *testing.T) (Source, string, int64, int64) {
	t.Helper()
	home := t.TempDir()
	repo := filepath.Join(home, ".cache", "huggingface", "hub", "models--acme--Vision-GGUF")
	snap := filepath.Join(repo, "snapshots", visionRev)
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "refs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(visionRev), 0o644); err != nil {
		t.Fatal(err)
	}
	model := ggufFile(map[string]string{"general.architecture": "qwen3", "general.type": "model",
		"tokenizer.chat_template": "{%- if tools %}<tools>{{ tools }}</tools>{%- endif %}{%- if enable_thinking %}<think>{%- endif %}"},
		map[string]uint32{"qwen3.embedding_length": 2048}, nil)
	projector := ggufFile(map[string]string{"general.architecture": "clip", "general.type": "mmproj"},
		map[string]uint32{"clip.vision.projection_dim": 2048}, map[string]bool{"clip.has_vision_encoder": true, "clip.has_audio_encoder": true})
	for name, body := range map[string][]byte{"Vision-4B-Q4_K_M.gguf": model, "mmproj-F16.gguf": projector} {
		if err := os.WriteFile(filepath.Join(snap, name), body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	src := NewLocalStores(source.Env{GOOS: runtime.GOOS, Home: home, Getenv: func(string) string { return "" }})
	return src, "models--acme--Vision-GGUF/snapshots/" + visionRev + "/Vision-4B-Q4_K_M.gguf", int64(len(model)), int64(len(projector))
}

// A model whose store holds its vision projector launches with --mmproj
// naming that file, and its lease counts the projector's bytes
// (MODELHOST-L12).
func TestAModelLaunchesWithItsProjector(t *testing.T) {
	requireEnginePeer(t)
	src, locator, modelSize, projectorSize := visionStore(t)
	argsFile := filepath.Join(t.TempDir(), "arguments")
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Source = src
		o.Models = []Model{{Name: "vision:4b", Store: "huggingface", Object: locator}}
		o.Engine.fixture = []string{"--fake-args-file", argsFile}
	})
	if _, err := h.Ensure(context.Background(), "vision:4b"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := strings.Split(string(raw), "\n")
	i := slices.Index(args, "--mmproj")
	if i < 0 || i+1 >= len(args) || filepath.Base(args[i+1]) != "mmproj-F16.gguf" {
		t.Fatalf("engine arguments %q carry no --mmproj mmproj-F16.gguf", args)
	}
	leases.mu.Lock()
	grants := slices.Clone(leases.grants)
	leases.mu.Unlock()
	if len(grants) != 1 || grants[0] != modelSize+projectorSize {
		t.Fatalf("lease grants %v, want the model %d and its projector %d", grants, modelSize, projectorSize)
	}
}

// The facts a chat model's own files state reach ChatModelFacts: image and
// audio input from its projector, tools from its template, and the default
// profile's reasoning for a template with a thinking section.
func TestChatModelFactsReadTheModelsFiles(t *testing.T) {
	src, locator, _, _ := visionStore(t)
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Source = src
		o.Models = []Model{{Name: "vision:4b", Store: "huggingface", Object: locator}, {Name: "gone:1b", Store: "huggingface", Object: "missing"}}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	h.RefreshFacts(ctx)
	names, facts := h.ChatModelFacts()
	if !slices.Equal(names, []string{"gone:1b", "vision:4b"}) {
		t.Fatalf("names %v", names)
	}
	if _, ok := facts["gone:1b"]; ok {
		t.Fatal("a model no store holds has facts")
	}
	f, ok := facts["vision:4b"]
	if !ok || !f.Image || !f.Audio || f.Tools == nil || !*f.Tools || f.Reasoning == nil {
		t.Fatalf("facts %+v", f)
	}
	want := engineProfiles["default"]
	if f.Reasoning.Default != want.DefaultReasoningEffort || f.Reasoning.Budgets["low"] != want.ReasoningBudgets["low"] || f.Reasoning.Budgets["high"] != want.ReasoningBudgets["high"] {
		t.Fatalf("reasoning %+v, want the default profile", f.Reasoning)
	}
	var _ bridge.Projectors = src.(*bridge.LocalStores)
}

func TestChatModelFactsRefuseAProjectorRedirectedAfterBinding(t *testing.T) {
	src, locator, _, _ := visionStore(t)
	model := Model{Name: "vision:4b", Store: "huggingface", Object: locator}
	h := &Host{options: Options{Source: src}}
	obj, err := src.Resolve(model.Store, model.Object, bridge.FileOnly)
	if err != nil {
		t.Fatal(err)
	}
	proj, ok := h.projector(obj)
	if !ok {
		t.Fatal("fixture has no projector")
	}
	external := t.TempDir()
	data, err := os.ReadFile(proj.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(external, proj.File), data, 0o600); err != nil {
		t.Fatal(err)
	}
	parent := filepath.Dir(proj.Path)
	original := parent + "-original"
	if err := os.Rename(parent, original); err != nil {
		t.Fatal(err)
	}
	obj.Path = filepath.Join(original, obj.File)
	if runtime.GOOS == "windows" {
		err = placetest.Junction(parent, external)
	} else {
		err = os.Symlink(external, parent)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Remove(parent); err != nil {
			t.Error(err)
		}
	})
	if f := h.factsFor(model, obj, proj, true); f.Image || f.Audio {
		t.Fatalf("redirected projector supplied metadata: %+v", f)
	}
}
