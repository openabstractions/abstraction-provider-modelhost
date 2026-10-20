package host

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

type defaultProofSource struct {
	Source
	proof   content.Object
	current bool
}

func (s defaultProofSource) ContentProof(string, string) (content.Object, bool) {
	return s.proof, s.current
}

type legacyDefaultSource struct{ Source }

func TestCuratedModelDefaultsRequireCurrentMatchingProof(t *testing.T) {
	fit := modelservice.FitCatalog(context.Background(), modelwire.CatalogUseCode, modelservice.CatalogSources{Pool: func(context.Context) (int64, error) { return 0, nil }, Served: func(context.Context, string) ([]modelservice.ServedFamily, error) { return nil, nil }})
	if len(fit.Entries) == 0 {
		t.Fatal("no curated code model")
	}
	entry := fit.Entries[0]
	model := Model{Name: "legacy-alias", Store: "store", Object: "opaque-object"}
	object := Object{Store: model.Store, ID: model.Object, Locator: "weights.gguf", Size: entry.DownloadBytes}
	proof := content.Object{Locator: object.Locator, Size: object.Size, Modified: "2026-11-02T00:00:00Z", Digest: entry.Digest, Evidence: content.EvidenceHashed}
	base := []string{"--parallel", "1"}
	h := &Host{options: Options{Engine: Engine{Arguments: base}, Source: defaultProofSource{proof: proof, current: true}}}
	got := h.modelEngine(model, object)
	if got.modelPin != nil {
		t.Fatal("current Hashed observation created a launch pin without Content")
	}
	want := []string{"--parallel", "1", "--ctx-size", "32768", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"}
	if !reflect.DeepEqual(got.Arguments, want) {
		t.Fatalf("defaults %v", got.Arguments)
	}
	got.Arguments[0] = "changed"
	if !reflect.DeepEqual(h.options.Engine.Arguments, base) {
		t.Fatal("per-model defaults mutated shared engine")
	}
	for name, change := range map[string]func(*content.Object){"named": func(p *content.Object) { p.Evidence = content.EvidenceNamed }, "digest": func(p *content.Object) {
		p.Digest = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	}, "size": func(p *content.Object) { p.Size-- }, "binding": func(p *content.Object) { p.Locator = "other.gguf" }, "stamp": func(p *content.Object) { p.Modified = "" }, "partial": func(p *content.Object) { p.Partial = true }} {
		t.Run(name, func(t *testing.T) {
			p := proof
			change(&p)
			h.options.Source = defaultProofSource{proof: p, current: true}
			if args := h.modelEngine(model, object).Arguments; !reflect.DeepEqual(args, base) {
				t.Fatalf("unproven defaults %v", args)
			}
		})
	}
	for _, src := range []Source{legacyDefaultSource{}, defaultProofSource{proof: proof, current: false}} {
		h.options.Source = src
		if args := h.modelEngine(model, object).Arguments; !reflect.DeepEqual(args, base) {
			t.Fatalf("missing/stale optional proof changed args %v", args)
		}
	}
	h.options.Source = defaultProofSource{proof: proof, current: true}
	h.options.Engine.Arguments = []string{"-c", "4096", "--spec-type", "draft-mtp", "--spec-draft-n-max", "5"}
	want, err := EngineArguments(h.options.Engine.Arguments)
	if err != nil {
		t.Fatal(err)
	}
	if args := h.modelEngine(model, object).Arguments; !reflect.DeepEqual(args, want) {
		t.Fatalf("operator override lost: %v", args)
	}
}

func TestExplicitContentPinSurvivesOptionalDefaultsEarlyReturns(t *testing.T) {
	model := Model{Store: "oa-chat-spoof", Object: "opaque", Content: &programfiles.File{Size: 7, SHA256: strings.Repeat("a", 64)}}
	for _, source := range []Source{legacyDefaultSource{}, defaultProofSource{current: false}} {
		for _, embeddings := range []bool{false, true} {
			h := &Host{options: Options{Source: source, Engine: Engine{Record: "record", Embeddings: embeddings}}}
			engine := h.modelEngine(model, Object{Path: "model.gguf", Size: 7})
			if engine.modelPin == nil || engine.modelPin.SHA256 != model.Content.SHA256 || engine.modelPin.Size != 7 || !engine.modelPin.Owners.Account {
				t.Fatalf("explicit pin bypassed: %+v", engine.modelPin)
			}
			model.Content = nil
			if engine := h.modelEngine(model, Object{}); engine.modelPin != nil {
				t.Fatal("a store prefix created Content")
			}
			model.Content = &programfiles.File{Size: 7, SHA256: strings.Repeat("a", 64)}
		}
	}
}

func TestReservedExtraRootConfigurationSuppliesNoLaunchPin(t *testing.T) {
	root := t.TempDir()
	src := NewLocalStores(Env{GOOS: "test", Home: t.TempDir(), Getenv: func(string) string { return "" },
		Config: source.Config{ExtraRoots: []source.ExtraRoot{{Store: "oa-chat-spoof", Path: root}}}})
	h := &Host{options: Options{Source: src}}
	model := Model{Name: "coding", Store: "oa-chat-spoof", Object: "opaque"}
	if engine := h.modelEngine(model, Object{Path: root + "/model.gguf", Size: 7}); model.Content != nil || engine.modelPin != nil {
		t.Fatal("caller-authored extra root configuration supplied origin or a launch pin")
	}
}
