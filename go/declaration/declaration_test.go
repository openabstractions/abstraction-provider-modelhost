package declaration

import (
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
	"testing"

	registration "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
)

func TestBuildConfiguredFacts(t *testing.T) {
	facet := registration.ProviderFacet{Name: "configured", Endpoint: "logical", Activation: registration.ActivationOnDemand,
		Arguments: []string{"serve", "--endpoint", "{endpoint}"}, Services: []string{"abstraction.inference/chat@1"}, Models: []registration.ScopedModel{{Modality: "chat", Model: "m"}}}
	spec, err := Build("configured-program", facet, "configured-engine", []Model{{"m", "store", "object"}, {"unassociated", "store", "other"}})
	if err != nil {
		t.Fatal(err)
	}
	got := spec.Providers[0]
	if len(got.Engines) != 1 || got.Engines[0].Version != "" || len(got.Recipes) != 1 || got.Recipes[0].Objects[0].Object != "object" {
		t.Fatalf("facts: %+v", got)
	}
	// Every command that declares a chat facet names the key its loads read.
	if len(got.ConfigKeys) != 1 || got.ConfigKeys[0] != ContextKey {
		t.Fatalf("chat facet config keys %v", got.ConfigKeys)
	}
	video := facet
	video.Services = []string{"abstraction.inference/video@1"}
	video.ConfigKeys = []string{ContextKey}
	if spec, err := Build("configured-program", video, "", nil); err != nil || len(spec.Providers[0].ConfigKeys) != 0 {
		t.Fatalf("video-only facet config keys %+v %v", spec.Providers, err)
	}
	got.Arguments[0] = "changed"
	if facet.Arguments[0] != "serve" {
		t.Fatal("builder aliased input")
	}
	unknown, err := Build("configured-program", facet, "", []Model{{"m", "store", "object"}})
	if err != nil || len(unknown.Providers[0].Engines) != 0 || len(unknown.Providers[0].Recipes) != 0 {
		t.Fatalf("unknown selection: %+v %v", unknown, err)
	}
	partial, err := Build("configured-program", facet, "configured-engine", []Model{{Name: "m", Store: "store"}})
	if err != nil || len(partial.Providers[0].Recipes) != 0 {
		t.Fatalf("partial object: %+v %v", partial, err)
	}
	for _, endpoint := range []string{"", "{endpoint}"} {
		facet.Endpoint = endpoint
		if _, err := Build("program", facet, "engine", nil); err == nil {
			t.Fatalf("accepted endpoint %q", endpoint)
		}
	}
}

// Keep this shared package independent of producer runtime and execution code.
func pureImports(source string) bool {
	f, err := parser.ParseFile(token.NewFileSet(), "source.go", source, parser.ImportsOnly)
	if err != nil {
		return false
	}
	for _, item := range f.Imports {
		path, err := strconv.Unquote(item.Path.Value)
		if err != nil {
			return false
		}
		if path == "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration" {
			continue
		}
		if strings.Contains(strings.Split(path, "/")[0], ".") || path == "os" || path == "os/exec" || path == "net" || strings.HasPrefix(path, "net/") || path == "syscall" {
			return false
		}
	}
	return true
}

func TestDeclarationPackagePurity(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".go") && !strings.HasSuffix(entry.Name(), "_test.go") {
			source, err := os.ReadFile(entry.Name())
			if err != nil || !pureImports(string(source)) {
				t.Fatalf("impure declaration source %s: %v", entry.Name(), err)
			}
		}
	}
	for _, path := range []string{"os/exec", "os", "net/http", "github.com/openabstractions/abstraction-provider-modelhost/go/host", "github.com/openabstractions/abstraction-provider-modelhost/go/service"} {
		if pureImports("package fixture\nimport " + strconv.Quote(path)) {
			t.Fatalf("accepted runtime dependency %s", path)
		}
	}
}
