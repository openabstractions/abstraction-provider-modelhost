package host

import (
	"testing"

	registration "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
)

func TestDeclareEngineFactsUsesExactConfiguredObjects(t *testing.T) {
	facet := &registration.ProviderFacet{Models: []registration.ScopedModel{{Modality: "chat", Model: "literal:model"}, {Modality: "embed", Model: "literal:model"}}}
	DeclareEngineFacts(facet, "private/engine", []Model{{Name: "literal:model", Store: "store", Object: "opaque/object"}, {Name: "unknown", Store: "store", Object: "elsewhere"}})
	if len(facet.Engines) != 1 || facet.Engines[0].Version != "" || len(facet.Recipes) != 1 || len(facet.Recipes[0].Modalities) != 2 || facet.Recipes[0].Objects[0].Object != "opaque/object" {
		t.Fatalf("configured facts: %+v", facet)
	}
}

func TestDeclareEngineFactsKeepsUnknownAndPartialObjectsUnknown(t *testing.T) {
	for _, engine := range []string{"", "private/engine"} {
		facet := &registration.ProviderFacet{Models: []registration.ScopedModel{{Modality: "chat", Model: "m"}}}
		DeclareEngineFacts(facet, engine, []Model{{Name: "m", Store: "store"}})
		if len(facet.Recipes) != 0 || engine == "" && len(facet.Engines) != 0 {
			t.Fatalf("inferred object/engine: %+v", facet)
		}
	}
}
