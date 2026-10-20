// Package declaration builds configured provider facts without reading files,
// observing installed engines, or starting a native process.
package declaration

import (
	"errors"
	"regexp"
	"slices"

	registration "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
)

// Model is one explicitly configured storage object. Names and opaque object
// identities are supplied by configuration; formats and capabilities are unknown.
type Model struct{ Name, Store, Object string }

// ProviderName is the compiled declaration identity and chat history owner.
const ProviderName = "modelhost"

// ContextKey is the configuration key holding the person's per-conversation
// context for each local model family. A facet that serves chat names it, so
// the runtime designates the model host to read it (config CONTRACT.md CFG-X1,
// architect decision D, 2026-11-08).
const ContextKey = "model_context"

// chatContract is the service whose loads read ContextKey.
const chatContract = "abstraction.inference/chat@1"

// ReadsContext reports whether facet's loads read ContextKey: a facet that
// serves chat does. The runtime applies the same rule to a kept model host
// registration written before its declaration named the key.
func ReadsContext(facet registration.ProviderFacet) bool {
	return slices.Contains(facet.Services, chatContract)
}

var logicalEndpoint = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]{0,63}$`)

// Facts reports exact configured model/engine associations. An empty engine
// configuration leaves selection unknown. Version and installed state remain
// unknown even when configuration names an engine executable.
func Facts(facet registration.ProviderFacet, engine string, models []Model) ([]registration.EngineFact, []registration.ModelRecipe) {
	if engine == "" {
		return nil, nil
	}
	engines := []registration.EngineFact{{ID: "llamacpp", Name: "llama.cpp"}}
	var recipes []registration.ModelRecipe
	for _, model := range models {
		var modalities []string
		for _, scoped := range facet.Models {
			if scoped.Model == model.Name {
				modalities = append(modalities, scoped.Modality)
			}
		}
		if len(modalities) == 0 || model.Store == "" || model.Object == "" {
			continue
		}
		recipes = append(recipes, registration.ModelRecipe{Model: model.Name, Engine: "llamacpp", Modalities: modalities,
			Objects: []registration.RecipeObject{{Role: "model", Store: model.Store, Object: model.Object}}})
	}
	return engines, recipes
}

// Build returns the generated native declaration from explicit logical inputs.
// The facet carries the logical endpoint and literal activation arguments as
// separate fields; no endpoint token or argument is interpreted here. Existing
// facts are rebuilt from this configuration, while other metadata is preserved.
// A facet that serves chat names ContextKey whichever command declared it:
// `modelhostd declare` and the runtime's `models host`, which Setup runs.
func Build(program string, facet registration.ProviderFacet, engine string, models []Model) (spec registration.RegistrationSpec, err error) {
	defer func() {
		if value := recover(); value != nil {
			if refusal, ok := value.(*registration.Refusal); ok {
				spec = registration.RegistrationSpec{}
				err = refusal
				return
			}
			panic(value)
		}
	}()
	if program == "" {
		return spec, errors.New("declaration: program is required")
	}
	if !logicalEndpoint.MatchString(facet.Endpoint) {
		return spec, errors.New("declaration: logical endpoint is required")
	}
	if len(facet.Services) == 0 {
		return spec, errors.New("declaration: services are required")
	}
	facet.Engines, facet.Recipes = nil, nil
	facet.ConfigKeys = nil
	if ReadsContext(facet) {
		facet.ConfigKeys = []string{ContextKey}
	}
	raw := registration.EncodeProviderFacet(&facet)
	copy, err := registration.DecodeProviderFacet(raw)
	if err != nil {
		return spec, err
	}
	copy.Engines, copy.Recipes = Facts(*copy, engine, slices.Clone(models))
	spec = registration.RegistrationSpec{Program: program, Providers: []registration.ProviderFacet{*copy}}
	decoded, err := registration.DecodeRegistrationSpec(registration.EncodeRegistrationSpec(&spec))
	if err != nil {
		return registration.RegistrationSpec{}, err
	}
	return *decoded, nil
}
