package host

import (
	registration "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
)

// DeclareEngineFacts adds the producer's exact configured object associations.
// An empty engine leaves the selection unknown. Version is unknown until the
// producer has established the selected pin; an executable path proves none.
func DeclareEngineFacts(facet *registration.ProviderFacet, engine string, models []Model) {
	configured := make([]declaration.Model, 0, len(models))
	for _, model := range models {
		configured = append(configured, declaration.Model{Name: model.Name, Store: model.Store, Object: model.Object})
	}
	engines, recipes := declaration.Facts(*facet, engine, configured)
	facet.Engines = append(facet.Engines, engines...)
	facet.Recipes = append(facet.Recipes, recipes...)
}
