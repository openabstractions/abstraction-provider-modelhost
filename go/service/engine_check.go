package service

import wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"

// EngineCheck is ModelHost's captured observation of its startup preparation.
// The producer supplies it after its existing verification owner has returned.
type EngineCheck = wire.EndpointProducerCheck

const (
	EngineCheckKind  = "abstraction.provider.modelhost/engine"
	EngineCurrent    = wire.EndpointProducerCheckStateCurrent
	EngineRepaired   = wire.EndpointProducerCheckStateRepaired
	EngineRefused    = wire.EndpointProducerCheckStateRefused
	EngineUnverified = wire.EndpointProducerCheckStateUnverified
)

// EngineChecks snapshots startup evidence for Describe. Reading it performs no
// file access, preparation or inference readiness check.
func EngineChecks(check EngineCheck) func() []wire.EndpointProducerCheck {
	return func() []wire.EndpointProducerCheck { return []wire.EndpointProducerCheck{check} }
}
