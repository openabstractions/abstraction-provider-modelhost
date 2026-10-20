// Package service serves abstraction.inference/chat@1 and
// abstraction.inference/embed@1 on the shared identity-bound native transport,
// for callers of this account only. Its one caller is the runtime, which binds
// the application, decides its rights and forwards; applications never reach
// this endpoint.
//
// Every admitted call passes through the model host first: the model the
// request names is loaded, under a lease, before the call is mediated to the
// engine. A load that cannot happen is the request's refusal, in the host's
// own words. Admit (abstraction-inference CONTRACT.md INF-A5) is how this
// provider gets that boundary from the shared inference service host instead
// of serving its own dispatcher.
package service

import (
	"context"

	inference "github.com/openabstractions/abstraction-inference/go"
	inferenceservice "github.com/openabstractions/abstraction-inference/go/service"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// MaxFrameBytes bounds one request or reply frame, as the inference service's
// own endpoint does.
const MaxFrameBytes = inferenceservice.MaxFrameBytes

// Host serves the inference profiles on one endpoint, binding every caller by
// native Program proof and refusing any program outside its runtime designation.
type Host = inferenceservice.Host

// Admit is the inference.Admitter this provider runs after authority and
// request preflight, before accepted execution: it makes model the object this host's
// engine holds, so a load that cannot happen is the request's refusal in the
// host's own word (CONTRACT.md MODELHOST-L3, MODELHOST-H2) rather than an upstream
// failure. Pass it as Config.Admit before inference.New, so the provider this
// package's Listen serves already carries it.
func Admit(model *host.Host) inference.Admitter {
	return func(ctx context.Context, _ *router.Host, name string) (string, bool) {
		ensure := model.Ensure
		if inference.PreloadAdmission(ctx) {
			// A preload never asks another holder to yield (MODELHOST-W2).
			ensure = model.EnsureSpeculative
		}
		if _, err := ensure(ctx, name); err != nil {
			word := host.WordOf(err)
			if word == "" {
				word = host.Unavailable
			}
			return word, false
		}
		return "", true
	}
}

// ProviderConfig is the inference provider configuration modelhostd serves:
// r routes each allowlist name to its engine, Admit loads the model, the
// runtime's own decision stands for the caller's rights, and every engine
// request passes host.EngineClient's media guard (CONTRACT.md MODELHOST-L8).
// report receives provider faults and each refused engine request.
func ProviderConfig(model *host.Host, r *router.Router, report func(error)) inference.Config {
	var evaluate inference.PrefixEvaluator
	if model != nil {
		// preload@1: the engine evaluates the caller's own prefix after
		// Admit loaded the model (CONTRACT.md MODELHOST-W2).
		evaluate = model.EvaluatePrefix
	}
	return inference.Config{
		EvaluatePrefix: evaluate,
		Router: r,
		// The runtime bound the application and decided its rules before it
		// forwarded; this endpoint admits that runtime's account alone.
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit:   Admit(model),
		HTTP:    host.EngineClient(engines(model), report),
		OnError: report,
	}
}

// engines is model as the engine client reads it; nil reaches no engine.
func engines(model *host.Host) host.Engines {
	if model == nil {
		return nil
	}
	return model
}

// Listen binds endpoint for one model host's inference provider as a one-hop
// leaf: calls admitted here use the local engine directly and refuse another
// native or remote OA delegation (CONTRACT.md MODELHOST-R4). provider's
// Config.Admit should be Admit(model), set before inference.New, so a load
// that cannot happen reaches the caller as this provider's own refusal word.
func Listen(endpoint string, provider *inference.Provider, runtimePrograms ...string) (*Host, error) {
	return inferenceservice.ListenDesignatedLeaf(endpoint, provider, inference.ExecutionLocal, runtimePrograms...)
}
