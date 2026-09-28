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

// Admit is the inference.Admitter this provider runs after a host is picked
// and before the rights decision: it makes model the object this host's
// engine holds, so a load that cannot happen is the request's refusal in the
// host's own word (CONTRACT.md MODELHOST-L3, MODELHOST-H2) rather than an upstream
// failure. Pass it as Config.Admit before inference.New, so the provider this
// package's Listen serves already carries it.
func Admit(model *host.Host) inference.Admitter {
	return func(ctx context.Context, _ *router.Host, name string) (string, bool) {
		if _, err := model.Ensure(ctx, name); err != nil {
			word := host.WordOf(err)
			if word == "" {
				word = host.Unavailable
			}
			return word, false
		}
		return "", true
	}
}

// Listen binds endpoint for one model host's inference provider as a one-hop
// leaf: calls admitted here use the local engine directly and refuse another
// native or remote OA delegation (CONTRACT.md MODELHOST-R4). provider's
// Config.Admit should be Admit(model), set before inference.New, so a load
// that cannot happen reaches the caller as this provider's own refusal word.
func Listen(endpoint string, provider *inference.Provider, runtimePrograms ...string) (*Host, error) {
	return inferenceservice.ListenDesignatedLeaf(endpoint, provider, inference.ExecutionLocal, runtimePrograms...)
}
