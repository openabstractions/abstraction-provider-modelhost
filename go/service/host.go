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
	"slices"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	inferenceservice "github.com/openabstractions/abstraction-inference/go/service"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/declaration"
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
		if model == nil {
			return host.Unavailable, false
		}
		ensure := model.Ensure
		if inference.PreloadAdmission(ctx) || inference.FreeCapacityAdmission(ctx) {
			// A preload never asks another holder to yield (MODELHOST-W2).
			ensure = model.EnsureSpeculative
		}
		reason, ready := admitLoad(ctx, name, ensure)
		if inference.FreeCapacityAdmission(ctx) && reason == "resources:held" {
			reason = "resource:held"
		}
		return reason, ready
	}
}

func admitLoad(ctx context.Context, name string, ensure func(context.Context, string) (string, error)) (string, bool) {
	// Start clients have a five-second transport budget. Return the typed
	// loading reason while the provider keeps its bounded load.
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	if _, err := ensure(ctx, name); err != nil {
		word := host.WordOf(err)
		if word == "" {
			word = host.Unavailable
		}
		return word, false
	}
	return "", true
}

// Routes is the router modelhostd serves. Each resident's engine is its own
// OpenAI-compatible host, reached at the one address this provider reserved
// for it: asking the engine would mean starting it, so the router is told
// the address up front, one host per allowlist name, never a shared one two
// engine processes could collide on (CONTRACT.md MODELHOST-H5). Each host
// declares chat, and embed only when the engines start with embeddings, so
// an embed request without them reads no_host instead of an upstream failure.
func Routes(model *host.Host, embeddings bool) *router.Router {
	profiles := []string{router.ProfileChat}
	if embeddings {
		profiles = append(profiles, router.ProfileEmbed)
	}
	var names []string
	if model != nil {
		names = model.Models()
	}
	var facts map[string]host.ModelFacts
	if model != nil {
		_, facts = model.ChatModelFacts()
	}
	hosts := make([]*router.Host, 0, len(names))
	for _, name := range names {
		engineHost := router.OpenAICompatible("modelhost-engine-"+name, model.Base(name), "/v1", []string{name})
		engineHost.Profiles = slices.Clone(profiles)
		// Each engine host declares what its model's files state, read
		// before the router is built (host.RefreshFacts); a model whose
		// facts are not read yet declares none.
		if f, ok := facts[name]; ok {
			caps, control := Capabilities(f)
			engineHost.DeclareModel(name, caps, control)
		}
		hosts = append(hosts, engineHost)
	}
	return router.New(hosts...)
}

// Capabilities is one chat model's facts as router capabilities: text in
// and out, image and audio input from its projector, tools from its
// template, and for a thinking family its llama.cpp reasoning control and
// the native extensions that switch thinking and bound it
// (CONTRACT.md MODELHOST-L12). An unknown fact is left out. The control is
// nil for a model with no reasoning profile.
func Capabilities(f host.ModelFacts) (*router.ModelCapabilities, *router.ReasoningControl) {
	caps := &router.ModelCapabilities{Input: []string{"text"}, Output: []string{"text"}}
	if f.Image {
		caps.Input = append(caps.Input, "image")
	}
	if f.Audio {
		caps.Input = append(caps.Input, "audio")
	}
	if f.Tools != nil {
		tools := *f.Tools
		caps.Tools = &tools
	}
	if f.Reasoning == nil {
		return caps, nil
	}
	control := &router.ReasoningControl{Kind: router.ReasoningLlamaCpp, Budgets: map[string]int64{}, Default: f.Reasoning.Default}
	for _, effort := range []string{"low", "medium", "high"} {
		if budget := f.Reasoning.Budgets[effort]; budget > 0 {
			control.Budgets[effort] = int64(budget)
		}
	}
	caps.Reasoning = control.Support()
	caps.Extensions = []string{router.ExtensionLlamaCppEnableThinking, router.ExtensionLlamaCppReasoningBudget}
	return caps, control
}

// ModelCapabilities is what the endpoint's chat@3 description states for
// each allowlisted chat model (inferenceservice.Host.ModelCapabilities): the
// facts model has cached, never waiting for a file read.
func ModelCapabilities(model *host.Host) func() ([]string, map[string]*router.ModelCapabilities) {
	return func() ([]string, map[string]*router.ModelCapabilities) {
		if model == nil {
			return nil, nil
		}
		names, facts := model.ChatModelFacts()
		out := make(map[string]*router.ModelCapabilities, len(facts))
		for name, f := range facts {
			out[name], _ = Capabilities(f)
		}
		return names, out
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
		ChatLogicalOwner: declaration.ProviderName,
		FreeCapacity:     true,
		EvaluatePrefix:   evaluate,
		Router:           r,
		// The runtime bound the application and decided its rules before it
		// forwarded; this endpoint admits that runtime's account alone.
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: func(ctx context.Context, routed *router.Host, name string) (string, bool) {
			if inference.AdmissionProfile(ctx) == "chat" && !inference.PreloadAdmission(ctx) && !inference.FreeCapacityAdmission(ctx) {
				if model == nil {
					return host.Unavailable, false
				}
				if err := model.CheckModel(name); err != nil {
					return host.WordOf(err), false
				}
				return "", true
			}
			reason, ready := Admit(model)(ctx, routed, name)
			if inference.FreeCapacityAdmission(ctx) && reason == "engine:loading" {
				// The bounded speculative load continues after the admission
				// wait; accepted execution joins it without a yielding acquire.
				return "", true
			}
			return reason, ready
		},
		WaitExecution: func(ctx context.Context, _ *router.Host, name string) inference.ExecutionPreparation {
			if model == nil {
				return inference.ExecutionPreparation{Reason: host.Unavailable}
			}
			progressCtx := host.WithLoadProgress(ctx, projectExecutionProgress(func(phase string) { inference.ReportExecutionProgress(ctx, phase) }))
			ensure := model.EnsureMeasured
			if inference.FreeCapacityAdmission(ctx) {
				ensure = model.EnsureSpeculativeMeasured
			}
			_, measured, err := ensure(progressCtx, name)
			duration := measured.Milliseconds
			result := inference.ExecutionPreparation{Ready: err == nil, Timing: inference.PreparationTiming{
				LoadMS: &duration, Load: measured.State, LoadBytes: measured.Bytes}}
			if err != nil {
				word := host.WordOf(err)
				if inference.FreeCapacityAdmission(ctx) && word == "resources:held" {
					word = "resource:held"
				}
				if word == "" {
					word = host.Unavailable
				}
				result.Reason = word
			}
			return result
		},
		HTTP:    host.EngineClient(engines(model), report),
		OnError: report,
	}
}

// projectExecutionProgress keeps engine parser words at the model-host boundary.
// The raw parser still advances the readiness watchdog on every genuine phase.
func projectExecutionProgress(report func(string)) func(string) {
	last := ""
	return func(raw string) {
		phase := ""
		switch raw {
		case "metadata", "tensors":
			phase = "loading"
		case "context", "graph":
			phase = "warming"
		}
		if phase != "" && phase != last {
			last = phase
			report(phase)
		}
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
