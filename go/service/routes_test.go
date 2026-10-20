package service

import (
	"context"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"

	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

// twoModels is a model host of two allowlist names and no engine program:
// a load is refused no_engine before anything starts.
func twoModels(t *testing.T) *host.Host {
	t.Helper()
	model, err := host.New(host.Options{
		Models: []host.Model{{Name: "tiny:1b", Store: "fake", Object: "tiny"}, {Name: "small:3b", Store: "fake", Object: "small"}},
		Engine: pinnedEngine(t, ""),
		Source: oneObject{object: host.Object{Store: "fake", ID: "tiny"}},
		Leases: granting{released: make(chan string, 4)},
		Idle:   time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(model.Close)
	return model
}

// MODELHOST-H5: modelhostd's router holds one engine host per allowlist name,
// named for it and at its own reserved address, declaring that name alone.
func TestRoutesHoldOneEngineHostPerModel(t *testing.T) {
	model := twoModels(t)
	r := Routes(model, false)
	hosts := r.Hosts()
	names := model.Models()
	if len(hosts) != len(names) {
		t.Fatalf("%d hosts for %d models", len(hosts), len(names))
	}
	for i, name := range names {
		h := hosts[i]
		if h.Name != "modelhost-engine-"+name || h.Base == "" || h.Base != model.Base(name) {
			t.Fatalf("%s: host %q at %q, wanted modelhost-engine-%s at %q", name, h.Name, h.Base, name, model.Base(name))
		}
		if !h.ServesModel(router.ProfileChat, name) {
			t.Fatalf("host %s declares no chat", h.Name)
		}
		if h.ServesModel(router.ProfileEmbed, name) {
			t.Fatalf("host %s declares embed without embeddings", h.Name)
		}
	}
	if model.Base(names[0]) == model.Base(names[1]) {
		t.Fatal("two engines share one address")
	}
	r.Survey()
	for _, name := range names {
		d, _ := r.Route(router.Request{Model: name, Profile: router.ProfileChat})
		if d.Verdict == router.NoHost || d.Host != "modelhost-engine-"+name {
			t.Fatalf("chat %s routed %s to %q (%s)", name, d.Verdict, d.Host, d.Why)
		}
	}
	for _, h := range Routes(model, true).Hosts() {
		if !h.ServesModel(router.ProfileEmbed, h.Name[len("modelhost-engine-"):]) {
			t.Fatalf("host %s declares no embed with embeddings", h.Name)
		}
	}
}

// The composed provider routes chat for each model to its engine, where the
// host's own admission refuses it no_engine. Embed reads no_host without
// --embeddings, and reaches the same admission with them.
func TestTheComposedProviderRoutesChatAndEmbed(t *testing.T) {
	model := twoModels(t)
	caller := inference.Subject{Account: "S-1-5-21-fixture", Program: `C:\apps\opencode.exe`}
	ctx := context.Background()
	compose := func(embeddings bool) *inference.Provider {
		p, err := inference.New(ProviderConfig(model, Routes(model, embeddings), nil))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = p.Close() })
		return p
	}
	without := compose(false)
	for _, name := range model.Models() {
		request := chat()
		request.Model = name
		a := without.Start(ctx, caller, request)
		if a.Outcome != wire.StartOutcomeUnavailable || a.Reason != host.NoEngine {
			t.Fatalf("chat %s: %+v, wanted unavailable %s from the host's admission", name, a, host.NoEngine)
		}
	}
	embed := wire.EmbedRequest{Model: "tiny:1b", Inputs: []string{"hello"}}
	if e := without.Embed(ctx, caller, embed); e.Outcome != wire.EmbedOutcomeNoHost {
		t.Fatalf("embed without embeddings: %+v, wanted no_host", e)
	}
	if e := compose(true).Embed(ctx, caller, embed); e.Outcome == wire.EmbedOutcomeNoHost || e.Outcome == wire.EmbedOutcomeCompleted {
		t.Fatalf("embed with embeddings: %+v, wanted the host's own refusal", e)
	}
	for _, name := range model.Models() {
		if model.Loaded(name) {
			t.Fatalf("a refused request loaded %s", name)
		}
	}
}
