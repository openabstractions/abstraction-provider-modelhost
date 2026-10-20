package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	registration "github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
	inference "github.com/openabstractions/abstraction-inference/go"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
)

func TestDeclareEmitsTypedFactsWithoutStartup(t *testing.T) {
	var out, diagnostics bytes.Buffer
	args := []string{"declare", "--endpoint", "declared", "--engine", "absent-engine", "--model", "literal:model=store/opaque/object", "--embeddings", "--config", "absent-config", "--state", "absent-state"}
	if code := run(args, &out, &diagnostics, func(string) string { return "" }); code != 0 {
		t.Fatalf("declaration read startup state: %d %s", code, &diagnostics)
	}
	spec, err := registration.DecodeRegistrationSpec(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	facet := spec.Providers[0]
	if owner := service.ProviderConfig(nil, nil, nil).ChatLogicalOwner; owner != facet.Name {
		t.Fatalf("history owner %q differs from declared provider %q", owner, facet.Name)
	}
	if !slices.Equal(facet.Services, []string{inference.Contract, inference.PreloadContract, inference.EmbedContract}) {
		t.Fatalf("configured chat/preload/embed declarations: %v", facet.Services)
	}
	if spec.Program == "" || facet.Arguments[0] != "serve" || len(facet.Models) != 2 || facet.Models[0].Model != "literal:model" || len(facet.Recipes) != 1 || facet.Recipes[0].Objects[0].Object != "opaque/object" || facet.Engines[0].Version != "" {
		t.Fatalf("typed configured facts: %+v", spec)
	}
}

func TestDeclareRejectsMalformedInputsAndCaps(t *testing.T) {
	for _, args := range [][]string{
		{"declare", "--endpoint", "declared", "--model", "partial=store/"},
		{"declare", "--endpoint", "declared", "--model", strings.Repeat("m", 257) + "=s/o"},
		{"declare", "--endpoint", "declared", "--model", "m=s/o", "--video-kind", "torch"},
	} {
		var out, diagnostics bytes.Buffer
		if code := run(args, &out, &diagnostics, func(string) string { return "" }); code == 0 || out.Len() != 0 {
			t.Fatalf("invalid declaration emitted: %v %d %s", args, code, &out)
		}
	}
	if err := writeServeDeclaration(failedWriter{}, []string{"--endpoint", "declared"}, "declared", "engine", false, "card:0", []string{"m=s/o"}, ""); err == nil {
		t.Fatal("lost declaration output reported success")
	}
}

func TestDeclarePersistsSelectedEnvironmentEngineInArguments(t *testing.T) {
	var out, diagnostics bytes.Buffer
	if code := run([]string{"declare", "--endpoint", "declared", "--model", "m=s/o"}, &out, &diagnostics, func(key string) string {
		if key == "MODELHOST_ENGINE" {
			return "selected-engine"
		}
		return ""
	}); code != 0 {
		t.Fatalf("declare: %d %s", code, &diagnostics)
	}
	spec, err := registration.DecodeRegistrationSpec(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	args := spec.Providers[0].Arguments
	if args[len(args)-2] != "--engine" || args[len(args)-1] != "selected-engine" {
		t.Fatalf("selected engine absent from declared configuration: %v", args)
	}
}
