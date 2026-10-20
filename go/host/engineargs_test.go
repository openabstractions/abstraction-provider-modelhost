package host

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-inference/go/providerhost/llamacpp"
)

// b10825 review condition 2 (CONTRACT.md MODELHOST-L5): a person's engine
// arguments pass only when each flag is on the reviewed list with a value of
// the reviewed form. Each refusal names the flag.
func TestEngineArgumentsPassOnlyTheReviewedList(t *testing.T) {
	allowed := map[string][]string{
		"pooling":           {"--pooling", "mean"},
		"context":           {"-c", "4096"},
		"equals form":       {"--ctx-size=8192"},
		"gpu layers":        {"-ngl", "99"},
		"gpu layers auto":   {"--n-gpu-layers", "auto"},
		"flash attention":   {"--flash-attn", "on"},
		"cache types":       {"-ctk", "q8_0", "-ctv", "q8_0"},
		"device":            {"--device", "Vulkan0,Vulkan1"},
		"switches":          {"--no-warmup", "--no-mmap", "--cont-batching"},
		"negative predict":  {"--n-predict", "-1"},
		"reasoning budget":  {"--reasoning-budget", "0"},
		"reasoning format":  {"--reasoning-format", "deepseek"},
		"parallel and seed": {"-np", "2", "--seed", "42"},
		"mtp":               {"--spec-type", "draft-mtp", "--spec-draft-n-max", "3"},
		"mtp equals form":   {"--spec-type=draft-mtp", "--spec-draft-n-max=8"},
		"mtp draft one":     {"--spec-draft-n-max", "1"},
	}
	for name, args := range allowed {
		t.Run("allowed "+name, func(t *testing.T) {
			got, err := EngineArguments(args)
			if err != nil {
				t.Fatalf("%q refused: %v", args, err)
			}
			if len(got) == 0 {
				t.Fatalf("%q produced no arguments", args)
			}
		})
	}
	refused := map[string]struct {
		args []string
		flag string
	}{
		"rpc":                {[]string{"--rpc", "10.0.0.2:50052"}, "--rpc"},
		"lora":               {[]string{"--lora", `C:\x\adapter.gguf`}, "--lora"},
		"chat template file": {[]string{"--chat-template-file", `C:\x\t.jinja`}, "--chat-template-file"},
		"chat template":      {[]string{"--chat-template", "{{ x }}"}, "--chat-template"},
		"host any":           {[]string{"--host", "0.0.0.0"}, "--host"},
		"host loopback":      {[]string{"--host", "127.0.0.1"}, "--host"},
		"port":               {[]string{"--port", "9"}, "--port"},
		"model":              {[]string{"-m", `C:\x\other.gguf`}, "-m"},
		"api key file":       {[]string{"--api-key-file", `C:\x\k`}, "--api-key-file"},
		"ssl key":            {[]string{"--ssl-key-file", `C:\x\k.pem`}, "--ssl-key-file"},
		"static path":        {[]string{"--path", `C:\`}, "--path"},
		"hf short":           {[]string{"-hf", "org/repo"}, "-hf"},
		"hf repo":            {[]string{"--hf-repo", "org/repo"}, "--hf-repo"},
		"model url":          {[]string{"--model-url", "https://example.invalid/m.gguf"}, "--model-url"},
		"mmproj":             {[]string{"--mmproj", `C:\x\p.gguf`}, "--mmproj"},
		"tools":              {[]string{"--tools", "all"}, "--tools"},
		"agent":              {[]string{"--agent"}, "--agent"},
		"mcp":                {[]string{"--mcp-servers-json", "{}"}, "--mcp-servers-json"},
		"sleep":              {[]string{"--sleep-idle-seconds", "5"}, "--sleep-idle-seconds"},
		"media path":         {[]string{"--media-path", `C:\`}, "--media-path"},
		"ffmpeg dir":         {[]string{"--video-ffmpeg-dir", `C:\ff`}, "--video-ffmpeg-dir"},
		"log file":           {[]string{"--log-file", `C:\x\l`}, "--log-file"},
		"slot save":          {[]string{"--slot-save-path", `C:\x`}, "--slot-save-path"},
		"models dir":         {[]string{"--models-dir", `C:\x`}, "--models-dir"},
		"embeddings":         {[]string{"--embeddings"}, "--embeddings"},
		"alias":              {[]string{"--alias", "x"}, "--alias"},
		"unknown":            {[]string{"--frobnicate"}, "--frobnicate"},
		"positional":         {[]string{"model.gguf"}, "model.gguf"},
		"bad number":         {[]string{"--ctx-size", "lots"}, "--ctx-size"},
		"bad choice":         {[]string{"--pooling", "evil"}, "--pooling"},
		"missing value":      {[]string{"--ctx-size"}, "--ctx-size"},
		"value on switch":    {[]string{"--no-mmap=1"}, "--no-mmap"},
		"web ui":             {[]string{"--no-webui"}, "--no-webui"},
		"device path":        {[]string{"--device", `..\x`}, "--device"},
		"repeated":           {[]string{"-c", "1", "--ctx-size", "2"}, "--ctx-size"},
		"spec type other":    {[]string{"--spec-type", "ngram-simple"}, "--spec-type"},
		"spec type draft":    {[]string{"--spec-type=draft"}, "--spec-type"},
		"spec type path":     {[]string{"--spec-type", `C:\x\d.gguf`}, "--spec-type"},
		"draft n zero":       {[]string{"--spec-draft-n-max", "0"}, "--spec-draft-n-max"},
		"draft n nine":       {[]string{"--spec-draft-n-max=9"}, "--spec-draft-n-max"},
		"draft n word":       {[]string{"--spec-draft-n-max", "many"}, "--spec-draft-n-max"},
		"model draft":        {[]string{"--model-draft", `C:\x\d.gguf`}, "--model-draft"},
		"model draft short":  {[]string{"-md", `C:\x\d.gguf`}, "-md"},
		"hf repo draft":      {[]string{"--hf-repo-draft", "org/repo"}, "--hf-repo-draft"},
		"hf draft short":     {[]string{"-hfd", "org/repo"}, "-hfd"},
		"lookup static":      {[]string{"--lookup-cache-static", `C:\x\c`}, "--lookup-cache-static"},
		"lookup dynamic":     {[]string{"--lookup-cache-dynamic", `C:\x\c`}, "--lookup-cache-dynamic"},
		"repeated spec type": {[]string{"--spec-type", "draft-mtp", "--spec-type=draft-mtp"}, "--spec-type"},
		"repeated draft n":   {[]string{"--spec-draft-n-max", "3", "--spec-draft-n-max", "4"}, "--spec-draft-n-max"},
	}
	for name, c := range refused {
		t.Run("refused "+name, func(t *testing.T) {
			got, err := EngineArguments(c.args)
			if err == nil {
				t.Fatalf("%q passed as %q", c.args, got)
			}
			if !errors.Is(err, ErrEngineArgument) || !strings.Contains(err.Error(), c.flag) {
				t.Fatalf("%q: %v, wanted ErrEngineArgument naming %s", c.args, err, c.flag)
			}
		})
	}
}

// The reviewed list sits on top of the provider kit's forbidden options
// (inference CONTRACT.md INF-LC1): no spelling it admits is one the kit
// forbids, and every forbidden spelling is refused as ErrEngineArgument
// wrapping the kit's error.
func TestEngineArgumentListIsDisjointFromTheForbiddenFlags(t *testing.T) {
	forbidden := llamacpp.ForbiddenSpellings()
	if len(forbidden) == 0 {
		t.Fatal("the kit lists no forbidden option")
	}
	for _, name := range forbidden {
		if _, ok := engineFlags[name]; ok {
			t.Errorf("the reviewed list admits the forbidden %s", name)
		}
		if long, ok := engineFlagNames[name]; ok {
			t.Errorf("the reviewed list admits the forbidden %s as %s", name, long)
		}
		for _, args := range [][]string{{name, "x"}, {name + "=x"}} {
			if _, err := EngineArguments(args); !errors.Is(err, ErrEngineArgument) || !errors.Is(err, llamacpp.ErrForbiddenFlag) {
				t.Errorf("%q: %v, wanted the kit's refusal", args, err)
			}
		}
	}
	for long := range engineFlags {
		if _, bad := llamacpp.ForbiddenFlag(long); bad {
			t.Errorf("the reviewed list admits the forbidden %s", long)
		}
	}
	for short, long := range engineFlagNames {
		if _, bad := llamacpp.ForbiddenFlag(short); bad {
			t.Errorf("the reviewed list admits the forbidden %s (%s)", short, long)
		}
	}
}

// The passed list is the canonical one: one token per flag and one per value.
func TestEngineArgumentsAreCanonical(t *testing.T) {
	got, err := EngineArguments([]string{"-c=2048", "-ngl", "99", "--no-warmup"})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--ctx-size", "2048", "--n-gpu-layers", "99", "--no-warmup"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q, wanted %q", got, want)
	}
}

// A host configured with an argument outside the list is refused at New, so
// no call ever starts an engine with it.
func TestHostRefusesAnEngineArgumentOutsideTheList(t *testing.T) {
	options := Options{
		Models: []Model{{Name: "tiny:1b", Store: "fake", Object: "tiny"}},
		Engine: Engine{Arguments: []string{"--rpc", "10.0.0.2:50052"}},
		Source: fakeSource{}, Leases: newFakeLeases(),
	}
	h, err := New(options)
	if err == nil {
		h.Close()
		t.Fatal("a host with --rpc was made")
	}
	if !errors.Is(err, ErrEngineArgument) || !strings.Contains(err.Error(), "--rpc") {
		t.Fatalf("New: %v, wanted ErrEngineArgument naming --rpc", err)
	}
}
