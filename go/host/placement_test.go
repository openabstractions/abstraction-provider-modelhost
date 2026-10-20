package host

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// CONTRACT.md MODELHOST-L9: the default placement follows the lease. A card
// lease reads the file into the engine's buffers and leaves the layer count
// to the engine's --fit; a memory lease keeps the layers in system memory;
// any other resource adds nothing. The pinned engines are one backend,
// Vulkan on Windows x64 (pin.go), and the rule names none: it reads only the
// lease's resource and the file size.
func TestEnginePlacementFollowsTheLease(t *testing.T) {
	const size = 22_853_663_008
	unmapped := []string{"--no-mmap"}
	cases := map[string]struct {
		resource         string
		size             int64
		configured, want []string
	}{
		"card lease":                   {"card:0", size, nil, unmapped},
		"second card":                  {"card:1", size, nil, unmapped},
		"card with no size":            {"card:0", 0, nil, nil},
		"memory lease":                 {"memory", size, nil, []string{"--n-gpu-layers", "0"}},
		"other resource":               {"awake", size, nil, nil},
		"person sets layers":           {"card:0", size, []string{"--n-gpu-layers", "20"}, unmapped},
		"person sets layers on memory": {"memory", size, []string{"--n-gpu-layers", "8"}, nil},
		"person maps the file":         {"card:0", size, []string{"--mmap"}, nil},
		"person sets the load mode":    {"card:0", size, []string{"--load-mode", "dio"}, nil},
		"person locks the file":        {"card:0", size, []string{"--mlock"}, nil},
		"person gives no-mmap":         {"card:0", size, []string{"--no-mmap"}, nil},
		"person maps on memory":        {"memory", size, []string{"--load-mode", "mmap"}, []string{"--n-gpu-layers", "0"}},
		"unrelated person arguments":   {"card:0", size, []string{"--ctx-size", "32768", "--flash-attn", "on"}, unmapped},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			got := enginePlacement(c.resource, c.size, c.configured)
			if !slices.Equal(got, c.want) {
				t.Fatalf("enginePlacement(%q, %d, %q) = %q, wanted %q",
					c.resource, c.size, c.configured, got, c.want)
			}
		})
	}
}

// A person's arguments override the default whatever spelling they used:
// EngineArguments canonicalises -ngl before the placement reads it.
func TestEnginePlacementReadsCanonicalSpellings(t *testing.T) {
	for resource, given := range map[string][]string{"card:0": {"--load-mode=mmap"}, "memory": {"-ngl=12"}} {
		configured, err := EngineArguments(given)
		if err != nil {
			t.Fatal(err)
		}
		if got := enginePlacement(resource, 10, configured); len(got) != 0 {
			t.Fatalf("%s: placement %q beside the person's %q", resource, got, configured)
		}
	}
}

// The reviewed list admits the load options a person needs to replace the
// default, and only their reviewed values.
func TestEngineArgumentsAdmitTheLoadOptions(t *testing.T) {
	for _, args := range [][]string{{"--mmap"}, {"--load-mode", "none"}, {"--load-mode=mmap+mlock"}, {"--load-mode", "dio"}} {
		if _, err := EngineArguments(args); err != nil {
			t.Errorf("%q refused: %v", args, err)
		}
	}
	for _, args := range [][]string{{"--load-mode", `C:\x`}, {"--load-mode"}, {"--mmap=1"}} {
		if got, err := EngineArguments(args); err == nil {
			t.Errorf("%q passed as %q", args, got)
		}
	}
}

// The engine a card lease starts receives the default placement once, after
// the arguments the provider sets itself; one a person configured receives
// theirs in its place.
func TestTheEngineStartsWithTheLeasePlacement(t *testing.T) {
	requireEnginePeer(t)
	cases := map[string]struct {
		configured []string
		want       map[string]string
		absent     []string
	}{
		"default":  {nil, map[string]string{"--no-mmap": ""}, []string{"--n-gpu-layers"}},
		"override": {[]string{"-ngl", "8", "--load-mode", "mmap"}, map[string]string{"--n-gpu-layers": "8", "--load-mode": "mmap"}, []string{"--no-mmap"}},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			argsFile := filepath.Join(t.TempDir(), "args")
			h := composed(t, newFakeLeases(), func(o *Options) {
				o.Engine.Arguments = c.configured
				o.Engine.fixture = []string{"--fake-args-file", argsFile}
			})
			if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(argsFile)
			if err != nil {
				t.Fatal(err)
			}
			args := strings.Split(string(raw), "\n")
			for flag, value := range c.want {
				at := slices.Index(args, flag)
				if at < 0 {
					t.Fatalf("%s missing from %q", flag, args)
				}
				if slices.Index(args[at+1:], flag) >= 0 {
					t.Fatalf("%s given twice in %q", flag, args)
				}
				if value != "" && (at+1 >= len(args) || args[at+1] != value) {
					t.Fatalf("%s is not %s in %q", flag, value, args)
				}
			}
			for _, flag := range c.absent {
				if slices.Contains(args, flag) {
					t.Fatalf("%s beside the person's load mode in %q", flag, args)
				}
			}
		})
	}
}
