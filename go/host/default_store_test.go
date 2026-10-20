package host_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/snapshot"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// The setup step hosts a model found in one of inventoryd's default stores,
// not Setup's own folder (ADD-DL 2026-10-31): the model host's own source,
// reading every default adapter, resolves a Hugging Face hub snapshot file
// by the store and manifest id the inventory listed.
func TestModelHostResolvesAnObjectFromTheHubCache(t *testing.T) {
	home := t.TempDir()
	rev := strings.Repeat("a", 40)
	repo := filepath.Join(home, ".cache", "huggingface", "hub", "models--acme--Coder-GGUF")
	file := filepath.Join(repo, "snapshots", rev, "Coder-Q4_K_M.gguf")
	for path, body := range map[string]string{filepath.Join(repo, "refs", "main"): rev, file: "coder weights"} {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	env := host.Env{GOOS: "test", Home: home, Getenv: func(string) string { return "" }, Digests: source.NewDigests("")}
	store, id := "", ""
	for _, m := range snapshot.Take(env, snapshot.Adapters()).Page.Manifests {
		for _, n := range m.Manifest.Names {
			if strings.HasSuffix(n.Name, ":Coder-Q4_K_M.gguf") {
				store, id = m.Manifest.Store, m.Manifest.ID
			}
		}
	}
	if store == "" {
		t.Fatal("the inventory listed no manifest for the hub file")
	}
	object, err := host.NewLocalStores(env).Resolve(store, id, bridge.FileOnly)
	if err != nil || object.Path != file {
		t.Fatalf("resolve %s/%s = %+v, %v", store, id, object, err)
	}
}
