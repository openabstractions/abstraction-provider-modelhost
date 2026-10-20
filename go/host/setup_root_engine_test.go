package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/adapters/extra"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/snapshot"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// Owner's preview, 2026-11-01 (native:object:unknown:oa-chat-…): a model in
// Setup's model folder, a store only Setup's inventory source reads through
// its --config, starts its engine once that configuration is among the
// model host's --config files.
func TestAModelInSetupsModelFolderStartsItsEngine(t *testing.T) {
	requireEnginePeer(t)
	home := t.TempDir()
	folder := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	// A GGUF magic and version: a tiny file the fake engine is given.
	if err := os.WriteFile(filepath.Join(folder, "coder.gguf"), append([]byte("GGUF"), 3, 0, 0, 0), 0o644); err != nil {
		t.Fatal(err)
	}
	resolved, err := filepath.EvalSymlinks(folder)
	if err != nil {
		t.Fatal(err)
	}
	key, err := identity.FileKey(resolved)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte("directory-file:" + key))
	root, err := source.NewSetupInventoryRoot(folder, "oa-chat-"+hex.EncodeToString(sum[:12]))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := source.SetupInventoryRootConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(t.TempDir(), "inventoryd.json")
	if err := os.WriteFile(config, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfigs([]string{config})
	if err != nil {
		t.Fatal(err)
	}
	env := Env{GOOS: runtime.GOOS, Home: home, Getenv: func(string) string { return "" }, Config: cfg}
	object := ""
	for _, m := range snapshot.Take(Env{GOOS: runtime.GOOS, Home: home, Getenv: func(string) string { return "" }, Config: cfg,
		IgnoreEnvironmentRoots: true}, []source.Adapter{extra.New()}).Page.Manifests {
		if m.Manifest.Store == root.Store {
			object = m.Manifest.ID
		}
	}
	if object == "" {
		t.Fatal("Setup's source listed no manifest for the model folder's file")
	}
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Models = []Model{{Name: "qwen3.6-35b-a3b", Store: root.Store, Object: object}}
		o.Source = NewLocalStores(env)
	})
	if _, err := h.Ensure(context.Background(), "qwen3.6-35b-a3b"); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if !h.Loaded("qwen3.6-35b-a3b") {
		t.Fatal("the model is not resident")
	}
}
