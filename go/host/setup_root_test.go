package host_test

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"

	identity "github.com/openabstractions/abstraction-identity"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/adapters/extra"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/snapshot"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// Owner's preview, 2026-11-01: every local chat failed with
// native:object:unknown:oa-chat-…/<id>. Setup's model folder is a store only
// Setup's own inventoryd reads, through its --config extra root
// (--config-roots-only); `models host` found the object through the
// runtime's inventory, but modelhostd read no configuration and its own
// store reading never saw the store. With that source's configuration among
// its --config files, the model host resolves the store and manifest id the
// Setup source listed, beside the stores another configuration names.
func TestModelHostResolvesAnObjectInSetupsModelFolder(t *testing.T) {
	home := t.TempDir()
	folder := filepath.Join(t.TempDir(), "models")
	if err := os.MkdirAll(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	const filename = "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1.gguf"
	const weights = "coder weights"
	file := filepath.Join(folder, filename)
	if err := os.WriteFile(file, []byte(weights), 0o644); err != nil {
		t.Fatal(err)
	}
	// Setup's store name for the folder: oa-chat- and the first 12 bytes of
	// the folder identity's SHA-256 (source.NewSetupInventoryRoot checks it).
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
	setupConfig := filepath.Join(t.TempDir(), "inventoryd.json")
	if err := os.WriteFile(setupConfig, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(t.TempDir(), "other.json")
	if err := os.WriteFile(other, []byte(`{"comfyui":["`+filepath.ToSlash(filepath.Join(home, "ComfyUI"))+`"]}`), 0o600); err != nil {
		t.Fatal(err)
	}

	// What Setup's inventoryd lists: its extra root alone.
	cfg, err := host.LoadConfig(setupConfig)
	if err != nil {
		t.Fatal(err)
	}
	listed := source.Env{GOOS: "test", Home: home, Getenv: func(string) string { return "" }, Config: cfg, IgnoreEnvironmentRoots: true}
	var object string
	for _, m := range snapshot.Take(listed, []source.Adapter{extra.New()}).Page.Manifests {
		if m.Manifest.Store == root.Store {
			object = m.Manifest.ID
		}
	}
	if object == "" {
		t.Fatal("Setup's source listed no manifest for the model folder's file")
	}

	env := func(c source.Config) host.Env {
		return host.Env{GOOS: "test", Home: home, Getenv: func(string) string { return "" }, Config: c}
	}
	// Without the source's configuration the store is unknown: the defect.
	if _, err := host.NewLocalStores(env(source.Config{})).Resolve(root.Store, object, bridge.FileOnly); !errors.Is(err, bridge.ErrUnknownObject) {
		t.Fatalf("resolve without the configuration = %v", err)
	}
	joined, err := host.LoadConfigs([]string{other, setupConfig})
	if err != nil {
		t.Fatal(err)
	}
	if len(joined.ComfyUI) != 1 || len(joined.ExtraRoots) != 1 {
		t.Fatalf("joined configuration %+v", joined)
	}
	got, err := host.NewLocalStores(env(joined)).Resolve(root.Store, object, bridge.FileOnly)
	if err != nil || got.Store != root.Store || got.ID != object || got.Locator != filename ||
		got.Binding.Kind != bridge.BindFile || got.Binding.Length != int64(len(weights)) {
		t.Fatalf("resolve %s/%s = %+v, %v", root.Store, object, got, err)
	}
	for _, path := range []string{got.Path, got.Binding.Path} {
		if same, err := identity.SameFile(path, file); err != nil || !same {
			t.Fatalf("resolved path %q does not name %q: %v", path, file, err)
		}
	}
	if _, err := host.LoadConfigs([]string{setupConfig, filepath.Join(t.TempDir(), "missing.json")}); err == nil {
		t.Fatal("a configuration that cannot be read was skipped")
	}
}
