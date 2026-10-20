// Package host loads model files with an engine this machine already has, one
// engine process per resident object, holds each one's own residency under
// its own lease of abstraction.resource/leases@1, serves it for as long as
// calls arrive, and gives it back when its lease is asked for or nothing has
// used it for its own idle time.
//
// It bundles no engine and downloads nothing. The engine is a path a person
// configured; a model file is an object the storage inventory names.
package host

import (
	"github.com/openabstractions/abstraction-provider-kit/go/modelplace"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// A Source resolves the inventory's naming — a store and an id — to a binding
// on this machine. It is the seam abstraction.storage/lend@1 resolves through
// (abstraction-provider-modelbridge CONTRACT.md MODELBRIDGE-L1, MODELBRIDGE-L19),
// reused here so this provider's contract carries no path either; this
// provider asks it for a whole file.
type Source = bridge.Source

// An Object is one weight file as that seam reports it: its store, its id,
// its binding with the absolute path of a file, and its size.
type Object = bridge.Object

// Env is the machine one Source reads.
type Env = source.Env

// NewLocalStores reads the stores abstraction-storage-over-local-stores reads,
// through its own adapters. It hashes nothing and writes nothing.
func NewLocalStores(env Env) Source { return bridge.NewLocalStores(env) }

// StoreFiles is src as the provider kit's store reader for a models install
// (KIT-M1): the kit names no store provider, and the model host reads stores
// through the model bridge (its recorded coupling, architecture review
// 2026-10-02 D8). A nil src reads no store.
func StoreFiles(src Source) modelplace.StoreFiles {
	if src == nil {
		return nil
	}
	return bridgeStores{src}
}

// bridgeStores answers a store search with a whole stored file's path, the
// file binding of bridge.Source.Resolve.
type bridgeStores struct{ src Source }

func (b bridgeStores) Stores() []string { return b.src.Stores() }

func (b bridgeStores) FilePath(store, id string) (string, bool) {
	o, err := b.src.Resolve(store, id, bridge.FileOnly)
	if err != nil || o.Binding.Kind != bridge.BindFile || o.Path == "" {
		return "", false
	}
	return o.Path, true
}

// LoadConfig reads the same operator store configuration the inventory source
// and the model bridge read.
func LoadConfig(path string) (source.Config, error) { return source.LoadConfig(path) }
