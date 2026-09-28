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
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
)

// A Source resolves the inventory's naming — a store and an id — to a file on
// this machine. It is the seam abstraction.storage/lend@1 asks for a path
// through (abstraction-provider-modelbridge CONTRACT.md MODELBRIDGE-L1), reused here
// so this provider's contract carries no path either.
type Source = bridge.Source

// An Object is one weight file as that seam reports it: its store, its id, the
// absolute path on this machine, and its size.
type Object = bridge.Object

// Env is the machine one Source reads.
type Env = source.Env

// NewLocalStores reads the stores abstraction-storage-over-local-stores reads,
// through its own adapters. It hashes nothing and writes nothing.
func NewLocalStores(env Env) Source { return bridge.NewLocalStores(env) }

// LoadConfig reads the same operator store configuration the inventory source
// and the model bridge read.
func LoadConfig(path string) (source.Config, error) { return source.LoadConfig(path) }
