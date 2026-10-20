package host

import (
	"archive/zip"
	"context"
	"fmt"

	"github.com/openabstractions/abstraction-inference/go/providerhost/modelfiles"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
)

// memoryExtent is the extent the provider kit fills a section from, for an
// object the source bound as a file, a region or a DEFLATE member
// (inference CONTRACT.md INF-MS1; modelbridge CONTRACT.md MODELBRIDGE-L19).
func memoryExtent(b bridge.ObjectBinding) (modelfiles.Extent, error) {
	switch {
	case b.Kind == bridge.BindFile:
		return modelfiles.Extent{Path: b.Path, Packed: b.Length, Length: b.Length, Method: zip.Store}, nil
	case b.Kind == bridge.BindRegion:
		return modelfiles.Extent{Path: b.Path, Offset: b.Offset, Packed: b.Length, Length: b.Length, Method: zip.Store}, nil
	case b.Kind == bridge.BindCompressed && b.Method == "deflate":
		return modelfiles.Extent{Path: b.Path, Offset: b.Offset, Packed: b.Packed, Length: b.Length, Method: zip.Deflate, CRC32: b.CRC32}, nil
	}
	return modelfiles.Extent{}, fmt.Errorf("a %s binding (method %q) fills no memory section", b.Kind, b.Method)
}

// fillMemory places the object b binds in a read-only section proved against
// digest, and answers the section with the memory binding that describes it
// (MODELHOST-L1). release ends the resource holding for the section's memory
// and is called exactly once: when the section is freed, or on a refusal.
// The model host's pinned engine reads files only, so no load calls this
// yet; an engine that reads an inherited section takes a Loan of it at
// launch through supervision's Options.Inherit.
func fillMemory(ctx context.Context, b bridge.ObjectBinding, digest string, release func()) (*modelfiles.Section, bridge.ObjectBinding, error) {
	extent, err := memoryExtent(b)
	if err != nil {
		if release != nil {
			release()
		}
		return nil, bridge.ObjectBinding{}, err
	}
	section, err := modelfiles.Fill(ctx, extent, digest, release)
	if err != nil {
		return nil, bridge.ObjectBinding{}, err
	}
	from := b
	return section, bridge.ObjectBinding{Kind: bridge.BindMemory, Length: section.Length(), Handle: section.Handle(),
		Digest: section.Digest(), ChunkDigests: section.ChunkDigests(), From: &from}, nil
}
