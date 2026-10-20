package host

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-provider-kit/go/enginelaunch"
)

func fixtureModelPin(t *testing.T, options *Options) string {
	t.Helper()
	object := options.Source.(fakeSource).objects["fake/tiny"]
	raw, err := os.ReadFile(object.Path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	options.Models[0].Content = &programfiles.File{Size: object.Size, SHA256: hex.EncodeToString(sum[:])}
	return object.Path
}

func TestPinnedGGUFRejectsTamperedBytesBeforeLaunch(t *testing.T) {
	launched := false
	leases := newFakeLeases()
	h := composed(t, leases, func(options *Options) {
		path := fixtureModelPin(t, options)
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		raw[len(raw)-1] ^= 1
		if err := os.WriteFile(path, raw, 0600); err != nil {
			t.Fatal(err)
		}
		options.Engine.beforeLaunch = func(*enginelaunch.SocketDir) { launched = true }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != Unavailable {
		t.Fatalf("tampered model: %v", err)
	}
	if launched {
		t.Fatal("tampered model reached launch")
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("failed proof released %v leases", released)
	}
}

func TestUnpinnedGGUFKeepsLegacyLaunchWithoutContentRecord(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), nil)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(h.options.Engine.Record), "models")); !os.IsNotExist(err) {
		t.Fatalf("unpinned launch created model records: %v", err)
	}
}

func TestPinnedGGUFSizeMismatchRefusesBeforeLease(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(options *Options) {
		fixtureModelPin(t, options)
		options.Models[0].Content.Size++
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != Unavailable {
		t.Fatalf("size mismatch: %v", err)
	}
	if acquired, _, _ := leases.counts(); acquired != 0 {
		t.Fatalf("mismatched size acquired %d leases", acquired)
	}
}
