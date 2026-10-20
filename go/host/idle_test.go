package host

import (
	"context"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	res "github.com/openabstractions/abstraction-resource/go/client"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

// idleClock is a host clock the test moves forward.
type idleClock struct{ offset atomic.Int64 }

var idleEpoch = time.Date(2026, 11, 10, 9, 0, 0, 0, time.UTC)

func (c *idleClock) now() time.Time          { return idleEpoch.Add(time.Duration(c.offset.Load())) }
func (c *idleClock) advance(d time.Duration) { c.offset.Add(int64(d)) }
func (c *idleClock) adjust(o *Options) {
	o.Now = c.now
	o.Idle = DefaultIdle
	o.Renew = 20 * time.Millisecond
}
func waitIdleSweeps(t *testing.T, h *Host, name string, loaded bool) bool {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if h.Loaded(name) != loaded {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
	return true
}

// familyHost is a host whose one model is the curated Qwen3.6 35B, whose
// engine profile names a 30-minute idle time, on clock.
func familyHost(t *testing.T, leases *fakeLeases, clock *idleClock) *Host {
	t.Helper()
	requireEnginePeer(t)
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	if entry.Family != contextFamily || engineProfiles[contextFamily].IdleMinutes != 30 {
		t.Fatalf("curated entry %s, profile idle %d minutes", entry.Family, engineProfiles[contextFamily].IdleMinutes)
	}
	return composed(t, leases, func(o *Options) {
		clock.adjust(o)
		o.Engine.cache = nil
		original := o.Source.(fakeSource)
		object := original.objects["fake/tiny"]
		object.Size = entry.DownloadBytes
		original.objects["fake/tiny"] = object
		o.Source = defaultProofSource{Source: original, current: true, proof: content.Object{Locator: object.Locator, Size: object.Size, Modified: "current", Digest: entry.Digest, Evidence: content.EvidenceHashed}}
		o.Capacity = func(context.Context) int64 { return 128 << 30 }
	})
}

// MODELHOST-H8: a family whose engine profile names 30 minutes stays loaded
// past the declared 5 minutes and unloads after its own 30.
func TestFamilyIdleTimeComesFromItsEngineProfile(t *testing.T) {
	leases := newFakeLeases()
	clock := &idleClock{}
	h := familyHost(t, leases, clock)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	clock.advance(6 * time.Minute)
	if !waitIdleSweeps(t, h, "tiny:1b", true) {
		t.Fatal("the 30-minute family unloaded at 6 minutes")
	}
	clock.advance(25 * time.Minute)
	if waitIdleSweeps(t, h, "tiny:1b", true) {
		t.Fatal("the 30-minute family stayed loaded at 31 minutes")
	}
	if _, released, _ := leases.counts(); len(released) != 1 {
		t.Fatalf("released %v, want the resident's lease", released)
	}
}

// A family without an idle time in its profile unloads after --idle.
func TestFamilyWithoutAnIdleTimeUsesTheDeclaredIdle(t *testing.T) {
	requireEnginePeer(t)
	leases := newFakeLeases()
	clock := &idleClock{}
	h := composed(t, leases, clock.adjust)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	clock.advance(4 * time.Minute)
	if !waitIdleSweeps(t, h, "tiny:1b", true) {
		t.Fatal("unloaded before the declared idle time")
	}
	clock.advance(2 * time.Minute)
	if waitIdleSweeps(t, h, "tiny:1b", true) {
		t.Fatal("a family without a profile idle time stayed loaded past --idle")
	}
}

// A longer idle time changes no yield (MODELHOST-H7), and an idle resident
// holds no wake lease: its only lease is the card.
func TestALongIdleResidentStillYieldsAndHoldsNoWakeLease(t *testing.T) {
	leases := newFakeLeases()
	clock := &idleClock{}
	h := familyHost(t, leases, clock)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	lease := h.Lease("tiny:1b")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	clock.advance(10 * time.Minute)
	if !waitIdleSweeps(t, h, "tiny:1b", true) {
		t.Fatal("unloaded within its idle time")
	}
	leases.mu.Lock()
	resources := slices.Clone(leases.resources)
	amount := leases.amounts[lease]
	leases.mu.Unlock()
	if !slices.Equal(resources, []string{"card:0"}) {
		t.Fatalf("an idle resident acquired %v, want the card alone and no wake lease", resources)
	}
	leases.pending <- res.YieldRequest{ID: "picture", Resource: "card:0", Amount: amount, Reason: "a picture job", Lease: lease}
	deadline := time.Now().Add(10 * time.Second)
	for {
		_, released, answers := leases.counts()
		if !h.Loaded("tiny:1b") && len(released) == 1 && slices.Equal(answers, []string{"picture=yielded"}) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("yield during the long idle time: loaded=%v released=%v answers=%v", h.Loaded("tiny:1b"), released, answers)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
