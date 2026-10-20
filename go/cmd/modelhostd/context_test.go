package main

import (
	"bytes"
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	configwire "github.com/openabstractions/abstraction-config/go/abstraction/config"
	"github.com/openabstractions/abstraction-facade/go/abstraction/facade/registration"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	resource "github.com/openabstractions/abstraction-resource/go/client"
	"github.com/openabstractions/abstraction-resource/go/instrument"
	resourceservice "github.com/openabstractions/abstraction-resource/go/service"
	rights "github.com/openabstractions/abstraction-rights/go/abstraction/rights/api"
)

type fakeTable map[string]int64

func (f fakeTable) HoldersContext(_ context.Context, name string, _ bool) (resource.ResourceState, error) {
	capacity, ok := f[name]
	if !ok {
		return resource.ResourceState{}, errors.New("unreadable")
	}
	return resource.ResourceState{Resource: name, Capacity: capacity}, nil
}

func TestMachineCapacityFollowsPoolRule(t *testing.T) {
	if got := machineCapacity(fakeTable{"card:0": 96 << 30, "memory": 128 << 30}, "card:0")(context.Background()); got != 96<<30 {
		t.Fatalf("card capacity %d", got)
	}
	// The table's figure is the pool; the model host adds no rule of its own.
	if got := machineCapacity(fakeTable{"card:0": 0, "memory": 128 << 30}, "card:0")(context.Background()); got != 0 {
		t.Fatalf("carved card capacity %d", got)
	}
	if got := machineCapacity(fakeTable{}, "card:0")(context.Background()); got != 0 {
		t.Fatalf("unknown capacity %d", got)
	}
}

// The owner's Strix Halo (R27): the table reported card:0's capacity as
// unknown, so every load read an unknown pool and Automatic stayed at the
// catalogue's 32768 in two slots while the runtime told OpenCode the 128 GB
// plan. The table now reports the card's pool, DXGI's 102.7 GiB there, and
// the model host reads that figure: Automatic takes the trained context.
func TestMachineCapacityOnUnifiedMemoryReadsTheTable(t *testing.T) {
	fixed := instrument.NewFixed()
	fixed.Set(instrument.Card0, instrument.FixedResource{Pool: instrument.Memory, Capacity: 110_324_465_664})
	capacity := machineCapacity(tableReader{resourceservice.NewTable(fixed, nil, 0)}, instrument.Card0)(context.Background())
	if capacity != 110_324_465_664 {
		t.Fatalf("unified-memory pool %d, want the adapter's 102.75 GiB", capacity)
	}
	plan, ok := modelservice.PlanContext("qwen3.6-35b-a3b", capacity, 0)
	if !ok || plan.Automatic != 262144 || plan.Context != plan.Automatic {
		t.Fatalf("Automatic on the Strix Halo: %+v", plan)
	}
	// A table that does not answer stays unknown.
	if got := machineCapacity(fakeTable{}, "card:0")(context.Background()); got != 0 {
		t.Fatalf("unanswered table read as %d", got)
	}
}

// tableReader reads a real resource table the way the runtime's resource
// service answers HoldersContext.
type tableReader struct{ table *resourceservice.Table }

func (r tableReader) HoldersContext(_ context.Context, name string, fresh bool) (resource.ResourceState, error) {
	return r.table.Holders(name, fresh)
}

// The runtime reports a hosted model's context from Table.Pool (serve's
// tablePool) and the model host loads it from machineCapacity: one pool, so
// the context OpenCode is told is the context each conversation is loaded
// with, on a shared-memory machine and on a discrete card.
func TestMachineCapacityEqualsTheRuntimePool(t *testing.T) {
	for _, c := range []struct {
		name   string
		card   instrument.FixedResource
		memory instrument.FixedResource
	}{
		{"unified memory, no capacities", instrument.FixedResource{Pool: instrument.Memory}, instrument.FixedResource{}},
		{"unified memory with memory capacity", instrument.FixedResource{Pool: instrument.Memory}, instrument.FixedResource{Capacity: 96 << 30}},
		{"discrete card", instrument.FixedResource{Capacity: 24 << 30}, instrument.FixedResource{Capacity: 64 << 30}},
	} {
		fixed := instrument.NewFixed()
		fixed.Set(instrument.Card0, c.card)
		fixed.Set(instrument.Memory, c.memory)
		table := resourceservice.NewTable(fixed, nil, 0)
		loaded := machineCapacity(tableReader{table}, instrument.Card0)(context.Background())
		reported := table.Pool(instrument.Card0)
		if loaded != reported || loaded <= 0 {
			t.Fatalf("%s: model host pool %d, runtime pool %d", c.name, loaded, reported)
		}
		load, _ := modelservice.PlanContext("qwen3.6-35b-a3b", loaded, 0)
		report, _ := modelservice.PlanContext("qwen3.6-35b-a3b", reported, 0)
		if load.Context != report.Context || load.Slots != report.Slots {
			t.Fatalf("%s: loaded %+v, reported %+v", c.name, load, report)
		}
	}
}

type fakeConfig struct {
	result configwire.ReadResult
	err    error
}

func (f fakeConfig) ReadTypedContext(context.Context, []string) (configwire.ReadResult, error) {
	return f.result, f.err
}

func contextRead(values map[string]string) configwire.ReadResult {
	return configwire.ReadResult{Outcome: configwire.ReadOutcomeRead, Entries: []configwire.Entry{
		{Key: modelContextKey, Value: &configwire.Value{Kind: configwire.ValueKindTextMap, TextMap: values}}}}
}

func TestContextChoiceReadsModelContext(t *testing.T) {
	ctx := context.Background()
	var logged []error
	logf := func(err error) { logged = append(logged, err) }
	if got := contextChoice(fakeConfig{result: contextRead(map[string]string{"qwen3.6-35b-a3b": "131072"})}, logf)(ctx, "qwen3.6-35b-a3b"); got != 131072 || len(logged) != 0 {
		t.Fatalf("choice %d, logged %v", got, logged)
	}
	// Automatic is quiet.
	if got := contextChoice(fakeConfig{result: contextRead(map[string]string{"other": "65536"})}, logf)(ctx, "qwen3.6-35b-a3b"); got != 0 || len(logged) != 0 {
		t.Fatalf("automatic %d, logged %v", got, logged)
	}
	// A refused, failed or unreadable read is logged, never silent.
	refusal := &rights.RefusalDetail{Source: rights.RefusalSourceRights, Decision: rights.RefusalDecisionNotGranted, Action: "abstraction.config/key.read"}
	for name, config := range map[string]fakeConfig{
		"refused-key": {result: configwire.ReadResult{Outcome: configwire.ReadOutcomeRead,
			Refused: []configwire.KeyRefusal{{Key: modelContextKey, Reason: configwire.KeyRefusalReasonNotGranted, Refusal: refusal}}}},
		"forbidden": {result: configwire.ReadResult{Outcome: configwire.ReadOutcomeForbidden}},
		"failed":    {err: errors.New("no service")},
		"invalid":   {result: contextRead(map[string]string{"qwen3.6-35b-a3b": "lots"})},
	} {
		logged = nil
		if got := contextChoice(config, logf)(ctx, "qwen3.6-35b-a3b"); got != 0 || len(logged) != 1 {
			t.Fatalf("%s: choice %d, logged %v", name, got, logged)
		}
		if name == "refused-key" && !strings.Contains(logged[0].Error(), "refused") {
			t.Fatalf("refusal log %v", logged[0])
		}
	}
}

// fakeObserver answers each Observe from its script, then waits for ctx.
type fakeObserver struct {
	mu      sync.Mutex
	answers []configwire.ConfigObservationV2
	cursors []string
}

func (f *fakeObserver) ObserveTypedContext(ctx context.Context, cursor string, _ int64) (configwire.ConfigObservationV2, error) {
	f.mu.Lock()
	f.cursors = append(f.cursors, cursor)
	if len(f.answers) > 0 {
		next := f.answers[0]
		f.answers = f.answers[1:]
		f.mu.Unlock()
		return next, nil
	}
	f.mu.Unlock()
	<-ctx.Done()
	return configwire.ConfigObservationV2{}, ctx.Err()
}

func TestWatchModelContextReportsChangedFamilies(t *testing.T) {
	snapshot := func(values map[string]string) *configwire.ReadResult { r := contextRead(values); return &r }
	observer := &fakeObserver{answers: []configwire.ConfigObservationV2{
		{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "1", Snapshot: snapshot(map[string]string{"a": "65536", "b": "32768"})},
		{Outcome: configwire.ObservationOutcomeV2Unchanged, Cursor: "1"},
		{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "2", Snapshot: snapshot(map[string]string{"a": "131072", "b": "32768", "c": "65536"})},
		{Outcome: configwire.ObservationOutcomeV2Changed, Cursor: "3", Snapshot: snapshot(map[string]string{"a": "131072", "c": "65536"})},
	}}
	ctx, cancel := context.WithCancel(context.Background())
	var mu sync.Mutex
	var changed []string
	done := make(chan struct{})
	go func() {
		defer close(done)
		watchModelContext(ctx, observer, func(family string) { mu.Lock(); changed = append(changed, family); mu.Unlock() }, func(err error) { t.Error(err) })
	}()
	deadline := time.Now().Add(5 * time.Second)
	for {
		mu.Lock()
		n := len(changed)
		mu.Unlock()
		if n >= 3 || time.Now().After(deadline) {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	<-done
	if want := []string{"a", "c", "b"}; !slices.Equal(changed, want) {
		t.Fatalf("changed %v, want %v (the baseline reports none)", changed, want)
	}
	if want := []string{"", "1", "1", "2", "3"}; !slices.Equal(observer.cursors, want) {
		t.Fatalf("cursors %v", observer.cursors)
	}
}

func TestWatchModelContextLogsARefusal(t *testing.T) {
	saved := contextWatchBackoff
	contextWatchBackoff = time.Millisecond
	t.Cleanup(func() { contextWatchBackoff = saved })
	observer := &fakeObserver{answers: []configwire.ConfigObservationV2{{Outcome: configwire.ObservationOutcomeV2NotPermitted,
		Refusal: &rights.RefusalDetail{Source: rights.RefusalSourceRights, Decision: rights.RefusalDecisionNotGranted, Action: "abstraction.config/key.read"}}}}
	ctx, cancel := context.WithCancel(context.Background())
	logged := make(chan error, 4)
	done := make(chan struct{})
	go func() {
		defer close(done)
		watchModelContext(ctx, observer, func(string) { t.Error("refusal reported a change") }, func(err error) { logged <- err })
	}()
	select {
	case err := <-logged:
		if !strings.Contains(err.Error(), "not_permitted") {
			t.Fatalf("refusal log %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("refusal was not logged")
	}
	cancel()
	<-done
}

// The model host's own registration names model_context, the key its loads
// read, so the runtime grants that exact read with its provider facet.
func TestServeDeclarationNamesModelContext(t *testing.T) {
	var out bytes.Buffer
	if err := writeServeDeclaration(&out, []string{"--endpoint", "modelhost"}, "modelhost", "engine", false, "card:0", []string{"qwen=fake/tiny"}, ""); err != nil {
		t.Fatal(err)
	}
	spec, err := registration.DecodeRegistrationSpec(out.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Providers) != 1 || !slices.Equal(spec.Providers[0].ConfigKeys, []string{modelContextKey}) {
		t.Fatalf("provider facets %+v", spec.Providers)
	}
}
