package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go-core/bootstrap"
	facade "github.com/openabstractions/abstraction-facade/go-core/go/abstraction/facade"
	"github.com/openabstractions/abstraction-facade/go-core/resolution"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	inference "github.com/openabstractions/abstraction-inference/go"
	logging "github.com/openabstractions/abstraction-logging/go"
	"github.com/openabstractions/abstraction-logging/go/service"
)

func TestCacheObservationRecordIsNarrowAndSingle(t *testing.T) {
	var output bytes.Buffer
	record := cacheObservationRecord(slog.New(slog.NewJSONHandler(&output, nil)))
	evictions, prefill := int64(1), int64(40)
	read, reused := int64(31245), int64(0)
	engineClaim := 999.0
	record(inference.Record{Operation: "operation-1", Program: "SECRET PROGRAM", Model: "SECRET MODEL", Claim: "SECRET EXTENSION", Reason: "SECRET REASON", Timing: &inference.RequestTiming{Restore: "miss", RestoreCause: "not_resident", CacheEvictions: &evictions, PrefillMS: &prefill, PrefillTokens: &read, ReusedTokens: &reused, Engine: inference.EngineTiming{PromptMS: &engineClaim}}})
	if strings.Count(output.String(), "\n") != 1 || strings.Contains(output.String(), "SECRET") {
		t.Fatal(output.String())
	}
	var value map[string]any
	if err := json.Unmarshal(output.Bytes(), &value); err != nil {
		t.Fatal(err)
	}
	if value["slot"] != "changed" || value["slot_cause"] != "not_resident" || value["cache_evictions"] != float64(1) || value["prefill_ms"] != float64(40) || value["engine_prompt_ms"] != float64(999) ||
		value["prefill_tokens"] != float64(31245) || value["reused_tokens"] != float64(0) || value["reuse"] != "none" {
		t.Fatal(value)
	}
	// The kept slot of an edited conversation reads the whole prompt again:
	// the line says the slot was kept and the engine reused nothing.
	for _, tc := range []struct {
		read, reused int64
		reuse        string
	}{{31245, 0, "none"}, {2788, 15104, "most"}, {20000, 1000, "some"}} {
		output.Reset()
		record(inference.Record{Operation: "operation-3", Timing: &inference.RequestTiming{Restore: "hit", RestoreCause: "resident", PrefillTokens: &tc.read, ReusedTokens: &tc.reused}})
		value = nil
		if err := json.Unmarshal(output.Bytes(), &value); err != nil {
			t.Fatal(err)
		}
		if value["slot"] != "kept" || value["slot_cause"] != "resident" || value["reuse"] != tc.reuse || value["restore"] != nil {
			t.Fatal(value)
		}
	}
	before := output.Len()
	record(inference.Record{Operation: "invalid\ncorrelation", Timing: &inference.RequestTiming{Restore: "hit", RestoreCause: "resident"}})
	record(inference.Record{Operation: "operation-2"})
	if output.Len() != before {
		t.Fatal("invalid or missing observations logged")
	}
}

type providerLogRecords chan logging.Record

func (r providerLogRecords) Write(record logging.Record) error { r <- record; return nil }

// The child exercises the same private provider composition using an inert
// selected runtime with independent native executable/principal evidence.
// Production continues to pass logging.Resolve, whose owner selects installed
// registration; this fixture never reads or changes installed registration.
func TestResolvedDiagnosticsChild(t *testing.T) {
	if os.Getenv("OA_PROVIDER_LOG_CHILD") != "1" {
		return
	}
	resolve := func(ctx context.Context, program string) (*logging.Handler, error) {
		endpoint := os.Getenv("OA_PROVIDER_LOG_ENDPOINT")
		if endpoint == "" {
			return nil, bootstrap.ErrNoTrustedInstallation
		}
		exe, err := os.Executable()
		if err != nil {
			return nil, err
		}
		selected, err := bootstrap.SelectDevelopment(endpoint, exe)
		if err != nil {
			return nil, err
		}
		return logging.ResolveVerified(ctx, program, selected.Endpoint, selected.Server)
	}
	var handler slog.Handler
	func() {
		var closeDiagnostics func()
		handler, closeDiagnostics = logging.ResolveDiagnostics("modelhostd", os.Stderr, resolve)
		defer closeDiagnostics()
		slog.New(handler).Info("provider diagnostics fixture")
	}()
	if _, ok := handler.(*logging.Handler); ok {
		if err := handler.Handle(context.Background(), slog.NewRecord(time.Now(), slog.LevelInfo, "after teardown", 0)); !errors.Is(err, logging.ErrSinkClosed) {
			t.Fatalf("owner teardown left async handler open: %v", err)
		}
	}
}

func TestProviderDiagnosticsThroughRuntimeAndStderr(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("native Program-bound local transport is not implemented on Darwin")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	logEndpoint := listen.Endpoint(fmt.Sprintf("provider-log-%d", os.Getpid()))
	records := make(providerLogRecords, 4)
	sink, err := service.Listen(logEndpoint, records)
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()
	sinkDone := make(chan error, 1)
	go func() { sinkDone <- sink.Serve(ctx) }()
	catalog, err := resolution.New([]resolution.Candidate{{Ready: true, Reference: facade.ServiceReference{
		Provider: "logging-fixture", Capability: "abstraction.logging", Contract: "abstraction.logging/sink@1",
		Scope: facade.ScopeLocal, Transport: resolution.LocalTransport, Endpoint: logEndpoint, Guarantees: []string{},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	endpoint := listen.Endpoint(fmt.Sprintf("provider-resolver-%d", os.Getpid()))
	resolver, err := resolution.Listen(endpoint, catalog, func(*identity.Peer, facade.ServiceReference) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	defer resolver.Close()
	resolverDone := make(chan error, 1)
	go func() { resolverDone <- resolver.Serve(ctx) }()
	t.Cleanup(func() {
		cancel()
		sink.Close()
		resolver.Close()
		<-sinkDone
		<-resolverDone
	})
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, runtimeEndpoint := range []string{endpoint, ""} {
		cmd := exec.Command(exe, "-test.run=^TestResolvedDiagnosticsChild$")
		cmd.Env = append(os.Environ(), "OA_PROVIDER_LOG_CHILD=1", "OA_PROVIDER_LOG_ENDPOINT="+runtimeEndpoint)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		if output, err := cmd.Output(); err != nil {
			t.Fatalf("provider fixture: %v %s %s", err, output, stderr.String())
		}
		if runtimeEndpoint == "" {
			if !strings.Contains(stderr.String(), "provider diagnostics fixture") {
				t.Fatalf("absent runtime lost stderr diagnostics: %q", stderr.String())
			}
			select {
			case record := <-records:
				t.Fatalf("absent runtime unexpectedly wrote through service: %+v", record)
			default:
			}
			continue
		}
		if strings.Contains(stderr.String(), "provider diagnostics fixture") {
			t.Fatalf("adopted record fell back to stderr: %q", stderr.String())
		}
		select {
		case record := <-records:
			if record.Msg != "provider diagnostics fixture" || len(record.Identity) != 2 ||
				record.Identity[0].Program != "modelhostd" || !record.Identity[1].Verified || record.Identity[1].Exe == "" {
				t.Fatalf("native service record: %+v", record)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("provider record did not reach logging service")
		}
	}
}
