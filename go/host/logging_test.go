package host

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go-core/bootstrap"
	facade "github.com/openabstractions/abstraction-facade/go-core/go/abstraction/facade"
	"github.com/openabstractions/abstraction-facade/go-core/resolution"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/listen"
	logging "github.com/openabstractions/abstraction-logging/go"
	"github.com/openabstractions/abstraction-logging/go/service"
)

type hostLogRecords chan logging.Record

func (r hostLogRecords) Write(record logging.Record) error { r <- record; return nil }

func TestHostOwnsAdoptedDiagnosticsThroughCloseAndFailedConstruction(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("native Program-bound local transport is not implemented on Darwin")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	records := make(hostLogRecords, 8)
	logEndpoint := listen.Endpoint(fmt.Sprintf("host-log-%d", os.Getpid()))
	sink, err := service.Listen(logEndpoint, records)
	if err != nil {
		t.Fatal(err)
	}
	sinkDone := make(chan error, 1)
	go func() { sinkDone <- sink.Serve(ctx) }()
	catalog, err := resolution.New([]resolution.Candidate{{Ready: true, Reference: facade.ServiceReference{
		Provider: "logging-fixture", Capability: "abstraction.logging", Contract: "abstraction.logging/sink@1",
		Scope: facade.ScopeLocal, Transport: resolution.LocalTransport, Endpoint: logEndpoint, Guarantees: []string{},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	endpoint := listen.Endpoint(fmt.Sprintf("host-log-resolver-%d", os.Getpid()))
	resolver, err := resolution.Listen(endpoint, catalog, func(*identity.Peer, facade.ServiceReference) bool { return true })
	if err != nil {
		sink.Close()
		t.Fatal(err)
	}
	resolverDone := make(chan error, 1)
	go func() { resolverDone <- resolver.Serve(ctx) }()
	t.Cleanup(func() { cancel(); sink.Close(); resolver.Close(); <-sinkDone; <-resolverDone })
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	selected, err := bootstrap.SelectDevelopment(endpoint, exe)
	if err != nil {
		t.Fatal(err)
	}
	for iteration := 0; iteration < 4; iteration++ {
		var adopted *logging.Handler
		closes := 0
		open := func(program string) (slog.Handler, func()) {
			handler, closeDiagnostics := logging.ResolveDiagnostics(program, io.Discard, func(ctx context.Context, program string) (*logging.Handler, error) {
				var err error
				adopted, err = logging.ResolveVerified(ctx, program, selected.Endpoint, selected.Server)
				return adopted, err
			})
			return handler, func() { closes++; closeDiagnostics() }
		}
		model := Model{Name: "m", Store: "store", Object: "object"}
		if iteration == 3 {
			model.Object = ""
		}
		h, err := newHost(Options{Models: []Model{model}, Source: fakeSource{}, Leases: newFakeLeases()}, open)
		if iteration == 3 {
			if err == nil || h != nil {
				t.Fatal("malformed model unexpectedly constructed")
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			h.loadLogger.Info("owned host diagnostics")
			h.Close()
			select {
			case record := <-records:
				if record.Msg != "owned host diagnostics" {
					t.Fatalf("record %+v", record)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("owner close did not drain accepted record")
			}
		}
		if adopted == nil || closes != 1 {
			t.Fatalf("lifetime %d: adopted=%v closes=%d", iteration, adopted != nil, closes)
		}
		if err := adopted.Handle(context.Background(), slog.NewRecord(time.Now(), slog.LevelInfo, "after teardown", 0)); !errors.Is(err, logging.ErrSinkClosed) {
			t.Fatalf("lifetime %d left async handler open: %v", iteration, err)
		}
		// Close returns only after the admitted queue's worker has joined.
		closed, cancelClosed := context.WithTimeout(context.Background(), time.Second)
		if _, err := adopted.Close(closed); err != nil {
			t.Fatalf("lifetime %d worker not joined: %v", iteration, err)
		}
		cancelClosed()
	}
}
