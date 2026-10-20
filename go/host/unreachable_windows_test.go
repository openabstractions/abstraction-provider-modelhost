package host

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
)

// A socket path the client can never connect to ends the load within
// enginehttp.UnreachableFor, refused engine:unreachable with the system's
// error, and the engine is stopped and its lease given back. The stand-in
// makes the path a folder, which a Windows AF_UNIX connect answers with
// WSAEINVAL, the error a packaged app's redirected socket folder gave the
// 2026-10-03 live run that waited out its whole ten-minute budget.
func TestAnUnreachableEngineSocketFailsFast(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-no-listen"}
		o.Engine.Ready = time.Minute
		o.Engine.beforeLaunch = func(d *enginelaunch.SocketDir) {
			if err := os.Mkdir(d.Socket, 0o700); err != nil {
				t.Errorf("the folder in the socket's place: %v", err)
			}
		}
	})
	began := time.Now()
	_, err := h.Ensure(context.Background(), "tiny:1b")
	took := time.Since(began)
	if WordOf(err) != EngineUnreachable || !strings.Contains(err.Error(), "invalid argument") {
		t.Fatalf("got %v, want %s naming the connect's invalid argument", err, EngineUnreachable)
	}
	if took > enginehttp.UnreachableFor+5*time.Second {
		t.Fatalf("the load took %s, want about %s", took, enginehttp.UnreachableFor)
	}
	if h.Loaded("tiny:1b") {
		t.Fatal("the engine stayed resident")
	}
	if acquires, released, _ := leases.counts(); acquires != 1 || len(released) != 1 {
		t.Fatalf("acquired %d, released %v; wanted the one lease given back", acquires, released)
	}
}

// The video kind's probe reports the same: port 0 can never be connected to
// (WSAEADDRNOTAVAIL), and the probe returns enginehttp.ErrEngineUnreachable for
// the kit's wait to end on.
func TestTheVideoProbeReportsAnUnreachablePort(t *testing.T) {
	base := "http://127.0.0.1:0"
	client, err := enginehttp.NewClientWithVerifier(base, func(identity.Process) (bool, error) { return true, nil })
	if err != nil {
		t.Fatal(err)
	}
	ready, err := videoReady(base)(context.Background(), client)
	if ready || !errors.Is(err, enginehttp.ErrEngineUnreachable) {
		t.Fatalf("ready %v, err %v; want enginehttp.ErrEngineUnreachable", ready, err)
	}
}
