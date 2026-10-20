package host

import (
	"time"

	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-provider-kit/go/refusal"
)

// EngineNetworkSocket is the admission reason for a socket engine whose
// contained process tree holds a TCP listener or a bound UDP socket on any
// address, or whose sockets could not be read (CONTRACT.md MODELHOST-L7;
// inference CONTRACT.md INF-EL2).
const EngineNetworkSocket = "engine:network-socket"

// ListenerNotLoopback is the reason for a loopback engine, the video kind's
// sd-server, that holds a socket beyond loopback or whose own listener on
// 127.0.0.1 could not be observed (CONTRACT.md MODELHOST-V5; inference
// CONTRACT.md INF-EL2's loopback rule).
const ListenerNotLoopback = "engine:listener-not-loopback"

// EnginePeer is the admission reason for an engine whose socket is served
// by a process outside the engine's contained tree (CONTRACT.md
// MODELHOST-L7; abstraction-identity ID-T8).
const EnginePeer = "engine:peer"

// EngineUnreachable is the admission reason for an engine whose socket or
// port every readiness connect failed to reach for
// enginehttp.UnreachableFor, with an error a starting engine never gives,
// such as a Windows AF_UNIX path in a packaged app's redirected folder
// (CONTRACT.md MODELHOST-L7; provider kit KIT-P5).
const EngineUnreachable = refusal.EngineUnreachable

// ownerSockets reads the operating system's socket owner tables. It is
// abstraction-identity's reader: GetExtendedTcpTable and GetExtendedUdpTable
// on Windows, IPv4 and IPv6. Other platforms have no reader there, and the
// check below refuses.
var ownerSockets = identity.Sockets

// ownerAllSockets reads every TCP socket in any state and every bound UDP
// socket with its owner, for the rule that a socket engine holds no network
// socket at all.
var ownerAllSockets = identity.AllSockets

// socketWatch is how often a running engine's sockets are checked again
// after readiness (MODELHOST-L7); New reads it once per host.
var socketWatch = 30 * time.Second

// engineSockets is the one check of an engine's sockets for both launch
// paths, the chat engine on its socket and the video engine on its loopback
// port, at readiness and every socketWatch while it runs: the provider
// kit's INF-EL2 rule for the engine's transport, with membership in the
// contained tree proved by process id and creation time through the
// supervised job.
var engineSockets = func(p *Process) error {
	return enginelaunch.CheckEngineSockets(ownerSockets, ownerAllSockets, p.contains, p.port)
}

// networkWord is the refusal reason for p's failed socket check.
func networkWord(p *Process) string {
	if p.port != 0 {
		return ListenerNotLoopback
	}
	return EngineNetworkSocket
}
