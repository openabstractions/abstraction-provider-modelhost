//go:build !windows

package host

// abstraction-identity reads socket owner tables on Windows only, so the
// provider refuses every engine elsewhere at readiness (MODELHOST-L7), and no
// release is pinned there either. These tests measure the lease, the load and
// the unload with the stand-in engines, so on these platforms they replace the
// owner-table check; the Windows tests run the real one. The socket's peer
// check runs on every platform.
func init() { engineSockets = func(*Process) error { return nil } }
