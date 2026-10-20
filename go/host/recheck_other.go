//go:build !windows

package host

import "time"

// userIdleTime has no session input source here; the host's own quiet
// (nothing loading, no video engine) decides alone.
func userIdleTime() time.Duration { return time.Duration(1<<63 - 1) }

// stillBeingWritten is false outside Windows, where an open for writing
// blocks no reader.
func stillBeingWritten(error) bool { return false }
