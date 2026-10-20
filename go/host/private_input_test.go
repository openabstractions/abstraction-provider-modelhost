package host

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// privateInput names cross-repository input used by the private workspace gate.
// The published module retains its existing absent-input skip; the source gate
// runs this check when either the runtime row or its provider changes.
func privateInput(t *testing.T, parts ...string) string {
	t.Helper()
	path := filepath.Join(parts...)
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("%s is private-workspace input this repository does not publish; the private gate runs this check", filepath.ToSlash(path))
		}
		t.Fatal(err)
	}
	return path
}
