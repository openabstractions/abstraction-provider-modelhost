package service

import (
	"os"
	"testing"

	"github.com/openabstractions/abstraction-facade/go/programfiles/statetest"
	"github.com/openabstractions/abstraction-provider-modelhost/go/internal/personfolders"
)

// TestMain runs this package's tests with the person's per-user folders
// inside a temporary folder (personfolders.Main), below a temporary root
// ACL'd as a provider's state folder is (statetest), so check records prove
// files across provider processes as they do in an installed provider.
func TestMain(m *testing.M) {
	done := statetest.ProtectTemp()
	code := personfolders.Main(m)
	done()
	os.Exit(code)
}
