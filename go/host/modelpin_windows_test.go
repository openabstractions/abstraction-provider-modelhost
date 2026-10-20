//go:build windows

package host

import (
	"context"
	"os"
	"testing"
)

func TestPinnedGGUFIsHeldUntilConfirmedExit(t *testing.T) {
	var path string
	h := composed(t, newFakeLeases(), func(options *Options) {
		path = fixtureModelPin(t, options)
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	if file, err := os.OpenFile(path, os.O_WRONLY, 0); err == nil {
		file.Close()
		t.Fatal("running engine's pinned model allowed a writer")
	}
	if err := os.Rename(path, path+".moved"); err == nil {
		t.Fatal("running engine's pinned model allowed rename")
	}
	h.Close()
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatalf("confirmed exit retained the model hold: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
