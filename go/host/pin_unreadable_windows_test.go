package host

import (
	"context"
	"strings"
	"testing"

	"golang.org/x/sys/windows"
)

// An engine program another process holds open without sharing, as a scanner
// can, is engine:unreadable before the lease, naming the sharing violation;
// it is no claim that the folder differs from the pin. Once the file is free
// the engine loads.
func TestALockedEngineFileIsUnreadableNotUnpinned(t *testing.T) {
	e := copiedEngine(t)
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) { o.Engine = e })
	name, err := windows.UTF16PtrFromString(e.Program)
	if err != nil {
		t.Fatal(err)
	}
	lock, err := windows.CreateFile(name, windows.GENERIC_READ, 0, nil, windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = h.Ensure(context.Background(), "tiny:1b")
	windows.CloseHandle(lock)
	if WordOf(err) != EngineUnreadable || !strings.Contains(err.Error(), "being used by another process") {
		t.Fatalf("a locked engine program: %v, wanted %s with the sharing violation", err, EngineUnreadable)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 {
		t.Fatalf("a locked engine took %d leases", acquires)
	}
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the freed engine: %v", err)
	}
}
