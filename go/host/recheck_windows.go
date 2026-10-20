package host

import (
	"errors"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	user32           = windows.NewLazySystemDLL("user32.dll")
	getLastInputInfo = user32.NewProc("GetLastInputInfo")
	getTickCount64   = windows.NewLazySystemDLL("kernel32.dll").NewProc("GetTickCount64")
)

// lastInputInfo is LASTINPUTINFO.
type lastInputInfo struct {
	size uint32
	time uint32
}

// userIdleTime is how long the session has had no keyboard or mouse
// input; a session without input reads as idle.
func userIdleTime() time.Duration {
	info := lastInputInfo{size: uint32(unsafe.Sizeof(lastInputInfo{}))}
	//unchecked: the BOOL primary return is checked below
	ok, _, _ := getLastInputInfo.Call(uintptr(unsafe.Pointer(&info)))
	if ok == 0 {
		return time.Duration(1<<63 - 1)
	}
	//unchecked: GetTickCount64 cannot fail
	now, _, _ := getTickCount64.Call()
	// LASTINPUTINFO keeps the low 32 bits of the tick count.
	elapsed := uint32(uint64(now)) - info.time
	return time.Duration(elapsed) * time.Millisecond
}

// stillBeingWritten reports a file another program holds open for writing:
// the proof opens it with read sharing only.
func stillBeingWritten(err error) bool {
	return errors.Is(err, windows.ERROR_SHARING_VIOLATION)
}
