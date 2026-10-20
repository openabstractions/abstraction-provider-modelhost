package host

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

var globalMemoryStatusEx = windows.NewLazySystemDLL("kernel32.dll").NewProc("GlobalMemoryStatusEx")

// memoryStatusEx is MEMORYSTATUSEX; only availPhys is read.
type memoryStatusEx struct {
	length               uint32
	memoryLoad           uint32
	totalPhys            uint64
	availPhys            uint64
	totalPageFile        uint64
	availPageFile        uint64
	totalVirtual         uint64
	availVirtual         uint64
	availExtendedVirtual uint64
}

// systemAvailableMemory is GlobalMemoryStatusEx's available physical
// memory: free, zeroed and standby pages, the file cache included.
func systemAvailableMemory() int64 {
	var status memoryStatusEx
	status.length = uint32(unsafe.Sizeof(status))
	//unchecked: the BOOL primary return is checked below; Call's GetLastError field adds nothing here
	ok, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&status)))
	if ok == 0 || status.availPhys > 1<<62 {
		return 0
	}
	return int64(status.availPhys)
}
