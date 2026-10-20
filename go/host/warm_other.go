//go:build !windows

package host

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

// systemAvailableMemory is Linux's MemAvailable; 0 elsewhere, where
// warming is skipped.
func systemAvailableMemory() int64 {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0
	}
	//unchecked: a read-only handle
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) >= 2 && fields[0] == "MemAvailable:" {
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil || kb < 0 {
				return 0
			}
			return kb << 10
		}
	}
	return 0
}
