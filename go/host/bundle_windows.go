//go:build windows

package host

import (
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/windows"
)

// holdEngineArchive keeps a ZIP read-only until the kit has finished its
// install, including an idempotent install of an already-proven engine tree.
func holdEngineArchive(path string) (*os.File, error) {
	name := filepath.Clean(path)
	if strings.HasPrefix(name, `\\?\`) {
		// Already an extended path.
	} else if strings.HasPrefix(name, `\\`) {
		name = `\\?\UNC\` + strings.TrimPrefix(name, `\\`)
	} else {
		name = `\\?\` + name
	}
	wide, err := windows.UTF16PtrFromString(name)
	if err != nil {
		return nil, err
	}
	h, err := windows.CreateFile(wide, windows.GENERIC_READ, windows.FILE_SHARE_READ, nil,
		windows.OPEN_EXISTING, windows.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(h), path), nil
}
