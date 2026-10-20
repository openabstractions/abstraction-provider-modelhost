//go:build !windows

package host

import "os"

func holdEngineArchive(path string) (*os.File, error) { return os.Open(path) }
