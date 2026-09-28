package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

type failedWriter struct{}

func (failedWriter) Write([]byte) (int, error) { return 0, errors.New("output unavailable") }

func TestHelpOutputFailureIsNotSuccess(t *testing.T) {
	if status := run([]string{"help"}, failedWriter{}, &bytes.Buffer{}, func(string) string { return "" }); status == 0 {
		t.Fatal("failed help output reported success")
	}
}

func TestRuntimeDesignationComesFromProviderInstallation(t *testing.T) {
	dir := t.TempDir()
	runtimeName := "openabstractions"
	if runtime.GOOS == "windows" {
		runtimeName += ".exe"
	}
	installed := filepath.Join(dir, runtimeName)
	if err := os.WriteFile(installed, []byte("runtime"), 0o600); err != nil {
		t.Fatal(err)
	}
	programs, err := runtimeProgramsAt(filepath.Join(dir, "modelhostd"), nil)
	if err != nil || len(programs) != 1 || programs[0] != installed {
		t.Fatalf("installed programs = %v, %v", programs, err)
	}
	if runtime.GOOS == "windows" {
		windowless := filepath.Join(dir, "openabstractionsw.exe")
		if err := os.WriteFile(windowless, []byte("runtime"), 0o600); err != nil {
			t.Fatal(err)
		}
		programs, err = runtimeProgramsAt(filepath.Join(dir, "modelhostd.exe"), nil)
		if err != nil || len(programs) != 2 || programs[1] != windowless {
			t.Fatalf("windowless runtime = %v, %v", programs, err)
		}
	}
	explicit := filepath.Join(t.TempDir(), "custom-runtime")
	if err := os.WriteFile(explicit, []byte("runtime"), 0o600); err != nil {
		t.Fatal(err)
	}
	programs, err = runtimeProgramsAt(filepath.Join(dir, "modelhostd"), []string{explicit})
	if err != nil || len(programs) != 1 || programs[0] != explicit {
		t.Fatalf("explicit program = %v, %v", programs, err)
	}
}
