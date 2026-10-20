package host

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginelaunch"
	"github.com/openabstractions/abstraction-inference/go/providerhost/llamacpp"
)

// ErrEngineEnvironment reports an engine environment entry outside the
// allowlist (CONTRACT.md MODELHOST-L6; b10825 review condition 3).
var ErrEngineEnvironment = errors.New("modelhost: engine environment variable outside the allowlist")

// engineEnvironmentNames is the complete engine environment; nothing is
// inherited. On Windows SystemRoot, windir and SystemDrive let Winsock and
// the system libraries load, TEMP and TMP name the provider's private
// folder, PATH names System32 alone, USERPROFILE, LOCALAPPDATA, APPDATA,
// ProgramData and ALLUSERSPROFILE are this account's known folders (driver
// pipeline caches, llama.cpp's cache directory), and NUMBER_OF_PROCESSORS
// and PROCESSOR_ARCHITECTURE come from GetNativeSystemInfo. The Vulkan
// loader needs no variable: vulkan-1.dll is in System32 and the loader finds
// drivers through the registry, so VK_ICD_FILENAMES, VK_DRIVER_FILES,
// VK_LAYER_PATH, VK_INSTANCE_LAYERS and the other loader variables, which
// only add or replace drivers and layers, are dropped. GGML_*
// (GGML_BACKEND_PATH loads a DLL), LLAMA_ARG_*, LLAMA_CACHE, HF_*, MTMD_*,
// OMP_*/KMP_* and every proxy variable are dropped with the rest. Elsewhere
// PATH and TMPDIR are the whole environment.
var engineEnvironmentNames = map[string]bool{
	"SYSTEMROOT": true, "WINDIR": true, "SYSTEMDRIVE": true, "PATH": true, "TEMP": true, "TMP": true, "TMPDIR": true,
	"USERPROFILE": true, "LOCALAPPDATA": true, "APPDATA": true, "PROGRAMDATA": true, "ALLUSERSPROFILE": true,
	"NUMBER_OF_PROCESSORS": true, "PROCESSOR_ARCHITECTURE": true,
	// LLAMA_API_KEY is this launch's own key, made by the provider; an
	// inherited one never reaches the engine.
	"LLAMA_API_KEY": true,
}

// engineSystemNames are the Windows variables of the engine environment the
// provider kit computes from the token, the profile registry and
// GetNativeSystemInfo (inference CONTRACT.md INF-EL1).
var engineSystemNames = []string{"USERPROFILE", "LOCALAPPDATA", "APPDATA", "ProgramData", "ALLUSERSPROFILE", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"}

// registryString reads the profile registry for the builder; tests replace
// it to model a profile whose folders name other variables.
var registryString enginelaunch.RegistryValue = enginelaunch.SystemRegistry

// engineEnvironment builds the engine environment from nothing through the
// provider kit's clean-environment builder, with PATH of System32 alone, and
// names the profile variables it left out. Every value comes from the
// kernel, the token or the profile registry, so an inherited SystemRoot,
// PATH or LOCALAPPDATA cannot point the engine at planted files. The profile
// folders are there for the GPU driver's pipeline caches under LOCALAPPDATA
// and for llama.cpp's fs_get_cache_directory, which reads LOCALAPPDATA
// without a check. Elsewhere PATH and TMPDIR are the whole environment.
func engineEnvironment(temp string) ([]string, []error, error) {
	var omitted []error
	env, err := enginelaunch.CleanEnvironment{Temp: temp, Names: engineSystemNames, Registry: registryString,
		OnError: func(err error) { omitted = append(omitted, err) }}.Build()
	if err != nil {
		return nil, nil, err
	}
	return env, omitted, nil
}

// newLaunchKey is a fresh key for one engine start: 32 random bytes, hex.
func newLaunchKey() (string, error) {
	var raw [32]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", fmt.Errorf("modelhost: launch key: %w", err)
	}
	key := hex.EncodeToString(raw[:])
	clear(raw[:])
	return key, nil
}

// engineTemp is the engine's private temporary folder: Temp, or engine-temp
// beside the folder record in the provider's private state.
func (e Engine) engineTemp() (string, error) {
	temp := e.Temp
	if temp == "" {
		if !filepath.IsAbs(e.Record) {
			return "", errors.New("the engine has no private temporary folder")
		}
		temp = filepath.Join(filepath.Dir(e.Record), "engine-temp")
	}
	if !filepath.IsAbs(temp) {
		return "", fmt.Errorf("the engine temporary folder %s is not absolute", temp)
	}
	if err := os.MkdirAll(temp, 0o700); err != nil {
		return "", err
	}
	return filepath.Clean(temp), nil
}

// launchEnvironment is the checked environment the engine starts with,
// carrying key as LLAMA_API_KEY, and one diagnostic per profile variable it
// left out (MODELHOST-L6).
func (e Engine) launchEnvironment(key string) ([]string, []error, error) {
	env, omitted, err := e.cleanEnvironment()
	if err != nil {
		return nil, nil, err
	}
	env = sortedEnvironment(append(env, "LLAMA_API_KEY="+key))
	if err := checkEngineEnvironment(env); err != nil {
		return nil, nil, err
	}
	return env, omitted, nil
}

// cleanEnvironment is the engine environment built from nothing in the
// engine's private temporary folder, sorted, with one diagnostic per profile
// variable left out; every engine kind starts from it (MODELHOST-L6, V4).
func (e Engine) cleanEnvironment() ([]string, []error, error) {
	temp, err := e.engineTemp()
	if err != nil {
		return nil, nil, err
	}
	env, omitted, err := engineEnvironment(temp)
	if err != nil {
		return nil, nil, err
	}
	return sortedEnvironment(env), omitted, nil
}

// checkEngineEnvironment refuses a name outside the allowlist, a name the
// provider kit's llama.cpp policy forbids whatever the allowlist says
// (inference CONTRACT.md INF-LC2), an empty value and a repeated name.
func checkEngineEnvironment(env []string) error {
	seen := map[string]bool{}
	for _, item := range env {
		name, value, ok := strings.Cut(item, "=")
		upper := strings.ToUpper(name)
		if !ok || !engineEnvironmentNames[upper] || llamacpp.ForbiddenEnvironment(upper) || seen[upper] || value == "" {
			return fmt.Errorf("%w: %q", ErrEngineEnvironment, name)
		}
		seen[upper] = true
	}
	if len(env) == 0 {
		return fmt.Errorf("%w: empty", ErrEngineEnvironment)
	}
	return nil
}

// sortedEnvironment orders env case-insensitively by name, as CreateProcess
// expects an environment block.
func sortedEnvironment(env []string) []string {
	sort.Slice(env, func(i, j int) bool { return strings.ToUpper(env[i]) < strings.ToUpper(env[j]) })
	return env
}
