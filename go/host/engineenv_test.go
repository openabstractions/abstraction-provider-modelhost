package host

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// b10825 review condition 3 (CONTRACT.md MODELHOST-L6): the engine starts
// with an environment built from nothing. A variable set in the provider's
// own environment never reaches it: GGML_BACKEND_PATH loads a DLL, the
// LLAMA_ARG_* forms set any option, the VK_* loader variables add drivers
// and layers, and proxy variables redirect fetches. The stand-in engine
// reports its variables' names; only the clean ones carry a value in its
// report, and this test prints names alone.
func TestTheEngineStartsWithACleanEnvironment(t *testing.T) {
	requireEnginePeer(t)
	for _, name := range []string{
		"GGML_BACKEND_PATH", "GGML_VK_VISIBLE_DEVICES", "LLAMA_ARG_HOST", "LLAMA_ARG_RPC", "LLAMA_API_KEY",
		"LLAMA_CACHE", "HF_TOKEN", "MTMD_BACKEND_DEVICE", "VK_ICD_FILENAMES", "VK_DRIVER_FILES",
		"VK_LAYER_PATH", "VK_ADD_LAYER_PATH", "VK_INSTANCE_LAYERS", "HTTP_PROXY", "HTTPS_PROXY",
		"ALL_PROXY", "NO_PROXY", "OMP_NUM_THREADS", "OA_MODELHOST_TEST_SECRET",
	} {
		t.Setenv(name, "planted-by-test")
	}
	e := copiedEngine(t)
	// The variables the engine does receive are planted too, after the
	// stand-in is built: the engine's values come from the system, never
	// from the provider's environment. SystemRoot and TEMP stay unplanted
	// because this test process needs them itself; their values are
	// asserted below.
	planted := []string{"USERPROFILE", "LOCALAPPDATA", "APPDATA", "ProgramData", "ALLUSERSPROFILE",
		"SystemDrive", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"}
	for _, name := range planted {
		t.Setenv(name, "planted-by-test")
	}
	report := filepath.Join(t.TempDir(), "env")
	e.fixture = []string{"--fake-env-file", report}
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine = e })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(report)
	if err != nil {
		t.Fatal(err)
	}
	values := map[string]string{}
	var names []string
	for _, line := range strings.Split(string(raw), "\n") {
		name, value, _ := strings.Cut(line, "\t")
		if name == "" {
			continue
		}
		upper := strings.ToUpper(name)
		names = append(names, upper)
		values[upper] = value
	}
	sort.Strings(names)
	// LLAMA_API_KEY is the provider's own per-launch key; that the engine
	// holds the provider's key and never the planted one is
	// TestTheEngineRequiresThePerLaunchKey's to show.
	// The core is always delivered; a profile variable is delivered when
	// its folder could be read (MODELHOST-L6), and nothing else is.
	core := []string{"LLAMA_API_KEY", "PATH", "TMPDIR"}
	profile := map[string]bool{}
	if runtime.GOOS == "windows" {
		core = []string{"LLAMA_API_KEY", "NUMBER_OF_PROCESSORS", "PATH", "PROCESSOR_ARCHITECTURE",
			"SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
		profile = map[string]bool{"USERPROFILE": true, "LOCALAPPDATA": true, "APPDATA": true, "PROGRAMDATA": true, "ALLUSERSPROFILE": true}
	}
	for name, value := range values {
		if strings.Contains(value, "planted-by-test") {
			t.Fatalf("%s reached the engine from the provider's environment", name)
		}
	}
	delivered := map[string]bool{}
	for _, name := range names {
		delivered[name] = true
	}
	for _, name := range core {
		if !delivered[name] {
			t.Fatalf("the engine's environment %v lacks %s", names, name)
		}
	}
	isCore := map[string]bool{}
	for _, name := range core {
		isCore[name] = true
	}
	for _, name := range names {
		if !isCore[name] && !profile[name] {
			t.Fatalf("the engine's environment holds %s, outside the core %v and the profile folders", name, core)
		}
		if profile[name] && !filepath.IsAbs(values[name]) {
			t.Fatalf("the delivered %s is not an absolute folder", name)
		}
	}
	temp := filepath.Join(filepath.Dir(e.Record), "engine-temp")
	same := func(a, b string) bool { return strings.EqualFold(filepath.Clean(a), filepath.Clean(b)) }
	if runtime.GOOS == "windows" {
		if !same(values["TEMP"], temp) || !same(values["TMP"], temp) {
			t.Fatal("TEMP and TMP are not the engine's private folder beside its record")
		}
		root := values["SYSTEMROOT"]
		if !filepath.IsAbs(root) || !same(values["WINDIR"], root) {
			t.Fatal("SystemRoot and windir are not one absolute Windows folder")
		}
		if !same(values["PATH"], filepath.Join(root, "System32")) {
			t.Fatal("PATH is not the System32 folder alone")
		}
		if delivered["PROGRAMDATA"] != delivered["ALLUSERSPROFILE"] || !same(values["PROGRAMDATA"], values["ALLUSERSPROFILE"]) {
			t.Fatal("ProgramData and ALLUSERSPROFILE are not delivered together as one folder")
		}
		if values["SYSTEMDRIVE"] != filepath.VolumeName(root) {
			t.Fatal("SystemDrive is not the Windows folder's volume")
		}
		if n, err := strconv.Atoi(values["NUMBER_OF_PROCESSORS"]); err != nil || n < 1 {
			t.Fatal("NUMBER_OF_PROCESSORS is not a processor count")
		}
		switch values["PROCESSOR_ARCHITECTURE"] {
		case "AMD64", "x86", "ARM64", "ARM", "IA64":
		default:
			t.Fatal("PROCESSOR_ARCHITECTURE is not a Windows architecture name")
		}
	} else {
		if !same(values["TMPDIR"], temp) {
			t.Fatal("TMPDIR is not the engine's private folder beside its record")
		}
		if values["PATH"] != "/usr/bin:/bin" {
			t.Fatal("PATH is not the system folders alone")
		}
	}
	if info, err := os.Stat(temp); err != nil || !info.IsDir() {
		t.Fatalf("the engine's temporary folder was not made: %v", err)
	}
}

// The provider kit's forbidden names (inference CONTRACT.md INF-LC2) are
// refused even when a name reaches the allowlist by mistake.
func TestForbiddenEngineNamesRefusedWhateverTheAllowlist(t *testing.T) {
	env, _, err := engineEnvironment(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"GGML_BACKEND_PATH", "ggml_backend_path", "MTMD_BACKEND_DEVICE", "LLAMA_ARG_HOST", "LLAMA_CACHE", "HF_TOKEN", "HTTPS_PROXY", "no_proxy"} {
		upper := strings.ToUpper(name)
		engineEnvironmentNames[upper] = true
		err := checkEngineEnvironment(append(append([]string(nil), env...), name+"=x"))
		delete(engineEnvironmentNames, upper)
		if !errors.Is(err, ErrEngineEnvironment) {
			t.Fatalf("%s accepted once allow-listed: %v", name, err)
		}
	}
}

// The environment is one list whose every name is on the allowlist; a name
// outside it is refused before any process starts.
func TestTheEngineEnvironmentRefusesAnyOtherName(t *testing.T) {
	temp := t.TempDir()
	env, _, err := engineEnvironment(temp)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkEngineEnvironment(env); err != nil {
		t.Fatalf("the built environment was refused: %v", err)
	}
	for _, planted := range []string{"GGML_BACKEND_PATH=x", "VK_ICD_FILENAMES=x", "LLAMA_ARG_HOST=0.0.0.0", "HTTPS_PROXY=x"} {
		err := checkEngineEnvironment(append(append([]string(nil), env...), planted))
		name, _, _ := strings.Cut(planted, "=")
		if err == nil || !strings.Contains(err.Error(), name) {
			t.Fatalf("%s: %v, wanted a refusal naming it", name, err)
		}
	}
}
