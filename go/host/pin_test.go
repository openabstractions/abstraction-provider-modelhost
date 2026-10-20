package host

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
)

// VISION 2026-10-01: this provider launches only the pinned llama.cpp
// release. An engine path outside the pin is refused with engine:none before
// any lease is asked for, and nothing starts.
func TestAnEngineOutsideThePinIsRefusedBeforeTheLease(t *testing.T) {
	leases := newFakeLeases()
	h := composed(t, leases, func(o *Options) {
		o.Engine = Engine{Program: engineProgram(t), Ready: 30 * time.Second, Sockets: socketFolder(t)}
	})
	_, err := h.Ensure(context.Background(), "tiny:1b")
	if WordOf(err) != NoEngine {
		t.Fatalf("an unpinned engine: %v, wanted %s", err, NoEngine)
	}
	if acquires, _, _ := leases.counts(); acquires != 0 || h.Loaded("tiny:1b") {
		t.Fatalf("an unpinned engine took %d leases, loaded=%v", acquires, h.Loaded("tiny:1b"))
	}
}

// copiedEngine is the stand-in engine copied into a folder of its own, under
// a pin of that copy.
func copiedEngine(t *testing.T) Engine {
	t.Helper()
	source := engineProgram(t)
	raw, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	program := filepath.Join(t.TempDir(), filepath.Base(source))
	if err := os.WriteFile(program, raw, 0o755); err != nil {
		t.Fatal(err)
	}
	e := pinnedEngine(t)
	e.Program = program
	e.Pin = fakePin(t, program)
	return e
}

// An engine folder that differs from the pin in any file is refused before
// the lease: a planted file beside the program (Windows loads a DLL found
// there) and a program whose bytes changed.
func TestAnEngineFolderThatDiffersFromThePinIsRefused(t *testing.T) {
	cases := map[string]func(t *testing.T, e Engine){
		"extra file": func(t *testing.T, e Engine) {
			if err := os.WriteFile(filepath.Join(filepath.Dir(e.Program), "ggml-planted.dll"), []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
		},
		"changed program": func(t *testing.T, e Engine) {
			raw, err := os.ReadFile(e.Program)
			if err != nil {
				t.Fatal(err)
			}
			raw[len(raw)-1] ^= 0xff
			if err := os.WriteFile(e.Program, raw, 0o755); err != nil {
				t.Fatal(err)
			}
		},
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			e := copiedEngine(t)
			change(t, e)
			leases := newFakeLeases()
			h := composed(t, leases, func(o *Options) { o.Engine = e })
			_, err := h.Ensure(context.Background(), "tiny:1b")
			if WordOf(err) != NoEngine || !strings.Contains(err.Error(), "not the pinned llama.cpp") {
				t.Fatalf("%s: %v, wanted %s naming the pin", name, err, NoEngine)
			}
			if acquires, _, _ := leases.counts(); acquires != 0 {
				t.Fatalf("%s: took %d leases", name, acquires)
			}
		})
	}
}

// The pinned folder loads, and the same folder answers again after an unload.
func TestThePinnedEngineFolderLoads(t *testing.T) {
	requireEnginePeer(t)
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine = copiedEngine(t) })
	base, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	if answer := answered(t, h, base); !strings.Contains(answer, "tiny:1b") {
		t.Fatalf("the pinned engine answered %q", answer)
	}
	h.Unload("tiny:1b", "test")
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatalf("the second load: %v", err)
	}
}

// llama.cpp's backend loader also loads ggml-*.dll from the working
// directory, so the engine runs in its own proven folder (b10825 review,
// condition 1).
func TestTheEngineRunsInItsProvenFolder(t *testing.T) {
	requireEnginePeer(t)
	e := copiedEngine(t)
	cwdFile := filepath.Join(t.TempDir(), "cwd")
	e.fixture = []string{"--fake-cwd-file", cwdFile}
	h := composed(t, newFakeLeases(), func(o *Options) { o.Engine = e })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(cwdFile)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := filepath.EvalSymlinks(string(raw))
	want, _ := filepath.EvalSymlinks(filepath.Dir(e.Program))
	if !strings.EqualFold(filepath.Clean(got), filepath.Clean(want)) {
		t.Fatalf("the engine ran in %s, wanted its folder %s", got, want)
	}
}

// The pin is the reviewed b10825 Windows Vulkan archive, and it is the only
// pin: no other platform launches an engine.
func TestThePinnedReleaseIsTheReviewedB10825(t *testing.T) {
	pin, ok := PinnedEngine("windows", "amd64")
	if !ok || pin.Release != "b10825" {
		t.Fatalf("windows/amd64 pin = %+v, %v", pin.Release, ok)
	}
	for _, platform := range [][2]string{{"linux", "amd64"}, {"darwin", "arm64"}, {"windows", "arm64"}} {
		if _, ok := PinnedEngine(platform[0], platform[1]); ok {
			t.Fatalf("%s/%s has a pin and no review", platform[0], platform[1])
		}
	}
	if err := pin.valid(); err != nil {
		t.Fatal(err)
	}
	if pin.ArchiveSize != 35227898 || pin.ArchiveSHA256 != "8907e5f3b74cb57d93fd94e187a87d85be24dc64e3b0ff2bc1b287da17b31446" {
		t.Fatalf("archive pin %d %s", pin.ArchiveSize, pin.ArchiveSHA256)
	}
	if len(pin.Files)+len(pin.Excluded) != 52 {
		t.Fatalf("%d kept and %d excluded entries, wanted the archive's 52", len(pin.Files), len(pin.Excluded))
	}
	for _, name := range []string{"llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "llama.dll",
		"mtmd.dll", "ggml.dll", "ggml-base.dll", "libomp.dll", "ggml-vulkan.dll", "ggml-cpu-x64.dll"} {
		if !kept(pin, name) {
			t.Fatalf("%s is not kept", name)
		}
	}
	for _, name := range []string{"ggml-rpc-server.exe", "ggml-rpc.dll", "llama-cli.exe", "llama.exe"} {
		if kept(pin, name) {
			t.Fatalf("%s is kept; review condition 5 excludes it", name)
		}
	}
	// A subtest, so a skipped manifest comparison leaves the review check
	// below to run.
	t.Run("review manifest.tsv", func(t *testing.T) { compareReviewManifest(t, pin) })
	// The digests are the review's first observation. The review lives in
	// the private repository; a split module has no copy to compare with.
	review, err := os.ReadFile(filepath.Join("..", "..", "..", "..", filepath.FromSlash(pin.Review)))
	if err != nil {
		t.Skipf("review not in this tree: %v", err)
	}
	for _, f := range append(pin.Files, enginefiles.File{Path: pin.Archive, SHA256: pin.ArchiveSHA256}) {
		if !bytes.Contains(review, []byte(f.SHA256)) {
			t.Fatalf("%s %s is not the review's digest", f.Path, f.SHA256)
		}
	}
}

// compareReviewManifest holds every kept file's size and digest to the
// review's manifest.tsv ("<name>\t<size>\t<sha256>\t<signature>"), which the
// review keeps beside its archive copy in the private .build folder. It reads
// the copy in this checkout's .build or the one beside OA_LLAMA_B10825_ZIP,
// and skips with the paths it tried when neither is present.
func compareReviewManifest(t *testing.T, pin EnginePin) {
	t.Helper()
	candidates := []string{filepath.Join("..", "..", "..", "..", ".build", "security-review", "llama.cpp-b10825", "manifest.tsv")}
	if archive := os.Getenv("OA_LLAMA_B10825_ZIP"); archive != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(archive), "manifest.tsv"))
	}
	for _, candidate := range candidates {
		raw, err := os.ReadFile(candidate)
		if err != nil {
			continue
		}
		recorded := map[string]string{}
		for _, line := range strings.Split(strings.TrimPrefix(string(raw), string(rune(0xfeff))), "\n") {
			fields := strings.Split(strings.TrimRight(line, "\r"), "\t")
			if len(fields) >= 3 {
				recorded[fields[0]] = fields[1] + "\t" + fields[2]
			}
		}
		if len(recorded) != len(pin.Files)+len(pin.Excluded) {
			t.Fatalf("%s lists %d entries, the pin accounts for %d", candidate, len(recorded), len(pin.Files)+len(pin.Excluded))
		}
		for _, f := range pin.Files {
			if got, want := recorded[f.Path], strconv.FormatInt(f.Size, 10)+"\t"+f.SHA256; got != want {
				t.Fatalf("%s: %s records %q, the pin %q", f.Path, candidate, got, want)
			}
		}
		t.Logf("sizes and digests equal %s", candidate)
		return
	}
	t.Skipf("no review manifest.tsv at %s; set OA_LLAMA_B10825_ZIP to the reviewed archive beside it", strings.Join(candidates, " or "))
}

func kept(pin EnginePin, name string) bool {
	for _, f := range pin.Files {
		if f.Path == name {
			return true
		}
	}
	return false
}

// testArchive writes a ZIP of entries and returns its path and a pin that
// keeps keep and excludes the rest.
func testArchive(t *testing.T, entries map[string]string, keep ...string) (string, EnginePin) {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		f, err := w.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(entries[name])); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "engine.zip")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(buf.Bytes())
	pin := EnginePin{Release: "test", Archive: "engine.zip", ArchiveSize: int64(buf.Len()),
		ArchiveSHA256: hex.EncodeToString(sum[:]), Program: keep[0]}
	for _, name := range names {
		keepIt := false
		for _, k := range keep {
			keepIt = keepIt || k == name
		}
		if !keepIt {
			pin.Excluded = append(pin.Excluded, name)
			continue
		}
		digest := sha256.Sum256([]byte(entries[name]))
		pin.Files = append(pin.Files, enginefiles.File{Path: name, Size: int64(len(entries[name])), SHA256: hex.EncodeToString(digest[:])})
	}
	return path, pin
}

type bundleEntry struct {
	name string
	body string
	mode os.FileMode
}

func testBundle(t *testing.T, pin *EnginePin, entries ...bundleEntry) string {
	t.Helper()
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	for _, e := range entries {
		h := &zip.FileHeader{Name: e.name, Method: zip.Store}
		h.SetMode(e.mode)
		f, err := w.CreateHeader(h)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write([]byte(e.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "bundle.zip")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(buf.Bytes())
	pin.Bundle = EngineBundle{Name: "bundle.zip", Size: int64(buf.Len()), SHA256: hex.EncodeToString(sum[:])}
	return path
}

func testBundleFiles() []bundleEntry {
	return []bundleEntry{{"ggml.dll", "library", 0o644}, {"llama-server.exe", "server", 0o644}}
}

func folderNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

var testEntries = map[string]string{"llama-server.exe": "server", "ggml.dll": "library", "llama-cli.exe": "tool"}

// Install stages the kept entries of the pinned archive and nothing else,
// then proves and records the folder.
func TestInstallEngineStagesOnlyThePinnedFiles(t *testing.T) {
	archive, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
	dir := filepath.Join(t.TempDir(), "engine")
	record := filepath.Join(t.TempDir(), "state", "engine.files")
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	program, err := InstallEngine(context.Background(), v, pin, archive, dir, record)
	if err != nil {
		t.Fatal(err)
	}
	if program != filepath.Join(dir, "llama-server.exe") {
		t.Fatalf("program %s", program)
	}
	if got := strings.Join(folderNames(t, dir), ","); got != "ggml.dll,llama-server.exe" {
		t.Fatalf("staged %s", got)
	}
	if _, err := os.Stat(record); err != nil {
		t.Fatalf("no record: %v", err)
	}
	e := Engine{Program: program, Pin: pin, Record: record, Files: v}
	if err := e.present(context.Background()); err != nil {
		t.Fatalf("the installed folder: %v", err)
	}
}

func TestInstallBundledEngineProvesInputEvenWhenFolderAlreadyMatches(t *testing.T) {
	_, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
	bundle := testBundle(t, &pin, testBundleFiles()...)
	dir := filepath.Join(t.TempDir(), "engine")
	record := filepath.Join(t.TempDir(), "state", "engine.files")
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	program, err := InstallBundledEngine(context.Background(), v, pin, bundle, dir, record)
	if err != nil {
		t.Fatal(err)
	}
	if program != filepath.Join(dir, pin.Program) {
		t.Fatalf("program %q", program)
	}
	if got := strings.Join(folderNames(t, dir), ","); got != "ggml.dll,llama-server.exe" {
		t.Fatalf("staged %s", got)
	}
	if _, err := InstallBundledEngine(context.Background(), v, pin, bundle, dir, record); err != nil {
		t.Fatalf("idempotent install: %v", err)
	}
	bad := filepath.Join(t.TempDir(), "another-bundle.zip")
	if err := os.WriteFile(bad, []byte("wrong bundle"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := InstallBundledEngine(context.Background(), v, pin, bad, dir, record); err == nil {
		t.Fatal("an already-proven folder hid the wrong bundle")
	}
	if got := strings.Join(folderNames(t, dir), ","); got != "ggml.dll,llama-server.exe" {
		t.Fatalf("refused bundle changed folder: %s", got)
	}
}

func TestInstallBundledEngineRefusesNonExactMembers(t *testing.T) {
	cases := map[string][]bundleEntry{
		"directory":      append(testBundleFiles(), bundleEntry{"unused/", "", os.ModeDir | 0o755}),
		"extra file":     append(testBundleFiles(), bundleEntry{"llama-cli.exe", "tool", 0o644}),
		"case duplicate": append(testBundleFiles(), bundleEntry{"GGML.dll", "library", 0o644}),
		"link":           append(testBundleFiles(), bundleEntry{"other", "target", os.ModeSymlink | 0o644}),
	}
	for name, entries := range cases {
		t.Run(name, func(t *testing.T) {
			_, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
			bundle := testBundle(t, &pin, entries...)
			dir := filepath.Join(t.TempDir(), "engine")
			record := filepath.Join(t.TempDir(), "engine.files")
			v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
			if _, err := InstallBundledEngine(context.Background(), v, pin, bundle, dir, record); err == nil {
				t.Fatal("non-exact bundle was installed")
			}
			if _, err := os.Stat(dir); !os.IsNotExist(err) {
				t.Fatalf("refused bundle left folder: %v", err)
			}
		})
	}
}

func TestBundledEnginePinNamesOnlyABasename(t *testing.T) {
	_, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
	bundle := testBundle(t, &pin, testBundleFiles()...)
	pin.Bundle.Name = "../bundle.zip"
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	if _, err := InstallBundledEngine(context.Background(), v, pin, bundle,
		filepath.Join(t.TempDir(), "engine"), filepath.Join(t.TempDir(), "engine.files")); err == nil ||
		!strings.Contains(err.Error(), "one basename") {
		t.Fatalf("bundle path in public pin: %v", err)
	}
}

// Install refuses an archive other than the pinned one, a folder that holds
// anything, and an entry whose bytes differ from the pin, and leaves nothing
// of its own behind.
func TestInstallEngineRefusesAnythingOutsideThePin(t *testing.T) {
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	record := filepath.Join(t.TempDir(), "engine.files")
	t.Run("another archive", func(t *testing.T) {
		archive, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
		pin.ArchiveSHA256 = strings.Repeat("0", 64)
		dir := filepath.Join(t.TempDir(), "engine")
		if _, err := InstallEngine(context.Background(), v, pin, archive, dir, record); err == nil || !strings.Contains(err.Error(), "is not the pinned") {
			t.Fatalf("another archive: %v", err)
		}
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("a refused archive left %s: %v", dir, err)
		}
	})
	t.Run("a folder that holds something", func(t *testing.T) {
		archive, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "other"), nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := InstallEngine(context.Background(), v, pin, archive, dir, record); err == nil || !strings.Contains(err.Error(), "not empty") {
			t.Fatalf("a used folder: %v", err)
		}
	})
	t.Run("an entry that differs from the pin", func(t *testing.T) {
		archive, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
		pin.Files[0].SHA256 = strings.Repeat("1", 64)
		dir := filepath.Join(t.TempDir(), "engine")
		if _, err := InstallEngine(context.Background(), v, pin, archive, dir, record); err == nil || !strings.Contains(err.Error(), "wanted") {
			t.Fatalf("a changed entry: %v", err)
		}
		if _, err := os.Stat(dir); !os.IsNotExist(err) {
			t.Fatalf("a refused entry left %s: %v", dir, err)
		}
	})
}

// CheckEngine is the call's own proof against this platform's pin, for a
// configuring command: the stand-in engine is no pinned llama-server.
func TestCheckEngineRefusesAnUnpinnedProgram(t *testing.T) {
	err := CheckEngine(context.Background(), engineProgram(t), filepath.Join(t.TempDir(), "engine.files"))
	if err == nil {
		t.Fatal("the stand-in engine passed the platform's pin")
	}
	if !strings.Contains(err.Error(), "pinned") {
		t.Fatalf("the refusal %q does not name the pin", err)
	}
}

// RecordedEngine accepts only the folder the record names.
func TestRecordedEngineNamesOnlyTheInstalledFolder(t *testing.T) {
	archive, pin := testArchive(t, testEntries, "llama-server.exe", "ggml.dll")
	dir := filepath.Join(t.TempDir(), "engine")
	record := filepath.Join(t.TempDir(), "engine.files")
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	if _, err := InstallEngine(context.Background(), v, pin, archive, dir, record); err != nil {
		t.Fatal(err)
	}
	if err := RecordedEngine(record, dir); err != nil {
		t.Fatalf("the installed folder: %v", err)
	}
	if err := RecordedEngine(record, t.TempDir()); err == nil {
		t.Fatal("another folder matched the record")
	}
	if err := RecordedEngine(filepath.Join(t.TempDir(), "absent.files"), dir); err == nil {
		t.Fatal("a missing record was accepted")
	}
}

// OA_LLAMA_B10825_ZIP names a copy of the reviewed archive. The real pin
// installs from it, and the installed folder proves against the pin.
func TestInstallTheReviewedArchive(t *testing.T) {
	archive := os.Getenv("OA_LLAMA_B10825_ZIP")
	if archive == "" {
		t.Skip("OA_LLAMA_B10825_ZIP names no copy of llama-b10825-bin-win-vulkan-x64.zip")
	}
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	dir := filepath.Join(t.TempDir(), "llama.cpp-b10825")
	record := EngineRecord(t.TempDir(), LlamaCppB10825WindowsVulkan)
	program, err := InstallEngine(context.Background(), v, LlamaCppB10825WindowsVulkan, archive, dir, record)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(folderNames(t, dir)); n != len(LlamaCppB10825WindowsVulkan.Files) {
		t.Fatalf("staged %d files, wanted %d", n, len(LlamaCppB10825WindowsVulkan.Files))
	}
	e := Engine{Program: program, Pin: LlamaCppB10825WindowsVulkan, Record: record, Files: v}
	if err := e.present(context.Background()); err != nil {
		t.Fatal(err)
	}
}

// OA_LLAMA_B10825_BUNDLE names the deterministic reduced ZIP derived from the
// reviewed publisher archive. This installs its exact 24-file manifest without
// starting llama-server or any other executable.
func TestInstallTheReviewedBundle(t *testing.T) {
	bundle := os.Getenv("OA_LLAMA_B10825_BUNDLE")
	if bundle == "" {
		t.Skip("OA_LLAMA_B10825_BUNDLE names no reduced b10825 package ZIP")
	}
	v := enginefiles.NewVerifier(enginefiles.Options{ACL: openACL{}})
	dir := filepath.Join(t.TempDir(), "llama.cpp-b10825")
	record := EngineRecord(t.TempDir(), LlamaCppB10825WindowsVulkan)
	program, err := InstallBundledEngine(context.Background(), v, LlamaCppB10825WindowsVulkan, bundle, dir, record)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(folderNames(t, dir)); n != len(LlamaCppB10825WindowsVulkan.Files) {
		t.Fatalf("staged %d files, wanted %d", n, len(LlamaCppB10825WindowsVulkan.Files))
	}
	e := Engine{Program: program, Pin: LlamaCppB10825WindowsVulkan, Record: record, Files: v}
	if err := e.present(context.Background()); err != nil {
		t.Fatal(err)
	}
}
