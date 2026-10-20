package host

import (
	"archive/zip"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"

	identity "github.com/openabstractions/abstraction-identity"
)

// An EnginePin is the one llama.cpp release this provider launches: the
// publisher's archive and OA's reduced package ZIP each by size and SHA-256,
// and every runtime file kept from them by size and SHA-256 (VISION 2026-10-01).
// The engine folder holds exactly Files;
// enginefiles refuses a missing, changed or extra file before each start
// (inference CONTRACT.md INF-F1..INF-F4).
type EnginePin struct {
	// Release is llama.cpp's release tag, for example "b10825".
	Release string
	// Archive is the publisher's asset name, ArchiveURL where it is published.
	Archive    string
	ArchiveURL string
	// ArchiveSize and ArchiveSHA256 are the publisher's record of the asset.
	ArchiveSize   int64
	ArchiveSHA256 string
	// SourceCommit is the publisher's reviewed release commit.
	SourceCommit string
	// Bundle is OA's deterministic, reviewed-files-only ZIP derived from Archive.
	// Its member manifest is Files, never a second list of engine files.
	Bundle EngineBundle
	// Program is the file of Files that is started.
	Program string
	// Files are the archive entries the engine folder holds.
	Files []enginefiles.File
	// Excluded are the archive entries the security review requires absent.
	Excluded []string
	// Review is the security review that recorded Files.
	Review string
}

// EngineBundle pins the reduced package ZIP, distinct from the publisher ZIP.
type EngineBundle struct {
	Name   string
	Size   int64
	SHA256 string
}

// EnginePinInfo is the versioned public metadata Setup and the package builder
// read from modelhostd engine pin --json. Names are basenames; the caller owns
// the installed package path and the engine's destination folder.
type EnginePinInfo struct {
	Schema  int    `json:"schema"`
	Release string `json:"release"`
	Program string `json:"program"`
	Archive struct {
		Name         string `json:"name"`
		URL          string `json:"url"`
		SourceCommit string `json:"source_commit"`
		SizeBytes    int64  `json:"size_bytes"`
		SHA256       string `json:"sha256"`
	} `json:"archive"`
	Bundle struct {
		Name      string `json:"name"`
		SizeBytes int64  `json:"size_bytes"`
		SHA256    string `json:"sha256"`
	} `json:"bundle"`
	Files []struct {
		Path      string `json:"path"`
		SizeBytes int64  `json:"size_bytes"`
		SHA256    string `json:"sha256"`
	} `json:"files"`
}

// Info returns one read-only view of the compiled pin. Setup reads Bundle;
// package derivation reads Archive and Files from this same value.
func (p EnginePin) Info() EnginePinInfo {
	info := EnginePinInfo{Schema: 1, Release: p.Release, Program: p.Program}
	info.Archive.Name, info.Archive.URL = p.Archive, p.ArchiveURL
	info.Archive.SourceCommit = p.SourceCommit
	info.Archive.SizeBytes, info.Archive.SHA256 = p.ArchiveSize, p.ArchiveSHA256
	info.Bundle.Name, info.Bundle.SizeBytes, info.Bundle.SHA256 = p.Bundle.Name, p.Bundle.Size, p.Bundle.SHA256
	for _, f := range p.Files {
		info.Files = append(info.Files, struct {
			Path      string `json:"path"`
			SizeBytes int64  `json:"size_bytes"`
			SHA256    string `json:"sha256"`
		}{f.Path, f.Size, f.SHA256})
	}
	return info
}

func (p EnginePin) manifest() enginefiles.Manifest {
	return enginefiles.Manifest{Files: append([]enginefiles.File(nil), p.Files...)}
}

// valid reports a pin that cannot be enforced as written.
func (p EnginePin) valid() error {
	if p.Release == "" || p.Program == "" {
		return errors.New("the engine pin names no release or program")
	}
	if err := p.manifest().Validate(); err != nil {
		return err
	}
	kept := map[string]bool{}
	for _, f := range p.Files {
		if strings.Contains(f.Path, "/") {
			return fmt.Errorf("the engine pin's file %s is not at the archive root", f.Path)
		}
		kept[identity.PathNameKey(f.Path)] = true
	}
	if !kept[identity.PathNameKey(p.Program)] {
		return fmt.Errorf("the engine pin does not list its program %s", p.Program)
	}
	if p.Bundle.Name != "" || p.Bundle.Size != 0 || p.Bundle.SHA256 != "" {
		if !fs.ValidPath(p.Bundle.Name) || strings.ContainsAny(p.Bundle.Name, `/\\:`) || p.Bundle.Size <= 0 ||
			len(p.Bundle.SHA256) != 64 || strings.Trim(p.Bundle.SHA256, "0123456789abcdef") != "" {
			return errors.New("the reduced engine bundle pin needs one basename, size and lower-case SHA-256")
		}
	}
	for _, name := range p.Excluded {
		if kept[identity.PathNameKey(name)] {
			return fmt.Errorf("the engine pin both keeps and excludes %s", name)
		}
	}
	return nil
}

// PinnedEngine is the reviewed llama.cpp release for a platform. Only the
// Windows x64 Vulkan build has a security review; every other platform has
// no pin, and this provider launches no engine there.
func PinnedEngine(goos, goarch string) (EnginePin, bool) {
	if goos == "windows" && goarch == "amd64" {
		return LlamaCppB10825WindowsVulkan, true
	}
	return EnginePin{}, false
}

// LlamaCppB10825WindowsVulkan is llama.cpp b10825, Windows x64 Vulkan. The
// archive's size and SHA-256 equal GitHub's asset digest for the release
// (publisher record). The file sizes and digests are the first observation
// recorded by research/security-reviews/llama.cpp-b10825-win-vulkan-x64.md and
// its manifest.tsv, and each equals the archive entry's bytes. Review
// condition 5 removes ggml-rpc-server.exe and the command-line tools: the
// folder keeps llama-server.exe, the libraries its import tables reach
// (llama-server-impl, llama-common, llama, mtmd, ggml, ggml-base, libomp),
// the ggml CPU and Vulkan backends and the OpenMP licence. ggml-rpc.dll is
// excluded as well, so no RPC client loads.
var LlamaCppB10825WindowsVulkan = EnginePin{
	Release:       "b10825",
	Archive:       "llama-b10825-bin-win-vulkan-x64.zip",
	ArchiveURL:    "https://github.com/ggml-org/llama.cpp/releases/download/b10825/llama-b10825-bin-win-vulkan-x64.zip",
	ArchiveSize:   35227898,
	ArchiveSHA256: "8907e5f3b74cb57d93fd94e187a87d85be24dc64e3b0ff2bc1b287da17b31446",
	SourceCommit:  "9e0e220594af405a62835dc3a27495729fd8506b",
	Bundle: EngineBundle{
		Name:   "llama-b10825-oa-win-vulkan-x64.zip",
		Size:   96902073,
		SHA256: "bfed6c062e89fec3ba594da6d826703258079ca57fcea10c834e4014b9bafceb",
	},
	Program: "llama-server.exe",
	Review:  "research/security-reviews/llama.cpp-b10825-win-vulkan-x64.md",
	Files: []enginefiles.File{
		{Path: "LICENSE-LLVM-OpenMP", Size: 19741, SHA256: "fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4"},
		{Path: "ggml-base.dll", Size: 796160, SHA256: "40090a7e59b1e26f1e8c6483dd7b0d935a5dad7036c29347616d7c88bc7754e1"},
		{Path: "ggml-cpu-alderlake.dll", Size: 1230336, SHA256: "45ede07bf41f84a15fcdf62e572259ce6051ca6512e2ff0701cd19ecc9c3897e"},
		{Path: "ggml-cpu-cannonlake.dll", Size: 1448448, SHA256: "2afa5395a56476fd2defb1320715f5374da4f88222c98cdfc82b7860e881077d"},
		{Path: "ggml-cpu-cascadelake.dll", Size: 1434112, SHA256: "8670dbb1ec128a8a14c0a505624c103ce00958bc8de3a141aea440d7f590bf01"},
		{Path: "ggml-cpu-cooperlake.dll", Size: 1434624, SHA256: "300897211946ab72c2d55e1bf41ea86cc1b4313dc2aa4effc9efedc41b8e7ece"},
		{Path: "ggml-cpu-haswell.dll", Size: 1235968, SHA256: "998bdb373f5c328122a83f559e806f3d87d9ac9175031e8fd98ec84efc4c6363"},
		{Path: "ggml-cpu-icelake.dll", Size: 1440768, SHA256: "472972a1c0b4cb4574a4137e21fc1714cf37a18236bae06fa0e204fc9fbd33c4"},
		{Path: "ggml-cpu-ivybridge.dll", Size: 1127424, SHA256: "01e1629b38f0521352f1867b3b78972b55f2914cfdb34eb1d694c5753f573950"},
		{Path: "ggml-cpu-piledriver.dll", Size: 1131520, SHA256: "41aac7b7f6cdeff98b23fd6e3a8024c9de51139d19b2a91e26153d6b3c2b76ac"},
		{Path: "ggml-cpu-sandybridge.dll", Size: 1107968, SHA256: "3772d31c15c4b1b4e948e649e7babcc03d7890b4deb945c4f4cb28cc3de8d644"},
		{Path: "ggml-cpu-sapphirerapids.dll", Size: 1711616, SHA256: "deea88459bdbc6f1bc180306ea1e8551c7720bd0a4cdadc3847b690a89f01fc0"},
		{Path: "ggml-cpu-skylakex.dll", Size: 1441792, SHA256: "d8d44d32c72b497e95bc3e02d35df799742821f42b02adc0a74ecade9aaf34f0"},
		{Path: "ggml-cpu-sse42.dll", Size: 932352, SHA256: "f9e9a57219508df92f7b276e984f58e2dbedd83ee337d356618da3da6b3da7c9"},
		{Path: "ggml-cpu-x64.dll", Size: 924160, SHA256: "3e05d0c094941d5678fcfa55152663abe375b3555f563e5779eeaac74b40ad48"},
		{Path: "ggml-cpu-zen4.dll", Size: 1441280, SHA256: "f39567b48214d230a65f3534befd53d3c897359734c55bc151e619fbc4e4e329"},
		{Path: "ggml-vulkan.dll", Size: 55572992, SHA256: "cb03e41776af38d0e28dbc3815a8eefbe4538be5661157aeb38a6b48cc65c540"},
		{Path: "ggml.dll", Size: 79872, SHA256: "8f8173ea400583a20a74bd66ed4a940971c891a33b8f9b6384b33e95248c73f7"},
		{Path: "libomp.dll", Size: 768000, SHA256: "a12116ba72d1d6820407cf30be23da04ce79d6bb8a71a5ee71759c5a1faa6f1c"},
		{Path: "llama-common.dll", Size: 7802880, SHA256: "30232986786d21d8795c2c6afed58e022f9730a0ba33d952d97510686f199180"},
		{Path: "llama-server-impl.dll", Size: 8893952, SHA256: "685165c39d202aa76aac5852d79ff970d5f96f8130943e3c7b3cb98f78e7faec"},
		{Path: "llama-server.exe", Size: 9216, SHA256: "de24bae95807c16ae93afca2ffdd57a85a252d7bfa59c768d201422c6e8dd8f0"},
		{Path: "llama.dll", Size: 3143168, SHA256: "46d129df4f270ee91ef49d95173548321d6ab3ab08d03452f8985de9a783e5b4"},
		{Path: "mtmd.dll", Size: 1771008, SHA256: "e774481b5f4c874477e4958cff4d0ca2741dcd3078561c1b9e43ac6b067bceee"},
	},
	Excluded: []string{
		"ggml-rpc-server.exe", "ggml-rpc.dll", "llama.exe",
		"llama-batched-bench.exe", "llama-batched-bench-impl.dll", "llama-bench.exe", "llama-bench-impl.dll",
		"llama-cli.exe", "llama-cli-impl.dll", "llama-completion.exe", "llama-completion-impl.dll",
		"llama-fit-params.exe", "llama-fit-params-impl.dll", "llama-gemma3-cli.exe", "llama-gguf-split.exe",
		"llama-imatrix.exe", "llama-llava-cli.exe", "llama-minicpmv-cli.exe", "llama-mtmd-cli.exe",
		"llama-mtmd-debug.exe", "llama-perplexity.exe", "llama-perplexity-impl.dll", "llama-quantize.exe",
		"llama-quantize-impl.dll", "llama-qwen2vl-cli.exe", "llama-results.exe", "llama-tokenize.exe", "llama-tts.exe",
	},
}

// EngineRecord is the metadata record of pin's engine folder in the
// provider's private state folder (inference CONTRACT.md INF-F2).
func EngineRecord(stateDir string, pin EnginePin) string {
	return filepath.Join(stateDir, "engine-llama.cpp-"+pin.Release+".files")
}

// CheckEngine proves program's folder against this platform's pinned release
// the way a call does before its lease, writes record and releases the
// folder. A configuring command uses it so an unpinned engine is refused
// before a declaration names it.
func CheckEngine(ctx context.Context, program, record string) error {
	//unchecked: a platform with no pin leaves pin zero, which present refuses by name
	pin, _ := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	e := Engine{Program: program, Pin: pin, Record: record, Files: enginefiles.NewVerifier(enginefiles.Options{})}
	return e.present(ctx)
}

// RecordedEngine refuses dir unless record is an engine folder record whose
// root is dir: the folder an install or a proven start recorded.
func RecordedEngine(record, dir string) error {
	root, err := RecordedRoot(record)
	if err != nil {
		return err
	}
	want, err := filepath.Abs(dir)
	if err != nil {
		return err
	}
	if resolved, err := filepath.EvalSymlinks(want); err == nil {
		want = resolved
	}
	if same, err := identity.SameFile(root, want); err == nil && same {
		return nil
	}
	return fmt.Errorf("the engine record %s names %s, not %s", record, root, dir)
}

// RecordedRoot is the engine folder an engine folder record names. An
// absent record wraps fs.ErrNotExist.
func RecordedRoot(record string) (string, error) {
	data, err := os.ReadFile(record)
	if err != nil {
		return "", fmt.Errorf("no engine record at %s: %w", record, err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		root, ok := strings.CutPrefix(line, "root\t")
		if !ok {
			continue
		}
		if unc, ok := strings.CutPrefix(root, `\\?\UNC\`); ok {
			root = `\\` + unc
		} else {
			root = strings.TrimPrefix(root, `\\?\`)
		}
		return filepath.Clean(root), nil
	}
	return "", fmt.Errorf("%s is no engine folder record", record)
}

// InstallEngine stages pin's engine from the publisher's archive into dir and
// returns the program path to configure. The provider kit's archive install
// does the work (inference CONTRACT.md INF-F6): the archive must have the
// pinned size and SHA-256, each kept entry is written only when its own size
// and SHA-256 equal the pin, every other entry must be one pin.Excluded
// lists and is never written, and files then hashes the folder, gives it the
// read-only ACL and writes record (INF-F1, INF-F3). dir must be absent,
// empty, or already hold exactly the pinned files. A refused install leaves
// dir as it was.
func InstallEngine(ctx context.Context, files *enginefiles.Verifier, pin EnginePin, archive, dir, record string) (string, error) {
	if err := pin.valid(); err != nil {
		return "", err
	}
	if !filepath.IsAbs(archive) || !filepath.IsAbs(dir) || !filepath.IsAbs(record) {
		return "", errors.New("the archive, the engine folder and the record must be absolute paths")
	}
	_, err := files.InstallFromArchive(ctx, enginefiles.ArchiveInstall{
		Archive:  enginefiles.PinnedArchive{Path: archive, Size: pin.ArchiveSize, SHA256: pin.ArchiveSHA256, Name: pin.Archive},
		Tree:     enginefiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: pin.manifest(), Record: record},
		Excluded: pin.Excluded,
	})
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, pin.Program), nil
}

// InstallBundledEngine accepts the reduced package ZIP, whose entries are
// exactly Files. It checks the supplied ZIP even when the destination already
// proves, because the kit's idempotent install returns before reading input.
// The held archive file prevents a Windows writer from changing bytes between
// that proof and the install. The kit still owns staging, ACLs and the record.
func InstallBundledEngine(ctx context.Context, files *enginefiles.Verifier, pin EnginePin, bundle, dir, record string) (string, error) {
	if err := pin.valid(); err != nil {
		return "", err
	}
	if pin.Bundle.Name == "" {
		return "", errors.New("no reduced engine bundle is pinned")
	}
	if !filepath.IsAbs(bundle) || !filepath.IsAbs(dir) || !filepath.IsAbs(record) {
		return "", errors.New("the bundle, the engine folder and the record must be absolute paths")
	}
	archive := enginefiles.PinnedArchive{Path: bundle, Name: pin.Bundle.Name, Size: pin.Bundle.Size, SHA256: pin.Bundle.SHA256}
	held, err := proveBundledEngineArchive(ctx, pin, archive)
	if err != nil {
		return "", err
	}
	defer held.Close()
	_, err = files.InstallFromArchive(ctx, enginefiles.ArchiveInstall{
		Archive: archive,
		Tree: enginefiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir,
			Manifest: pin.manifest(), Record: record},
	})
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, pin.Program), nil
}

// proveBundledEngineArchive holds the source across the kit install so a valid
// destination cannot bypass proof of the supplied ZIP. ArchiveManifest owns
// the archive/content hashing; this bundle also requires flat ZIP entries.
func proveBundledEngineArchive(ctx context.Context, pin EnginePin, archive enginefiles.PinnedArchive) (*os.File, error) {
	held, err := holdEngineArchive(archive.Path)
	if err != nil {
		return nil, fmt.Errorf("opening the engine archive: %w", err)
	}
	closeOnError := func(err error) (*os.File, error) {
		held.Close()
		return nil, err
	}
	stat, err := held.Stat()
	if err != nil {
		return closeOnError(err)
	}
	if stat.Size() != archive.Size {
		return closeOnError(fmt.Errorf("the reduced engine bundle is %d bytes, wanted %d", stat.Size(), archive.Size))
	}
	zr, err := zip.NewReader(held, stat.Size())
	if err != nil {
		return closeOnError(fmt.Errorf("the reduced engine bundle is not ZIP: %w", err))
	}
	for _, f := range zr.File {
		if f.Name == "" || strings.ContainsAny(f.Name, `/\\`) || f.FileInfo().IsDir() {
			return closeOnError(fmt.Errorf("the reduced engine bundle has a non-root file or directory %q", f.Name))
		}
	}
	manifest, err := enginefiles.ArchiveManifest(ctx, archive, "")
	if err != nil {
		return closeOnError(err)
	}
	want := make(map[string]enginefiles.File, len(pin.Files))
	for _, f := range pin.Files {
		want[f.Path] = f
	}
	if len(manifest.Files) != len(want) {
		return closeOnError(fmt.Errorf("the engine archive has %d kept files, wanted %d", len(manifest.Files), len(want)))
	}
	for _, got := range manifest.Files {
		if expected, ok := want[got.Path]; !ok || expected != got {
			return closeOnError(fmt.Errorf("the engine archive file %q differs from the pin", got.Path))
		}
	}
	return held, nil
}
