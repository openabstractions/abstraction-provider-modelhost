package host

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/openabstractions/abstraction-identity/go/trustedpath"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginefiles"
	"github.com/openabstractions/abstraction-inference/go/providerhost/sdcpp"

	identity "github.com/openabstractions/abstraction-identity"
)

// videoReview is the security review whose conditions the video engine kind
// meets (CONTRACT.md MODELHOST-V1..MODELHOST-V9).
const videoReview = "research/security-reviews/stable-diffusion.cpp-master-843-462d675-video-addendum.md"

// hummingbirdReview is the file review of Hummingbird-XT image to video,
// whose conditions add to the video addendum's (amendment 1 there).
const hummingbirdReview = "research/security-reviews/hummingbird-xt-i2v-files.md"

// PinnedVideoEngine is the reviewed stable-diffusion.cpp release for a
// platform. Only the Windows x64 Vulkan build has a review.
func PinnedVideoEngine(goos, goarch string) (EnginePin, bool) {
	if goos == "windows" && goarch == "amd64" {
		return SdCppMaster843WindowsVulkan, true
	}
	return EnginePin{}, false
}

// SdCppMaster843WindowsVulkan is stable-diffusion.cpp master-843-462d675,
// Windows x64 Vulkan. The archive's size and SHA-256 equal GitHub's asset
// digest (research/security-reviews/stable-diffusion.cpp-master-843-462d675.md);
// the 21 files are the archive's entries, all kept (base review condition 2,
// video addendum condition 1), recorded byte for byte from the reviewed
// archive as the Lemonade provider's SDCppVulkan manifest records them.
var SdCppMaster843WindowsVulkan = EnginePin{
	Release:       "master-843-462d675",
	Archive:       "sd-master-462d675-bin-win-vulkan-x64.zip",
	ArchiveURL:    "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-843-462d675/sd-master-462d675-bin-win-vulkan-x64.zip",
	ArchiveSize:   38865230,
	ArchiveSHA256: "b9248856707de2e125538ede8a7dc71a9d800bba67e469e60881dae2a5a1c0a2",
	Program:       "sd-server.exe",
	Review:        videoReview,
	Files: []enginefiles.File{
		{Path: "ggml-base.dll", Size: 657920, SHA256: "fc4b7d28e9057d915e68a06debe65b4d89eea2a2a0b1d716d6d7a442ad4519c0"},
		{Path: "ggml-cpu-alderlake.dll", Size: 832000, SHA256: "3a5ab1f52648ff40f73be578f24fba2bfd4eb3680daf66e24b009a70db25488c"},
		{Path: "ggml-cpu-cannonlake.dll", Size: 876544, SHA256: "8720650111b2856cfdfb8f7a51a5ed7a129fbf78400564f511201a4778218cde"},
		{Path: "ggml-cpu-cascadelake.dll", Size: 874496, SHA256: "10cf847bdd439ed2598e49b701c0565a516aae359242f7b205914b9d08c529d2"},
		{Path: "ggml-cpu-haswell.dll", Size: 833024, SHA256: "85afeea59a3f613b60b0415f1d766cd11ea9a600e5d8be0f9bc26e73a692dbdf"},
		{Path: "ggml-cpu-icelake.dll", Size: 874496, SHA256: "99aec7f7b42db3b0aca854a0903180fd934f1378c6d6dda04333b23a54d06be0"},
		{Path: "ggml-cpu-sandybridge.dll", Size: 806400, SHA256: "70fc87cc9b6bd54d71c67750f5d5c96de85bf811da89081062f65690b314749b"},
		{Path: "ggml-cpu-skylakex.dll", Size: 876544, SHA256: "87a86bf4262e5c3bc03d261088eeaa2711e9dca2bf7f35c35ff8014a6080e739"},
		{Path: "ggml-cpu-sse42.dll", Size: 793600, SHA256: "72c99ca5c9bd4572fb2a4db3e07a752d924d73ed2192246e98fa146a7a3c4747"},
		{Path: "ggml-cpu-x64.dll", Size: 794112, SHA256: "8e9ca8084955e08b5dc82373f18003c1e8e970bd6d0147e9ffb8773377448a67"},
		{Path: "ggml-vulkan.dll", Size: 50544640, SHA256: "76eb7a34115da8a5ef8a8e2d065ade87ef624fc966a9be38501f533e3d59b9ba"},
		{Path: "ggml.dll", Size: 63488, SHA256: "5c1bd056e0019853b5cd3273685a355db3a7a0d89ec0c705a63dbef94f30e565"},
		{Path: "ggml.txt", Size: 1099, SHA256: "bcd8ec749126d45cb06737d0690295d73df4b6e7e194205bcf91190368f27285"},
		{Path: "libsharpyuv.dll", Size: 24576, SHA256: "6ea16345927abecd4aab1ae403b2f8d1fdce7108c564d27fae31ceb697c465bd"},
		{Path: "libwebp.dll", Size: 373248, SHA256: "5106ef770c86a86ce608c3552392cbc1b6f7239765eced7715d862cce308de01"},
		{Path: "libwebpmux.dll", Size: 41984, SHA256: "a1d37240c9b0c88f3da51475e831b0e57d45f14cf70e5215624c0b193f38f6f2"},
		{Path: "sd-cli.exe", Size: 642560, SHA256: "c8ca2f7f415c0e0a5df7ded0837a5810e22e326870af72b3bedf207f4508f65d"},
		{Path: "sd-server.exe", Size: 1112064, SHA256: "e45bbbe59f30746a7933c63c400a00996b8a3059335bae878b11bbdc555dceae"},
		{Path: "stable-diffusion.cpp.txt", Size: 1082, SHA256: "4e76049398df734a97db46f3bd4b5cbc460cb5f6bfaf737cdc3d6fbb8ab3de27"},
		{Path: "stable-diffusion.dll", Size: 52770816, SHA256: "e48ad012df24d1098feb46e2127a3c37b4035f16e44b5a485da019597074b742"},
		{Path: "webm.dll", Size: 328192, SHA256: "898533cd95c668f9dc500e28e4bccbd4f84e8c7bd8074ddea753ee591d3c1791"},
	},
}

// Model roles: the sd-server option each pinned file is passed with.
const (
	RoleDiffusionModel = "--diffusion-model"
	RoleT5XXL          = "--t5xxl"
	RoleTAE            = "--tae"
)

// A VideoModelSet is one video model as data: the name callers ask for, its
// reviewed files by role and digest, the reviewed sd-server switches it
// always runs with, its preset and the sizes, frames, rate and first frame
// it takes. A new model is a row of VideoModelSets plus the review its
// Review names; no code changes.
type VideoModelSet struct {
	// Name is the model name callers ask for and the router lists.
	Name   string
	Review string
	Files  []VideoModelFile
	// Options are reviewed sd-server switches every launch of this set
	// passes beside the ones a person configured (video addendum condition
	// 3's list); a valued option is refused.
	Options []string
	// Preset is the set's generation preset (INF-G8); its width, height
	// and rate are what a request of size auto and rate zero gets.
	Preset sdcpp.VideoPreset
	// Sizes, MaxFrames and MaxFPS bound a caller's request, within the
	// engine's reviewed bounds (INF-SD6).
	Sizes     [][2]int
	MaxFrames int
	MaxFPS    int
	// FirstFrame is how the set takes a caller's first frame.
	FirstFrame FirstFrameUse
	// UsePolicy is a use-based policy that travels with the set's weights;
	// wherever the set's licence is shown (Setup, Panel), it is shown too
	// and passed to the person. Zero names none.
	UsePolicy UsePolicy
}

// VideoModelSets are the reviewed video models, by name.
var VideoModelSets = []VideoModelSet{FastWan22TI2V5B, HummingbirdXTI2V}

// VideoModelSetNamed is the set of VideoModelSets named name.
func VideoModelSetNamed(name string) (VideoModelSet, bool) {
	for _, s := range VideoModelSets {
		if s.Name == name {
			return s, true
		}
	}
	return VideoModelSet{}, false
}

// FastWanPreset is FastWan 2.2 TI2V 5B's preset: three DMD steps, no
// classifier-free guidance, Euler with the discrete schedule and flow shift
// 3 (stable-diffusion.cpp's documented Wan 2.2 TI2V 5B recipe), 24 fps,
// 832×480 for size auto, WebM.
var FastWanPreset = sdcpp.VideoPreset{Steps: 3, TxtCfg: 1, FlowShift: 3, SampleMethod: "euler", Scheduler: "discrete",
	FPS: 24, Width: 832, Height: 480, OutputFormat: sdcpp.FormatWebM, OutputCompression: 90}

// FastWan22TI2V5B is FastWan 2.2 TI2V 5B q8_0 with the umt5-xxl Q8_0
// encoder and the taew2_2 decoder, pinned by the video addendum's model
// table (metadata from the Hugging Face and GitHub APIs on 2026-10-01).
var FastWan22TI2V5B = VideoModelSet{
	Name:   "fastwan2.2-ti2v-5b",
	Review: videoReview,
	// Without flash attention the diffusion model's attention at
	// 1280x704x121 passed 87 GB of commit and ran the device out of memory;
	// with it and a tiled decode the run peaked at 32.8 GB
	// (research/live-runs/2026-10-01.md, runs 3b and 3d).
	Options:    []string{"--diffusion-fa", "--vae-tiling"},
	Preset:     FastWanPreset,
	Sizes:      [][2]int{{832, 480}, {480, 832}, {1280, 704}, {704, 1280}},
	MaxFrames:  121,
	MaxFPS:     30,
	FirstFrame: FirstFrameOptional,
	Files: []VideoModelFile{
		{Name: "FastWan2.2-TI2V-5B-q8_0.gguf", Role: RoleDiffusionModel, Size: 5412844128,
			SHA256: "b62f50ff87c4dfa2910c6883d45015e05b709366581698b302e259e7f25c9208",
			Source: "https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF", Revision: "3e8fe5537b1200654868aa24ea8d0f4012fb3a1e", License: "Apache-2.0"},
		umt5XXLEncoderQ8, taew22,
	},
}

// umt5XXLEncoderQ8 is the umt5-xxl encoder both Wan 2.2 TI2V 5B sets pass
// with --t5xxl (video addendum's model table).
var umt5XXLEncoderQ8 = VideoModelFile{Name: "umt5-xxl-encoder-Q8_0.gguf", Role: RoleT5XXL, Size: 6043068256,
	SHA256: "2521d4de0bf9e1cc6549866463ceae85e4ec3239bc6063f7488810be39033bbc",
	Source: "https://huggingface.co/city96/umt5-xxl-encoder-gguf", Revision: "b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7", License: "Apache-2.0"}

// taew22 is the tiny Wan 2.2 autoencoder both sets pass with --tae. Its
// SHA-256 was recorded on 2026-10-01 from the owner-approved download whose
// Git blob SHA-1 matched the publisher's commit.
var taew22 = VideoModelFile{Name: "taew2_2.safetensors", Role: RoleTAE, Size: 22848048,
	SHA256:      "b84609b2a133d48434bd9636bfcb44bf05168dc436e2d3cecf26256faa1f5325",
	GitBlobSHA1: "9d0ef19504c0b918eb6a8efea1ed8596cf666400",
	Source:      "https://github.com/madebyollin/taehv/blob/011dfc2112197741c540e0bdd5b7b67bcc930771/safetensors/taew2_2.safetensors",
	Revision:    "011dfc2112197741c540e0bdd5b7b67bcc930771", License: "MIT"}

// HummingbirdPreset is AMD's Hummingbird-XT recipe as sd-server master-843
// runs it: three steps of the lcm sampler, which re-noises between steps as
// AMD's DMD backward simulation does, at guidance 1, on AMD's own sigma
// table (1, 0.9677, 0.9091: shift 10 at timesteps 1000, 750, 500;
// infer/run_i2v.sh at 929e90a2), 16 fps, 512x512, WebM. FlowShift is AMD's
// shift and goes unused beside the table.
var HummingbirdPreset = sdcpp.VideoPreset{Steps: 3, TxtCfg: 1, FlowShift: 10, SampleMethod: "lcm", Scheduler: "lcm",
	FPS: 16, Width: 512, Height: 512, OutputFormat: sdcpp.FormatWebM, OutputCompression: 90,
	Sigmas: []float64{1, 0.967742, 0.909091, 0}}

// HummingbirdXTI2V is AMD's Hummingbird-XT image-to-video transformer, the
// one MuseTree's Idea Clip runs, read where ASUS's AI store keeps it, with
// OA's pinned umt5-xxl encoder and the taew2_2 decoder (TurboVAED needs a
// PyTorch engine), under the file review hummingbirdReview names and AMD's
// Responsible AI Use Policy.
var HummingbirdXTI2V = VideoModelSet{
	Name:   "hummingbird-xt-i2v",
	Review: hummingbirdReview,
	// Flash attention as for FastWan; 512x512 decodes without tiling.
	Options:    []string{"--diffusion-fa"},
	Preset:     HummingbirdPreset,
	Sizes:      [][2]int{{512, 512}},
	MaxFrames:  81,
	MaxFPS:     16,
	FirstFrame: FirstFrameRequired,
	UsePolicy: UsePolicy{Name: "AMD Responsible AI Use Policy", License: "Open RAIL-MS",
		URL: "https://github.com/AMD-AGI/HummingbirdXT/blob/929e90a26c2d023c89f4428aa43869c88817a55e/NOTICES"},
	Files: []VideoModelFile{
		{Name: "diffusion_pytorch_model_i2v.safetensors", Role: RoleDiffusionModel, Size: 9999659672,
			SHA256: "6937e10a78a130d5f3a4d44705db476d98af5932b3cad1e011180294ca2b8291",
			Source: "https://huggingface.co/amd/HummingbirdXT", Revision: "4d3bd2e3a8c96a189ed158ab74b575ec6a424d2c", License: "Apache-2.0",
			Locators: []string{"hummingbird/HM_2.2_5B/transformer_20251218/diffusion_pytorch_model_i2v.safetensors",
				"hm_2.2_5B/diffusion_pytorch_model.safetensors"}},
		umt5XXLEncoderQ8, taew22,
	},
}

// Size is the set's total size, the lease's admission estimate.
func (s VideoModelSet) Size() int64 {
	var n int64
	for _, f := range s.Files {
		n += f.Size
	}
	return n
}

// File is the file of the set passed with role.
func (s VideoModelSet) File(role string) (VideoModelFile, bool) {
	for _, f := range s.Files {
		if f.Role == role {
			return f, true
		}
	}
	return VideoModelFile{}, false
}

// valid refuses a set that cannot be enforced as written.
func (s VideoModelSet) valid() error {
	if s.Name == "" || len(s.Files) == 0 {
		return errors.New("the video model set names no files")
	}
	if err := validFiles(s.Name, s.Files); err != nil {
		return err
	}
	roles := map[string]bool{}
	for _, f := range s.Files {
		roles[f.Role] = true
	}
	for _, role := range []string{RoleDiffusionModel, RoleT5XXL, RoleTAE} {
		if !roles[role] {
			return fmt.Errorf("the video model set has no file for %s", role)
		}
	}
	if len(roles) != 3 {
		return errors.New("the video model set names a role sd-server is not given")
	}
	options, err := sdcpp.ParseArgs(s.Options)
	if err != nil {
		return fmt.Errorf("the video model set's options: %w", err)
	}
	if _, err := withSwitches(sdcpp.Args{}, options); err != nil {
		return err
	}
	if err := s.Preset.Validate(); err != nil {
		return err
	}
	if len(s.Sizes) == 0 {
		return errors.New("the video model set takes no size")
	}
	auto := false
	for _, size := range s.Sizes {
		if err := sdcpp.CheckVideoSize(size[0], size[1]); err != nil {
			return err
		}
		auto = auto || size == [2]int{s.Preset.Width, s.Preset.Height}
	}
	if !auto {
		return errors.New("the video model set's preset size is not one of its sizes")
	}
	switch {
	case sdcpp.AlignFrames(s.MaxFrames) != s.MaxFrames || s.MaxFrames > sdcpp.MaxVideoFrames:
		return fmt.Errorf("the video model set's %d frames is not 4n+1 within the engine's bound", s.MaxFrames)
	case s.MaxFPS < 1 || s.MaxFPS > sdcpp.MaxVideoFPS || s.Preset.FPS > s.MaxFPS:
		return fmt.Errorf("the video model set's rate %d is outside the engine's bound or below its preset", s.MaxFPS)
	}
	switch s.FirstFrame {
	case FirstFrameOptional, FirstFrameRequired, FirstFrameNone:
	default:
		return fmt.Errorf("the video model set's first frame %q is none of optional, required and none", s.FirstFrame)
	}
	if p := s.UsePolicy; p != (UsePolicy{}) && (p.Name == "" || p.License == "" || !strings.HasPrefix(p.URL, "https://")) {
		return fmt.Errorf("the video model set's use policy %+v needs a name, a licence and an https link", p)
	}
	return nil
}

// VideoModelPlaces is the record of where each pinned file of set was
// found, and VideoModelRecord the metadata record its placed proof keeps,
// both in the provider's private state.
func VideoModelPlaces(stateDir string, set VideoModelSet) string {
	return filepath.Join(stateDir, "models-"+set.Name+".places")
}

// VideoModelRecord is the metadata record of the set's placed files.
func VideoModelRecord(stateDir string, set VideoModelSet) string {
	return filepath.Join(stateDir, "models-"+set.Name+".files")
}

// VideoEngineRecord is the metadata record of the sd-server folder.
func VideoEngineRecord(stateDir string, pin EnginePin) string {
	return filepath.Join(stateDir, "engine-sd.cpp-"+pin.Release+".files")
}

// PlaceVideoModels finds each pinned file of set through searches, in
// order, and proves it where it lies by size, SHA-256 and, when pinned, Git
// blob SHA-1. It copies nothing (VISION 2026-10-01: OA never copies a model
// for an engine). A candidate whose digest differs is refused and the
// search goes on; a file no search proves is the error, naming each refused
// candidate. The record at places then names where each file was found.
func PlaceVideoModels(ctx context.Context, set VideoModelSet, searches []ModelSearch, places string) (map[string]string, error) {
	if err := set.valid(); err != nil {
		return nil, err
	}
	return placeFiles(ctx, set.Files, searches, places)
}

// readPlaces reads the record PlaceVideoModels wrote, refusing one that
// names a file of no pin or misses one.
func readPlaces(places string, set VideoModelSet) (map[string]string, error) {
	where, _, err := readPlacesOwned(places, set)
	return where, err
}

// readPlacesOwned reads the record PlaceVideoModels wrote: where each file
// lies and the owners it may have to be trusted in place.
func readPlacesOwned(places string, set VideoModelSet) (map[string]string, map[string]trustedpath.Owners, error) {
	return readFilePlaces(places, set.Name, set.Files, kindSdCpp)
}

// RecordedSetFolders are the folders directly under root that a recorded
// set uses: one holding a file a places record in state names
// (VideoModelPlaces), or one named as a set of VideoModelSets is. Only these
// are the model host's set folders; the PyTorch provider records its own.
func RecordedSetFolders(root, state string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	// One folder named twice is listed once, keyed by file identity
	// (identity.FileKey); a path that is not an existing folder is skipped.
	add := func(dir string) {
		if dir == "" {
			return
		}
		key, err := identity.FileKey(dir)
		if err != nil || seen[key] {
			return
		}
		if info, err := os.Lstat(dir); err == nil && info.IsDir() {
			seen[key] = true
			out = append(out, dir)
		}
	}
	for _, set := range VideoModelSets {
		add(filepath.Join(root, set.Name))
		if state == "" {
			continue
		}
		where, err := readPlaces(VideoModelPlaces(state, set), set)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		for _, p := range where {
			add(SetFolderOf(root, p))
		}
	}
	slices.Sort(out)
	return out, nil
}

// kindSdCpp names the models install kind a refusal points to.
const kindSdCpp = "sdcpp"
