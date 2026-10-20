package host

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/openabstractions/abstraction-provider-kit/go/sdcpp"
)

// The LTX-2.5 row pins each file by name, role, size, SHA-256 and a Hugging
// Face source at the reviewed revision (LTX addendum's file table, and the
// Q8_0 encoder of its condition 3a), and is a valid set of VideoModelSets.
func TestLTXSetIsPinnedBySizeDigestAndSource(t *testing.T) {
	set, ok := VideoModelSetNamed("ltx-2.5")
	if !ok || set.DisplayTitle() != "LTX-2.5 (video with sound)" || set.Review != ltxReview {
		t.Fatalf("the ltx-2.5 row is %+v", set)
	}
	if err := set.valid(); err != nil {
		t.Fatal(err)
	}
	type pin struct {
		role   string
		size   int64
		sha256 string
		source string
	}
	const gguf = "https://huggingface.co/vantagewithai/LTX-2.5-GGUF/resolve/9a5c1bed3ff27b5e5d055fc04e3ae884381c2343/"
	const lightricks = "https://huggingface.co/Lightricks/LTX-2.5/resolve/5e6e71018ee1756ed329b697a7b4aedc934dfce9/"
	want := map[string]pin{
		"ltx-2.5-22b-distilled-transformer-Q8_0.gguf": {RoleDiffusionModel, 23604666752,
			"e268b018768e00fa987c128d76384f487f9cf2ac3e3bddb077b979224ded467a", gguf + "distilled/ltx-2.5-22b-distilled-transformer-Q8_0.gguf"},
		"gemma4-12b-with-proj-ltx-2.5-Q8_0.gguf": {RoleLLM, 15043601968,
			"0acd76d05ecf2c81a13d9b11885d4478a5c084cdc1a7841304a5063fe91cbe65", lightricks + "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"},
		"ltx-2.5-video-vae-conv-bf16.safetensors": {RoleVAE, 1452269922,
			"685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d", lightricks + "vae/ltx-2.5-video-vae-conv-bf16.safetensors"},
		"ltx-2.5-audio-vae-bf16.safetensors": {RoleAudioVAE, 364866540,
			"c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5", lightricks + "vae/ltx-2.5-audio-vae-bf16.safetensors"},
		"ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors": {RoleUpscaler, 995778752,
			"eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8", lightricks + "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"},
	}
	if len(set.Files) != len(want) {
		t.Fatalf("%d files, want %d", len(set.Files), len(want))
	}
	for _, f := range set.Files {
		w, ok := want[f.Name]
		if !ok || f.Role != w.role || f.Size != w.size || f.SHA256 != w.sha256 || f.Source != w.source || !strings.Contains(f.Source, f.Revision) {
			t.Fatalf("%s is pinned as %+v", f.Name, f)
		}
	}
	// The BF16 encoder, which aborts master-843 on Vulkan, is never a file
	// of the set; it is only what the Q8_0 encoder is converted from.
	from, derived := Derived(LTXGemmaQ8)
	if !derived || from.SHA256 != "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1" || from.Size != 26263858182 {
		t.Fatalf("the Q8_0 encoder derives from %+v", from)
	}
	for _, f := range set.Files {
		if f.SHA256 == from.SHA256 {
			t.Fatal("the BF16 encoder is a file of the set")
		}
	}
	// The lease asks for the measured peak, above the files' size, and the
	// engine's limit is raised for it.
	if set.LeaseBytes() != 53<<30 || set.Size() >= set.LeaseBytes() {
		t.Fatalf("lease %d for files of %d", set.LeaseBytes(), set.Size())
	}
	if got := (VideoEngine{Set: set, MemoryLimit: DefaultVideoMemoryLimit}).memoryLimit(); got != 64<<30 {
		t.Fatalf("memory limit %d", got)
	}
	if got := (VideoEngine{Set: FastWan22TI2V5B, MemoryLimit: DefaultVideoMemoryLimit}).memoryLimit(); got != DefaultVideoMemoryLimit {
		t.Fatalf("FastWan's memory limit %d", got)
	}
}

// The research runner's known-good sd-cli command lines of LTX-2.5 on
// master-843 (research/live-runs/2026-10-02.md, phase 3: run 3c, 512x512x25
// in 160 s, and run 3d, the two-stage 1280x704x121 in 903 s; LTX addendum
// condition 3 lists the options). File paths are by role; the prompt is a
// placeholder.
const (
	knownGoodQuick = `-M vid_gen --diffusion-model {--diffusion-model} --llm {--llm} --vae {--vae} --audio-vae {--audio-vae}` +
		` -p {prompt} -W 512 -H 512 --video-frames 25 --fps 24 --steps 8 --cfg-scale 1 --sampling-method euler` +
		` --sigmas 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0 -s 42 --diffusion-fa --vae-tiling --temporal-tiling -o {out} -v`
	knownGoodQuality = `-M vid_gen --diffusion-model {--diffusion-model} --llm {--llm} --vae {--vae} --audio-vae {--audio-vae}` +
		` -p {prompt} -W 640 -H 352 --video-frames 121 --fps 24 --steps 8 --cfg-scale 1 --sampling-method euler` +
		` --sigmas 1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0 -s 42 --diffusion-fa --vae-tiling --temporal-tiling` +
		` --hires --hires-upscalers-dir {--hires-upscalers-dir} --hires-upscaler ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0` +
		` --hires-steps 3 --hires-sigmas 0.909375,0.725,0.421875,0 -o {out} -v`
)

// Each preset is the known-good command as sd-server takes it: the
// engine's command line carries the model files and the switches, and the
// request body carries every generation option at the value the research
// run passed. -M, -p, -o and -v belong to sd-cli alone.
func TestLTXPresetsAreTheKnownGoodCommand(t *testing.T) {
	models := t.TempDir()
	where := map[string]string{}
	for _, f := range LTX25.Files {
		where[f.Name] = filepath.Join(models, f.Name)
	}
	server, err := (VideoEngine{Set: LTX25}).arguments(41000, where)
	if err != nil {
		t.Fatal(err)
	}
	serverValue := func(flag string) (string, bool) {
		at := slices.Index(server, flag)
		if at < 0 || at+1 >= len(server) {
			return "", false
		}
		return server[at+1], true
	}
	for _, c := range []struct {
		command       string
		width, height int
		frames        int
	}{{knownGoodQuick, 512, 512, 25}, {knownGoodQuality, 1280, 704, 121}} {
		raw, err := sdcpp.BuildVideoRequest(LTX25.Preset, sdcpp.VideoParams{Prompt: "a lighthouse", Width: c.width, Height: c.height,
			Frames: sdcpp.AlignFramesTo(c.frames, LTX25.FrameStep), FPS: LTX25.Preset.FPS, Seed: 42})
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Width, Height int
			VideoFrames   int `json:"video_frames"`
			FPS           int
			Seed          int64
			SampleParams  struct {
				SampleSteps  int       `json:"sample_steps"`
				SampleMethod string    `json:"sample_method"`
				CustomSigmas []float64 `json:"custom_sigmas"`
				Guidance     struct {
					TxtCfg float64 `json:"txt_cfg"`
				}
			} `json:"sample_params"`
			Hires *struct {
				Enabled      bool
				Upscaler     string
				Steps        int
				CustomSigmas []float64 `json:"custom_sigmas"`
			}
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatal(err)
		}
		s := body.SampleParams
		words := strings.Fields(c.command)
		seen := map[string]bool{}
		for i := 0; i < len(words); i++ {
			flag, value := words[i], ""
			if i+1 < len(words) && !strings.HasPrefix(words[i+1], "-") {
				value = words[i+1]
				i++
			}
			seen[flag] = true
			got := ""
			switch flag {
			case "-M", "-p", "-o", "-v":
				continue
			case "--diffusion-model", "--llm", "--vae", "--audio-vae", "--hires-upscalers-dir":
				path, ok := serverValue(flag)
				if !ok || value != "{"+flag+"}" {
					t.Fatalf("%s: the engine is not given %s", c.command, flag)
				}
				role, _ := LTX25.File(flag)
				want := where[role.Name]
				if flag == RoleUpscaler {
					want = filepath.Dir(want)
				}
				if path != want {
					t.Fatalf("%s is %s, want %s", flag, path, want)
				}
				continue
			case "--diffusion-fa", "--vae-tiling", "--temporal-tiling":
				if !slices.Contains(server, flag) {
					t.Fatalf("the engine runs without %s", flag)
				}
				continue
			case "-W":
				got = strconv.Itoa(body.Width)
			case "-H":
				got = strconv.Itoa(body.Height)
			case "--video-frames":
				got = strconv.Itoa(body.VideoFrames)
			case "--fps":
				got = strconv.Itoa(body.FPS)
			case "--steps":
				got = strconv.Itoa(s.SampleSteps)
			case "--cfg-scale":
				got = strconv.FormatFloat(s.Guidance.TxtCfg, 'g', -1, 64)
			case "--sampling-method":
				got = s.SampleMethod
			case "--sigmas":
				got = sigmaList(s.CustomSigmas)
				value = sigmaList(parseSigmas(t, value))
			case "-s":
				got = strconv.FormatInt(body.Seed, 10)
			case "--hires":
				if body.Hires == nil || !body.Hires.Enabled {
					t.Fatalf("%dx%d sends no refine pass", c.width, c.height)
				}
				continue
			case "--hires-upscaler":
				got = body.Hires.Upscaler
			case "--hires-steps":
				got = strconv.Itoa(body.Hires.Steps)
			case "--hires-sigmas":
				got = sigmaList(body.Hires.CustomSigmas)
				value = sigmaList(parseSigmas(t, value))
			default:
				t.Fatalf("the known-good option %s has no place in the preset", flag)
			}
			if got != value {
				t.Fatalf("%dx%d: %s is %s, the known-good command passes %s", c.width, c.height, flag, got, value)
			}
		}
		if !seen["--hires"] && body.Hires != nil {
			t.Fatalf("%dx%d sends a refine pass the known-good command has not", c.width, c.height)
		}
	}
	// No option outside the known-good commands reaches the engine.
	for _, word := range server {
		if strings.HasPrefix(word, "-") && !strings.Contains(knownGoodQuality, word+" ") &&
			!slices.Contains([]string{"--listen-ip", "--listen-port"}, word) && !gpuDispatchArg(word) {
			t.Fatalf("the engine is given %s, which the known-good command does not pass", word)
		}
	}
}

// gpuDispatchArg reports an argument the sd.cpp GPU dispatch row adds to
// every engine (resource RES-G5).
func gpuDispatchArg(word string) bool {
	args, err := (VideoEngine{Set: FastWan22TI2V5B}).arguments(41000, map[string]string{
		FastWan22TI2V5B.Files[0].Name: filepath.Join(string(filepath.Separator), "a"),
		FastWan22TI2V5B.Files[1].Name: filepath.Join(string(filepath.Separator), "b"),
		FastWan22TI2V5B.Files[2].Name: filepath.Join(string(filepath.Separator), "c")})
	if err != nil {
		return false
	}
	return slices.Contains(args, word) && !slices.Contains([]string{"--t5xxl", "--tae", "--diffusion-model", "--diffusion-fa", "--vae-tiling"}, word)
}

func parseSigmas(t *testing.T, list string) []float64 {
	t.Helper()
	var out []float64
	for _, v := range strings.Split(list, ",") {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, f)
	}
	return out
}

func sigmaList(sigmas []float64) string {
	parts := make([]string, len(sigmas))
	for i, v := range sigmas {
		parts[i] = fmt.Sprint(v)
	}
	return strings.Join(parts, ",")
}
