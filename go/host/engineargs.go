package host

import (
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/openabstractions/abstraction-inference/go/providerhost/llamacpp"
)

// ErrEngineArgument reports a llama-server argument outside the reviewed list
// (CONTRACT.md MODELHOST-L5; b10825 review condition 2).
var ErrEngineArgument = errors.New("modelhost: engine argument outside the reviewed list")

// engineFlag is one llama-server option a person may add. value is nil for a
// switch, and otherwise accepts the only values the option may carry.
type engineFlag struct {
	value func(string) bool
}

// engineFlags are the llama-server options of b10825 (common/arg.cpp) a person
// may add with --engine-arg, under their long names. Each one sizes, places
// or schedules the model this provider already chose, or turns a feature off:
// --spec-type admits only draft-mtp, the model's own draft head, and
// --spec-draft-n-max bounds its draft length to 1..8; every option that names
// a draft model (--model-draft, -md, --hf-repo-draft, -hfd) or a lookup cache
// file is refused. --load-mode, --mmap and --no-mmap choose how the engine
// reads the model file this provider named, so a person can replace the
// placement default (MODELHOST-L9).
// None loads code, reads or writes a file a person names, opens a listener,
// or reaches the network. Everything else is refused, including every option
// the b10825 review's condition 2 names (--host, --rpc, --tools, --agent, the
// MCP options, -hf and the --hf-* forms, --model-url, --mmproj-url,
// --docker-repo, --models-dir, --models-preset, --media-path, --path,
// --ui-config-file, --slot-save-path, --log-file, --log-prompts-dir,
// --sleep-idle-seconds), the options that read a file (--lora,
// --chat-template-file, --api-key-file, --ssl-*, --mmproj, --grammar-file),
// --chat-template, whose value is a Jinja template, and the options this
// provider sets itself (--model, --host, --port, --alias, --embeddings,
// --no-webui).
var engineFlags = map[string]engineFlag{
	"--ctx-size":              {atLeast(0)},
	"--n-predict":             {atLeast(-2)},
	"--batch-size":            {atLeast(1)},
	"--ubatch-size":           {atLeast(1)},
	"--keep":                  {atLeast(-1)},
	"--threads":               {atLeast(-1)},
	"--threads-batch":         {atLeast(-1)},
	"--threads-http":          {atLeast(-1)},
	"--parallel":              {atLeast(-1)},
	"--n-gpu-layers":          {gpuLayers},
	"--split-mode":            {oneOf("none", "layer", "row")},
	"--main-gpu":              {atLeast(0)},
	"--device":                {devices},
	"--flash-attn":            {oneOf("on", "off", "auto")},
	"--cache-type-k":          {oneOf(cacheTypes...)},
	"--cache-type-v":          {oneOf(cacheTypes...)},
	"--pooling":               {oneOf("none", "mean", "cls", "last", "rank")},
	"--cache-reuse":           {atLeast(0)},
	"--cache-ram":             {atLeast(-1)},
	"--timeout":               {atLeast(1)},
	"--seed":                  {atLeast(-1)},
	"--reasoning-format":      {oneOf("none", "deepseek", "deepseek-legacy", "auto")},
	"--reasoning-budget":      {atLeast(-1)},
	"--log-verbosity":         {atLeast(0)},
	"--spec-type":             {oneOf("draft-mtp")},
	"--spec-draft-n-max":      {between(1, 8)},
	"--cont-batching":         {},
	"--no-cont-batching":      {},
	"--load-mode":             {oneOf("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")},
	"--mlock":                 {},
	"--mmap":                  {},
	"--no-mmap":               {},
	"--no-kv-offload":         {},
	"--kv-unified":            {},
	"--swa-full":              {},
	"--context-shift":         {},
	"--no-context-shift":      {},
	"--no-warmup":             {},
	"--no-reasoning-preserve": {},
	"--log-disable":           {},
}

// engineFlagNames are the other spellings llama-server accepts for an
// option on the list.
var engineFlagNames = map[string]string{
	"-c": "--ctx-size", "-n": "--n-predict", "--predict": "--n-predict",
	"-b": "--batch-size", "-ub": "--ubatch-size", "-t": "--threads", "-tb": "--threads-batch",
	"-np": "--parallel", "-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers",
	"-sm": "--split-mode", "-mg": "--main-gpu", "-dev": "--device", "-fa": "--flash-attn",
	"-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-cram": "--cache-ram", "-to": "--timeout",
	"-s": "--seed", "-lv": "--log-verbosity", "--verbosity": "--log-verbosity",
	"-cb": "--cont-batching", "-nocb": "--no-cont-batching", "-nkvo": "--no-kv-offload",
	"-kvu": "--kv-unified",
}

var cacheTypes = []string{"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}

func atLeast(least int) func(string) bool {
	return func(v string) bool {
		n, err := strconv.Atoi(v)
		return err == nil && n >= least && n <= 1<<30
	}
}

func between(least, most int) func(string) bool {
	return func(v string) bool {
		n, err := strconv.Atoi(v)
		return err == nil && n >= least && n <= most
	}
}

func oneOf(values ...string) func(string) bool {
	return func(v string) bool {
		for _, allowed := range values {
			if v == allowed {
				return true
			}
		}
		return false
	}
}

func gpuLayers(v string) bool { return v == "auto" || v == "all" || atLeast(-1)(v) }

// The resources a lease can name for an engine's weights (resource
// CONTRACT.md): an accelerator's memory, card:<n>, or system memory.
const (
	cardResourcePrefix = "card:"
	memoryResource     = "memory"
)

// loadFlags are the reviewed options that choose how llama-server reads the
// model file. b10825 reads them as one setting, the last one given winning.
var loadFlags = []string{"--load-mode", "--mmap", "--no-mmap", "--mlock"}

// enginePlacement is the placement a llama-server engine starts with under a
// lease of resource for a model file of size bytes (CONTRACT.md
// MODELHOST-L9). The holder starts an engine only after its lease granted
// the file's size, an admission estimate of the weights alone (MODELHOST-H1).
// configured are the person's arguments in canonical form (EngineArguments);
// an option they set is never defaulted.
//
//   - A card lease reads the file into the engine's own buffers: --no-mmap.
//     The layer count is left to the engine's --fit, which places what fits
//     the device memory it measures, so a discrete card smaller than the
//     model keeps a partial offload.
//   - A memory lease keeps every layer in system memory, the resource the
//     lease admitted: --n-gpu-layers 0, and the file is mapped as the
//     engine does by default.
//   - Any other resource adds nothing.
func enginePlacement(resource string, size int64, configured []string) []string {
	var layers string
	unmapped := false
	switch {
	case strings.HasPrefix(resource, cardResourcePrefix) && size > 0:
		unmapped = true
	case resource == memoryResource:
		layers = "0"
	}
	set := func(names ...string) bool {
		for _, name := range names {
			if slices.Contains(configured, name) {
				return true
			}
		}
		return false
	}
	var out []string
	if layers != "" && !set("--n-gpu-layers") {
		out = append(out, "--n-gpu-layers", layers)
	}
	if unmapped && !set(loadFlags...) {
		out = append(out, "--no-mmap")
	}
	return out
}

// devices accepts llama.cpp's device list: "none", or backend device names
// such as Vulkan0 separated by commas.
func devices(v string) bool {
	if v == "none" {
		return true
	}
	for _, name := range strings.Split(v, ",") {
		if name == "" || len(name) > 64 {
			return false
		}
		for _, r := range name {
			if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_') {
				return false
			}
		}
	}
	return true
}

// EngineArguments checks the llama-server arguments a person configured
// against the reviewed list and returns them in canonical form: each flag
// under its long name, each value its own token. The first argument outside
// the list is refused by name (CONTRACT.md MODELHOST-L5).
func EngineArguments(args []string) ([]string, error) {
	out := make([]string, 0, len(args))
	seen := map[string]bool{}
	for i := 0; i < len(args); i++ {
		given := args[i]
		flag, value, inline := strings.Cut(given, "=")
		// The provider kit's list of review condition 2 (inference
		// CONTRACT.md INF-LC1) is refused first; the reviewed list below
		// admits none of it.
		if name, forbidden := llamacpp.ForbiddenFlag(given); forbidden {
			return nil, fmt.Errorf("%w: %s is not on the reviewed llama-server list (%w)", ErrEngineArgument, name, llamacpp.ErrForbiddenFlag)
		}
		if !strings.HasPrefix(flag, "-") {
			return nil, fmt.Errorf("%w: %q is not a llama-server option", ErrEngineArgument, given)
		}
		name := flag
		if long, ok := engineFlagNames[flag]; ok {
			name = long
		}
		rule, ok := engineFlags[name]
		if !ok {
			return nil, fmt.Errorf("%w: %s is not on the reviewed llama-server list", ErrEngineArgument, flag)
		}
		if seen[name] {
			return nil, fmt.Errorf("%w: %s is given more than once", ErrEngineArgument, name)
		}
		seen[name] = true
		if rule.value == nil {
			if inline {
				return nil, fmt.Errorf("%w: %s takes no value", ErrEngineArgument, flag)
			}
			out = append(out, name)
			continue
		}
		if !inline {
			if i+1 >= len(args) {
				return nil, fmt.Errorf("%w: %s needs a value", ErrEngineArgument, flag)
			}
			i++
			value = args[i]
		}
		if !rule.value(value) {
			return nil, fmt.Errorf("%w: %s does not take the value %q", ErrEngineArgument, flag, value)
		}
		out = append(out, name, value)
	}
	return out, nil
}
