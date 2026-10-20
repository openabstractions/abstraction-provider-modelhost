package host

import (
	"bytes"
	"io"
	"regexp"
	"strings"
	"sync"
)

// EngineDoesNotFit is a llama-server that printed an allocation failure and
// then did not become ready: the file, its KV cache and its compute buffers
// together exceed what the card holds (MODELHOST-L9).
const EngineDoesNotFit = "engine:does_not_fit"

// doesNotFitRemedy is the text a person acts on after EngineDoesNotFit.
const doesNotFitRemedy = "place fewer layers on the card with --engine-arg --n-gpu-layers N, or give a smaller --ctx-size"

// allocationFailure matches, in any case, llama.cpp b10825's allocation
// failures: ggml-alloc's "failed to allocate <buffer> buffer", llama's
// "unable to allocate", "failed to allocate compute buffers", the CUDA and
// HIP "out of memory" and Vulkan's ErrorOutOfDeviceMemory.
var allocationFailure = regexp.MustCompile(`(?i)failed to allocate|unable to allocate|out of memory|outofdevicememory`)

// pinnedWarning is Vulkan's pinned-memory warning.
var pinnedWarning = regexp.MustCompile(`(?i)pinned memory`)

// fitTap passes an engine's output on and remembers whether a line named an
// allocation failure. Vulkan's "Failed to allocate pinned memory" is a
// warning the engine continues past and is not counted.
type fitTap struct {
	out     io.Writer
	mu      sync.Mutex
	partial []byte
	failed  string
}

func newFitTap(out io.Writer) *fitTap { return &fitTap{out: out} }

func (f *fitTap) Write(b []byte) (int, error) {
	f.mu.Lock()
	f.partial = append(f.partial, b...)
	for {
		i := bytes.IndexByte(f.partial, '\n')
		if i < 0 {
			break
		}
		f.note(string(f.partial[:i]))
		f.partial = f.partial[i+1:]
	}
	if len(f.partial) > 4096 {
		f.note(string(f.partial))
		f.partial = nil
	}
	f.mu.Unlock()
	if f.out == nil {
		return len(b), nil
	}
	return f.out.Write(b)
}

func (f *fitTap) note(line string) {
	if f.failed != "" {
		return
	}
	if allocationFailure.MatchString(line) && !pinnedWarning.MatchString(line) {
		f.failed = strings.TrimSpace(line)
	}
}

// failure is the first allocation-failure line the engine printed, or "".
func (f *fitTap) failure() string {
	if f == nil {
		return ""
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failed == "" && len(f.partial) > 0 {
		f.note(string(f.partial))
	}
	return f.failed
}
