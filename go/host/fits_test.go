package host

import (
	"context"
	"strings"
	"testing"
)

// An engine that prints llama.cpp's allocation failure and exits reads
// engine:does_not_fit with its remedy, and gives the lease back
// (MODELHOST-L9).
func TestAnEngineThatDoesNotFitIsNamed(t *testing.T) {
	for _, line := range []string{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory",
		"alloc_tensor_range: failed to allocate Vulkan0 buffer of size 9663676416",
		"llama_init_from_model: failed to initialize the context: failed to allocate compute buffers",
		"ggml_vulkan: Device memory allocation of size 9663676416 failed. vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
	} {
		leases := newFakeLeases()
		h := composed(t, leases, func(o *Options) { o.Engine.fixture = []string{"--fake-fail-line", line} })
		_, err := h.Ensure(context.Background(), "tiny:1b")
		if WordOf(err) != EngineDoesNotFit {
			t.Fatalf("%q: %v, wanted %s", line, err, EngineDoesNotFit)
		}
		if !strings.Contains(err.Error(), "--engine-arg --n-gpu-layers N") || !strings.Contains(err.Error(), "--ctx-size") {
			t.Fatalf("the refusal names no remedy: %v", err)
		}
		if _, released, _ := leases.counts(); len(released) != 1 {
			t.Fatalf("a load that did not fit released %v, wanted the lease it took", released)
		}
	}
}

// Any other start failure stays unavailable, and Vulkan's pinned-memory
// warning, which the engine continues past, is not an allocation failure.
func TestOtherStartFailuresStayUnavailable(t *testing.T) {
	for _, line := range []string{
		"llama_model_load: error loading model: tensor data is not within the file bounds",
		"ggml_vulkan: Failed to allocate pinned memory (Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory)",
	} {
		h := composed(t, newFakeLeases(), func(o *Options) { o.Engine.fixture = []string{"--fake-fail-line", line} })
		if _, err := h.Ensure(context.Background(), "tiny:1b"); WordOf(err) != Unavailable {
			t.Fatalf("%q: %v, wanted %s", line, err, Unavailable)
		}
	}
}
