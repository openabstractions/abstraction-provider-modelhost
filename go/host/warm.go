package host

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Warming reads an engine's model files into the operating system's file
// cache while the engine starts (MODELHOST-L10). The pinned engines read a
// cold file far below the disk's rate: llama-server b10825 reads one 1 MiB
// piece at a time, and sd.cpp master-843 reads through std::ifstream, which
// on Windows stays near 3 GB/s however many threads read. Eight parallel
// 8 MiB reads move a cold file at the disk's rate (6.1-6.3 GB/s on the
// Strix Halo NVMe, measured 2026-11-12 with scripts/load_speed.py), and the
// engine then reads from memory.
const (
	warmWorkers = 8
	warmChunk   = 8 << 20
)

// warmShare is the part of the available physical memory warming may fill:
// files larger in total than half of it are left to the engine, so warming
// never pushes out what the engine has just read.
const warmShare = 2

// warmResult is what one warming did, for the engine log.
type warmResult struct {
	Bytes   int64
	Elapsed time.Duration
	// Skipped says why nothing was read; empty when the files were read.
	Skipped string
	Err     error
}

func (r warmResult) String() string {
	if r.Skipped != "" {
		return "skipped: " + r.Skipped
	}
	rate := 0.0
	if s := r.Elapsed.Seconds(); s > 0 {
		rate = float64(r.Bytes) / 1e9 / s
	}
	out := fmt.Sprintf("%.2f GB in %.2f s (%.2f GB/s)", float64(r.Bytes)/1e9, r.Elapsed.Seconds(), rate)
	if r.Err != nil {
		out += "; stopped: " + r.Err.Error()
	}
	return out
}

// availableMemory reports the physical memory the system can give without
// paging, 0 when unknown; tests replace it.
var availableMemory = systemAvailableMemory

// warmFiles reads each file in turn, in the order the engine needs them,
// with warmWorkers parallel reads into a discarded buffer. A read error ends
// the warming early and is reported; the engine reads the files itself in
// any case, so warming never fails a launch.
func warmFiles(ctx context.Context, paths []string) warmResult {
	var total int64
	sizes := make([]int64, len(paths))
	for i, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return warmResult{Skipped: "the file size: " + err.Error()}
		}
		sizes[i] = info.Size()
		total += sizes[i]
	}
	if total == 0 {
		return warmResult{Skipped: "no bytes"}
	}
	avail := availableMemory()
	if avail <= 0 {
		return warmResult{Skipped: "the available memory is unknown"}
	}
	if total > avail/warmShare {
		return warmResult{Skipped: fmt.Sprintf("%.2f GB of files against %.2f GB of available memory", float64(total)/1e9, float64(avail)/1e9)}
	}
	started := time.Now()
	var read int64
	for i, p := range paths {
		n, err := warmFile(ctx, p, sizes[i])
		read += n
		if err != nil {
			return warmResult{Bytes: read, Elapsed: time.Since(started), Err: err}
		}
	}
	return warmResult{Bytes: read, Elapsed: time.Since(started)}
}

// warmFile reads one file of size bytes in warmChunk pieces on warmWorkers
// goroutines, each with its own handle and buffer.
func warmFile(ctx context.Context, path string, size int64) (int64, error) {
	var next, read atomic.Int64
	var wg sync.WaitGroup
	var once sync.Once
	var first error
	fail := func(err error) { once.Do(func() { first = err }) }
	for w := 0; w < warmWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			f, err := os.Open(path)
			if err != nil {
				fail(err)
				return
			}
			//unchecked: a read-only handle; nothing written is lost on a failed close
			defer f.Close()
			buf := make([]byte, warmChunk)
			for ctx.Err() == nil {
				off := next.Add(warmChunk) - warmChunk
				if off >= size {
					return
				}
				n, err := f.ReadAt(buf, off)
				read.Add(int64(n))
				if err != nil && err != io.EOF {
					fail(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	if first == nil {
		first = ctx.Err()
	}
	return read.Load(), first
}

// warmRoleOrder is the order sd-server master-843 reads a set's files in:
// it loads each model when a request first needs it, the text encoder
// before the diffusion model and the decoders last.
var warmRoleOrder = []string{RoleLLM, RoleT5XXL, RoleDiffusionModel, RoleAudioVAE, RoleVAE, RoleTAE, RoleUpscaler}

// warmOrder is set's held files in warmRoleOrder; where maps each file name
// to its held final path.
func warmOrder(set VideoModelSet, where map[string]string) []string {
	var out []string
	for _, role := range warmRoleOrder {
		for _, f := range set.Files {
			if f.Role == role && where[f.Name] != "" {
				out = append(out, where[f.Name])
			}
		}
	}
	return out
}

// startWarming warms paths on a goroutine until done or until stop closes,
// and writes the outcome as one line to log when log is not nil.
func startWarming(paths []string, stop <-chan struct{}, log io.Writer) {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		select {
		case <-stop:
		case <-ctx.Done():
		}
		cancel()
	}()
	go func() {
		defer cancel()
		r := warmFiles(ctx, paths)
		if log != nil {
			//unchecked: the line is diagnostic; the engine log keeps its own errors
			fmt.Fprintf(log, "%s phase=warm %s\n", time.Now().UTC().Format(time.RFC3339Nano), r)
		}
	}()
}
