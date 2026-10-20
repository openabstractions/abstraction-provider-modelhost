//go:build windows

package host

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/openabstractions/abstraction-facade/go/supervision"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/source"
	"golang.org/x/sys/windows"
)

func sha256Of(b []byte) string {
	d := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(d[:])
}

// sectionDigest maps value as a section and hashes its first size bytes;
// empty when the value maps no section.
func sectionDigest(value uintptr, size int64) string {
	addr, err := windows.MapViewOfFile(windows.Handle(value), windows.FILE_MAP_READ, 0, 0, 0)
	if err != nil {
		return ""
	}
	defer windows.UnmapViewOfFile(addr)
	var info windows.MemoryBasicInformation
	if windows.VirtualQuery(addr, &info, unsafe.Sizeof(info)) != nil || int64(info.RegionSize) < size {
		return ""
	}
	p := *(*unsafe.Pointer)(unsafe.Pointer(&addr))
	return sha256Of(unsafe.Slice((*byte)(p), size))
}

// memoryPackage is a store root holding one zip with a 64 KiB aligned STORED
// GGUF member and a DEFLATE GGUF member of the same bytes.
func memoryPackage(t *testing.T) (Source, []byte) {
	t.Helper()
	data := make([]byte, 3<<20+777)
	copy(data, "GGUF")
	for i := 8; i < len(data); i++ {
		data[i] = byte(i*7) & 0x3f
	}
	home := t.TempDir()
	root := filepath.Join(home, "packages")
	if err := os.MkdirAll(root, 0o700); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	w := zip.NewWriter(&buf)
	start := int64(30 + len("aligned.gguf") + 6)
	pad := (65536 - start%65536) % 65536
	extra := make([]byte, 6+pad)
	binary.LittleEndian.PutUint16(extra[0:], 0xD935)
	binary.LittleEndian.PutUint16(extra[2:], uint16(2+pad))
	fw, err := w.CreateRaw(&zip.FileHeader{Name: "aligned.gguf", Method: zip.Store, CRC32: crc32.ChecksumIEEE(data),
		CompressedSize64: uint64(len(data)), UncompressedSize64: uint64(len(data)), Extra: extra})
	if err != nil {
		t.Fatal(err)
	}
	fw.Write(data)
	if fw, err = w.CreateHeader(&zip.FileHeader{Name: "packed.gguf", Method: zip.Deflate}); err != nil {
		t.Fatal(err)
	}
	fw.Write(data)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "pkg.zip"), buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	env := source.Env{GOOS: "windows", Home: home, Getenv: func(string) string { return "" }}
	env.Config.ExtraRoots = []source.ExtraRoot{{Store: "packages", Path: root}}
	return NewLocalStores(env), data
}

type releases struct{ n atomic.Int32 }

func (r *releases) release() { r.n.Add(1) }

// TestAMemoryBindingReachesOneEngine: a STORED region and a DEFLATE member,
// resolved through the storage source, each fill a section whose memory
// binding carries the object's digest. An engine stand-in started by
// supervision with one loan inherits that handle only: it reads the object
// through it and cannot reach the other section's inheritable loan. The
// section is freed and its holding released once after the engine exits
// (MODELHOST-L1; inference INF-MS1..INF-MS5; facade FAC-P1).
func TestAMemoryBindingReachesOneEngine(t *testing.T) {
	if spec := os.Getenv("OA_MODELHOST_MEMORY_CHILD"); spec != "" {
		args := map[string]string{}
		for _, kv := range strings.Split(spec, ",") {
			k, v, _ := strings.Cut(kv, "=")
			args[k] = v
		}
		number := func(k string) uint64 { n, _ := strconv.ParseUint(args[k], 10, 64); return n }
		lent := sectionDigest(uintptr(number("handle")), int64(number("size")))
		decoy := "absent"
		if sectionDigest(uintptr(number("decoy")), int64(number("size"))) == args["want"] {
			decoy = "present"
		}
		fmt.Printf("lent=%s decoy=%s\n", lent, decoy)
		os.Exit(0)
	}
	src, data := memoryPackage(t)
	digest := sha256Of(data)
	region, err := src.Resolve("packages", "pkg.zip!/aligned.gguf", bridge.Accepts{Kinds: []bridge.BindingKind{bridge.BindRegion}})
	if err != nil || region.Binding.Kind != bridge.BindRegion {
		t.Fatalf("region: %+v %v", region.Binding, err)
	}
	packed, err := src.Resolve("packages", "pkg.zip!/packed.gguf", bridge.FileOnly)
	if !errors.Is(err, bridge.ErrUnaccepted) || packed.Binding.Kind != bridge.BindCompressed {
		t.Fatalf("compressed: %+v %v", packed.Binding, err)
	}
	var regionHeld, packedHeld releases
	regionSection, regionMemory, err := fillMemory(context.Background(), region.Binding, digest, regionHeld.release)
	if err != nil {
		t.Fatal(err)
	}
	packedSection, packedMemory, err := fillMemory(context.Background(), packed.Binding, digest, packedHeld.release)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []bridge.ObjectBinding{regionMemory, packedMemory} {
		if m.Kind != bridge.BindMemory || m.Digest != digest || m.Length != int64(len(data)) || m.Handle == 0 || len(m.ChunkDigests) != 1 || m.From == nil {
			t.Fatalf("memory binding %+v", m)
		}
	}
	if regionMemory.From.Kind != bridge.BindRegion || packedMemory.From.Kind != bridge.BindCompressed {
		t.Fatalf("filled from %s and %s", regionMemory.From.Kind, packedMemory.From.Kind)
	}

	loan, err := packedSection.Lend()
	if err != nil {
		t.Fatal(err)
	}
	decoy, err := regionSection.Lend()
	if err != nil {
		t.Fatal(err)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	spec := fmt.Sprintf("handle=%d,size=%d,decoy=%d,want=%s", loan.Handle(), len(data), decoy.Handle(), digest)
	child, err := supervision.Start(context.Background(), supervision.Options{
		Executable: exe, Args: []string{"-test.run=^TestAMemoryBindingReachesOneEngine$"},
		Env: append(os.Environ(), "OA_MODELHOST_MEMORY_CHILD="+spec), Stdout: w,
		Inherit: []uintptr{loan.Handle()}, ForceTimeout: 5 * time.Second,
	})
	w.Close()
	if err != nil {
		t.Fatal(err)
	}
	loan.Started()
	exited := make(chan struct{})
	loan.EndWhen(exited)
	report, readErr := io.ReadAll(r)
	code, waitErr := child.Wait()
	close(exited)
	if readErr != nil || waitErr != nil || code != 0 {
		t.Fatalf("engine: exit %d, %v, %v", code, waitErr, readErr)
	}
	if got := strings.TrimSpace(string(report)); got != "lent="+digest+" decoy=absent" {
		t.Fatalf("the engine reported %q", got)
	}
	deadline := time.Now().Add(10 * time.Second)
	for !packedSection.Freed() || packedHeld.n.Load() != 1 {
		if time.Now().After(deadline) {
			t.Fatalf("after the engine exited: freed %v, released %d times", packedSection.Freed(), packedHeld.n.Load())
		}
		time.Sleep(5 * time.Millisecond)
	}
	if regionSection.Freed() || regionHeld.n.Load() != 0 {
		t.Fatal("the section no engine received was freed")
	}
	decoy.End()
	if !regionSection.Freed() || regionHeld.n.Load() != 1 {
		t.Fatalf("the other section: freed %v, released %d times", regionSection.Freed(), regionHeld.n.Load())
	}
}

// TestMemoryRefusesAnotherObject: a digest that is not the object's frees
// the section and releases the holding once; a binding with no bytes to
// read is refused before any section exists.
func TestMemoryRefusesAnotherObject(t *testing.T) {
	src, data := memoryPackage(t)
	packed, _ := src.Resolve("packages", "pkg.zip!/packed.gguf", bridge.FileOnly)
	var held releases
	if _, _, err := fillMemory(context.Background(), packed.Binding, sha256Of(data[1:]), held.release); err == nil || held.n.Load() != 1 {
		t.Fatalf("another digest: %v, released %d times", err, held.n.Load())
	}
	var device releases
	if _, _, err := fillMemory(context.Background(), bridge.ObjectBinding{Kind: bridge.BindDevice}, sha256Of(data), device.release); err == nil || device.n.Load() != 1 {
		t.Fatalf("a device binding: %v, released %d times", err, device.n.Load())
	}
}
