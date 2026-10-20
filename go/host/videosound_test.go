package host

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

// An LTX clip's sound reaches the delivery as a Vorbis track. The engine's
// result is recorded: the first two clusters of the 512x512 clip
// sd.cpp master-843's sd-server made on 2026-11-12 (VP8 frames and 16-bit
// PCM at 48 kHz stereo, DocType matroska). The stored delivery is a WebM
// whose audio track is A_VORBIS, with the engine's video frames unchanged.
func TestLTXSoundIsDeliveredAsVorbis(t *testing.T) {
	requireEnginePeer(t)
	recorded, err := filepath.Abs(filepath.Join("testdata", "ltx-512x512-sound.webm"))
	if err != nil {
		t.Fatal(err)
	}
	engine, err := os.ReadFile(recorded)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(engine, []byte("A_PCM/INT/LIT")) {
		t.Fatal("the recorded clip carries no PCM track")
	}
	f := newVideoFixture(t, "--fake-result-file="+recorded)
	h := f.host(t, newFakeLeases(), nil)
	op := startVideo(t, h, videoRequest("job-sound"))
	observed := awaitVideo(t, h, op, wire.VideoStateCompleted, wire.VideoStateFailed)
	if observed.Status != wire.VideoStateCompleted || observed.Delivery == nil {
		t.Fatalf("observed %+v", observed)
	}
	ref, ok := f.store.Find(observed.Delivery.Digest)
	if !ok || observed.Delivery.MediaType != "video/webm" {
		t.Fatalf("delivery %+v", observed.Delivery)
	}
	clip, err := os.ReadFile(f.store.Path(ref))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\x42\x82\x84webm", "\x86\x88A_VORBIS", "\x01vorbis", "\x03vorbis", "\x05vorbis", "\x86\x85V_VP8"} {
		if !bytes.Contains(clip, []byte(want)) {
			t.Fatalf("the delivered clip lacks %q", want)
		}
	}
	if bytes.Contains(clip, []byte("A_PCM")) {
		t.Fatal("the delivered clip still carries the PCM track")
	}
	// The first VP8 keyframe's bytes, from its frame tag on, are delivered
	// as the engine wrote them.
	key := bytes.Index(engine, []byte("\x9d\x01\x2a"))
	if key < 0 || !bytes.Contains(clip, engine[key-3:key+4096]) {
		t.Fatal("the engine's video frame is not in the delivered clip")
	}
}
