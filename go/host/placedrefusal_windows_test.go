package host

import (
	"context"
	"os"
	"strings"
	"testing"
)

// Decision 2: the check before a video's lease says plainly when a model
// file is still being written, and when it no longer matches its download,
// with what fixes it.
func TestVideoProofSaysPlainlyWhyAModelFileFails(t *testing.T) {
	f := trustedVideoFixture(t)
	ctx := context.Background()
	if err := f.engine.present(ctx); err != nil {
		t.Fatal(err)
	}
	path := f.placed["diffusion.gguf"]
	w, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	err = f.engine.present(ctx)
	w.Close()
	if err == nil || !strings.Contains(err.Error(), "diffusion.gguf is still being written by another program") {
		t.Fatalf("a file open for writing: %v", err)
	}
	appendFile(t, path, "!")
	err = f.engine.present(ctx)
	if err == nil || !strings.Contains(err.Error(), "diffusion.gguf no longer matches its download; download it again") {
		t.Fatalf("a changed file: %v", err)
	}
}

func appendFile(t *testing.T, path, s string) {
	t.Helper()
	w, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if _, err := w.WriteString(s); err != nil {
		t.Fatal(err)
	}
}
