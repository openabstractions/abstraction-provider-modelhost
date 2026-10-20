package host

import (
	"context"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	"github.com/openabstractions/abstraction-identity/go/trustedpath"
)

// trustedVideoFixture is the video fixture with its placed files trusted in
// place, as the owner's model folder is.
func trustedVideoFixture(t *testing.T) *videoFixture {
	t.Helper()
	f := newVideoFixture(t)
	f.engine.Files = programfiles.NewVerifier(programfiles.Options{ACL: openACL{}, TrustPlaced: func(string, trustedpath.Owners) error { return nil }})
	return f
}

// MODELHOST-L11: the background re-hash runs only while the host is quiet,
// and a file whose bytes no longer match its record gets one plain line.
func TestRecheckReportsAFileThatNoLongerMatchesWhenQuiet(t *testing.T) {
	f := trustedVideoFixture(t)
	var mu sync.Mutex
	var lines []string
	h := f.host(t, newFakeLeases(), func(o *Options) {
		o.OnError = func(err error) {
			if !strings.Contains(err.Error(), "no longer matches") {
				return // the start-up lines about the state folder
			}
			mu.Lock()
			defer mu.Unlock()
			lines = append(lines, err.Error())
		}
	})
	ctx := context.Background()
	if err := CheckVideoModels(ctx, f.engine.Files, f.engine.Set, f.engine.Places, f.engine.ModelRecord); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(f.engine.ModelRecord)
	if err != nil {
		t.Fatal(err)
	}
	// Stand-in for media that changed under unchanged metadata: the record
	// is 40 days old and expects other bytes for the diffusion model.
	old := strconv.FormatInt(time.Now().Add(-40*24*time.Hour).Unix(), 10)
	record := regexp.MustCompile(`(?m)^proved\t\d+$`).ReplaceAllString(string(raw), "proved\t"+old)
	record = regexp.MustCompile(`(?m)^(.*diffusion\.gguf\t.*\t)[0-9a-f]{64}$`).ReplaceAllString(record, "${1}"+strings.Repeat("0", 64))
	if record == string(raw) || !strings.Contains(record, strings.Repeat("0", 64)) {
		t.Fatalf("record layout %q", raw)
	}
	if err := os.WriteFile(f.engine.ModelRecord, []byte(record), 0o600); err != nil {
		t.Fatal(err)
	}

	h.userIdle = func() time.Duration { return time.Minute }
	h.recheckOnce(ctx)
	if now, _ := os.ReadFile(f.engine.ModelRecord); string(now) != record || len(lines) != 0 {
		t.Fatalf("a re-hash ran while the person used the machine: %q", lines)
	}

	h.userIdle = func() time.Duration { return time.Hour }
	h.recheckOnce(ctx)
	mu.Lock()
	defer mu.Unlock()
	if len(lines) != 1 || !strings.Contains(lines[0], "diffusion.gguf no longer matches its pinned bytes") {
		t.Fatalf("lines %q", lines)
	}
}
