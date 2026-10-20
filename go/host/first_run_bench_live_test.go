package host

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestLiveFirstRunRankOneBench measures the accepted 32K/tg128/pp2048 row
// against one already pinned weight and engine. The external runner bounds the
// whole process tree and runs plain and MTP as separate, serial processes.
func TestLiveFirstRunRankOneBench(t *testing.T) {
	if os.Getenv("OA_LIVE_FIRST_RUN_BENCH") != "1" {
		t.Skip("set OA_LIVE_FIRST_RUN_BENCH=1 through the bounded runner")
	}
	need := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is not set", name)
		}
		return value
	}
	mode := need("MODELHOST_BENCH_MODE")
	if mode != "plain" && mode != "mtp" && mode != "mtp-n2" && mode != "plain-fit" {
		t.Fatalf("unrecognized benchmark mode %q", mode)
	}
	program := need("MODELHOST_LIVE_ENGINE")
	state := need("MODELHOST_LIVE_STATE")
	model := need("MODELHOST_LIVE_MODEL")
	out := need("MODELHOST_LIVE_OUT")
	sockets := need("MODELHOST_LIVE_SOCKETS")
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	log, err := os.Create(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()

	const name = "live:qwen36-rank-one-bench"
	pin, ok := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	if !ok {
		t.Fatal("no pinned engine for this platform")
	}
	args := []string{"--ctx-size", "32768", "--ubatch-size", "512", "--parallel", "1", "--flash-attn", "on"}
	// mtp is the catalogue's MTP draft length, which
	// research/first-run/DECISION.md §2 owns: 3 since 2026-10-06. mtp-n2 is
	// the length the 2026-10-03 and first 2026-10-06 legs measured.
	if mode == "mtp" {
		args = append(args, "--spec-type", "draft-mtp", "--spec-draft-n-max", "3")
	}
	if mode == "mtp-n2" {
		args = append(args, "--spec-type", "draft-mtp", "--spec-draft-n-max", "2")
	}
	// MODELHOST_BENCH_LOG_VERBOSITY, when set, is passed as --log-verbosity so
	// the engine log shows its layer placement (b10825 prints load_tensors
	// lines from 4).
	if verbosity := os.Getenv("MODELHOST_BENCH_LOG_VERBOSITY"); verbosity != "" {
		args = append(args, "--log-verbosity", verbosity)
	}
	// plain-fit names MODELHOST-L9's card default as a person's arguments:
	// the file read unmapped and the layer count left to the engine's own
	// --fit, b10825's "auto". It measured that default on 2026-10-06.
	if mode == "plain-fit" {
		args = append(args, "--no-mmap", "--n-gpu-layers", "auto")
	}
	h, err := New(Options{
		Models: []Model{{Name: name, Store: "live", Object: "rank-one"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin),
			Arguments: args, Sockets: sockets, MemoryLimit: 48 << 30,
			Ready: 10 * time.Minute, Stderr: log},
		Source: fakeSource{objects: map[string]Object{"live/rank-one": {
			Store: "live", ID: "rank-one", Path: model, File: filepath.Base(model),
			Family: "qwen35moe", Size: info.Size()}}},
		Leases: newFakeLeases(), Idle: time.Hour,
		OnError: func(err error) { t.Logf("modelhost: %v", err) },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		h.Close()
		entries, err := os.ReadDir(sockets)
		if err != nil && !os.IsNotExist(err) {
			t.Errorf("inspect socket root after Close: %v", err)
		} else if len(entries) != 0 {
			t.Errorf("Close left %d entries in the isolated socket root", len(entries))
		}
		if err := log.Sync(); err != nil {
			t.Errorf("sync engine log: %v", err)
		}
		engineLog, err := os.ReadFile(filepath.Join(out, "engine.log"))
		if err != nil {
			t.Errorf("read engine log: %v", err)
		} else {
			if !bytes.Contains(engineLog, []byte("n_ctx_slot = 32768")) {
				t.Error("engine did not confirm the fitted 32768-token slot")
			}
			if strings.HasPrefix(mode, "mtp") && !bytes.Contains(engineLog, []byte("creating MTP draft context")) {
				t.Error("engine did not create an MTP draft context")
			}
			if mode == "plain" && bytes.Contains(engineLog, []byte("creating MTP draft context")) {
				t.Error("plain engine unexpectedly created an MTP draft context")
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 11*time.Minute)
	defer cancel()
	if _, err := h.Ensure(ctx, name); err != nil {
		t.Fatal(err)
	}
	client := EngineClient(h, func(err error) { t.Logf("engine client: %v", err) })
	base := h.Base(name)
	// Numeric token IDs avoid a chat template, BOS insertion and text
	// rendering differences between the two fresh processes.
	synthetic := strings.Repeat("Synthetic local benchmark observation 0123456789; compare the next numbered observation.\n", 600)
	var tokenized struct {
		Tokens []int64 `json:"tokens"`
	}
	benchPOST(t, ctx, client, base+"/tokenize", map[string]any{
		"content": synthetic, "add_special": false,
	}, &tokenized)
	if len(tokenized.Tokens) < 2048 {
		t.Fatalf("synthetic fixture has %d tokens, needs 2048", len(tokenized.Tokens))
	}
	prompt := tokenized.Tokens[:2048]
	encoded, err := json.Marshal(prompt)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(encoded)
	promptSHA := hex.EncodeToString(digest[:])
	var completed struct {
		StopType        string `json:"stop_type"`
		TokensCached    int64  `json:"tokens_cached"`
		TokensEvaluated int64  `json:"tokens_evaluated"`
		Timings         struct {
			CacheN             int64   `json:"cache_n"`
			PromptN            int64   `json:"prompt_n"`
			PromptMS           float64 `json:"prompt_ms"`
			PromptPerSecond    float64 `json:"prompt_per_second"`
			PredictedN         int64   `json:"predicted_n"`
			PredictedMS        float64 `json:"predicted_ms"`
			PredictedPerSecond float64 `json:"predicted_per_second"`
		} `json:"timings"`
	}
	benchPOST(t, ctx, client, base+"/completion", map[string]any{
		"prompt": prompt, "n_predict": 128, "ignore_eos": true,
		"cache_prompt": false, "seed": 42, "stream": false,
	}, &completed)
	if completed.StopType != "limit" || completed.Timings.CacheN != 0 || completed.Timings.PromptN != 2048 ||
		completed.Timings.PredictedN != 128 || completed.Timings.PredictedMS <= 0 ||
		completed.Timings.PredictedPerSecond <= 0 {
		t.Fatalf("invalid benchmark counts/timing: cache=%d prompt=%d generated=%d ms=%.3f rate=%.3f",
			completed.Timings.CacheN, completed.Timings.PromptN, completed.Timings.PredictedN,
			completed.Timings.PredictedMS, completed.Timings.PredictedPerSecond)
	}
	summary := struct {
		Mode               string  `json:"mode"`
		PromptTokenSHA256  string  `json:"prompt_token_sha256"`
		PromptN            int64   `json:"prompt_n"`
		CacheN             int64   `json:"cache_n"`
		PredictedN         int64   `json:"predicted_n"`
		PromptMS           float64 `json:"prompt_ms"`
		PromptPerSecond    float64 `json:"prompt_per_second"`
		PredictedMS        float64 `json:"predicted_ms"`
		PredictedPerSecond float64 `json:"predicted_per_second"`
		StopType           string  `json:"stop_type"`
		TokensCached       int64   `json:"tokens_cached"`
		TokensEvaluated    int64   `json:"tokens_evaluated"`
	}{mode, promptSHA, completed.Timings.PromptN, completed.Timings.CacheN,
		completed.Timings.PredictedN, completed.Timings.PromptMS,
		completed.Timings.PromptPerSecond, completed.Timings.PredictedMS,
		completed.Timings.PredictedPerSecond, completed.StopType,
		completed.TokensCached, completed.TokensEvaluated}
	raw, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "result.json"), append(raw, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: prompt=2048 cache=0 generated=128, tg=%.3f tok/s, prompt_sha256=%s",
		mode, completed.Timings.PredictedPerSecond, promptSHA)
}

func benchPOST(t *testing.T, ctx context.Context, client *http.Client, url string, value, result any) {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("%s returned HTTP %d", request.URL.Path, response.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, (4<<20)+1))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 4<<20 {
		t.Fatalf("%s response exceeds 4 MiB", request.URL.Path)
	}
	if err := json.Unmarshal(raw, result); err != nil {
		t.Fatalf("decode %s response: %v", request.URL.Path, err)
	}
}
