package host

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

// The measured run of the chat prefix preload (MODELHOST-W1..W3) on a real
// engine and model. It loads the model through this provider's holder, has
// the engine evaluate a prefix as preload@1 does after its load, and
// measures each session start from its request to its first token and its
// first answer text:
//
//	cold                the first session start on a fresh engine
//	                    (today's cost)
//	preloaded           a session start after the Local assistant's prefix
//	                    was evaluated
//	busy-after-turn     a preload after a turn, which must evaluate nothing
//	changed             another project's session start after the evaluation
//	long-preloaded      the evaluation of a prefix of at least four logical
//	                    batches, which gives the prefill rate
//	same-mid-preload    the session start a long preload prepares, sent while
//	                    the engine evaluates; it waits for the evaluation
//	changed-mid-preload another project's long session start, sent while the
//	                    engine evaluates; it cancels the evaluation
//
// It runs only with OA_LIVE_PRELOAD=1 and these variables:
//
//	MODELHOST_LIVE_ENGINE      llama-server of the pinned release, staged by
//	                           modelhostd engine install
//	MODELHOST_LIVE_STATE       the --state that install was given
//	MODELHOST_LIVE_MODEL       the GGUF to load
//	MODELHOST_LIVE_OUT         a folder for the engine log and results.json
//	MODELHOST_LIVE_SOCKETS     a short, private socket root outside packaged
//	                           app profile redirection
//	MODELHOST_LIVE_ENGINE_ARGS reviewed engine arguments, space separated
//	                           (default: --ctx-size 32768 --ubatch-size 512
//	                           --parallel 1 --flash-attn on --no-mmap)
//
// The run fails when changed-mid-preload's queue wait, its first token less
// the engine's own prompt time (timings.prompt_ms), passes the cancellation
// bound of MODELHOST-W3: one second of polling, one logical batch of 2048
// tokens at the measured prefill rate, and one checkpoint of 2.5 s. The long
// prefix is at least four batches, so an evaluation that was not cancelled
// passes the bound. Whether the engine's log names the cancelled task
// ("cancel task", a warning in b10825's server_response_reader::stop) is
// reported for the step's own part of the log and is unverified until a GPU
// run shows the line; the run does not depend on it. The lease is the tests'
// own book (fakeLeases): this measures the engine and the preload, not the
// resource service. Run it under the GPU rules: a job memory cap, a wake
// hold and the GPU event check before and after.
func TestLivePreloadFirstAnswer(t *testing.T) {
	if os.Getenv("OA_LIVE_PRELOAD") != "1" {
		t.Skip("set OA_LIVE_PRELOAD=1 and the MODELHOST_LIVE_* variables to measure the preload on a real engine")
	}
	need := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is not set", name)
		}
		return value
	}
	program, state, model, out := need("MODELHOST_LIVE_ENGINE"), need("MODELHOST_LIVE_STATE"), need("MODELHOST_LIVE_MODEL"), need("MODELHOST_LIVE_OUT")
	sockets := need("MODELHOST_LIVE_SOCKETS")
	arguments := strings.Fields(os.Getenv("MODELHOST_LIVE_ENGINE_ARGS"))
	if len(arguments) == 0 {
		arguments = strings.Fields("--ctx-size 32768 --ubatch-size 512 --parallel 1 --flash-attn on --no-mmap")
	}
	pin, _ := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	steps := measurePreload(t, Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin), Arguments: arguments,
		Sockets: sockets, MemoryLimit: 48 << 30, Ready: 10 * time.Minute}, model, out)
	byName := map[string]preloadStep{}
	for _, s := range steps {
		byName[s.Name] = s
	}
	long, mid := byName["long-preloaded"], byName["changed-mid-preload"]
	if long.PreloadTokens < 4*2048 {
		t.Errorf("the long prefix evaluated %d tokens, fewer than four logical batches", long.PreloadTokens)
	}
	if mid.PreloadOutcome != "busy" || mid.PreloadReason != "engine:used" {
		t.Errorf("changed-mid-preload: the preload answered %s %s, want busy engine:used", mid.PreloadOutcome, mid.PreloadReason)
	}
	if long.PreloadSeconds > 0 && long.PreloadTokens > 0 && mid.PromptMs > 0 {
		rate := float64(long.PreloadTokens) / long.PreloadSeconds
		bound := 1 + 2048/rate + 2.5
		t.Logf("changed-mid-preload queue wait %.2f s; bound %.2f s at %.0f tok/s; cancel line in its log: %v (unverified)",
			mid.QueueWait, bound, rate, mid.CancelLogged)
		if mid.QueueWait > bound {
			t.Errorf("changed-mid-preload waited %.2f s in the queue, past the bound %.2f s", mid.QueueWait, bound)
		}
	} else {
		t.Error("the run measured no prefill rate or no prompt time for changed-mid-preload")
	}
}

// TestLivePreloadTokenParity isolates the cache question from the longer W3
// measurement. It uses the checked-in synthetic Local session and records
// lengths and common-prefix counts only, never rendered prompt text or tokens.
// A single warm and chat then show whether the engine actually reused its KV.
func TestLivePreloadTokenParity(t *testing.T) {
	if os.Getenv("OA_LIVE_PRELOAD_DIAG") != "1" {
		t.Skip("set OA_LIVE_PRELOAD_DIAG=1 and the MODELHOST_LIVE_* variables for the bounded token diagnostic")
	}
	need := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is not set", name)
		}
		return value
	}
	program, state, model, out := need("MODELHOST_LIVE_ENGINE"), need("MODELHOST_LIVE_STATE"), need("MODELHOST_LIVE_MODEL"), need("MODELHOST_LIVE_OUT")
	sockets := need("MODELHOST_LIVE_SOCKETS")
	arguments := strings.Fields(os.Getenv("MODELHOST_LIVE_ENGINE_ARGS"))
	if len(arguments) == 0 {
		arguments = strings.Fields("--ctx-size 32768 --ubatch-size 512 --parallel 1 --flash-attn on --no-mmap")
	}
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	engineLog, err := os.Create(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer engineLog.Close()
	const name = "live:chat"
	pin, _ := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	h, err := New(Options{
		Models: []Model{{Name: name, Store: "live", Object: "model"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin), Arguments: arguments,
			Sockets: sockets, MemoryLimit: 48 << 30, Ready: 10 * time.Minute, Stderr: engineLog},
		Source: fakeSource{objects: map[string]Object{"live/model": {Store: "live", ID: "model", Path: model,
			File: filepath.Base(model), Family: "live", Size: info.Size()}}},
		Leases: newFakeLeases(), Idle: time.Hour,
		OnError: func(err error) { t.Logf("provider: %v", err) },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if _, err := h.Ensure(ctx, name); err != nil {
		t.Fatal(err)
	}
	session := readLocalSession(t)
	prefixBody := session.chatBody(t, session.System)
	fields, messages, ok := readChatBody(prefixBody)
	if !ok {
		t.Fatal("synthetic prefix is not a chat body")
	}
	prefix, err := newChatPrefix(fields, messages)
	if err != nil {
		t.Fatal(err)
	}
	warmed, err := warmTokens(ctx, h.warmClient, h.Base(name), prefix)
	if err != nil {
		t.Fatal(err)
	}
	fullBody := session.chatBody(t, session.System, user("Write a fizzbuzz function in fizzbuzz.py and add a short test."))
	var full map[string]json.RawMessage
	if err := json.Unmarshal(fullBody, &full); err != nil {
		t.Fatal(err)
	}
	full["max_tokens"] = json.RawMessage("512")
	fullBody, err = json.Marshal(full)
	if err != nil {
		t.Fatal(err)
	}
	fullTokens := liveTemplateTokens(t, ctx, h.warmClient, h.Base(name), fullBody)
	withoutSampling := make(map[string]json.RawMessage, len(full))
	for key, value := range full {
		if !unrenderedFields[key] {
			withoutSampling[key] = value
		}
	}
	plainBody, err := json.Marshal(withoutSampling)
	if err != nil {
		t.Fatal(err)
	}
	plainTokens := liveTemplateTokens(t, ctx, h.warmClient, h.Base(name), plainBody)
	if len(warmed) < minWarmTokens {
		t.Fatalf("synthetic prefix has only %d warm tokens", len(warmed))
	}
	result := map[string]any{
		"warm_tokens": len(warmed), "full_tokens": len(fullTokens),
		"full_lcp":                liveTokenPrefix(warmed, fullTokens),
		"without_sampling_tokens": len(plainTokens),
		"without_sampling_lcp":    liveTokenPrefix(warmed, plainTokens),
	}
	server := router.OpenAICompatible("modelhost-engine", h.Base(name), "/v1", h.Models())
	outcome, reason, evaluated := h.EvaluatePrefix(ctx, server, name, prefixBody)
	result["preload_outcome"], result["preload_reason"], result["evaluated_tokens"] = outcome.String(), reason, evaluated
	if outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("synthetic preload: %s %s", outcome, reason)
	}
	client := EngineClient(h, func(err error) { t.Logf("engine client: %v", err) })
	prompt, cache, promptMs := liveDiagnosticChat(t, ctx, client, h.Base(name), fullBody)
	result["chat_prompt_tokens"], result["chat_cached_tokens"], result["chat_prompt_ms"] = prompt, cache, promptMs
	// A separate fresh slot distinguishes a default cache policy from an
	// explicit cache_prompt=true request. The renderer must treat this field
	// as unrendered, so no prompt content changes between the two cycles.
	h.Unload(name, "diagnostic")
	if _, err := h.Ensure(ctx, name); err != nil {
		t.Fatal(err)
	}
	outcome, reason, evaluated = h.EvaluatePrefix(ctx, server, name, prefixBody)
	result["explicit_preload_outcome"], result["explicit_preload_reason"], result["explicit_evaluated_tokens"] = outcome.String(), reason, evaluated
	if outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("explicit-cache preload: %s %s", outcome, reason)
	}
	full["cache_prompt"] = json.RawMessage("true")
	explicitBody, err := json.Marshal(full)
	if err != nil {
		t.Fatal(err)
	}
	prompt, cache, promptMs = liveDiagnosticChat(t, ctx, client, h.Base(name), explicitBody)
	result["explicit_chat_prompt_tokens"], result["explicit_chat_cached_tokens"], result["explicit_chat_prompt_ms"] = prompt, cache, promptMs
	raw, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "diagnostic.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("synthetic token diagnostic: %s", raw)
}

func liveTemplateTokens(t *testing.T, ctx context.Context, client *http.Client, base string, body []byte) []int64 {
	t.Helper()
	var template struct {
		Prompt *string `json:"prompt"`
	}
	if err := postEngine(ctx, client, base+applyTemplatePath, body, &template); err != nil {
		t.Fatal(err)
	}
	if template.Prompt == nil {
		t.Fatal("engine rendered no prompt")
	}
	request, err := json.Marshal(map[string]any{"content": *template.Prompt, "add_special": true, "parse_special": true})
	if err != nil {
		t.Fatal(err)
	}
	var tokenized struct {
		Tokens []int64 `json:"tokens"`
	}
	if err := postEngine(ctx, client, base+tokenizePath, request, &tokenized); err != nil {
		t.Fatal(err)
	}
	return tokenized.Tokens
}

func liveTokenPrefix(a, b []int64) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func liveDiagnosticChat(t *testing.T, ctx context.Context, client *http.Client, base string, body []byte) (int64, int64, float64) {
	t.Helper()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, base+chatPath, bytes.NewReader(body))
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
		t.Fatalf("synthetic chat answered %s", response.Status)
	}
	var prompt, cache int64
	var promptMs float64
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 1<<20), 1<<24)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Usage *struct {
				Prompt  int64 `json:"prompt_tokens"`
				Details *struct {
					Cached int64 `json:"cached_tokens"`
				} `json:"prompt_tokens_details"`
			} `json:"usage"`
			Timings *struct {
				PromptMs float64 `json:"prompt_ms"`
				Cache    int64   `json:"cache_n"`
			} `json:"timings"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		if chunk.Usage != nil {
			prompt = chunk.Usage.Prompt
			if chunk.Usage.Details != nil {
				cache = chunk.Usage.Details.Cached
			}
		}
		if chunk.Timings != nil {
			promptMs, cache = chunk.Timings.PromptMs, chunk.Timings.Cache
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return prompt, cache, promptMs
}

// The measurement runs every step on the stand-in engine, so a scheduled
// GPU run never meets a broken harness.
func TestPreloadMeasurementRunsOnTheStandInEngine(t *testing.T) {
	requireEnginePeer(t)
	steps := measurePreload(t, pinnedEngine(t), weightsFile(t, 4096), t.TempDir())
	want := map[string]string{"cold": "", "preloaded": "evaluated", "busy-after-turn": "busy", "changed": "evaluated",
		"long-preloaded": "evaluated", "same-mid-preload": "evaluated", "changed-mid-preload": "evaluated"}
	for _, s := range steps {
		// The stand-in answers within the clock's resolution, so its reply is
		// proved read by its usage, not by a positive time. Its preload ends
		// before a request sent a second later arrives.
		asks := s.Name != "busy-after-turn" && s.Name != "long-preloaded"
		if s.RequestError != "" || asks && s.PromptTokens == 0 || s.PreloadOutcome != want[s.Name] {
			t.Errorf("step %+v", s)
		}
	}
}

// preloadStep is one measured step.
type preloadStep struct {
	Name           string  `json:"name"`
	LoadSeconds    float64 `json:"load_s,omitempty"`
	PreloadSeconds float64 `json:"preload_s,omitempty"`
	PreloadTokens  int64   `json:"preload_tokens,omitempty"`
	PreloadOutcome string  `json:"preload_outcome,omitempty"`
	PreloadReason  string  `json:"preload_reason,omitempty"`
	FirstToken     float64 `json:"first_token_s,omitempty"`
	FirstAnswer    float64 `json:"first_answer_s,omitempty"`
	PromptTokens   int64   `json:"prompt_tokens,omitempty"`
	CachedTokens   int64   `json:"cached_tokens,omitempty"`
	// PromptMs is the engine's own prompt time (timings.prompt_ms), and
	// QueueWait the first token's time less it: the time the request waited
	// before the engine began it.
	PromptMs     float64 `json:"prompt_ms,omitempty"`
	QueueWait    float64 `json:"queue_wait_s,omitempty"`
	Timings      any     `json:"timings,omitempty"`
	CancelLogged bool    `json:"cancel_logged_unverified,omitempty"`
	RequestError string  `json:"request_error,omitempty"`
}

// longSystem is system followed by an example project's notes until the
// prefix is at least four logical batches: about 40,000 characters, over
// 8,192 tokens in any tokenizer of this decade.
func longSystem(system string) string {
	var b strings.Builder
	b.WriteString(system)
	for i := 0; b.Len() < 40000; i++ {
		b.WriteString("\n\n## Module notes ")
		b.WriteString(strings.Repeat("x", i%7+1))
		b.WriteString("\n\nEach module keeps its tests beside it, names its public functions after what they return, ")
		b.WriteString("reads files as UTF-8 and reports a failure with the path it could not read. Counting ")
		b.WriteString("splits on Unicode whitespace, lines end at LF or CRLF, and a table prints right-aligned numbers.")
	}
	return b.String()
}

// measurePreload runs the steps and returns them.
func measurePreload(t *testing.T, engine Engine, model, out string) []preloadStep {
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(out, "engine.log")
	engineLog, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	defer engineLog.Close()
	engine.Stderr = engineLog
	const name = "live:chat"
	h, err := New(Options{
		Models: []Model{{Name: name, Store: "live", Object: "model"}},
		Engine: engine,
		Source: fakeSource{objects: map[string]Object{"live/model": {Store: "live", ID: "model", Path: model,
			File: filepath.Base(model), Family: "live", Size: info.Size()}}},
		Leases:  newFakeLeases(),
		Idle:    time.Hour,
		OnError: func(err error) { t.Logf("provider: %v", err) },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	session := readLocalSession(t)
	moved := func(system string) string {
		return strings.ReplaceAll(system, `C:\Users\person\projects\wordstat`, `C:\Users\person\projects\other`)
	}
	long := longSystem(session.System[0])
	server := router.OpenAICompatible("modelhost-engine", h.Base(name), "/v1", h.Models())
	client := EngineClient(h, func(err error) { t.Logf("engine client: %v", err) })
	logSize := func() int64 {
		engineLog.Sync()
		info, err := os.Stat(logPath)
		if err != nil {
			return 0
		}
		return info.Size()
	}
	logSince := func(offset int64) string {
		engineLog.Sync()
		raw, err := os.ReadFile(logPath)
		if err != nil || int64(len(raw)) < offset {
			return ""
		}
		return string(raw[offset:])
	}
	ask := func(s preloadStep, system, words string) preloadStep {
		body := session.chatBody(t, []string{system}, user(words))
		// Keep nested tool schemas as the inference provider emits them.
		// Decoding into map[string]any and remarshalling would reorder their
		// keys, giving the warm and the chat different token prefixes.
		var raw map[string]json.RawMessage
		json.Unmarshal(body, &raw)
		raw["max_tokens"] = json.RawMessage("512")
		body, _ = json.Marshal(raw)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		request, _ := http.NewRequestWithContext(ctx, http.MethodPost, h.Base(name)+chatPath, bytes.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		began := time.Now()
		response, err := client.Do(request)
		if err != nil {
			s.RequestError = err.Error()
			return s
		}
		defer response.Body.Close()
		lines := bufio.NewScanner(response.Body)
		lines.Buffer(make([]byte, 1<<20), 1<<24)
		for lines.Scan() {
			data, ok := strings.CutPrefix(lines.Text(), "data: ")
			if !ok || data == "[DONE]" {
				continue
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content   string `json:"content"`
						Reasoning string `json:"reasoning_content"`
					} `json:"delta"`
				} `json:"choices"`
				Usage *struct {
					Prompt  int64 `json:"prompt_tokens"`
					Details *struct {
						Cached int64 `json:"cached_tokens"`
					} `json:"prompt_tokens_details"`
				} `json:"usage"`
				Timings json.RawMessage `json:"timings"`
			}
			if json.Unmarshal([]byte(data), &chunk) != nil {
				continue
			}
			for _, choice := range chunk.Choices {
				if s.FirstToken == 0 && (choice.Delta.Content != "" || choice.Delta.Reasoning != "") {
					s.FirstToken = time.Since(began).Seconds()
				}
				if s.FirstAnswer == 0 && choice.Delta.Content != "" {
					s.FirstAnswer = time.Since(began).Seconds()
				}
			}
			if chunk.Usage != nil {
				s.PromptTokens = chunk.Usage.Prompt
				if chunk.Usage.Details != nil {
					s.CachedTokens = chunk.Usage.Details.Cached
				}
			}
			if len(chunk.Timings) > 0 && string(chunk.Timings) != "null" {
				var timings struct {
					PromptMs float64 `json:"prompt_ms"`
				}
				json.Unmarshal(chunk.Timings, &timings)
				s.Timings, s.PromptMs = json.RawMessage(chunk.Timings), timings.PromptMs
			}
		}
		if response.StatusCode != http.StatusOK {
			s.RequestError = response.Status
		}
		if s.FirstToken > 0 && s.PromptMs > 0 {
			s.QueueWait = s.FirstToken - s.PromptMs/1000
		}
		return s
	}
	// load is a fresh engine of the model, loaded as a call loads it.
	load := func(s preloadStep) preloadStep {
		h.Unload(name, "measurement")
		began := time.Now()
		if _, err := h.Ensure(context.Background(), name); err != nil {
			t.Fatal(err)
		}
		s.LoadSeconds = time.Since(began).Seconds()
		return s
	}
	preload := func(s preloadStep, system string) preloadStep {
		began := time.Now()
		outcome, reason, tokens := h.EvaluatePrefix(context.Background(), server, name, session.chatBody(t, []string{system}))
		s.PreloadSeconds, s.PreloadOutcome, s.PreloadReason, s.PreloadTokens = time.Since(began).Seconds(), outcome.String(), reason, tokens
		return s
	}
	midPreload := func(s preloadStep, prefix, system string) preloadStep {
		s = load(s)
		offset := logSize()
		done := make(chan preloadStep, 1)
		go func() { done <- preload(preloadStep{}, prefix) }()
		time.Sleep(time.Second)
		s = ask(s, system, "What does this project do?")
		ended := <-done
		s.PreloadSeconds, s.PreloadOutcome, s.PreloadReason, s.PreloadTokens = ended.PreloadSeconds, ended.PreloadOutcome, ended.PreloadReason, ended.PreloadTokens
		s.CancelLogged = strings.Contains(logSince(offset), "cancel task")
		return s
	}

	var steps []preloadStep
	steps = append(steps, ask(load(preloadStep{Name: "cold"}), session.System[0], "hi"))
	steps = append(steps, ask(preload(load(preloadStep{Name: "preloaded"}), session.System[0]), session.System[0],
		"Write a fizzbuzz function in fizzbuzz.py and add a short test."))
	steps = append(steps, preload(preloadStep{Name: "busy-after-turn"}, session.System[0]))
	steps = append(steps, ask(preload(load(preloadStep{Name: "changed"}), session.System[0]), moved(session.System[0]), "hi"))
	steps = append(steps, preload(load(preloadStep{Name: "long-preloaded"}), long))
	steps = append(steps, midPreload(preloadStep{Name: "same-mid-preload"}, long, long))
	steps = append(steps, midPreload(preloadStep{Name: "changed-mid-preload"}, long, moved(long)))
	if steps[2].PreloadOutcome != wire.PreloadOutcomeBusy.String() {
		t.Errorf("a preload after a turn answered %s %s", steps[2].PreloadOutcome, steps[2].PreloadReason)
	}

	raw, _ := json.MarshalIndent(steps, "", "  ")
	if err := os.WriteFile(filepath.Join(out, "results.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("%-19s %7s %9s %8s %-10s %9s %9s %8s %8s %9s", "step", "load s", "preload s", "tokens", "outcome", "1st tok s", "answer s", "prompt", "cached", "queue s")
	for _, s := range steps {
		t.Logf("%-19s %7.2f %9.2f %8d %-10s %9.2f %9.2f %8d %8d %9.2f %s%s", s.Name, s.LoadSeconds, s.PreloadSeconds, s.PreloadTokens,
			s.PreloadOutcome, s.FirstToken, s.FirstAnswer, s.PromptTokens, s.CachedTokens, s.QueueWait, s.PreloadReason, s.RequestError)
	}
	t.Logf("results: %s", filepath.Join(out, "results.json"))
	return steps
}
