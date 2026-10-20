package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openabstractions/abstraction-identity/go/privatedir"
	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	acceptance "github.com/openabstractions/abstraction-job/go/abstraction/job/acceptance"
	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
	"github.com/openabstractions/abstraction-resource/go/gpufault"
	router "github.com/openabstractions/abstraction-router/go"
	content "github.com/openabstractions/abstraction-storage/go/abstraction/storage/content"
)

func cacheKnowledgeRequest(ctx context.Context, base, path string, body any) (*http.Request, error) {
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+path, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

func cacheKnowledgeSlotPath(action string) string { return "/slots/0?action=" + action }

func cacheKnowledgeCompletion(prompt []any) map[string]any {
	return map[string]any{"prompt": prompt, "id_slot": 0, "cache_prompt": true, "n_predict": 1, "temperature": 0, "seed": 1}
}

func cacheKnowledgeProductionCompletion(prompt []any, predict int) map[string]any {
	body := cacheKnowledgeCompletion(prompt)
	delete(body, "id_slot")
	body["n_predict"] = predict
	return body
}

// Join both initial requests before advancing the main conversation. No testing
// callbacks run in these workers; HTTP cancellation retains the shared context.
func cacheKnowledgeProductionSequence(tokens []any, complete func([]any, int) (map[string]any, error)) (map[string]map[string]any, error) {
	type answer struct {
		label string
		value map[string]any
		err   error
	}
	done := make(chan answer, 2)
	for _, request := range []struct {
		label   string
		prompt  []any
		predict int
	}{
		{"cold", tokens[:2048], 128}, {"concurrent_title", tokens[6000:6550], 32},
	} {
		go func() {
			value, err := complete(request.prompt, request.predict)
			done <- answer{request.label, value, err}
		}()
	}
	values := map[string]map[string]any{}
	var failures []error
	for range 2 {
		a := <-done
		values[a.label] = a.value
		if a.err != nil {
			failures = append(failures, a.err)
		}
	}
	if len(failures) != 0 {
		return values, errors.Join(failures...)
	}
	for _, request := range []struct {
		label   string
		prompt  []any
		predict int
	}{
		{"turn_two", tokens[:2112], 128}, {"idle_title", tokens[6000:6550], 32}, {"after_title", tokens[:2176], 128},
	} {
		value, err := complete(request.prompt, request.predict)
		if err != nil {
			return values, err
		}
		values[request.label] = value
	}
	return values, nil
}

func cacheKnowledgeTurnTwo(timings map[string]any) error {
	if timings["cache_n"].(float64) < 2048-8 || timings["prompt_n"].(float64) > 64+8 {
		return fmt.Errorf("turn_two checkpoint reuse cache_n=%v prompt_n=%v; want >=2040 and <=72", timings["cache_n"], timings["prompt_n"])
	}
	n, ms := timings["predicted_n"].(float64), timings["predicted_ms"].(float64)
	if n < 64 || ms <= 0 || n*1000/ms < 40 {
		return fmt.Errorf("turn_two insufficient decode n=%v ms=%v; want >=40 tokens/s", n, ms)
	}
	return nil
}

func cacheKnowledgeConcurrentSlots(raw string, mainResponse, titleResponse map[string]any) (int, int, error) {
	pattern := regexp.MustCompile(`print_timing: id\s+(\d+)\s+\| task\s+(\d+)\s+\| prompt eval time =.* /\s*(\d+) tokens`)
	type timing struct{ slot, task int }
	slots := map[int]timing{}
	for _, match := range pattern.FindAllStringSubmatch(raw, -1) {
		slot, _ := strconv.Atoi(match[1])
		task, _ := strconv.Atoi(match[2])
		count, _ := strconv.Atoi(match[3])
		if _, found := slots[count]; !found {
			slots[count] = timing{slot, task}
		}
	}
	mainTimings, mainErr := cacheKnowledgeTimings(mainResponse)
	titleTimings, titleErr := cacheKnowledgeTimings(titleResponse)
	if mainErr != nil || titleErr != nil {
		return 0, 0, errors.New("concurrent response timing diagnostics unavailable")
	}
	mainCount, titleCount := int(mainTimings["prompt_n"].(float64)), int(titleTimings["prompt_n"].(float64))
	main, mainOK := slots[mainCount]
	title, titleOK := slots[titleCount]
	mainSlot, mainResponseOK := mainResponse["id_slot"].(float64)
	titleSlot, titleResponseOK := titleResponse["id_slot"].(float64)
	if mainCount <= titleCount || !mainOK || !titleOK || main.slot == title.slot || main.task == title.task || !mainResponseOK || !titleResponseOK || mainSlot != float64(main.slot) || titleSlot != float64(title.slot) {
		return main.slot, title.slot, errors.New("concurrent main/title print_timing task/token/response slots missing, mismatched or equal")
	}
	return main.slot, title.slot, nil
}

func TestCacheKnowledgeProductionSequence(t *testing.T) {
	tokens := make([]any, 6550)
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	var mu sync.Mutex
	var lengths []int
	done := make(chan error, 1)
	go func() {
		_, err := cacheKnowledgeProductionSequence(tokens, func(prompt []any, predict int) (map[string]any, error) {
			mu.Lock()
			lengths = append(lengths, len(prompt))
			first := len(lengths) <= 2
			mu.Unlock()
			if first {
				entered <- struct{}{}
				<-release
			}
			if _, forced := cacheKnowledgeProductionCompletion(prompt, predict)["id_slot"]; forced {
				return nil, errors.New("forced slot")
			}
			return map[string]any{}, nil
		})
		done <- err
	}()
	<-entered
	<-entered
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(lengths[2:], []int{2112, 550, 2176}) || lengths[0]+lengths[1] != 2598 {
		t.Fatal(lengths)
	}
	valid := map[string]any{"cache_n": float64(2040), "prompt_n": float64(72), "predicted_n": float64(128), "predicted_ms": float64(3200)}
	if err := cacheKnowledgeTurnTwo(valid); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]float64{"cache_n": 2039, "prompt_n": 73, "predicted_n": 63, "predicted_ms": 3201} {
		bad := make(map[string]any)
		for k, v := range valid {
			bad[k] = v
		}
		bad[key] = value
		if cacheKnowledgeTurnTwo(bad) == nil {
			t.Fatalf("accepted %s=%v", key, value)
		}
	}
	log := "slot print_timing: id  3 | task 2 | prompt eval time = 10 ms / 550 tokens\nslot print_timing: id  2 | task 1 | prompt eval time = 12 ms / 2048 tokens\nslot print_timing: id  2 | task 1 | eval time = 4 ms / 128 tokens\nslot print_timing: id  0 | task 9 | prompt eval time = 10 ms / 550 tokens"
	mainResponse, titleResponse := map[string]any{"id_slot": float64(2), "timings": map[string]any{"prompt_n": float64(2048), "cache_n": float64(0), "predicted_n": float64(128), "prompt_ms": float64(10), "predicted_ms": float64(1000)}}, map[string]any{"id_slot": float64(3), "timings": map[string]any{"prompt_n": float64(550), "cache_n": float64(0), "predicted_n": float64(32), "prompt_ms": float64(10), "predicted_ms": float64(500)}}
	if _, _, err := cacheKnowledgeConcurrentSlots(log, mainResponse, titleResponse); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{"", strings.ReplaceAll(log, "id  3", "id  2")} {
		if _, _, err := cacheKnowledgeConcurrentSlots(bad, mainResponse, titleResponse); err == nil {
			t.Fatal("accepted missing/equal slots")
		}
	}
	if _, _, err := cacheKnowledgeConcurrentSlots(log, titleResponse, mainResponse); err == nil {
		t.Fatal("accepted swapped concurrent responses")
	}
}

func cacheKnowledgeTokenize(content string) map[string]any {
	return map[string]any{"content": content, "add_special": false}
}

func cacheKnowledgeTemplate(kwarg string, preserve bool) map[string]any {
	body := map[string]any{"messages": []map[string]string{{"role": "user", "content": "one"}, {"role": "assistant", "content": "<think>cache-probe-reasoning-marker</think>answer"}, {"role": "user", "content": "two"}}}
	if kwarg != "" {
		body["chat_template_kwargs"] = map[string]bool{kwarg: preserve}
	}
	return body
}

func cacheKnowledgeFilenames(out string) []string {
	return []string{"../escape.bin", `..\escape.bin`, filepath.Join(out, "absolute-escape.bin"), "sub/escape.bin", "sub\\escape.bin"}
}

// The live probe shares this request builder with the existing inert client
// fixture. All its paths pass media admission and receive the launch key;
// the unkeyed probe explicitly uses the raw client's documented host.
func TestCacheKnowledgeClientAdmission(t *testing.T) {
	type call struct {
		path string
		body any
		want string
	}
	calls := []call{
		{cacheKnowledgeSlotPath("erase"), map[string]any{}, `{}`},
		{tokenizePath, cacheKnowledgeTokenize("synthetic"), `{"content":"synthetic","add_special":false}`},
		{completionPath, cacheKnowledgeCompletion([]any{1, 2, 3, 4}), `{"prompt":[1,2,3,4],"id_slot":0,"cache_prompt":true,"n_predict":1,"temperature":0,"seed":1}`},
		{completionPath, cacheKnowledgeProductionCompletion([]any{1, 2, 3, 4}, 128), `{"prompt":[1,2,3,4],"cache_prompt":true,"n_predict":128,"temperature":0,"seed":1}`},
	}
	for _, action := range []string{"save", "restore"} {
		for _, filename := range append(cacheKnowledgeFilenames(t.TempDir()), "synthetic.bin") {
			calls = append(calls, call{cacheKnowledgeSlotPath(action), map[string]any{"filename": filename}, `{"filename":` + strconv.Quote(filename) + `}`})
		}
	}
	const messages = `"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"<think>cache-probe-reasoning-marker</think>answer"},{"role":"user","content":"two"}]`
	calls = append(calls, call{applyTemplatePath, cacheKnowledgeTemplate("", false), `{` + messages + `}`})
	for _, kwarg := range []string{"preserve_thinking", "preserve_reasoning", "enable_thinking"} {
		for _, preserve := range []bool{false, true} {
			calls = append(calls, call{applyTemplatePath, cacheKnowledgeTemplate(kwarg, preserve), `{` + messages + `,"chat_template_kwargs":{` + strconv.Quote(kwarg) + `:` + strconv.FormatBool(preserve) + `}}`})
		}
	}
	for index, call := range calls {
		t.Run(fmt.Sprintf("%d%s", index, call.path), func(t *testing.T) {
			seen := false
			client := engineClient(warmEngines{newWarmer()}, nil, inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen = true
				if r.URL.Host != "engine0.modelhost.invalid" || r.URL.RequestURI() != call.path || r.Method != http.MethodPost {
					t.Errorf("unexpected request %s %s", r.Method, r.URL)
				}
				if r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Authorization") != "Bearer key" {
					t.Error("request omitted its JSON content type or provider launch key")
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				var want map[string]any
				if err := json.Unmarshal([]byte(call.want), &want); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(body, want) {
					t.Errorf("request body = %#v; want %#v", body, want)
				}
				fmt.Fprint(w, "{}")
			})})
			req, err := cacheKnowledgeRequest(context.Background(), inProcessBase, call.path, call.body)
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			if !seen || response.StatusCode != http.StatusOK {
				t.Fatalf("guarded request did not reach the fixture: seen=%v status=%d", seen, response.StatusCode)
			}
		})
	}
	raw, err := cacheKnowledgeRequest(context.Background(), "http://"+enginehttp.UnixHost, cacheKnowledgeSlotPath("erase"), map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if raw.URL.Host != enginehttp.UnixHost || raw.Header.Get("Authorization") != "" {
		t.Fatal("unkeyed probe must use the raw socket client's engine host and omit its launch key")
	}
	seen := false
	client := &http.Client{Transport: inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = true
		if r.URL.Host != enginehttp.UnixHost || r.URL.RequestURI() != "/slots/0?action=erase" || r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Authorization") != "" {
			t.Error("raw request host, route or headers differ")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil || string(body) != "{}" {
			t.Errorf("raw erase body = %q, %v; want {}", body, err)
		}
		w.WriteHeader(http.StatusUnauthorized)
	})}}
	response, err := client.Do(raw)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if !seen || response.StatusCode != http.StatusUnauthorized {
		t.Fatal("unkeyed request did not reach the inert raw transport's authentication boundary")
	}
}

func cacheKnowledgeTimings(value map[string]any) (map[string]any, error) {
	timings, ok := value["timings"].(map[string]any)
	if !ok || len(timings) == 0 {
		return nil, errors.New("missing timings object")
	}
	for _, key := range []string{"prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms"} {
		n, ok := timings[key].(float64)
		if !ok || n < 0 || math.IsNaN(n) || math.IsInf(n, 0) || strings.HasSuffix(key, "_n") && n != math.Trunc(n) {
			return nil, fmt.Errorf("invalid timing %s", key)
		}
	}
	if timings["predicted_n"].(float64) == 0 || timings["prompt_n"].(float64)+timings["cache_n"].(float64) == 0 {
		return nil, errors.New("timings contain no completed prompt or generated token")
	}
	return timings, nil
}

func cacheKnowledgeRendered(value map[string]any) (string, error) {
	prompt, ok := value["prompt"].(string)
	if !ok || strings.TrimSpace(prompt) == "" {
		return "", errors.New("missing nonempty rendered prompt")
	}
	return prompt, nil
}

type cacheKnowledgeMonitor struct {
	mu     sync.Mutex
	errors []string
}

func (m *cacheKnowledgeMonitor) fail(reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errors = append(m.errors, reason)
}

func (m *cacheKnowledgeMonitor) outcome(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	problems := append([]string(nil), m.errors...)
	if err := ctx.Err(); err != nil {
		problems = append(problems, err.Error())
	}
	if len(problems) > 0 {
		return errors.New(strings.Join(problems, "; "))
	}
	return nil
}

func (m *cacheKnowledgeMonitor) watch(ctx context.Context, cancelWork context.CancelFunc, watch func(context.Context, func(gpufault.Event)) error) func() {
	watchCtx, cancelWatch := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		err := watch(watchCtx, func(event gpufault.Event) {
			if event.Kind == gpufault.Reset {
				m.fail("GPU reset: " + event.Source)
				cancelWork()
			}
		})
		if err == nil && watchCtx.Err() == nil {
			err = errors.New("watch ended before shutdown")
		}
		if err != nil && (watchCtx.Err() == nil || !errors.Is(err, context.Canceled)) {
			m.fail("GPU records unavailable: " + err.Error())
			cancelWork()
		}
	}()
	return func() { cancelWatch(); <-done }
}

func TestCacheKnowledgeMeasurementValidity(t *testing.T) {
	valid := map[string]any{"prompt_n": float64(1), "cache_n": float64(1024), "predicted_n": float64(1), "prompt_ms": float64(2), "predicted_ms": float64(3)}
	if _, err := cacheKnowledgeTimings(map[string]any{"timings": valid}); err != nil {
		t.Fatal(err)
	}
	for _, value := range []map[string]any{nil, {"timings": nil}, {"timings": "unknown"}, {"timings": map[string]any{}}, {"timings": map[string]any{"prompt_n": "1"}}} {
		if _, err := cacheKnowledgeTimings(value); err == nil {
			t.Errorf("malformed timing reply passed: %#v", value)
		}
	}
	for _, key := range []string{"prompt_n", "cache_n", "predicted_n", "prompt_ms", "predicted_ms"} {
		for _, bad := range []any{nil, "1", float64(-1), math.NaN(), math.Inf(1)} {
			changed := make(map[string]any, len(valid))
			for k, v := range valid {
				changed[k] = v
			}
			changed[key] = bad
			if _, err := cacheKnowledgeTimings(map[string]any{"timings": changed}); err == nil {
				t.Errorf("invalid %s=%v passed", key, bad)
			}
		}
	}
	for _, value := range []map[string]any{nil, {"prompt": nil}, {"prompt": false}, {"prompt": ""}, {"prompt": " \n"}} {
		if _, err := cacheKnowledgeRendered(value); err == nil {
			t.Errorf("malformed template reply passed: %#v", value)
		}
	}
	if got, err := cacheKnowledgeRendered(map[string]any{"prompt": "synthetic"}); err != nil || got != "synthetic" {
		t.Fatalf("valid template reply = %q, %v", got, err)
	}
}

func cacheKnowledgeMemoryLimit(raw string) (uint64, error) {
	if raw == "" {
		return 24 << 30, nil
	}
	n, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || n != 24<<30 && n != 32<<30 {
		return 0, errors.New("live fixture needs its exact admitted 24 or 32 GiB limit")
	}
	return n, nil
}

func TestCacheKnowledgeMemoryLimit(t *testing.T) {
	for raw, want := range map[string]uint64{"": 24 << 30, "25769803776": 24 << 30, "34359738368": 32 << 30} {
		if got, err := cacheKnowledgeMemoryLimit(raw); err != nil || got != want {
			t.Fatalf("%q: %d %v", raw, got, err)
		}
	}
	for _, raw := range []string{"0", "-1", "24GiB", "34359738369", "68719476736"} {
		if _, err := cacheKnowledgeMemoryLimit(raw); err == nil {
			t.Errorf("unadmitted fixture cap %q", raw)
		}
	}
}

func cacheKnowledgeEngineOutcome(exit EngineExit, admitted uint64) error {
	if exit.ProcessMemoryLimit != admitted {
		return fmt.Errorf("engine limit %d differs from admitted %d", exit.ProcessMemoryLimit, admitted)
	}
	if exit.MemoryLimitHit {
		return errors.New("engine memory limit was hit")
	}
	if exit.PeakProcessMemory == 0 || exit.PeakJobMemory == 0 {
		return errors.New("engine peak memory accounting unavailable")
	}
	return nil
}

func TestCacheKnowledgeEngineOutcome(t *testing.T) {
	for _, limit := range []uint64{24 << 30, 32 << 30} {
		var clean EngineExit
		clean.ProcessMemoryLimit = limit
		clean.PeakProcessMemory, clean.PeakJobMemory = 1, 1
		if err := cacheKnowledgeEngineOutcome(clean, limit); err != nil {
			t.Fatal(err)
		}
		hit := clean
		hit.MemoryLimitHit = true
		if err := cacheKnowledgeEngineOutcome(hit, limit); err == nil {
			t.Error("completed requests hid an engine cap hit")
		}
		wrong := clean
		wrong.ProcessMemoryLimit++
		if err := cacheKnowledgeEngineOutcome(wrong, limit); err == nil {
			t.Error("unadmitted engine cap qualified")
		}
		missing := clean
		missing.PeakJobMemory = 0
		if err := cacheKnowledgeEngineOutcome(missing, limit); err == nil {
			t.Error("missing peak accounting qualified")
		}
	}
}

func cacheKnowledgePrivateCommit(ctx context.Context, pid int) (uint64, error) {
	if runtime.GOOS != "windows" || pid <= 0 {
		return 0, errors.New("private commit needs this Windows fixture's owned PID")
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	// The stock OS process reader samples only the existing owned engine PID.
	// No profile, module, GPU code or process-control operation is involved.
	program := filepath.Join(os.Getenv("SystemRoot"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
	command := "[System.Diagnostics.Process]::GetProcessById(" + strconv.Itoa(pid) + ").PrivateMemorySize64.ToString([System.Globalization.CultureInfo]::InvariantCulture)"
	raw, err := exec.CommandContext(ctx, program, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command).Output()
	if err != nil {
		return 0, fmt.Errorf("owned process private commit: %w", err)
	}
	n, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil || n == 0 || n > math.MaxInt64 {
		return 0, fmt.Errorf("invalid private commit %q", raw)
	}
	return n, nil
}

func TestCacheKnowledgePrivateCommit(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows live fixture measurement")
	}
	if n, err := cacheKnowledgePrivateCommit(context.Background(), os.Getpid()); err != nil || n == 0 {
		t.Fatalf("owned current PID: %d %v", n, err)
	}
	if _, err := cacheKnowledgePrivateCommit(context.Background(), 0); err == nil {
		t.Error("unbound PID accepted")
	}
}

type cacheKnowledgeAbortWriter struct {
	tap  *fitTap
	stop func(string)
	once sync.Once
}

func (w *cacheKnowledgeAbortWriter) Write(p []byte) (int, error) {
	n, err := w.tap.Write(p)
	if line := w.tap.failure(); line != "" {
		w.once.Do(func() { w.stop(line) })
	}
	return n, err
}

func TestCacheKnowledgeAbortOnOOM(t *testing.T) {
	var log bytes.Buffer
	stops := 0
	w := &cacheKnowledgeAbortWriter{tap: newFitTap(&log), stop: func(string) { stops++ }}
	w.Write([]byte("failed to allocate pinned memory\n"))
	if stops != 0 {
		t.Fatal("ordinary pinned-memory warning aborted")
	}
	w.Write([]byte("ggml_vulkan: ErrorOutOfDeviceMemory\n"))
	w.Write([]byte("failed to allocate compute buffers\n"))
	if stops != 1 || !strings.Contains(log.String(), "ErrorOutOfDeviceMemory") {
		t.Fatalf("first OOM stops=%d log=%s", stops, log.String())
	}
}

func TestCacheKnowledgeMonitorOutcome(t *testing.T) {
	for _, fault := range []string{"reset", "unavailable", "clean"} {
		t.Run(fault, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m := &cacheKnowledgeMonitor{}
			afterFinalRequest := make(chan struct{})
			stop := m.watch(ctx, cancel, func(watchCtx context.Context, deliver func(gpufault.Event)) error {
				<-afterFinalRequest
				if fault == "unavailable" {
					return errors.New("fixture watch unavailable")
				}
				if fault == "reset" {
					deliver(gpufault.Event{Kind: gpufault.Reset, Source: "late fixture reset"})
				}
				<-watchCtx.Done()
				return watchCtx.Err()
			})
			if err := m.outcome(ctx); err != nil {
				t.Fatal(err)
			}
			close(afterFinalRequest)
			stop()
			if err := m.outcome(ctx); (err != nil) != (fault != "clean") {
				t.Fatalf("monitor outcome after final request = %v for %s", err, fault)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := (&cacheKnowledgeMonitor{}).outcome(ctx); err == nil {
		t.Fatal("final work cancellation passed")
	}
}

// TestLiveCacheKnowledge uses the existing pinned-engine live fixture owner.
// It writes only synthetic KV in a disposable slot directory, removed before
// return. No provider default is selected by this opt-in knowledge probe.
func TestLiveCacheKnowledge(t *testing.T) {
	if os.Getenv("OA_LIVE_CACHE_KNOWLEDGE") != "1" {
		t.Skip("set OA_LIVE_CACHE_KNOWLEDGE=1 through the bounded source owner")
	}
	if os.Getenv("OA_LIVE_CACHE_PRODUCTION") == "1" {
		liveProductionCacheKnowledge(t, false)
		return
	}
	need := func(key string) string {
		value := os.Getenv(key)
		if value == "" {
			t.Fatalf("%s is not set", key)
		}
		return value
	}
	program, state, model := need("MODELHOST_LIVE_ENGINE"), need("MODELHOST_LIVE_STATE"), need("MODELHOST_LIVE_MODEL")
	out, sockets := need("MODELHOST_LIVE_OUT"), need("MODELHOST_LIVE_SOCKETS")
	if !filepath.IsAbs(out) || !filepath.IsAbs(sockets) {
		t.Fatal("live output and socket roots must be absolute fixture paths")
	}
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	slots := filepath.Join(out, "synthetic-slots")
	if err := privatedir.Make(slots); err != nil {
		t.Fatal(err)
	}
	defer func() {
		// slots is the newly created, fixed child of the admitted absolute
		// output fixture. It contains only this probe's synthetic KV.
		if err := os.RemoveAll(slots); err != nil {
			t.Error(err)
		}
	}()
	info, err := os.Stat(model)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 22853663008 {
		t.Fatalf("model has %d bytes; want the exact admitted weight", info.Size())
	}
	log, err := os.Create(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	pin, _ := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	const name = "live:cache-knowledge"
	h, err := newTestHost(Options{
		Models: []Model{{Name: name, Store: "live", Object: "model"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin),
			Arguments: []string{"--ctx-size", "8192", "--ubatch-size", "512", "--parallel", "1", "--flash-attn", "on", "--no-mmap", "--log-verbosity", "4"},
			Sockets:   sockets, MemoryLimit: 24 << 30, Ready: 90 * time.Second, Stderr: log},
		Source: fakeSource{objects: map[string]Object{"live/model": {Store: "live", ID: "model", Path: model, File: filepath.Base(model), Family: "qwen35moe", Size: info.Size()}}},
		Leases: newFakeLeases(), Idle: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		h.Close()
		entries, err := os.ReadDir(sockets)
		if err != nil && !os.IsNotExist(err) {
			t.Error(err)
		} else if len(entries) != 0 {
			t.Errorf("Close left %d socket entries", len(entries))
		}
	}()
	// A private provider recipe supplies these flags. They never pass through
	// the person's Engine.Arguments admission, which refuses every spelling.
	h.options.Engine.cache = &engineCache{ramMiB: 256, checkpoints: 4, checkpointMiB: 64, reuse: 64, slotPath: slots}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	monitor := &cacheKnowledgeMonitor{}
	stopMonitor := monitor.watch(ctx, cancel, gpufault.PlatformRecords().Watch)
	defer stopMonitor()
	h.options.Engine.deviceLost = func(p *Process, marker string) {
		monitor.fail("GPU device loss: " + marker)
		cancel()
		go p.stop(2 * time.Second)
	}
	if _, err := h.Ensure(ctx, name); err != nil {
		t.Fatal(err)
	}
	client := EngineClient(h, nil)
	base := h.Base(name)
	results := map[string]any{"cache_ram_mib": 256, "ctx_checkpoints": 4, "cache_reuse": 64}
	post := func(path string, body any, want int) map[string]any {
		t.Helper()
		req, err := cacheKnowledgeRequest(ctx, base, path, body)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		data, err := io.ReadAll(io.LimitReader(response.Body, (2<<20)+1))
		if err != nil || len(data) > 2<<20 {
			t.Fatalf("%s: response read failed or exceeded its bound: %v", path, err)
		}
		if response.StatusCode != want {
			t.Errorf("%s: status %d; want %d", path, response.StatusCode, want)
		}
		var value map[string]any
		if err := json.Unmarshal(data, &value); err != nil {
			t.Errorf("%s: invalid response JSON: %v", path, err)
		}
		return value
	}
	// Provider access is keyed. The underlying, peer-verified socket client
	// supplies no Authorization header and must fail before a slot action.
	h.engineMu.Lock()
	bare := h.engines[h.hosts[name]].client
	h.engineMu.Unlock()
	req, err := cacheKnowledgeRequest(ctx, "http://"+enginehttp.UnixHost, cacheKnowledgeSlotPath("erase"), map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	response, err := bare.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	results["unkeyed_slot_status"] = response.StatusCode
	if response.StatusCode != http.StatusUnauthorized {
		t.Errorf("unkeyed slot request answered %d", response.StatusCode)
	}
	for _, filename := range cacheKnowledgeFilenames(out) {
		post(cacheKnowledgeSlotPath("save"), map[string]any{"filename": filename}, http.StatusBadRequest)
		post(cacheKnowledgeSlotPath("restore"), map[string]any{"filename": filename}, http.StatusBadRequest)
	}
	results["confined_save_restore_cases"] = 10
	tokenized := post(tokenizePath, cacheKnowledgeTokenize(strings.Repeat("Synthetic cache observation alpha beta gamma 0123456789.\n", 200)), http.StatusOK)
	tokens, ok := tokenized["tokens"].([]any)
	if !ok {
		t.Fatal("tokenization prerequisite returned no token list")
	}
	if len(tokens) < 1100 {
		t.Fatal("synthetic fixture has too few tokens")
	}
	complete := func(label string, prompt []any) {
		t.Helper()
		value := post(completionPath, cacheKnowledgeCompletion(prompt), http.StatusOK)
		timings, err := cacheKnowledgeTimings(value)
		if err != nil {
			t.Errorf("%s: %v", label, err)
			return
		}
		results[label] = timings
	}
	complete("cold", tokens[:1024])
	complete("turn_two", tokens[:1088])
	post(cacheKnowledgeSlotPath("save"), map[string]any{"filename": "synthetic.bin"}, http.StatusOK)
	complete("title", []any{tokens[0], tokens[4], tokens[8], tokens[12]})
	complete("after_title", tokens[:1100])
	post(cacheKnowledgeSlotPath("erase"), map[string]any{}, http.StatusOK)
	post(cacheKnowledgeSlotPath("restore"), map[string]any{"filename": "synthetic.bin"}, http.StatusOK)
	complete("after_restore", tokens[:1100])
	render := func(label string, body map[string]any) {
		value := post(applyTemplatePath, body, http.StatusOK)
		prompt, err := cacheKnowledgeRendered(value)
		if err != nil {
			t.Errorf("%s: %v", label, err)
			return
		}
		results[label] = strings.Contains(prompt, "cache-probe-reasoning-marker")
	}
	render("template_default", cacheKnowledgeTemplate("", false))
	for _, kwarg := range []string{"preserve_thinking", "preserve_reasoning"} {
		for _, preserve := range []bool{false, true} {
			render(kwarg+"_"+strconv.FormatBool(preserve), cacheKnowledgeTemplate(kwarg, preserve))
		}
	}
	// Keep the GPU monitor alive through engine shutdown, then join it before
	// accepting measurements. A late reset or unavailable watcher fails too.
	h.Close()
	stopMonitor()
	if err := monitor.outcome(ctx); err != nil {
		t.Error(err)
		return
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "knowledge.json"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("cache knowledge measurements: %s", encoded)
}

// This mode qualifies the stock production owners, including digest-selected
// settings data, with synthetic requests. It changes no installed state.
type cacheKnowledgeCallKey struct{}

type cacheKnowledgeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func cacheKnowledgeMessagesKey(messages []wire.ChatMessage) (string, error) {
	values := make([]cacheKnowledgeMessage, 0, len(messages))
	for _, message := range messages {
		var text strings.Builder
		for _, part := range message.Parts {
			if part.Kind != wire.PartKindText {
				return "", errors.New("fixture history contains non-text content")
			}
			text.WriteString(part.Text)
		}
		values = append(values, cacheKnowledgeMessage{message.Role.String(), text.String()})
	}
	raw, err := json.Marshal(values)
	return string(raw), err
}

// The trace is below EngineClient's media/slot owner. It observes the admitted
// id_slot and tees the actual SSE bytes without delaying delivery to inference.
type cacheKnowledgeResidentTransport struct {
	next       http.RoundTripper
	mu         sync.Mutex
	traces     map[string]*mtpChatTrace
	slots      map[string]int
	bodies     map[string][]byte
	pending    map[string][]string
	diagnostic *cacheKnowledgeColdObserver
}

func (transport *cacheKnowledgeResidentTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.Method != http.MethodPost || request.URL.Path != "/v1/chat/completions" {
		return transport.next.RoundTrip(request)
	}
	raw, err := io.ReadAll(request.Body)
	if err != nil {
		return nil, err
	}
	request.Body.Close()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	var messages []cacheKnowledgeMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil || len(messages) == 0 || messages[len(messages)-1].Role != "user" {
		return nil, errors.New("resident fixture completion has no synthetic user history")
	}
	key, err := json.Marshal(messages)
	if err != nil {
		return nil, err
	}
	transport.mu.Lock()
	labels := transport.pending[string(key)]
	if len(labels) == 0 {
		transport.mu.Unlock()
		return nil, errors.New("resident fixture completion has no pending call identity")
	}
	label := labels[0]
	transport.pending[string(key)] = labels[1:]
	transport.mu.Unlock()
	var slot *int
	if value, ok := body["id_slot"]; !ok {
		return nil, errors.New("resident owner supplied no slot")
	} else if err := json.Unmarshal(value, &slot); err != nil {
		return nil, err
	}
	if slot == nil || *slot < 0 || *slot >= 2 {
		return nil, errors.New("resident owner supplied invalid two-slot identity")
	}
	out := request.Clone(request.Context())
	if transport.diagnostic != nil {
		// Existing engine diagnostics only; leave sampling and total budget
		// unchanged. Counters avoid treating SSE chunks as generated tokens.
		body["timings_per_token"] = json.RawMessage("true")
		raw, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
	}
	out.Body = io.NopCloser(bytes.NewReader(raw))
	out.ContentLength = int64(len(raw))
	out.TransferEncoding = nil
	out.Header.Del("Content-Length")
	out.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(raw)), nil }
	trace := &mtpChatTrace{}
	transport.mu.Lock()
	transport.traces[label], transport.slots[label] = trace, *slot
	transport.bodies[label] = bytes.Clone(raw)
	transport.mu.Unlock()
	response, err := transport.next.RoundTrip(out)
	if err == nil {
		response.Body = &mtpTraceBody{ReadCloser: response.Body, trace: trace}
		if transport.diagnostic != nil {
			response.Body = transport.diagnostic.body(label, response.Body)
		}
	}
	return response, err
}

func cacheKnowledgeResidentRequest(label, conversation, prompt, epoch, model string) wire.RequestV3 {
	temperature := wire.Temperature{Milli: 0}
	request := wire.RequestV3{Identity: acceptance.RequestIdentity{Key: label, HistoryEpoch: epoch}, Conversation: conversation,
		Target: wire.ChatTarget{Model: model}, Content: wire.ChatContent{Messages: []wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: prompt}}}}, Options: &wire.ChatOptions{MaxOutput: 128, Temperature: &temperature},
			// The harness measures one engine call per request. It names the
			// engine profile's default effort, medium, which plans the same
			// reasoning budget as no effort; a named effort opts out of the
			// service's one further attempt without reasoning (INF-Q8).
			Extensions: map[string]string{"openai/reasoning_effort": "medium"}}}
	if label == "concurrent_title" || label == "idle_title" {
		use := wire.ChatUseTitle
		request.Use = &use
	}
	return request
}

type cacheKnowledgeResidentCall struct{ label, conversation, prompt string }

func cacheKnowledgeResidentFollowups(main, title string) []cacheKnowledgeResidentCall {
	subagent := "Explain this independent synthetic task in 80 to 120 words: " + strings.Repeat("Independent subagent task zeta theta. ", 400)
	return []cacheKnowledgeResidentCall{{"turn_two", "main", main + "\nContinue with another useful detail."}, {"idle_title", "", title + " Pick a name."}, {"after_title", "main", main + "\nContinue with another useful detail.\nExplain the next detail."}, {"subagent", "subagent", subagent}, {"subagent_two", "subagent", subagent}, {"after_subagent", "main", main + "\nContinue with another useful detail.\nExplain the next detail.\nFinish the final detail."}}
}

func cacheKnowledgeResidentReuse(previous, current map[string]any) error {
	prior, total := previous["rendered_prompt_tokens"].(float64), current["rendered_prompt_tokens"].(float64)
	shared := current["rendered_shared_prefix_tokens"].(float64)
	stable := previous["fixture_user_tokens"].(float64)
	timings := current["timings"].(map[string]any)
	if shared < 2048 || shared > min(prior, total) || shared < stable {
		return fmt.Errorf("unexpected fixture divergence prior=%v current=%v shared=%v stable_user=%v changed_tail=%v", prior, total, shared, stable, prior-shared)
	}
	var failures []error
	if timings["cache_n"].(float64) < shared-8 || timings["prompt_n"].(float64) > total-shared+8 || timings["cache_n"].(float64)+timings["prompt_n"].(float64) != total {
		failures = append(failures, fmt.Errorf("resident checkpoint reuse rendered_prior=%v rendered_current=%v shared=%v timings=%v", prior, total, shared, timings))
	}
	n, ms := timings["predicted_n"].(float64), timings["predicted_ms"].(float64)
	if n < 64 || ms <= 0 || n*1000/ms < 40 {
		failures = append(failures, fmt.Errorf("resident decode n=%v ms=%v; want >=40 tokens/s", n, ms))
	}
	return errors.Join(failures...)
}

func TestCacheKnowledgeResidentRequestAndReuse(t *testing.T) {
	request := cacheKnowledgeResidentRequest("operation", "main", "synthetic", "epoch", "model")
	raw, err := wire.ToJSON(request)
	if err != nil || bytes.Contains(raw, []byte("id_slot")) || request.Conversation != "main" {
		t.Fatal(string(raw), err)
	}
	if cacheKnowledgeResidentRequest("title", "", "title", "epoch", "model").Conversation != "" {
		t.Fatal("title gained affinity")
	}
	for _, label := range []string{"concurrent_title", "idle_title", "cold", "turn_two", "after_title", "subagent", "subagent_two", "after_subagent"} {
		value := cacheKnowledgeResidentRequest(label, "", "synthetic", "epoch", "model")
		title := label == "concurrent_title" || label == "idle_title"
		if (value.Use != nil) != title || title && *value.Use != wire.ChatUseTitle {
			t.Fatalf("%s use = %v", label, value.Use)
		}
		raw, err := wire.ToJSON(value)
		if err != nil || bytes.Contains(raw, []byte(`"use"`)) != title {
			t.Fatalf("%s request = %s: %v", label, raw, err)
		}
	}
	calls := cacheKnowledgeResidentFollowups("main prefix", "title")
	var labels, conversations []string
	for _, call := range calls {
		labels = append(labels, call.label)
		conversations = append(conversations, call.conversation)
	}
	if !reflect.DeepEqual(labels, []string{"turn_two", "idle_title", "after_title", "subagent", "subagent_two", "after_subagent"}) || !reflect.DeepEqual(conversations, []string{"main", "", "main", "subagent", "subagent", "main"}) || calls[4].prompt != calls[3].prompt {
		t.Fatal("resident followup sequence lost labelled two-turn subagent", calls)
	}
	// The changed template suffix exceeds eight tokens, while the long user
	// payload is unchanged. The denominator is its independently measured LCP.
	prior := map[string]any{"rendered_prompt_tokens": float64(13020), "fixture_user_tokens": float64(13000)}
	timings := map[string]any{"cache_n": float64(13006), "prompt_n": float64(24), "predicted_n": float64(128), "predicted_ms": float64(3200)}
	current := map[string]any{"rendered_prompt_tokens": float64(13030), "rendered_shared_prefix_tokens": float64(13010), "timings": timings}
	if err := cacheKnowledgeResidentReuse(prior, current); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]float64{"cache_n": 13001, "prompt_n": 29, "predicted_n": 63, "predicted_ms": 3201} {
		bad := map[string]any{}
		for k, v := range timings {
			bad[k] = v
		}
		bad[key] = value
		if cacheKnowledgeResidentReuse(prior, map[string]any{"rendered_prompt_tokens": float64(13030), "rendered_shared_prefix_tokens": float64(13010), "timings": bad}) == nil {
			t.Fatalf("accepted %s=%v", key, value)
		}
	}
	for _, shared := range []float64{2047, 3000} {
		bad := map[string]any{"rendered_prompt_tokens": float64(13030), "rendered_shared_prefix_tokens": shared, "timings": timings}
		if cacheKnowledgeResidentReuse(prior, bad) == nil {
			t.Fatalf("vacuous/divergent shared prefix %v passed", shared)
		}
	}
}

type cacheKnowledgeCheckpoint struct{ first, last, tokens int }

// Characterize b10825's checkpoint selection without changing the live LCP-8
// acceptance. Pinned owner: server-context.cpp, 3037-3061 and 3221-3237 at
// 9e0e220594af405a62835dc3a27495729fd8506b. This fixture's SWA window is zero.
func cacheKnowledgeCheckpointFrontier(shared, current int, checkpoints []cacheKnowledgeCheckpoint) int {
	threshold := shared
	if shared == current {
		threshold--
	}
	for index := len(checkpoints) - 1; index >= 0; index-- {
		checkpoint := checkpoints[index]
		if checkpoint.last <= shared && (checkpoint.first < threshold || checkpoint.first == 0) {
			return min(shared, max(checkpoint.first+1, checkpoint.last), checkpoint.tokens)
		}
	}
	return 0
}

func cacheKnowledgeCheckpointReuse(current map[string]any, checkpoints []cacheKnowledgeCheckpoint) error {
	shared, total := int(current["rendered_shared_prefix_tokens"].(float64)), int(current["rendered_prompt_tokens"].(float64))
	frontier := cacheKnowledgeCheckpointFrontier(shared, total, checkpoints)
	timings := current["timings"].(map[string]any)
	cached, evaluated := timings["cache_n"].(float64), timings["prompt_n"].(float64)
	if shared < 2048 || frontier <= 0 || cached != float64(frontier) || evaluated != float64(total-frontier) || cached+evaluated != float64(total) {
		return fmt.Errorf("pinned checkpoint reuse shared=%d frontier=%d cached=%v evaluated=%v total=%d", shared, frontier, cached, evaluated, total)
	}
	return nil
}

func TestCacheKnowledgeEditedUserCheckpointFrontier(t *testing.T) {
	for _, example := range []struct{ prior, current, shared, cached int }{{24028, 24035, 24021, 23512}, {24035, 24042, 24028, 23519}, {24042, 24048, 24035, 23526}} {
		checkpoints := []cacheKnowledgeCheckpoint{{example.prior - 517, example.prior - 517, example.prior - 516}, {example.prior - 5, example.prior - 5, example.prior - 4}}
		frontier := cacheKnowledgeCheckpointFrontier(example.shared, example.current, checkpoints)
		if frontier != example.cached || example.shared-frontier != 509 {
			t.Fatal("pinned checkpoint frontier differs from live4 metadata", example, frontier)
		}
		if cacheKnowledgeCheckpointFrontier(example.shared, example.current, checkpoints[1:]) != 0 {
			t.Fatal("checkpoint after changed prefix became reusable")
		}
		if cacheKnowledgeCheckpointFrontier(example.shared, example.current, nil) != 0 {
			t.Fatal("cleared checkpoint control gained reuse")
		}
		if got := cacheKnowledgeCheckpointFrontier(example.prior, example.current, checkpoints); got != example.prior-4 || got < example.prior-8 {
			t.Fatal("append-only prefix lost eligible latest checkpoint", got)
		}
		prior := map[string]any{"rendered_prompt_tokens": float64(example.prior), "fixture_user_tokens": float64(example.shared - 2)}
		current := map[string]any{"rendered_prompt_tokens": float64(example.current), "rendered_shared_prefix_tokens": float64(example.shared), "timings": map[string]any{"cache_n": float64(frontier), "prompt_n": float64(example.current - frontier), "predicted_n": float64(128), "predicted_ms": float64(2500)}}
		if err := cacheKnowledgeCheckpointReuse(current, checkpoints); err != nil {
			t.Fatal(err)
		}
		for _, counts := range [][2]int{{frontier - 1, example.current - frontier + 1}, {frontier, example.current - frontier + 1}, {0, example.current}} {
			bad := map[string]any{"rendered_prompt_tokens": float64(example.current), "rendered_shared_prefix_tokens": float64(example.shared), "timings": map[string]any{"cache_n": float64(counts[0]), "prompt_n": float64(counts[1])}}
			if cacheKnowledgeCheckpointReuse(bad, checkpoints) == nil {
				t.Fatal("accepted missed eligible reuse or wrong token identity", counts)
			}
		}
		if cacheKnowledgeResidentReuse(prior, current) == nil {
			t.Fatal("historical strict LCP-8 failure was silently accepted")
		}
		current["timings"].(map[string]any)["predicted_ms"] = float64(4050.9)
		if err := cacheKnowledgeResidentReuse(prior, current); err == nil || !strings.Contains(err.Error(), "resident checkpoint reuse") || !strings.Contains(err.Error(), "resident decode") {
			t.Fatalf("independent acceptance failures were hidden: %v", err)
		}
	}
}

func TestCacheKnowledgeResidentTracePreservesStreaming(t *testing.T) {
	const stream = "data: {\"timings\":{\"prompt_n\":12}}\n\ndata: [DONE]\n\n"
	trace := &cacheKnowledgeResidentTransport{next: inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body["id_slot"] != float64(1) {
			t.Errorf("owner body changed: %v %v", body, err)
		}
		w.Write([]byte(stream))
	})}, traces: map[string]*mtpChatTrace{}, slots: map[string]int{}, bodies: map[string][]byte{}}
	key, err := cacheKnowledgeMessagesKey([]wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "synthetic"}}}})
	if err != nil {
		t.Fatal(err)
	}
	trace.pending = map[string][]string{key: {"main"}}
	request, err := cacheKnowledgeRequest(t.Context(), "http://engine.fixture", "/v1/chat/completions", map[string]any{"id_slot": 1, "cache_prompt": true, "messages": []map[string]string{{"role": "user", "content": "synthetic"}}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := trace.RoundTrip(request)
	if err != nil {
		t.Fatal(err)
	}
	if trace.traces["main"].stream.Len() != 0 {
		t.Fatal("trace consumed response before inference")
	}
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	if err := response.Body.Close(); err != nil {
		t.Fatal(err)
	}
	if string(raw) != stream || trace.traces["main"].stream.String() != stream || trace.slots["main"] != 1 {
		t.Fatal("trace changed stream/owner correlation")
	}
}

func TestCacheKnowledgeResidentTraceDuplicatePrompts(t *testing.T) {
	key, err := cacheKnowledgeMessagesKey([]wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "same prompt"}}}})
	if err != nil {
		t.Fatal(err)
	}
	trace := &cacheKnowledgeResidentTransport{next: inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: [DONE]\n\n")
	})}, traces: map[string]*mtpChatTrace{}, slots: map[string]int{}, bodies: map[string][]byte{}, pending: map[string][]string{key: {"subagent", "subagent_two"}}}
	call := func(path string) error {
		req, err := cacheKnowledgeRequest(t.Context(), "http://engine", path, map[string]any{"id_slot": 0, "messages": []map[string]string{{"role": "user", "content": "same prompt"}}})
		if err != nil {
			return err
		}
		response, err := trace.RoundTrip(req)
		if err == nil {
			_, err = io.Copy(io.Discard, response.Body)
			response.Body.Close()
		}
		return err
	}
	for _, path := range []string{tokenizePath, applyTemplatePath, "/v1/chat/completions", "/v1/chat/completions"} {
		if err := call(path); err != nil {
			t.Fatal(err)
		}
	}
	if len(trace.traces) != 2 || trace.traces["subagent"] == trace.traces["subagent_two"] || trace.slots["subagent"] != trace.slots["subagent_two"] {
		t.Fatal("duplicate prompt collapsed per-call identity")
	}
	if err := call("/v1/chat/completions"); err == nil {
		t.Fatal("unregistered duplicate prompt gained a call identity")
	}
}

const (
	cacheKnowledgeFunctionalOutput = int64(4096)       // OpenCode's actual caller default, including reasoning.
	cacheKnowledgeColdBudget       = 300 * time.Second // Live5's concurrent cold pair reached 218.8 seconds.
	cacheKnowledgeWarmBudget       = 45 * time.Second  // Independent headroom over retained warm requests.
	cacheKnowledgeCleanupBudget    = 15 * time.Second
)

func cacheKnowledgeObserve(ctx context.Context, observe func(context.Context) wire.ChatPage) (wire.ChatPage, error) {
	if err := ctx.Err(); err != nil {
		return wire.ChatPage{}, err
	}
	page := observe(ctx)
	if err := ctx.Err(); err != nil {
		return wire.ChatPage{}, err
	}
	if page.Outcome != wire.PageOutcomePage {
		return page, fmt.Errorf("observation: %s", page.Outcome)
	}
	return page, nil
}

func cacheKnowledgeResidentSequence(ctx context.Context, h *Host, name string, next http.RoundTripper, admit inference.Admitter, diagnose func(string, map[string]any) error) (map[string]map[string]any, error) {
	return cacheKnowledgeResidentSequenceBudget(ctx, h, name, next, admit, cacheKnowledgeFunctionalOutput, false, diagnose)
}

func cacheKnowledgeResidentSequenceBudget(ctx context.Context, h *Host, name string, next http.RoundTripper, admit inference.Admitter, maxOutput int64, useModelTemperature bool, diagnose func(string, map[string]any) error) (map[string]map[string]any, error) {
	return cacheKnowledgeResidentSequenceMode(ctx, h, name, next, admit, maxOutput, useModelTemperature, nil, diagnose)
}

func cacheKnowledgeResidentSequenceMode(ctx context.Context, h *Host, name string, next http.RoundTripper, admit inference.Admitter, maxOutput int64, useModelTemperature bool, coldDiagnostic *cacheKnowledgeColdObserver, diagnose func(string, map[string]any) error, awakeMode ...bool) (map[string]map[string]any, error) {
	awake := len(awakeMode) == 1 && awakeMode[0]
	if maxOutput != 128 && maxOutput != cacheKnowledgeFunctionalOutput {
		return nil, errors.New("fixture requires its explicit comparative or functional output budget")
	}
	resolve := func(host string) (enginehttp.UnixEndpoint, bool) { return h.EngineEndpoint(host) }
	if next == nil {
		next = enginehttp.NewUnixTransport(resolve)
	}
	if admit == nil {
		admit = modelhostAdmit(h)
	}
	trace := &cacheKnowledgeResidentTransport{next: next, traces: map[string]*mtpChatTrace{}, slots: map[string]int{}, bodies: map[string][]byte{}, pending: map[string][]string{}}
	trace.diagnostic = coldDiagnostic
	client := engineClient(h, nil, trace)
	engine := router.OpenAICompatible("modelhost", h.Base(name), "/v1", h.Models())
	engine.Profiles = []string{router.ProfileChat}
	var mu sync.Mutex
	records := map[string]inference.Record{}
	changed := make(chan struct{})
	provider, err := inference.New(inference.Config{ChatLogicalOwner: "modelhost", Router: router.New(engine), HTTP: client,
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		Admit: admit, Record: func(record inference.Record) {
			if record.Operation == "" || record.Outcome == "seal_omitted" {
				return
			}
			mu.Lock()
			records[record.Operation] = record
			close(changed)
			changed = make(chan struct{})
			mu.Unlock()
		}})
	if err != nil {
		return nil, err
	}
	defer provider.Close()
	subject := inference.Subject{Account: "synthetic-live", Program: "conversation-memory-fixture"}
	window := provider.GetChatHistoryWindow(subject)
	if window.Window == nil {
		return nil, errors.New("fixture history owner unavailable")
	}
	measure := func(phaseCtx context.Context, label, conversation string, messages []wire.ChatMessage) (_ map[string]any, resultErr error) {
		started := time.Now()
		output := &cacheKnowledgeOutput{}
		var observedTimings map[string]any
		observedFinish := "not_observed"
		var requestedMaxOutput int64
		diagnosed := false
		emitDiagnostic := func() error {
			if diagnose != nil && !diagnosed {
				diagnosed = true
				trace.mu.Lock()
				stream := trace.traces[label]
				trace.mu.Unlock()
				if stream != nil {
					stream.mu.Lock()
					raw := bytes.Clone(stream.stream.Bytes())
					stream.mu.Unlock()
					observedTimings, observedFinish, _ = cacheKnowledgeEngineMetadata(raw)
				}
				if coldDiagnostic != nil {
					observedTimings, observedFinish = coldDiagnostic.metadata(label)
				}
				diagnostic := output.diagnostic()
				diagnostic["phase_wall_ms"] = time.Since(started).Milliseconds()
				diagnostic["upstream_finish_reason"] = observedFinish
				diagnostic["requested_max_output"] = requestedMaxOutput
				diagnostic["temperature_omitted"] = useModelTemperature
				var measurementErr error
				if coldDiagnostic != nil {
					diagnostic["diagnostic_only"] = !awake
					diagnostic["acceptance_credit"] = false
					trace.mu.Lock()
					diagnostic["id_slot"] = trace.slots[label]
					trace.mu.Unlock()
					result, err := coldDiagnostic.report(phaseCtx, label, client, h.Base(name))
					measurementErr = err
					if result == nil {
						result = map[string]any{"availability": "unavailable", "error_category": "observer_failure"}
					}
					diagnostic["cold_measurement"] = result
					if awake {
						diagnostic["awake_remeasurement"] = true
						diagnostic["cold300_report_only"] = true
						if first, ok := result["first_generated_arrival_ms"].(int64); ok && first >= 0 {
							diagnostic["request_first_stream_ms"] = first - started.Sub(coldDiagnostic.start).Milliseconds()
						}
					}
					diagnostic["cold300_acceptance"] = "within_budget"
					if result["cold300_missed"] == true {
						diagnostic["cold300_acceptance"] = "failed"
					}
				}
				// These requests contain only this fixture's synthetic transcript.
				// Reuse the existing response tap for a bounded public-reasoning
				// tail in the real live diagnostic; never log replay-seal bytes.
				if useModelTemperature && stream != nil && coldDiagnostic == nil {
					stream.mu.Lock()
					diagnostic["synthetic_reasoning_tail"] = cacheKnowledgeSyntheticReasoningTail(stream.stream.Bytes())
					stream.mu.Unlock()
				}
				timings := map[string]float64{}
				for _, key := range []string{"prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_ms", "draft_n", "draft_n_accepted"} {
					if value, ok := observedTimings[key].(float64); ok && !math.IsNaN(value) && !math.IsInf(value, 0) {
						timings[key] = value
					}
				}
				diagnostic["engine_timings"] = timings
				// Persist typed terminal and bounded partial measurements before
				// returning a measurement failure. Never serialize the raw error.
				return errors.Join(measurementErr, diagnose(label, diagnostic))
			}
			return nil
		}
		defer func() {
			resultErr = errors.Join(resultErr, emitDiagnostic())
			if resultErr != nil {
				resultErr = fmt.Errorf("%s phase_wall_ms=%d: %w", label, time.Since(started).Milliseconds(), resultErr)
			}
		}()
		callCtx := inference.WithRoute(phaseCtx, inference.RouteNative, "program")
		key, err := cacheKnowledgeMessagesKey(messages)
		if err != nil {
			return nil, err
		}
		trace.mu.Lock()
		trace.pending[key] = append(trace.pending[key], label)
		trace.mu.Unlock()
		request := cacheKnowledgeResidentRequest(label, conversation, "", window.Window.HistoryEpoch, name)
		request.Content.Messages = messages
		request.Content.Options.MaxOutput = maxOutput
		if useModelTemperature {
			// OpenCode omits an unsupplied temperature. Keep the real live
			// request faithful to that caller, using the pinned model default.
			request.Content.Options.Temperature = nil
		}
		if request.Content.Options != nil {
			requestedMaxOutput = request.Content.Options.MaxOutput
		}
		admission := provider.StartV3(callCtx, subject, request)
		if admission.Outcome != wire.ChatStartOutcomeAccepted {
			return nil, fmt.Errorf("%s admission: %+v", label, admission)
		}
		cursor, ended := int64(0), false
		defer func() {
			if ended {
				return
			}
			// StartV3 owns execution independently of the caller's observation
			// context. Cancel only this accepted fixture operation, then drain
			// its terminal page under an independent finite cleanup context.
			cleanupCtx, cancelCleanup := context.WithTimeout(context.Background(), cacheKnowledgeCleanupBudget)
			defer cancelCleanup()
			provider.CancelV3(subject, admission.Request)
			for {
				page, err := cacheKnowledgeObserve(cleanupCtx, func(waitCtx context.Context) wire.ChatPage {
					return provider.ObserveV3(waitCtx, subject, admission.Request, cursor, 256, 65536, 1000)
				})
				if err != nil {
					resultErr = errors.Join(resultErr, fmt.Errorf("%s cleanup: %w", label, err))
					return
				}
				cursor = page.Next
				output.consume(page.Events)
				if page.AtEnd {
					return
				}
			}
		}()
		for {
			if err := callCtx.Err(); err != nil {
				return nil, fmt.Errorf("%s: %w", label, err)
			}
			mu.Lock()
			record, ok := records[admission.Request]
			wake := changed
			mu.Unlock()
			if ended && !ok && output.end != nil && output.end.Outcome == wire.ChatResultOutcomeCompleted {
				select {
				case <-wake:
				case <-callCtx.Done():
					return nil, callCtx.Err()
				}
				continue
			}
			if ended {
				trace.mu.Lock()
				stream, slot, body := trace.traces[label], trace.slots[label], bytes.Clone(trace.bodies[label])
				trace.mu.Unlock()
				if stream == nil {
					return nil, errors.New("missing engine stream")
				}
				stream.mu.Lock()
				raw := bytes.Clone(stream.stream.Bytes())
				truncated := stream.truncated
				stream.mu.Unlock()
				if truncated && coldDiagnostic == nil {
					return nil, io.ErrShortBuffer
				}
				// Actual engine diagnostics remain separately attributed. Their
				// token facts are used only by this opt-in stock-engine fixture.
				timings, finish, err := cacheKnowledgeEngineMetadata(raw)
				if coldDiagnostic != nil {
					timings, finish = coldDiagnostic.metadata(label)
					err = nil
				}
				if err != nil {
					return nil, err
				}
				observedTimings = timings
				observedFinish = finish
				if err := emitDiagnostic(); err != nil {
					return nil, err
				}
				if coldDiagnostic != nil && !awake {
					// Preserve the real Q8/length terminal as a measurement. No
					// answer is fabricated and no append/acceptance gate is run.
					if output.end == nil || output.end.Outcome != wire.ChatResultOutcomeCompleted && output.end.Cause != "upstream:reasoning_only" {
						return nil, errors.New("cold diagnostic ended without natural completion or reasoning-only terminal")
					}
					return map[string]any{"id_slot": float64(slot), "timings": timings}, nil
				}
				reply, err := output.assistant()
				if err != nil {
					return nil, err
				}
				if timings == nil {
					return nil, errors.New("stock engine supplied no timing diagnostics")
				}
				if record.Outcome != "completed" || record.Timing == nil {
					return nil, fmt.Errorf("%s terminal record unavailable", label)
				}
				var template struct {
					Prompt *string `json:"prompt"`
				}
				if err := postEngine(callCtx, client, h.Base(name)+applyTemplatePath, body, &template); err != nil {
					return nil, err
				}
				if template.Prompt == nil {
					return nil, errors.New("engine rendered no production prompt")
				}
				tokenBody, err := json.Marshal(map[string]any{"content": *template.Prompt, "add_special": true, "parse_special": true})
				if err != nil {
					return nil, err
				}
				var tokenized struct {
					Tokens []int64 `json:"tokens"`
				}
				if err := postEngine(callCtx, client, h.Base(name)+tokenizePath, tokenBody, &tokenized); err != nil {
					return nil, err
				}
				if len(tokenized.Tokens) == 0 {
					return nil, errors.New("empty production prompt token oracle")
				}
				userBody, err := json.Marshal(map[string]any{"content": messages[0].Parts[0].Text, "add_special": false, "parse_special": false})
				if err != nil {
					return nil, err
				}
				var userTokens struct {
					Tokens []int64 `json:"tokens"`
				}
				if err := postEngine(callCtx, client, h.Base(name)+tokenizePath, userBody, &userTokens); err != nil {
					return nil, err
				}
				if len(userTokens.Tokens) == 0 {
					return nil, errors.New("empty unchanged user payload oracle")
				}
				result := map[string]any{"phase_wall_ms": float64(time.Since(started).Milliseconds()), "id_slot": float64(slot), "timings": timings, "rendered_prompt_tokens": float64(len(tokenized.Tokens)), "rendered_tokens": tokenized.Tokens, "fixture_user_tokens": float64(len(userTokens.Tokens)), "stream_bytes": len(raw), "oa_timing": record.Timing, "assistant_reply": reply}
				if awake && os.Getenv("MODELHOST_LIVE_OUT") != "" {
					checkpoints, err := cacheKnowledgeObservedCheckpoints(filepath.Join(os.Getenv("MODELHOST_LIVE_OUT"), "engine.log"), slot)
					if err != nil {
						return nil, err
					}
					result["checkpoint_metadata"] = checkpoints
				}
				return result, nil
			}
			if !ended {
				page, err := cacheKnowledgeObserve(callCtx, func(waitCtx context.Context) wire.ChatPage {
					return provider.ObserveV3(waitCtx, subject, admission.Request, cursor, 256, 65536, 1000)
				})
				if err != nil {
					return nil, fmt.Errorf("%s: %w", label, err)
				}
				output.consume(page.Events)
				cursor, ended = page.Next, page.AtEnd
				continue
			}
			select {
			case <-wake:
			case <-callCtx.Done():
				return nil, callCtx.Err()
			}
		}
	}
	main := "Summarize the following synthetic transcript in 80 to 120 words. Do not quote the transcript.\n" + strings.Repeat("Synthetic long conversation alpha beta gamma 0123456789.\n", 1200)
	title := "Give a short title for this unrelated synthetic request: " + strings.Repeat("title topic delta. ", 120)
	user := func(text string) wire.ChatMessage {
		return wire.ChatMessage{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: text}}}
	}
	mainHistory := []wire.ChatMessage{user(main)}
	results := map[string]map[string]any{}
	var initial [2]map[string]any
	var failures [2]error
	var workers sync.WaitGroup
	coldCtx, cancelCold := context.WithTimeout(ctx, cacheKnowledgeColdBudget)
	if coldDiagnostic != nil || awake {
		cancelCold()
		// Architect's measurement only: cold300 remains report-only. The
		// owning Go/native ceilings and cancellation cleanup remain finite.
		coldCtx, cancelCold = context.WithCancel(ctx)
	}
	defer cancelCold()
	workers.Add(2)
	go func() { defer workers.Done(); initial[0], failures[0] = measure(coldCtx, "cold", "main", mainHistory) }()
	go func() {
		defer workers.Done()
		initial[1], failures[1] = measure(coldCtx, "concurrent_title", "", []wire.ChatMessage{user(title)})
	}()
	workers.Wait()
	cancelCold()
	if err := errors.Join(failures[:]...); err != nil {
		return nil, err
	}
	results["cold"], results["concurrent_title"] = initial[0], initial[1]
	if coldDiagnostic != nil && !awake {
		return results, nil // No warm sequence or acceptance credit.
	}
	var subagentHistory []wire.ChatMessage
	lastMain, lastSubagent := "cold", ""
	for _, call := range cacheKnowledgeResidentFollowups(main, title) {
		var history []wire.ChatMessage
		switch call.conversation {
		case "main":
			mainHistory = append(mainHistory, results[lastMain]["assistant_reply"].(wire.ChatMessage), user("Continue with another useful detail in 80 to 120 words."))
			history = mainHistory
		case "subagent":
			if lastSubagent == "" {
				subagentHistory = []wire.ChatMessage{user(call.prompt)}
			} else {
				subagentHistory = append(subagentHistory, results[lastSubagent]["assistant_reply"].(wire.ChatMessage), user("Explain one further detail in 80 to 120 words."))
			}
			history = subagentHistory
		default:
			history = []wire.ChatMessage{user(call.prompt)}
		}
		warmCtx, cancelWarm := context.WithTimeout(ctx, cacheKnowledgeWarmBudget)
		if awake {
			cancelWarm()
			warmCtx, cancelWarm = context.WithCancel(ctx)
		}
		value, err := measure(warmCtx, call.label, call.conversation, history)
		cancelWarm()
		if err != nil {
			return nil, err
		}
		results[call.label] = value
		if call.conversation == "main" {
			lastMain = call.label
		}
		if call.conversation == "subagent" {
			lastSubagent = call.label
		}
	}
	for _, pair := range [][2]string{{"cold", "turn_two"}, {"turn_two", "after_title"}, {"after_title", "after_subagent"}, {"subagent", "subagent_two"}} {
		previous := results[pair[0]]["rendered_tokens"].([]int64)
		label := pair[1]
		current := results[label]["rendered_tokens"].([]int64)
		shared := commonTokens(previous, current)
		results[label]["rendered_prior_prompt_tokens"], results[label]["rendered_shared_prefix_tokens"], results[label]["rendered_changed_tail_tokens"] = len(previous), float64(shared), len(previous)-shared
	}
	for _, value := range results {
		delete(value, "rendered_tokens")
		delete(value, "assistant_reply")
	}
	return results, nil
}

func TestCacheKnowledgeObservationCancellation(t *testing.T) {
	for _, during := range []bool{false, true} {
		ctx, cancel := context.WithCancel(t.Context())
		calls := 0
		if !during {
			cancel()
		}
		_, err := cacheKnowledgeObserve(ctx, func(context.Context) wire.ChatPage {
			calls++
			cancel()
			// ObserveV3 returns this empty page when its context expires.
			return wire.ChatPage{Outcome: wire.PageOutcomePage}
		})
		cancel()
		if !errors.Is(err, context.Canceled) || calls != map[bool]int{false: 0, true: 1}[during] {
			t.Fatalf("cancellation during=%v: calls=%d err=%v", during, calls, err)
		}
	}
}

func TestCacheKnowledgeResidentProviderCancellationJoinsOwnedRequests(t *testing.T) {
	h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(),
		engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(2)}}}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	entered, departed := make(chan struct{}, 2), make(chan struct{}, 2)
	var diagnosticsMu sync.Mutex
	diagnostics := map[string]map[string]any{}
	transport := inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected canceled fixture path: %s", r.URL.Path)
		}
		entered <- struct{}{}
		<-r.Context().Done()
		departed <- struct{}{}
	})}
	done := make(chan error, 1)
	go func() {
		_, err := cacheKnowledgeResidentSequence(ctx, h, "fixture", transport, func(context.Context, *router.Host, string) (string, bool) { return "", true }, func(label string, value map[string]any) error {
			diagnosticsMu.Lock()
			defer diagnosticsMu.Unlock()
			diagnostics[label] = value
			return nil
		})
		done <- err
	}()
	for range 2 {
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("owned requests did not enter")
		}
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled sequence: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("canceled observation spun or owned requests did not join")
	}
	if len(departed) != 2 {
		t.Fatalf("owned HTTP executions still active: departures=%d", len(departed))
	}
	if len(diagnostics) != 2 {
		t.Fatal("canceled terminal evidence missing")
	}
	for _, value := range diagnostics {
		if value["terminal_outcome"] != "cancelled" {
			t.Fatal("owned cancellation terminal lost")
		}
	}
}

func TestCacheKnowledgeResidentProviderSequence(t *testing.T) {
	h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(),
		engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(2)}}}
	var mu sync.Mutex
	var prompts []string
	var histories [][]string
	entered, release := make(chan struct{}, 2), make(chan struct{})
	transport := inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer fixture-key" {
			t.Error("engine key missing")
		}
		var body struct {
			Messages []struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"messages"`
			Content string `json:"content"`
			Slot    *int   `json:"id_slot"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		switch r.URL.Path {
		case "/v1/chat/completions":
			if r.Context().Value(cacheKnowledgeCallKey{}) != nil {
				t.Error("caller marker reached production execution context")
			}
			if len(body.Messages) == 0 || body.Slot == nil {
				t.Error("production body lost prompt/slot")
				w.WriteHeader(400)
				return
			}
			mu.Lock()
			var roles []string
			for _, message := range body.Messages {
				roles = append(roles, message.Role)
				if message.Role == "assistant" && (message.Content != "synthetic answer" || message.Reasoning != "opaque replay") {
					t.Errorf("assistant history changed answer/seal: %+v", message)
				}
			}
			histories = append(histories, roles)
			prompts = append(prompts, body.Messages[len(body.Messages)-1].Content)
			index := len(prompts)
			mu.Unlock()
			if index <= 2 {
				entered <- struct{}{}
				if index == 2 {
					<-entered
					<-entered
					close(release)
				}
				<-release
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"opaque replay\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"synthetic answer\"}}]}\n\n")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":1,\"cache_n\":0,\"prompt_ms\":1,\"predicted_n\":128,\"predicted_ms\":1000}}\n\ndata: [DONE]\n\n")
		case applyTemplatePath:
			if len(body.Messages) == 0 {
				t.Error("template body lost messages")
				w.WriteHeader(400)
				return
			}
			var rendered strings.Builder
			for _, message := range body.Messages {
				rendered.WriteString("<" + message.Role + ">" + message.Reasoning + message.Content + "</" + message.Role + ">\n")
			}
			rendered.WriteString("<assistant>")
			json.NewEncoder(w).Encode(map[string]any{"prompt": rendered.String()})
		case tokenizePath:
			tokens := make([]int64, 0, len(body.Content))
			for _, value := range body.Content {
				tokens = append(tokens, int64(value))
			}
			json.NewEncoder(w).Encode(map[string]any{"tokens": tokens})
		default:
			t.Errorf("unexpected engine path %s", r.URL.Path)
			w.WriteHeader(404)
		}
	})}
	ctx, cancel := context.WithTimeout(context.WithValue(t.Context(), cacheKnowledgeCallKey{}, "caller-only"), 10*time.Second)
	defer cancel()
	values, err := cacheKnowledgeResidentSequence(ctx, h, "fixture", transport, func(context.Context, *router.Host, string) (string, bool) { return "", true }, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(values) != 8 || len(prompts) != 8 {
		t.Fatalf("incomplete actual-provider sequence: %d/%d", len(values), len(prompts))
	}
	counts := map[int]int{}
	for _, roles := range histories {
		counts[len(roles)]++
		for index, role := range roles {
			want := "user"
			if index%2 == 1 {
				want = "assistant"
			}
			if role != want {
				t.Fatalf("non-production history roles %v", roles)
			}
		}
	}
	if !reflect.DeepEqual(counts, map[int]int{1: 4, 3: 2, 5: 1, 7: 1}) {
		t.Fatalf("history did not append assistant+user turns: %v", histories)
	}
	for _, label := range []string{"cold", "concurrent_title", "turn_two", "idle_title", "after_title", "subagent", "subagent_two", "after_subagent"} {
		if values[label]["stream_bytes"].(int) == 0 || values[label]["oa_timing"] == nil {
			t.Fatalf("missing per-call trace/record %s", label)
		}
	}
	mainSlot := values["cold"]["id_slot"]
	if mainSlot == values["concurrent_title"]["id_slot"] {
		t.Fatal("concurrent calls shared a slot")
	}
	for _, label := range []string{"turn_two", "after_title", "after_subagent"} {
		if values[label]["id_slot"] != mainSlot || values[label]["rendered_shared_prefix_tokens"].(float64) < 2048 {
			t.Fatalf("main call correlation lost: %s %+v", label, values[label])
		}
	}
	if values["subagent"]["id_slot"] != values["subagent_two"]["id_slot"] {
		t.Fatal("same-labelled subagent lost affinity")
	}
}

func liveProductionCacheKnowledge(t *testing.T, diagnosticOnly bool, awakeMode ...bool) {
	awake := len(awakeMode) == 1 && awakeMode[0]
	need := func(key string) string {
		value := os.Getenv(key)
		if value == "" {
			t.Fatalf("%s is not set", key)
		}
		return value
	}
	program, model := need("MODELHOST_LIVE_ENGINE"), need("MODELHOST_LIVE_MODEL")
	memoryLimit, err := cacheKnowledgeMemoryLimit(os.Getenv("MODELHOST_LIVE_MEMORY_BYTES"))
	if err != nil {
		t.Fatal(err)
	}
	state, out, sockets := need("MODELHOST_LIVE_STATE"), need("MODELHOST_LIVE_OUT"), need("MODELHOST_LIVE_SOCKETS")
	for _, path := range []string{state, out, sockets} {
		if !filepath.IsAbs(path) {
			t.Fatal("fixture roots must be absolute")
		}
	}
	entry := profileCatalog(modelwire.CatalogUseCode).Entries[0]
	info, err := os.Stat(model)
	if err != nil || info.Size() != entry.DownloadBytes {
		t.Fatalf("model size: %v %v", info, err)
	}
	if err := os.MkdirAll(out, 0o700); err != nil {
		t.Fatal(err)
	}
	log, err := os.Create(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	defer log.Close()
	pin, ok := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	if !ok {
		t.Fatal("no platform engine pin")
	}
	name := filepath.Base(model)
	object := Object{Store: "live", ID: "model", Path: model, File: name, Locator: model, Size: info.Size()}
	source := defaultProofSource{Source: fakeSource{objects: map[string]Object{"live/model": object}}, current: true,
		proof: content.Object{Locator: object.Locator, Size: object.Size, Modified: "pinned-live-fixture", Digest: entry.Digest, Evidence: content.EvidenceHashed}}
	engineExits := make(chan EngineExit, 1)
	var engineExit *EngineExit
	collectExit := func() {
		if engineExit == nil {
			select {
			case value := <-engineExits:
				engineExit = &value
			default:
			}
		}
	}
	ctx, cancel := context.WithCancel(t.Context())
	if awake {
		cancel()
		ctx, cancel = context.WithTimeout(t.Context(), 720*time.Second)
	}
	defer cancel()
	monitor := &cacheKnowledgeMonitor{}
	output := &cacheKnowledgeAbortWriter{tap: newFitTap(log), stop: func(line string) { monitor.fail("allocation failure: " + line); cancel() }}
	configured := []string{"--log-verbosity", "4"}
	h, err := newTestHost(Options{Models: []Model{{Name: name, Store: "live", Object: "model"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin), Arguments: configured, Sockets: sockets, MemoryLimit: memoryLimit, Ready: 90 * time.Second, Stderr: output},
		Source: source, Leases: newFakeLeases(), Idle: time.Hour, OnEngineExit: func(value EngineExit) { engineExits <- value }})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		h.Close()
		collectExit()
		if engineExit != nil {
			t.Logf("engine peak_process_bytes=%d peak_job_bytes=%d process_limit_bytes=%d limit_hit=%v", engineExit.PeakProcessMemory, engineExit.PeakJobMemory, engineExit.ProcessMemoryLimit, engineExit.MemoryLimitHit)
		}
		if err := monitor.outcome(ctx); err != nil {
			t.Logf("production monitor: %v", err)
		}
		if entries, err := os.ReadDir(sockets); err != nil && !os.IsNotExist(err) {
			t.Error(err)
		} else if len(entries) != 0 {
			t.Errorf("Close left %d socket entries", len(entries))
		}
	}()
	engine := h.modelEngine(h.options.Models[0], object)
	if !reflect.DeepEqual(engine.Arguments, curatedModelArguments(configured, entry)) || engine.cacheRecipe().slots != 0 {
		t.Fatal("production argument/profile owner changed")
	}
	forecast, err := engine.cacheRecipe().leaseBytes(object.Size)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("production engine args=%q cache=%q lease_bytes=%d", engine.Arguments, engine.cacheRecipe().arguments(), forecast)
	watchReady := make(chan struct{})
	var watching sync.Once
	stopMonitor := monitor.watch(ctx, cancel, func(watchCtx context.Context, deliver func(gpufault.Event)) error {
		return gpufault.PlatformRecords().Watch(watchCtx, func(event gpufault.Event) {
			if event.Kind == gpufault.Watching {
				watching.Do(func() { close(watchReady) })
			}
			deliver(event)
		})
	})
	defer stopMonitor()
	h.options.Engine.deviceLost = func(p *Process, marker string) {
		monitor.fail("GPU device loss: " + marker)
		cancel()
		go p.stop(2 * time.Second)
	}
	select {
	case <-watchReady:
	case <-ctx.Done():
		t.Fatal(monitor.outcome(ctx))
	case <-time.After(5 * time.Second):
		t.Fatal("GPU records subscription did not stand")
	}
	startupBegan := time.Now()
	startupCtx, cancelStartup := context.WithTimeout(ctx, 90*time.Second)
	defer cancelStartup()
	if _, err := h.Ensure(startupCtx, name); err != nil {
		t.Fatal(err)
	}
	pid := h.resources.PID(name)
	readyCommit, err := cacheKnowledgePrivateCommit(startupCtx, pid)
	if err != nil {
		t.Fatal(err)
	}
	startupMS := time.Since(startupBegan).Milliseconds()
	cancelStartup()
	t.Logf("phase budgets startup=90s cold=%s each_followup=%s cleanup=%s; startup_wall_ms=%d", cacheKnowledgeColdBudget, cacheKnowledgeWarmBudget, cacheKnowledgeCleanupBudget, startupMS)
	results := map[string]any{"startup_wall_ms": startupMS, "arguments": engine.Arguments, "cache_arguments": engine.cacheRecipe().arguments(), "lease_bytes": forecast, "cache_forecast_bytes": forecast - object.Size, "resident_slots": engine.cacheRecipe().residentSlots, "model_digest": entry.Digest, "engine_release": pin.Release, "engine_source_commit": pin.SourceCommit, "fit_need_bytes": entry.NeedBytes, "private_commit_ready_bytes": readyCommit}
	if awake {
		results["shared_context_budget_seconds"] = 720
		results["warm_decode_required_tokens_per_second"] = 40
		results["warm_decode_context_source"] = "each turn's rendered_prompt_tokens; output count separately recorded"
		results["checkpoint_gate"] = "observed b10825 checkpoint frontier at rendered LCP; 8192 spacing may reevaluate shared prefix"
		t.Log("awake phases share the finite 720s context; cold300 and 45s warm comparators remain historical report values")
	}
	terminalLog, err := os.OpenFile(filepath.Join(out, "terminal.jsonl"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	defer terminalLog.Close()
	var terminalMu sync.Mutex
	terminalCount := 0
	firstStreams := map[string]int64{}
	var coldDiagnostic *cacheKnowledgeColdObserver
	if diagnosticOnly || awake {
		coldDiagnostic = newCacheKnowledgeColdObserver()
	}
	responses, err := cacheKnowledgeResidentSequenceMode(ctx, h, name, nil, nil, cacheKnowledgeFunctionalOutput, true, coldDiagnostic, func(label string, diagnostic map[string]any) error {
		diagnostic["call"] = label
		raw, err := json.Marshal(diagnostic)
		if err != nil {
			return err
		}
		terminalMu.Lock()
		defer terminalMu.Unlock()
		if first, ok := diagnostic["request_first_stream_ms"].(int64); ok {
			firstStreams[label] = first
		}
		if terminalCount >= 8 || len(raw) > 8192 {
			return errors.New("fixture terminal diagnostic bound exceeded")
		}
		terminalCount++
		if _, err := terminalLog.Write(append(raw, '\n')); err != nil {
			return err
		}
		return terminalLog.Sync()
	}, awake)
	if err != nil {
		t.Fatal(err)
	}
	if coldDiagnostic != nil && !awake {
		h.Close()
		collectExit()
		if engineExit == nil {
			t.Fatal("cold diagnostic engine exit accounting unavailable")
		}
		if err := cacheKnowledgeEngineOutcome(*engineExit, memoryLimit); err != nil {
			t.Fatal(err)
		}
		stopMonitor()
		if err := monitor.outcome(ctx); err != nil {
			t.Fatal(err)
		}
		t.Log("COLD DIAGNOSTIC ONLY: no full-eight/warm-rate/LCP/growth/default credit; cold300 remains the acceptance budget")
		return
	}
	results["raw_responses"] = responses
	if awake {
		first, ok := firstStreams["cold"]
		results["cold_first_stream_ms"], results["cold_first_stream_limit_ms"] = first, 90000
		results["total_completion_report_only"], results["first_visible_report_only"] = true, true
		if err := cacheKnowledgeAwakeFirstStreamGate(first, ok); err != nil {
			t.Error(err)
		}
	}
	cohort := map[string]map[string]any{}
	for label, response := range responses {
		timings, err := cacheKnowledgeTimings(response)
		if err != nil {
			t.Fatalf("%s timings: %v", label, err)
		}
		cohort[label] = timings
		results[label] = timings
		t.Logf("%s timings=%v", label, timings)
	}
	previous := responses["cold"]
	for _, label := range []string{"turn_two", "after_title", "after_subagent"} {
		turn := responses[label]
		check := cacheKnowledgeResidentReuse
		if awake {
			check = cacheKnowledgeAwakeReuse
		}
		if err := check(previous, turn); err != nil {
			t.Errorf("%s: %v", label, err)
		}
		previous = turn
	}
	check := cacheKnowledgeResidentReuse
	if awake {
		check = cacheKnowledgeAwakeReuse
	}
	if err := check(responses["subagent"], responses["subagent_two"]); err != nil {
		t.Errorf("subagent_two: %v", err)
	}
	if awake {
		probe, err := cacheKnowledgeThinkingProbe(ctx, h, name)
		results["thinking_template_observations"] = probe
		if err != nil {
			t.Errorf("same-cohort template observation: %v", err)
		}
	}
	for _, label := range []string{"cold", "turn_two", "after_title", "after_subagent", "subagent_two"} {
		timings := cohort[label]
		results[label+"_tokens_per_second"] = timings["predicted_n"].(float64) * 1000 / timings["predicted_ms"].(float64)
	}
	lastCommit, err := cacheKnowledgePrivateCommit(ctx, pid)
	if err != nil {
		t.Fatal(err)
	}
	growth := int64(lastCommit) - int64(readyCommit)
	growthLimit := forecast - object.Size + (256 << 20)
	results["private_commit_last_turn_bytes"], results["private_commit_growth_bytes"], results["private_commit_growth_limit_bytes"] = lastCommit, growth, growthLimit
	if growth > growthLimit {
		t.Errorf("engine private commit growth %d exceeds cache forecast+tolerance %d", growth, growthLimit)
	}
	h.Close()
	collectExit()
	if engineExit == nil {
		t.Error("engine exit memory accounting unavailable")
	} else {
		results["engine_peak_process_bytes"] = engineExit.PeakProcessMemory
		results["engine_peak_job_bytes"] = engineExit.PeakJobMemory
		results["engine_process_limit_bytes"] = engineExit.ProcessMemoryLimit
		results["engine_memory_limit_hit"] = engineExit.MemoryLimitHit
		results["engine_peak_minus_fit_need_bytes"] = int64(engineExit.PeakJobMemory) - entry.NeedBytes
		if err := cacheKnowledgeEngineOutcome(*engineExit, memoryLimit); err != nil {
			t.Error(err)
		}
	}
	stopMonitor()
	if err := monitor.outcome(ctx); err != nil {
		t.Fatal(err)
	}
	if err := log.Sync(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(out, "engine.log"))
	if err != nil {
		t.Fatal(err)
	}
	mainSlot, titleSlot, slotErr := cacheKnowledgeConcurrentSlots(string(raw), responses["cold"], responses["concurrent_title"])
	results["concurrent_main_slot"], results["concurrent_title_slot"] = mainSlot, titleSlot
	if slotErr != nil {
		t.Error(slotErr)
	}
	var idleCache []string
	var draft []string
	var buffers []string
	haveRecurrent, haveCompute := false, false
	for _, line := range strings.Split(string(raw), "\n") {
		lower := strings.ToLower(line)
		if strings.Contains(lower, "get_availabl") || strings.Contains(lower, "load") && strings.Contains(lower, "cache") {
			idleCache = append(idleCache, line)
		}
		if strings.Contains(lower, "buffer size") || strings.Contains(lower, "llama_memory_recurrent") || strings.Contains(lower, "n_slots") || strings.Contains(lower, "n_ctx_slot") || strings.Contains(lower, "kv_unified") || strings.Contains(lower, "n_seq_max") {
			buffers = append(buffers, line)
		}
		haveRecurrent = haveRecurrent || strings.Contains(lower, "rs buffer size")
		haveCompute = haveCompute || strings.Contains(lower, "compute buffer size")
		if strings.Contains(lower, "draft") && strings.Contains(lower, "accept") {
			draft = append(draft, line)
		}
	}
	results["idle_cache_selection_lines"] = idleCache
	results["draft_acceptance_lines"] = draft
	results["production_buffer_lines"] = buffers
	if !haveRecurrent || !haveCompute {
		t.Error("production recurrent/compute buffer evidence unavailable")
	}
	if len(draft) == 0 {
		t.Error("no draft-acceptance lines in stock engine log")
	}
	encoded, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "knowledge.json"), encoded, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("production cache measurements: %s", encoded)
}
