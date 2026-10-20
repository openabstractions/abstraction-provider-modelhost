package host

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/openabstractions/abstraction-provider-kit/go/enginehttp"
	"io"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func cacheKnowledgeThinkingProbe(ctx context.Context, h *Host, name string) (map[string]any, error) {
	client := engineClient(h, nil, enginehttp.NewUnixTransport(func(host string) (enginehttp.UnixEndpoint, bool) { return h.EngineEndpoint(host) }))
	observations := map[string]any{}
	for _, label := range []string{"absent", "true", "false"} {
		body := cacheKnowledgeTemplate("preserve_thinking", true)
		if label != "absent" {
			body["chat_template_kwargs"] = map[string]bool{"preserve_thinking": true, "enable_thinking": label == "true"}
		}
		request, err := cacheKnowledgeRequest(ctx, h.Base(name), applyTemplatePath, body)
		if err != nil {
			return observations, err
		}
		response, err := client.Do(request)
		if err != nil {
			return observations, err
		}
		raw, readErr := io.ReadAll(io.LimitReader(response.Body, 65537))
		response.Body.Close()
		if readErr != nil || len(raw) > 65536 {
			return observations, errors.New("template observation unavailable or over bound")
		}
		row := map[string]any{"status": response.StatusCode, "response_bytes": len(raw)}
		observations[label] = row
		if response.StatusCode != http.StatusOK {
			row["render_availability"] = "refused"
			continue
		}
		var value map[string]any
		if err := json.Unmarshal(raw, &value); err != nil {
			return observations, errors.New("template observation malformed")
		}
		prompt, err := cacheKnowledgeRendered(value)
		if err != nil {
			return observations, err
		}
		suffix := []rune(prompt)
		if len(suffix) > 1024 {
			suffix = suffix[len(suffix)-1024:]
		}
		row["prompt_bytes"], row["prompt_sha256"], row["synthetic_prompt_suffix"] = len(prompt), fmt.Sprintf("%x", sha256.Sum256([]byte(prompt))), string(suffix)
		row["render_availability"] = "observed_not_yet_interpreted"
	}
	return observations, nil
}

func (c cacheKnowledgeCheckpoint) MarshalJSON() ([]byte, error) {
	return json.Marshal([3]int{c.first, c.last, c.tokens})
}

func cacheKnowledgeAwakeFirstStreamGate(first int64, observed bool) error {
	if !observed || first < 0 || first > 90000 {
		return fmt.Errorf("awake cold24k first streamed token %dms (observed=%v), want <=90000ms", first, observed)
	}
	return nil
}

func TestCacheKnowledgeAwakeFirstStreamGate(t *testing.T) {
	for _, first := range []int64{0, 89999, 90000} {
		if err := cacheKnowledgeAwakeFirstStreamGate(first, true); err != nil {
			t.Fatal(err)
		}
	}
	for _, row := range []struct {
		first    int64
		observed bool
	}{{90001, true}, {-1, true}, {0, false}} {
		if err := cacheKnowledgeAwakeFirstStreamGate(row.first, row.observed); err == nil {
			t.Fatal(row)
		}
	}
}

// These are the pinned b10825 checkpoint creation/deletion records, not a
// widened LCP tolerance. The source selection algorithm remains above.
func cacheKnowledgeObservedCheckpoints(path string, slot int) ([]cacheKnowledgeCheckpoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, (32<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(raw) > 32<<20 {
		return nil, errors.New("checkpoint log bound exceeded")
	}
	return cacheKnowledgeParseCheckpoints(string(raw), slot)
}

func cacheKnowledgeParseCheckpoints(raw string, slot int) ([]cacheKnowledgeCheckpoint, error) {
	linePattern := regexp.MustCompile(`id\s+(\d+) \| task [-\d]+ \| (.*)`)
	checkpointPattern := regexp.MustCompile(`pos_min = (\d+), pos_max = (\d+), n_tokens = (\d+)`)
	var checkpoints []cacheKnowledgeCheckpoint
	for _, line := range strings.Split(raw, "\n") {
		match := linePattern.FindStringSubmatch(line)
		if match == nil || match[1] != strconv.Itoa(slot) {
			continue
		}
		text := match[2]
		if strings.HasPrefix(text, "clearing prompt with ") {
			checkpoints = nil
			continue
		}
		created := strings.HasPrefix(text, "created context checkpoint ")
		deleted := strings.HasPrefix(text, "erasing old context checkpoint ") || strings.HasPrefix(text, "erasing context checkpoint too close ") || strings.HasPrefix(text, "erased invalidated context checkpoint ")
		if !created && !deleted {
			continue
		}
		fields := checkpointPattern.FindStringSubmatch(text)
		if fields == nil {
			return nil, errors.New("malformed pinned checkpoint record")
		}
		values := [3]int{}
		for i := range values {
			value, err := strconv.Atoi(fields[i+1])
			if err != nil {
				return nil, err
			}
			values[i] = value
		}
		checkpoint := cacheKnowledgeCheckpoint{values[0], values[1], values[2]}
		if checkpoint.last < checkpoint.first || checkpoint.tokens <= 0 {
			return nil, errors.New("invalid checkpoint dimensions")
		}
		if created {
			checkpoints = append(checkpoints, checkpoint)
			if len(checkpoints) > 2 {
				return nil, errors.New("checkpoint deletion record unavailable")
			}
		} else {
			found := false
			for i, value := range checkpoints {
				if value == checkpoint {
					checkpoints = append(checkpoints[:i], checkpoints[i+1:]...)
					found = true
					break
				}
			}
			if !found {
				return nil, errors.New("checkpoint deletion has no observed creation")
			}
		}
	}
	return checkpoints, nil
}

func cacheKnowledgeAwakeReuse(previous, current map[string]any) error {
	shared, total := current["rendered_shared_prefix_tokens"].(float64), current["rendered_prompt_tokens"].(float64)
	prior := previous["rendered_prompt_tokens"].(float64)
	if shared < 2048 || shared > min(prior, total) || shared < previous["fixture_user_tokens"].(float64) || previous["id_slot"] != current["id_slot"] {
		return errors.New("awake reuse lost stable user prefix or resident slot")
	}
	checkpoints, ok := previous["checkpoint_metadata"].([]cacheKnowledgeCheckpoint)
	if !ok {
		return errors.New("observed prior checkpoint metadata unavailable")
	}
	frontier := cacheKnowledgeCheckpointFrontier(int(shared), int(total), checkpoints)
	current["eligible_checkpoint_tokens"] = frontier
	current["checkpoint_reevaluated_prefix_tokens"] = int(shared) - frontier
	timings := current["timings"].(map[string]any)
	n, ms := timings["predicted_n"].(float64), timings["predicted_ms"].(float64)
	current["decode_context_prompt_tokens"], current["decode_output_tokens"] = total, n
	var failures []error
	if err := cacheKnowledgeCheckpointReuse(current, checkpoints); err != nil {
		failures = append(failures, err)
	}
	if n < 64 || ms <= 0 || n*1000/ms < 40 {
		failures = append(failures, fmt.Errorf("awake decode at prompt context %.0f tokens: n=%v ms=%v; want >=40 tokens/s", total, n, ms))
	}
	return errors.Join(failures...)
}

func TestCacheKnowledgeAwakeCheckpointGate(t *testing.T) {
	raw := "I slot create_check: id 1 | task 2 | created context checkpoint 1 of 2 (pos_min = 23511, pos_max = 23511, n_tokens = 23512, size = 109.0 MiB)\n" +
		"I slot create_check: id 1 | task 2 | created context checkpoint 2 of 2 (pos_min = 24023, pos_max = 24023, n_tokens = 24024, size = 110.0 MiB)\n"
	checkpoints, err := cacheKnowledgeParseCheckpoints(raw, 1)
	if err != nil || len(checkpoints) != 2 {
		t.Fatal(checkpoints, err)
	}
	prior := map[string]any{"rendered_prompt_tokens": float64(24028), "fixture_user_tokens": float64(24000), "id_slot": float64(1), "checkpoint_metadata": checkpoints}
	current := map[string]any{"rendered_shared_prefix_tokens": float64(24021), "rendered_prompt_tokens": float64(24035), "id_slot": float64(1), "timings": map[string]any{"cache_n": float64(23512), "prompt_n": float64(523), "predicted_n": float64(128), "predicted_ms": float64(3200)}}
	if err := cacheKnowledgeAwakeReuse(prior, current); err != nil {
		t.Fatal(err)
	}
	if current["checkpoint_reevaluated_prefix_tokens"] != 509 {
		t.Fatal(current)
	}
	timings := current["timings"].(map[string]any)
	timings["cache_n"], timings["prompt_n"], timings["predicted_ms"] = float64(23511), float64(524), float64(3201)
	if err := cacheKnowledgeAwakeReuse(prior, current); err == nil || !strings.Contains(err.Error(), "checkpoint reuse") || !strings.Contains(err.Error(), "context 24035") {
		t.Fatal(err)
	}
	if _, err := cacheKnowledgeParseCheckpoints(raw+raw, 1); err == nil {
		t.Fatal("unobserved deletion passed")
	}
	if other, err := cacheKnowledgeParseCheckpoints(raw, 0); err != nil || len(other) != 0 {
		t.Fatal(other, err)
	}
	deleted := raw + "I slot checkpoint: id 1 | task 3 | erased invalidated context checkpoint (pos_min = 24023, pos_max = 24023, n_tokens = 24024, n_swa = 0, pos_next = 24021, size = 110.0 MiB)\n"
	if observed, err := cacheKnowledgeParseCheckpoints(deleted, 1); err != nil || len(observed) != 1 {
		t.Fatal(observed, err)
	}
}

func TestLiveCacheAwakeKnowledge(t *testing.T) {
	if os.Getenv("OA_LIVE_CACHE_AWAKE_737") != "1" {
		t.Skip("requires the bounded awake737 owner")
	}
	if os.Getenv("OA_LIVE_POWER_GUARDED") != "1" {
		t.Fatal("native mains/display validity owner required")
	}
	liveProductionCacheKnowledge(t, false, true)
}
