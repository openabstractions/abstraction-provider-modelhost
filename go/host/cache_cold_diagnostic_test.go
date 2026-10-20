package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

type cacheKnowledgeRoundTripFunc func(*http.Request) (*http.Response, error)

func (f cacheKnowledgeRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCacheKnowledgeColdDiagnosticHTTPFramingAndReplay(t *testing.T) {
	received := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil || r.ContentLength != int64(len(body)) {
			t.Error("real HTTP body framing mismatch", err)
		}
		received <- body
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	wireTransport := &http.Transport{}
	defer wireTransport.CloseIdleConnections()
	var sent *http.Request
	next := cacheKnowledgeRoundTripFunc(func(r *http.Request) (*http.Response, error) { sent = r; return wireTransport.RoundTrip(r) })
	key, err := cacheKnowledgeMessagesKey([]wire.ChatMessage{{Role: wire.RoleUser, Parts: []wire.ChatInputPart{{Kind: wire.PartKindText, Text: "synthetic"}}}})
	if err != nil {
		t.Fatal(err)
	}
	transport := &cacheKnowledgeResidentTransport{next: next, diagnostic: newCacheKnowledgeColdObserver(), traces: map[string]*mtpChatTrace{}, slots: map[string]int{}, bodies: map[string][]byte{}, pending: map[string][]string{key: {"cold"}}}
	original := []byte(`{"id_slot":0,"messages":[{"role":"user","content":"synthetic"}],"max_tokens":4096}`)
	request, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	response, err := transport.RoundTrip(request)
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, response.Body)
	response.Body.Close()
	if sent == nil || sent.ContentLength <= int64(len(original)) || sent.GetBody == nil {
		t.Fatal("rewritten body retained stale framing or replay")
	}
	replay, err := sent.GetBody()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(replay)
	replay.Close()
	if err != nil || !bytes.Equal(raw, <-received) || sent.ContentLength != int64(len(raw)) {
		t.Fatal("replay differs from real HTTP body", err)
	}
	var body map[string]any
	if json.Unmarshal(raw, &body) != nil || body["timings_per_token"] != true || body["max_tokens"] != float64(4096) {
		t.Fatal("diagnostic rewrite changed body semantics")
	}
}

func TestCacheKnowledgeColdDiagnosticFailureEvidence(t *testing.T) {
	for _, failure := range []string{"tokenizer", "observer", "missing_stream"} {
		t.Run(failure, func(t *testing.T) {
			h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(), engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(2)}}}
			transport := inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == tokenizePath {
					if failure == "tokenizer" {
						w.WriteHeader(http.StatusForbidden)
						fmt.Fprint(w, "DO_NOT_LOG")
						return
					}
					json.NewEncoder(w).Encode(map[string]any{"tokens": []int64{1}})
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				text := "public thought"
				if failure == "observer" {
					text = strings.Repeat("DO_NOT_LOG", 8000)
				}
				chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"reasoning_content": text}}}})
				fmt.Fprintf(w, "data: %s\n\n", chunk)
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"predicted_n\":8}}\n\ndata: [DONE]\n\n")
			})}
			next := cacheKnowledgeRoundTripFunc(func(r *http.Request) (*http.Response, error) {
				if failure == "missing_stream" {
					return nil, errors.New("DO_NOT_LOG")
				}
				return transport.RoundTrip(r)
			})
			var mu sync.Mutex
			rows := 0
			_, err := cacheKnowledgeResidentSequenceMode(t.Context(), h, "fixture", next, func(context.Context, *router.Host, string) (string, bool) { return "", true }, 4096, true, newCacheKnowledgeColdObserver(), func(_ string, row map[string]any) error {
				mu.Lock()
				defer mu.Unlock()
				rows++
				measurement, ok := row["cold_measurement"].(map[string]any)
				if !ok || measurement["availability"] == "complete" || measurement["error_category"] == "none" || row["terminal_outcome"] == "not_observed" || row["diagnostic_only"] != true || row["acceptance_credit"] != false {
					t.Error("failure lost partial measurement or typed terminal")
				}
				raw, _ := json.Marshal(row)
				if len(raw) > 8192 || bytes.Contains(raw, []byte("DO_NOT_LOG")) {
					t.Error("failure leaked raw content/error or exceeded diagnostic bound")
				}
				return nil
			})
			if err == nil || rows != 2 {
				t.Fatalf("failure did not fail after persisting both terminals: rows=%d err=%v", rows, err)
			}
		})
	}
}

// A distinct opt-in test entry keeps the report-only diagnostic out of the
// full-eight qualification path even if a caller inherits this environment.
func TestLiveCacheColdDiagnostic(t *testing.T) {
	if os.Getenv("OA_LIVE_CACHE_COLD_DIAGNOSTIC") != "1" {
		t.Skip("private architect-selected cold diagnostic requires its bounded owner")
	}
	liveProductionCacheKnowledge(t, true)
}

// Private synthetic-stream observer for the architect's single cold-pair
// measurement. It changes no request options or production stream handling.
// Arrival times bracket the hidden reasoning-end delimiter; they do not claim
// the engine's exact token timestamp. Text token counts are explicitly a
// retokenization, kept separate from final engine-generated-token accounting.
type cacheKnowledgeColdObserver struct {
	mu    sync.Mutex
	start time.Time
	calls map[string]*cacheKnowledgeColdCall
}

type cacheKnowledgeColdCall struct {
	pending                         []byte
	reasoning, visible              strings.Builder
	lastReasoningMS, firstVisibleMS int64
	finish                          string
	reasoningEvents, visibleEvents  int
	overflow                        bool
	timings                         map[string]any
	points                          [][2]int64
	firstGeneratedMS, finishMS      int64
	lastReasoningN, firstVisibleN   int64
}

func newCacheKnowledgeColdObserver() *cacheKnowledgeColdObserver {
	return &cacheKnowledgeColdObserver{start: time.Now(), calls: map[string]*cacheKnowledgeColdCall{}}
}

type cacheKnowledgeColdBody struct {
	io.ReadCloser
	owner *cacheKnowledgeColdObserver
	label string
}

func (o *cacheKnowledgeColdObserver) body(label string, body io.ReadCloser) io.ReadCloser {
	o.mu.Lock()
	o.calls[label] = &cacheKnowledgeColdCall{lastReasoningMS: -1, firstVisibleMS: -1, firstGeneratedMS: -1, finishMS: -1, lastReasoningN: -1, firstVisibleN: -1, finish: "not_observed"}
	o.mu.Unlock()
	return &cacheKnowledgeColdBody{ReadCloser: body, owner: o, label: label}
}

func (b *cacheKnowledgeColdBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.owner.observe(b.label, p[:n], time.Since(b.owner.start).Milliseconds())
	return n, err
}

func (o *cacheKnowledgeColdObserver) observe(label string, raw []byte, ms int64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	c := o.calls[label]
	if c == nil || c.overflow {
		return
	}
	c.pending = append(c.pending, raw...)
	if len(c.pending) > 65536 {
		c.overflow = true
		return
	}
	for {
		line, rest, found := bytes.Cut(c.pending, []byte("\n"))
		if !found {
			return
		}
		c.pending = rest
		data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data: "))
		if !ok || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var event struct {
			Timings map[string]any `json:"timings"`
			Choices []struct {
				Delta struct {
					Reasoning string `json:"reasoning_content"`
					Alias     string `json:"reasoning"`
					Content   string `json:"content"`
				} `json:"delta"`
				Finish *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal(data, &event) != nil {
			c.overflow = true
			return
		}
		if event.Timings != nil {
			c.timings = event.Timings
		}
		n := int64(-1)
		if count, ok := event.Timings["predicted_n"].(float64); ok {
			n = int64(count)
			if n > 0 && c.firstGeneratedMS < 0 {
				c.firstGeneratedMS = ms
			}
			if len(c.points) == 0 || ms-c.points[len(c.points)-1][0] >= 5000 {
				if len(c.points) >= 160 {
					c.overflow = true
					return
				}
				c.points = append(c.points, [2]int64{ms, n})
			}
		}
		for _, choice := range event.Choices {
			reasoning := choice.Delta.Reasoning
			if reasoning == "" {
				reasoning = choice.Delta.Alias
			}
			if reasoning != "" {
				c.reasoning.WriteString(reasoning)
				c.reasoningEvents++
				c.lastReasoningMS = ms
				c.lastReasoningN = n
			}
			if choice.Delta.Content != "" {
				if c.firstVisibleMS < 0 {
					c.firstVisibleMS = ms
					c.firstVisibleN = n
				}
				c.visible.WriteString(choice.Delta.Content)
				c.visibleEvents++
			}
			if choice.Finish != nil {
				c.finish = *choice.Finish
				c.finishMS = ms
			}
		}
		if c.reasoning.Len()+c.visible.Len() > 256<<10 {
			c.overflow = true
			return
		}
	}
}

func (o *cacheKnowledgeColdObserver) metadata(label string) (map[string]any, string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	c := o.calls[label]
	if c == nil {
		return nil, "not_observed"
	}
	return c.timings, c.finish
}

func cacheKnowledgeRepeatedNgrams(text string, n int) int {
	words := strings.Fields(text)
	seen := map[string]bool{}
	repeats := 0
	for i := 0; i+n <= len(words); i++ {
		key := strings.Join(words[i:i+n], " ")
		if seen[key] {
			repeats++
		}
		seen[key] = true
	}
	return repeats
}

func (o *cacheKnowledgeColdObserver) report(ctx context.Context, label string, client *http.Client, base string) (map[string]any, error) {
	o.mu.Lock()
	c := o.calls[label]
	if c == nil {
		o.mu.Unlock()
		return map[string]any{"availability": "unavailable", "error_category": "missing_stream", "cold300_missed": time.Since(o.start) > cacheKnowledgeColdBudget}, errors.New("cold diagnostic stream unavailable")
	}
	reasoning, visible := c.reasoning.String(), c.visible.String()
	result := map[string]any{"last_reasoning_arrival_ms": c.lastReasoningMS, "first_visible_arrival_ms": c.firstVisibleMS,
		"reasoning_end_exact_token_time": "not_exposed_by_openai_chat_stream", "reasoning_events": c.reasoningEvents,
		"visible_events": c.visibleEvents, "finish_reason": c.finish, "overflow": c.overflow,
		"cold300_missed": time.Since(o.start) > cacheKnowledgeColdBudget}
	result["reasoning_end_generated_counter_bracket"] = [2]int64{c.lastReasoningN, c.firstVisibleN}
	result["first_generated_arrival_ms"], result["finish_arrival_ms"] = c.firstGeneratedMS, c.finishMS
	result["generated_progress_5s_samples"] = append([][2]int64(nil), c.points...)
	result["availability"], result["error_category"] = "complete", "none"
	result["reasoning_bytes"], result["visible_bytes"] = len(reasoning), len(visible)
	o.mu.Unlock()
	if result["overflow"] == true {
		result["availability"], result["error_category"] = "partial", "observer_invalid_or_bound"
		return result, errors.New("cold diagnostic observer bound or JSON failure")
	}
	// Retokenize only fixture-generated public fields; no seal bytes or owner
	// transcript are read. Cancellation keeps already-observed evidence honest.
	for kind, text := range map[string]string{"reasoning": reasoning, "visible": visible} {
		result[kind+"_bytes"] = len(text)
		if ctx.Err() != nil {
			result["availability"], result["error_category"] = "partial", "cancelled_tokenization"
			result[kind+"_retokenized_tokens"] = "unavailable_after_cancellation"
			continue
		}
		body, err := json.Marshal(map[string]any{"content": text, "add_special": false, "parse_special": false})
		if err != nil {
			result["availability"], result["error_category"] = "partial", "tokenizer_encoding"
			return result, errors.New("cold diagnostic tokenizer encoding failed")
		}
		var tokenized struct {
			Tokens []int64 `json:"tokens"`
		}
		if err := postEngine(ctx, client, base+tokenizePath, body, &tokenized); err != nil {
			result["availability"], result["error_category"] = "partial", "tokenizer_unavailable"
			return result, errors.New("cold diagnostic tokenizer unavailable")
		}
		result[kind+"_retokenized_tokens"] = len(tokenized.Tokens)
	}
	// Existing bounded suffix has a separate count, never a whole-stream claim.
	raw, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"reasoning_content": reasoning}}}})
	suffix := cacheKnowledgeSyntheticReasoningTail(append(append([]byte("data: "), raw...), '\n'))
	words := strings.Fields(suffix)
	result["reasoning_suffix_words"] = len(words)
	for _, n := range []int{4, 8, 16} {
		result["suffix_repeated_"+strconv.Itoa(n)+"_grams"] = cacheKnowledgeRepeatedNgrams(suffix, n)
	}
	// Reuse the existing public-only suffix helper, never the replay field.
	result["synthetic_reasoning_tail"] = suffix
	return result, nil
}

func TestCacheKnowledgeColdObserver(t *testing.T) {
	o := newCacheKnowledgeColdObserver()
	o.body("cold", io.NopCloser(strings.NewReader("")))
	first := []byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"one two three four one two three four\",\"reasoning\":\"alias ignored\"}}]}\n\n")
	o.observe("cold", first[:17], 10)
	o.observe("cold", first[17:], 20)
	o.observe("cold", []byte("data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n"), 30)
	c := o.calls["cold"]
	if c.reasoningEvents != 1 || c.visibleEvents != 1 || c.lastReasoningMS != 20 || c.firstVisibleMS != 30 || c.finish != "stop" || c.reasoning.String() != "one two three four one two three four" || cacheKnowledgeRepeatedNgrams(c.reasoning.String(), 4) != 1 {
		t.Fatal("observer lost split/alias/phase/finish evidence")
	}
}

func TestCacheKnowledgeColdDiagnosticStopsAfterOriginalPair(t *testing.T) {
	for _, visible := range []bool{true, false} {
		t.Run(strconv.FormatBool(visible), func(t *testing.T) {
			h := &Host{names: []string{"fixture"}, bases: map[string]string{"fixture": "http://engine"}, warm: newWarmer(),
				engines: map[string]*Process{"engine": {key: "fixture-key", cacheSlots: newCacheSlots(2)}}}
			var mu sync.Mutex
			requests, reports := 0, 0
			transport := inProcess{handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == tokenizePath {
					json.NewEncoder(w).Encode(map[string]any{"tokens": []int64{1, 2}})
					return
				}
				if r.URL.Path != "/v1/chat/completions" {
					t.Errorf("unexpected diagnostic endpoint %s", r.URL.Path)
					w.WriteHeader(400)
					return
				}
				var body struct {
					Messages []struct {
						Content string `json:"content"`
					} `json:"messages"`
					Temperature *float64 `json:"temperature"`
					Max         int64    `json:"max_tokens"`
					Timings     bool     `json:"timings_per_token"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body.Max != 4096 || body.Temperature != nil || len(body.Messages) != 1 || !body.Timings {
					t.Error("diagnostic changed request shape")
				}
				if !strings.Contains(body.Messages[0].Content, "Synthetic long conversation") && !strings.Contains(body.Messages[0].Content, "title topic delta") {
					t.Error("diagnostic substituted original prompt")
				}
				mu.Lock()
				requests++
				mu.Unlock()
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"public thought\"}}]}\n\n")
				if visible {
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n")
				}
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"timings\":{\"prompt_n\":1,\"cache_n\":0,\"prompt_ms\":1,\"predicted_n\":8,\"predicted_ms\":1000,\"draft_n\":8,\"draft_n_accepted\":5}}\n\ndata: [DONE]\n\n")
			})}
			observer := newCacheKnowledgeColdObserver()
			values, err := cacheKnowledgeResidentSequenceMode(t.Context(), h, "fixture", transport,
				func(context.Context, *router.Host, string) (string, bool) { return "", true }, 4096, true, observer,
				func(_ string, value map[string]any) error {
					mu.Lock()
					defer mu.Unlock()
					reports++
					want := wire.ChatResultOutcomeCompleted.String()
					if !visible {
						want = wire.ChatResultOutcomeFailed.String()
					}
					if value["terminal_outcome"] != want || value["cold_measurement"] == nil || value["diagnostic_only"] != true || value["acceptance_credit"] != false {
						t.Error("diagnostic lost actual terminal/measurement")
					}
					return nil
				})
			if err != nil || requests != 2 || reports != 2 || len(values) != 2 {
				t.Fatalf("diagnostic ran followups or lost evidence: requests=%d reports=%d values=%d err=%v", requests, reports, len(values), err)
			}
		})
	}
}
