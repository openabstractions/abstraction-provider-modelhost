package host

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

// The chat prefix preload (CONTRACT.md MODELHOST-W1..MODELHOST-W3; inference
// INF-PL1..INF-PL4).
//
// An application sends, through preload@1, the messages and tools its next
// chat request will open with. The provider makes the model's engine evaluate
// them, so that request begins where they end. llama-server reuses its KV
// cache only for an identical token prefix, and a hybrid or sliding-window
// model resumes only from a context checkpoint at or before the first
// differing token (b10825 tools/server/server-context.cpp:3283-3369).
//
// The engine itself renders and tokenizes the prefix: /apply-template runs
// the same oaicompat_chat_params_parse as /v1/chat/completions
// (server-context.cpp:4914-4928, 5044-5054), and /tokenize with add_special
// and parse_special calls the same tokenize_mixed the chat path calls
// (server-context.cpp:5068-5077; server-common.cpp:850-880, 964-971). No
// template is rendered and no tokenizer runs in this process. The provider
// keeps nothing of the prefix after the call.

const (
	// chatPath is the route every chat request reaches the engine on.
	chatPath = "/v1/chat/completions"
	// applyTemplatePath renders a chat request without running it.
	applyTemplatePath = "/apply-template"
	// tokenizePath tokenizes text as the chat path does.
	tokenizePath = "/tokenize"
	// completionPath runs a prompt of token ids.
	completionPath = "/completion"
	// minWarmTokens is the shortest prefix worth evaluating: a shorter one
	// costs the engine more in context checkpoints than it saves in prefill.
	minWarmTokens = 256
	// maxWarmReply bounds a rendered prompt or a token list read back.
	maxWarmReply = 64 << 20
	// exitConfirm is how long a preload whose engine connection broke
	// waits for the engine's exit to be confirmed before it reads the break
	// as unreachable.
	exitConfirm = 3 * time.Second
)

// warmPlaceholders are two user messages that share no first character. The
// tokens their two renderings after the prefix share are the tokens every
// request continuing the prefix begins with, up to where the user's words
// start.
var warmPlaceholders = [2]string{"A", "B"}

// unrenderedFields are request fields that change sampling, streaming or the
// answer's length and never the rendered prompt. Every other field stays in
// the prefix: tools, tool_choice, response_format, reasoning_effort and any
// template argument decide what the engine renders.
var unrenderedFields = map[string]bool{
	"stream": true, "stream_options": true, "max_tokens": true, "max_completion_tokens": true, "n_predict": true,
	"temperature": true, "top_p": true, "top_k": true, "min_p": true, "seed": true, "stop": true, "user": true,
	"cache_prompt": true, "n_probs": true, "logprobs": true, "top_logprobs": true,
	"presence_penalty": true, "frequency_penalty": true, "repeat_penalty": true,
}

// chatPrefix is a chat request's rendered fields and the messages a
// conversation opens with, verbatim.
type chatPrefix struct {
	// key is the prefix as canonical JSON: two equal keys render the same
	// tokens up to the next user message.
	key      []byte
	fields   map[string]json.RawMessage
	messages []json.RawMessage
}

// readChatBody reads a chat request body into its rendered fields and its
// messages; ok is false for a body that is no chat request.
func readChatBody(raw []byte) (map[string]json.RawMessage, []json.RawMessage, bool) {
	var body map[string]json.RawMessage
	if json.Unmarshal(raw, &body) != nil || body == nil {
		return nil, nil, false
	}
	var messages []json.RawMessage
	if json.Unmarshal(body["messages"], &messages) != nil || len(messages) == 0 {
		return nil, nil, false
	}
	fields := map[string]json.RawMessage{}
	for name, value := range body {
		if name != "messages" && !unrenderedFields[name] {
			fields[name] = value
		}
	}
	return fields, messages, true
}

// newChatPrefix is the prefix of fields and messages.
func newChatPrefix(fields map[string]json.RawMessage, messages []json.RawMessage) (*chatPrefix, error) {
	prefix := &chatPrefix{fields: fields, messages: messages}
	key, err := prefix.request(nil)
	if err != nil {
		return nil, err
	}
	prefix.key = key
	return prefix, nil
}

// continuedKey is the key of the prefix a chat request body continues: its
// messages before its last one, when the last is the user's. It is nil for
// any other body.
func continuedKey(raw []byte) []byte {
	fields, messages, ok := readChatBody(raw)
	if !ok || messageRole(messages[len(messages)-1]) != "user" {
		return nil
	}
	prefix, err := newChatPrefix(fields, messages[:len(messages)-1])
	if err != nil {
		return nil
	}
	return prefix.key
}

func messageRole(raw json.RawMessage) string {
	var m struct {
		Role string `json:"role"`
	}
	//unchecked: a message that is not an object has no role
	json.Unmarshal(raw, &m)
	return m.Role
}

// request is the prefix as a chat request body: its fields, its messages
// and, when user is set, one user message holding it.
func (p *chatPrefix) request(user *string) ([]byte, error) {
	body := make(map[string]any, len(p.fields)+1)
	for name, value := range p.fields {
		body[name] = value
	}
	messages := make([]any, 0, len(p.messages)+1)
	for _, message := range p.messages {
		messages = append(messages, message)
	}
	if user != nil {
		messages = append(messages, map[string]string{"role": "user", "content": *user})
	}
	body["messages"] = messages
	return encodeJSON(body)
}

// errNothingToWarm is a prefix shorter than minWarmTokens; the engine is
// sent nothing to evaluate.
var errNothingToWarm = errors.New("the prefix is too short to evaluate")

// errNotPosted is a preload that a request overtook before its evaluation
// was posted; the engine was sent nothing to evaluate.
var errNotPosted = errors.New("a request reached the engine before the evaluation was posted")

// engineRefusal is the engine's own refusal of a request of the warm.
type engineRefusal struct{ status int }

func (e engineRefusal) Error() string { return fmt.Sprintf("the engine answered %d", e.status) }

// warmTokens asks the engine at base for the tokens prefix renders to up to
// the next user message: the tokens the renderings of both placeholders
// share.
func warmTokens(ctx context.Context, client *http.Client, base string, prefix *chatPrefix) ([]int64, error) {
	var rendered [2][]int64
	for i := range warmPlaceholders {
		body, err := prefix.request(&warmPlaceholders[i])
		if err != nil {
			return nil, err
		}
		var template struct {
			Prompt *string `json:"prompt"`
		}
		if err := postEngine(ctx, client, base+applyTemplatePath, body, &template); err != nil {
			return nil, err
		}
		if template.Prompt == nil {
			return nil, errors.New("the engine rendered no prompt")
		}
		// The chat path tokenizes its prompt with special tokens added and
		// parsed (tokenize_input_prompts(..., true, true)); so does this.
		request, err := json.Marshal(map[string]any{"content": *template.Prompt, "add_special": true, "parse_special": true})
		if err != nil {
			return nil, err
		}
		var tokenized struct {
			Tokens []int64 `json:"tokens"`
		}
		if err := postEngine(ctx, client, base+tokenizePath, request, &tokenized); err != nil {
			return nil, err
		}
		rendered[i] = tokenized.Tokens
	}
	n := 0
	for n < len(rendered[0]) && n < len(rendered[1]) && rendered[0][n] == rendered[1][n] {
		n++
	}
	return rendered[0][:n], nil
}

// warmPrefix makes the engine at base evaluate prefix's tokens and returns
// how many it evaluated. The prompt is token ids, which the engine takes
// verbatim (server-common.cpp:972-975); one token is generated, the least
// /completion produces. The engine's own checkpoints 4 and 4+n_ubatch tokens
// before the prompt's end (server-context.cpp:3543-3562) let a hybrid model
// resume a request whose tokens part from the prefix in its last tokens.
//
// post is asked just before the evaluation is posted; when it answers false,
// nothing is posted and warmPrefix returns errNotPosted.
func warmPrefix(ctx context.Context, client *http.Client, base string, prefix *chatPrefix, post func() bool) (int, error) {
	tokens, err := warmTokens(ctx, client, base, prefix)
	if err != nil {
		return 0, err
	}
	if len(tokens) < minWarmTokens {
		return 0, errNothingToWarm
	}
	request, err := json.Marshal(map[string]any{"prompt": tokens, "n_predict": 1, "cache_prompt": true, "stream": false})
	if err != nil {
		return 0, err
	}
	if !post() {
		return 0, errNotPosted
	}
	if err := postEngine(ctx, client, base+completionPath, request, nil); err != nil {
		return 0, err
	}
	return len(tokens), nil
}

// warmRequest marks the context of the warm's own engine requests, which
// the media guard does not count as a use of the engine.
type warmRequest struct{}

// postEngine posts body to url and decodes a 200 answer into out, when set.
func postEngine(ctx context.Context, client *http.Client, url string, body []byte, out any) error {
	request, err := http.NewRequestWithContext(context.WithValue(ctx, warmRequest{}, true), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	reply := io.LimitReader(response.Body, maxWarmReply)
	if response.StatusCode != http.StatusOK {
		//unchecked: the status decides the error; the body is drained for the connection's reuse
		io.Copy(io.Discard, reply)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return engineRefusal{status: response.StatusCode}
	}
	if out == nil {
		_, err = io.Copy(io.Discard, reply)
	} else {
		err = json.NewDecoder(reply).Decode(out)
	}
	// An answer that arrived after the warm was cancelled is no warm.
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

// warmer holds, per engine, whether a request has run there since it
// started and the one warm in flight. Every field is keyed by the engine
// host of the model's base URL. It holds no prefix beyond the warm in
// flight (MODELHOST-W1).
type warmer struct {
	mu sync.Mutex
	// used is set once a request other than a warm reached the engine since
	// it started: its slot then holds someone's conversation.
	used map[string]bool
	runs map[string]*warmRun
}

// warmRun is one warm in flight.
type warmRun struct {
	key    []byte
	cancel context.CancelFunc
	// why is set under warmer.mu when something other than the caller ended
	// the warm: another request, or the engine's end.
	why string
	// posted is set under warmer.mu just before the evaluation is posted to
	// the engine; until then a request overtakes the warm and the warm posts
	// nothing (MODELHOST-W3).
	posted bool
	// done is closed when the warm's last engine request has returned.
	done chan struct{}
}

func newWarmer() *warmer {
	return &warmer{used: map[string]bool{}, runs: map[string]*warmRun{}}
}

// observe hears one request on its way to host's engine (MODELHOST-W3) and
// returns what the request must wait for before it leaves the provider, or
// nil. The engine counts as used from here on.
//
//   - A warm whose evaluation is not yet posted is stopped: it posts nothing,
//     and the request goes at once.
//   - A chat request that continues the prefix of a warm whose evaluation is
//     posted waits for that evaluation to return, then goes: the engine has
//     the prefix and serves the request from it, and nothing of the warm can
//     run after the request.
//   - Any other request cancels a posted evaluation and waits only until its
//     engine request has returned; the engine drops the cancelled
//     evaluation at its next poll and batch.
func (w *warmer) observe(host, path string, raw []byte) <-chan struct{} {
	var continued []byte
	if path == chatPath {
		continued = continuedKey(raw)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.used[host] = true
	run := w.runs[host]
	if run == nil {
		return nil
	}
	if run.posted && continued != nil && bytes.Equal(continued, run.key) {
		return run.done
	}
	run.why = "request"
	run.cancel()
	delete(w.runs, host)
	if run.posted {
		return run.done
	}
	return nil
}

// engineChanged forgets what host's engine held and cancels its warm: the
// engine started anew or stopped.
func (w *warmer) engineChanged(host string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if run := w.runs[host]; run != nil {
		run.why = "engine"
		run.cancel()
		delete(w.runs, host)
	}
	delete(w.used, host)
}

// evaluate has host's engine at base evaluate the prefix body, unless a
// request has used the engine since it started or a warm runs there
// (MODELHOST-W2). The check and the warm's start are one step under the
// warmer's lock. It returns the preload@1 outcome, its reason and the tokens
// evaluated. exited ends the warm with the engine.
func (w *warmer) evaluate(ctx context.Context, host, base string, client *http.Client, exited <-chan struct{}, body []byte) (wire.PreloadOutcome, string, int64) {
	fields, messages, ok := readChatBody(body)
	if !ok {
		return wire.PreloadOutcomeInvalid, "messages", 0
	}
	prefix, err := newChatPrefix(fields, messages)
	if err != nil {
		return wire.PreloadOutcomeInvalid, "messages", 0
	}
	warmCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	w.mu.Lock()
	switch {
	case w.used[host]:
		w.mu.Unlock()
		return wire.PreloadOutcomeBusy, "engine:used", 0
	case w.runs[host] != nil:
		w.mu.Unlock()
		return wire.PreloadOutcomeBusy, "preload:running", 0
	}
	run := &warmRun{key: prefix.key, cancel: cancel, done: make(chan struct{})}
	w.runs[host] = run
	w.mu.Unlock()
	post := func() bool {
		w.mu.Lock()
		defer w.mu.Unlock()
		if w.runs[host] != run || run.why != "" {
			return false
		}
		run.posted = true
		return true
	}
	go func() {
		select {
		case <-exited:
			w.mu.Lock()
			if run.why == "" {
				run.why = "engine"
			}
			w.mu.Unlock()
			cancel()
		case <-warmCtx.Done():
		}
	}()
	warmed, err := warmPrefix(warmCtx, client, base, prefix, post)
	// The warm's last engine request has returned: a request waiting for it
	// goes now, whatever the warm still has to decide (MODELHOST-W3).
	close(run.done)
	w.mu.Lock()
	unexplained := run.why == ""
	w.mu.Unlock()
	var refusal engineRefusal
	if err != nil && unexplained && ctx.Err() == nil && !errors.As(err, &refusal) && !errors.Is(err, errNothingToWarm) && !errors.Is(err, errNotPosted) {
		// Nobody cancelled the warm, yet its connection broke: most often the
		// engine stopping, whose exit is confirmed a moment later.
		select {
		case <-exited:
			w.mu.Lock()
			if run.why == "" {
				run.why = "engine"
			}
			w.mu.Unlock()
		case <-time.After(exitConfirm):
		}
	}
	w.mu.Lock()
	if w.runs[host] == run {
		delete(w.runs, host)
	}
	why := run.why
	w.mu.Unlock()
	var refused engineRefusal
	switch {
	case err == nil:
		return wire.PreloadOutcomeEvaluated, "", int64(warmed)
	case errors.Is(err, errNothingToWarm):
		return wire.PreloadOutcomeLoaded, "prefix:short", 0
	case why == "request" || errors.Is(err, errNotPosted) && why != "engine":
		// A request reached the engine first and took it over; the warm
		// posted nothing, or the engine dropped what it posted.
		return wire.PreloadOutcomeBusy, "engine:used", 0
	case why == "engine":
		return wire.PreloadOutcomeUnavailable, "engine:stopped", 0
	case ctx.Err() != nil:
		return wire.PreloadOutcomeUnavailable, "caller", 0
	case errors.As(err, &refused) && refused.status >= 400 && refused.status < 500:
		return wire.PreloadOutcomeRefused, fmt.Sprintf("engine:%d", refused.status), 0
	case errors.As(err, &refused):
		return wire.PreloadOutcomeUnavailable, fmt.Sprintf("engine:%d", refused.status), 0
	}
	return wire.PreloadOutcomeUnavailable, "engine:unreachable", 0
}

// EvaluatePrefix is the inference provider's prefix evaluator for this
// host's engines (inference INF-PL2, MODELHOST-W2): the inference provider has
// already loaded model under its lease through Ensure.
func (h *Host) EvaluatePrefix(ctx context.Context, server *router.Host, model string, body []byte) (wire.PreloadOutcome, string, int64) {
	if server == nil || h.bases[model] == "" || server.Base != h.bases[model] {
		return wire.PreloadOutcomeUnsupportedFeature, "preload", 0
	}
	parsed, err := url.Parse(server.Base)
	if err != nil {
		return wire.PreloadOutcomeUnsupportedFeature, "preload", 0
	}
	h.engineMu.Lock()
	p := h.engines[parsed.Host]
	h.engineMu.Unlock()
	if p == nil {
		return wire.PreloadOutcomeUnavailable, "engine:stopped", 0
	}
	return h.warm.evaluate(ctx, parsed.Host, server.Base, h.warmClient, p.exited, body)
}
