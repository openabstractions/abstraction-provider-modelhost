package host

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
	res "github.com/openabstractions/abstraction-resource/go/client"
	router "github.com/openabstractions/abstraction-router/go"
)

// localSession is the chat@1 content OpenCode 1.18.31's Local assistant
// sends to open a conversation, as OA's provider forwards it: the Local
// prompt with OpenCode's environment block and a project's AGENTS.md, and
// the nine core tools with the OA plugin's compact text and schemas, made
// from the pinned OpenCode 1.18.31 tool text
// (adopters/opencode/test/fixtures/opencode-1.18.31-tool). The adopter's
// preload-prefix test derives the tools again from that text and fails when
// this file falls behind.
type localSession struct {
	System []string `json:"system"`
	Tools  []struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  string `json:"parameters"`
	} `json:"tools"`
}

func readLocalSession(t *testing.T) localSession {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "opencode-1.18.31-local.json"))
	if err != nil {
		t.Fatal(err)
	}
	var session localSession
	if err := json.Unmarshal(raw, &session); err != nil {
		t.Fatal(err)
	}
	if len(session.System) == 0 || len(session.Tools) == 0 {
		t.Fatal("the fixture holds no system message or no tools")
	}
	return session
}

// chatBody is the engine request OA's inference provider sends for the
// session's messages: the shape openAIBody gives a chat@1 request (system
// and user text as string content, tools as functions with their parameters
// verbatim, streaming with usage, the answer reserve).
func (s localSession) chatBody(t *testing.T, system []string, turns ...[2]string) []byte {
	t.Helper()
	var messages []map[string]any
	for _, text := range system {
		messages = append(messages, map[string]any{"role": "system", "content": text})
	}
	for _, turn := range turns {
		messages = append(messages, map[string]any{"role": turn[0], "content": turn[1]})
	}
	var tools []map[string]any
	for _, tool := range s.Tools {
		tools = append(tools, map[string]any{"type": "function", "function": map[string]any{
			"name": tool.Name, "description": tool.Description, "parameters": json.RawMessage(tool.Parameters)}})
	}
	raw, err := json.Marshal(map[string]any{"model": "qwen3.6-35b-a3b", "messages": messages, "tools": tools,
		"stream": true, "stream_options": map[string]any{"include_usage": true}, "max_tokens": 4096})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func user(text string) [2]string { return [2]string{"user", text} }

// The live measurement must preserve the nested tool schema bytes when it
// changes answer length. Qwen's template prints those schemas before the
// system prompt; decoding them into map[string]any reorders their keys and
// makes an otherwise identical session start miss the warmed token prefix.
func TestLiveFixtureToolSchemaRoundTripKeepsWarmablePrefix(t *testing.T) {
	session := readLocalSession(t)
	fields, messages, ok := readChatBody(session.chatBody(t, session.System))
	if !ok {
		t.Fatal("synthetic prefix is not a chat body")
	}
	prefix, err := newChatPrefix(fields, messages)
	if err != nil {
		t.Fatal(err)
	}
	engine := newQwenEngine()
	tokens := func(body []byte) []int64 {
		t.Helper()
		prompt, err := engine.render(body)
		if err != nil {
			t.Fatal(err)
		}
		return engine.tokenize(prompt)
	}
	a, err := prefix.request(&warmPlaceholders[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := prefix.request(&warmPlaceholders[1])
	if err != nil {
		t.Fatal(err)
	}
	aTokens, bTokens := tokens(a), tokens(b)
	warm := aTokens[:liveTokenPrefix(aTokens, bTokens)]
	full := session.chatBody(t, session.System, user("Write a fizzbuzz function in fizzbuzz.py and add a short test."))
	var kept map[string]json.RawMessage
	if err := json.Unmarshal(full, &kept); err != nil {
		t.Fatal(err)
	}
	kept["max_tokens"] = json.RawMessage("512")
	keptBody, err := json.Marshal(kept)
	if err != nil {
		t.Fatal(err)
	}
	if common := liveTokenPrefix(warm, tokens(keptBody)); common != len(warm) {
		t.Fatalf("preserved tool schemas share %d/%d warm tokens", common, len(warm))
	}
	var normalized map[string]any
	if err := json.Unmarshal(full, &normalized); err != nil {
		t.Fatal(err)
	}
	normalized["max_tokens"] = 512
	normalizedBody, err := json.Marshal(normalized)
	if err != nil {
		t.Fatal(err)
	}
	if common := liveTokenPrefix(warm, tokens(normalizedBody)); common >= len(warm)/2 {
		t.Fatalf("normalized tool schemas unexpectedly share %d/%d warm tokens", common, len(warm))
	} else {
		t.Logf("synthetic normalized-body prefix %d/%d tokens; preserved body %d/%d", common, len(warm), len(warm), len(warm))
	}
}

// qwenEngine stands where llama-server stands for the warm: it renders chat
// requests with Qwen3.6-35B-A3B's layout (tool definitions first, then the
// system text, in one system turn; the GGUF's own chat template), and
// tokenizes with special tokens kept whole and Qwen2's pre-tokenizer
// splitting the rest, one token per piece. It records every token sequence a
// chat request and a completion brought, and a completion can be held open.
type qwenEngine struct {
	mu          sync.Mutex
	vocab       map[string]int64
	pieces      []string
	chats       [][]int64
	completions [][]int64
	// hold, when set, keeps each completion open until it is closed or the
	// request is cancelled; cancelled then counts the cancelled ones.
	hold      chan struct{}
	entered   chan struct{}
	cancelled int
	// refuseTemplate answers every /apply-template 400, as an engine whose
	// template refuses the prefix.
	refuseTemplate bool
	// templates counts the /apply-template requests.
	templates int
	// holdTokenize, when set, keeps each /tokenize open until it is closed;
	// tokenizing is told on tokenizeEntered.
	holdTokenize    chan struct{}
	tokenizeEntered chan struct{}
	// events is the order requests reached the engine: "completion" at a
	// /completion's arrival, "chat" and "embed" at theirs.
	events []string
	// cancelledAt is when a held completion saw its request cancelled.
	cancelledAt time.Time
}

func newQwenEngine() *qwenEngine { return &qwenEngine{vocab: map[string]int64{}} }

var qwenSpecial = regexp.MustCompile(`<\|im_start\|>|<\|im_end\|>`)

// qwenPiece is Qwen2's pre-tokenizer pattern without its one lookahead
// alternative, \s+(?!\S), which qwenPieces applies by hand.
var qwenPiece = regexp.MustCompile(`^(?:'[sStTmMdD]|'(?:[rR][eE]|[vV][eE]|[lL][lL])|[^\r\n\pL\pN]?\pL+|\pN| ?[^\s\pL\pN]+[\r\n]*|\s*[\r\n]+)`)

func qwenPieces(text string) []string {
	var pieces []string
	for text != "" {
		if m := qwenPiece.FindString(text); m != "" {
			pieces = append(pieces, m)
			text = text[len(m):]
			continue
		}
		// \s+(?!\S), then \s+: a run of spaces before a non-space leaves its
		// last space to the piece that follows.
		end := 0
		for end < len(text) && (text[end] == ' ' || text[end] == '\t') {
			end++
		}
		if end == 0 {
			end = 1
		} else if end > 1 && end < len(text) {
			end--
		}
		pieces = append(pieces, text[:end])
		text = text[end:]
	}
	return pieces
}

func (e *qwenEngine) tokenize(text string) []int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	var tokens []int64
	add := func(piece string) {
		id, ok := e.vocab[piece]
		if !ok {
			id = int64(len(e.pieces))
			e.vocab[piece] = id
			e.pieces = append(e.pieces, piece)
		}
		tokens = append(tokens, id)
	}
	last := 0
	for _, loc := range qwenSpecial.FindAllStringIndex(text, -1) {
		for _, piece := range qwenPieces(text[last:loc[0]]) {
			add(piece)
		}
		add(text[loc[0]:loc[1]])
		last = loc[1]
	}
	for _, piece := range qwenPieces(text[last:]) {
		add(piece)
	}
	return tokens
}

func (e *qwenEngine) text(tokens []int64) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	var b strings.Builder
	for _, id := range tokens {
		b.WriteString(e.pieces[id])
	}
	return b.String()
}

// render is the Qwen3.6 template's layout for system, user and assistant
// turns with tools.
func (e *qwenEngine) render(raw []byte) (string, error) {
	var body struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return "", err
	}
	var b strings.Builder
	rest := body.Messages
	system := ""
	if len(rest) > 0 && rest[0].Role == "system" {
		system = strings.TrimSpace(rest[0].Content)
		rest = rest[1:]
	}
	if len(body.Tools) > 0 {
		b.WriteString("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, tool := range body.Tools {
			var compact bytes.Buffer
			if err := json.Compact(&compact, tool); err != nil {
				return "", err
			}
			b.WriteString("\n")
			b.Write(compact.Bytes())
		}
		b.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n</function>\n</tool_call>")
		if system != "" {
			b.WriteString("\n\n" + system)
		}
		b.WriteString("<|im_end|>\n")
	} else if system != "" {
		b.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	for _, m := range rest {
		b.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n<think>\n")
	return b.String(), nil
}

func (e *qwenEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	raw, _ := io.ReadAll(r.Body)
	switch r.URL.Path {
	case applyTemplatePath:
		e.mu.Lock()
		e.templates++
		refuse := e.refuseTemplate
		e.mu.Unlock()
		prompt, err := e.render(raw)
		if err != nil || refuse {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"prompt": prompt})
	case tokenizePath:
		e.mu.Lock()
		hold, entered := e.holdTokenize, e.tokenizeEntered
		e.mu.Unlock()
		if entered != nil {
			entered <- struct{}{}
		}
		if hold != nil {
			select {
			case <-hold:
			case <-r.Context().Done():
				return
			}
		}
		var body struct {
			Content      string `json:"content"`
			AddSpecial   bool   `json:"add_special"`
			ParseSpecial bool   `json:"parse_special"`
		}
		json.Unmarshal(raw, &body)
		if !body.AddSpecial || !body.ParseSpecial {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"tokens": e.tokenize(body.Content)})
	case completionPath:
		var body struct {
			Prompt      []int64 `json:"prompt"`
			NPredict    int     `json:"n_predict"`
			CachePrompt bool    `json:"cache_prompt"`
		}
		json.Unmarshal(raw, &body)
		e.mu.Lock()
		e.completions = append(e.completions, body.Prompt)
		e.events = append(e.events, "completion")
		hold, entered := e.hold, e.entered
		e.mu.Unlock()
		if entered != nil {
			entered <- struct{}{}
		}
		if hold != nil {
			select {
			case <-hold:
			case <-r.Context().Done():
				e.mu.Lock()
				e.cancelled++
				e.cancelledAt = time.Now()
				e.mu.Unlock()
				return
			}
		}
		if body.NPredict != 1 || !body.CachePrompt {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"content": "", "tokens_evaluated": len(body.Prompt)})
	case chatPath:
		prompt, err := e.render(raw)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		tokens := e.tokenize(prompt)
		e.mu.Lock()
		e.chats = append(e.chats, tokens)
		e.events = append(e.events, "chat")
		e.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	case "/v1/embeddings":
		e.mu.Lock()
		e.events = append(e.events, "embed")
		e.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"index": 0, "embedding": []float64{1}}}})
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

func (e *qwenEngine) lastChat() []int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.chats[len(e.chats)-1]
}

func (e *qwenEngine) chatAt(i int) []int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.chats[i]
}

func (e *qwenEngine) warmed() [][]int64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([][]int64(nil), e.completions...)
}

// warmEngines is one engine host whose requests a warmer hears, as *Host
// hears its own.
type warmEngines struct{ w *warmer }

func (warmEngines) EngineKey(string) (string, bool) { return "key", true }
func (warmEngines) EngineEndpoint(string) (enginehttp.UnixEndpoint, bool) {
	return enginehttp.UnixEndpoint{}, false
}
func (e warmEngines) observeRequest(host, path string, body []byte) <-chan struct{} {
	return e.w.observe(host, path, body)
}

// warmRig is a warmer and the in-process engine it reaches through the
// provider's own engine client.
type warmRig struct {
	engine *qwenEngine
	warm   *warmer
	client *http.Client
	host   string
	exited chan struct{}
}

func newWarmRig(t *testing.T) *warmRig {
	t.Helper()
	engine := newQwenEngine()
	w := newWarmer()
	rig := &warmRig{engine: engine, warm: w, host: "engine0.modelhost.invalid", exited: make(chan struct{})}
	rig.client = engineClient(warmEngines{w}, nil, inProcess{engine})
	return rig
}

// post sends one request through the provider's engine client.
func (r *warmRig) post(t *testing.T, path string, body []byte) {
	t.Helper()
	response, err := r.client.Post(inProcessBase+path, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, response.Body)
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("the request to %s was answered %s", path, response.Status)
	}
}

func (r *warmRig) chat(t *testing.T, body []byte) { t.Helper(); r.post(t, chatPath, body) }

// preloadResult is one evaluate call's answer.
type preloadResult struct {
	outcome wire.PreloadOutcome
	reason  string
	tokens  int64
}

// preload has the engine evaluate body, the engine request of a prefix.
func (r *warmRig) preload(body []byte) preloadResult {
	outcome, reason, tokens := r.warm.evaluate(context.Background(), r.host, inProcessBase, r.client, r.exited, body)
	return preloadResult{outcome, reason, tokens}
}

// preloadAsync runs preload in the background and answers on the channel.
func (r *warmRig) preloadAsync(body []byte) chan preloadResult {
	out := make(chan preloadResult, 1)
	go func() { out <- r.preload(body) }()
	return out
}

func (r *warmRig) holdCompletions() {
	r.engine.mu.Lock()
	r.engine.hold = make(chan struct{})
	r.engine.entered = make(chan struct{}, 4)
	r.engine.mu.Unlock()
}

func (r *warmRig) cancelled() int {
	r.engine.mu.Lock()
	defer r.engine.mu.Unlock()
	return r.engine.cancelled
}

// state is what the warmer holds for the rig's engine.
func (r *warmRig) state() (used bool, running bool) {
	r.warm.mu.Lock()
	defer r.warm.mu.Unlock()
	return r.warm.used[r.host], r.warm.runs[r.host] != nil
}

func wait(t *testing.T, results chan preloadResult) preloadResult {
	t.Helper()
	select {
	case result := <-results:
		return result
	case <-time.After(10 * time.Second):
		t.Fatal("the preload did not end")
		return preloadResult{}
	}
}

func commonTokens(a, b []int64) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// MODELHOST-W2: the Local assistant's prefix, sent as preload@1's engine
// request, makes the engine evaluate, token for token, the tokens the
// session start that follows it begins with, through to the user turn's
// header; the person's first words are the first token it leaves out.
func TestPreloadedPrefixIsTheSessionStartTokenForToken(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	result := rig.preload(session.chatBody(t, session.System))
	if result.outcome != wire.PreloadOutcomeEvaluated || result.tokens < minWarmTokens {
		t.Fatalf("the preload answered %+v", result)
	}
	warmed := rig.engine.warmed()
	if len(warmed) != 1 || int64(len(warmed[0])) != result.tokens {
		t.Fatalf("the engine evaluated %d prompts; the reply counts %d tokens", len(warmed), result.tokens)
	}
	prefix := warmed[0]
	if !strings.HasSuffix(rig.engine.text(prefix), "<|im_end|>\n<|im_start|>user\n") {
		t.Fatalf("the evaluated prefix ends %q, not at the user turn's header", tail(rig.engine.text(prefix), 40))
	}
	for _, first := range []string{"hi", "Write a fizzbuzz function in fizzbuzz.py and add a short test.",
		`"Fix" the failing test`, "1. read the README", "  indented request", "ünïcödé request", "/init"} {
		rig.chat(t, session.chatBody(t, session.System, user(first)))
		next := rig.engine.lastChat()
		if n := commonTokens(prefix, next); n != len(prefix) {
			t.Fatalf("first words %q: the session start shares %d of the %d evaluated tokens; it parts at %q",
				first, n, len(prefix), rig.engine.text(next[n:min(n+3, len(next))]))
		}
		if len(next)-len(prefix) > 40 {
			t.Fatalf("first words %q: %d of %d tokens are left to evaluate", first, len(next)-len(prefix), len(next))
		}
	}
	t.Logf("evaluated %d tokens; the session start %q has %d", len(prefix), "hi", len(rig.engine.chatAt(0)))
}

// MODELHOST-W1: the provider holds no prefix. Application A's session start
// leaves nothing B's preload can evaluate: on A's used engine B's preload
// evaluates nothing at all, and on a fresh engine it evaluates B's own text
// alone. After every call the warmer holds no request bytes.
func TestAPreloadEvaluatesOnlyTheCallersOwnText(t *testing.T) {
	session := readLocalSession(t)
	other := strings.ReplaceAll(session.System[0], `C:\Users\person\projects\wordstat`, `C:\Users\person\projects\private`)
	rig := newWarmRig(t)
	rig.chat(t, session.chatBody(t, []string{other}, user("hi")))
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeBusy || len(rig.engine.warmed()) != 0 {
		t.Fatalf("B's preload on A's used engine answered %+v and sent %d prompts", result, len(rig.engine.warmed()))
	}
	rig.warm.engineChanged(rig.host)
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("B's preload on a fresh engine answered %+v", result)
	}
	evaluated := rig.engine.text(rig.engine.warmed()[0])
	if strings.Contains(evaluated, `projects\private`) || !strings.Contains(evaluated, `projects\wordstat`) {
		t.Fatal("the evaluated prefix is not B's own text")
	}
	rig.warm.mu.Lock()
	defer rig.warm.mu.Unlock()
	if len(rig.warm.runs) != 0 {
		t.Fatalf("the warmer holds %d preloads after the calls ended", len(rig.warm.runs))
	}
}

// MODELHOST-W2: a turn on the engine makes a later preload a no-op that
// answers busy, so no conversation is displaced; a second preload while one
// runs is busy too. A new engine is free again.
func TestAPreloadNeverDisplacesAConversation(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	rig.chat(t, session.chatBody(t, session.System, user("hi"), [2]string{"assistant", "Hello."}, user("more")))
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeBusy || result.reason != "engine:used" {
		t.Fatalf("a preload after a turn answered %+v", result)
	}
	rig.engine.mu.Lock()
	templates := rig.engine.templates
	rig.engine.mu.Unlock()
	if templates != 0 || len(rig.engine.warmed()) != 0 {
		t.Fatalf("a busy preload reached the engine %d times", templates)
	}
	rig.warm.engineChanged(rig.host)
	rig.holdCompletions()
	first := rig.preloadAsync(session.chatBody(t, session.System))
	<-rig.engine.entered
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeBusy || result.reason != "preload:running" {
		t.Fatalf("a second preload answered %+v", result)
	}
	close(rig.engine.hold)
	if result := wait(t, first); result.outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("the first preload answered %+v", result)
	}
}

// MODELHOST-W3: every request but the one a preload prepares cancels its
// posted evaluation before the request leaves the provider, and the request
// then waits only until the evaluation's engine request has returned: a
// changed session start, a conversation's later turn, and an embedding
// request on the same engine. The request the preload prepares is
// TestTheRequestAPreloadPreparesFollowsItsPostedEvaluation's.
func TestEveryOtherRequestCancelsAPreload(t *testing.T) {
	session := readLocalSession(t)
	moved := strings.ReplaceAll(session.System[0], `C:\Users\person\projects\wordstat`, `C:\Users\person\projects\other`)
	for _, c := range []struct {
		name string
		path string
		body func() []byte
	}{
		{"another project's session start", chatPath, func() []byte { return session.chatBody(t, []string{moved}, user("hi")) }},
		{"a conversation's later turn", chatPath, func() []byte {
			return session.chatBody(t, session.System, user("hi"), [2]string{"assistant", "Hello."}, user("Write fizzbuzz"))
		}},
		{"an embedding request", "/v1/embeddings", func() []byte { return []byte(`{"model":"m","input":["x"]}`) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			rig := newWarmRig(t)
			rig.holdCompletions()
			results := rig.preloadAsync(session.chatBody(t, session.System))
			<-rig.engine.entered
			rig.post(t, c.path, c.body())
			returned := time.Now()
			rig.engine.mu.Lock()
			cancelledAt := rig.engine.cancelledAt
			rig.engine.mu.Unlock()
			// Review R1: the request goes as soon as the evaluation's engine
			// request has returned, with no wait for an engine exit.
			if cancelledAt.IsZero() || returned.Sub(cancelledAt) > 200*time.Millisecond {
				t.Fatalf("the request returned %s after the stand-in saw the cancel", returned.Sub(cancelledAt))
			}
			result := wait(t, results)
			if result.outcome != wire.PreloadOutcomeBusy || result.reason != "engine:used" {
				t.Fatalf("the preload answered %+v", result)
			}
			if rig.cancelled() != 1 {
				t.Fatalf("the engine saw %d cancelled evaluations, want 1", rig.cancelled())
			}
			if order := rig.engine.order(); len(order) != 2 || order[0] != "completion" {
				t.Fatalf("the engine saw %v, want the evaluation before the request", order)
			}
			if used, running := rig.state(); !used || running {
				t.Fatalf("after the request the engine reads used = %v, a preload running = %v", used, running)
			}
		})
	}
}

// MODELHOST-W2: an engine stopped by an unload or a yield while it
// evaluates ends the preload unavailable with reason engine:stopped. No
// report is made: the caller has the outcome.
func TestAnEngineStoppedDuringAPreloadEndsIt(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	rig.holdCompletions()
	results := rig.preloadAsync(session.chatBody(t, session.System))
	<-rig.engine.entered
	close(rig.exited)
	result := wait(t, results)
	if result.outcome != wire.PreloadOutcomeUnavailable || result.reason != "engine:stopped" {
		t.Fatalf("the preload answered %+v", result)
	}
	if _, running := rig.state(); running {
		t.Fatal("a preload is still registered after its engine stopped")
	}
}

// MODELHOST-W1: a prefix the engine refuses answers refused with the
// engine's status and leaves nothing behind: the engine is not marked used
// and the next prefix is evaluated as if nothing came before. A prefix too
// short to be worth its checkpoints answers loaded and sends nothing to
// evaluate.
func TestARefusedPrefixLeavesNothingBehind(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	rig.engine.refuseTemplate = true
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeRefused || result.reason != "engine:400" {
		t.Fatalf("a refused prefix answered %+v", result)
	}
	if used, running := rig.state(); used || running {
		t.Fatalf("a refused prefix left used = %v, running = %v", used, running)
	}
	rig.engine.refuseTemplate = false
	if result := rig.preload(session.chatBody(t, session.System)); result.outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("the next prefix answered %+v", result)
	}
	short := rig.preload([]byte(`{"model":"m","messages":[{"role":"system","content":"Be brief."}]}`))
	if short.outcome != wire.PreloadOutcomeLoaded || short.reason != "prefix:short" || len(rig.engine.warmed()) != 1 {
		t.Fatalf("a short prefix answered %+v after %d evaluations", short, len(rig.engine.warmed()))
	}
}

// MODELHOST-W2: the provider's evaluator serves its own engines alone: a
// server it did not start is unsupported, and a stopped engine unavailable.
func TestEvaluatePrefixServesOnlyThisHostsEngines(t *testing.T) {
	h := composed(t, newFakeLeases(), nil)
	other := router.OpenAICompatible("elsewhere", "http://127.0.0.1:1", "/v1", []string{"tiny:1b"})
	if outcome, _, _ := h.EvaluatePrefix(context.Background(), other, "tiny:1b", []byte(`{}`)); outcome != wire.PreloadOutcomeUnsupportedFeature {
		t.Fatalf("another server answered %v", outcome)
	}
	own := router.OpenAICompatible("modelhost-engine", h.Base("tiny:1b"), "/v1", h.Models())
	if outcome, reason, _ := h.EvaluatePrefix(context.Background(), own, "tiny:1b", []byte(`{}`)); outcome != wire.PreloadOutcomeUnavailable || reason != "engine:stopped" {
		t.Fatalf("a stopped engine answered %v %s", outcome, reason)
	}
}

func (e *qwenEngine) order() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.events...)
}

// MODELHOST-W3 (review N1): the session start a preload prepares, arriving
// before the evaluation is posted, overtakes it: the preload posts nothing
// and answers busy, so nothing of it can run after the request.
func TestTheRequestAPreloadPreparesOvertakesItBeforeItIsPosted(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	rig.engine.mu.Lock()
	rig.engine.holdTokenize = make(chan struct{})
	rig.engine.tokenizeEntered = make(chan struct{}, 4)
	rig.engine.mu.Unlock()
	results := rig.preloadAsync(session.chatBody(t, session.System))
	<-rig.engine.tokenizeEntered
	rig.chat(t, session.chatBody(t, session.System, user("hi")))
	close(rig.engine.holdTokenize)
	result := wait(t, results)
	if result.outcome != wire.PreloadOutcomeBusy || result.reason != "engine:used" {
		t.Fatalf("the overtaken preload answered %+v", result)
	}
	if order := rig.engine.order(); len(order) != 1 || order[0] != "chat" {
		t.Fatalf("the engine saw %v, want the chat alone", order)
	}
}

// MODELHOST-W3: the session start a preload prepares, arriving once the
// evaluation is posted, waits in the provider for the evaluation to return
// and then reaches the engine, so the engine evaluates the prefix first and
// nothing of the preload runs after the request.
func TestTheRequestAPreloadPreparesFollowsItsPostedEvaluation(t *testing.T) {
	session := readLocalSession(t)
	rig := newWarmRig(t)
	rig.holdCompletions()
	results := rig.preloadAsync(session.chatBody(t, session.System))
	<-rig.engine.entered
	sent := make(chan struct{})
	go func() { rig.chat(t, session.chatBody(t, session.System, user("hi"))); close(sent) }()
	select {
	case <-sent:
		t.Fatal("the session start reached the engine while the evaluation ran")
	case <-time.After(100 * time.Millisecond):
	}
	close(rig.engine.hold)
	<-sent
	if result := wait(t, results); result.outcome != wire.PreloadOutcomeEvaluated {
		t.Fatalf("the preload answered %+v", result)
	}
	if order := rig.engine.order(); len(order) != 2 || order[0] != "completion" || order[1] != "chat" {
		t.Fatalf("the engine saw %v, want the evaluation and then the chat", order)
	}
}

// MODELHOST-W2, W3 (review F4, N7): a preload and another request racing on
// a fresh engine never leave an evaluation reaching the engine after the
// request, whichever wins the lock, and the preload answers what happened.
func TestAPreloadAndARequestInterleaveSafely(t *testing.T) {
	session := readLocalSession(t)
	prefix := session.chatBody(t, session.System)
	for i := 0; i < 200; i++ {
		rig := newWarmRig(t)
		results := rig.preloadAsync(prefix)
		if i%2 == 0 {
			rig.post(t, "/v1/embeddings", []byte(`{"model":"m","input":["x"]}`))
		} else {
			rig.chat(t, session.chatBody(t, session.System, user("hi")))
		}
		result := wait(t, results)
		order := rig.engine.order()
		for j, event := range order {
			if event == "completion" && j > 0 {
				t.Fatalf("round %d: the engine saw %v", i, order)
			}
		}
		evaluated := len(order) > 0 && order[0] == "completion"
		switch {
		case result.outcome == wire.PreloadOutcomeEvaluated && evaluated:
		case result.outcome == wire.PreloadOutcomeBusy && result.reason == "engine:used":
		default:
			t.Fatalf("round %d: the preload answered %+v after %v", i, result, order)
		}
	}
}

// freeBook is the fake lease book with the no-yield acquire (resource
// RES-L6): held answers held; otherwise it grants a "free-N" lease. frees
// counts its calls apart from Acquire's.
type freeBook struct {
	*fakeLeases
	held  bool
	frees int
}

func (b *freeBook) AcquireFree(resource string, amount int64) (res.AcquireFreeResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.frees++
	if b.held {
		return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeHeld, Asked: []res.YieldRecord{}}, nil
	}
	b.seq++
	id := "free-" + strconv.Itoa(b.seq)
	b.live[id] = true
	if b.amounts == nil {
		b.amounts = map[string]int64{}
	}
	b.amounts[id] = amount
	return res.AcquireFreeResult{Outcome: res.AcquireFreeOutcomeAcquired, Lease: &res.Lease{ID: id, Resource: resource, Amount: amount},
		Asked: []res.YieldRecord{}}, nil
}

func (b *freeBook) setHeld(held bool) { b.mu.Lock(); b.held = held; b.mu.Unlock() }

func (b *freeBook) freeCalls() int { b.mu.Lock(); defer b.mu.Unlock(); return b.frees }

func freeHost(t *testing.T, book *freeBook, adjust func(*Options)) *Host {
	t.Helper()
	return composed(t, book.fakeLeases, func(o *Options) {
		o.Leases = book
		if adjust != nil {
			adjust(o)
		}
	})
}

// MODELHOST-W2 (resource RES-L6): a preload's load goes through the
// holder's speculative ensure. A held card answers resources:held with no
// Acquire, so nobody is asked to yield; a model evicted between a residency
// check and the ensure is the same missing model; a free card loads through
// the no-yield acquire; and a book without it is held.
func TestAPreloadNeverAsksAnotherHolderToYield(t *testing.T) {
	requireEnginePeer(t)
	book := &freeBook{fakeLeases: newFakeLeases(), held: true}
	h := freeHost(t, book, nil)
	if _, err := h.EnsureSpeculative(context.Background(), "tiny:1b"); WordOf(err) != resourcesHeld || h.Loaded("tiny:1b") {
		t.Fatalf("a held card: %v, loaded %v", err, h.Loaded("tiny:1b"))
	}
	// Loaded, then evicted before the speculative ensure.
	book.setHeld(false)
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	h.Unload("tiny:1b", "evicted")
	book.setHeld(true)
	if _, err := h.EnsureSpeculative(context.Background(), "tiny:1b"); WordOf(err) != resourcesHeld || h.Loaded("tiny:1b") {
		t.Fatalf("an evicted model: %v, loaded %v", err, h.Loaded("tiny:1b"))
	}
	if acquires, _, _ := book.counts(); acquires != 1 || book.freeCalls() != 2 {
		t.Fatalf("Acquire %d times, the no-yield acquire %d times; want the one plain Ensure and two speculative ones", acquires, book.freeCalls())
	}
	book.setHeld(false)
	if _, err := h.EnsureSpeculative(context.Background(), "tiny:1b"); err != nil || !h.Loaded("tiny:1b") || !strings.HasPrefix(h.Lease("tiny:1b"), "free-") {
		t.Fatalf("a free card: %v, loaded %v, lease %q", err, h.Loaded("tiny:1b"), h.Lease("tiny:1b"))
	}
	plain := composed(t, newFakeLeases(), nil)
	if _, err := plain.EnsureSpeculative(context.Background(), "tiny:1b"); WordOf(err) != resourcesHeld {
		t.Fatalf("a book with no no-yield acquire: %v", err)
	}
}

// MODELHOST-W2 (review R2): a preload of a resident model leaves its idle
// clock alone, so a model used at t0 and preloaded at t0+Idle-1s is still
// unloaded at t0+Idle.
func TestAPreloadLeavesTheIdleClockAlone(t *testing.T) {
	requireEnginePeer(t)
	clock := time.Unix(2_000_000, 0)
	var mu sync.Mutex
	now := func() time.Time { mu.Lock(); defer mu.Unlock(); return clock }
	advance := func(d time.Duration) { mu.Lock(); clock = clock.Add(d); mu.Unlock() }
	book := &freeBook{fakeLeases: newFakeLeases()}
	h := freeHost(t, book, func(o *Options) { o.Now, o.Idle = now, time.Minute })
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	advance(time.Minute - time.Second)
	if _, err := h.EnsureSpeculative(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	advance(time.Second)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	deadline := time.Now().Add(10 * time.Second)
	for h.Loaded("tiny:1b") {
		if time.Now().After(deadline) {
			t.Fatal("the preload kept the model past its idle time")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// MODELHOST-W2 (review R4): a yield request during a preload's load is
// answered yielded at once and cancels the load, which answers
// resources:held.
func TestAYieldDuringAPreloadsLoadCancelsIt(t *testing.T) {
	book := &freeBook{fakeLeases: newFakeLeases()}
	h := freeHost(t, book, func(o *Options) { o.Engine.fixture = []string{"--fake-load", "30s"} })
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	loaded := make(chan error, 1)
	go func() { _, err := h.EnsureSpeculative(context.Background(), "tiny:1b"); loaded <- err }()
	lease := ""
	deadline := time.Now().Add(10 * time.Second)
	for lease == "" {
		book.mu.Lock()
		for id := range book.live {
			lease = id
		}
		book.mu.Unlock()
		if time.Now().After(deadline) {
			t.Fatal("the speculative load never acquired")
		}
		time.Sleep(10 * time.Millisecond)
	}
	began := time.Now()
	book.pending <- res.YieldRequest{ID: "ask-1", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: lease}
	var err error
	select {
	case err = <-loaded:
	case <-time.After(10 * time.Second):
		t.Fatal("the yield did not cancel the load")
	}
	if WordOf(err) != resourcesHeld {
		t.Fatalf("the cancelled load answered %v", err)
	}
	for {
		if _, _, answers := book.counts(); len(answers) == 1 {
			if answers[0] != "ask-1=yielded" {
				t.Fatalf("the yield was answered %v", answers)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the yield was never answered")
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Logf("the yield was answered %s after it was sent", time.Since(began).Round(time.Millisecond))
}

// heldPreload loads tiny:1b on a stand-in engine whose evaluation is held
// open, starts a preload, and returns its answer channel once the engine
// has the evaluation.
func heldPreload(t *testing.T, leases *fakeLeases, reported *[]error) (*Host, chan preloadResult) {
	t.Helper()
	session := readLocalSession(t)
	dir := t.TempDir()
	warmFile, gate := filepath.Join(dir, "warm.txt"), filepath.Join(dir, "never")
	var mu sync.Mutex
	h := composed(t, leases, func(o *Options) {
		o.Engine.fixture = []string{"--fake-warm-file", warmFile, "--fake-completion-gate", gate}
		o.OnError = func(err error) { mu.Lock(); *reported = append(*reported, err); mu.Unlock() }
	})
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	server := router.OpenAICompatible("modelhost-engine", h.Base("tiny:1b"), "/v1", h.Models())
	results := make(chan preloadResult, 1)
	go func() {
		outcome, reason, tokens := h.EvaluatePrefix(context.Background(), server, "tiny:1b", session.chatBody(t, session.System))
		results <- preloadResult{outcome, reason, tokens}
	}()
	deadline := time.Now().Add(10 * time.Second)
	for {
		if raw, err := os.ReadFile(warmFile); err == nil && len(raw) > 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the engine never received the evaluation")
		}
		time.Sleep(10 * time.Millisecond)
	}
	return h, results
}

// MODELHOST-W2 (review N7): an unload while the engine evaluates ends the
// preload unavailable, engine:stopped, and nothing is reported.
func TestAnUnloadDuringAPreloadEndsItSilently(t *testing.T) {
	requireEnginePeer(t)
	var reported []error
	h, results := heldPreload(t, newFakeLeases(), &reported)
	h.Unload("tiny:1b", "test")
	result := wait(t, results)
	if result.outcome != wire.PreloadOutcomeUnavailable || result.reason != "engine:stopped" {
		t.Fatalf("the preload answered %+v", result)
	}
	if len(reported) != 0 {
		t.Fatalf("the provider reported %v", reported)
	}
}

// MODELHOST-W2 (decision N2a): a preload's evaluation is no model use, so a
// yield request for the lease is answered yielded at once, the engine stops
// and the preload ends unavailable, engine:stopped, with nothing reported.
func TestAYieldDuringAPreloadIsAnsweredAtOnce(t *testing.T) {
	requireEnginePeer(t)
	var reported []error
	leases := newFakeLeases()
	h, results := heldPreload(t, leases, &reported)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go h.Run(ctx)
	leases.pending <- res.YieldRequest{ID: "ask-1", Resource: "card:0", Amount: 4096, Reason: "another program", Lease: h.Lease("tiny:1b")}
	result := wait(t, results)
	if result.outcome != wire.PreloadOutcomeUnavailable || result.reason != "engine:stopped" {
		t.Fatalf("the preload answered %+v", result)
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, _, answers := leases.counts(); len(answers) == 1 {
			if answers[0] != "ask-1=yielded" {
				t.Fatalf("the yield was answered %v", answers)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the yield was never answered")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(reported) != 0 {
		t.Fatalf("the provider reported %v", reported)
	}
}
