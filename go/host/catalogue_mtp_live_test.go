package host

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	inference "github.com/openabstractions/abstraction-inference/go"
	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
	router "github.com/openabstractions/abstraction-router/go"
)

// TestLiveCatalogueMTPChat qualifies one exact, already pinned catalogue
// weight through the modelhost provider's chat route. The guarded test owns an
// isolated source declaration, fake lease book, engine child and socket root;
// the external runner enforces the overall job memory and time limits. It does
// not exercise public IPC, Setup placement or a real resource service.
func TestLiveCatalogueMTPChat(t *testing.T) {
	if os.Getenv("OA_LIVE_CATALOGUE_MTP") != "1" {
		t.Skip("set OA_LIVE_CATALOGUE_MTP=1 and MODELHOST_LIVE_* for one cached MTP chat")
	}
	need := func(name string) string {
		value := os.Getenv(name)
		if value == "" {
			t.Fatalf("%s is not set", name)
		}
		return value
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

	const name = "live:qwen36-mtp"
	pin, ok := PinnedEngine(runtime.GOOS, runtime.GOARCH)
	if !ok {
		t.Fatal("no pinned engine for this platform")
	}
	var provider *inference.Provider
	h, err := New(Options{
		Models: []Model{{Name: name, Store: "live", Object: "mtp"}},
		Engine: Engine{Program: program, Pin: pin, Record: EngineRecord(state, pin),
			Arguments: []string{"--ctx-size", "32768", "--ubatch-size", "512", "--parallel", "1",
				"--flash-attn", "on", "--no-mmap", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"},
			Sockets: sockets, MemoryLimit: 48 << 30, Ready: 10 * time.Minute, Stderr: log},
		Source: fakeSource{objects: map[string]Object{"live/mtp": {Store: "live", ID: "mtp", Path: model,
			File: filepath.Base(model), Family: "qwen35moe", Size: info.Size()}}},
		Leases: newFakeLeases(), Idle: time.Hour,
		Active:  func(model string) bool { return provider != nil && provider.ActiveForModel(model) },
		OnError: func(err error) { t.Logf("provider: %v", err) },
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
	}()

	r := router.New()
	engine := router.OpenAICompatible("modelhost", h.Base(name), "/v1", h.Models())
	engine.Profiles = []string{router.ProfileChat}
	r.SetHosts(engine)
	trace := &mtpChatTrace{}
	client := EngineClient(h, func(err error) { t.Logf("engine client: %v", err) })
	client.Transport = mtpTraceTransport{next: client.Transport, trace: trace}
	provider, err = inference.New(inference.Config{
		Router: r, Admit: modelhostAdmit(h),
		Decide: func(context.Context, inference.Subject, string, string) (string, error) { return "permitted", nil },
		Apply: func(context.Context, inference.Subject, string, string, string) (map[string]string, string) {
			return nil, "applied"
		},
		HTTP: client,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 11*time.Minute)
	defer cancel()
	subject := inference.Subject{Account: "live", Program: "catalogue-mtp-test"}
	admission := provider.Start(ctx, subject, wire.Request{
		Model: name,
		Messages: []wire.Message{{Role: wire.RoleUser, Parts: []wire.Part{{Kind: wire.PartKindText,
			Text: "In one short sentence, say that the model is ready."}}}},
		Options: &wire.Options{MaxOutput: 512},
	})
	if admission.Outcome != wire.StartOutcomeAccepted {
		t.Fatalf("chat admission=%s reason=%q", admission.Outcome, admission.Reason)
	}
	var answer strings.Builder
	cursor := int64(0)
	for ctx.Err() == nil {
		page := provider.Observe(ctx, subject, admission.Operation, cursor, 256, 65536, 1000)
		if page.Outcome != wire.PageOutcomePage {
			t.Fatalf("chat observe=%s", page.Outcome)
		}
		for _, delta := range page.Deltas {
			if delta.Sequence != cursor {
				t.Fatalf("delta sequence=%d cursor=%d", delta.Sequence, cursor)
			}
			cursor++
			if delta.Kind == wire.DeltaKindPart && delta.Part.Kind == wire.PartKindText {
				answer.WriteString(delta.Part.Text)
			}
			if delta.Kind == wire.DeltaKindEnd {
				reasoning, content, reasoningChunks, contentChunks, err := trace.streamFields()
				if err != nil {
					t.Fatal(err)
				}
				reasoningTokens := mtpTokenCount(t, ctx, h.warmClient, h.Base(name), reasoning)
				contentTokens := mtpTokenCount(t, ctx, h.warmClient, h.Base(name), content)
				var outputTokens int64
				if delta.End != nil {
					outputTokens = delta.End.Usage.Output
				}
				t.Logf("synthetic stream: reasoning=%d tokenized tokens in %d chunks, content=%d tokenized tokens in %d chunks; engine output=%d tokens",
					reasoningTokens, reasoningChunks, contentTokens, contentChunks, outputTokens)
				if delta.End == nil || delta.End.Outcome != wire.ReplyOutcomeCompleted ||
					strings.TrimSpace(answer.String()) == "" {
					t.Fatalf("chat end=%+v text_bytes=%d", delta.End, answer.Len())
				}
				t.Logf("chat@1 completed: %d text bytes, %d input tokens, %d output tokens",
					answer.Len(), delta.End.Usage.Input, delta.End.Usage.Output)
				return
			}
		}
		if page.Next != cursor {
			t.Fatalf("chat next=%d cursor=%d", page.Next, cursor)
		}
	}
	t.Fatalf("chat did not complete: %v", ctx.Err())
}

// mtpChatTrace observes the same synthetic SSE stream inference reads. It
// retains at most 1 MiB in test memory and writes no generated text to logs.
type mtpChatTrace struct {
	mu        sync.Mutex
	stream    bytes.Buffer
	truncated bool
}

type mtpTraceTransport struct {
	next  http.RoundTripper
	trace *mtpChatTrace
}

func (transport mtpTraceTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := transport.next.RoundTrip(request)
	if err == nil && request.URL.Path == "/v1/chat/completions" {
		response.Body = &mtpTraceBody{ReadCloser: response.Body, trace: transport.trace}
	}
	return response, err
}

type mtpTraceBody struct {
	io.ReadCloser
	trace *mtpChatTrace
}

func (body *mtpTraceBody) Read(p []byte) (int, error) {
	n, err := body.ReadCloser.Read(p)
	body.trace.mu.Lock()
	if body.trace.stream.Len()+n <= 1<<20 {
		_, _ = body.trace.stream.Write(p[:n])
	} else {
		body.trace.truncated = true
	}
	body.trace.mu.Unlock()
	return n, err
}

func (trace *mtpChatTrace) streamFields() (string, string, int, int, error) {
	trace.mu.Lock()
	raw := bytes.Clone(trace.stream.Bytes())
	truncated := trace.truncated
	trace.mu.Unlock()
	if truncated {
		return "", "", 0, 0, io.ErrShortBuffer
	}
	var reasoning, content strings.Builder
	reasoningChunks, contentChunks := 0, 0
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 64<<10), 1<<20)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var event struct {
			Choices []struct {
				Delta struct {
					Reasoning string `json:"reasoning_content"`
					Content   string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return "", "", 0, 0, err
		}
		for _, choice := range event.Choices {
			if choice.Delta.Reasoning != "" {
				reasoning.WriteString(choice.Delta.Reasoning)
				reasoningChunks++
			}
			if choice.Delta.Content != "" {
				content.WriteString(choice.Delta.Content)
				contentChunks++
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", "", 0, 0, err
	}
	return reasoning.String(), content.String(), reasoningChunks, contentChunks, nil
}

func mtpTokenCount(t *testing.T, ctx context.Context, client *http.Client, base, content string) int {
	t.Helper()
	if content == "" {
		return 0
	}
	request, err := json.Marshal(map[string]any{"content": content, "add_special": false, "parse_special": true})
	if err != nil {
		t.Fatal(err)
	}
	var result struct {
		Tokens []int64 `json:"tokens"`
	}
	if err := postEngine(ctx, client, base+tokenizePath, request, &result); err != nil {
		t.Fatal(err)
	}
	return len(result.Tokens)
}

func TestMTPTraceSyntheticFields(t *testing.T) {
	const stream = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"first\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"ready\"}}]}\n\n" +
		"data: [DONE]\n\n"
	trace := &mtpChatTrace{}
	body := &mtpTraceBody{ReadCloser: io.NopCloser(strings.NewReader(stream)), trace: trace}
	got, err := io.ReadAll(body)
	if err != nil || string(got) != stream {
		t.Fatalf("stream pass-through: bytes=%d err=%v", len(got), err)
	}
	reasoning, content, reasoningChunks, contentChunks, err := trace.streamFields()
	if err != nil || reasoning != "think first" || content != "ready" || reasoningChunks != 2 || contentChunks != 1 {
		t.Fatalf("stream fields: reasoning=%q content=%q chunks=%d/%d err=%v",
			reasoning, content, reasoningChunks, contentChunks, err)
	}
}
