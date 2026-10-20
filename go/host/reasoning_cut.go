package host

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
)

// thinkEnd is the end-of-thinking tag of the thinking families modelhost
// runs (Qwen3, Qwen3.6). After llama.cpp has closed the reasoning section,
// the tag can only reach content when the model wrote it itself.
const thinkEnd = "</think>"

// maxCutHold bounds the content held after a cut while the model's own
// end-of-thinking tag is awaited; beyond it the held text goes out as content.
const maxCutHold = 64 << 10

// cutReasoningMessage is the reasoning_budget_message of a streamed engine
// request that sets a reasoning budget, or "" when its response needs no
// cut filter.
func cutReasoningMessage(raw []byte) string {
	var body struct {
		Stream  bool             `json:"stream"`
		Budget  *json.RawMessage `json:"reasoning_budget_tokens"`
		Message string           `json:"reasoning_budget_message"`
	}
	if json.Unmarshal(raw, &body) != nil || !body.Stream || body.Budget == nil || strings.TrimSpace(body.Message) == "" {
		return ""
	}
	return body.Message
}

// reasoningCutBody keeps a cut thought out of the answer. When the reasoning
// budget runs out, llama.cpp writes the budget message into the reasoning
// and forces the end-of-thinking tag. Qwen3.6 often goes on with the thought
// it was cut in anyway, as content, and then writes its own </think> before
// the answer (R28: "Say hi in five words." over chat@2 returned the counting
// and a literal </think> as text). Once the reasoning has ended with the
// budget message, the body holds the following plain content chunks; when
// the model's own </think> arrives, the text before it is sent as
// reasoning_content and the text after it as content. A response that ends,
// or carries anything other than plain content, before that tag sends the
// held chunks unchanged: the model obeyed the cut.
type reasoningCutBody struct {
	io.ReadCloser
	message string
	in      []byte
	out     bytes.Buffer
	err     error
	tail    string
	cut     bool
	settled bool
	held    [][]byte
	text    strings.Builder
}

func newReasoningCutBody(body io.ReadCloser, message string) *reasoningCutBody {
	return &reasoningCutBody{ReadCloser: body, message: strings.TrimSpace(message)}
}

func (b *reasoningCutBody) Read(p []byte) (int, error) {
	buf := make([]byte, 32<<10)
	for b.out.Len() == 0 && b.err == nil {
		n, err := b.ReadCloser.Read(buf)
		b.in = append(b.in, buf[:n]...)
		for {
			i := bytes.IndexByte(b.in, '\n')
			if i < 0 {
				break
			}
			b.line(b.in[:i])
			b.in = b.in[i+1:]
		}
		if err != nil {
			if len(b.in) > 0 {
				b.release()
				b.out.Write(b.in)
				b.in = nil
			}
			b.release()
			b.err = err
		}
	}
	if b.out.Len() > 0 {
		return b.out.Read(p)
	}
	return 0, b.err
}

// line handles one SSE line without its newline.
func (b *reasoningCutBody) line(raw []byte) {
	line := bytes.TrimSuffix(raw, []byte("\r"))
	if b.settled {
		b.write(raw)
		return
	}
	if len(line) == 0 {
		if len(b.held) == 0 {
			b.write(raw)
		}
		return
	}
	data, ok := bytes.CutPrefix(line, []byte("data:"))
	data = bytes.TrimSpace(data)
	if !ok || bytes.Equal(data, []byte("[DONE]")) {
		b.release()
		b.write(raw)
		return
	}
	var chunk map[string]json.RawMessage
	reasoning, content, plain := "", "", false
	if json.Unmarshal(data, &chunk) == nil {
		reasoning, content, plain = chunkText(chunk)
	}
	switch {
	case reasoning != "" && len(b.held) == 0:
		b.tail = lastBytes(b.tail+reasoning, len(b.message)+256)
		b.cut = strings.HasSuffix(strings.TrimSpace(b.tail), b.message)
		b.write(raw)
	case b.cut && plain && content != "":
		b.held = append(b.held, append([]byte(nil), line...))
		b.text.WriteString(content)
		if before, after, found := strings.Cut(b.text.String(), thinkEnd); found {
			b.held, b.settled = nil, true
			b.emit(chunk, "reasoning_content", before)
			if after = strings.TrimLeft(after, "\r\n"); after != "" {
				b.emit(chunk, "content", after)
			}
		} else if b.text.Len() > maxCutHold {
			b.release()
		}
	default:
		b.release()
		b.write(raw)
	}
}

// release sends the held chunks as they came and stops filtering.
func (b *reasoningCutBody) release() {
	if len(b.held) == 0 {
		return
	}
	for _, line := range b.held {
		b.out.Write(line)
		b.out.WriteString("\n\n")
	}
	b.held, b.settled = nil, true
}

func (b *reasoningCutBody) write(raw []byte) {
	b.out.Write(raw)
	b.out.WriteByte('\n')
}

// emit writes one chunk shaped like chunk whose only delta field is field.
func (b *reasoningCutBody) emit(chunk map[string]json.RawMessage, field, text string) {
	out := map[string]any{}
	for key, value := range chunk {
		out[key] = value
	}
	out["choices"] = []any{map[string]any{"index": 0, "delta": map[string]any{field: text}, "finish_reason": nil}}
	encoded, err := encodeJSON(out)
	if err != nil {
		return
	}
	b.out.WriteString("data: ")
	b.out.Write(encoded)
	b.out.WriteString("\n\n")
}

// chunkText reads one chat completion chunk: its reasoning and content
// text, and whether it is plain content, one choice whose delta carries
// nothing but content and a role, with no finish, tool call, usage, timings
// or error.
func chunkText(chunk map[string]json.RawMessage) (reasoning, content string, plain bool) {
	var choices []map[string]json.RawMessage
	if json.Unmarshal(chunk["choices"], &choices) != nil || len(choices) != 1 {
		return "", "", false
	}
	var delta map[string]json.RawMessage
	if json.Unmarshal(choices[0]["delta"], &delta) != nil {
		return "", "", false
	}
	//unchecked: an absent or non-string field reads as no text
	json.Unmarshal(delta["reasoning_content"], &reasoning)
	//unchecked: an absent or non-string field reads as no text
	json.Unmarshal(delta["content"], &content)
	plain = isNull(choices[0]["finish_reason"])
	for key, value := range delta {
		if key != "content" && key != "role" && !isNull(value) {
			plain = false
		}
	}
	for key, value := range chunk {
		switch key {
		case "id", "object", "created", "model", "system_fingerprint", "choices":
		default:
			if !isNull(value) {
				plain = false
			}
		}
	}
	return reasoning, content, plain
}

func isNull(value json.RawMessage) bool {
	value = bytes.TrimSpace(value)
	return len(value) == 0 || bytes.Equal(value, []byte("null"))
}

func lastBytes(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}
