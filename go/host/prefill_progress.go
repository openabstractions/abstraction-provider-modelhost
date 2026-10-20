package host

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"sync"
)

// Pinned b10825 server-schema.cpp:31 and server-context.cpp:1883–1888,
// 3095–3103, 3439–3444: return_progress reports request-local prompt counters
// after decode. processed includes the cached prefix; only processed-cache
// represents newly evaluated tokens. No elapsed time or label is authority.
func requestPrefillProgress(raw []byte) ([]byte, bool, error) {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, false, err
	}
	if !bytes.Equal(bytes.TrimSpace(body["stream"]), []byte("true")) {
		return raw, false, nil
	}
	body["return_progress"] = json.RawMessage("true")
	result, err := json.Marshal(body)
	return result, err == nil, err
}

type prefillProgressBody struct {
	io.ReadCloser
	mu                       sync.Mutex
	line                     []byte
	overflow, closed, output bool
	id                       string
	processed, total, cache  int64
	bound                    bool
	report                   func(int64, int64) bool
}

func (b *prefillProgressBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	b.observe(p[:n])
	return n, err
}

func (b *prefillProgressBody) Close() error {
	b.mu.Lock()
	b.closed = true
	b.line = nil
	b.mu.Unlock()
	return b.ReadCloser.Close()
}

func (b *prefillProgressBody) observe(raw []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.output {
		return
	}
	for _, c := range raw {
		if c == '\n' {
			if !b.overflow {
				b.event(bytes.TrimSpace(b.line))
			}
			b.line = b.line[:0]
			b.overflow = false
		} else if !b.overflow && len(b.line) < 64<<10 {
			b.line = append(b.line, c)
		} else {
			b.overflow = true
		}
	}
}

// Reject duplicate keys and trailing JSON rather than accepting ambiguous
// counter authority. Unknown ordinary event fields remain byte-for-byte intact.
func prefillObject(raw []byte) (map[string]json.RawMessage, bool) {
	d := json.NewDecoder(bytes.NewReader(raw))
	first, err := d.Token()
	if err != nil || first != json.Delim('{') {
		return nil, false
	}
	fields := map[string]json.RawMessage{}
	for d.More() {
		key, err := d.Token()
		name, ok := key.(string)
		if err != nil || !ok {
			return nil, false
		}
		if _, duplicate := fields[name]; duplicate {
			return nil, false
		}
		var value json.RawMessage
		if d.Decode(&value) != nil {
			return nil, false
		}
		fields[name] = value
	}
	last, err := d.Token()
	if err != nil || last != json.Delim('}') {
		return nil, false
	}
	var extra any
	if d.Decode(&extra) != io.EOF {
		return nil, false
	}
	return fields, true
}

func prefillInteger(raw []byte) (int64, bool) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	token, err := d.Token()
	number, ok := token.(json.Number)
	if err != nil || !ok {
		return 0, false
	}
	value, err := number.Int64()
	if err != nil {
		return 0, false
	}
	var extra any
	return value, d.Decode(&extra) == io.EOF
}

func (b *prefillProgressBody) event(line []byte) {
	data, ok := bytes.CutPrefix(line, []byte("data: "))
	if !ok || bytes.Equal(data, []byte("[DONE]")) {
		return
	}
	event, ok := prefillObject(data)
	if !ok {
		return
	}
	var id string
	if json.Unmarshal(event["id"], &id) != nil || id == "" || len(id) > 256 {
		return
	}
	if b.id != "" && id != b.id {
		return
	}
	var choices []json.RawMessage
	if json.Unmarshal(event["choices"], &choices) != nil || len(choices) != 1 {
		return
	}
	choice, ok := prefillObject(choices[0])
	if !ok {
		return
	}
	index, ok := prefillInteger(choice["index"])
	if !ok || index != 0 {
		return
	}
	if finish, exists := choice["finish_reason"]; exists && !bytes.Equal(bytes.TrimSpace(finish), []byte("null")) {
		b.output = true
		return
	}
	delta, ok := prefillObject(choice["delta"])
	if !ok {
		return
	}
	// The pinned progress event is an assistant role with null content. Token,
	// tool and reasoning deltas end this parser's prefill authority instead.
	for key, value := range delta {
		if key == "role" {
			continue
		}
		if key != "content" || !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			b.output = true
			return
		}
	}
	var role string
	if json.Unmarshal(delta["role"], &role) != nil || role != "assistant" {
		return
	}
	progress, ok := prefillObject(event["prompt_progress"])
	if !ok {
		return
	}
	total, ok := prefillInteger(progress["total"])
	if !ok || total <= 0 {
		return
	}
	processed, ok := prefillInteger(progress["processed"])
	if !ok || processed < 0 || processed > total {
		return
	}
	cache, ok := prefillInteger(progress["cache"])
	if !ok || cache < 0 || cache > processed {
		return
	}
	var elapsed float64
	if json.Unmarshal(progress["time_ms"], &elapsed) != nil || math.IsNaN(elapsed) || math.IsInf(elapsed, 0) || elapsed < 0 {
		return
	}
	if b.bound && (total != b.total || cache != b.cache || processed <= b.processed) {
		return
	}
	if !b.bound {
		b.bound = true
		b.id = id
		b.total = total
		b.cache = cache
		b.processed = cache
	}
	if processed <= b.processed {
		return
	}
	b.processed = processed
	if b.report != nil {
		b.report(processed-cache, total-cache)
	}
}
