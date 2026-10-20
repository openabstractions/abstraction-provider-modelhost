package host

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/openabstractions/abstraction-inference/go/providerhost/enginehttp"
	"github.com/openabstractions/abstraction-inference/go/providerhost/llamacpp"
)

// ErrEngineMedia reports an engine request that carries media the engine
// must not fetch, read or decode (CONTRACT.md MODELHOST-L8; b10825 review
// condition 6). Its message names one of the words below.
var ErrEngineMedia = errors.New("modelhost: media the engine must not fetch or decode")

const (
	// MediaRemote is an http(s) URL, which llama-server fetches itself.
	MediaRemote = "media:remote"
	// MediaFile is a file:// URL, which llama-server reads under --media-path.
	MediaFile = "media:file"
	// MediaType is any other media: a non-data URL, an image other than a
	// base64 PNG or JPEG of at most 16 MP whose bytes are the type it
	// declares (WebP and video spawn FFmpeg), or a content part other than
	// text and image_url.
	MediaType = "media:type"
	// MediaRequest is a body this guard cannot read as one JSON object.
	MediaRequest = "media:request"
)

// engineMedia is the provider kit's chat media policy (providerhost/llamacpp,
// inference CONTRACT.md INF-LC3) without audio: chat@1 carries no audio, and
// no engine request of this provider needs it.
var engineMedia = llamacpp.Media{}

// mediaWords are this provider's words for the kit's refusal reasons.
var mediaWords = map[string]string{llamacpp.RefusedRemote: MediaRemote, llamacpp.RefusedFile: MediaFile, llamacpp.RefusedType: MediaType}

// maxEngineRequest bounds one request body the guard reads: inference's 64
// MiB vision bound, base64-encoded, with room for the text around it.
const maxEngineRequest = 128 << 20

// Engines are the running engines an EngineClient reaches, by the host of
// each engine's base URL. *Host is one.
type Engines interface {
	// EngineKey is the engine's per-launch key (MODELHOST-L8).
	EngineKey(host string) (string, bool)
	// EngineEndpoint is the engine's socket and the check its server
	// process must pass (MODELHOST-L7).
	EngineEndpoint(host string) (enginehttp.UnixEndpoint, bool)
}

// requestObserver hears each POST an EngineClient sends, with its path and
// body, after the media guard passed it and before it leaves this process,
// and answers what the request must wait for first, or nil. *Host is one.
type requestObserver interface {
	observeRequest(host, path string, body []byte) <-chan struct{}
}

// EngineClient is the HTTP client the inference provider reaches this
// provider's engines with. Each request goes to the socket engines names
// for its host, and the socket's server process must be that engine before
// the request's first byte leaves this process (MODELHOST-L7). Its
// transport refuses any request body that carries media the engine would
// fetch, read or decode outside this process's reach, answering 403 before
// the body leaves this process; refused reports each one. Every request it
// sends carries the engine's per-launch LLAMA_API_KEY as "Authorization:
// Bearer KEY". It follows no redirect and uses no proxy. A nil engines
// reaches no engine.
func EngineClient(engines Engines, refused func(error)) *http.Client {
	resolve := func(host string) (enginehttp.UnixEndpoint, bool) {
		if engines == nil {
			return enginehttp.UnixEndpoint{}, false
		}
		return engines.EngineEndpoint(host)
	}
	return engineClient(engines, refused, enginehttp.NewUnixTransport(resolve))
}

// engineClient is EngineClient over next; tests pass an in-process engine.
func engineClient(engines Engines, refused func(error), next http.RoundTripper) *http.Client {
	return &http.Client{
		Transport:     mediaGuard{next: next, engines: engines, refused: refused},
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

type mediaGuard struct {
	next    http.RoundTripper
	engines Engines
	refused func(error)
}

func (g mediaGuard) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body == nil || req.Body == http.NoBody {
		return g.next.RoundTrip(g.keyed(req.Clone(req.Context()), req))
	}
	raw, err := io.ReadAll(io.LimitReader(req.Body, maxEngineRequest+1))
	//unchecked: the body was read to its end or its bound; the read error decides below
	req.Body.Close()
	if err != nil {
		return nil, err
	}
	if len(raw) > maxEngineRequest {
		return g.refuse(req, fmt.Errorf("%w: %s: the request exceeds %d bytes", ErrEngineMedia, MediaRequest, maxEngineRequest))
	}
	raw, err = checkEngineMedia(raw)
	if err != nil {
		return g.refuse(req, err)
	}
	// Every request that runs work leaves this process here: it marks the
	// engine used and cancels a preload in flight it does not continue
	// (MODELHOST-W3). The preload's own requests are not counted.
	if observer, ok := g.engines.(requestObserver); ok && req.Method == http.MethodPost && req.URL != nil && req.Context().Value(warmRequest{}) == nil {
		// The request leaves only once nothing of a preload can run on the
		// engine after it (MODELHOST-W3).
		if wait := observer.observeRequest(req.URL.Host, req.URL.Path, raw); wait != nil {
			select {
			case <-wait:
			case <-req.Context().Done():
				return nil, req.Context().Err()
			}
		}
	}
	// A RoundTripper must not change the caller's request; the clone carries
	// the checked bytes.
	out := req.Clone(req.Context())
	out.Body = io.NopCloser(bytes.NewReader(raw))
	out.ContentLength = int64(len(raw))
	out.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(raw)), nil }
	return g.next.RoundTrip(g.keyed(out, req))
}

// keyed sets the per-launch key of the engine at req's host on out, the
// request's clone, in place of any Authorization a caller set. A host with
// no running engine gets no Authorization, and the transport refuses it.
func (g mediaGuard) keyed(out, req *http.Request) *http.Request {
	out.Header.Del("Authorization")
	if g.engines != nil && req.URL != nil {
		if key, ok := g.engines.EngineKey(req.URL.Host); ok {
			out.Header.Set("Authorization", "Bearer "+key)
		}
	}
	return out
}

// refuse answers 403 in place of the engine. inference reads it as the
// reply outcome refused with reason upstream:403.
func (g mediaGuard) refuse(req *http.Request, err error) (*http.Response, error) {
	if g.refused != nil {
		g.refused(err)
	}
	body, _ := json.Marshal(map[string]any{"error": map[string]any{
		"code": http.StatusForbidden, "type": "oa_media_refused", "message": err.Error()}})
	return &http.Response{
		Status: "403 Forbidden", StatusCode: http.StatusForbidden,
		Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
		Header:        http.Header{"Content-Type": []string{"application/json"}},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)), Request: req,
	}, nil
}

// checkEngineMedia reads one engine request body and returns the bytes to
// send. Chat messages may carry string content, text parts and image_url
// parts; the provider kit's media policy decodes each image and replaces it
// with the provider's own PNG, and the body is then sent as the guard encoded
// it again, the image parts carrying only bytes this process produced. A
// body without an image part is sent as it came. Embedding and completion
// inputs may carry text and token ids. Everything else that could name media
// is refused.
func checkEngineMedia(raw []byte) ([]byte, error) {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(raw, &body); err != nil || body == nil {
		return nil, fmt.Errorf("%w: %s: the body is not one JSON object", ErrEngineMedia, MediaRequest)
	}
	rewritten := false
	if messages, ok := body["messages"]; ok {
		dec := json.NewDecoder(bytes.NewReader(messages))
		dec.UseNumber()
		var list []map[string]any
		if err := dec.Decode(&list); err != nil {
			return nil, fmt.Errorf("%w: %s: messages is not a list of objects", ErrEngineMedia, MediaRequest)
		}
		for _, m := range list {
			images, err := checkContent(m["content"])
			if err != nil {
				return nil, err
			}
			rewritten = rewritten || images
		}
		if rewritten {
			encoded, err := encodeJSON(list)
			if err != nil {
				return nil, fmt.Errorf("%w: %s: %v", ErrEngineMedia, MediaRequest, err)
			}
			body["messages"] = encoded
		}
	}
	for _, key := range []string{"input", "prompt"} {
		if value, ok := body[key]; ok {
			if err := checkTextInput(key, value); err != nil {
				return nil, err
			}
		}
	}
	if !rewritten {
		return raw, nil
	}
	out, err := encodeJSON(body)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrEngineMedia, MediaRequest, err)
	}
	return out, nil
}

// encodeJSON encodes v without HTML escaping or a trailing newline.
func encodeJSON(v any) ([]byte, error) {
	var out bytes.Buffer
	enc := json.NewEncoder(&out)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimSuffix(out.Bytes(), []byte("\n")), nil
}

// checkContent applies the kit's media policy to one message's content and
// reports whether it rewrote an image part.
func checkContent(content any) (bool, error) {
	switch c := content.(type) {
	case nil, string:
		return false, nil
	case []any:
		images := false
		for _, part := range c {
			if err := engineMedia.Part(part); err != nil {
				var media *llamacpp.MediaError
				if errors.As(err, &media) {
					return false, fmt.Errorf("%w: %s: %s", ErrEngineMedia, mediaWords[media.Reason], media.Detail)
				}
				return false, err
			}
			if p, _ := part.(map[string]any); p["type"] == "image_url" {
				images = true
			}
		}
		return images, nil
	}
	return false, fmt.Errorf("%w: %s: message content is neither text nor a list of parts", ErrEngineMedia, MediaType)
}

// checkTextInput accepts a string, a list of strings, a list of token ids,
// or a list of such lists.
func checkTextInput(key string, value json.RawMessage) error {
	var text string
	if json.Unmarshal(value, &text) == nil {
		return nil
	}
	var list []json.RawMessage
	if err := json.Unmarshal(value, &list); err != nil {
		return fmt.Errorf("%w: %s: %s is neither text nor a list", ErrEngineMedia, MediaType, key)
	}
	for _, item := range list {
		var s string
		var n json.Number
		var ids []json.Number
		if json.Unmarshal(item, &s) == nil || json.Unmarshal(item, &n) == nil || json.Unmarshal(item, &ids) == nil {
			continue
		}
		return fmt.Errorf("%w: %s: %s carries an entry other than text or token ids", ErrEngineMedia, MediaType, key)
	}
	return nil
}
