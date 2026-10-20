package host

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// engineRecorder stands where the engine's HTTP server stands and records
// every body that reached it.
type engineRecorder struct {
	mu     sync.Mutex
	bodies []string
}

func (r *engineRecorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	raw, _ := io.ReadAll(req.Body)
	r.mu.Lock()
	r.bodies = append(r.bodies, string(raw))
	r.mu.Unlock()
	w.WriteHeader(http.StatusOK)
}

func (r *engineRecorder) reached() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.bodies...)
}

// inProcessBase is the engine base the media tests post to; inProcess
// answers it in this process, with no socket.
const inProcessBase = "http://engine0.modelhost.invalid"

// inProcess serves each request with its handler in this process, where the
// engine's socket transport stands in EngineClient.
type inProcess struct{ handler http.Handler }

func (p inProcess) RoundTrip(req *http.Request) (*http.Response, error) {
	recorder := httptest.NewRecorder()
	p.handler.ServeHTTP(recorder, req)
	response := recorder.Result()
	response.Request = req
	return response, nil
}

func encodedImage(t *testing.T, format string) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 9, 5))
	img.Set(1, 1, color.RGBA{G: 200, A: 255})
	var buf bytes.Buffer
	var err error
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpeg":
		err = jpeg.Encode(&buf, img, nil)
	case "gif":
		err = gif.Encode(&buf, img, nil)
	}
	if err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func dataURL(mime string, data []byte) string {
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func chatWith(part string) string {
	return `{"model":"tiny:1b","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"look"},` + part + `]}]}`
}

// b10825 review condition 6 (CONTRACT.md MODELHOST-L8): llama-server fetches
// an http(s) media URL itself, reads file:// under --media-path, and spawns
// FFmpeg for video and WebP. No request the provider sends its engine
// carries any of these: the client the inference provider reaches the
// engines with refuses such a body before it leaves this process.
func TestNoMediaURLReachesTheEngine(t *testing.T) {
	png := "data:image/png;base64,iVBORw0KGgo="
	refused := map[string]string{
		"http image":  chatWith(`{"type":"image_url","image_url":{"url":"http://10.0.0.1/a.png"}}`),
		"https image": chatWith(`{"type":"image_url","image_url":{"url":"HTTPS://example.invalid/a.png"}}`),
		"file image":  chatWith(`{"type":"image_url","image_url":{"url":"file:///C:/Windows/win.ini"}}`),
		"webp data":   chatWith(`{"type":"image_url","image_url":{"url":"data:image/webp;base64,UklGRg=="}}`),
		"svg data":    chatWith(`{"type":"image_url","image_url":{"url":"data:image/svg+xml;base64,PHN2Zz4="}}`),
		"bare path":   chatWith(`{"type":"image_url","image_url":{"url":"C:\\x\\a.png"}}`),
		// mtmd detects an image's format by its bytes, so a declared type
		// must match them: WebP bytes under a PNG label still reach FFmpeg.
		"webp as png": chatWith(`{"type":"image_url","image_url":{"url":"data:image/png;base64,UklGRiQAAABXRUJQ"}}`),
		"jpeg as png": chatWith(`{"type":"image_url","image_url":{"url":"data:image/png;base64,/9j/4AAQ"}}`),
		"png as gif":  chatWith(`{"type":"image_url","image_url":{"url":"data:image/gif;base64,iVBORw0KGgo="}}`),
		"bmp as jpeg": chatWith(`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,Qk02AAAA"}}`),
		"not base64":  chatWith(`{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!!"}}`),
		// Every image is decoded before it is forwarded: a signature
		// alone, GIF and BMP are refused (MODELHOST-L8).
		"png signature":    chatWith(`{"type":"image_url","image_url":{"url":"` + png + `"}}`),
		"jpeg signature":   chatWith(`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/4AAQ"}}`),
		"gif data":         chatWith(`{"type":"image_url","image_url":{"url":"` + dataURL("image/gif", encodedImage(t, "gif")) + `"}}`),
		"bmp data":         chatWith(`{"type":"image_url","image_url":{"url":"data:image/bmp;base64,Qk02AAAA"}}`),
		"real jpeg as png": chatWith(`{"type":"image_url","image_url":{"url":"` + dataURL("image/png", encodedImage(t, "jpeg")) + `"}}`),
		"empty png data":   chatWith(`{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}`),
		"string image_url": chatWith(`{"type":"image_url","image_url":"http://10.0.0.1/a.png"}`),
		"audio url":        chatWith(`{"type":"input_audio","input_audio":{"url":"http://10.0.0.1/a.wav"}}`),
		"audio data":       chatWith(`{"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}`),
		"video data":       chatWith(`{"type":"input_video","input_video":{"url":"data:video/mp4;base64,AAAA"}}`),
		"video url part":   chatWith(`{"type":"video_url","video_url":{"url":"https://example.invalid/v.mp4"}}`),
		"file part":        chatWith(`{"type":"file","file":{"file_data":"x"}}`),
		"untyped part":     chatWith(`{"image_url":{"url":"http://10.0.0.1/a.png"}}`),
		"embed object":     `{"model":"tiny:1b","input":[{"multimodal_data":["AAAA"]}]}`,
		"not json":         `model=tiny`,
	}
	allowed := map[string]string{
		"string content": `{"model":"tiny:1b","messages":[{"role":"user","content":"see http://example.invalid/a.png"}]}`,
		"text parts":     chatWith(`{"type":"text","text":"http://example.invalid/a.png"}`),
		"tool turn": `{"model":"tiny:1b","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},` +
			`{"role":"tool","tool_call_id":"c1","content":"http://example.invalid"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"image_url":{"type":"string"}}}}}]}`,
		"embed strings": `{"model":"tiny:1b","input":["a","b"]}`,
	}
	engine := &engineRecorder{}
	var reported []error
	client := engineClient(nil, func(err error) { reported = append(reported, err) }, inProcess{engine})
	post := func(body string) *http.Response {
		t.Helper()
		resp, err := client.Post(inProcessBase+"/v1/chat/completions", "application/json", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatalf("post: %v", err)
		}
		//unchecked: test drains the body for connection reuse
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp
	}
	for name, body := range refused {
		before := len(engine.reached())
		resp := post(body)
		if resp.StatusCode != http.StatusForbidden {
			t.Errorf("%s: status %d, wanted %d", name, resp.StatusCode, http.StatusForbidden)
		}
		if after := len(engine.reached()); after != before {
			t.Errorf("%s: the body reached the engine", name)
		}
	}
	if len(reported) != len(refused) {
		t.Errorf("reported %d refusals, wanted %d", len(reported), len(refused))
	}
	for name, body := range allowed {
		before := len(engine.reached())
		resp := post(body)
		got := engine.reached()
		if resp.StatusCode != http.StatusOK || len(got) != before+1 || got[len(got)-1] != body {
			t.Errorf("%s: status %d, the engine received the body unchanged: %v", name, resp.StatusCode, len(got) == before+1 && got[len(got)-1] == body)
		}
	}
	for _, err := range reported {
		if !strings.Contains(err.Error(), "media") {
			t.Errorf("refusal %q does not name the media rule", err)
		}
	}
}

// MODELHOST-L8 through the provider kit's media policy (inference
// CONTRACT.md INF-LC3): a PNG or JPEG image reaches the engine only as the
// provider's own 8-bit NRGBA PNG, so stb_image parses bytes this process
// produced. Large integers elsewhere in the body keep their digits.
func TestEngineImagesAreForwardedAsProviderPNG(t *testing.T) {
	engine := &engineRecorder{}
	client := engineClient(nil, nil, inProcess{engine})
	for name, given := range map[string]string{
		"png":  dataURL("image/png", encodedImage(t, "png")),
		"jpeg": dataURL("image/jpeg", encodedImage(t, "jpeg")),
	} {
		body := `{"model":"tiny:1b","seed":12345678901234567,"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"` + given + `"}}]}]}`
		resp, err := client.Post(inProcessBase+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		got := engine.reached()
		if resp.StatusCode != http.StatusOK || len(got) == 0 {
			t.Fatalf("%s: status %d", name, resp.StatusCode)
		}
		sent := got[len(got)-1]
		// Go's PNG encoder may write a PNG it decoded byte for byte; a JPEG
		// always changes.
		if name == "jpeg" && strings.Contains(sent, given) || !strings.Contains(sent, `"seed":12345678901234567`) {
			t.Fatalf("%s: the engine received the caller's JPEG or a changed seed", name)
		}
		var request struct {
			Messages []struct {
				Content []struct {
					ImageURL *struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal([]byte(sent), &request); err != nil {
			t.Fatal(err)
		}
		uri := request.Messages[0].Content[1].ImageURL.URL
		payload, ok := strings.CutPrefix(uri, "data:image/png;base64,")
		if !ok {
			t.Fatalf("%s: forwarded as %.30s", name, uri)
		}
		data, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			t.Fatal(err)
		}
		img, format, err := image.Decode(bytes.NewReader(data))
		if err != nil || format != "png" || img.Bounds().Dx() != 9 || img.Bounds().Dy() != 5 || data[24] != 8 {
			t.Fatalf("%s: forwarded %s %v", name, format, err)
		}
	}
}
