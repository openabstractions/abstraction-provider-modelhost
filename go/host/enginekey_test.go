package host

import (
	"bytes"
	"context"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// lockedBuffer is a bytes.Buffer the engine's output goroutine and the test
// share.
type lockedBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *lockedBuffer) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *lockedBuffer) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

// directSocket is a client of another local program of this account that
// dials the engine's socket itself, with no key and no peer check.
func directSocket(h *Host, host string) *http.Client {
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			endpoint, ok := h.EngineEndpoint(host)
			if !ok {
				return nil, net.ErrClosed
			}
			return (&net.Dialer{}).DialContext(ctx, "unix", endpoint.Path)
		},
	}}
}

// b10825 review condition 6 closure (CONTRACT.md MODELHOST-L8): each engine
// start gets its own 32-byte key as LLAMA_API_KEY in the engine's
// environment, and only this provider's engine client sends it. A local
// process of this account that reaches the engine's socket directly is
// refused; /health stays open for readiness. The key never appears on the engine's command line or
// in its output or this provider's reports.
func TestTheEngineRequiresThePerLaunchKey(t *testing.T) {
	requireEnginePeer(t)
	e := copiedEngine(t)
	t.Setenv("LLAMA_API_KEY", "planted-by-test")
	argsFile := filepath.Join(t.TempDir(), "args")
	e.fixture = []string{"--fake-args-file", argsFile}
	output := &lockedBuffer{}
	e.Stderr = output
	var reports []string
	var mu sync.Mutex
	h := composed(t, newFakeLeases(), func(o *Options) {
		o.Engine = e
		o.OnError = func(err error) { mu.Lock(); reports = append(reports, err.Error()); mu.Unlock() }
	})
	base, err := h.Ensure(context.Background(), "tiny:1b")
	if err != nil {
		t.Fatal(err)
	}
	address, err := url.Parse(base)
	if err != nil {
		t.Fatal(err)
	}
	chat := `{"model":"tiny:1b","messages":[{"role":"user","content":"hi"}]}`
	post := func(client *http.Client, authorization string) int {
		t.Helper()
		req, err := http.NewRequest(http.MethodPost, base+"/v1/chat/completions", strings.NewReader(chat))
		if err != nil {
			t.Fatal(err)
		}
		if authorization != "" {
			req.Header.Set("Authorization", authorization)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		//unchecked: test drains the body for connection reuse
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp.StatusCode
	}
	if status := post(directSocket(h, address.Host), ""); status != http.StatusUnauthorized {
		t.Fatalf("a direct request without the key read %d, wanted 401", status)
	}
	if status := post(directSocket(h, address.Host), "Bearer planted-by-test"); status != http.StatusUnauthorized {
		t.Fatalf("a direct request with the provider's inherited LLAMA_API_KEY read %d, wanted 401", status)
	}
	if status := post(EngineClient(h, nil), ""); status != http.StatusOK {
		t.Fatalf("the provider's engine client read %d, wanted 200", status)
	}
	health, err := directSocket(h, address.Host).Get(base + "/health")
	if err != nil {
		t.Fatal(err)
	}
	health.Body.Close()
	if health.StatusCode != http.StatusOK {
		t.Fatalf("/health without the key read %d", health.StatusCode)
	}
	key, ok := h.EngineKey(address.Host)
	if !ok || len(key) != 64 {
		t.Fatalf("the engine at %s has no 32-byte key", address.Host)
	}
	argv, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(argv), key) {
		t.Fatal("the key is on the engine's command line")
	}
	if !strings.Contains(string(argv), "--no-webui") {
		t.Fatal("the provider did not pass --no-webui")
	}
	mu.Lock()
	reported := strings.Join(reports, "\n")
	mu.Unlock()
	if strings.Contains(output.String(), key) || strings.Contains(reported, key) {
		t.Fatal("the key is in the engine's output or the provider's reports")
	}
	h.Unload("tiny:1b", "test")
	if _, err := h.Ensure(context.Background(), "tiny:1b"); err != nil {
		t.Fatal(err)
	}
	again, ok := h.EngineKey(address.Host)
	if !ok || again == key {
		t.Fatal("the second launch reused the first launch's key")
	}
}
