// Command fakeengine is a stand-in for llama-server: enough of the
// OpenAI-compatible API to report readiness and answer one completion. It is
// the engine the model host's tests spawn, so those tests measure this
// provider's own behaviour — the lease, the load, the reuse, the unload — and
// not llama.cpp's.
//
// It lives under testdata so the module's own build never produces it; the
// tests build it by path into a temporary directory.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"time"
)

func main() {
	model := flag.String("model", "", "the model file this engine was told to load")
	alias := flag.String("alias", "", "the name it answers under")
	host := flag.String("host", "127.0.0.1", "")
	port := flag.Int("port", 0, "")
	embeddings := flag.Bool("embeddings", false, "")
	load := flag.Duration("fake-load", 0, "how long this engine pretends to be loading")
	// Unknown llama-server flags are tolerated the way a real one tolerates
	// the extra arguments a person configured.
	flag.CommandLine.Init("fakeengine", flag.ContinueOnError)
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *model == "" || *port == 0 {
		fmt.Fprintln(os.Stderr, "fakeengine: --model and --port are required")
		os.Exit(2)
	}
	if _, err := os.Stat(*model); err != nil {
		fmt.Fprintf(os.Stderr, "fakeengine: %v\n", err)
		os.Exit(2)
	}
	name := *alias
	if name == "" {
		name = filepath.Base(*model)
	}
	var ready atomic.Bool
	if *load <= 0 {
		ready.Store(true)
	} else {
		go func() { time.Sleep(*load); ready.Store(true) }()
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprint(w, `{"status":"loading model"}`)
			return
		}
		fmt.Fprint(w, `{"status":"ok"}`)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"object": "list",
			"data": []map[string]any{{"id": name, "object": "model"}}})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		text := "hosted by " + name
		chunk := func(v any) {
			raw, _ := json.Marshal(v)
			fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}
		chunk(map[string]any{"choices": []map[string]any{{"delta": map[string]any{"content": text}}}})
		chunk(map[string]any{"choices": []map[string]any{{"delta": map[string]any{}, "finish_reason": "stop"}},
			"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 3}})
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if !*embeddings {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"message":"this engine was not started with embeddings"}}`)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"object": "list", "model": name,
			"data":  []map[string]any{{"index": 0, "embedding": []float64{0.25, -0.5, 0.75, 1}}},
			"usage": map[string]any{"prompt_tokens": 2}})
	})
	address := *host + ":" + strconv.Itoa(*port)
	if err := http.ListenAndServe(address, mux); err != nil {
		fmt.Fprintf(os.Stderr, "fakeengine: %v\n", err)
		os.Exit(1)
	}
}
