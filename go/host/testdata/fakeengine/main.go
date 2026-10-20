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
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

var heldMemory []byte

func main() {
	model := flag.String("model", "", "the model file this engine was told to load")
	alias := flag.String("alias", "", "the name it answers under")
	host := flag.String("host", "127.0.0.1", "")
	port := flag.Int("port", 0, "")
	embeddings := flag.Bool("embeddings", false, "")
	load := flag.Duration("fake-load", 0, "how long this engine pretends to be loading")
	alloc := flag.Uint64("fake-alloc", 0, "commit this many bytes before readiness")
	childPIDFile := flag.String("fake-child-pid-file", "", "write a spawned child's PID here")
	childMode := flag.Bool("fake-child-mode", false, "wait as a test descendant")
	exitAfterChild := flag.Bool("fake-exit-after-child", false, "exit after spawning a test descendant")
	cwdFile := flag.String("fake-cwd-file", "", "write the working directory here")
	envFile := flag.String("fake-env-file", "", "write the environment's names here")
	argsFile := flag.String("fake-args-file", "", "write the command line here")
	pidFile := flag.String("fake-pid-file", "", "write this engine's process id here")
	noListen := flag.Bool("fake-no-listen", false, "start without serving, as an engine whose socket path another process took")
	warmFile := flag.String("fake-warm-file", "", "append the prompt length of each /completion here")
	completionGate := flag.String("fake-completion-gate", "", "hold each /completion until this file exists")
	failLine := flag.String("fake-fail-line", "", "print this line to stderr and exit 1 before serving, as a load that failed")
	flag.Bool("no-webui", false, "")
	// The placement options the provider passes by default (MODELHOST-L9)
	// and a person may give instead.
	flag.String("n-gpu-layers", "", "")
	flag.String("load-mode", "", "")
	flag.Bool("no-mmap", false, "")
	flag.Bool("mmap", false, "")
	// Unknown llama-server flags are tolerated the way a real one tolerates
	// the extra arguments a person configured.
	flag.CommandLine.Init("fakeengine", flag.ContinueOnError)
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *failLine != "" {
		fmt.Fprintln(os.Stderr, *failLine)
		os.Exit(1)
	}
	if *childMode {
		if err := os.WriteFile(*childPIDFile, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(2)
		}
		for {
			time.Sleep(time.Hour)
		}
	}
	if *cwdFile != "" {
		cwd, err := os.Getwd()
		if err != nil || os.WriteFile(*cwdFile, []byte(cwd), 0o600) != nil {
			os.Exit(2)
		}
	}
	if *envFile != "" {
		// One line per variable: its name, and its value only for the few
		// variables a clean engine environment holds. No other value is
		// written, so an inherited secret never lands in a test file.
		shown := map[string]bool{"PATH": true, "TEMP": true, "TMP": true, "TMPDIR": true, "SYSTEMROOT": true, "WINDIR": true,
			"SYSTEMDRIVE": true, "USERPROFILE": true, "LOCALAPPDATA": true, "APPDATA": true, "PROGRAMDATA": true,
			"ALLUSERSPROFILE": true, "NUMBER_OF_PROCESSORS": true, "PROCESSOR_ARCHITECTURE": true}
		var lines []string
		for _, item := range os.Environ() {
			name, value, _ := strings.Cut(item, "=")
			if name == "" {
				continue
			}
			if shown[strings.ToUpper(name)] {
				lines = append(lines, name+"\t"+value)
			} else {
				lines = append(lines, name)
			}
		}
		if os.WriteFile(*envFile, []byte(strings.Join(lines, "\n")), 0o600) != nil {
			os.Exit(2)
		}
	}
	if *argsFile != "" {
		if os.WriteFile(*argsFile, []byte(strings.Join(os.Args, "\n")), 0o600) != nil {
			os.Exit(2)
		}
	}
	if *pidFile != "" {
		if os.WriteFile(*pidFile, []byte(strconv.Itoa(os.Getpid())), 0o600) != nil {
			os.Exit(2)
		}
	}
	// llama-server b10825 serves AF_UNIX when --host ends in ".sock" and
	// ignores the port then (tools/server/server-http.cpp:438-444).
	socket := strings.HasSuffix(*host, ".sock")
	if *model == "" || (*port == 0 && !socket) {
		fmt.Fprintln(os.Stderr, "fakeengine: --model, and --port or a .sock --host, are required")
		os.Exit(2)
	}
	if _, err := os.Stat(*model); err != nil {
		fmt.Fprintf(os.Stderr, "fakeengine: %v\n", err)
		os.Exit(2)
	}
	if *alloc > 0 {
		heldMemory = make([]byte, int(*alloc))
		for i := 0; i < len(heldMemory); i += 4096 {
			heldMemory[i] = 1
		}
	}
	if *childPIDFile != "" {
		child := exec.Command(os.Args[0], "--fake-child-mode", "--fake-child-pid-file", *childPIDFile)
		if err := child.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "fakeengine: spawn child: %v\n", err)
			os.Exit(2)
		}
		if *exitAfterChild {
			deadline := time.Now().Add(5 * time.Second)
			for time.Now().Before(deadline) {
				if _, err := os.Stat(*childPIDFile); err == nil {
					break
				}
				time.Sleep(5 * time.Millisecond)
			}
			os.Exit(13)
		}
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
	// The warm's routes (MODELHOST-W2): a template that joins each message's
	// text content, a tokenizer of one token per byte, and a completion that
	// records how many token ids it was given.
	mux.HandleFunc("/apply-template", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		var prompt strings.Builder
		for _, m := range body.Messages {
			prompt.WriteString("[" + m.Content + "]")
		}
		json.NewEncoder(w).Encode(map[string]string{"prompt": prompt.String()})
	})
	mux.HandleFunc("/tokenize", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Content string `json:"content"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		tokens := make([]int, 0, len(body.Content))
		for _, b := range []byte(body.Content) {
			tokens = append(tokens, int(b))
		}
		json.NewEncoder(w).Encode(map[string]any{"tokens": tokens})
	})
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Prompt []int `json:"prompt"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if *warmFile != "" {
			f, err := os.OpenFile(*warmFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
			if err == nil {
				fmt.Fprintln(f, len(body.Prompt))
				f.Close()
			}
		}
		// A gate holds the evaluation open until its file exists, as a long
		// prefix holds a real engine.
		for *completionGate != "" {
			if _, err := os.Stat(*completionGate); err == nil {
				break
			}
			select {
			case <-r.Context().Done():
				return
			case <-time.After(10 * time.Millisecond):
			}
		}
		json.NewEncoder(w).Encode(map[string]any{"content": "", "tokens_evaluated": len(body.Prompt)})
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
	// llama-server with LLAMA_API_KEY answers 401 to every route but its
	// public ones (/health, /models) without "Authorization: Bearer KEY".
	key := os.Getenv("LLAMA_API_KEY")
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		public := r.URL.Path == "/health" || r.URL.Path == "/v1/health" || r.URL.Path == "/models" || r.URL.Path == "/v1/models"
		if key != "" && !public && r.Header.Get("Authorization") != "Bearer "+key {
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}`)
			return
		}
		mux.ServeHTTP(w, r)
	})
	if *noListen {
		for {
			time.Sleep(time.Hour)
		}
	}
	// The stand-in listens on its socket path or on loopback alone: a
	// listener beyond loopback makes Windows Defender Firewall prompt the
	// person at the machine.
	var listener net.Listener
	var err error
	switch {
	case socket:
		listener, err = net.Listen("unix", *host)
	case *host == "127.0.0.1":
		listener, err = net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(*port))
	default:
		fmt.Fprintf(os.Stderr, "fakeengine listens on a .sock path or 127.0.0.1 only, not %q\n", *host)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "fakeengine: %v\n", err)
		os.Exit(1)
	}
	if err := http.Serve(listener, handler); err != nil {
		fmt.Fprintf(os.Stderr, "fakeengine: %v\n", err)
		os.Exit(1)
	}
}
