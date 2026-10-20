// Command fakesdserver is a stand-in for stable-diffusion.cpp's sd-server
// with a video model loaded: /sdcpp/v1/capabilities, /sdcpp/v1/vid_gen,
// /sdcpp/v1/jobs/{id} and its cancel route, answering as master-843's
// sd.cpp routes and async job table do, with a small WebM
// as every result. The model host's video tests spawn it, so they measure
// this provider and not sd-server.
//
// It lives under testdata so the module's own build never produces it; the
// tests build it by path into a temporary directory.
package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// webm is a container whose leading bytes net/http sniffs as video/webm.
var webm = []byte("\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\x82\x84webm")

type job struct {
	ID        string
	Status    string
	Frames    int
	FPS       int
	Format    string
	created   time.Time
	cancelled bool
}

type server struct {
	mu       sync.Mutex
	jobs     map[string]*job
	order    []string
	seq      int
	queue    time.Duration
	generate time.Duration
	result   string
	requests string
	cancels  string
	// stall holds every vid_gen answer this long after the job is taken.
	stall time.Duration
	// crash ends the process once a job is read while it generates.
	crash bool
	// deviceLost does the same after printing ggml-vulkan's device-lost
	// line, the way a driver reset ends a Vulkan engine.
	deviceLost bool
	// echoPrompt prints each request's prompt to the engine's output.
	echoPrompt bool
	// lostLineOnly prints ggml-vulkan's device-lost line once while a job
	// generates and keeps serving.
	lostLineOnly bool
	lostPrinted  bool
}

func main() {
	opts := map[string]string{}
	var switches []string
	values := map[string]bool{"--listen-ip": true, "-l": true, "--listen-port": true, "--diffusion-model": true,
		"--t5xxl": true, "--tae": true, "--max-vram": true, "--backend": true, "-t": true, "--vae-tile-size": true}
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		word := args[i]
		if strings.HasPrefix(word, "--fake-") {
			name, value, _ := strings.Cut(word, "=")
			opts[name] = value
			continue
		}
		if values[word] && i+1 < len(args) {
			opts[word] = args[i+1]
			i++
			continue
		}
		switches = append(switches, word)
	}
	if path := opts["--fake-args-file"]; path != "" {
		if os.WriteFile(path, []byte(strings.Join(os.Args[1:], "\n")), 0o600) != nil {
			os.Exit(2)
		}
	}
	if path := opts["--fake-env-file"]; path != "" {
		var names []string
		for _, item := range os.Environ() {
			name, _, _ := strings.Cut(item, "=")
			if name != "" {
				names = append(names, name)
			}
		}
		if os.WriteFile(path, []byte(strings.Join(names, "\n")), 0o600) != nil {
			os.Exit(2)
		}
	}
	if path := opts["--fake-cwd-file"]; path != "" {
		cwd, err := os.Getwd()
		if err != nil || os.WriteFile(path, []byte(cwd), 0o600) != nil {
			os.Exit(2)
		}
	}
	if path := opts["--fake-pid-file"]; path != "" {
		if os.WriteFile(path, []byte(strconv.Itoa(os.Getpid())), 0o600) != nil {
			os.Exit(2)
		}
	}
	for _, flag := range []string{"--diffusion-model", "--t5xxl", "--tae"} {
		if _, err := os.Stat(opts[flag]); err != nil {
			fmt.Fprintf(os.Stderr, "%s: no file\n", flag)
			os.Exit(1)
		}
	}
	if n := opts["--fake-alloc"]; n != "" {
		size, _ := strconv.Atoi(n)
		held := make([]byte, size)
		for i := range held {
			held[i] = 1
		}
		runtime.KeepAlive(held)
	}
	fmt.Fprintln(os.Stderr, "fakesdserver: prompt text an engine log would carry")
	s := &server{jobs: map[string]*job{}, queue: duration(opts["--fake-queue"], 0), generate: duration(opts["--fake-generate"], 100*time.Millisecond),
		result: opts["--fake-result"], requests: opts["--fake-requests-dir"], cancels: opts["--fake-cancels-file"],
		stall: duration(opts["--fake-stall-submit"], 0), crash: opts["--fake-crash-generating"] != "",
		deviceLost: opts["--fake-device-lost"] != "", echoPrompt: opts["--fake-echo-prompt"] != "",
		lostLineOnly: opts["--fake-lost-line-only"] != ""}
	loaded := time.Now().Add(duration(opts["--fake-load"], 0))
	mode := opts["--fake-mode"]
	if mode == "" {
		mode = "vid_gen"
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/sdcpp/v1/capabilities", func(w http.ResponseWriter, r *http.Request) {
		if time.Now().Before(loaded) {
			http.Error(w, "loading", http.StatusServiceUnavailable)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"supported_modes": []string{mode}, "model": opts["--diffusion-model"]})
	})
	mux.HandleFunc("/sdcpp/v1/vid_gen", s.vidGen)
	mux.HandleFunc("/sdcpp/v1/jobs/", s.jobRoute)
	// The stand-in listens on loopback alone: a listener beyond it makes
	// Windows Defender Firewall prompt the person at the machine. Tests of
	// the loopback check inject the socket table instead.
	if ip := opts["--listen-ip"]; ip != "127.0.0.1" {
		fmt.Fprintf(os.Stderr, "fakesdserver listens on 127.0.0.1 only, not %q\n", ip)
		os.Exit(2)
	}
	if _, ok := opts["--fake-no-listen"]; ok {
		// An engine whose port another process took first.
		for {
			time.Sleep(time.Hour)
		}
	}
	listener, err := net.Listen("tcp", "127.0.0.1:"+opts["--listen-port"])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if code := opts["--fake-exit-after"]; code != "" {
		go func() { time.Sleep(duration(code, time.Second)); os.Exit(3) }()
	}
	_ = http.Serve(listener, mux)
}

func duration(v string, fallback time.Duration) time.Duration {
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *server) vidGen(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method", http.StatusMethodNotAllowed)
		return
	}
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, 400, map[string]any{"error": "body"})
		return
	}
	var body struct {
		Prompt string `json:"prompt"`
		Frames int    `json:"video_frames"`
		FPS    int    `json:"fps"`
		Format string `json:"output_format"`
	}
	if json.Unmarshal(raw, &body) != nil {
		writeJSON(w, 400, map[string]any{"error": "invalid json"})
		return
	}
	if s.echoPrompt {
		// sd.cpp logs the prompt as given, line breaks included.
		fmt.Fprintf(os.Stderr, "[INFO ] stable-diffusion.cpp:2104 - prompt: '%s'\n", body.Prompt)
	}
	s.mu.Lock()
	s.seq++
	id := fmt.Sprintf("job_%x_%08d", time.Now().Unix(), s.seq)
	if s.requests != "" {
		_ = os.WriteFile(filepath.Join(s.requests, fmt.Sprintf("req-%d.json", s.seq)), raw, 0o600)
	}
	s.jobs[id] = &job{ID: id, Status: "queued", Frames: body.Frames, FPS: body.FPS, Format: body.Format, created: time.Now()}
	s.order = append(s.order, id)
	stall := s.stall
	s.mu.Unlock()
	// The job is taken before the answer, as sd-server queues it before
	// it replies; a stalled answer leaves a job the caller never learned.
	time.Sleep(stall)
	writeJSON(w, http.StatusAccepted, map[string]any{"id": id, "kind": "vid_gen", "status": "queued", "poll_url": "/sdcpp/v1/jobs/" + id})
}

// advance moves a job through queued, generating and completed by time.
func (s *server) advance(j *job) {
	if j.cancelled || j.Status == "completed" || j.Status == "failed" {
		return
	}
	age := time.Since(j.created)
	switch {
	case age < s.queue:
		j.Status = "queued"
	case age < s.queue+s.generate:
		j.Status = "generating"
	case s.result == "fail":
		j.Status = "failed"
	default:
		j.Status = "completed"
	}
}

func (s *server) jobRoute(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/sdcpp/v1/jobs/")
	id, action, _ := strings.Cut(rest, "/")
	s.mu.Lock()
	defer s.mu.Unlock()
	j, ok := s.jobs[id]
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "job not found"})
		return
	}
	s.advance(j)
	if action == "cancel" && r.Method == http.MethodPost {
		if s.cancels != "" {
			if f, err := os.OpenFile(s.cancels, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
				_, _ = f.WriteString(j.Status + " " + id + "\n")
				_ = f.Close()
			}
		}
		if j.Status == "generating" {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "job is currently generating and cannot be interrupted yet"})
			return
		}
		if j.Status == "queued" {
			j.Status, j.cancelled = "cancelled", true
		}
		writeJSON(w, http.StatusOK, s.view(j))
		return
	}
	if action != "" {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, http.StatusOK, s.view(j))
	if j.Status == "generating" {
		s.progress(j)
	}
	if s.lostLineOnly && j.Status == "generating" && !s.lostPrinted {
		s.lostPrinted = true
		fmt.Fprintln(os.Stderr, "ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost")
	}
	if (s.crash || s.deviceLost) && j.Status == "generating" {
		if s.deviceLost {
			fmt.Fprintln(os.Stderr, "ggml_vulkan: vk::Device::waitForFences: ErrorDeviceLost")
		}
		go func() { time.Sleep(20 * time.Millisecond); os.Exit(3) }()
	}
}

// progress prints what sd.cpp prints for the generation so far, in its
// order: a weight load's '#' bar and the first frame's one-step encode in
// the first fifth, then the generate_video line and three sampling steps up
// to three fifths, then the closing line and eight decoded tiles, as
// print_progress_line and the library's log draw them
// (research/live-runs/2026-10-02.md, run 2a).
func (s *server) progress(j *job) {
	if s.generate <= 0 {
		return
	}
	done := float64(time.Since(j.created)-s.queue) / float64(s.generate)
	bar := func(fill string, step, steps int, rate string) string {
		return "\r  |" + strings.Repeat(fill, 50*step/steps) + strings.Repeat(" ", 50-50*step/steps) + fmt.Sprintf("| %d/%d - %s", step, steps, rate)
	}
	switch {
	case done < 0.2:
		// A finished load bar ends its line, as the library's does.
		fmt.Print(bar("#", 64, 64, "13.91MB/s") + "\n" + bar("=", 1, 1, "1.51it/s") + "\n")
	case done < 0.6:
		fmt.Print("[INFO ] stable-diffusion.cpp:6941 - generate_video 832x480x5\n" + bar("=", max(1, int((done-0.2)/0.4*3)), 3, "1.00s/it"))
	default:
		fmt.Print(bar("=", 3, 3, "1.00s/it") + "\n[INFO ] stable-diffusion.cpp:7038 - sampling completed, taking 1.00s\n" +
			bar("=", max(1, min(8, int((done-0.6)/0.4*8))), 8, "1.00s/it"))
	}
}

func (s *server) view(j *job) map[string]any {
	out := map[string]any{"id": j.ID, "kind": "vid_gen", "status": j.Status, "queue_position": 0, "result": nil, "error": nil}
	if j.Status == "queued" {
		out["queue_position"] = 1
	}
	switch j.Status {
	case "completed":
		data, mime, frames, fps := webm, "video/webm", j.Frames, j.FPS
		switch s.result {
		case "frames":
			frames++
		case "fps":
			fps++
		case "mime":
			mime = "video/mp4"
		case "magic":
			data = []byte("\x00\x00\x00\x18ftypmp42")
		case "huge":
			data = make([]byte, 65<<20)
			copy(data, webm)
		}
		out["result"] = map[string]any{"output_format": j.Format, "mime_type": mime, "fps": fps, "frame_count": frames,
			"b64_json": base64.StdEncoding.EncodeToString(data)}
	case "failed", "cancelled":
		out["error"] = map[string]any{"code": "generation_failed", "message": "engine text carrying the prompt"}
	}
	return out
}
