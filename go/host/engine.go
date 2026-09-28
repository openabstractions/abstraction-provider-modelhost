package host

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"
)

// An Engine is a llama.cpp server program already installed on this machine.
// The path is configured; this package looks for none, downloads none and
// ships none (research/model-bridge/DECISION.md §4 rules 4 and 6).
type Engine struct {
	// Program is the absolute path of llama-server on this machine.
	Program string
	// Arguments are extra arguments a person configured, placed after the
	// model, host and port this provider chooses.
	Arguments []string
	// Embeddings asks the engine to serve the embeddings endpoint as well.
	Embeddings bool
	// Start, ready and stop budgets; zero reads the defaults below.
	Ready time.Duration
	// Stderr, when set, receives the engine's own output.
	Stderr io.Writer
}

// DefaultReady is how long a model has to load before the call that asked for
// it gives up. A cold read of a multi-gigabyte file off a spinning disk is the
// case this has to fit.
const DefaultReady = 180 * time.Second

// readyPoll is how often the engine's own health endpoint is read while it
// loads.
const readyPoll = 250 * time.Millisecond

// present reports the engine this machine has, or why there is none. It is the
// only check made before a lease is asked for, because asking the card for
// bytes no engine can use would take them from a holder for nothing.
func (e Engine) present() error {
	if e.Program == "" {
		return errors.New("no llama.cpp engine is configured; declare one with --engine")
	}
	if !filepath.IsAbs(e.Program) {
		return fmt.Errorf("the engine path %s is not absolute", e.Program)
	}
	info, err := os.Stat(e.Program)
	if err != nil {
		return fmt.Errorf("no engine at %s", e.Program)
	}
	if info.IsDir() {
		return fmt.Errorf("the engine path %s is a directory", e.Program)
	}
	return nil
}

// A Process is one running engine, owned by this provider: it was started
// here, it is stopped here by its own process id, and no other program's
// engine is ever touched.
type Process struct {
	cmd    *exec.Cmd
	base   string
	exited chan struct{}
}

// Base is where this engine answers.
func (p *Process) Base() string { return p.base }

// PID is the process id this provider started.
func (p *Process) PID() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

// FreePort is a loopback port nothing holds right now. The provider takes one
// at start and reuses it for every load, so the address a call is mediated to
// never changes while it serves.
func FreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// start runs one engine for modelPath on port, under alias, and returns as
// soon as the child is running. It does not wait for the model to load.
func (e Engine) start(modelPath, alias string, port int) (*Process, error) {
	if err := e.present(); err != nil {
		return nil, err
	}
	args := []string{"--model", modelPath, "--host", "127.0.0.1", "--port", strconv.Itoa(port)}
	if alias != "" {
		args = append(args, "--alias", alias)
	}
	if e.Embeddings {
		args = append(args, "--embeddings")
	}
	args = append(args, e.Arguments...)
	cmd := exec.Command(e.Program, args...)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if e.Stderr != nil {
		cmd.Stdout, cmd.Stderr = e.Stderr, e.Stderr
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &Process{cmd: cmd, base: "http://127.0.0.1:" + strconv.Itoa(port), exited: make(chan struct{})}
	//unchecked: this reaper goroutine only signals p.exited for the health/stop watchers below; the exit error itself is never surfaced anywhere
	go func() { cmd.Wait(); close(p.exited) }()
	return p, nil
}

// wait polls the engine's own health endpoint until it reports the model
// loaded, the child exits, or the budget passes.
func (p *Process) wait(ctx context.Context, budget time.Duration, client *http.Client) error {
	if budget <= 0 {
		budget = DefaultReady
	}
	deadline := time.Now().Add(budget)
	for {
		select {
		case <-p.exited:
			return fmt.Errorf("the engine exited before it was ready")
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if p.healthy(ctx, client) {
			return nil
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("the engine was not ready within %s", budget)
		}
		timer := time.NewTimer(readyPoll)
		select {
		case <-timer.C:
		case <-p.exited:
			timer.Stop()
			return fmt.Errorf("the engine exited before it was ready")
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		}
	}
}

// healthy reads llama.cpp's own /health, which answers 503 while the weights
// are still being read and 200 once the model can answer.
func (p *Process) healthy(ctx context.Context, client *http.Client) bool {
	call, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(call, http.MethodGet, p.base+"/health", nil)
	if err != nil {
		return false
	}
	response, err := client.Do(request)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	//unchecked: draining the body for connection reuse; the health verdict below depends only on the status code
	io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
	return response.StatusCode == http.StatusOK
}

// stop ends the engine this provider started, by its own process id, and waits
// for it to go. No process is ever named by image: the owner's own engines are
// not this provider's to stop.
func (p *Process) stop(within time.Duration) {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return
	}
	//unchecked: best-effort termination; a failed Kill (e.g. the process already exited) is fine since the timeout wait below still proceeds
	p.cmd.Process.Kill()
	timer := time.NewTimer(within)
	defer timer.Stop()
	select {
	case <-p.exited:
	case <-timer.C:
	}
}
