package host

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log/slog"
	"os"
	"strings"
	"testing"
)

// Ordinary and live host fixtures own their diagnostics locally. Relocating
// per-user folders does not isolate the installed runtime's native endpoint.
func newTestHost(options Options) (*Host, error) {
	return newTestHostDiagnostics(options, io.Discard, nil)
}

func newTestHostDiagnostics(options Options, output io.Writer, resolved func(string)) (*Host, error) {
	return newHost(options, func(program string) (slog.Handler, func()) {
		if resolved != nil {
			resolved(program)
		}
		return slog.NewTextHandler(output, nil), func() {}
	})
}

func TestHostTestDiagnosticsStayLocal(t *testing.T) {
	var output bytes.Buffer
	resolutions := 0
	options := Options{Models: []Model{{Name: "fixture", Store: "store", Object: "object"}}, Source: fakeSource{}, Leases: newFakeLeases()}
	h, err := newTestHostDiagnostics(options, &output, func(program string) {
		resolutions++
		if program != "modelhostd" {
			t.Errorf("diagnostics program %q", program)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	h.loadLogger.Info("fixture-owned diagnostics", "result", "captured")
	h.Close()
	if resolutions != 1 || !strings.Contains(output.String(), "fixture-owned diagnostics") || !strings.Contains(output.String(), "result=captured") {
		t.Fatalf("local resolutions=%d captured=%q", resolutions, output.String())
	}
	ordinary, err := newTestHost(options)
	if err != nil {
		t.Fatal(err)
	}
	defer ordinary.Close()
	if _, ok := ordinary.loadLogger.Handler().(*slog.TextHandler); !ok {
		t.Fatal("ordinary fixture did not select its local text handler")
	}
}

// This regression inspects source only: its failing control never resolves or
// connects to the installed runtime. The explicit owned logging fixture uses
// newHost with ResolveVerified and remains independently exercised.
func TestHostTestConstructorsUseOwnedDiagnostics(t *testing.T) {
	unsafe, err := parser.ParseFile(token.NewFileSet(), "control.go", `package host
func control() { New(Options{}); resolvedDiagnostics("fixture") }`, 0)
	if err != nil || len(ambientTestConstructorCalls(unsafe)) != 2 {
		t.Fatalf("inert source control did not expose ambient constructors: %v", err)
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), entry.Name(), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, name := range ambientTestConstructorCalls(file) {
			t.Errorf("%s calls ambient %s; use the test-owned constructor", entry.Name(), name)
		}
	}
}

func ambientTestConstructorCalls(file *ast.File) []string {
	var names []string
	if file == nil {
		return names
	}
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		name, ok := call.Fun.(*ast.Ident)
		if ok && (name.Name == "New" || name.Name == "resolvedDiagnostics") {
			names = append(names, name.Name)
		}
		return true
	})
	return names
}
