package main

import (
	"context"
	"errors"
	"fmt"
	timestampwire "github.com/openabstractions/abstraction-job/go/rec"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/openabstractions/abstraction-facade/go/programfiles"
	identity "github.com/openabstractions/abstraction-identity"
	"github.com/openabstractions/abstraction-identity/fileid"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
	"github.com/openabstractions/abstraction-provider-modelhost/go/service"
)

// prepareServeEngine reconciles the engine this provider previously staged in
// OA's cache. Retained declarations survive package upgrades, including a pin
// whose release name stays the same while its required runtime files change.
// Custom paths retain the ordinary per-call proof and are never replaced here.
func prepareServeEngine(ctx context.Context, pin host.EnginePin, program, state, self, dir string, files *programfiles.Verifier) error {
	_, err := prepareServeEngineObserved(ctx, pin, program, state, self, dir, files, io.Discard)
	return err
}

// prepareServeEngineObserved writes one line to diagnostics when it sets the
// read-only engine ACL again on a folder whose files all prove.
func prepareServeEngineObserved(ctx context.Context, pin host.EnginePin, program, state, self, dir string, files *programfiles.Verifier, diagnostics io.Writer) (check service.EngineCheck, result error) {
	check = service.EngineCheck{Kind: service.EngineCheckKind, Copy: pin.Release, State: service.EngineUnverified, Reason: "engine:custom_path"}
	defer func() {
		check.Checked = timestampwire.FormatTimestamp(time.Now())
		if result != nil {
			check.State = service.EngineRefused
			check.Reason = host.WordOf(result)
			if check.Reason == "" {
				check.Reason = "engine:unavailable"
			}
		}
	}()
	if !filepath.IsAbs(program) || !identity.SamePathName(filepath.Base(program), pin.Program) {
		return check, nil
	}
	if _, err := os.Lstat(dir); errors.Is(err, fs.ErrNotExist) {
		canonical, canonicalErr := fileid.ResolveLeaf(ctx, fileid.Self, dir)
		declared, declaredErr := fileid.ResolveLeaf(ctx, fileid.Self, filepath.Dir(program))
		if canonicalErr != nil || declaredErr != nil || canonical.Key() != declared.Key() {
			return check, nil
		}
		if err := restoreMissingServeEngine(ctx, pin, program, state, self, dir, files); err != nil {
			if errors.Is(err, errUnprovenManagedEngine) {
				return check, nil
			}
			return check, err
		}
		check.State, check.Reason = service.EngineRepaired, ""
		return check, nil
	}
	if same, err := identity.SameFile(filepath.Dir(program), dir); err != nil || !same {
		return check, nil
	}
	record := host.EngineRecord(state, pin)
	if err := host.RecordedEngine(record, dir); err != nil {
		return check, err
	}
	manifest := append(append([]programfiles.File{}, pin.Files...), pin.Runtime.Files...)
	sort.Slice(manifest, func(i, j int) bool { return manifest[i].Path < manifest[j].Path })
	tree := programfiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: programfiles.Manifest{Files: manifest}, Record: record}
	// A removal of OpenAbstractions runs `engine unlock` (INF-F3) and a later
	// install keeps the unlocked folder. CheckRepairing sets the read-only ACL
	// again when every file proves against the pin. A file that differs goes
	// on to the bundled copy below.
	if repaired, err := files.CheckRepairing(ctx, tree, diagnostics); err == nil {
		check.State, check.Reason = service.EngineCurrent, ""
		if repaired {
			check.State = service.EngineRepaired
		}
		return check, nil
	} else if !errors.Is(err, programfiles.ErrMismatch) {
		return check, err
	}
	_, err := host.InstallBundledEngine(ctx, files, pin, filepath.Join(filepath.Dir(self), "engines", pin.Bundle.Name), dir, record)
	if err != nil {
		return check, fmt.Errorf("%s: the files in %s differ from the pinned release and the copy that ships with OpenAbstractions could not replace them: %w. "+
			"Repair OpenAbstractions in Windows Settings > Apps > Installed apps, then start a new chat", tree.Name, dir, err)
	}
	check.State, check.Reason = service.EngineRepaired, ""
	return check, nil
}

// The missing-leaf path has no directory FileID to compare. Bind its existing
// native parent and leaf identity instead, under the ordinary namespace hold.
// The record and declared target must name that same canonical placement.
var errUnprovenManagedEngine = errors.New("engine record does not prove the managed leaf")

func restoreMissingServeEngine(ctx context.Context, pin host.EnginePin, program, state, self, dir string, files *programfiles.Verifier) error {
	parent, err := programfiles.HoldDirectory(filepath.Dir(dir))
	if err != nil {
		return err
	}
	defer parent.Release()
	record := host.EngineRecord(state, pin)
	if err = programfiles.CheckPlainPath(record); err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return errUnprovenManagedEngine
		}
		return err
	}
	root, err := host.RecordedRoot(record)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return errUnprovenManagedEngine
		}
		return err
	}
	canonical, err := fileid.ResolveLeaf(ctx, fileid.Self, dir)
	if err != nil {
		return err
	}
	for _, target := range []string{filepath.Dir(program), root} {
		hold, err := programfiles.HoldDirectory(filepath.Dir(target))
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return errUnprovenManagedEngine
			}
			return err
		}
		defer hold.Release()
		leaf, err := fileid.ResolveLeaf(ctx, fileid.Self, target)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return errUnprovenManagedEngine
			}
			return err
		}
		if leaf.Key() != canonical.Key() {
			return errUnprovenManagedEngine
		}
	}
	if h, err := fileid.Open(ctx, fileid.Self, dir, fileid.Options{Links: fileid.RefuseLinks}); err == nil {
		h.Close()
		return fs.ErrExist
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if _, err = host.RestoreBundledEngine(ctx, files, pin, filepath.Join(filepath.Dir(self), "engines", pin.Bundle.Name), dir, record); err != nil {
		return err
	}
	if err = host.RecordedEngine(record, dir); err != nil {
		return err
	}
	manifest := append(append([]programfiles.File{}, pin.Files...), pin.Runtime.Files...)
	return files.Check(ctx, programfiles.Tree{Name: "llama.cpp " + pin.Release, Root: dir, Manifest: programfiles.Manifest{Files: manifest}, Record: record})
}
