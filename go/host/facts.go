package host

import (
	"context"
	"maps"
	"strings"
	"sync"
	"time"

	"github.com/openabstractions/abstraction-identity/fileid"
	bridge "github.com/openabstractions/abstraction-provider-modelbridge/go/bridge"
	"github.com/openabstractions/abstraction-storage-over-local-stores/go/gguf"
)

// ModelFacts is what one allowlisted chat model takes and gives, read from
// its own files and its family's engine profile (MODELHOST-L12). A fact the
// files do not state stays unknown: Tools nil, Reasoning nil.
type ModelFacts struct {
	// Image is a vision projector that loads with the model; Audio is that
	// projector's audio encoder (clip.has_audio_encoder).
	Image, Audio bool
	// Tools is whether the model's chat template formats tool calls; nil
	// when the file states no template.
	Tools *bool
	// Reasoning is the family's reasoning profile, for a model whose chat
	// template has a thinking section; nil otherwise.
	Reasoning *ReasoningProfile
}

// ReasoningProfile is how a thinking family reasons on llama.cpp: the
// reasoning budget in tokens of each effort word (low, medium, high) and the
// effort a request that names none gets (reserveAnswer).
type ReasoningProfile struct {
	Budgets map[string]int
	Default string
}

// factsRefresh is how old the cached facts may be before a read starts a
// fresh one in the background.
const factsRefresh = time.Minute

type factsState struct {
	mu      sync.Mutex
	facts   map[string]ModelFacts
	at      time.Time
	running bool
	closed  bool
	wg      sync.WaitGroup
}

// projector is object's vision projector in its store, when the source
// finds one.
func (h *Host) projector(object Object) (Object, bool) {
	p, ok := h.options.Source.(bridge.Projectors)
	if !ok || object.Binding.Kind != bridge.BindFile || object.Path == "" {
		return Object{}, false
	}
	return p.Projector(object)
}

// ChatModelFacts is every allowlisted chat model's name and the facts known
// for it now. It never waits for a store or file read: facts older than a
// minute are read again in the background, and a load that resolves its
// model reads that model's at once.
func (h *Host) ChatModelFacts() ([]string, map[string]ModelFacts) {
	h.facts.mu.Lock()
	defer h.facts.mu.Unlock()
	h.startRefreshLocked()
	return h.Models(), maps.Clone(h.facts.facts)
}

// RefreshFacts reads every allowlisted chat model's facts, waiting at most
// until ctx ends; a read still running then completes in the background.
func (h *Host) RefreshFacts(ctx context.Context) {
	h.facts.mu.Lock()
	done := h.startRefreshLocked()
	h.facts.mu.Unlock()
	if done == nil {
		return
	}
	select {
	case <-done:
	case <-ctx.Done():
	}
}

// startRefreshLocked starts a background read of every model's facts when
// none is running and the last began a minute ago or more. It answers the
// read's end, or nil when none started.
func (h *Host) startRefreshLocked() <-chan struct{} {
	if h.facts.running || h.facts.closed || (!h.facts.at.IsZero() && time.Since(h.facts.at) < factsRefresh) {
		return nil
	}
	h.facts.running = true
	h.facts.at = time.Now()
	done := make(chan struct{})
	h.facts.wg.Add(1)
	go func() {
		defer h.facts.wg.Done()
		defer close(done)
		read := map[string]ModelFacts{}
		for _, name := range h.names {
			model := h.allow[name]
			object, err := h.options.Source.Resolve(model.Store, model.Object, h.options.Engine.accepts())
			if err != nil || object.Path == "" {
				continue
			}
			projector, ok := h.projector(object)
			read[name] = h.factsFor(model, object, projector, ok)
		}
		h.facts.mu.Lock()
		h.facts.facts = read
		h.facts.running = false
		h.facts.mu.Unlock()
	}()
	return done
}

// noteResolved reads name's facts in the background once a load resolved
// its object and projector.
func (h *Host) noteResolved(name string, model Model, object Object, projector Object, hasProjector bool) {
	h.facts.mu.Lock()
	defer h.facts.mu.Unlock()
	if h.facts.closed || object.Path == "" {
		return
	}
	h.facts.wg.Add(1)
	go func() {
		defer h.facts.wg.Done()
		f := h.factsFor(model, object, projector, hasProjector)
		h.facts.mu.Lock()
		if h.facts.facts == nil {
			h.facts.facts = map[string]ModelFacts{}
		}
		h.facts.facts[name] = f
		h.facts.mu.Unlock()
	}()
}

// closeFacts stops new reads and waits for those running, which hold files
// open.
func (h *Host) closeFacts() {
	h.facts.mu.Lock()
	h.facts.closed = true
	h.facts.mu.Unlock()
	h.facts.wg.Wait()
}

// factsFor reads model's facts from its weights file, its projector and its
// family's engine profile.
func (h *Host) factsFor(model Model, object Object, projector Object, hasProjector bool) ModelFacts {
	var f ModelFacts
	template := ""
	if read, err := gguf.ReadFacts(object.Path); err == nil && read.ChatTemplate != "" {
		template = read.ChatTemplate
		tools := toolTemplate(template)
		f.Tools = &tools
	}
	if hasProjector && projector.Path != "" {
		held, err := fileid.Open(context.Background(), fileid.Self, projector.Path,
			fileid.Options{Links: fileid.RefuseLinks, Read: true})
		if err == nil {
			defer held.Close()
			if st, err := held.File().Stat(); err == nil && st.Mode().IsRegular() {
				read, err := gguf.ReadFactsFrom(held.File())
				// A projector that states no encoder flags is a vision projector,
				// the only kind llama.cpp loaded before audio encoders existed.
				f.Image = err != nil || read.HasVisionEncoder == nil || *read.HasVisionEncoder
				f.Audio = err == nil && read.HasAudioEncoder != nil && *read.HasAudioEncoder
			}
		}
	}
	family := "default"
	if entry, ok := h.curatedEntry(model, object); ok {
		family = entry.Family
	}
	reserve := profileCache(family, nil, 0).answerReserve
	if thinkingTemplate(template) && reserve.tokens > 0 && reserve.budgets[reserve.effort] > 0 {
		f.Reasoning = &ReasoningProfile{Budgets: maps.Clone(reserve.budgets), Default: reserve.effort}
	}
	return f
}

// toolTemplate reports whether a chat template formats tool definitions or
// tool calls.
func toolTemplate(template string) bool {
	return strings.Contains(template, "tools") || strings.Contains(template, "tool_call")
}

// thinkingTemplate reports whether a chat template has a thinking section
// llama.cpp switches with enable_thinking.
func thinkingTemplate(template string) bool {
	return strings.Contains(template, "enable_thinking") || strings.Contains(template, "<think>") || strings.Contains(template, "reasoning_content")
}
