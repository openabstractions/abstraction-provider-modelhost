package host

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/url"
	"slices"
)

// titleMovedKey marks the context of a title request moved to another
// model's engine (titleElsewhere). A moved title is never moved again.
type titleMovedKey struct{}

// titleModel is the smallest allowlisted chat model, other than the one
// whose engine answers at host from, that serves a title now without
// evicting a resident model: one already loaded, or one the resource lease
// admits beside every resident model. EnsureSpeculative decides that through
// the no-yield acquire (resource RES-L6) and the model's own context plan;
// it never asks a holder to yield. Unknown sizes come last; ties go by
// name. ok is false when no model can serve.
func (h *Host) titleModel(ctx context.Context, from string) (string, bool) {
	type candidate struct {
		name string
		size int64
	}
	if h.options.Source == nil {
		return "", false
	}
	var candidates []candidate
	for _, name := range h.names {
		if h.hosts[name] == from {
			continue
		}
		model := h.allow[name]
		size := int64(math.MaxInt64)
		if object, err := h.options.Source.Resolve(model.Store, model.Object, h.options.Engine.accepts()); err == nil && object.Size > 0 {
			size = object.Size
		}
		candidates = append(candidates, candidate{name, size})
	}
	slices.SortFunc(candidates, func(a, b candidate) int { return cmp.Or(cmp.Compare(a.size, b.size), cmp.Compare(a.name, b.name)) })
	for _, c := range candidates {
		if _, err := h.EnsureSpeculative(ctx, c.name); err == nil {
			return c.name, true
		}
	}
	return "", false
}

// titleElsewhere is req, a title request whose own engine keeps its slots
// for conversations, addressed to the engine of titleModel with body's
// model named for it. ok is false for a request already moved and when no
// model can serve.
func (h *Host) titleElsewhere(req *http.Request, body []byte) (*http.Request, bool) {
	if moved, _ := req.Context().Value(titleMovedKey{}).(bool); moved || req.URL == nil {
		return nil, false
	}
	name, ok := h.titleModel(req.Context(), req.URL.Host)
	if !ok {
		return nil, false
	}
	base, err := url.Parse(h.Base(name))
	if err != nil || base.Host == "" {
		return nil, false
	}
	var fields map[string]any
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, false
	}
	fields["model"] = name
	raw, err := encodeJSON(fields)
	if err != nil {
		return nil, false
	}
	out := req.Clone(context.WithValue(req.Context(), titleMovedKey{}, true))
	out.URL.Host, out.Host = base.Host, base.Host
	out.Body = io.NopCloser(bytes.NewReader(raw))
	out.ContentLength = int64(len(raw))
	out.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(raw)), nil }
	return out, true
}
