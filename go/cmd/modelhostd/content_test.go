package main

import (
	"bytes"
	"context"
	"strconv"
	"strings"
	"testing"

	modelwire "github.com/openabstractions/abstraction-model/go/abstraction/model/api"
	modelservice "github.com/openabstractions/abstraction-model/go/service"
	"github.com/openabstractions/abstraction-provider-modelhost/go/host"
)

func TestModelContentBindsOnlyReviewedCataloguePinToDeclaredObject(t *testing.T) {
	fit := modelservice.FitCatalog(context.Background(), modelwire.CatalogUseCode, modelservice.CatalogSources{
		Pool:   func(context.Context) (int64, error) { return 0, nil },
		Served: func(context.Context, string) ([]modelservice.ServedFamily, error) { return nil, nil },
	})
	entry := fit.Entries[0]
	pin := entry.Digest + ":" + strconv.FormatInt(entry.DownloadBytes, 10)
	models := []host.Model{{Name: "coding", Store: "oa-chat-spoof", Object: "result"}}
	if err := applyModelContents(models, nil); err != nil || models[0].Content != nil {
		t.Fatalf("store configuration inferred a pin: %+v %v", models, err)
	}
	if err := applyModelContents(models, []string{"oa-chat-spoof/result=" + pin}); err != nil || models[0].Content == nil || models[0].Content.Size != entry.DownloadBytes || "sha256:"+models[0].Content.SHA256 != entry.Digest {
		t.Fatalf("explicit pin: %+v %v", models, err)
	}
	for name, values := range map[string][]string{
		"digest":    {"oa-chat-spoof/result=sha256:" + strings.Repeat("a", 64) + ":" + strconv.FormatInt(entry.DownloadBytes, 10)},
		"size":      {"oa-chat-spoof/result=" + entry.Digest + ":1"},
		"missing":   {"oa-chat-spoof/missing=" + pin},
		"repeat":    {"oa-chat-spoof/result=" + pin, "oa-chat-spoof/result=" + pin},
		"malformed": {"oa-chat-spoof/result=unverified"},
	} {
		t.Run(name, func(t *testing.T) {
			args := []string{"serve", "--model", "coding=oa-chat-spoof/result"}
			for _, value := range values {
				args = append(args, "--model-content", value)
			}
			var out, errs bytes.Buffer
			if status := run(args, &out, &errs, func(string) string { return "" }); status != 2 || !strings.Contains(errs.String(), "--model-content") {
				t.Fatalf("status %d: %s", status, errs.String())
			}
		})
	}
}
