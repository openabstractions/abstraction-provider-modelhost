package host

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	wire "github.com/openabstractions/abstraction-inference/go/abstraction/inference/api"
)

func TestRequestUseSettingsWithAndWithoutResidentSlots(t *testing.T) {
	for _, resident := range []bool{false, true} {
		for _, row := range []bool{false, true} {
			for _, title := range []bool{false, true} {
				process := &Process{}
				if resident {
					process.cacheSlots = newCacheSlots(2)
				}
				if row {
					process.requestSettings = requestSettingsByUse{"title": {ChatTemplateKwargs: map[string]any{"enable_thinking": false, "preserve_thinking": false}}}
				}
				h := &Host{engines: map[string]*Process{"engine": process}}
				raw := []byte(`{ "messages":[], "id_slot":7, "cache_prompt":false, "chat_template_kwargs":{"unchanged":true} }`)
				uses := []wire.ChatUse(nil)
				if title {
					uses = []wire.ChatUse{wire.ChatUseTitle}
				}
				out, release, err := h.prepareScopedCacheRequest(context.Background(), "engine", "/v1/chat/completions", raw, "scope", "partition", uses...)
				if err != nil {
					t.Fatal(err)
				}
				if release != nil {
					release(true)
				}
				if !resident && !(row && title) && !bytes.Equal(out, raw) {
					t.Fatal("absent use or row changed the ordinary request bytes")
				}
				var body map[string]any
				if err := json.Unmarshal(out, &body); err != nil {
					t.Fatal(err)
				}
				kwargs := body["chat_template_kwargs"].(map[string]any)
				value, changed := kwargs["enable_thinking"]
				if changed != (row && title) || changed && value != false || kwargs["unchanged"] != true {
					t.Fatal("request-use row selection or unrelated setting changed", kwargs)
				}
				if resident || row && title {
					if kwargs["preserve_thinking"] != true {
						t.Fatal("profile row overrode the provider's replay setting")
					}
				}
				if resident {
					// A title takes the highest empty slot; a conversation the lowest.
					want := float64(0)
					if title {
						want = 1
					}
					if body["id_slot"] != want || body["cache_prompt"] != true {
						t.Fatal("request-use row changed the admitted slot policy")
					}
				} else if body["id_slot"] != float64(7) || body["cache_prompt"] != false {
					t.Fatal("request-use row changed unowned slot settings")
				}
			}
		}
	}
}
