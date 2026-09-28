# History: abstraction-provider-modelhost

Linked from [CONTRACT.md](CONTRACT.md)'s "Reading this page."

## Superseded ids

- **`HOST-1`..`HOST-4`** became `MODELHOST-L1`..`MODELHOST-L4` (loading).
- **`HOST-5`..`HOST-13`** became `MODELHOST-H1`..`MODELHOST-H9` (holding).
- **`HOST-14`..`HOST-17`** became `MODELHOST-R1`..`MODELHOST-R4` (rights).

  Per `research/vocabulary/RENAME-PLAN.md`'s modelhost step 1 and
  `research/vocabulary/DECISION.md` S3/S4. `HOST` is retired and never
  reused.

| retired | now | rule |
| --- | --- | --- |
| `HOST-1` | MODELHOST-L1 | object naming |
| `HOST-2` | MODELHOST-L2 | the allowlist is exhaustive |
| `HOST-3` | MODELHOST-L3 | the engine is a configured path |
| `HOST-4` | MODELHOST-L4 | the engine process is this provider's own |
| `HOST-5` | MODELHOST-H1 | a load takes a lease first |
| `HOST-6` | MODELHOST-H2 | each acquire outcome is one reason |
| `HOST-7` | MODELHOST-H3 | an unfinished load gives the lease back |
| `HOST-8` | MODELHOST-H4 | the lease is renewed while loaded |
| `HOST-9` | MODELHOST-H5 | every resident is its own engine |
| `HOST-10` | MODELHOST-H6 | a request in flight is a model in use |
| `HOST-11` | MODELHOST-H7 | a yield request unloads oldest first |
| `HOST-12` | MODELHOST-H8 | the idle time unloads the same way |
| `HOST-13` | MODELHOST-H9 | withdrawal ends the provider |
| `HOST-14` | MODELHOST-R1 | the program holds the resource |
| `HOST-15` | MODELHOST-R2 | withdrawing yield |
| `HOST-16` | MODELHOST-R3 | the peer check |
| `HOST-17` | MODELHOST-R4 | no onward delegation |

## Measured

- **`TestLiveHostTheSmallestGGUFUnderALease`** (`OA_LIVE_MODELHOST=1`),
  2026-09-22: hosted `unsloth/SmolLM2-135M-Instruct-GGUF` (88,201,792 bytes)
  with `llama-server --embeddings --pooling mean`; the first chat call
  loaded the model in 868 ms, the second reused it in 145 ms, and one
  `embed@1` call answered in 10.6 ms with 576 dimensions for 6 input
  tokens. See [README.md](README.md#measured) for the walkthrough.

## Design records

- `DREAM.md` §2 "One host", §10 — the design for one host holding one
  resident copy per model.
- `research/architecture-review-2026-09-22.md` F3, R3 — the review that set
  this layer's scope.
- `research/model-bridge/DECISION.md` §4 rules 4 and 6 — the decision that
  this provider bundles and downloads nothing.
- `research/vocabulary/DECISION.md` D2, D11, D88 — request, server, and the
  inference right's resource string. Applied in this release: D2 (request,
  MODELHOST-H6), D11/D88 (the `server:modelhost` prose word, S13 table entry
  pending rights' own dual reader).
- `research/vocabulary/DECISION.md` §3 (S1-S13) — the contract shape this
  page and its "Reading this page" section follow.

## Reviewed and applied

- `research/reviews/providers-2026-09-23.md` F2 — the abstraction did not
  define `insufficient`, `holders_refused`, `no_engine`. MODELHOST-H2 and
  MODELHOST-L3 now carry `unavailable` with reasons `lease:insufficient`,
  `lease:holders_refused`, `lease:not_permitted` and `engine:none`
  (`go/host/host.go`'s `Insufficient`, `HoldersRefused`, `NotPermitted` and
  `NoEngine` constants changed value to match; `go/host/host_test.go` and
  `go/service/host_test.go` reference the constants, not literal strings,
  so no test assertion needed a separate edit). `abstraction-inference`'s
  own `INF-S2` needs the matching amendment listing these reason families;
  that contract is not this worker's to change.
- `research/reviews/providers-2026-09-23.md` F4 — HOST-15 wrote a rule
  nothing decides. MODELHOST-R2 now states that no `yield` rule exists for
  this provider and that withdrawing the declaration is the person's actual
  switch.
- `research/reviews/providers-2026-09-23.md` F10 — HOST-2 cited REG-8, the
  non-blocking activation rule, for the router listing the allowlist.
  MODELHOST-L2 cites REG-2 (facade CONTRACT.md lines 84-90, 111) instead,
  noting it becomes `FAC-R2` once facade's own contract carries its id
  rename.
- `research/reviews/providers-2026-09-23.md` F12 — one seam, two words for
  an unknown id. MODELHOST-L1 now reads `object:unknown:<store>/<id>`,
  the same family the lend seam uses (`go/host/host.go`'s `unknownObject`
  helper; `go/host/host_test.go`'s "unknown object when no store holds the
  object" case).

## Reviewed and not changed here

- `research/reviews/providers-2026-09-23.md` F11 — `card:0` beside the
  platforms' device words. The word is `abstraction-resource`'s and
  MODELHOST-H1 already uses it consistently; the fix (the glossary's
  Resource entry naming the equivalents) belongs to the site glossary, not
  this contract.
