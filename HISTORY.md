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

### The engine serves a socket, 2026-10-01

The owner asked that engines stop raising firewall prompts and use OA IPC
(VISION 2026-10-01, "Engines adopt model bytes…; engines OA launches move
off TCP ports to endpoints only OA can reach";
`research/model-bytes/ENGINE-TRANSPORT.md` stage 1, T1). The b10825
review's condition 7 asked for a loopback-only listener; a socket path in a
private directory with the server bound to the engine meets it with no
listener at all.

| id | before | now |
| --- | --- | --- |
| MODELHOST-L4 | started with the loopback address and a chosen port | started with the socket path the provider chose |
| MODELHOST-L5 | `--port` refused as an option the provider sets | `--port` refused; the provider passes none |
| MODELHOST-L7 | `--host 127.0.0.1 --port N`; the tree holds its listener on 127.0.0.1 and no socket beyond loopback; reason `engine:listener-not-loopback` | `--host <state>/sockets/<random>/e.sock` in a directory only this account and SYSTEM open, no port; every request binds the socket's server to the engine's tree (reason `engine:peer`); the tree holds no TCP listener or bound UDP socket on any address (reason `engine:network-socket`); the directory is removed after exit and stale ones are swept at start |
| MODELHOST-L8 | a local process reaching the loopback port directly is refused 401 | a local process of this account reaching the socket directly is refused 401 |

Review of the same day, with the coordinator's decisions: the socket
root is the kit's fixed per-user folder (`%LOCALAPPDATA%\oa\s`,
`$XDG_RUNTIME_DIR/oa` or `/tmp/oa-<uid>`, `$TMPDIR/oa`), independent of
`--state`, created private and refused for another owner or wider access;
the engine may hold no TCP socket in any state and no UDP socket,
outbound included, checked at readiness and every 30 seconds while it runs
(a later one stops the engine and reports `engine:network-socket`); each
launch directory holds a locked `held` file and the sweep keeps held ones.
The video kind's `sd-server` keeps the loopback rule through the same
check, and its readiness client binds to the engine's tree. Re-check of
the same day: video submission, polling and cancellation use that bound
client too, and the running check stops only the engine it checked.

`modelhostd serve --engine-port` is removed with the port. The engine's
base URL names its model's engine slot (`http://engineN.modelhost.invalid`)
and the engine client maps it to the running engine's socket.

### A model in memory, 2026-10-01

The inference provider kit fills a model into a read-only memory section
and lends it by inherited handle (VISION 2026-10-01; inference
INF-MS1..INF-MS6; facade FAC-P1). This provider maps a source binding to
the kit's extent and the filled section to a `memory` binding
(`go/host/memory.go`), proved end to end with an engine stand-in started by
supervision. `llama-server` at b10825 reads `-m FNAME` only, so `accepts`
stays `file`.

| id | before | now |
| --- | --- | --- |
| MODELHOST-L1 | `file` binding only | `file` binding only for every load; the provider holds the memory fill for an engine that reads an inherited section, and none at its pin does |

### Load is its own stage of progress, 2026-10-02

The first live run against the real engine (`research/live-runs/2026-10-02.md`,
phase 2 finding 5) showed `Observe` at 60 percent four to six seconds into a
cold job, before sampling began: the first frame's encode prints a one-step
'=' bar, which the provider read as all of sampling, and sampling then read
as the decode. The provider now follows the engine's own stage lines.

| id | before | now |
| --- | --- | --- |
| MODELHOST-V10 | every '=' series counts: the first is sampling (60), later ones the decode (40) | three stages from the engine's lines: load and preparation (10) until `generate_video`, sampling (60) until `sampling completed`, the decode (30); a bar before the opening line is preparation; never going back |

### The page states what the provider serves, 2026-10-02

The architecture review of 2026-10-02 (`research/architecture-review-2026-10-02.md`,
"Coherence" and D5) found the opening and "Not built" saying chat,
embeddings and one video model, image "not built" and an engine bundled
nowhere, while MODELHOST-T11 serves `image@1` and `modelhostd` carries the
PyTorch engine script. Branch `docs-match-tree` stated the scope as built;
no rule's behaviour changed. The owner's decision of the same day (VISION
"OA may supply an engine process") moves that engine into its own provider
module.

| id or section | before | now |
| --- | --- | --- |
| opening | chat, embeddings and one video model on the pinned llama.cpp release | a table of four services on three engine kinds, what ships (the optional `ModelHost` feature) and what is gated (the PyTorch kind), and the move to its own provider |
| MODELHOST-L3 | "This layer ships no engine, downloads none and looks for none." | "This layer ships no llama.cpp build, downloads none and looks for none." |
| MODELHOST-R1 | the runtime writes `hold` on `awake` beside `card:0` | the same, named as 0.4.0's provisional exception that the requester-held wake lease replaces after research (VISION 2026-10-02, D2) |
| Not built | "bundles neither"; images belong with the `image@1` providers; FLUX.2 [klein] `image@1` "is not built" | bundles no llama.cpp, stable-diffusion.cpp or PyTorch build; the PyTorch script is the one engine it carries and moves to its own provider; images on the PyTorch engine alone, sd.cpp images being the Lemonade provider's |
### FLUX.2 [klein]'s lease covers its measured peak, 2026-10-02

The live walkthrough of 2026-10-02
(`research/live-runs/2026-10-02-walkthrough.md`, scene 2, D9) reserved
18,828,230,656 bytes for one 512x512 image against the 18,790,481,920-byte
lease, and the provider reported the excess. The lease is now that peak with
about 5 percent above it, as Hummingbird-XT's was sized from its diagnostic
rerun.

| id | before | now |
| --- | --- | --- |
| MODELHOST-T5 | `flux2-klein-4b` leases 17.5 GiB (phase 1 of 2026-10-02) | `flux2-klein-4b` leases 18.5 GiB (the walkthrough's 18.83 GB peak with about 5 percent above it) |

### Progress waits for the device, 2026-10-02

The live walkthrough of 2026-10-02 (D10) saw FLUX.2 [klein]'s four step
reports arrive within 0.1 s after the steps had run, and a video's F32
decode at 99 percent for 15 s. The engine's callbacks ran when Python had
queued a step's kernels, before the GPU ran them. The engine script now
waits for the device before each step and decode report (`settled`), so
the steps' share and the decode's share fill as the work finishes.

| id | before | now |
| --- | --- | --- |
| MODELHOST-T6 | progress per denoising step and decode chunk | the same, reported once the device has finished each |
| MODELHOST-T11 | progress per step and for the decode, steps 85 of 100 | the same, reported once the device has finished each; the decode fills the rest |

### The chat prefix preload, 2026-10-03

`research/live-runs/2026-10-02-demo-model-opencode.md` measured the Local
assistant's cold first answer at 10.6-12 s, and 3.6 s with its fixed prefix
evaluated ahead. A first design, never reviewed as applied, learned the
last session start's prefix from chat requests and evaluated it at the next
load; its review found that a prefix learned from one application and
evaluated for another is readable token by token through `cached_tokens`,
and that it had no caller. The coordinator's decision of the same review:
the application supplies its own prefix through inference `preload@1`
(INF-PL1..PL4), and this provider keeps nothing.

| id | before | now |
| --- | --- | --- |
| MODELHOST-W1 | none | binding: the prefix is the `Preload` call's; nothing kept or learned; per engine only whether it was used and the preload in flight |
| MODELHOST-W2 | none | binding: the lowest standing (the 2026-10-03 decision under VISION 2026-09-06's open arbitration): the load through the holder's speculative ensure (resource RES-L6), else `busy` `resources:held`; the load and the evaluation no model use; then `busy` on a used engine or a running preload, else evaluate through `/apply-template`, `/tokenize` and `/completion`; `loaded`, `refused` and `engine:stopped` |
| MODELHOST-W3 | none | binding: nothing of a preload runs after another request: before the evaluation is posted a request overtakes it; after, the continuing chat request waits for it and every other request cancels it and waits for its engine request to return; the wait bound of one second, one logical batch and one checkpoint |

The re-review of the same day (N1) found that the continuing request could
reach the engine before the evaluation it continued, which then ran after
it; the posted phase and the waits of MODELHOST-W3 close that. The third
round found every request that cancelled a posted preload waiting 3 s for
an engine exit (R1), and three windows of a check-then-act emulation of the
speculative standing (R2-R4); the standing moved into abstraction-resource
as RES-L6, and `EnsureSpeculative` calls the holder's.

Tests: `go/host/preload_test.go` (the OpenCode 1.18.31 Local prefix token
for token against `testdata/opencode-1.18.31-local.json`, another caller's
text, a turn before a preload, the continuing request before and after the
evaluation is posted, every other request during one, an embedding during
one, 200 raced rounds of a preload and a request, an engine stopped during
one, an unload and a yield during a held evaluation, a load against a held
card and after an eviction, the idle clock at Idle less a second, a yield
during the load, a refused prefix),
`go/service/preload_test.go` (the endpoint end to end), and the gated
measurement `TestLivePreloadFirstAnswer` (`go/host/preload_live_test.go`),
which the stand-in engine runs in `TestPreloadMeasurementRunsOnTheStandInEngine`.
Not run on the GPU. Review pending.

### The demo model's MTP draft head, 2026-10-03

`research/live-runs/2026-10-02-demo-model-opencode.md` measured
Qwen3.6-35B-A3B at about 60 tok/s with `--spec-type draft-mtp
--spec-draft-n-max 3` and about 48 tok/s without.
`feedback/2026-10-02-prefix-warm.md` found both flags refused. The reviewed
list admits them with bounded values. Draft-model and lookup-cache options
stay refused.

| id | before | now |
| --- | --- | --- |
| MODELHOST-L5 | no speculative decoding option | `--spec-type draft-mtp` and `--spec-draft-n-max` 1..8; options naming a draft model or a cache file refused |

### The engine's placement follows its lease, 2026-10-06

The rank-one benchmark of 2026-10-03 ran the demo model through modelhost on
llama.cpp b10825's own placement (`--fit on`, automatic layers, a mapped
file) at 16 tok/s; the same family ran at about 48 tok/s with
`--n-gpu-layers 999 --no-mmap` on 2026-10-02
(`research/engines/STATUS.md`). Nothing in the product path set either. The
card lease already admits the file's bytes, so the default placement now
follows it. `--load-mode` and `--mmap` joined the reviewed list so a person
can replace the load default; both choose how the engine reads the file this
provider named.

| id | before | now |
| --- | --- | --- |
| MODELHOST-L5 | `--no-mmap` and `--mlock` the only load options | `--load-mode` (`auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`) and `--mmap` added |
| MODELHOST-L9 | (none; the engine's own placement) | `card:<n>` lease: `--no-mmap`, layers left to the engine's `--fit`; `memory` lease: `--n-gpu-layers 0`; a person's own option replaces the matching default |

The first draft also set `--n-gpu-layers all` under a card lease. A rerun
with `--no-mmap` and the layers left to `--fit` decoded at 41.7 tok/s
against 44.5 with `all` and 16.1 with neither
(`research/live-runs/2026-10-06-gpu-offload.md`): the mapped file caused the
10-03 slowness. The layer count stays with `--fit`, so a discrete card
smaller than the model keeps a partial offload. Review of 2026-10-06 also
narrowed the rule to what the lease admits, the file's bytes, and named the
start failure when the file, KV cache and compute buffers exceed the card.

### A start that does not fit is named, 2026-10-06

| id | before | now |
| --- | --- | --- |
| MODELHOST-L9 | an engine that ran out of device memory at start read `unavailable` with no reason | an allocation-failure line before a failed readiness reads `engine:does_not_fit` with the remedy |

The engine's output is read for b10825's allocation-failure lines; the
reason applies only when readiness then fails, so a warning the engine
survives changes nothing. Tested with the stand-in engine's
`--fake-fail-line` (`TestAnEngineThatDoesNotFitIsNamed`); no GPU run.

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
- 2026-10-01 GPU faults (MODELHOST-G1). Three driver resets on the owner's
  Radeon 8060S that day (abstraction-resource HISTORY.md, same date) showed
  that an engine ended by a reset read `engine:lost`, the other engines on
  the adapter stayed listed, and an identical video could start again at
  once. The provider now composes the resource kit's `gpufault` Guard;
  `--diffusion-fa` became the sd.cpp dispatch row after live run 3b
  (`research/live-runs/2026-10-01.md`) needed 87 GB without it. The stand-in
  sd-server gained `--fake-device-lost` to print ggml-vulkan's
  device-lost line before it exits.

## Reviewed and applied

- Engine exits are recorded (coordinator, 2026-10-02; the torch-live run in
  `research/live-runs/2026-10-02.md`, feedback/gpu-runs-2026-10-02.md item
  17). The first cold PyTorch job failed `native:engine:lost` with nothing
  in `torch.log` after `loaded`, no reason from modelhostd and no crash
  event. New rule MODELHOST-E1: every engine kind's exit is one OA logging
  record and one stderr line with its code or exception status, run time,
  job memory peaks and limit (from facade FAC-P2), and the line closes the
  engine's log; the PyTorch engine script enables `faulthandler` before
  torch is imported and logs the job's start and phases. The supervised
  child keeps `modelhostd`'s inherited error mode: measured, a Python child
  started through `supervision` reads `GetErrorMode()` 0x8003
  (`SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX`,
  which Go's runtime sets in `preventErrorDialogs`), and two access
  violations of the stand-in engine left no Application 1000 event. The
  alternative, `CREATE_DEFAULT_ERROR_MODE`, would also clear
  `SEM_FAILCRITICALERRORS` and let a missing-DLL loader dialog hold an
  unattended engine; Go's `WER_FAULT_REPORTING_NO_UI` is per process and
  cannot be set for a foreign child. Tests: `engineexit_test.go`, the
  chat idle stop (`TestIdleUnloadsAndReleasesTheLease`), the sd-server
  crash (`TestVideoEngineCrashMidGenerationIsLost`), the PyTorch engine's
  exit code and native fault (`TestTorchVideoEngineCrashIsLost`,
  `TestTorchVideoEngineFaultIsRecorded`). With the FLUX.2 [klein] image set
  merged in, the job's start line and the phase hook moved to the engine's
  `Session` (load stages, unload, generate) and `TorchRuntime.phased`, so
  both sets log their phases; the image set logs its text encode, each
  step's end and its decode (`TestTorchImageMakesAPNGThroughTheEngine`).
- `lease:ended` (coordinator, 2026-10-02; abstraction-resource HISTORY.md
  "renewal never waits for a load"). The resource kit's holder now reloads
  a component once when the service ends its lease during readiness, and
  returns an acquire failure wrapping `holder.ErrLeaseEnded` on a second
  ending. `mapFailure` read that failure, which carries no acquire outcome,
  as a lease service that did not answer. MODELHOST-H2 gains the reason
  `lease:ended` (`go/host/host.go`'s `LeaseEnded`; `TestRefusals`' "lease
  ended" case).
- The PyTorch-on-ROCm engine kind (coordinator, 2026-10-02;
  `research/live-runs/2026-10-01-rocm.md` "What OA needs to offer this as a
  provider"; stack review `pytorch-rocm-windows-2.11.0-rocm7.13.0.md`
  conditions 2 to 4a; file review `hummingbird-xt-i2v-files.md`). New rules
  MODELHOST-T1..T9: the research harness became the provider's own engine
  script (`go/host/torch/oa_torch_engine.py`) beside `llama-server` and
  `sd-server`, reached over its stdin and stdout with a framed protocol
  (`go/host/torchproto.go`), launched by the provider kit's launcher with
  the harness's allowlisted environment and the resource kit's
  `pytorch-rocm` dispatch row (`go/host/torchengine.go`), under the measured
  29.3 GiB device lease and a 43 GiB per-process limit, with progress per
  step and decode chunk, a cancel between steps, and the provider's own
  Motion-JPEG AVI (`go/host/torchvideo.go`, `avi.go`). Its model set is the
  `TorchModelSets` row `hummingbird-xt-i2v` (`go/host/torchset.go`): the
  AICreator i2v transformer and TurboVAED, Wan-AI's Diffusers Wan 2.2 VAE
  and umT5 shards, the umT5 tokenizer, and AMD's eight source modules at
  `929e90a2`, all pinned by SHA-256 and bound where they lie. The BF16
  decode is the option `decode-dtype=bf16`, opt-in. The video kind became an
  interface over the two engines (`videoKind` in `go/host/video.go`); the
  sd-server path is unchanged. `providerhost/devicelost` gained the
  engine's form `PyTorchEngine`. The engine does not call
  `SetThreadExecutionState`: the wake lease and the standby refusal are the
  holder's (VISION 2026-10-01, GPU faults). Tests: `go/host/torchvideo_test.go`
  against `go/host/testdata/faketorch/fake_engine.py`, a stand-in that runs
  the real script's protocol, containment and failure handling in a venv
  without torch, and `TestTorchVideoCommandsRefuseBeforeAnythingStarts`
  (`go/cmd/modelhostd/video_test.go`). Not run on the GPU in this change.
- Review of `779d7f0c2` (coordinator, 2026-10-02), revise. MODELHOST-T1:
  the interpreter runs with `-I -S -B` and the engine appends the venv's
  site-packages after its network hook, so no `.pth` file runs; the base
  interpreter `pyvenv.cfg` names is pinned (`python.exe`, `python312.dll`,
  `python3.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`) and held; the
  standard library and site-packages stay an accepted risk with a TODO row
  for a RECORD-based launch check. MODELHOST-T4: the dispatch row's naive
  solvers, and the provider kit's shared GPU library path gate (inference
  INF-EL3) before the lease and at every launch, reason `engine:gpu-path`.
  MODELHOST-T6: the sd path's prompt bound (`sdcpp.CheckPrompt`). Tests:
  `TestTorchVideoPromptWithALineBreakIsRefused`,
  `TestTorchProtocolRefusesAnUnterminatedLongHeader`,
  `TestTorchGPUPathGateRunsBeforeTheLease`, `TestTorchBaseInterpreterIsProved`,
  and the `-S` and site-packages checks in
  `TestTorchVideoGeneratesThroughTheEngine`.
- Second review of the PyTorch engine kind and the OA-models folder
  protection (coordinator, 2026-10-02). MODELHOST-V2 and T2: a PyTorch
  set's places record makes its folder under the models root a recorded
  set (`RecordedSetFolders` reads `TorchModelPlaces`), and `models install
  --kind torch` adopts it as a model folder as the sd-server sets do
  (`TestTorchSetFolderIsProtectedAndTrusted`). MODELHOST-T1: the base
  interpreter's `DLLs\_socket.pyd`, which the network audit hook depends
  on, is pinned and held; the other native modules of `DLLs` join the
  accepted standard-library risk.
- Delta review of the PyTorch engine kind (coordinator, 2026-10-02).
  MODELHOST-T1: `DLLs\select.pyd` is pinned beside `_socket.pyd`, because
  `socket` imports `selectors`, which imports `select`, before the network
  audit hook runs (`TestHummingbirdTorchRowIsValid`; the stand-in
  runtime of every engine test pins and holds it too). MODELHOST-V2:
  install-time adoption is tested for both kinds: `models install --kind
  sdcpp` and `--kind torch` with a `--from` folder directly under
  `--models-root` leave that folder a model folder and a sibling folder's
  owner and access list unchanged (`TestModelsInstallAdoptsTheSetFolder`;
  removing either `adoptSetFolders` call fails its subtest).
- FLUX.2 [klein] on the PyTorch engine (coordinator, 2026-10-02; spec from
  `research/live-runs/2026-10-01-rocm.md` and `2026-10-02.md`). New rules
  MODELHOST-T10 and T11; T1..T3 and T5 extended. The `TorchModelSets` row
  `flux2-klein-4b` (`go/host/torchset.go`) is an image set: Comfy-Org's
  single files in `OA-models\flux2-klein\` and `zimage\qwen_3_4b.safetensors`,
  BFL's Qwen3 tokenizer, a 17.5 GiB lease and a 25 GiB commit limit, and its
  figures beside `hummingbird-xt-i2v`. The engine script serves both sets
  (protocol 2: `load`, `unload` and `generate` name a set; `unloaded`
  answers an unload); the FLUX set builds the transformer through
  Diffusers' FLUX.2 key conversion and the VAE through
  `go/host/torch/flux2-vae-keys.json`, derived by pairing the Comfy-Org VAE,
  rounded to BF16, with AICreator's diffusers VAE tensor by tensor. The two
  sets share one process as two holder components (`go/host/torchshared.go`):
  the first set's lease is the process's, the joined set's is the measured
  lease together less the first's, since `leases@1` has no resize; a join
  that fails leaves the first set serving. `image@1` is the inference
  provider's composed image engine (`go/host/torchimage.go`), its PNG
  written by the provider kit's new `providerhost/rgbpng` (inference
  INF-IO1) and published through `DesignatedContent.OutputWriter`, which
  now owns the output publication `providerhost.NewEngine` had inline.
  `modelhostd serve --image-model`, `engine check --kind torch --set`.
  The coordinator's 40.5 GiB for both sets is run E of 2026-10-01; phase 1
  of 2026-10-02 reserved 40.9 GiB with the BF16 decode and 44.4 GiB with
  F32, and the row holds those, the larger measured peaks. Tests:
  `go/host/torchimage_test.go` against the stand-in engine,
  `TestTheServedConfigurationCarriesTheImageEngine`
  (`go/service/image_test.go`), the refusals in
  `TestTorchVideoCommandsRefuseBeforeAnythingStarts`, and
  `serve/provider_video_test.go`'s `TestTorchImageDeclarationRoutesFluxToTheProvider`.
  Not run on the GPU in this change.
- Review of `f6d3e3fbd` (coordinator, 2026-10-02). MODELHOST-T10: a joined
  set could be idled or yielded while the other set's job ran, because a
  set read busy only for its own work; its unload then queued behind the
  job in the engine, and past 20 seconds the whole process was stopped and
  the video failed `engine:lost`. Every engine operation (a load, an unload,
  a job) now holds the process's one operation slot, an unload waits for
  it, and every set reads busy while an operation holds it
  (`TestTorchJoinedSetIsNotIdledUnderARunningJob`,
  `TestTorchUnloadWaitsForTheRunningJob`; removing either half fails its
  test). `TestTorchBusyAnswersDuringAStart` holds a start inside its lock
  and reads busy.
- The PyTorch video's container (coordinator, 2026-10-02, once `videoenc`
  reached main). MODELHOST-T7 was "a Motion-JPEG AVI (`video/avi`) the
  provider writes"; it is now an H.264 MP4 (`video/mp4`) through
  `providerhost/videoenc`, hardware encoder first and software as the
  fallback, the completed `Observe` naming the encoder. No caller asked
  for AVI, so `go/host/avi.go` is gone. The tests take the software
  encoder (`TestTorchVideoGeneratesThroughTheEngine` reads the MP4's
  boxes and sample count).
- The video set's device lease (coordinator, 2026-10-02, from the
  diagnostic rerun, `feedback/gpu-runs-2026-10-02.md` items 22 and 23).
  MODELHOST-T5 was 29.3 GiB (31,460,635,443 bytes, run c'' of 2026-10-01);
  the rerun reserved 31,509,708,800 bytes at 512x512x81 with the F32
  decode, and modelhostd warned. It is now 31.0 GiB, the measured peak with
  about 5 percent above it. The 43 GiB process limit stays: the rerun's
  32.78 GiB peak did not reach it. The joined figures follow from the
  together peaks (MODELHOST-T10): the image set joining the video set holds
  9.9 or 13.4 GiB, the video set joining the image set 23.4 or 26.9 GiB.
- Delta review of `46682621b` (coordinator, 2026-10-02). MODELHOST-T10: the
  holder's idle and yield take a set's component lock, read busy and call
  Stop, which waited for the operation slot, while a job took the slot and
  then the component lock: a deadlock. The lock order is now the component
  lock, then the slot: a job takes its set's use and loads it before it
  holds the slot (`torchShared.acquire`), and a load takes the slot inside
  its Start and gives it up at readiness. A driver reset stops the process
  directly (`torchLoser`, attached to the Guard) and takes no slot. A join
  cancelled while the engine still loads the set unloads it within its own
  operation before its lease goes. MODELHOST-T7: an encode failure other
  than refused options or no system encoder reads `encoder:failed`.
  Tests: `TestTorchUnloadAndJobKeepTheLockOrder`,
  `TestTorchResetStopsTheEngineUnderARunningJob`,
  `TestTorchCancelledJoinUnloadsBeforeTheLeaseGoes` and
  `TestEncodeFailureReasons`; removing the reset's direct stop or the
  cancelled join's unload fails its test.

- VISION 2026-10-01, OA never copies a model for an engine (coordinator,
  2026-10-01). MODELHOST-V2 was "The model folder holds exactly the pinned
  set": `models install` copied the 11.5 GB set into a read-only folder of
  its own and `--video-models` named it. V2 now binds each pinned file where
  it lies, found by digest in `--from` folders and the inventory's stores,
  proved and held there (inference CONTRACT.md INF-F7, `enginefiles`
  `LaunchPlaced`); `--video-models` and `models unlock` are gone. Model sets
  became data (`VideoModelSets`), and V6 holds a request to its set's sizes,
  frames, rate and first-frame use, with the new reason `frame:required`.
  Tests: `TestPlaceVideoModelsBindsInPlace`,
  `TestVideoModelFilesAreHeldWhileTheEngineRuns`,
  `TestVideoRequestsAreHeldToTheModelSet` (`go/host/video_test.go`) and
  `TestVideoModelsInstallRefusesUnpinnedFiles`
  (`go/cmd/modelhostd/video_test.go`).
- `research/security-reviews/hummingbird-xt-i2v-files.md` conditions 1 to 8
  and video addendum amendment 1 (coordinator, 2026-10-01). The
  `hummingbird-xt-i2v` row of `VideoModelSets`: the AICreator i2v file by
  SHA-256 and locator, OA's umt5 Q8_0 and `taew2_2`, `--diffusion-fa`,
  AMD's sigma table as `custom_sigmas` with `lcm` (sdcpp `VideoPreset.Sigmas`,
  INF-SD6), 512×512, 81 frames, 16 fps, a required first frame, and the AMD
  Responsible AI Use Policy as the row's `UsePolicy`. Tests:
  `TestHummingbirdRowRunsAsData` (`go/host/video_test.go`) and sdcpp's
  `TestVideoPresetSigmaTable`.
- Review of da8e3b548 items 3 to 6 (coordinator, 2026-10-01). Item 3:
  sd-server is given each held file's final path, every folder on the path
  held and no reparse point on it (`TestVideoModelPathsCannotBeRedirectedByAJunction`).
  Items 4 and 5: the places record keeps the owners each file may have, and
  the metadata shortcut is taken only for a file identity's
  `trustedpath.CheckFile` passes (`TestPlacedModelOwnersReachTheTrustCheck`;
  enginefiles' `TestPlacedTrustDecidesTheShortcut` and
  `TestPlacedTrustedFileIsReadOnce`). Item 6: amendment 1 states the path
  binding and the trust condition and overrides the file review's
  conditions 4 and 5. The `hummingbird-xt-i2v` row is in `VideoModelSets`.
- Coordinator follow-ups of 2026-10-02. The places record's owners admit the
  machine's and this account alone (`parseOwners`; regression in
  `TestPlacedModelOwnersReachTheTrustCheck`). OA's set folders under
  `OA-models` are made model folders (identity ID-D3) by `models install`
  and, once, by `models protect` (`TestAdoptModelFolderTouchesOnlyOAsSetFolders`,
  `TestModelsProtectReprotectsSetFolders`): files moved in from `%TEMP%` had
  kept a sandbox group's entries and were hashed at every launch.
- Review of 6607b567d (2026-10-02): protection reached outside OA's set
  folders through a hard link. Now only recorded sets are protected
  (`RecordedSetFolders`), after placing in `models install`; the models root
  must pass `privatedir.CheckModelsRoot`; a file with a name outside the
  models root and the job store refuses its folder; `models protect` gained
  `--state`, `--job-store` and `--dry-run` (`TestRecordedSetFoldersAreTheOnlySetFolders`,
  `TestModelsProtectReprotectsSetFolders`).
- `research/security-reviews/stable-diffusion.cpp-master-843-462d675-video-addendum.md`
  conditions 1 to 13 (2026-10-01). New rules MODELHOST-V1..V9: the pinned
  `sd-server` folder and model set staged and proved through `enginefiles`
  (`go/host/videopin.go`), the command line and environment through
  `providerhost/sdcpp` (`go/host/videoengine.go`), the `video@1` operations
  with their preset, bounds, request ids, first frames, results, cancels and
  time limit (`go/host/video.go`), and the endpoint composition with the
  provider's content store (`go/cmd/modelhostd`). Condition 11's job memory
  limit is a hook (`--video-memory-limit`) with no value until measured;
  condition 13's exposure is stated in README.md "Video"; condition 14 is
  the re-review trigger. Tests: `go/host/video_test.go`,
  `video_windows_test.go`, `go/service/video_test.go` and
  `go/cmd/modelhostd/video_test.go`, against `go/host/testdata/fakesdserver`.

- `research/security-reviews/llama.cpp-b10825-win-vulkan-x64.md`
  conditions 2, 3, 6 and 7 (2026-10-01). New rules MODELHOST-L5 (the
  reviewed engine argument list, `go/host/engineargs.go`), MODELHOST-L6
  (the engine environment built from nothing, `go/host/engineenv*.go`),
  MODELHOST-L7 (the loopback listener confirmed after readiness through
  abstraction-identity's owner tables, `go/host/listener.go`, reason
  `engine:listener-not-loopback`) and MODELHOST-L8 (no media the engine
  would fetch or decode, `go/host/media.go` and preflight). Tests:
  `engineargs_test.go`, `engineenv_test.go`, `listener_windows_test.go`,
  `media_test.go`, `service/host_test.go` and
  `cmd/modelhostd/main_test.go`. Condition 4 (pinned GGUF digests) stays
  open. After review: L6 gains the profile folders and processor facts
  from the token, profile registry and GetNativeSystemInfo; L8 gains the
  per-launch `LLAMA_API_KEY`, `--no-webui` and data-URL byte sniffing
  (`enginekey_test.go`); L7 names UDP and runs once per load; `openabstractions
  models host` carries `--state` and `--engine-arg` to `engine check` and the
  declaration.
- `feedback/adopter-pinning-audit-2026-10-01.md` item 14 — `llama-server`
  had no version or digest. MODELHOST-L3 now pins llama.cpp b10825 Windows
  x64 Vulkan (`go/host/pin.go`; review
  `research/security-reviews/llama.cpp-b10825-win-vulkan-x64.md`), proves the
  engine folder before each start and holds it while the engine runs
  (`go/host/pin_test.go`, `engine_windows_test.go`).
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

## 2026-10-02: the PyTorch engine moves to its own provider

The owner decided that OA may supply an engine process when no engine on
the machine serves a model, as its own provider module (VISION 2026-10-02).
The PyTorch kind left with its history for `abstraction-provider-pytorch`,
and the parts it shared with the sd-server and llama.cpp kinds moved first
into `abstraction-provider-kit`: the engine process launch and its exit
record, the video@1 table, the placed model files and the refusals. The
rules MODELHOST-T1..T11 are retired here and live on as PYTORCH-T1..T11;
MODELHOST-E1 is the kit's KIT-P3, which this provider still cites for its
engines. `modelhostd` refuses the PyTorch kind's options with the new
provider's name, and `openabstractions models host` declares that provider.

