# abstraction-provider-modelhost

*For operators and provider authors. Applications reach it through the OA
runtime.*

abstraction-provider-modelhost is the model host: it starts an engine
process for each model on this machine and keeps the model loaded while
calls keep arriving.

**What's new here:** Every application that asks for a model shares one loaded
copy, held under a lease that other programs asking OA for the card can ask
back, and a model nothing has used for its idle time unloads on its own.

It serves chat and embeddings on the pinned `llama.cpp` release and video on
the pinned stable-diffusion.cpp `sd-server` (CONTRACT.md's opening table).
The OA PyTorch engine it carried until 2026-10-02 is its own provider,
`abstraction-provider-pytorch` (VISION 2026-10-02, "OA may supply an engine
process"). Installing the model host starts no model; the operator selects
an engine and model afterward. OA's inference service also connects
directly to registered Ollama, LM Studio and Lemonade servers.

For chat it loads a model file into the pinned `llama.cpp` release staged on
this machine, one engine process per model, and keeps a loaded model
**resident**, in memory and answering calls, for as long as calls keep
arriving.

Holding a resident model's memory takes a **lease**, a time-bound claim from
[`abstraction-resource`](https://github.com/openabstractions/abstraction-resource/blob/main/CONTRACT.md)
that this provider renews while the model stays loaded. When the resource is
asked for elsewhere, this provider must **yield**: give a resident and its
lease back. It also yields on its own, for a resident nothing has used in
its declared idle time.

It serves
[`abstraction.inference/chat@1`](https://github.com/openabstractions/abstraction-inference/blob/main/CONTRACT.md)
on its own OA endpoint like any other native provider, so an application
never has to know whether the model it asked for was already loaded or had
to start. Every application the runtime forwards shares whatever is
resident: asking for a second model beside a resident one does not displace
it, each gets its own engine, and giving memory back yields the resident
nothing has touched longest first.

## Start

Install the runtime first: https://openabstractions.org/adopt.html

```
modelhostd serve --endpoint NAME --engine PATH --model NAME=STORE/OBJECT
                 [--model NAME=STORE/OBJECT]... [--resource-endpoint PATH]
                 [--resource NAME] [--engine-arg ARG]... [--embeddings]
                 [--idle DURATION] [--wait DURATION]
                 [--ready DURATION] [--chat-memory-limit BYTES]
                 [--runtime-program PATH]
                 [--config FILE] [--home DIR] [--state DIR] [--verbose]
modelhostd engine install --archive ZIP [--dir DIR] [--state DIR]
modelhostd engine check --engine PATH [--engine-arg ARG]... [--state DIR]
modelhostd engine unlock [--dir DIR] [--state DIR]
modelhostd help
```

On Windows, `--chat-memory-limit` sets an optional per-process job limit in
bytes for each llama-server. The service leaves it unset at zero. A model's
resource lease reserves its file size for admission and does not bound its
runtime memory use. The live Qwen3.6 measurement used 48 GiB; that value is
specific to that test.

The model host launches one llama.cpp release, pinned in
[`go/host/pin.go`](go/host/pin.go) (VISION 2026-10-01). On Windows x64 it is
b10825 Vulkan: `llama-b10825-bin-win-vulkan-x64.zip`, 35,227,898 bytes,
SHA-256 `8907e5f3b74cb57d93fd94e187a87d85be24dc64e3b0ff2bc1b287da17b31446`,
published at
`https://github.com/ggml-org/llama.cpp/releases/download/b10825/`. Download
that archive, then run `modelhostd engine install --archive <zip>`. The
install checks the archive's size and SHA-256, writes the 24 pinned files and
none of the archive's command-line tools or RPC server, makes the folder
read-only for this account and prints the `llama-server.exe` path to pass as
`--engine`. Before every start the folder is hashed against the pin and held
unchanged while the engine runs (inference CONTRACT.md INF-F1..INF-F4). Any
other `llama-server`, an added or changed file, and every platform without a
reviewed release are refused `engine:none`. `engine check` runs the same proof
on demand; `openabstractions models host --engine` calls it before it writes
the declaration. `engine unlock` gives a folder this provider recorded back to
the account for removal, and refuses any other folder. Without `--dir` it
unlocks the folder the engine record names and does nothing when no record
exists; the Windows uninstaller runs it that way before it removes
`modelhostd`.

The b10825 review approved the build with conditions, and the provider
enforces conditions 2, 3, 6 and 7 around every start:

- **Arguments** (CONTRACT.md MODELHOST-L5): `--engine-arg` passes only the
  reviewed llama-server options that size, place or schedule the model and
  switches that turn a feature off, listed in
  [`go/host/engineargs.go`](go/host/engineargs.go). `--rpc`, `--lora`,
  `--host`, `--path`, `-hf`, `--model-url`, `--tools` and every other option
  are refused by name; `serve` and `engine check` exit 2.
- **Environment** (MODELHOST-L6): the engine starts with `SystemRoot`,
  `windir`, `SystemDrive`, `PATH` of System32 alone, `TEMP`/`TMP` at
  `engine-temp` in the provider's state, this account's `USERPROFILE`,
  `LOCALAPPDATA`, `APPDATA`, `ProgramData`/`ALLUSERSPROFILE` from the
  token and the profile registry, the processor count and architecture,
  and its own `LLAMA_API_KEY`. Nothing is inherited, so
  `GGML_BACKEND_PATH`, `LLAMA_ARG_*`, the Vulkan loader's `VK_*`,
  `OMP_*` and proxy variables never reach it.
- **Socket** (MODELHOST-L7): each engine start serves on
  `--host <root>/<random>/e.sock`, where the root is
  `%LOCALAPPDATA%\oa\s` on Windows, `$XDG_RUNTIME_DIR/oa` or
  `/tmp/oa-<uid>` on Linux and `$TMPDIR/oa` on macOS, in a directory only
  this account and SYSTEM can open, and gets no port, so no firewall rule or
  prompt applies. Every engine request binds the socket's server process to
  the engine's contained tree before its first byte; another process on the
  path reads `engine:peer` and receives nothing. Once per load, after
  readiness, the provider reads the TCP and UDP socket owner tables through
  `abstraction-identity`, and again every 30 seconds while the engine runs;
  an engine holding any TCP socket in any state or bound UDP socket,
  loopback and outbound included, is stopped, its lease released, and the
  call reads `engine:network-socket`. The directory is removed when the
  engine exits.
- **Media** (MODELHOST-L8): `chat@1` images are content digests this
  provider does not resolve, so an image request is refused at preflight.
  Every engine request also passes a guard that answers 403 for an http(s)
  or `file:` URL, GIF, BMP, WebP, video, audio, data whose bytes differ from
  its declared type, or any part other than text and base64 PNG or JPEG
  images, before the body leaves the process; each image is decoded and
  sent as the provider's own PNG. Each launch gets its own
  32-byte key as `LLAMA_API_KEY`, which only the guard sends, so a local
  process of this account reaching the socket directly is refused 401. The provider
  passes `--no-webui`.
- **Placement** (MODELHOST-L9): an engine started under a `card:<n>` lease
  gets `--no-mmap` and leaves the layer count to llama.cpp's `--fit`, which
  puts on the card what fits there; one under a `memory` lease gets
  `--n-gpu-layers 0`. A
  person's own `--n-gpu-layers`, `--load-mode`, `--mmap`, `--no-mmap` or
  `--mlock` replaces the matching default.

The review's condition 4, pinned SHA-256 digests for model files, is not
enforced: the allowlist names a store and an id, and the source hashes
nothing.

`serve` answers `Start`, `Observe` and `Cancel` on the named local endpoint
until interrupted. It accepts only the installed runtime executable beside
the provider, including the windowless executable on Windows. Pass
`--runtime-program` for a runtime in another directory. `--model` is the explicit allowlist: the name callers ask
for, and the object the storage inventory names. `--engine` is the
`llama-server.exe` that `engine install` staged. `--state` holds the engine
folder's metadata record. The lease endpoint, where `leases@1` answers, is
the declaration's `--resource-endpoint` flag first; else
`ABSTRACTION_RESOURCE_ENDPOINT`, which the runtime that starts the provider
sets to its own; else the installed runtime's endpoint. `openabstractions
models host` declares no `--resource-endpoint` unless the person passes
one, so an isolated runtime's provider holds its leases where that runtime
serves them. A provider the runtime starts on
demand writes its output to `<runtime state>/provider-logs/<name>.log`
(1 MiB, rotated once), the only place its output is kept whole; `provider
list` shows an exit with its last line, escapes and control characters
removed, credential-shaped values redacted and cut at 240 bytes, and that
path. Running `modelhostd help` (or with no arguments) prints the option
list in full. From the command line, once the runtime is installed:

```
openabstractions models host <store>/<id> --engine C:\...\llama-server.exe
openabstractions models hosts
openabstractions models unhost <store>/<id>
```

## What it needs

**Go** 1.26 or later ([`go/go.mod`](go/go.mod)), plus
[`abstraction-identity`](https://github.com/openabstractions/abstraction-identity),
`abstraction-inference/go`, `abstraction-router/go`, `abstraction-resource/go`,
`abstraction-storage-over-local-stores/go` and
`abstraction-provider-modelbridge/go`, whose `bridge.Source` is the seam this
layer resolves an object through. `llama-server` and `sd-server` come from
the publishers' archives a person downloaded and `engine install` staged;
this provider downloads nothing. The engine process, the video table and
the placed model files are the provider kit's
(`abstraction-provider-kit`).

## How the runtime uses it

The declaration is an ordinary `abstraction.facade/registry@1` provider of
role `provider` with an explicit model allowlist, which makes it a mediated
local resolver candidate and a router model server (facade CONTRACT.md
FAC-R2).
`models host` reads the runtime's own storage inventory for the store's name
for that object and hosts it under that name, so the router's family and the
inventory's `held_in` name the same model. `models unhost` of the last
hosted model withdraws the declaration, which ends the provider and its
engine.

The runtime writes the provider's program two rules when the declaration
lands: `abstraction.resource/hold` on `card:0` and on `awake` (CONTRACT.md
MODELHOST-R1). The provider-held wake is 0.4.0's stated provisional
exception, which a wake lease held for the requesting application replaces
after research (VISION 2026-10-02). Nothing else is granted. An
application still needs `abstraction.inference/complete` on resource
`server:modelhost` (the wire still carries `host:modelhost` until rights'
own dual reader ships, CONTRACT.md's "Reading this page"), decided by the
runtime before any call reaches here. Applications never reach this
endpoint directly: their calls travel through the runtime, which binds the
caller from the connection's program proof, decides the rule, and forwards.

## Measured

`serve`'s `TestLiveHostTheSmallestGGUFUnderALease` (`OA_LIVE_MODELHOST=1`)
builds this provider and the real `inventoryd`, stages the pinned engine
with `modelhostd engine install` from the archive `OA_LLAMA_B10825_ZIP`
names, declares both to an isolated runtime with `--embeddings`, hosts the
smallest GGUF the composed inventory lists, answers one chat through the
runtime, answers a second off the same load, answers one `embed@1` call
straight to this provider's own endpoint off the same load, then asks
`card:0` for its whole capacity and asserts the lease is gone. Without
`OA_LLAMA_B10825_ZIP` it skips and names the archive to download.

On 2026-09-22, before the pin, hosting `unsloth/SmolLM2-135M-Instruct-GGUF` (88,201,792
bytes) with an unpinned `llama-server --embeddings --pooling mean`: the first chat call
loaded the model in 868 ms, the second reused it in 145 ms, and one
`embed@1` call answered in 10.6 ms with 576 dimensions for 6 input tokens.
The chat model carries no pooling head of its own, so `--pooling mean` is
one of the engine's own arguments (`--engine-arg`) a person adds for a model
that is not already built for embeddings; llama-server otherwise refuses
`/v1/embeddings` on it as "pooling type 'none' is not OAI compatible"
(CONTRACT.md MODELHOST-L3). `--pooling` is on the reviewed argument list
(MODELHOST-L5).

## Why it is its own provider

The model bridge lends a file to an engine someone else runs. This provider
runs the engine itself: it starts a process, holds bytes on the card and answers
for them. Starting processes and holding a scarce resource is a different
trust and lifecycle from writing one link, so it is a different provider —
and the runtime can withdraw it, which stops the engine, without touching
lending. Both resolve an object through the same seam: the bridge's
`Source` is how `lend@1` asks for a path, and this layer reuses it rather
than scanning the machine a third time.

## Wrap, do not write

The engine is someone else's. `llama-server` from llama.cpp is what loads
the weights and answers the completion; this layer chooses the file, the
address and the moment, and adds the policy nobody else has: one copy, a
lease, and a yield. `sd-server` is stable-diffusion.cpp's in the same way.
An engine OA writes is its own provider (VISION 2026-10-02): the OA PyTorch
engine is `abstraction-provider-pytorch`.

## What it is not

It is one more model server that behaves, not the only one. An engine another
application started keeps its own copy; this provider makes no claim about
what ComfyUI, LM Studio or Ollama hold, and nothing here shares a loaded
model with a program that did not call `chat@1`. It schedules no compute.

## Video

With `--video-model NAME` the provider also serves
`abstraction.inference/video@1` for NAME with stable-diffusion.cpp's
`sd-server` and the reviewed video model set NAME (today FastWan 2.2 TI2V
5B, `fastwan2.2-ti2v-5b`), under the conditions of
`research/security-reviews/stable-diffusion.cpp-master-843-462d675-video-addendum.md`
(CONTRACT.md MODELHOST-V1..V9). The runtime's job executor sends a video job
for NAME here, stages its first frame in this provider's content store, and
reads the finished WebM from it.

Staging the engine and placing the model files, once (nothing here
downloads):

```
modelhostd engine install --kind sdcpp --archive sd-master-462d675-bin-win-vulkan-x64.zip
modelhostd models install --kind sdcpp --from <a folder holding the files>
```

| File | Size | Licence | Pin |
| --- | --- | --- | --- |
| `sd-master-462d675-bin-win-vulkan-x64.zip` | 38,865,230 bytes | MIT | SHA-256 `b9248856…a1c0a2`, 21 files |
| `FastWan2.2-TI2V-5B-q8_0.gguf` (Green-Sky, revision `3e8fe553…`) | 5.41 GB | Apache-2.0 | SHA-256 `b62f50ff…9c9208` |
| `umt5-xxl-encoder-Q8_0.gguf` (city96, revision `b535255b…`) | 6.04 GB | Apache-2.0 | SHA-256 `2521d4de…033bbc` |
| `taew2_2.safetensors` (madebyollin/taehv, commit `011dfc21…`) | 23 MB | MIT | Git blob SHA-1 `9d0ef195…666400` and SHA-256 `b84609b2…1f5325` (recorded by OA from a download whose blob matched) |

`models install` copies nothing. It finds each file in the `--from`
folders and in every store the storage inventory reads, proves its size and
SHA-256 where it lies, and records that path; a file of the right name and
other bytes is refused. `sd-server` reads the files where they lie, held
read-only while it runs, so they stay in place for as long as video is
served. A model set is a row of `VideoModelSets` in `go/host/videopin.go`;
`--set` and `--video-model` name it. `hummingbird-xt-i2v` is AMD's
Hummingbird-XT image-to-video transformer, the one ASUS MuseTree runs, read
where MuseTree keeps it (`--from C:\ProgramData\ASUS\AICreator`) with the
umt5 encoder and `taew2_2` above; it takes a first frame at 512×512, 81
frames, 16 fps, and carries AMD's Responsible AI Use Policy
(research/security-reviews/hummingbird-xt-i2v-files.md). A placed file is
proved by its metadata after one full hash only while no one beyond its
trusted owners can change it (`trustedpath.CheckFile`); under `%TEMP%`, or
anywhere another principal may write, it is hashed at every launch. Then
serve:

```
modelhostd serve --endpoint modelhost-video --video-model fastwan2.2-ti2v-5b ^
  --video-engine <path printed by engine install>
```

The engine starts on the first video, with fifteen minutes to load
(`--video-ready`), runs one video at a time with at most 32 waiting, and
stops five minutes (`--idle`) after the last. The preset is the provider's: three
steps, guidance 1, Euler, flow shift 3, 24 fps, WebM; sizes 832×480,
480×832, 1280×704 and 704×1280; at most 121 frames (5 seconds at 24 fps).
`sd-server` always runs with `--diffusion-fa` and `--vae-tiling`, and its
job object holds each process to 40 GiB of commit on Windows
(`--video-memory-limit`). The first live run on the Radeon 8060S
(research/live-runs/2026-10-01.md) generated 1280×704×121 in 644 s in the
engine, 669 s end to end, at a peak commit of 32.8 GB; without flash
attention the same request passed 87 GB and ran the device out of memory.
`Observe` reports the generation's percent from `sd-server`'s progress bars
and stage lines: loading and preparing fill the first 10, sampling the next
60, the decode the rest.

The runtime reaches the provider through its declaration: profile `video`,
the contract `abstraction.inference/video@1`, the entry
`video:fastwan2.2-ti2v-5b` and the card its engine loads onto, `card:0`.
The card is what makes the runtime write the provider's
`abstraction.resource/hold` rules on `card:0` and on `awake`, which the
GPU fault guard's wake lease needs while a video runs. One command writes
that declaration, beside any chat model already hosted, after proving the
engine with `engine check`:

```
openabstractions models host --video-model fastwan2.2-ti2v-5b --video-engine <path printed by engine install>
```

Running it again changes nothing; a model host declaration written by hand
with `provider add` is replaced in place, its other `serve` options kept.

A video job for that model goes to the provider through the runtime's job
executor; applications submit video as a job and never call `video@1`
themselves.

On Windows `modelhostd` reports "holder: binding fastwan2.2-ti2v-5b:
unverifiable" at every engine start. The lease service binds an engine to
its lease only when its GPU instrument names processes by a verified
identity, and Windows' GPU process memory counters name a PID without the
process's creation time (abstraction-resource
`go/instrument/counters_windows.go`, `VerifiedProcessSamples`). The lease
then keeps its full reservation, as MODELHOST-H5 says; nothing is wrong
with the engine.

### Exposure while the engine runs

`sd-server` has no authentication and reflects any page's `Origin` in its
CORS headers. While it runs, any process on this machine, under any
Windows account, and any browser page that finds its loopback port can
submit work to it, read every result it keeps for ten minutes, cancel
queued work, learn the model's path, and crash the engine: through a heap
overflow from an image size it trusts (finding V1 of the review), a long
prompt (V2), or the published cpp-httplib advisories. This provider limits
what it controls: its own requests carry only reviewed keys and bounds and
its own re-encoded first frame (MODELHOST-V6, V7), the port is a fresh
loopback port per launch, and the engine runs only while video work does
and the idle time after it (MODELHOST-V5). It cannot add authentication to
`sd-server`. Finding V1 is to be reported to stable-diffusion.cpp as a
security advisory once the owner agrees; it has not been reported.

### Video and images on PyTorch

`hummingbird-xt-i2v` on the OA PyTorch engine, with AMD's TurboVAED decoder,
and FLUX.2 [klein] images are the PyTorch provider's,
`abstraction-provider-pytorch`.
`modelhostd` refuses `--video-kind torch`, `--torch-python`,
`--torch-source`, `--video-model-option`, `--image-model` and `--kind torch`
with that provider's name; `openabstractions models host` takes the same
options and declares that provider.

## Several residents

Version 1 places no limit on how many objects on the allowlist may be
resident together. Each gets its own engine process, on its own socket
in its own launch directory, under its own lease and its own idle clock (CONTRACT.md MODELHOST-H5):
asking for a second object while the first is resident starts a second
engine rather than waiting for or replacing the first. A yield request
names an amount of the resource, not one lease, so a provider holding more
than one resident gives up whichever has gone longest untouched first,
stopping once enough is freed (MODELHOST-H7).

## Layout

- **`go/host/`** — the operation. `object.go` is the seam that resolves a
  store and an id to a binding, of which this provider accepts a whole file
  only; `pin.go` is the pinned llama.cpp release and
  its install from the publisher's archive; `engine.go` proves the engine
  folder against the pin, starts one `llama-server` on a socket in a fresh
  launch directory, waits for its own `/health` over that socket and stops
  it by its process id; `engineargs.go` is the
  reviewed argument list, `engineenv*.go` the environment built from
  nothing, `listener.go` the network-socket check after readiness and `media.go`
  the guard on every engine request; `host.go` is
  `Ensure`, `Unload` and the loops that renew each resident's lease, answer
  a yield request and unload a resident that has gone idle, with the typed
  refusals.
- **`go/service/`** — the endpoint. It serves `chat@1`, and `embed@1` when
  the engine was started with embeddings, on the shared identity-bound
  native transport, admitting only the designated runtime program. Every admitted
  call loads the model the request names before it is mediated to the
  engine, so a load that cannot happen is the call's refusal in this
  provider's own words. `ProviderConfig` is the inference configuration
  `modelhostd` serves.
- **`go/cmd/modelhostd/`** — the one binary this layer produces.
- **`go/host/testdata/fakeengine/`** — a stand-in engine, enough of the
  OpenAI-compatible API to report readiness and answer one completion. The
  tests spawn it so they measure this provider and not llama.cpp.
- **`go/host/videopin.go`, `videoengine.go`, `video.go`** — the video kind:
  the pinned `sd-server` release and model set with their staging, the
  engine's start, readiness and bounded log, and the `video@1` operations
  with their queue, request ids, first frames, results and cancels.
- **`go/host/testdata/fakesdserver/`** — a stand-in `sd-server` with the
  capabilities, video job, job status and cancel routes, returning a small
  WebM.

## Limits

- **Only a llama.cpp-compatible engine is wired in today.** Two NPU engines
  were surveyed on the machine this module is developed on and are not
  wired in: `research/model-host/ENGINES-2026-09-22.md` records what was
  found and what wiring either one in would still need.
- **The runtime's own `embed@1` dispatch has no adapter yet for a
  natively-declared local model server like this one** — only `router.WireRemote`
  and `router.WireOpenAICompatible` are reached
  (`abstraction-inference/go/embed.go`), unlike `Complete`'s `nativeChat`
  adapter. The measurement above calls this provider's own endpoint
  directly rather than through `machine.ResolveEmbeddings`; closing that gap
  is outside this module.
- **This provider delegates no call onward.** Its own inference provider
  admits every request as a local leaf execution, so a call that reached
  it is answered by the engine it started or not at all.
- **It serves chat, embeddings and one video model.** Image diffusion
  belongs with the `image@1` providers: sd.cpp on Lemonade's, PyTorch on the
  PyTorch provider.
- **The video engine runs on Windows x64 alone.** No other platform has a
  reviewed `sd-server` build.

## Platforms

The model host serves engines on Windows x64 alone. The Linux and macOS
packages carry `modelhostd` as an opt-in feature with no pinned engine, and
there it refuses every engine with a stated reason:

- **Linux:** abstraction-identity reads socket-owner tables on Windows
  alone, so an engine's sockets cannot be checked at readiness
  (MODELHOST-L7, MODELHOST-V5): `engine:network-socket` for a chat engine,
  `engine:listener-not-loopback` for the video engine.
- **macOS:** `modelhostd serve` stops at startup, because its socket
  endpoint cannot carry the Program proof it requires of the runtime
  (abstraction-identity `CanEver`). Behind it, no primitive binds an
  engine's server process, on its AF_UNIX socket (ID-T8) or its loopback
  port (`LoopbackCeiling`), so every engine request would fail its peer
  check (inference INF-EL2) with `engine:peer` or `engine:unavailable`.

The host package's tests replace the owner-table check off Windows to
measure the lease, load and unload logic with the stand-in engines; tests
that send a request to a running engine skip on macOS.

## Status

Development provider for OA 0.3.0. The
[Go source release](https://github.com/openabstractions/abstraction-provider-modelhost/releases/tag/go/v0.1.0)
is tagged `go/v0.1.0`. This repository has no standalone release assets.
The optional Windows installer feature is defined in the source tree. The
[installation guide](https://openabstractions.org/adopt.html) describes available
release packages.

The OA PyTorch engine is its own provider module and process,
`abstraction-provider-pytorch`, gated until it has OS-level network
containment (VISION 2026-10-02, "OA may supply an engine process").

## Licence

Apache-2.0. See [LICENSE](LICENSE), published beside this file at the
repository root.

## Design records

- [HISTORY.md](HISTORY.md) — measurements, superseded ids and the reviews
  applied to this release.

- `DREAM.md` §2 "One host", §10 — the design for one host holding one
  resident copy per model.
- `research/architecture-review-2026-09-22.md` F3, R3 — the review that set
  this layer's scope.
- `research/model-bridge/DECISION.md` §4 rules 4 and 6 — the decision that
  this provider bundles and downloads nothing.
- `VISION.md` 2026-10-02, "OA may supply an engine process" — the decision
  that moved the PyTorch engine into its own provider module.
