# provider-modelhost contract

The contracts this provider serves are
[`abstraction.inference/chat@1`](https://github.com/openabstractions/abstraction-inference/blob/main/CONTRACT.md),
`abstraction.inference/embed@1` and `abstraction.inference/video@1`,
whose words bind every call; the rules
below say what this provider does around them — what it loads, what it
holds, and when it lets go. This file states the obligations a provider
meets; [README.md](README.md) holds the walkthrough. `research/architecture-review-2026-09-22.md`
R3 carries the design, `research/model-bridge/DECISION.md` §4 rules 4 and 6
the engine rule, and [`abstraction-resource`](https://github.com/openabstractions/abstraction-resource/blob/main/CONTRACT.md)
RES-L1 to RES-L4 the lease this provider holds.

Binds: none (records only)

The service loads each model under a [lease](https://openabstractions.github.io/reference.html#concept-resource.lease), serves it for as long as calls
arrive, and stops the engine and releases the lease when the resource is
asked for or the idle time passes. It serves three things on two engine
kinds:

| Service | Engine | Models | Rules |
| --- | --- | --- | --- |
| `chat@1`, `embed@1` | the pinned llama.cpp `llama-server`, one process per model | the model files the declaration's allowlist names | L1..L9, W1..W3 |
| `video@1` | the pinned stable-diffusion.cpp `sd-server` (`--video-kind sdcpp`, the default) | `fastwan2.2-ti2v-5b`, `hummingbird-xt-i2v` | V1..V10 |

The holding (H), GPU fault (G), engine exit (E) and rights (R) rules apply
to every kind.

`modelhostd` ships in the Windows installer's optional `ModelHost` feature,
off by default, and a person configures it with `openabstractions models
host`. The llama.cpp and stable-diffusion.cpp kinds run the publishers'
pinned builds. `modelhostd engine install --bundle` stages llama.cpp from
OA's reduced bundle, which the Windows package carries with the app-local
Visual C++ runtime the build needs; `modelhostd engine install --kind
sdcpp --archive` stages stable-diffusion.cpp from the archive the person
downloads. The OA PyTorch engine, which the model host carried
until 2026-10-02, is its own provider, `abstraction-provider-pytorch`
(VISION 2026-10-02, "OA may supply an engine process"); its rules moved there
as PYTORCH-T1..T11, and `modelhostd` refuses its options with that
provider's name. The model host wraps engines the machine has.

## Reading this page

The conventions every OA contract carries are stated once in [CONVENTIONS.md](https://github.com/openabstractions/abstractions/blob/main/idl/CONVENTIONS.md); a rule here that applies one cites its `CONV-` id.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174]
when, and only when, they appear in all capitals, as shown here.

The rule ids below, such as `MODELHOST-L1`, are what tests and refusals
cite. A retired id is never reused; [HISTORY.md](HISTORY.md) keeps it with
the release it left.

| Letter | Meaning |
| --- | --- |
| L | loading |
| H | holding |
| R | rights |
| V | video |
| G | GPU faults |
| E | engine exits |
| W | preloading a chat prefix |

| Prose | Wire (until its own release) |
| --- | --- |
| resource `server:modelhost` | resource `host:modelhost`, owned by rights' stored policy; the dual reader lands with that module, not here (`research/vocabulary/RENAME-PLAN.md` §4, first release) |

## What it loads

**[MODELHOST-L1] Object naming.** The service MUST name an object the way
the storage inventory names it: a store and an id, never a path. The
declaration's allowlist is `NAME=STORE/OBJECT` entries. The service resolves
the id to a file through a source, the same seam `abstraction.storage/lend@1`
asks for a path through (modelbridge CONTRACT.md MODELBRIDGE-L1); the
declaration carries no path and no caller supplies one. An id no store the
service reads holds is `unavailable` with reason
`object:unknown:<store>/<id>`. The service MUST ask the source for a `file`
binding only, because `llama-server` at the pinned release takes a model as
`-m FNAME` alone (modelbridge CONTRACT.md MODELBRIDGE-L19). An object that
exists only as a member of a zip is `unavailable` with reason
`object:unbound:<store>/<id>`, refused before any lease is asked for. The
provider holds the inference kit's memory fill (inference CONTRACT.md
INF-MS1..INF-MS5): a `file`, `region` or DEFLATE `compressed` binding fills a
read-only section proved against the object's digest, described by a
`memory` binding, and lent to an engine through supervision's handle list.
No `llama-server` release this provider pins reads a model from an inherited
handle, so it declares `file` only and no load fills a section. An
object the source refuses as a PyTorch pickle, which OA never loads, is
`unavailable` with reason `object:pickle:<store>/<id>`, refused before any
lease as well (modelbridge CONTRACT.md MODELBRIDGE-L20).

An explicit `Model.Content` pin MUST be retained per allowlist object. The
additive `--model-content STORE/OBJECT=sha256:<hex>:<bytes>` `modelhostd` argument
MUST name an existing `--model` binding exactly once and match one compiled
coding catalogue entry's digest and size; otherwise the command exits 2.
A pinned source-size mismatch MUST refuse before lease acquisition. Launch
MUST use `programfiles.LaunchPlaced` to verify the retained expected bytes,
open the held final path and keep the file and its ancestors through confirmed
engine exit. Its record is keyed by store/object and its trusted owners are
the existing folder-search owners. A mismatch refuses before launch and never
creates a replacement pin from current bytes. Explicit pins apply to every source,
including embeddings. Missing Content retains the existing unverified
file launch; `ContentProof`, a store name and source settings supply none.

**[MODELHOST-L2] The allowlist is exhaustive.** A model the declaration does
not name MUST NOT be loaded, whatever the engine could open. The router
lists exactly those names as the model server's models, so a hosted model
is visible before anything is running and no survey starts an engine to
find out what it holds (facade CONTRACT.md FAC-R2). The runtime writes the
allowlist from the storage inventory: every chat GGUF a store on this
machine holds, one entry per family, named by its descriptor family, beside
what a person or Setup hosted; a model a person removed stays out until it
is hosted again. Writing it never starts an engine.

**[MODELHOST-L3] The engine is a configured path of the pinned release.**
`--engine` names the `llama-server` of the one llama.cpp release this
provider pins (VISION 2026-10-01). On Windows x64 the pin is b10825 Vulkan:
the publisher archive's size and SHA-256, the reduced bundle's size and
SHA-256, and the size and SHA-256 of each of the 27 files the engine folder
holds: 24 from the publisher archive and 3 of the Microsoft Visual C++
runtime the publisher's build links to, deployed app-locally
(`go/host/pin.go`). Before every
start the provider proves the folder against the pin with the inference
provider kit (inference CONTRACT.md INF-F1..INF-F4) and holds its files
until the engine's exit is confirmed. This layer ships no llama.cpp build,
downloads none and looks for none. No engine configured, a relative path, a path that
names no file, a program other than the pinned one, a folder that differs
from the pin in any file, and a platform with no pinned release are all
`unavailable` with reason `engine:none`, before any lease is asked for
(`research/model-bridge/DECISION.md` §4 rule 6). A folder with a file the
provider cannot open or read, such as one a scanner holds, is `unavailable`
with reason `engine:unreadable` and the system error, also before any
lease; it passes once the file is free (inference CONTRACT.md INF-F1). `modelhostd engine install
--bundle` stages the pinned files from the reduced bundle after checking its
size and SHA-256, and restages a folder staged from an earlier pin. A pin
that carries runtime files stages from the reduced bundle alone:
`--archive` refuses in one sentence naming the missing runtime and the
bundle to give instead.

**[MODELHOST-L4] The engine process is this provider's own.** The service
MUST start it with the model and the socket path this provider chose
(MODELHOST-L7), and MUST stop it by that process's own id. No process is
named by image, so no engine another program started is ever touched.

**[MODELHOST-L5] Engine arguments are the reviewed list.** A configured
engine argument MUST be one of the llama-server options of the pinned
release that size, place or schedule the model (`--ctx-size`,
`--n-gpu-layers`, `--threads`, `--batch-size`, `--parallel`, `--flash-attn`,
`--cache-type-k`/`-v`, `--pooling`, `--device` and the others
`go/host/engineargs.go` lists), each with a value of its reviewed form, or a
switch that turns a feature off. `--load-mode` (`auto`, `none`, `mmap`,
`mlock`, `mmap+mlock`, `dio`), `--mmap` and `--no-mmap` choose how the
engine reads the model file this provider named, so a person can replace
MODELHOST-L9's default. `--spec-type` MAY carry only `draft-mtp`
and `--spec-draft-n-max` only an integer from 1 to 8: they choose the
model's own draft head and a bounded draft length, name no file and open no
port. The options that name a draft model (`--model-draft`, `-md`,
`--hf-repo-draft`, `-hfd`) or a lookup cache file are refused. Every other option MUST be refused by
name before anything starts: the b10825 review's condition 2 list
(`--host`, `--rpc`, `--tools`, `--agent`, the MCP options, `-hf` and the
`--hf-*` forms, `--model-url`, `--mmproj-url`, `--docker-repo`,
`--models-dir`, `--models-preset`, `--media-path`, `--path`,
`--ui-config-file`, `--slot-save-path`, `--log-file`, `--log-prompts-dir`,
`--sleep-idle-seconds`), options that read a named file (`--lora`,
`--chat-template-file`, `--api-key-file`, `--ssl-*`, `--mmproj`),
`--chat-template`, an unknown option, `--port`, and the options this
provider sets itself (`--model`, `--host`, `--alias`, `--embeddings`,
`--no-webui`).
`modelhostd serve` and `engine check` exit 2 naming the option; a library
`New` returns the error. The condition 2 list is the provider kit's
(abstraction-inference CONTRACT.md INF-LC1), refused before the reviewed
list is read, and the reviewed list admits none of its spellings.

For each chat launch the provider MAY fill unset arguments from its curated
model rule, using model's pure `FitCatalog` metadata. It MUST require current
`hashed` evidence for the exact source object, matching the catalogue digest
and size. The optional concrete source `ContentProof` capability rechecks the
existing recorded digest's live file stamp; an earlier source or missing,
stale, named or mismatched evidence keeps the configured arguments. The
provider copies the Engine and argument slice per launch, and explicit
operator flags win. The catalogue supplies the context; the provider's
reviewed Qwen3.6 35B MTP rule supplies `draft-mtp` with draft length three.
These defaults MUST NOT become global arguments for other hosted models.

**[MODELHOST-L6] The engine environment is built from nothing.** The
engine MUST start with no inherited variable; the provider kit's
clean-environment builder makes it (abstraction-inference CONTRACT.md
INF-EL1). On Windows it holds
`SystemRoot` and `windir` from the kernel's Windows folder, `SystemDrive`
(that folder's volume), `PATH` of the System32 folder alone, `TEMP` and
`TMP` naming a private folder in the provider's state, `USERPROFILE` (the
token's profile directory), `LOCALAPPDATA` and `APPDATA` (this account's
User Shell Folders), `ProgramData` and `ALLUSERSPROFILE` (ProfileList), and
`NUMBER_OF_PROCESSORS` and `PROCESSOR_ARCHITECTURE` (GetNativeSystemInfo);
elsewhere `PATH=/usr/bin:/bin` and `TMPDIR`. Registry folder values are
expanded only with variables the provider computed, never its own
environment. A profile folder that cannot be read, or whose value names any
other variable, is left out (`ALLUSERSPROFILE` together with `ProgramData`)
and the engine starts without it. `NUMBER_OF_PROCESSORS` and
`PROCESSOR_ARCHITECTURE` are left out the same way when GetNativeSystemInfo
reports no processor count or an unknown architecture. For each variable
left out the provider reports one diagnostic through its error channel
naming the variable, never its value. `LLAMA_API_KEY` is the launch's own key (MODELHOST-L8).
`GGML_*` (`GGML_BACKEND_PATH` loads a DLL), `LLAMA_ARG_*`, an inherited
`LLAMA_API_KEY`, `LLAMA_CACHE`, `HF_*`, `MTMD_*`, `OMP_*`/`KMP_*`, the
Vulkan loader's `VK_*` variables and every proxy variable therefore never
reach it, and the provider kit's forbidden names (abstraction-inference
CONTRACT.md INF-LC2) are refused even if one were added to the allowlist.
The Vulkan loader needs no variable: it is in System32 and finds
drivers through the registry. The profile folders carry the GPU driver's
pipeline caches and llama.cpp's cache directory, which it reads from
`LOCALAPPDATA` without a check.

**[MODELHOST-L7] The engine holds no network socket, and its socket's
server is the engine.** The provider MUST make a launch directory for each
engine start through the provider kit (abstraction-inference CONTRACT.md
INF-EL2) under the kit's fixed per-user socket root, whatever `--state`
names: `%LOCALAPPDATA%\oa\s` on Windows, `$XDG_RUNTIME_DIR/oa` or
`/tmp/oa-<uid>` on Linux, `$TMPDIR/oa` on macOS. The root is created
private to this account and refused if another principal owns it, it is
open beyond this account (and SYSTEM on Windows), or it is a file or a
link. A Windows process with a virtualized or unreadable profile view
refuses this default root before making a launch folder; a provider may
name a private socket root outside the redirected profile. The kit proves
a bind and connect on each Windows launch path, including under an explicit
root. A failed connect is `engine:unreachable` before an engine starts.
The launch directory has a random name, is refused if the path
exists, admits only this account and SYSTEM on Windows and mode 0700
elsewhere, and holds a `held` file the provider keeps locked while the
directory is in use. It MUST pass `--host <launch directory>/e.sock`
itself and no `--port`; llama-server b10825 serves AF_UNIX when `--host`
ends in `.sock` (`tools/server/server-http.cpp:438-444`), Windows
included. A root whose socket path cannot fit 107 bytes of printable ASCII
is refused when the provider starts. Every engine request, the readiness
checks included, MUST go over that socket through a client that binds the
process the kernel names for the socket's server end before the request's
first byte (abstraction-identity ID-T8) and admits it only as a member of
the engine's contained tree by process id and creation time (the
supervised job on Windows; elsewhere, the process this provider started).
A server that fails the check receives no byte; at readiness the load is
refused with reason `engine:peer`. A readiness whose every connect for 3
seconds failed with an error a starting engine never gives is refused
with reason `engine:unreachable`, naming the system error (inference
INF-EL2, provider kit KIT-P5); the video kind's probe keeps the same rule. Once the engine reports ready, and
again every 30 seconds while it runs, the provider MUST confirm through
the socket owner tables that the engine's contained tree owns no TCP
socket in any state, outbound connections included, and no bound UDP
socket, on any address, loopback included. An engine that holds one at
readiness is refused with reason `engine:network-socket`, as is every load
on a platform with no socket owner tables. An engine found holding one
later is stopped by its own process, never by its model's name, and its
lease is released once the holder sees it exit; while it is still its
model's resident the provider reports `engine:network-socket` through its
error channel. A newer engine loaded for the same model in between keeps
running. A refused engine is a
load that did not finish (MODELHOST-H3): it is stopped and its lease
released. The launch directory is removed once the engine's exit is
confirmed, after an unload or a crash. `Host.Close` and the holder's
`Owned.Stop` wait within one 20 s stop budget for the exited engine's
provider-owned cleanup to finish: the chat socket directory and engine-file
hold, or the video engine/model-file holds and log. A cleanup still pending
at that deadline leaves the lease held for a later retry. A completed file
cleanup error is reported through `OnError`, but a confirmed-dead engine
releases its scarce-resource lease; a stale socket is not evidence of a live
GPU user. When it starts, the provider
removes the launch directories under the root that no provider holds and
whose socket no longer answers a dial; a second provider sharing the root
keeps a directory whose engine is still starting. The video kind's
`sd-server` serves TCP only and keeps INF-EL2's loopback rule through the
same check (MODELHOST-V5): readiness, job submission, polling and
cancellation all go through the engine's own client, which binds each
loopback connection to the engine's tree by the TCP owner table, so a
listener on its port outside the tree, before readiness or after the
engine exits, receives no request. llama-server opens its
socket before it reports ready, and its only later child process, FFmpeg
for video and WebP, is unreachable because MODELHOST-L8 refuses both and
`--video-ffmpeg-dir` is never passed.

**[MODELHOST-L8] No media the engine would fetch or decode reaches it.**
`chat@1` carries an image only as a content digest, and this provider
resolves none, so a request with an image part is refused at preflight
(`unavailable`, `content:unavailable`) before any lease or engine. Every
request the provider sends an engine MUST also pass a guard before it leaves
the process. The guard applies the provider kit's chat media policy
(abstraction-inference CONTRACT.md INF-LC3) without audio: chat content holds
only text and `image_url` parts whose URL is a base64 PNG or JPEG data URL
of at most 16 MP whose bytes are the type it declares, and each such image
is decoded and sent as the provider's own 8-bit NRGBA PNG data URL in a
body the guard encodes again. Embedding and completion inputs hold only
text and token ids. An http(s) URL (`media:remote`), a `file:` URL
(`media:file`), GIF, BMP, WebP, video, audio, data whose bytes differ from
its declared type or do not decode, any other part or data type
(`media:type`) and a body that is not one JSON object (`media:request`) are
answered 403 in the engine's place, which the call reads as `refused` with
reason `upstream:403`; the provider reports the word through its error
channel. Each engine start MUST get its own key of 32 random bytes as
`LLAMA_API_KEY` in the engine's environment, held only in the provider's
memory and that environment block, never on the command line or in a log.
The guard sends it as `Authorization: Bearer KEY` on every engine request,
so a local process of this account that reaches the engine's socket
directly is refused;
`/health` stays open for readiness. The provider also passes `--no-webui`.

**[MODELHOST-L9] The engine's placement follows its lease.** The provider
MUST derive a `llama-server` engine's default placement from the resource
of the lease it starts under and from nothing specific to the machine. The
lease admits the file's size plus the provider cache recipe's explicit RAM
and checkpoint forecast, as an admission estimate (MODELHOST-H1). Placement
uses the file size alone. It covers no other KV cache or compute buffer.
Where the resource has a pool,
a file larger than what can be freed reads `lease:insufficient` or
`lease:holders_refused` before any engine starts (resource RES-L2,
MODELHOST-H2). On a machine with no pool the acquire never reads
`lease:insufficient` (resource RES-L2), and the engine finds the limit
itself. On a
shared-memory adapter the card's pool is the machine's memory (resource
RES-L2).

An engine under a `card:<n>` lease MUST start with `--no-mmap`, which reads
the file into the engine's own buffers. The layer count is left to the
engine's own placement, b10825's `--fit`, which measures the device memory
free at start and keeps the layers that do not fit in system memory. A
discrete card smaller than the model therefore keeps a partial offload. The
system memory those layers take is outside the card lease, which is an
estimate and no quota (MODELHOST-H1, resource RES-L2). An engine under a
`memory` lease MUST start with `--n-gpu-layers 0`, which keeps its weights
in the resource the lease admitted. Any other resource adds no placement.

When the file, the KV cache and the compute buffers together exceed what the
device can hold at the placement the engine was given, such as a person's
`--n-gpu-layers` above what fits, the engine fails to load or to become
ready, and the lease is released (MODELHOST-H2, H3). When the engine
printed one of llama.cpp b10825's allocation failures before that (a line
holding `failed to allocate`, `unable to allocate`, `out of memory` or
`ErrorOutOfDeviceMemory`, other than Vulkan's pinned-memory warning, which
the engine continues past), the call MUST read `engine:does_not_fit` with
the line and the remedy: `--engine-arg --n-gpu-layers N` with a smaller N,
or a smaller `--ctx-size`. Any other start failure reads `unavailable`.

A person's own `--n-gpu-layers` replaces the layer default, and their own
`--load-mode`, `--mmap`, `--no-mmap` or `--mlock` replaces the load default,
whatever spelling they used (MODELHOST-L5). The placement holds for every
backend the provider pins; today that is Windows x64 Vulkan alone
(MODELHOST-L3).

**[MODELHOST-L10] The provider warms the model files while the engine
starts.** Once an engine is running, the provider SHOULD read the model
files that launch's proof did not read in full into the operating system's
file cache, with parallel reads, in the order the engine loads them: for
`sd-server` the text encoder, then the diffusion model, then the decoders.
It reads nothing when the files together exceed half of the physical memory
the system reports available, or when that amount is unknown. Warming stops
when the engine exits. A read error ends it early. It never fails or delays
a launch, and the engine log gets one `phase=warm` line with the bytes read
and the time taken, or the reason nothing was read. Warming touches the
held files read-only and changes no byte the proof covers.

**[MODELHOST-L11] The provider re-checks its model files monthly and names
a failing file plainly.** Every 10 minutes, while no model loads, no video
engine runs and the session has had no input for 10 minutes, the provider
re-hashes the files of each record (each engine tree's, the video set's and
each pinned chat model's) last fully hashed more than 30 days ago
(INF-F2, INF-F7). It stops
as soon as any of those conditions ends and never holds the proof's lock
while it reads. A file whose bytes no longer match writes one line, and its
next load is refused. A proof that fails because another program holds a
model file open for writing reads "the model file X is still being written
by another program; try again once it finishes". A file that no longer
matches reads "the model file X no longer matches its download" with how to
download it again from the archive or the internet. The proof's handles,
opened with read sharing only, stay open until the engine's exit is
confirmed, which covers the engine's own open of each file. Once per start
the provider writes one line when its state folder proves no record across
starts, with the reason, and one line for each app container or capability
SID (`S-1-15-*`) that may change a folder above its state folder.

**[MODELHOST-L12] A model's projector loads with it, and its files state
its capabilities.** When the model's store holds a vision projector for the
model (bridge `Projectors`: a projector entry in the model's own manifest,
else a projector beside its weights file whose projection width matches the
model or whose name carries its family), the provider MUST start the engine
with `--mmproj` naming that file and MUST count its size in the lease.
`--mmproj` stays off the reviewed argument list (MODELHOST-L5). The
endpoint's chat@3 description MUST state each allowlisted chat model's
capabilities (router `CapabilityFacts`) from cached reads that never wait on
a file: text input and output; image input with a projector, and audio
input when that projector's header states `clip.has_audio_encoder`; tools
when the GGUF chat template formats tools; and, for a template with a
thinking section whose family profile has reasoning budgets, the llama.cpp
reasoning control and the `llama.cpp/enable_thinking` and
`llama.cpp/reasoning_budget` extensions. A fact the files do not state is
omitted. The endpoint's content store takes the chat input images the
runtime stages before relaying a chat@3 turn, and the inference provider
resolves the request's image digests from it.

## What it serves as video

With `--video-model` the provider also serves `video@1` (inference
CONTRACT.md INF-G1..INF-G8) for that one name with stable-diffusion.cpp's
`sd-server`, under the conditions of
`research/security-reviews/stable-diffusion.cpp-master-843-462d675-video-addendum.md`.
The provider kit package `providerhost/sdcpp` holds the command line,
environment, request, result and first-frame policy (inference CONTRACT.md
INF-SD4..INF-SD8); the rules below say how this provider applies it.

**[MODELHOST-V1] The video engine is the pinned sd-server.**
`--video-engine` names the `sd-server.exe` of stable-diffusion.cpp
master-843-462d675, Windows x64 Vulkan: the publisher archive
`sd-master-462d675-bin-win-vulkan-x64.zip` by size and SHA-256 and its 21
files by size and SHA-256 (`go/host/videopin.go`).
`modelhostd engine install --kind sdcpp` stages exactly those files from
the archive (inference CONTRACT.md INF-F6). Before every start the provider
proves the folder against the pin and holds its files until the engine's
exit is confirmed (INF-F1..INF-F4); the engine runs in that folder. A
missing, unpinned or changed engine, and a platform with no pinned release,
fail the video with reason `engine:none` before any lease.

**[MODELHOST-V2] Model files are bound where they lie.** A video model set
is one row of `VideoModelSets` (`go/host/videopin.go`): the name callers ask
for, its files by role, size and SHA-256, its reviewed switches, its preset
and its bounds. `--video-model` names a row; a name no row carries exits 2.
A new model is a new row and the review its `Review` field names. OA never
copies a model for an engine (VISION.md, 2026-10-01; video addendum
amendment 1). A row may name a use policy its publisher attaches to the
weights; wherever the set's licence is shown, the policy is shown with it.
`hummingbird-xt-i2v` names the AMD Responsible AI Use Policy.
The record keeps, beside each path, the owners the file may have to be
trusted in place: the machine's (SYSTEM, Administrators, TrustedInstaller)
and this account for a `--from` folder, the machine's alone for a store.
Before the metadata shortcut is taken for a file, identity's
`trustedpath.CheckFile` (ID-D2) must pass for those owners; a file that
fails it is hashed in full at every launch (inference CONTRACT.md INF-F7).
The record's owners are the machine's and this account and nothing else; a
record naming any other SID is refused. OA's set folders are the recorded
sets alone: a folder directly under the models root (`--models-root`,
default `<home>\OA-models`) that holds a file an sd-server set's places
record names, or one named as a set of `VideoModelSets`. The PyTorch
provider records its own sets' folders (KIT-M3). `models install` makes the set folder of
each file it records there a model folder (identity ID-D3) after placing,
and `modelhostd models protect` does so once for every recorded set folder,
listing every other folder under the root as left alone; `--dry-run` reports
and writes nothing. The models root must pass `privatedir.CheckModelsRoot`:
no drive root, home or known folder, no link on its path, resolving to
itself. A set folder holding a link or junction, or a file with another name
outside the models root and the runtime's job store (`--job-store`, whose
results `LinkResult` hard-links out, job JOB-A17), is refused unchanged. No
file with a name outside the models root and the job store is changed; a
file whose second name lies in the job store is the same file. A job store
that exists must pass `privatedir.CheckModelsRoot` as the models root does,
so a drive root, the home or a linked folder given as `--job-store` is
refused. The job store defaults to the runtime's own
(`modelplace.OwnedRoots`); an empty `--job-store` is refused.

**[MODELHOST-V3] Engine arguments are the reviewed list.** The provider
builds the command line itself (INF-SD4): `--listen-ip 127.0.0.1`, the port
it chose, and `--diffusion-model`, `--t5xxl` and `--tae` naming a Wan set's
placed files, or `--diffusion-model`, `--llm`, `--vae`, `--audio-vae` and
`--hires-upscalers-dir` (the folder holding the placed upscaler) for an LTX
set (MODELHOST-V2). LTX-2.5 runs with `--diffusion-fa`, `--vae-tiling` and
`--temporal-tiling`. The model set's own reviewed switches follow: FastWan 2.2 TI2V 5B
runs with `--diffusion-fa` and `--vae-tiling`, without which its largest
size ran the device out of memory (research/live-runs/2026-10-01.md).
`--video-engine-arg` admits only the reviewed device, memory and
logging options; every option the review forbids is refused by name, as is
any option the provider sets itself or the reviewed list does not hold.
`modelhostd serve` exits 2 naming it and a library `New` returns the error.
`-M` is never passed: `sd-server` chooses video from the loaded model.

**[MODELHOST-V4] The video engine environment is built from nothing.** The
engine starts from the clean environment of MODELHOST-L6 without
`LLAMA_API_KEY`, and the provider refuses a name starting `GGML_`, `SD_`,
`VK_` or `HF_` and every proxy variable even if one were added to the
allowlist (INF-SD5). Its device is chosen with `--backend`.

**[MODELHOST-V5] The engine runs while video work does.** The first video
loads the engine under a lease of the model set's total size, taken as in
MODELHOST-H1..H4; one provider owns one `sd-server`. Each launch takes a
free loopback port. The engine is ready when `GET /sdcpp/v1/capabilities`
answers 200 with `vid_gen` in `supported_modes` within the load budget
(`--video-ready`, default 15 minutes, a cold read of the model set), and
the provider then runs
the loopback check of MODELHOST-L7; an engine that fails it is stopped, its
lease released, and the video fails with reason
`engine:listener-not-loopback`. The engine stays while video work is queued
or running and stops the declared idle time (`--idle`, default 5 minutes)
after the last video. A yield request for its lease is refused while video
work is queued or running (MODELHOST-H7).

**[MODELHOST-V6] The provider owns the preset and the bounds.** The
provider runs one video at a time in admission order, with at most 32
waiting behind it and 4,096 operations held in all; a `Start` beyond either
is `exhausted` with reason `queue:full` or `operations:full`. It builds every
engine request from its model set's preset (INF-G8, INF-SD6), and holds a
caller to the set's sizes, frames, rate and first-frame use, each within
the engine's reviewed bounds. FastWan 2.2 TI2V 5B runs 3 steps at guidance
1 with Euler, the discrete schedule and flow shift 3, at 24 fps, 832×480 for
size `auto`, as WebM; a caller's size is one of 832×480, 480×832, 1280×704
and 704×1280, its rate at most 30 fps, and its duration at most 121 frames
at the rate, and a first frame is optional. Hummingbird-XT image to video
runs 3 `lcm` steps at guidance 1 on AMD's sigma table (1, 0.967742,
0.909091, 0, sent as `custom_sigmas`), at 16 fps, 512×512, as WebM; a
caller's size is 512×512, its duration at most 81 frames at 16 fps, and a
first frame is required. LTX-2.5 (`ltx-2.5`, video with sound) runs 8 Euler
steps at guidance 1 on Lightricks' distilled sigma table with flow shift
2.37, at 24 fps, as WebM with its audio track: 512×512 for size `auto` (the
quick preset), and 1280×704 in two stages (the quality preset: 640×352,
the latent upscaled x2, 3 refine steps on the stage-2 table); its duration
is at most 121 frames at 24 fps aligned to 8n+1, and it takes no first
frame. Its lease asks for the measured peak of 53 GiB, and its engine's
memory limit is raised to 64 GiB. Every set takes a prompt of at most 2,048 bytes. Frames are the duration times the rate rounded up and
aligned to 4n+1 (8n+1 for LTX-2.5). Beyond these `Start` answers `unsupported_feature` with
reason `size:engine-bound` or `video:engine-bound`. A set that generates from
a first frame only answers a text-only `Start` with reason `frame:required`,
and one that generates from text only answers a first frame with reason
`frame:engine-bound`. A required extension other than
the caller claim is `unsupported_feature` with reason `extension:<key>`;
every optional extension is ignored. A `Start` repeating a `request_id` the
caller holds returns that operation's admission and reads nothing else.
Each `request_id` and its operation are held 24 hours after the operation
ends, then retired with the operation's result; `retention_ms` says so. An
operation is visible to the program that started it alone.

**[MODELHOST-V7] The first frame is the provider's own.** A `Start` naming an
`image_digest` the provider's content store does not hold is `invalid` with
reason `content:unknown`, and nothing is generated. A held frame is decoded
by the provider and re-encoded as its own PNG of the request's exact size
(INF-SD8); a frame other than PNG or JPEG within the bound is
`unsupported_feature` with reason `frame:engine-bound`.

**[MODELHOST-V8] Results, cancel and the engine's output.** The provider
reads the engine's job at a bounded interval and checks a completed result
against the request (INF-SD7) before it keeps the container in its content
store as one local delivery (INF-G3). A video fails with reason
`engine:result` for a refused result, `engine:failed` when the engine
reports failure or a cancel this provider did not make, `engine:timeout`
past the time limit (`--video-time-limit`, default 30 minutes), whose
engine is stopped, `engine:lost` when the engine stops answering, and
`engine:refused` when it refuses the request; a load that cannot happen
carries the reason of MODELHOST-H2 or V1..V5, or `engine:unavailable`. No
reason carries engine text. `Cancel` of a queued operation removes it; of a
running one it uses the engine's cancel route while the engine holds the
job queued, and otherwise stops the engine's process tree, which the next
video starts again. The engine's output is kept in
`<state>/logs/sd-server.log`, at most 8 MiB with one earlier file, and is
never forwarded to a caller; the provider reads only the step counts of
its progress bar from it (MODELHOST-V10). `--video-memory-limit` sets the
engine job's per-process memory limit: by default 40 GiB on Windows, the
measured peak commit of 1280×704×121 (32.8 GB) with about 30 percent above
it, and none elsewhere.

**[MODELHOST-V9] Video, its frames and its results share the endpoint.**
`video@1`, `content-writer@1` and `content-reader@1` answer on the
provider's own endpoint for the designated runtime and this provider's
executable alone: the runtime stages a first frame through the writer
before `Start` and reads a delivery through the reader. The runtime reaches
the provider through a declaration listing profile `video`, the contract
`abstraction.inference/video@1` and `video:<model>` entries; it offers
applications no `video@1` endpoint (inference CONTRACT.md INF-G5).

**[MODELHOST-V10] Progress comes from the engine's output.** `sd-server`
master-843 reports a job's status and queue position and no step count.
Its library prints a progress bar per weight load, per sampling step and per
decoded tile to its output, and log lines that open sampling
(`generate_video WxHxF`) and close it (`sampling completed`). The provider
reads those bars and lines in three stages: while a video generates,
`Observe` carries `step` of `steps` 100, loading and preparing the request
(weight loads, text conditioning, the first frame's encode) filling the first
10, sampling the next 60 and the decode the rest, never going back, and at
most 99 until the result is kept. A bar before the opening line is
preparation whatever it draws. A video whose engine has printed no bar yet
carries no progress.

## What it holds

**[MODELHOST-H1] A load takes a lease first.** The service MUST take a
lease using the file's size plus the provider cache recipe's RAM and checkpoint
forecast as an admission estimate before the engine starts: `Acquire` on the
declared resource — `card:0` by default — for the number of bytes the
source reports plus that forecast, waiting the declared time for holders to yield.
Every chat engine launch MUST set cache RAM and context checkpoints from the
provider's settings profile, the model's own or the provider default, and the
lease MUST count that profile's forecast. An embedding engine sets both to 0.
Engine defaults MUST NOT add unaccounted cache memory; a resident-slot profile MUST count its slots and preserve its declared context per slot.
A curated model with a context plan MUST lease the model owner's
memory for that plan (`modelservice.ContextNeed`): its weights, the KV of
its context in every resident slot, its startup residual and the margin,
plus the profile's cache forecast. A checkpoint's forecast is the profile's
fixed size plus its bytes per token of the slot's context, and the profile's
per-slot byte cap lowers the checkpoint count to fit at that context.
Its smaller launches (the planned context with the profile's smaller RAM
cache, then in fewer slots, then each smaller size in one slot) are offered
to the holder largest first, and the largest
the resource service grants without asking anybody starts; with none
granted, the smallest is acquired asking holders to yield (resource
RES-L9). The plan's capacity is the one the resource table reports for the
declared resource; the model host computes no memory figure of its own.
Reported model load bytes remain the source's file size. The runtime
asks the planned leases to yield on this provider's behalf during that
wait (RES-L2). The lease is taken first and the engine second. File size does not
bound runtime GPU memory: the loader may use more or less, and this lease is
not a physical memory quota or an assurance that the load will fit. Once
measured, actual residency above the estimate remains charged (RES-L2).

**[MODELHOST-H2] Each acquire outcome is one reason.** The service MUST
carry `unavailable` in an admission's `reason`, with `lease:not_permitted`
when no rule lets this provider's program hold the resource,
`lease:insufficient` when available capacity is below the file-size admission estimate,
`lease:holders_refused` when the holders were asked and did not free
enough, with the holders and their answers in the detail, and
`lease:ended` when the resource service ended the lease while the engine
was starting and again on the holder's one reload (RES-L4). A lease service
that did not answer, an engine that would not start and an engine that was
not ready inside the load budget are all `unavailable` with no reason
family. `chat@1` carries the reason in an admission's `reason`; `embed@1`
carries it in an `Embeddings` `reason`.

**[MODELHOST-H3] An unfinished load gives the lease back.** An engine that
would not start, and one that was not ready in time, MUST be stopped and
the lease MUST be released before its load failure is returned. A grant nobody
occupied never stands between the next asker and the bytes.

The provider owns the shared model load within its configured lease-wait
and engine-readiness budgets. Ending a caller's wait MUST NOT stop that
load. Normal and speculative waits share the holder's one engine; the
existing speculative yield rules still apply. Shutdown cancels and joins
all loads. Admission waits at most one second and answers `unavailable`,
reason `engine:loading`, while the load continues; a later call can use the
ready engine. Lease revocation, idle policy and load failure keep their
existing meaning. This also applies to `preload@1` (INF-PL3).

A finished load failure is retained per model for 30 seconds across normal
and speculative waits, so a retry receives the failure instead of acquiring
another lease. Rebuilding the provider for a declaration or pin change
clears this cache. Load start, readiness and failure records use
`abstraction-logging`, carrying family, duration and outcome word. Arguments,
raw paths and engine error text stay out of these structured records.

Chat engine stdout and stderr are retained in provider state under
`logs/llama-<model hash>.log`, bounded by the provider kit's 8 MiB writer
with one earlier file. Each launch records its effective arguments, pinned
release and preparation duration; readiness records its duration and outcome.
These local logs may contain engine input and MUST NOT be forwarded to a
caller. The per-launch API key is excluded.

**[MODELHOST-H4] The lease is renewed while loaded.** The service MUST
renew the lease while the model stays loaded, and a lease the book no
longer holds unloads the model: a residency nobody granted is not one this
provider keeps (RES-L4).

**[MODELHOST-H5] Every resident is its own engine.** Every object on the
allowlist that is asked for gets its own engine process, its own lease of
the declared resource, and its own idle clock. The first call for an object
loads it inside that call; every call after it finds it resident and only
marks it used. Two callers arriving together load the same object once,
because the load is one serialized step. A call for a second object while
the first is still resident starts a second engine on its own socket
beside it, rather than giving the first back: a call for one object never waits
on, and never displaces, another. `Unload` names the object it lets go.
After starting an engine, the provider MUST request `BindProcess` for its
engine PID and that resident's exact lease. An unavailable or unverifiable
binding does not end the load; the resource service retains the conservative
reservation and the provider reports the failed attribution through its
error channel. A binding is accounting evidence, not a claim that the engine
used exactly the file size.

**[MODELHOST-H6] A request in flight is a model in use.** A request a call
started for the model, for as long as it is open or retained for
observation, counts as used: the idle clock asks the inference provider's
own record of requests for the model (`Provider.ActiveForModel`) before it
unloads one, not only this provider's own last-load time, so an idle timer
never takes a model out from under a reply that is still arriving.

**[MODELHOST-H7] A yield request unloads its lease's resident.** This
provider implements `leases@1` `Observe` and `Answer`: it is a lease holder
and answers for itself, then the service confirms what the instrument shows
(RES-L3). A request names one lease. The provider MUST stop only the engine
holding that lease and call `Release` for that lease before answering `yielded`.
The book ends a still-live grant after confirming the instrument's fall. It MUST
answer `refused` when the named resident has an active or retained operation,
when the resource or amount differs from that resident's lease, or when the
engine cannot be confirmed stopped within its bound. Another resident's
engine and lease MUST stay in place. The provider answers `refused` when the
named lease is no longer resident; a fresh table read can then grant from
any bytes already freed.

**[MODELHOST-H8] The idle time unloads the same way, per resident.** A
resident nothing has asked for within its idle time MUST be stopped and its
lease released, on its own clock; a resident still within its idle time is
untouched by another one's clock running out. The provider stays running
and serves the next call by loading again. A resident's idle time is its
model family's engine-profile value, from the same per-family data as its
cache recipe; a family without one uses the declared `--idle` (default 5
minutes). A longer idle time changes no yield: a resident with no active or
retained operation still yields its lease on request (MODELHOST-H7), and an
idle resident holds no wake lease.

**[MODELHOST-H9] Withdrawal ends the provider.** Withdrawing the
declaration MUST end the provider and its engine. The runtime stops the
child it launched; the child stops the engine it started.
`openabstractions models unhost` of the last hosted model withdraws the
declaration, which is how a person takes the model out of the machine's
list.

**[MODELHOST-H10] Describe names the residents.** The endpoint
description MUST state, on `abstraction.inference/chat@1`, the names the
provider holds loaded now as the display capabilities `resident` (their
count) and `resident.1` to `resident.<count>`, at most 16. The runtime's
router reads them as this host's residency under router ROUTE-D14, so the
Panel and routing see a loaded model. Beside each resident with a context
plan it MUST state `resident.<n>.context`, the per-conversation context
that model's engine was started with, while the description stays within
32 capabilities; the router reports it as that model's effective context
(ROUTE-D14), exposing the loaded engine’s actual context to clients. These
display facts grant nothing.

## What it preloads

This provider serves `abstraction.inference/preload@1` (inference
INF-PL1..PL4) for its chat models: an application sends the messages and
tools its next chat request opens with, and the model's engine evaluates
them before that request arrives. llama-server reuses its cache only for an
identical token prefix, and a hybrid or sliding-window model resumes only
from a context checkpoint at or before the first token that differs.

**[MODELHOST-W1] The prefix is the caller's, and the only thing kept is
the persisted prompt cache.** The provider evaluates only the prefix the
`Preload` call carries, as the inference provider converted it for the chat
path. It learns no prefix from chat requests. It keeps no prefix after the
call and writes none anywhere, with one exception: the persisted prompt
cache below, whose slot files the provider alone writes to its own private
directory (b10825 review condition 2 still refuses a person-supplied
`--slot-save-path`). Per engine it holds only whether a request other than a
preload has run there since the engine started, and the one preload in
flight.

Amendment, 2026-11-04 (owner direction 2026-11-03, "make these features great"; `.forks/evidence/review-arch-1101/VERDICTS.md`, "Chat cache design", Layer 3): the model host keeps one bounded cache of its own, the persisted prompt cache. The provider sets `--slot-save-path` itself to an OA-owned private directory under the account's cache root; a person-supplied `--slot-save-path` stays refused. Entries are keyed by (account, app, chat binding), sealed in AEAD envelopes bound to the model digest, engine build and cache-relevant launch facts, marked no-backup, LRU-bounded within the model host's declared limit, and removed by the person's Clear and by app uninstall. A preload's prefix is still the caller's and is never learned from chat requests.

**[MODELHOST-W2] The engine evaluates the prefix the chat route renders,
as speculative work.** A preload holds the lowest standing, the
coordinator's decision of 2026-10-03 for speculative work under the open
arbitration question of VISION 2026-09-06 ("automatic prioritisation of one
process over another ... is arbitration"), which it settles for nothing else.
Its load is the holder's speculative ensure (resource RES-L6): a model
already loaded is left as it is, its idle clock untouched; a model not loaded
loads only through `leases@1`'s `AcquireFree`, which grants from what is
free and asks nobody; any failure refuses with no fallback to an ordinary
load; and a yield request for its lease cancels the load. A load the
resource refuses, or a yield cancelled, answers `busy`, reason
`resources:held`. Its load and its evaluation are never model use: the
inference provider reserves no activity for a preload, so the idle clock
and a yield request treat the engine as unused, and a yield stops it under
the preload. Then, as one step under one lock, the provider answers `busy`,
reason `engine:used`, when a request other than a preload has run on the
engine since it started, and `busy`, reason `preload:running`, when a
preload runs there; otherwise it starts the evaluation. The engine renders
the prefix with two placeholder user messages through `/apply-template` and
tokenizes each through `/tokenize` with special tokens added and parsed, as
its chat route does; the tokens the two share are sent to `/completion` as
token ids with `n_predict` 1. No template or tokenizer runs in the provider.
A prefix of fewer than 256 tokens is not evaluated and answers `loaded`. A
rendering or evaluation the engine refuses with a 4xx answers `refused`,
reason `engine:<status>`, and leaves nothing behind: the engine is not
marked used. An engine stopped by an unload, a yield, a GPU reset or an
exit while it evaluates answers `unavailable`, reason `engine:stopped`; the
caller has the outcome and nothing is reported.

**[MODELHOST-W3] No preload runs on the engine after another request.**
Every request the provider sends the engine marks it used. The provider
marks, under the same lock, the moment just before the evaluation is
posted to `/completion`:

- A request that arrives before that moment overtakes the preload: the
  preload posts nothing and answers `busy`, reason `engine:used`, and the
  request goes at once.
- A chat request that continues the prefix, its messages being the
  prefix's followed by a user message, and arrives after that moment waits
  in the provider until the evaluation has returned, then goes: the engine
  serves it from the evaluated tokens, its own prefill.
- Any other request arriving after that moment, chat, embeddings or any
  other POST, cancels the evaluation, waits until the evaluation's engine
  request has returned, and goes; the preload answers `busy`, reason
  `engine:used`.

llama-server notices the closed connection when it next polls it, every
second (`HTTP_POLLING_SECONDS`), and drops the evaluation at its next
batch. A request that cancels a preload therefore waits at most one second,
one logical batch of `--batch-size` tokens (2048 by default) and any context
checkpoint the engine is writing, beyond what it would wait on an engine
that was never asked to preload. A changed prefix costs what the engine
reuses of the shared tokens and the evaluation of the rest, plus one
prompt-cache save when the request shares less than half of the evaluated
tokens and the engine keeps a prompt cache (`--cache-ram`, set by the provider's settings profile).
An engine whose requests name their cache slot runs with
`--slot-prompt-similarity 0`: llama-server b10825 otherwise loads the best
match from its one RAM prompt cache into a named slot, and that cache holds
every partition's parked prompts. A named slot then reuses only what it
holds itself, which the slot table keeps to one partition.
A request that fails or is canceled on a slot that held nothing or only its own
partition keeps that slot's conversation and partition: the slot holds only
that partition's tokens, and the conversation's next turn reuses its prefix.
A request that fails on a slot holding tokens of another or an unknown
partition leaves the slot with no conversation and no partition.

## GPU faults

**[MODELHOST-G1] The provider applies the resource kit's GPU fault rules.**
`modelhostd serve` MUST compose one `gpufault` Guard (abstraction-resource
CONTRACT.md RES-G1..RES-G7) over its holder: on Windows it reads the
platform's reset and standby records, and on every platform it reads each
engine's output, llama-server's and sd-server's, for a device-lost line in
that engine's own form, counted only once the engine exits or fails its
readiness twice (RES-G1). The device-lost writer reads every line beside the
progress tap; neither replaces the other. A reset stops every engine this
provider runs and releases their leases (RES-G2). A video whose engine was
lost in a reset fails `gpu:reset`, and its fingerprint (the model, the model
set's name and options, the pinned sd-server, size, frames, rate, whether a
first frame conditions it, and the configured engine arguments) is
quarantined, so two sets served under one alias never share a quarantine; a `Start` of an identical video is refused `unavailable` with
`gpu:quarantined` (RES-G3). A queued video waits while the machine is in
Modern Standby or the driver is coming back; a chat or embed load then reads
`gpu:standby` or `gpu:resetting` (RES-G4, RES-G5). While a video runs the
provider holds the wake lease through leases@1 under the rule
`abstraction.resource/hold` on `awake` the runtime writes beside the card's
(MODELHOST-R1); a person who denies that rule still gets the video, and the
refusal is reported. sd-server always carries `--diffusion-fa`, the sd.cpp
dispatch row (RES-G5), OR-ed with the model set's own switches. The quarantine lives in `gpu-quarantine.json` under
`--state`; `modelhostd gpu list` prints it and `modelhostd gpu clear`
releases a paused request, read by a running provider at its next check.
Each fault is one OA logging record from `modelhostd` with its Panel line
(RES-G6).

## Engine exits

**[MODELHOST-E1] Every engine's exit is recorded.** When an engine of any
kind (llama-server, sd-server) ends, the provider MUST
write one OA logging record from `modelhostd`, `engine exit`, and one line
on its stderr, naming the engine, the model, the process id and how it
ended: stopped by modelhost, crashed with its exception status (for
example `0xC0000005 STATUS_ACCESS_VIOLATION`), killed by a signal outside
Windows, exited with its code, or ended with its exit unknown (waiting on
it or reading its code failed, and the record says why). Each engine
process records exactly one exit. The record carries the run time, the
job's peak process and job memory, the per-process memory limit and
whether that limit refused an allocation, with its evidence (abstraction-
facade CONTRACT.md FAC-P2). A crash, a signal, a refused allocation, an
unknown exit or a nonzero code is a warning; a stop or a clean exit is information. Where the
provider writes the engine's log (sd-server's),
the same line, prefixed `modelhost:`, follows the engine's last output. The
child keeps the error mode it inherits from `modelhostd`, so Windows Error
Reporting records no crash of an engine and no dialog can hold one; this
record and the engine's own crash trace are the evidence.

On Windows, `modelhostd serve --chat-memory-limit` can set a nonzero
per-process ceiling for each llama-server job. Zero leaves this optional
ceiling unset. The service owns the setting; a model's resource lease
reserves its file size for admission and is not a process memory quota
(resource RES-L1). The process supervisor records a configured limit and
its hits under MODELHOST-E1. The 48 GiB cap used for the owner's Qwen3.6
live measurement is a test choice, not a default for other machines.

## The rights

**[MODELHOST-R1] The program holds the resource.** The runtime writes
`abstraction.resource/hold` on `card:0` for the declared program of a
native inference provider with an explicit allowlist, because that program
is what loads weights into the card; applications hold nothing here. Beside
it the runtime writes `abstraction.resource/hold` on `awake`, the wake lease
the program holds while GPU work runs (MODELHOST-G1). That provider-held
wake is 0.4.0's stated provisional exception, which a wake lease held for
the requesting application replaces after research (VISION 2026-10-02). A
person who does not want a provider holding the card denies that one rule,
and every call then reads `unavailable` with reason `lease:not_permitted`
(MODELHOST-H2).

**[MODELHOST-R2] Withdrawing yield.** The per-app switch resource RES-L3
names is `abstraction.resource/yield` on this provider's program. It is on
with no rule; a deny keeps the provider's models resident when another
program asks for the memory. Withdrawing the declaration stops the
provider entirely (MODELHOST-H9).

**[MODELHOST-R3] The peer check.** Applications never reach this endpoint.
Their calls travel through the runtime, which binds the caller from the
connection's program proof, decides `abstraction.inference/complete` on
resource `server:modelhost`, and forwards. The provider listens on the
shared identity-bound transport and admits only the runtime executable beside
its installed program, including the windowless runtime on Windows.
`--runtime-program` names an explicit executable for a custom installation.
A caller it cannot bind, one below the program proof, and a bound peer from
another account or program all read `forbidden`.

**[MODELHOST-R4] No onward delegation.** This provider MUST delegate no
call onward. Its own inference provider admits every request as a local
leaf execution, so a call that reached it is answered by the engine it
started or not at all.

## Outcomes

| Call | Outcome | Reason | Meaning |
| --- | --- | --- | --- |
| `chat@1`, `embed@1` admission | `unavailable` | `object:unknown:<store>/<id>` | The allowlist names a store and an id no source resolves (MODELHOST-L1). |
| `chat@1`, `embed@1` admission | `unavailable` | `object:unbound:<store>/<id>` | The object is a member of a zip, which the pinned engine cannot read and nothing unpacks yet (MODELHOST-L1). |
| `chat@1`, `embed@1` admission | `unavailable` | `object:pickle:<store>/<id>` | The object is a PyTorch pickle, which OA never loads (MODELHOST-L1). |
| `chat@1`, `embed@1` admission | `unavailable` | `engine:none` | No engine is configured, the configured path is not on this machine, or the engine folder is not the pinned release (MODELHOST-L3). |
| `chat@1`, `embed@1` admission | `unavailable` | `engine:unreadable` | A file of the engine folder could not be opened or read, such as one a scanner holds; the next admission retries (MODELHOST-L3). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:not_permitted` | No rule lets this provider's program hold the resource (MODELHOST-H2, MODELHOST-R1). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:insufficient` | Available capacity is below the file-size admission estimate and no holder could make up the difference (MODELHOST-H2). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:holders_refused` | The holders were asked and did not free enough (MODELHOST-H2). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:ended` | The resource service ended the lease while the engine was starting, on the holder's one reload too (MODELHOST-H2, RES-L4). |
| `chat@1`, `embed@1` admission | `unavailable` | `engine:network-socket` | The engine's tree held a TCP listener or bound UDP socket, or the socket owner tables could not be read; it was stopped and its lease released (MODELHOST-L7). |
| `chat@1`, `embed@1` admission | `unavailable` | `engine:peer` | A process outside the engine's tree served the engine's socket; it received no byte, and the engine was stopped and its lease released (MODELHOST-L7). |
| `chat@1`, `embed@1` admission | `unavailable` | `engine:unreachable` | Every readiness connect to the engine's socket failed for 3 s with an error a starting engine never gives, such as the WSAEINVAL of a path in a packaged app's redirected folder; the engine was stopped and its lease released (MODELHOST-L7). |
| `chat@1` admission | `unavailable` | `content:unavailable` | The request carries an image part; this provider resolves no content digest (MODELHOST-L8). |
| `chat@1`, `embed@1` reply | `refused` | `upstream:403` | The engine request carried media the engine would fetch or decode; it never left the provider (MODELHOST-L8). |
| `chat@1`, `embed@1` reply | `refused` | `upstream:401` | The server at the engine's address refused the request's key: it is no engine this provider holds a key for (MODELHOST-L8). |
| `chat@1`, `embed@1` admission | `unavailable` | (none) | The lease service did not answer, the engine would not start, or it was not ready inside the load budget (MODELHOST-H3). |
| any call | `forbidden` | — | The caller could not be bound, was below the program proof, or is a bound peer of another account or program (MODELHOST-R3). |
| `video@1` `Start` | `exhausted` | `queue:full`, `operations:full` | 32 videos already wait, or 4,096 operations are held (MODELHOST-V6). |
| `video@1` `Start` | `invalid` | `content:unknown` | The first frame named is not in the provider's content store (MODELHOST-V7). |
| `video@1` `Start` | `unsupported_feature` | `size:engine-bound`, `video:engine-bound`, `frame:engine-bound`, `frame:required`, `extension:<key>` | The size, rate, duration, prompt or first frame is beyond the reviewed bounds, or a required extension is not supported (MODELHOST-V6, V7). |
| `video@1` `Observe` | `failed` | `engine:none`, `lease:*`, `engine:listener-not-loopback`, `engine:unavailable` | The engine folder or a placed model file is not the pin, the lease was refused, or the engine failed its load (MODELHOST-V1, V2, V5). |
| `video@1` `Observe` | `failed` | `engine:result`, `engine:failed`, `engine:timeout`, `engine:lost`, `engine:refused` | The engine's result was refused, it failed, ran past the time limit, stopped answering, or refused the request (MODELHOST-V8). |
| `chat@1`, `embed@1` admission | `unavailable` | `gpu:standby`, `gpu:resetting` | The machine is in Modern Standby, or the graphics driver reset and its adapter is not back (MODELHOST-G1). |
| `video@1` `Start` | `unavailable` | `gpu:quarantined` | An identical video reset the graphics driver and is paused (MODELHOST-G1). |
| `video@1` `Observe` | `failed` | `gpu:reset`, `gpu:quarantined`, `gpu:known-bad` | The engine was lost in a driver reset, or the video is paused or its shape is recorded as hanging the GPU (MODELHOST-G1). |

## Bounds

The service owns no bytes it does not hold under a lease and starts no
process it cannot stop by id.

## Divergences

None declared for this release.

## Not built

This provider schedules no compute: it takes bytes under a lease and gives
them back, and it never kills, boosts or prioritises anything (RES-L5). It
downloads no engine and no model. It bundles no llama.cpp,
stable-diffusion.cpp or PyTorch build and carries no engine of its own; the
OA PyTorch engine is the PyTorch provider's (VISION 2026-10-02). It writes
into no other application's directory; that is `abstraction.storage/lend@1`,
a different provider with a different promise. It links that provider's
`bridge` package (`go/host/host.go`, `go/host/engine.go`) for the source
and binding seam MODELHOST-L1 names, a provider-on-provider dependency the
kit is to own (`research/release-0.8.0/audit/providers.md` §3.4, D8). It serves no images:
stable-diffusion.cpp image generation is the Lemonade provider's, and the
PyTorch engine's is the PyTorch provider's. It makes no claim about another application's
process: an engine ComfyUI or LM Studio started holds its own copy, and
nothing here shares a loaded model with a program that did not call this
contract.
