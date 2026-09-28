# abstraction-provider-modelhost

**Status: development provider for OA 0.3.0, with its first tagged release
in preparation.** The repository and optional Windows installer feature
exist in source. Build it in the OA development workspace while its dependency
versions are being prepared. The
[installation guide](https://openabstractions.org/adopt.html) describes available
release packages. Installing the model host starts no model; the operator
selects an engine and model afterward. OA's inference service also connects
directly to registered Ollama, LM Studio and Lemonade servers.

abstraction-provider-modelhost is the model host: it loads a model file into
a `llama.cpp` engine already on this machine, one engine process per model,
and keeps a loaded model **resident** — in memory and answering calls — for
as long as calls keep arriving. Holding a resident model's memory takes a
**lease**, a time-bound claim from
[`abstraction-resource`](https://github.com/openabstractions/abstraction-resource/blob/main/CONTRACT.md)
that this provider renews while the model stays loaded. When the resource is
asked for elsewhere, this provider must **yield**: give a resident and its
lease back. It also yields on its own, for a resident nothing has used in
its declared idle time. It serves
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
                 [--resource NAME] [--engine-arg ARG]... [--engine-port N]
                 [--embeddings] [--idle DURATION] [--wait DURATION]
                 [--ready DURATION] [--runtime-program PATH]
                 [--config FILE] [--home DIR] [--verbose]
modelhostd help
```

`serve` answers `Start`, `Observe` and `Cancel` on the named local endpoint
until interrupted. It accepts only the installed runtime executable beside
the provider, including the windowless executable on Windows. Pass
`--runtime-program` for a runtime in another directory. `--model` is the explicit allowlist: the name callers ask
for, and the object the storage inventory names. `--engine` is the
`llama-server` this machine already has. `--resource-endpoint` is where the
runtime serves `leases@1`; without it the installed runtime's endpoint is
read. Running `modelhostd help` (or with no arguments) prints the option
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
layer resolves an object through. The engine itself, `llama-server` from
llama.cpp, is a binary a person already installed: nothing is bundled and
nothing is downloaded.

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

The runtime writes the provider's program one rule when the declaration
lands: `abstraction.resource/hold` on `card:0`. Nothing else is granted. An
application still needs `abstraction.inference/complete` on resource
`server:modelhost` (the wire still carries `host:modelhost` until rights'
own dual reader ships, CONTRACT.md's "Reading this page"), decided by the
runtime before any call reaches here. Applications never reach this
endpoint directly: their calls travel through the runtime, which binds the
caller from the connection's program proof, decides the rule, and forwards.

## Measured

`serve`'s `TestLiveHostTheSmallestGGUFUnderALease` (`OA_LIVE_MODELHOST=1`)
builds this provider and the real `inventoryd`, declares both to an isolated
runtime with `--embeddings`, hosts the smallest GGUF the composed inventory
lists with the `llama-server` found under Lemonade's cache or the AMD
bundle, answers one chat through the runtime, answers a second off the same
load, answers one `embed@1` call straight to this provider's own endpoint
off the same load, then asks `card:0` for its whole capacity and asserts the
lease is gone. A machine with none of those engines skips and says so.

On 2026-09-22, hosting `unsloth/SmolLM2-135M-Instruct-GGUF` (88,201,792
bytes) with `llama-server --embeddings --pooling mean`: the first chat call
loaded the model in 868 ms, the second reused it in 145 ms, and one
`embed@1` call answered in 10.6 ms with 576 dimensions for 6 input tokens.
The chat model carries no pooling head of its own, so `--pooling mean` is
one of the engine's own arguments (`--engine-arg`) a person adds for a model
that is not already built for embeddings; llama-server otherwise refuses
`/v1/embeddings` on it as "pooling type 'none' is not OAI compatible"
(CONTRACT.md MODELHOST-L3).

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
lease, and a yield.

## What it is not

It is one more model server that behaves, not the only one. An engine another
application started keeps its own copy; this provider makes no claim about
what ComfyUI, LM Studio or Ollama hold, and nothing here shares a loaded
model with a program that did not call `chat@1`. It schedules no compute.

## Several residents

Version 1 places no limit on how many objects on the allowlist may be
resident together. Each gets its own engine process, on its own reserved
port, under its own lease and its own idle clock (CONTRACT.md MODELHOST-H5):
asking for a second object while the first is resident starts a second
engine rather than waiting for or replacing the first. A yield request
names an amount of the resource, not one lease, so a provider holding more
than one resident gives up whichever has gone longest untouched first,
stopping once enough is freed (MODELHOST-H7).

## Layout

- **`go/host/`** — the operation. `object.go` is the seam that resolves a
  store and an id to a file; `engine.go` starts one `llama-server`, waits
  for its own `/health` and stops it by its process id; `host.go` is
  `Ensure`, `Unload` and the loops that renew each resident's lease, answer
  a yield request and unload a resident that has gone idle, with the typed
  refusals.
- **`go/service/`** — the endpoint. It serves `chat@1`, and `embed@1` when
  the engine was started with embeddings, on the shared identity-bound
  native transport, admitting only the designated runtime program. Every admitted
  call loads the model the request names before it is mediated to the
  engine, so a load that cannot happen is the call's refusal in this
  provider's own words.
- **`go/cmd/modelhostd/`** — the one binary this layer produces.
- **`go/host/testdata/fakeengine/`** — a stand-in engine, enough of the
  OpenAI-compatible API to report readiness and answer one completion. The
  tests spawn it so they measure this provider and not llama.cpp.

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
- **It serves chat and embeddings only.** A diffusion checkpoint has one
  engine and belongs with the `image@1` service work, not yet built.

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
