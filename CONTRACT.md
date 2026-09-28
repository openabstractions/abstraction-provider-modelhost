# provider-modelhost contract

The contracts this provider serves are
[`abstraction.inference/chat@1`](https://github.com/openabstractions/abstraction-inference/blob/main/CONTRACT.md)
and `abstraction.inference/embed@1`, whose words bind every call; the rules
below say what this provider does around them — what it loads, what it
holds, and when it lets go. This file states the obligations a provider
meets; [README.md](README.md) holds the walkthrough. `research/architecture-review-2026-09-22.md`
R3 carries the design, `research/model-bridge/DECISION.md` §4 rules 4 and 6
the engine rule, and [`abstraction-resource`](https://github.com/openabstractions/abstraction-resource/blob/main/CONTRACT.md)
RES-L1 to RES-L4 the lease this provider holds.

The service loads one model file with an engine this machine already has,
holds its residency under a lease, serves it for as long as calls arrive,
and stops the engine and releases the lease when the resource is asked for
or the idle time passes.

Binds: none (records only)

## Reading this page

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
`object:unknown:<store>/<id>`.

**[MODELHOST-L2] The allowlist is exhaustive.** A model the declaration does
not name MUST NOT be loaded, whatever the engine could open. The router
lists exactly those names as the model server's models, so a hosted model
is visible before anything is running and no survey starts an engine to
find out what it holds (facade CONTRACT.md FAC-R2).

**[MODELHOST-L3] The engine is a configured path.** `--engine` names a
`llama-server` already installed on this machine. This layer ships no
engine, downloads none and looks for none: no engine configured, a relative
path, and a path that names no file are all `unavailable` with reason
`engine:none` (`research/model-bridge/DECISION.md` §4 rule 6).

**[MODELHOST-L4] The engine process is this provider's own.** The service
MUST start it with the model, the loopback address and the port this
provider chose, and MUST stop it by that process's own id. No process is
named by image, so no engine another program started is ever touched.

## What it holds

**[MODELHOST-H1] A load takes a lease first.** The service MUST take a
lease using the file's size as an admission estimate before the engine starts: `Acquire` on the
declared resource — `card:0` by default — for the number of bytes the
source reports, waiting the declared time for holders to yield. The runtime
yields attached holders on this provider's behalf during that wait
(RES-L2). The lease is taken first and the engine second. File size does not
bound runtime GPU memory: the loader may use more or less, and this lease is
not a physical memory quota or an assurance that the load will fit. Once
measured, actual residency above the estimate remains charged (RES-L2).

**[MODELHOST-H2] Each acquire outcome is one reason.** The service MUST
carry `unavailable` in an admission's `reason`, with `lease:not_permitted`
when no rule lets this provider's program hold the resource,
`lease:insufficient` when available capacity is below the file-size admission estimate, and
`lease:holders_refused` when the holders were asked and did not free
enough, with the holders and their answers in the detail. A lease service
that did not answer, an engine that would not start and an engine that was
not ready inside the load budget are all `unavailable` with no reason
family. `chat@1` carries the reason in an admission's `reason`; `embed@1`
carries it in an `Embeddings` `reason`.

**[MODELHOST-H3] An unfinished load gives the lease back.** An engine that
would not start, and one that was not ready in time, MUST be stopped and
the lease MUST be released before the caller is told. A grant nobody
occupied never stands between the next asker and the bytes.

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
the first is still resident starts a second engine on a second port beside
it, rather than giving the first back: a call for one object never waits
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
resident nothing has asked for within the declared idle time MUST be
stopped and its lease released, on its own clock; a resident still within
its idle time is untouched by another one's clock running out. The
provider stays running and serves the next call by loading again.

**[MODELHOST-H9] Withdrawal ends the provider.** Withdrawing the
declaration MUST end the provider and its engine. The runtime stops the
child it launched; the child stops the engine it started.
`openabstractions models unhost` of the last hosted model withdraws the
declaration, which is how a person takes the model out of the machine's
list.

## The rights

**[MODELHOST-R1] The program holds the resource.** The runtime writes
`abstraction.resource/hold` on `card:0` for the declared program of a
native inference provider with an explicit allowlist, because that program
is what loads weights into the card; applications hold nothing here. A
person who does not want a provider holding the card denies that one rule,
and every call then reads `unavailable` with reason `lease:not_permitted`
(MODELHOST-H2).

**[MODELHOST-R2] Withdrawing yield.** No `abstraction.resource/yield` rule
exists for this provider: the book decides no rule for a holder that
implements `leases@1` (RES-L3 scopes the yield rule to an attached holder,
yielded "through its host's mechanism"; this provider answers for itself).
The person's switch for whether this runtime may make the provider let go
is withdrawing the declaration (MODELHOST-H9), not a resource rule; denying
a rule that does not exist changes nothing.

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
| `chat@1`, `embed@1` admission | `unavailable` | `engine:none` | No engine is configured, or the configured path is not on this machine (MODELHOST-L3). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:not_permitted` | No rule lets this provider's program hold the resource (MODELHOST-H2, MODELHOST-R1). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:insufficient` | Available capacity is below the file-size admission estimate and no holder could make up the difference (MODELHOST-H2). |
| `chat@1`, `embed@1` admission | `unavailable` | `lease:holders_refused` | The holders were asked and did not free enough (MODELHOST-H2). |
| `chat@1`, `embed@1` admission | `unavailable` | (none) | The lease service did not answer, the engine would not start, or it was not ready inside the load budget (MODELHOST-H3). |
| any call | `forbidden` | — | The caller could not be bound, was below the program proof, or is a bound peer of another account or program (MODELHOST-R3). |

## Bounds

The service owns no bytes it does not hold under a lease and starts no
process it cannot stop by id.

## Divergences

None declared for this release.

## Not built

This provider schedules no compute: it takes bytes under a lease and gives
them back, and it never kills, boosts or prioritises anything (RES-L5). It
downloads no engine and no model, and it bundles neither. It writes into no
other application's directory — that is `abstraction.storage/lend@1`, a
different provider with a different promise. It serves chat and
embeddings; a diffusion checkpoint has one engine and belongs with the
`image@1` service work. It makes no claim about another application's
process: an engine ComfyUI or LM Studio started holds its own copy, and
nothing here shares a loaded model with a program that did not call this
contract.
