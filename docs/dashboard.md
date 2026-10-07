# Dashboard

An optional single-page dashboard for one running `ninfer-serve`. It answers, at a glance, what
the engine is doing for both of the systems it serves — chat generation on `/v1` and
[System One](serving.md#system-one-decisions) decisions on `/typesafe` — current prefill and
decode rates, whether requests are queueing for a lane, which lanes and adapter slots each system
holds, where prompts and decision states are being served from, how the VRAM budget is spent, and
whether the board is the limit. It also loads a `--request-log-jsonl` file to analyze a past
session offline. A second view, the [playground](#playground) at `/playground`, builds a System One
request, runs it against the same server, and shows each answer's distribution beside what the
engine did for that decision.

The application lives in [`apps/web/`](../apps/web/) and is built with Bun, Vite, and React.

## Build and run

The Docker image builds the dashboard in its own stage and serves it from `/opt/ninfer/web`, so
nothing below is needed to use it there — `http://127.0.0.1:8080/` is the dashboard and
`http://127.0.0.1:8080/v1` is the API, on one port.

To build it outside Docker:

```bash
cd apps/web
bun install
bun run build
```

Then serve it from the engine itself, same-origin with the API:

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --web-dir apps/web/dist
```

Open `http://127.0.0.1:8080/`, or `http://127.0.0.1:8080/playground` for the playground. Registered
API routes are matched before the static mount, so the dashboard cannot shadow an endpoint; any
other path resolves to the application shell. Neither view works on a server started with
`--api-key`: the key gates every path except `/health`, the static files included, and a browser
cannot attach it to a page load.

For development against a running engine:

```bash
cd apps/web
NINFER_BASE_URL=http://127.0.0.1:8080 bun run dev
```

The dev server proxies `/telemetry`, `/events`, `/metrics`, `/slots`, `/health`, `/v1`, and
`/typesafe` to that origin, so `--cors` is not required. `NINFER_BASE_URL` is validated as an
origin and defaults to `http://127.0.0.1:8080`.

| Command | Purpose |
|---|---|
| `bun run dev` | development server on `127.0.0.1:5180` |
| `bun run build` | typecheck and emit `dist/` |
| `bun run typecheck` | types only |
| `bun test` | derivation-layer and playground tests |
| `bun run format` | Prettier write over the app sources |
| `bun run format:check` | Prettier check |

## Two systems

Chat and System One run in one process over one resident weight copy, one scheduler, one set of
lanes, one KV pool, and one adapter bank, so the dashboard shows them side by side rather than as
two services. The top bar names both surfaces and the SDK alias binding. `Chat requests` and
`Chat latency` cover generation; the System One panel covers decisions. The Slots panel's `work`
column says which system each lane serves and under which adapter, and a lane retaining a decision
state reads as System One work between decisions, because that state is what makes the next
decision on it cheap. The Throughput panel stacks System One prefill on chat prefill whenever the
record carries `tokens.decision_prefill`; the two sum to the engine's reported prefill rate, and
decisions never decode. The headline adds decisions and their p50 latency once the pool holds a
decision adapter or a log recorded a decision.

## System One

A decision is a prefill-only request: the state, then one branch per question, each read out by
the adapter's pointer head. The panel reports decisions, questions, latency p50/p90, the share of
state tokens reused, decisions in flight, and errors, then decomposes the engine time into four
phases:

- **wait** — queued for a lane in the bounded FIFO shared with chat, which is where the two systems
  contend;
- **restore** — importing a cached state from L2 or L3;
- **state** — prefilling the state tokens that no lane or tier already held;
- **branches** — the branch passes, their readouts, and the pointer head.

Latency is System One's own `latency_ms`, restore plus execution, which excludes the wait: it is
the figure the decision response carries, so the panel and a client agree. The recent-decision table
lists each decision's adapter, question and option counts, state size, reused share and source, and
branch tokens with a per-decision phase bar; hovering a timestamp shows the full record. The empty
state distinguishes a pool with no decision adapter from one that has answered no decision yet, and
a target that cannot answer decisions at all says so.

## Adapter bank

Chat adapters and decision adapters are discovered from the same `--lora-dir` and staged through
the same device slots, so the panel shows one bank with two tables. The chat table lists the base
weights and every generative adapter with its requests, generated tokens, TTFT, decode rate, reuse,
and MTP acceptance; the System One table lists every decision adapter with its decisions,
questions, latency, state reuse, and calibrated temperature, and marks the one `jev-latest` binds.

The engine reports its pool, and usage is derived from completed records, so the panel
distinguishes three states that a traffic-only view would conflate: an adapter serving requests,
an adapter in the pool but idle (servable, carrying no traffic), and an adapter that appears in a
replayed log but is not in the pool of the engine now reporting. When a source reports no inventory
at all — a pre-schema-15 log, or an older engine — the panel says pool membership is unknown rather
than claiming an adapter is absent. A log from before schema 22 has no adapter kinds; its
`decision_names` still separate the two tables, and before schema 21 every adapter is a chat
adapter.

Pool membership is not device residency. A strip above the tables shows which adapter each slot
holds right now, colored by kind, and whether a running lane pins it; beside it are the swaps so
far with their mean cost and the admissions that waited because every slot was pinned. The strip is
the visible form of the shared bank: a chat request and a decision on different adapters trade
slots exactly as two chat adapters do. A pool larger than its slot count is normal, and a swap count
that climbs with traffic says the working set of adapters exceeds `--lora-slots`.

Rows are keyed on the resolved adapter, never on the requested model id: the Anthropic route passes
the client's own model string through and falls back to the base weights when it does not resolve,
and a decision's `model` may be the SDK alias, so only the resolved name identifies what actually
ran.

The bank is one device arena committed at startup, outside the weights arena and before KV
capacity is resolved. It is reported as its own segment in the VRAM panel; without that it would
be visible only as a reduction in free memory.

## Data sources

The dashboard reads two channels because they answer different questions.

| Channel | Kind | Carries |
|---|---|---|
| `GET /telemetry` | polled at 1 Hz | levels: board sensors, scheduler occupancy, lane work, VRAM, cache fill, adapter pool and slot residency, System One binding |
| `GET /events` | SSE | history: throughput samples, board energy, completed requests, and System One decisions |

Levels cannot be reconstructed by replaying deltas, and the event stream is bounded and lossy
under backpressure, so the poll is always authoritative for current state. `/metrics` is not read
by the dashboard: every counter it needs is already in the two channels above, in a form that does
not require differencing scrapes.

Board energy is the one sensor reading that is carried in the record stream rather than only in the
poll, because it is an interval total rather than a level. The Energy panel therefore works on a
replayed log; the GPU panel, which reports levels, does not.

## Energy

The Energy panel reports what the work costs, in joules. A watt-second is a joule, so tokens per
watt-second and tokens per joule are the same figure; the panel reports joules per token, because
energy composes additively across phases while a rate does not.

Two denominators are shown because they answer different questions and neither substitutes for the
other. **Served** divides all board energy by all tokens, including the draw while idle between
requests; it is what the work actually costs and it degrades on a mostly idle server even when
nothing about the engine changed. **Active** removes the measured idle baseline and tracks the
schedule rather than the duty cycle, which is the figure to compare between two builds. On a board
that idles near 76 W the two differ by a large factor, so showing only one would be misleading.

Prefill and decode are split apart because prefill is compute-bound and draws far more power than
memory-bound decode; attributing energy in proportion to time would systematically understate
prefill. Prefill divides by computed prefill tokens, so prefix-cache hits do not flatter the
kernels, and decode divides by committed decode tokens, which under MTP counts accepted tokens
only — speculation burns compute on drafts that may be rejected, so it can raise tokens per second
and joules per token at the same time.

The board total is exact; the split is not. It is integrated from power samples the board refreshes
at roughly 50 Hz, which is coarser than a decode round, so the panel publishes the residual — the
share of measured energy the split and idle baseline together fail to explain — and marks the phase
figures once it exceeds a tenth of the total. A few percent is the expected steady state and
shrinks with window length; see `docs/performance.md`. Boards with no cumulative energy counter
report the panel as unavailable rather than as zero.

The **per 1M tokens** stat restates the served figure in watt-hours, which is the denominator
inference is priced in and the unit electricity is billed in; one multiplication by a local rate
makes it comparable to a published $/1M-token price. It is an exact rescale of joules per token by
`1e6/3600` and carries no extra information. The per-token figures stay primary because energy
composes additively across phases and a rescaled rate does not — prefill and decode joules-per-token
can be combined against their own token counts, which is the operation the panel is built around.

## Reading the charts

Two sampling semantics are deliberately drawn differently, because conflating them would assert
measurements that were never taken.

- **Throughput** is an average over each reporting interval, so a sample fills its whole interval
  as a band. The reporter skips intervals with no activity and folds the skipped time into the
  next sample's `interval_seconds`, so a wide band is idle time folded forward, not a long
  sustained rate.
- **Scheduler occupancy** (`running`, `waiting`) is a snapshot taken when the report was emitted,
  not an interval average, so it is drawn as points at their own timestamps.

Decode throughput is engine-wide: the engine sums committed tokens over every lane in a round,
so the figure rises with concurrency even when no individual request gets faster. The panel pairs
it with the mean decode batch and the per-sequence rate (aggregate divided by batch), which is what
a single client experiences. Both decode means are taken over intervals that ran a decode round
rather than the whole window, so idle time does not deflate them. Per-request rates in the Requests
table are not a partition of the aggregate — every lane in a round is charged the full round wall
time — so they do not sum to it.

There is no per-lane throughput breakdown, because the engine keeps no per-lane token or round
counter; `SlotState` publishes occupancy and each lane's work kind and adapter only.

Prefill and decode are plotted on separate rate scales over one shared time axis. On this target
prefill runs roughly an order of magnitude faster than decode, so a single linear axis renders
decode as a flat sliver; each series is therefore read against its own axis, and the interval
tooltip reports both rates so they remain directly comparable at any instant.

Every reading carries an explanatory tooltip. Panel headings, stat captions, legend entries,
stacked-bar segments, and table headers are hover and focus targets that define the term, name the
units, and say what it means when the value moves. The wording lives in one glossary module rather
than beside each panel, so a reading and its explanation cannot drift apart.

The execution-thread wall-clock split is the panel that explains contention: one mutex serializes
decode, prefill, admission, publish, and upkeep, so a second spent in any of them is a second in
which no other resident lane advances. The admission decomposition (plan / restore / commit over
the calls that entered the attempt) separates continuation import cost from scheduling cost.

TTFT is decomposed into queue, restore, and prefill. A queue-dominated TTFT means requests are
waiting for a lane rather than being computed, which more lanes or shorter generations address
before any kernel work does.

## Cache churn

The occupancy panel answers what the cache is holding. The churn panel answers whether holding it
is still paying, which is a different question and is not answered by the hit rate: a working set
that has outgrown L1 keeps hitting and simply starts paying a host or disk import on every turn.

Eviction on its own is a cache working normally, so no single count here is pathological. The
readings that carry meaning are:

- **imported reuse** — share of restores served from L2 or L3 rather than a resident L1 lane.
  Drift down the tiers is the churn signal, and the restore chart shows it directly as the tier
  mix per interval.
- **lost sessions** — retained lanes evicted with no publication ticket (`l1_evictions` minus
  `l1_demotions`). Unlike a demotion, the session survives in no tier, so the next turn of that
  conversation has no state to import and prefills from zero.
- **recomputed coverage** — prefill spent on prefix the cache demonstrably held. Preflight reports
  how deep a candidate agreed with the prompt; when a request prefilled from zero anyway,
  everything beyond what the lane already reused was recomputed for nothing. Requests where
  nothing was preflighted are excluded rather than counted as clean, because they are not evidence
  either way, and agreement the resident frontier already covered is not counted either.
- **deferrals** — restores refused for shared-KV capacity with the candidate left live for a
  retry. Reported apart from restore failures because a deferral is recoverable and a failure is
  not.
- **superseded** — publications that completed and were then discarded because the session alias
  had already advanced. The export work was paid for and nothing can ever restore from it.

Host and disk evictions are counted only when a tier exceeded its byte budget. A TTL expiry is
deliberately not counted as one: reclaiming state that went cold is the cache working, while a
capacity eviction means the tier is too small for what is actually in use. A promotion that cannot
find room is refused rather than admitted and then evicted, and a refusal is not churn either.

Both charts are interval counts, so they are drawn as bands over the window each sample covers,
and the tier series are stacked because they partition one total.

## Replay

`load jsonl` reads a `--request-log-jsonl` file and renders it through the same components. When a
file contains records from more than one server instance — it is opened in append mode — only the
last instance is kept, because mixing two configurations on one axis would misattribute every
derived figure.

A log carries request, decision, and throughput history plus the `server_start` configuration,
adapter pool, and System One binding, so throughput, latency, cache occupancy against configured
capacity, per-request and per-decision analysis, and swap and slot-wait counts are all available
offline. Board telemetry, live lane occupancy, and which adapter holds which slot are sampled,
never recorded, and those panels say so rather than showing a stale or zero reading.

## Derived analytics

The request aggregations — percentiles, prompt-source distribution, miss-reason and
restore-failure partitioning, TTFT decomposition, and restore statistics — follow the definitions
in the maintainer script `cache_health.py`, including its truncating nearest-rank percentile rule.
`src/lib/derive.test.ts` pins the agreement on a fixture whose expected values were produced by
that script, so the dashboard and the script cannot report different numbers for the same log.
Decision summaries use the same percentile rule over `latency_ms`, restore plus execution; their
phase shares divide each phase's summed seconds by the window's summed queue plus execution time,
and state reuse divides reused state tokens by all state tokens, so a long state weighs as much as
the tokens it costs.

Churn and recomputed coverage are summed from the interval deltas the throughput record already
carries, rather than by differencing the cumulative endpoints of the window, so a server restart
inside the window cannot turn a counter reset into a negative or absurd reading. A counter that a
replayed log predates reads as zero rather than as `NaN` in a total.

## Playground

`/playground` is a decision console with a **protocol** selector for
[TypeSafe System One](serving.md#system-one-decisions) (`POST /typesafe/v1/systemone`) and
[OpenAI Decisions](serving.md#openai-decisions) (`POST /v1/decisions`). Request previews, submission,
answer distributions, errors, and code snippets follow the selected protocol. Each protocol keeps
its own draft and presets; switching does not translate or discard the other draft. The selector
is disabled during a run. Switching to the dashboard and back keeps the request and any run in
flight. A run is never cancelled, because the server would log an abandoned decision as a 499.

**TypeSafe request.** The state is edited as name/value fields or as any JSON value, and the questions as
cards or as JSON. The body is serialized from the parsed JSON, never from JavaScript numbers, so
number spellings, key order, and integers beyond 2^53 reach the server as typed. The checks follow
kev's schema. What it rejects is an error that blocks Run: JSON that does not parse, no questions, a
missing or unknown type, `choice` criteria that are not an object of 1 to 255 options, `score`
criteria that are not a list of 1 to 255 levels. What it accepts but is probably a mistake is a
warning, such as a repeated key (the last one counts), an empty key or label, a question without
instructions, an empty state, or a model the server does not list. F8 steps through both, and `?`
lists the keyboard shortcuts. Token budgets are left to the server.

**OpenAI request.** Input and questions use native JSON editors, not the TypeSafe forms. Input is
a quoted text string or an array of user messages with text content. Questions are an ordered array
of `predicate`, `choice`, and `score` objects with required string instructions. The checks enforce
1–200 questions, 2–255 string/boolean choices, and 2–10 score levels; unsupported fields and images
are errors. Duplicate names and choice values are valid and retain their positions and types.
Native presets cover the billing example, mixed boolean/string choices, and text-message input.
The server checks token budgets and rejects overflow rather than truncating input. These are local
text-only decision adapters, not a promise of Luna predictions, calibration, or refusal policy.

**Models.** Each protocol reads its catalog once per page load. TypeSafe uses
`GET /typesafe/v1/models`; a new request leaves `model` out, so the server's `jev-latest` alias
answers. OpenAI uses `GET /v1/models`, filtered to models advertising `/v1/decisions`, and selects
an actual decision adapter name; it requires `model` and cannot use `jev-latest`. Its card shows
the state and context limits. If discovery fails, a model name can be entered manually.
The TypeSafe card also reports the KV codec. On a rotated codec (`rk*`) it adds a caution: decisions
there keep task-level quality but are not qualified to kev's 0.03 bar, which `bf16` is
([System One fidelity](serving.md#system-one-fidelity)).

**Answers.** The headline is the probability of the answer shown: p(true) or p(false) for `noul`,
the chosen option for `choice`, the most likely level for `score`. Below 0.60 the answer is flagged
uncertain and names its runner-up. OpenAI `predicate` shows p(yes) or p(no); its answers are numbered
by position so unnamed or repeated names remain distinct, and mixed boolean/string choices use
JSON scalar spelling. The returned `confidence` is shown under the distribution for `choice`
and `score`; `noul` and `predicate` carry none. OpenAI also shows total, cached, and cache-write
token usage without inventing a response latency field. A failed run says what its status means.
OpenAI errors preserve their message, type, code, and parameter path. kev's question union
reports a 422 once per question type, so the list is folded to the type that was sent, and each
location jumps to the text it names. The JSON view shows the raw body; the Code view gives the
current request as curl and as Python for the selected SDK (TypeSafe or OpenAI 3.26.0+).

**Engine facts.** TypeSafe runs send their own `x-typesafe-request-id`; OpenAI runs obtain the
server-issued `x-request-id` from the response. The decision's records on `/events` carry that ID,
so each run shows what the engine did for that exact decision: state tokens
reused and where from (a lane, L2, or L3) against those computed, branch passes, the lane, and the
wait/restore/state/branch split. When no record arrives the run says why: the server refused the
request while preparing it, which logs nothing; the stream dropped the record; or the dashboard is
replaying a file instead of following the live stream. OpenAI cannot join its in-flight records
until its response ID arrives. Playground runs appear in the System One panel like any other decision.

**Sharing and drafts.** Copy link puts the whole request in the URL fragment (`#r=`, base64url of
the body text), which is never sent to a server. The native `state` or `input` field selects the
protocol when opening the link. Both drafts and the selected protocol are kept in the browser,
invalid text included, and restored on the next visit; a link takes precedence over `?preset=`,
which takes precedence over that protocol's draft, and `?model=` picks the model on top of either.

The original six presets come from [laya](https://github.com/NandhaKishorM/laya)'s playground
(Apache-2.0), rewritten as System One requests, and the editor and answer views follow its design.
Six more cover every example in TypeSafe's [Advanced: structure](https://docs.typesafe.ai/primitives/advanced):
invoice extraction with structured instructions, sender-identity comparison, support-routing rubrics,
product taxonomy, PR-scope score levels, and credential-request criteria. The five complete requests
retain their state and questions; the standalone sender-identity instruction is paired with a
`ticket.sender` state using the sender from the credential-request example. Structured questions
open in JSON mode so nested instructions and criteria stay intact.

The [Noul](https://docs.typesafe.ai/primitives/noul),
[Choice](https://docs.typesafe.ai/primitives/choice), and
[Score](https://docs.typesafe.ai/primitives/score) guides supply thirteen more presets: human
escalation, resume deduplication, programming language, meeting type, product category, shoe-exchange
routing, five-question shoe triage, return policy versus status, bug severity, three-question bug
triage, structured severity levels, outfit formality, and candidate fit. The complete requests retain
their state and questions; the three standalone Choice questions have sample inputs added. The Score
explorer's severity, frustration, and report-detail questions are already covered by the bug presets.
