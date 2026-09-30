# Dashboard

An optional single-page dashboard for one running `ninfer-serve`. It answers, at a glance, what
the engine is doing for both of the systems it serves — chat generation on `/v1` and
[System One](serving.md#system-one-decisions) decisions on `/typesafe` — current prefill and
decode rates, whether requests are queueing for a lane, which lanes and adapter slots each system
holds, where prompts and decision states are being served from, how the VRAM budget is spent, and
whether the board is the limit. It also loads a `--request-log-jsonl` file to analyze a past
session offline.

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

Open `http://127.0.0.1:8080/`. Registered API routes are matched before the static mount, so the
dashboard cannot shadow an endpoint; any other path resolves to the application shell.

For development against a running engine:

```bash
cd apps/web
NINFER_BASE_URL=http://127.0.0.1:8080 bun run dev
```

The dev server proxies `/telemetry`, `/events`, `/metrics`, `/slots`, `/health`, and `/v1` to that
origin, so `--cors` is not required. `NINFER_BASE_URL` is validated as an origin and defaults to
`http://127.0.0.1:8080`.

| Command | Purpose |
|---|---|
| `bun run dev` | development server on `127.0.0.1:5180` |
| `bun run build` | typecheck and emit `dist/` |
| `bun run typecheck` | types only |
| `bun test` | derivation-layer tests |
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
