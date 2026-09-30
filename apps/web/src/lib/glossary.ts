// Definitions for every term the dashboard shows.
//
// One place so a reading and its explanation cannot drift, and so the wording stays specific to
// this engine rather than generic inference vocabulary. Each entry says what the number measures,
// and where it is useful, what to conclude when it moves.

export interface Definition {
  title: string
  body: string
}

export const GLOSSARY = {
  // --- rates -------------------------------------------------------------------------------
  decodeRate: {
    title: 'Decode tokens/s (aggregate)',
    body: 'Tokens committed by decode rounds over the last interval, summed across every active lane — engine-wide, not per request. It rises with concurrency even when no individual request gets faster. The first token comes from prefill and is excluded; with MTP, accepted tokens count and rejected drafts do not.',
  },
  perSequenceRate: {
    title: 'Per-sequence decode rate',
    body: 'Aggregate decode divided by the mean decode batch: the rate one request experiences, because a sequence in a batch of b receives 1/b of what those rounds committed. This is the number to compare against a single-user benchmark; the aggregate is the number to compare against total server capacity.',
  },
  prefillRate: {
    title: 'Prefill tokens/s',
    body: 'Prompt tokens actually evaluated over the last interval. Tokens served from a reused prefix are excluded, so this measures real compute, not prompt size. Prefill and decode share one execution thread, so a high prefill rate usually coincides with a depressed decode rate.',
  },
  decodeBatch: {
    title: 'Average decode batch',
    body: 'Mean sequences advanced per decode round over the interval (round rows ÷ rounds). The engine forms one compact batch at every round boundary, so this rises toward the lane count when several requests decode concurrently. It is the multiplier between per-sequence and aggregate throughput: a value near 1 under load means requests are serialized behind prefill rather than batching.',
  },

  // --- scheduling --------------------------------------------------------------------------
  lanes: {
    title: 'Lanes',
    body: 'Running requests against --max-concurrency. A lane is a resident execution slot with its own KV pages. The count is fixed at startup between 1 and 8; there is no preemption, so an occupied lane is held until its request finishes.',
  },
  queued: {
    title: 'Queued requests',
    body: 'Requests admitted to the bounded FIFO but not yet given a lane. Ingress is bounded by --max-pending-requests; beyond that the server returns 429. A request that waits past --pending-timeout-ms is rejected with 503.',
  },
  meanQueue: {
    title: 'Mean queue delay',
    body: 'Cumulative time requests spent between submission and admission, divided by the number of admissions. This is the term that dominates time-to-first-token once the cache is working, and it responds to lane count and generation length, not to kernel speed.',
  },
  rejected: {
    title: 'Rejected requests',
    body: 'Requests refused before execution: the FIFO was full (429) or the admission deadline expired while waiting (503). A rejected request produces no request-log record, so this counter is the only place it appears.',
  },
  laneOccupancy: {
    title: 'Lane occupancy',
    body: 'Running requests as a fraction of the configured lane count. Sustained saturation with a non-empty queue means throughput is admission-bound, not compute-bound.',
  },
  ingressQueue: {
    title: 'Ingress queue',
    body: 'Waiting requests as a fraction of --max-pending-requests. Reaching the limit means new requests are being rejected with 429.',
  },

  // --- worker ------------------------------------------------------------------------------
  workerSplit: {
    title: 'Execution thread wall clock',
    body: 'How the single execution thread spent its time. Every unit runs behind one mutex, so a second spent in any of them is a second in which no other resident lane advances. Prefill share rising against decode is what starves a decoding request.',
  },
  workerDecode: {
    title: 'Decode',
    body: 'Time in decode rounds. Rounds are short, so a high share with a low decode rate means many small rounds rather than batched ones.',
  },
  workerPrefill: {
    title: 'Prefill',
    body: 'Time evaluating prompt chunks, sized by --prefill-chunk. Chunks are long relative to decode rounds, so prefill is the usual source of decode stalls.',
  },
  workerAdmission: {
    title: 'Admission',
    body: 'Time admitting requests to lanes, including continuation import. A large share here is cache restore work on the critical path, not scheduling overhead.',
  },
  workerPublish: {
    title: 'Publish',
    body: 'Time publishing continuation images so a later request can reuse the session. This is what populates L2 and queues L3 writes.',
  },
  workerUpkeep: {
    title: 'Upkeep',
    body: 'Periodic maintenance on the execution thread. Normally negligible.',
  },
  admissionSplit: {
    title: 'Admission decomposition',
    body: 'Admission time split into planning, continuation restore, and commit, over the scheduler iterations that entered the attempt. Most calls admit nothing, so the per-call figure is what determines how much of the execution thread admission consumes.',
  },

  // --- latency -----------------------------------------------------------------------------
  ttft: {
    title: 'Time to first token',
    body: 'Submission to first emitted token: queue wait, then continuation restore, then prefill. Reported from the engine, not from HTTP round-trip timing, so it excludes prompt preparation and vision encode.',
  },
  ttftSplit: {
    title: 'TTFT decomposition',
    body: 'Share of summed TTFT spent queueing for a lane, restoring cached state, and prefilling. Queue-dominated means requests are waiting, not computing — more lanes or shorter generations move it before any kernel work does. Prefill-dominated means prompts are genuinely being recomputed; check the prompt-source split.',
  },
  percentiles: {
    title: 'Percentiles',
    body: 'Nearest-rank over the retained request window, matching the maintainer cache_health.py rule. p50 is typical, p90 and p99 show the tail that a client actually notices.',
  },
  decodePerRequest: {
    title: 'Per-request decode rate',
    body: 'Generated tokens divided by that request’s own decode seconds. Not a partition of the aggregate: every lane in a round is charged the full round wall time, so these do not sum to the engine-wide rate. Read it as the speed that request experienced.',
  },

  // --- speculation -------------------------------------------------------------------------
  mtpAccept: {
    title: 'MTP acceptance',
    body: 'Fraction of speculatively drafted tokens the verifier accepted. Each accepted token is one the model did not have to decode serially, so acceptance translates almost directly into decode throughput. Rejected drafts cost the draft work but never change output.',
  },

  // --- cache -------------------------------------------------------------------------------
  prefillAvoided: {
    title: 'Prefill avoided',
    body: 'Share of prompt tokens across completed requests that never reached prefill because a resident or restored prefix already covered them. The single clearest measure of whether the continuation cache is doing its job.',
  },
  promptSource: {
    title: 'Prompt source',
    body: 'Which tier served each prompt. L1 is a session still resident in VRAM and costs nothing to reuse. L2 is a host-RAM image copied back to the device. L3 is read from disk. None means the prompt was prefilled from zero.',
  },
  tierL1: {
    title: 'L1 — resident VRAM',
    body: 'Sessions still held in a lane’s KV pages. Reuse requires no import at all, so an L1 hit is the cheapest possible path. Capacity is --continuation-cache-l1-mib; pressure shows up as evictions and demotions.',
  },
  tierL2: {
    title: 'L2 — host RAM',
    body: 'Continuation images in host memory. A hit costs a host-to-device transfer, which is fast but on the critical path of admission. Capacity is --continuation-cache-l2-mib.',
  },
  tierL3: {
    title: 'L3 — disk',
    body: 'Content-addressed images persisted under --continuation-cache-dir. A hit costs a disk read plus transfer, still far cheaper than re-prefilling a long prompt. Capacity is --continuation-cache-l3-mib; at capacity, older entries are evicted rather than new ones refused.',
  },
  catalogHit: {
    title: 'Catalog hit rate',
    body: 'Hit rate of the alias catalog lookup, which is the L2 and L3 path only. A prompt matched against a resident L1 lane never consults the catalog, so an all-L1 workload reads 0% here while still avoiding nearly all prefill.',
  },
  restores: {
    title: 'Restores',
    body: 'Imports that successfully brought cached state into a lane, across all tiers. Counted only when the restore was useful — deeper than the lane’s existing frontier.',
  },
  restoreFails: {
    title: 'Restore failures',
    body: 'Imports that were attempted and failed, attributed by cause: KV reservation exhausted, verification depth disagreement, segment inventory mismatch, metadata mismatch, decode or transfer error, or no lane able to accept the import. Failures fall back to prefill, so they cost latency but never correctness.',
  },
  missReason: {
    title: 'Miss reasons',
    body: 'Why a prompt ended up prefilling from zero. not_attempted means no candidate was evaluated (usually a cold alias); no_lane means nothing was free to import into; restore_failed means an import was tried and failed; not_deeper means the candidate added nothing over the lane’s current state.',
  },
  kvGrowth: {
    title: 'On-demand KV growth',
    body: 'A request reserves a bounded decode window at admission and acquires the rest as it generates. Attempts counts round boundaries that asked for more pages; forced spills counts retained sessions demoted to make room; curtailed counts requests that ended early at length because neither rung found pages.',
  },
  kvCurtailed: {
    title: 'Curtailed requests',
    body: 'Requests that stopped generating early because no KV pages could be found, reported to the client as finish reason length. Any non-zero value means output was truncated by capacity, not by the model.',
  },
  evictions: {
    title: 'L1 evictions and demotions',
    body: 'Retained lanes destroyed outright, versus demoted into L2 or L3 so their session survives. Demotions preserve reuse; evictions do not.',
  },
  churn: {
    title: 'Cache churn',
    body: 'Sessions losing residency and reuse being re-imported, per reporting interval. Eviction alone is a cache working normally, so read these against the restore mix rather than on their own.',
  },
  lostSessions: {
    title: 'Lost sessions',
    body: 'Retained lanes evicted with no publication ticket, so the session survives in no tier. Unlike a demotion, the next turn of that conversation has no state to import and must prefill from zero.',
  },
  importShare: {
    title: 'Imported reuse',
    body: 'Share of restores served from L2 host memory or L3 disk instead of a resident L1 lane. A working set that has outgrown L1 keeps its hit rate but starts paying an import on every turn, and this is the reading where that shows up.',
  },
  tierEvictions: {
    title: 'Host and disk evictions',
    body: 'Entries pushed out of L2 or L3 because the live working set exceeded that tier’s byte budget. A TTL expiry is deliberately not counted: reclaiming state that went cold is the cache working, while a capacity eviction means the tier is too small for what is actually in use.',
  },
  coverageWaste: {
    title: 'Recomputed coverage',
    body: 'Prefill spent on prefix the cache demonstrably held. Preflight reported how deep a candidate agreed with the prompt; if the request still prefilled from zero, everything beyond what the lane already reused was recomputed for nothing. Requests where nothing was preflighted are excluded, because there is no evidence either way.',
  },
  deferrals: {
    title: 'Restore deferrals',
    body: 'Restores refused for shared-KV capacity with the candidate left live, so the request can retry once pages could exist. Counted apart from restore failures because a deferral is recoverable and a failure is not.',
  },
  superseded: {
    title: 'Superseded publications',
    body: 'Publications that completed and were then discarded because the session alias had already advanced past them. The export work was done and paid for, and nothing can ever restore from it.',
  },
  reusePath: {
    title: 'Prefix reuse path',
    body: 'restore_turn_checkpoint means the prompt diverged mid-history and resumed from the nearest retained turn checkpoint. full_reset means the lane started from zero.',
  },
  sessionDigest: {
    title: 'Session digest',
    body: 'Stable identifier of a lane’s exact resident token ledger. Equal digests mean the identical session. Opaque to clients, but usable as an if_digest precondition on slot operations.',
  },
  checkpoints: {
    title: 'Turn checkpoints',
    body: 'Retained host snapshots a diverging prompt can rewind to, sized by --turn-checkpoints. Each holds a full state image, so they trade host memory for avoided re-prefill.',
  },
  slotReused: {
    title: 'Reused share',
    body: 'Portion of the lane’s prompt served from resident prefix rather than recomputed.',
  },
  retainedLane: {
    title: 'Retained lane',
    body: 'An idle lane still holding a resident session in VRAM. A matching prompt reuses it with no import at all; a non-matching one evicts or demotes it.',
  },

  // --- System One --------------------------------------------------------------------------
  systemOne: {
    title: 'System One',
    body: 'TypeSafe’s classification API, served under /typesafe with kev’s semantics. A decision is one state and a set of typed questions: the engine prefills the state once, then one branch per question that sees the state and itself, and a pointer head turns the readouts at each option into calibrated probabilities. It is prefill-only — it never samples, decodes or drafts — and runs on the same weights, lanes, KV pool and adapter bank as chat.',
  },
  decisionLatency: {
    title: 'Decision latency',
    body: 'What System One reports as latency_ms: the restore of a cached state plus execution, from admission to the probabilities. The wait for a lane is excluded, as it is in kev. Total is receipt to response as the client saw it, including that wait and request preparation.',
  },
  decisionPhases: {
    title: 'Decision phases',
    body: 'Summed engine time split into waiting for a lane, restoring a cached state from host memory or disk, prefilling the state, and the branch passes with their readout. State-dominated means states are being computed rather than reused; wait-dominated means the lanes are busy with chat or other decisions.',
  },
  stateReuse: {
    title: 'State reuse',
    body: 'Share of state tokens that were not prefilled because a lane still held the state (L1) or a tier restored it byte for byte (L2, L3). New questions over a known state cost only their branches.',
  },
  branchPasses: {
    title: 'Branch passes',
    body: 'Questions are packed first-fit into passes of at most --prefill-chunk columns, and each pass streams the weights once. Branch tokens per second is the rate of that phase, pooled over the window.',
  },
  decisionAdapter: {
    title: 'Decision adapter',
    body: 'A LoRA adapter plus a pointer head, converted with --decision-head. It is a System One model named by its file stem, selected on /typesafe and refused on the chat routes, just as a chat adapter is refused on /typesafe.',
  },
  chatAdapter: {
    title: 'Chat adapter',
    body: 'A generative LoRA adapter, selected on /v1 and the Anthropic route as <model>-<name>. Requests without one run on the base weights.',
  },
  sdkAlias: {
    title: 'SDK alias',
    body: 'The model name the TypeSafe SDK sends by default. It resolves to one decision adapter: the one named by --systemone-default, or the only decision adapter when the pool holds exactly one.',
  },
  decisionPrefill: {
    title: 'System One prefill',
    body: 'Decision state and branch tokens evaluated over the interval — the System One part of the prefill rate. Chat prefill is the rest. Both share one execution thread, so a burst of cold states shows up as slower chat decode in the same interval.',
  },
  engineFacts: {
    title: 'Engine facts',
    body: 'What the engine did for this exact decision, read from its decision_start and decision_done or decision_error records on /events. The playground sends its own x-typesafe-request-id with every run and the records carry it, so the match is exact. The stream drops records under backpressure and while it reconnects, and a request refused while it is prepared is never logged; the playground says which of those happened.',
  },
  answerProbability: {
    title: 'Probability and confidence',
    body: 'The headline is the probability of the answer shown: p(true) or p(false) for noul, the chosen option for choice, the most likely level for score. Below 0.60 the answer is flagged uncertain with its runner-up. Choice and score answers also carry kev’s confidence, a spread measure that is 0 at a uniform distribution; noul answers carry none. Probabilities arrive rounded to four decimals.',
  },
  kvCodecCaution: {
    title: 'Rotated KV codec',
    body: 'Decisions are qualified to kev’s bar (every probability within 0.03 of the evaluation path) on bf16 KV, which int8 also met. The Hadamard-rotated codecs keep task-level quality but miss that bar on a few questions, by at most 0.018 beyond it. The fidelity table is in docs/serving.md, System One fidelity.',
  },
  laneWork: {
    title: 'Lane work',
    body: 'What a lane is running — a chat generation or a System One decision — and under which adapter. An idle lane keeps the kind of state it retains: a chat session or a decision state, reusable only by the next request of the same kind on the same adapter.',
  },

  // --- adapters ----------------------------------------------------------------------------
  adapterBank: {
    title: 'Adapter bank',
    body: 'A fixed number of device slots in one arena, drawn from an unbounded pool discovered from --lora-dir. Chat and decision adapters share the pool and the slots. Every slot shares one rank and one site geometry, so both are kernel constants; a narrower or lower-rank adapter is zero-padded into them, which contributes nothing. Selection is a per-row index at execution time, and the engine stages an adapter into an empty or least recently used unpinned slot at admission — the slot address never moves, so a swap is invisible to the captured graph.',
  },
  adapterVram: {
    title: 'Adapter VRAM',
    body: 'Slots times one adapter slab. The bank lives outside the weights arena and is committed before KV capacity is resolved, so it silently reduces the KV budget. Pool size does not appear here: only resident slots cost VRAM.',
  },
  adapterUsage: {
    title: 'Per-adapter usage',
    body: 'Completed requests grouped by the adapter that actually served them. A pooled adapter with no rows costs disk and a directory entry, not VRAM — only the resident slots are committed.',
  },
  bankSlot: {
    title: 'Bank slot',
    body: 'One device-resident adapter slab and its current occupant. Colour gives the occupant’s kind; an empty slot has never been staged.',
  },
  pinned: {
    title: 'Pinned',
    body: 'A running lane — prefilling, decoding or deciding — executes against this slot’s occupant, so admission cannot swap it out. A lane that merely retains state under it does not pin it: that state is released before the swap, demoted to L2 or L3 where those tiers are on.',
  },
  swaps: {
    title: 'Adapter swaps',
    body: 'Adapters staged into a slot since startup, and the execution-thread time they took: reading and verifying the adapter file, demoting retained lanes that depended on the old occupant, and the upload.',
  },
  slotWaits: {
    title: 'Slot waits',
    body: 'Requests whose admission waited at least once because every slot was pinned by a running lane on another adapter. Non-zero under load means --lora-slots is smaller than the set of adapters in concurrent use.',
  },

  // --- GPU ---------------------------------------------------------------------------------
  gpuUtil: {
    title: 'SM utilization',
    body: 'Fraction of the sampling window in which at least one kernel was resident, from NVML. It says the GPU was busy, not that it was efficient — a memory-bound decode can report 100% while far from peak throughput.',
  },
  gpuMemBw: {
    title: 'Memory bandwidth utilization',
    body: 'Fraction of the window in which device memory was being read or written. Decode of a large model is memory-bound, so this sitting near 100% during decode is expected.',
  },
  gpuTemp: {
    title: 'Temperature',
    body: 'Core temperature. A 24 GB RTX 4090 under sustained prefill approaches its thermal limit and will reduce clocks; that is the documented way a prefill measurement gets misread as a kernel regression.',
  },
  gpuPower: {
    title: 'Power draw',
    body: 'Board draw against the enforced limit, averaged by the driver over a trailing second. Sitting at the limit means clocks are being capped by power, which appears as sw_power_cap in the throttle reasons.',
  },

  // --- energy ------------------------------------------------------------------------------
  energyServed: {
    title: 'Served energy per token',
    body: 'Every joule the board drew over the window divided by every token it produced, including the draw while idle between requests. This is what the work costs, so it gets worse on a mostly idle server even when nothing about the engine changed. A watt-second is a joule, so tokens per watt-second and tokens per joule are the same figure; energy is reported per token because it composes additively across phases and a rate does not.',
  },
  energyActive: {
    title: 'Active energy per token',
    body: 'Board energy with the measured idle baseline removed, divided by tokens produced. This tracks the schedule rather than the duty cycle, so it is the figure to compare between two builds or two settings. It ignores what idling costs, which is why it is reported alongside served rather than instead of it.',
  },
  energyPrefill: {
    title: 'Prefill energy per token',
    body: 'Energy charged to prefill units divided by tokens actually prefilled — prefix-cache hits are excluded from the denominator, so restored tokens do not make this look better than the kernels are. Prefill is compute-bound and draws far more power than decode, which is why time-proportional attribution would be wrong and each phase is bracketed separately.',
  },
  energyDecode: {
    title: 'Decode energy per token',
    body: 'Energy charged to decode rounds divided by committed decode tokens. With MTP the denominator counts accepted tokens only, so this is where speculation is revealed as a net energy win or loss: drafting burns compute for tokens that may be rejected, which can raise tokens per second and energy per token at the same time.',
  },
  energyIdle: {
    title: 'Idle draw',
    body: 'Board draw measured over intervals in which no execution unit ran and nothing was queued. It is measured, not assumed, and it is what the unclaimed part of an interval is priced at. A high idle baseline makes served energy per token mostly a statement about duty cycle.',
  },
  energyPerMillion: {
    title: 'Energy per million tokens',
    body: 'The served figure restated in the denominator inference is priced in, in watt-hours. It is the same measurement as joules per token scaled by 1e6/3600 and carries no extra information; what it buys is a shared denominator. A watt-hour is the unit electricity is billed in and a million tokens is the unit inference is sold in, so multiplying this by a local electricity rate gives a number directly comparable to a published $/1M-token price. The per-token figures stay primary above because energy composes additively across phases and this restatement does not.',
  },
  energyResidual: {
    title: 'Energy split residual',
    body: 'Share of measured board energy that the prefill, decode, and idle figures together fail to explain. The total is exact — it comes from the board’s own counter — but the split is integrated from power samples the board refreshes at roughly 50 Hz, which is slower than a decode round. A large residual means read the split as indicative and trust only the total.',
  },
  gpuClock: {
    title: 'SM clock',
    body: 'Current shader clock against the board maximum. A sustained gap with an active throttle reason explains a throughput drop that is not a code change.',
  },
  gpuThrottle: {
    title: 'Clock throttle reasons',
    body: 'Active NVML reasons the clocks are limited. gpu_idle and applications_clocks_setting are not faults and are not flagged. sw_power_cap, hw_slowdown, and the thermal reasons mean measured throughput is being limited by the board, not the implementation.',
  },
  boardMemory: {
    title: 'Board memory',
    body: 'Total VRAM in use on the device from NVML, including this process and anything else resident. Compare against the engine’s own arena budget to see whether another process is competing.',
  },

  // --- memory ------------------------------------------------------------------------------
  vramBudget: {
    title: 'VRAM budget',
    body: 'How the engine divided the board at startup. Arenas are reserved up front, so capacity — not current use — is what the device actually holds. Free is what remained after weights, KV, workspace, graph allowances, and any adapter bank. These are the engine’s own arenas; the board figure above also includes the CUDA context and any other process, so the two are not expected to match exactly.',
  },
  weightsArena: {
    title: 'Weights',
    body: 'Model weights uploaded once at load, in their registered storage format.',
  },
  kvArena: {
    title: 'KV arena',
    body: 'Reservation backing the paged KV cache and per-sequence state, sized from --kv-capacity and the selected --kv-dtype codec.',
  },
  kvPayload: {
    title: 'KV payload',
    body: 'Bytes of the KV arena currently holding real cache content, against the arena reserved for it.',
  },
  workspaceArena: {
    title: 'Workspace',
    body: 'Transient scratch for kernels. Peak use against capacity shows how much of the reservation is actually needed.',
  },
  cudaGraphs: {
    title: 'CUDA graphs',
    body: 'Memory captured graphs occupy against the allowance planned for them. Graphs require stable device addresses, which is why the allowance is reserved rather than allocated on demand.',
  },
  pageGroups: {
    title: 'KV page groups',
    body: 'Allocated page groups against the maximum the arena can address. This is the real capacity ceiling for concurrent context.',
  },
  textKv: {
    title: 'Text KV',
    body: 'KV cache for the main attention path.',
  },
  gdnState: {
    title: 'GDN state',
    body: 'Per-sequence recurrent state for the gated delta-net layers, held in FP32.',
  },
  mtpKv: {
    title: 'MTP KV',
    body: 'KV cache for the speculative draft head.',
  },

  // --- reading the charts ------------------------------------------------------------------
  intervalBands: {
    title: 'Why bars have different widths',
    body: 'Each bar covers its own reporting interval. The reporter skips intervals with no activity and folds the skipped time into the next sample, so a wide bar is idle time folded forward, not a long sustained rate.',
  },
  snapshotSeries: {
    title: 'Snapshots, not interval means',
    body: 'Lane and queue counts are read at the instant each report is emitted, unlike throughput which is averaged over the interval. They are drawn as points at their own timestamps so the chart does not claim an occupancy that was never measured.',
  },
  replayMode: {
    title: 'Replay',
    body: 'Showing a loaded request log instead of the live engine. Throughput, latency, per-request detail, System One decisions, and cache occupancy against configured capacity all come from the file. Board telemetry, live lane occupancy and bank residency are sampled, never recorded, so those readings stay empty.',
  },
  eventStream: {
    title: 'Live data',
    body: 'Levels come from GET /telemetry polled once a second; history comes from the GET /events record stream. The stream is bounded and drops its oldest records under backpressure rather than stalling the engine, so the poll is authoritative for current state.',
  },
} as const satisfies Record<string, Definition>

export type GlossaryKey = keyof typeof GLOSSARY
