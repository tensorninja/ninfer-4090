# Tiered continuation cache

NInfer can retain complete Qwen continuation state in VRAM, host RAM, and local storage. A
continuation contains more than attention KV: it includes the recurrent Gated DeltaNet state,
continuation hidden state, exact prepared-prefix identity, and the enabled MTP or DFlash state.
The cache stores model-prefix execution state, not generated results or HTTP response objects.

The implementation supports the Qwen3.8 27B and 35B-A3B runtime targets. Real-model cache parity depends on
the artifact, target, backend, KV codec, and media profile being qualified; absence of a published
parity result is not evidence that two configurations can share entries.

## Enable the cache

Both `ninfer` and `ninfer-serve` accept the same cache options:

```text
--continuation-cache off|l1|l1-l2|l1-l2-l3
--continuation-cache-policy adaptive
--continuation-cache-dir PATH
--continuation-cache-namespace local
--continuation-cache-l1-mib 768
--continuation-cache-l2-mib 16384
--continuation-cache-l3-mib 49152
--continuation-cache-l1-idle-seconds 600
--continuation-cache-l2-idle-seconds 7200
--continuation-cache-l3-ttl-seconds 86400
--continuation-cache-persist-interval-seconds 60
--continuation-cache-persist-min-tokens 8192
--continuation-cache-filesystem-reserve-mib 0
--prefix-checkpoint-history 4
```

The serve and public Engine default is `l1-l2`; the native one-shot CLI defaults to `off`.
Supplying `--continuation-cache-dir` without an explicit tier selection
changes the default to `l1-l2-l3`. Explicit `l1-l2-l3` requires a nonempty directory. `adaptive` is
the only implemented policy.

| Tier | Implemented meaning |
|---|---|
| `off` | disables continuation-tier retention; ordinary compatible resident-prefix reuse remains controlled separately by the server's `--no-prefix-reuse` flag |
| `l1` | retains completed lane state in the existing GPU KV pool, bounded by the L1 byte budget and idle timer |
| `l1-l2` | adds byte-budgeted complete continuation images in pageable host RAM and session/prompt-boundary aliases |
| `l1-l2-l3` | adds asynchronous, restart-persistent, content-addressed files under `PATH/NAMESPACE` |

Active request state is not a cache tier. It remains protected by normal admission accounting and
cannot be evicted to admit cache data.

## Current semantics

### L1

L1 consists of retained GPU lanes, not a separate pool of shareable GPU pages. The default
`--continuation-cache-l1-mib 768` is an eviction threshold over retained-lane resident bytes;
`--continuation-cache-l1-idle-seconds 600` evicts an idle retained lane after ten minutes. `0`
means no idle expiry. L1 uses the startup-fixed shared KV pool,
so it does not increase context capacity and may be evicted when an active request needs the lane.
When L2 is enabled, an evicted routed session is demoted asynchronously when possible.

### L2

L2 stores pointer-free complete images in host RAM. The default budget is 16,384 MiB. Admission and
eviction are byte-based; an image larger than the budget is not retained. Selection favors images
with greater recomputation value per byte, with recency used in the score. Restoring allocates
private runtime state and copies the image back to the selected lane.

`--continuation-cache-l2-idle-seconds 7200` expires an image after that many seconds without an L2
access. An L2 hit refreshes only the L2 idle time. `0` means no idle expiry. L2 and L3 expiration are
independent in `l1-l2-l3` mode, so expiration in one tier does not remove a still-valid copy in the
other. Pressure may evict unpinned data before its idle TTL.

### L3

L3 uses `--continuation-cache-dir PATH` plus the safe single-component
`--continuation-cache-namespace NAME`; the default namespace is `local`. For example,
`--continuation-cache-dir /var/cache/ninfer` stores the default namespace below
`/var/cache/ninfer/local/`, with `chunks`, `manifests`, `aliases`, `tmp`, and private artifact
fingerprint data under `artifacts`.
Namespaces may contain ASCII letters, digits, `.`, `_`, and `-`, but may not be `.` or `..`.

The default L3 budget is 49,152 MiB and its idle TTL is 86,400 seconds; `0` means no expiry. An L3
restore refreshes its own expiry and opportunistically republishes that manifest metadata. It does
not refresh L3 merely because the same content was read from L2. The implementation writes
immutable 4 MiB SHA-256-addressed chunks, a manifest, and an alias file using temporary files,
`fsync`, and atomic rename. Shared chunks are charged once. It verifies chunk and complete-image
SHA-256 values on restore. Missing, expired, malformed, incompatible, or corrupt data becomes a
safe miss; generation continues from another legal prefix or cold prefill.

`--continuation-cache-filesystem-reserve-mib N` refuses a new durable image if writing it would
leave less than `N` MiB available. Its default is `0`; set a nonzero production reserve because the
L3 byte budget alone does not protect unrelated filesystems.

The cache creates its directories as `0700` and files as `0600` on Linux. The server user must own
or be able to create and secure the namespace path. Keep the same writable volume mounted across
restarts. A read-only path, ownership mismatch, exhausted quota, or insufficient reserve causes
persistence failures without invalidating successful inference or usable L2 state.

One cache root is currently intended for one NInfer process at a time. Atomic files prevent torn
publication, but there is no implemented multi-process catalog/root coordination. Give concurrent
servers separate namespaces or directories.

## Persistence and history

Completed requests with a `prompt_cache_key` publish their latest complete continuation under that
session alias. Stable leading prefixes are published under internal immutable aliases. L3 work is
asynchronous and coalesces repeated updates to the newest head for each alias.

Persistence becomes due when either condition is met:

- the frontier has grown by `--continuation-cache-persist-min-tokens` since the prior durable head;
- `--continuation-cache-persist-interval-seconds` has elapsed since the prior durable head or the
  initial queue time.

Defaults are 8,192 tokens or 60 seconds. An interval of `0` disables only the timer trigger; token
growth still triggers persistence, and orderly shutdown flushes the latest queued publication. A
minimum token growth of `0` makes every publication due, including when the interval is also `0`.
Counters expose queued, coalesced, successful, and failed operations.

`--prefix-checkpoint-history N` is the bounded number of IDs retained by each mutable session alias,
including its current head; the default `4` therefore keeps the current head and up to three older
heads. The cache library implements destructive rollback of this alias history, but no public CLI or
HTTP rollback operation is currently exposed. Stable-prefix aliases are immutable and have no
history.

## Identity and routing safety

`prompt_cache_key` is a routing hint, not authorization to reuse state. Chat Completions and
Responses pass the string to the Engine as a session alias. On every candidate, NInfer still checks:

- continuation format and complete runtime compatibility key;
- exact `.ninfer` artifact bytes, `model_id`, and `weights_id`;
- capacity, KV layout/codec, speculative backend and draft configuration, proposal head, and Vision
  profile;
- exact token, token-type, three-axis position, media-ledger, and boundary identity through the
  deepest planner-usable frontier: either the saved execution frontier or its rolling turn
  checkpoint;
- the complete required segment inventory and encoded layouts;
- the selected LoRA adapter, when the process was started with `--lora-dir`.

Adapter scoping is not one check but three, because KV and GDN state produced under one adapter is
numerically invalid under another. Every session alias and stable-prefix alias is namespaced by the
selected adapter, unconditionally — base weights get a scope too, so no unscoped key exists and two
adapters cannot collide on one string. A resident lane records the adapter that produced it and
refuses in-place prefix reuse across a mismatch. And a saved slot image records the SHA-256 content
fingerprint of the adapter artifact that produced it.

That identity is content, not position. The adapter pool is enumerated from a directory, so a pool
index means nothing across restarts — adding one file renumbers everything after it. A fingerprint
survives the directory being extended, pruned or reordered, and an image whose adapter is no longer
present fails to resolve and is refused rather than replayed against whichever adapter inherited its
index. Reusing one `prompt_cache_key` across adapters is therefore a safe miss rather than a
correctness failure, in either direction.

The image's model binding separately carries the complete base artifact SHA-256. Registered
`model_id` and `weights_id` strings alone do not identify exact weight bytes, while folding the
whole adapter name list into the binding would unnecessarily invalidate an image when an unrelated
adapter was added.

The complete image is validated before import. Checkpoint fallback then trims paged state and
restores saved GDN, hidden, and backend checkpoint state before recomputing the divergent suffix.
Candidate ranking and restored-token metrics use that effective checkpoint depth, not the stale
generated frontier. Reusing a key for a different conversation, tenant, artifact, prompt,
tool schema, media input, reasoning rendering, or runtime profile produces a safe miss. Keys are
local strings, are not authentication boundaries, and must not be used to combine untrusted users
inside one namespace.

NInfer also derives an immutable, content-addressed alias for each prompt boundary: the end of the
system/developer/tools prefix, every assistant opener in the history, and every client
[prompt cache breakpoint](serving.md#prompt-cache-breakpoints). The alias hashes the exact
tokens, token types, and MRoPE positions through the boundary depth under the runtime
compatibility key, so any request sharing that prefix derives the same name whatever its suffix
or `prompt_cache_key`. A request looks up every boundary it carries (the eight newest openers,
and all system/tools and explicit boundaries), ranks the viable candidates with the routed session
and the resident lanes by reusable depth, and restores the deepest. It publishes the system/tools
boundary and every explicit boundary below its turn-rewrite frontier by capturing them during
prefill, and the boundary at the frontier itself by publishing the completed lane's image - the
same image a `prompt_cache_key` session head is made of - after the last token. Already-published
aliases are write-once and skipped. Concurrent cold requests that need the same unpublished
prefill-captured boundary use single-flight construction: one builds it, the others wait for it.
Each restore currently copies state into private GPU storage; reference-counted shared GPU pages
and partial-tail COW are later optimizations.

The artifact compatibility key includes SHA-256 of the complete `.ninfer` file. For the published
Qwen3.8-27B groupwise artifact used by this repository, that digest is
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`. Repacking or changing even
metadata produces a different compatibility domain and old entries safely miss.

When `--continuation-cache-dir` is configured, NInfer can avoid repeating that large-file SHA-256
scan by recording the result under `PATH/NAMESPACE/artifacts/`. A record is a bounded, versioned,
pointer-free local sidecar bound to the canonical artifact path, file size, device, inode, and
nanosecond modification and change times. An unchanged reopen reports the `fingerprint-cache` load
phase immediately; a full calculation continues to report scanned bytes in the `fingerprint`
phase. The digest itself remains in the continuation compatibility key and the sidecar never becomes
artifact authority.

Sidecar records and directories are owner-only (`0600` and `0700` on Linux), and publication uses an
owner-only temporary file, `fsync`, atomic rename, and directory `fsync`. Missing identity fields,
changed metadata, path or symlink resolution mismatch, malformed/truncated records, unsafe
permissions, and any cache I/O failure cause a normal full SHA-256 scan. A write failure does not
fail model loading. No directory means no sidecar lookup or output.

This optimization assumes the cache and artifact are controlled by the same local OS user; it is
not a defense against that user deliberately restoring or forging file metadata. Sidecar hits are
currently enabled only on Linux, where the complete identity above is available from the opened
file. Other platforms perform the full scan rather than weaken continuation compatibility.

The current cache manifest magic is a development format version (`NICMAN05`), separate from the
`.ninfer` artifact format. Unsupported/older manifests are ignored during discovery rather than
migrated, as are alias records below the current alias-record version. Until a stable cache-format
contract is published, expect development builds to require deleting or rotating the cache
namespace; this loses acceleration, not model or response data.

An alias record carries its `AliasKind` — mutable routed session or write-once stable prefix —
alongside its name and history. The kind is cache state, never inferred from the alias name, so the
engine is free to namespace a name by the adapter that produced its state without changing what the
name means.

`NICMAN05` is metadata-first: in addition to image bytes, TTLs, cost, and chunk inventory, every
manifest carries the exact opaque runtime compatibility key, continuation image format version,
frontier and boundary token depths, and 32-byte SHA-256 digests of the target's canonical exact
prefix identity at those depths. Session history lookup returns these bounded descriptors without
reading image chunks. Qwen rejects incompatible or nonmatching descriptors, orders viable history
by effective depth, and resolves a payload only while it can still improve the selected resident,
stable-prefix, or routed candidate. A digest match is only a negative-filter pass: complete payload
SHA-256, manifest/image metadata agreement, and exact image preflight remain required before import
or destructive history rollback. The corresponding complete-image framing is `NICIMG03`, with Qwen
target image version `3`; old cache namespaces must be deleted rather than migrated.

## System One decision states

A [System One decision](serving.md#system-one-decisions) caches its state row, the `Ls` tokens
`[<|fim_prefix|>, state...]` that every question branch continues, as a second Qwen image kind,
`decision_state`. The image holds exactly the main-text KV `[0, Ls)` and the lane's current Gated
DeltaNet state. It has no continuation hidden state, MTP or DFlash state, sampled token, or turn
checkpoint, because a decision never samples, decodes, or drafts.

- **Alias.** The image is published under one write-once alias derived from the exact state row and
  namespaced by the decision adapter, like every other alias. A decision has no routed session,
  no `prompt_cache_key`, and no boundary alias.
- **Identity.** The decision-state compatibility key is derived from the chat continuation key, so
  it carries the same artifact, KV codec, model-binding, and runtime-profile identity, and a chat
  image and a decision image reject each other at the first comparison. Preflight also requires
  the exact state row, the adapter's content fingerprint, and the exact segment inventory. Decisions
  consume only `decision_state` images, and chat never does.
- **Publication.** A decision that retains its state (`allow_prefix_reuse`) publishes it when it
  completes. The execution thread fences the lane and queues the export; the publication worker
  copies the state out and stores it. Publication happens only when the submit-time lookup found no
  usable image under the alias, so a state continued in L1 is republished only if its alias lapsed,
  and a state restored from L2 or L3 is not republished.
- **Restore.** Submission looks the alias up as descriptors only. Admission restores the image only
  when no idle lane already retains the whole state: the restore reserves the decision's planned
  entitlement, imports the KV and GDN state into an idle lane, and the decision then runs only its
  branches. The result reports `state_source` `l2` or `l3`, `reused_state_tokens` equal to `Ls`, and
  `restore_seconds`. Restoring copies the published state byte for byte. A decision's outcome (L1
  continuation, L2/L3 restore, or miss reason) counts in the continuation metrics like a chat
  request's.
- **Exact state only.** Continuing a shorter retained state as the prefix of a longer one remains L1
  lane planning. L2 and L3 match only the exact state: a longer state derives a different alias, and
  an image's GDN state exists only at its own frontier.
- **Sizing.** A retained decision state counts against `--continuation-cache-l1-mib` like a chat
  session. A 65k-token state is 4.1 GiB under BF16 KV, so at the default 768 MiB it leaves L1 as
  soon as its lane goes idle and every repeat pays an L2 restore (about 3 s on an RTX 4090) instead
  of running only its branches ([decision latency](performance.md#system-one-decision-latency)).

## OpenCode examples

Start one server with durable tiers:

```bash
mkdir -p "$HOME/.cache/ninfer/continuations"

./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity 262144 \
  --max-concurrency 1 --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prefix-checkpoint-policy rolling-tool \
  --continuation-cache l1-l2-l3 \
  --continuation-cache-dir "$HOME/.cache/ninfer/continuations" \
  --continuation-cache-namespace opencode \
  --continuation-cache-l1-mib 768 \
  --continuation-cache-l2-mib 16384 \
  --continuation-cache-l3-mib 49152 \
  --continuation-cache-filesystem-reserve-mib 8192
```

Configure two or three OpenCode sessions against the same base URL and give each a stable, distinct
OpenAI `prompt_cache_key`, for example `repo-a`, `repo-b`, and `repo-c`. Reuse the same key for later
turns of that session. L1 keeps a recently completed lane when space permits, L2 enables fast
switches after lane eviction, and L3 makes a coalesced head available after server restart. The
client must still send the complete conversation/prompt required by its API; the key alone cannot
reconstruct or authorize a prompt.

For a quick Responses request:

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "prompt_cache_key": "repo-a",
    "instructions": "You are working in repository A.",
    "input": "Inspect the failing test.",
    "max_output_tokens": 256
  }'
```

To verify persistence, wait for the interval or token threshold (or stop the server cleanly), restart
with the same artifact and all compatibility-affecting runtime options, and reuse both the mounted
path/namespace and session key. Watch the L3 and restore metrics below. Stored Responses and
`previous_response_id` remain process-local and do not survive this restart; continuation caching is
not response storage.

The native one-request CLI has no `prompt_cache_key` option. With L3 enabled it can still publish and
reuse prompt-boundary aliases across invocations, implicit or marked in the messages file:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages request.json --max-context 32768 --max-new 256 \
  --continuation-cache l1-l2-l3 \
  --continuation-cache-dir "$HOME/.cache/ninfer/continuations" \
  --continuation-cache-namespace native
```

## Sizing and metrics

Estimate a continuation image from actual metrics or disk use; it contains the complete logical KV
and recurrent/backend state, so size grows with frontier depth and depends strongly on KV codec.
For two or three OpenCode sessions, size L2 for the desired number of simultaneously switchable
heads plus stable prefixes. Size L3 for durable heads and history, then add filesystem reserve.
Budgets are ceilings, not reservations or guarantees: TTL and value-based pressure can remove data
earlier, while pinned queued writes can temporarily limit reclamation.

On a 24 GiB card, avoid increasing L1 merely because host RAM is available. L1 consumes the same
startup-fixed GPU KV capacity needed by requests. A practical starting profile is the defaults:
768 MiB L1, 16 GiB L2, and 48 GiB L3, then adjust from observed image bytes, restore frequency, host
memory pressure, and storage write rate.

### Measured L2 switching cost

A 2026-08-18 OpenCode run provides one concrete field measurement, not a general benchmark. It used
Qwen3.8-27B on a 24 GiB RTX 4090 with `rk4v4-e8`, MTP with three draft tokens, a 262,144-token shared
KV pool, `--max-concurrency 1`, and two subagents alternating on that lane. Across 44 L2 restores:

- 2,757,179 cached tokens and 64.0 GB (59.6 GiB) of continuation state were restored;
- L2 import took 88.8 seconds in aggregate, or 2.02 seconds per hit;
- exact preflight took another 39.2 seconds, making synchronous cache work approximately 128 seconds
  in aggregate, or 2.91 seconds per hit;
- routed-session hits alone averaged 3.26 seconds of restore plus preflight work;
- as routed heads grew to 120K-130K tokens, individual restore plus preflight cost rose to roughly
  5.5-7.3 seconds;
- asynchronous L2 admission consumed 525.9 worker-seconds across 61 publications, averaging 8.6
  seconds, but was not directly charged to request TTFT;
- observed prefill rates of roughly 1.1K-2.1K tokens/second imply that recomputing the restored
  prefixes would have taken approximately 22-42 minutes, versus about 2.1 minutes of synchronous L2
  work.

Average TTFT for an L2 hit was 19.5 seconds, so measured restore plus preflight accounted for about
15% of TTFT. Most remaining latency came from suffix prefill and FIFO contention. Four requests
expired under the 30-second admission deadline because both subagents competed for one execution
lane; these were scheduler queue timeouts, not cache restore failures. The run recorded zero restore
failures and no L3 payload reads. These numbers depend on frontier depth, codec, PCIe and host-memory
bandwidth, candidate history, and concurrent CPU/GPU activity; use the timing counters below for the
deployed workload rather than treating these values as fixed costs.

`GET /metrics` exports:

- `ninfer:continuation_lookup_hits_total`, `...lookup_misses_total`, and
  `...preflight_rejections_total`;
- `ninfer:continuation_restore_successes_total`, `...restore_failures_total`,
  `...restored_tokens_total`, and `...restored_bytes_total`;
- `ninfer:continuation_publication_successes_total`, `...publication_failures_total`, and
  `...publication_superseded_total`, plus the five attribution series that sum to the failures:
  `...publication_failed_capacity_total`, `...failed_evicted_total`, `...failed_alias_moved_total`,
  `...failed_lineage_total`, and `...failed_error_total`. `capacity` means one thing only: the
  image alone exceeded the L2 byte budget. An alias refused as malformed or as already owned at
  the other `AliasKind` is a caller defect and counts as `error`, so a nonzero `capacity` against
  an L2 tier with headroom is not a normal reading;
- `ninfer:continuation_persistence_queued_total`, `...coalesced_total`,
  `...successes_total`, and `...failures_total`;
- `ninfer:continuation_preparation_decoded_total`, `...preparation_hits_total`, and
  `...preparation_inline_total`. Decoding a restored image into host memory is the largest CPU term
  in a restore, so a preparation thread decodes ahead for the queue head and admission claims the
  payload by content identity. `hits / (hits + inline)` is the health signal: `inline` counts
  restores the executor had to decode itself, which is correct but costs the execution thread. A
  `decoded` count well above `hits` means preparation is choosing images admission does not use and
  is spending host memory and CPU for nothing;
- `ninfer:continuation_l2_entries`, `...l2_bytes`, `...l3_entries`, and `...l3_bytes`;
- `ninfer:continuation_l2_evictions_total`, `...l2_evicted_bytes_total`,
  `...l3_evictions_total`, and `...l3_evicted_bytes_total`. These count only capacity-driven
  eviction, never a TTL expiry: reclaiming state that went cold is the cache working, while a
  capacity eviction means the tier is too small for the live working set. A promotion that cannot
  find room is refused rather than admitted and then evicted, and a refusal is not counted here;
- `ninfer:l1_evictions_total`, `...l1_demotions_total`, `...l1_resident_entries`, and
  `...l1_resident_bytes`. `evictions` minus `demotions` is the number of retained sessions that
  were dropped with no publication ticket, so they survive in no tier and the next turn of that
  conversation prefills from zero;
- `ninfer:kv_growth_attempts_total`, `...forced_spills_total`, and `...curtailed_total`. A
  generating lane holds pages for its prompt plus a fixed decode window and asks for the rest per
  round. `forced_spills` counts retained sessions demoted to L2/L3 to satisfy such a request, and
  `curtailed` counts requests that ended at `length` because no rung found pages. Sustained
  `forced_spills` means L1 residency is being paid for by concurrent generation; sustained
  `curtailed` means the pool is too small for the configured concurrency.
- fixed L1/L2/L3 `...restore_successes_total`, `...restored_tokens_total`, and
  `...restored_bytes_total` series, plus `...session_restores_total` and
  `...stable_prefix_restores_total`;
- one fixed `ninfer:continuation_miss_<reason>_total` series for each stable terminal reason:
  `disabled`, `no_alias`, `not_attempted`, `entry_unavailable_or_corrupt`, `not_deeper`,
  `preflight_rejected`, `rollback_conflict`, `no_lane`, and `restore_failed`. `no_alias` means
  nothing was found to restore from, whether because the prompt is unaliasable or because no
  candidate exists under the aliases it has; `not_attempted` means a candidate existed and no
  restore step ever evaluated it, which is a scheduling observation rather than a cache one;
- cumulative microseconds and operation counts for L2/L3 lookup and restore, preflight,
  `export` (the device-to-host copy of a completed lane into its image, run by the publication
  worker off the execution thread), `l2_admission`, and `l3_persistence`.

Lookup hits count alias/catalog lookup before exact target preflight. Aggregate restore successes,
tokens, and bytes equal the sum of their L1, L2, and L3 series. Every exported `_total` is a
Prometheus counter; L1/L2/L3 entry and byte occupancy are gauges. Treat restore counters as the
authoritative useful-hit signals. L1 means the planner reused state already resident in VRAM and
therefore has no host restore operation or host restore time. L2 means the selected image bytes came
from host RAM. L3 means they were read and verified from storage for this request, even though that
read also admits the image to L2 for later requests.

A metadata-only session-history scan contributes to request lookup latency and alias/catalog hit
classification but does not increment an L2/L3 payload lookup operation. Each descriptor actually
resolved for exact preflight increments the tier and latency of the source that supplied its bytes.

Each completed request contributes at most one `continuation_miss_<reason>_total`, selected from its
terminal outcome. Candidate-level preflight rejection and restore-failure counters still include
failed candidates before a successful routed-history or stable-prefix fallback. Request lookup,
preflight, and restore times include all candidate work performed for that request; a successful L1
reuse can consequently have lookup/preflight time from an earlier rejected host candidate while its
host restore time remains zero unless an import was actually attempted. L2 admission and L3
persistence times are aggregate worker metrics. In particular, asynchronous L3 persistence time is
never attached to the request that queued publication.

For a concrete verification:

1. Start with `--request-log-jsonl requests.jsonl` and `--continuation-cache l1-l2-l3`.
2. Capture `/metrics`, issue a cold request, repeat it with the same `prompt_cache_key`, and capture
   `/metrics` again. The completion should report `source=l1` while its retained lane survives, or
   `source=l2` after L1 eviction.
3. Wait for `continuation_persistence_successes_total` to advance, stop cleanly, restart with the
   same artifact/options/path/namespace, and repeat the request. The completion should report
   `source=l3`; a following request should report L1 or L2.
4. Compare counter changes and inspect the final `request_done` record. No real-model artifact is
   needed for repository unit tests; the cache tests use synthetic continuation images.

```bash
curl -s http://127.0.0.1:8080/metrics | \
  grep -E 'continuation_(l[123]_restore|miss_|l[23]_(lookup|restore)|l2_admission|l3_persistence)'
```

The completion log deterministically prints `cache_source`, `cache_alias`, `cache_miss`, all three
phase latencies, restored tokens/bytes, rollback, and publication-queued state. JSONL schema v12
carries the same data in `continuation_cache`, in integer microseconds, without raw
routing/session/stable-alias values. Alias values are `none`, `routed_session`, and `stable_prefix`.
After a process restart, a successful durable lookup increments the L3 lookup and restore series;
a host-resident hit increments L2; planner reuse of an already retained GPU lane increments L1 and
has no host restore latency. Starting with `--no-prefix-reuse` produces `cache_miss=disabled`.

Throughput JSON records contain cumulative and interval tier counts/tokens/bytes, terminal miss
reasons, per-tier lookup/restore timing and operation counts, L2 admission, asynchronous L3
persistence, publication/persistence outcomes, occupancy, and L1 eviction/demotion. Interval deltas
are saturating and restart-safe: if a current counter is below the previous snapshot, the current
value is treated as the first delta of a new counter epoch rather than unsigned underflow.

## Implementation status

Implemented:

- pointer-free complete Qwen continuation images and exact prefix/compatibility preflight;
- retained-lane L1 policy, host L2 catalog, restore/demotion, bounded aliases, and history;
- automatic stable-prefix aliases, private restores, and single-flight construction;
- System One `decision_state` images: exact-state publication at completion and L2/L3 restore;
- asynchronous coalesced L3 persistence, chunk deduplication, SHA-256 verification, atomic
  publication, startup discovery, quotas, TTL, and filesystem reserve;
- Prometheus continuation, persistence, occupancy, and L1 counters.

Remaining optional optimizations and product work:

- reference-counted shared/COW GPU KV pages instead of private restore copies;
- coordinated safe sharing of one cache root by multiple processes;
- a public rollback API;
- broader published real-model parity and performance qualification across all targets, backends,
  codecs, media profiles, and hardware;
- stable cache-format migration/versioning.

Exact result caching and duplicate-result request coalescing are not implemented. They are separate
features because their identity must include sampling, stopping, output limits, and output format;
continuation-cache hits still execute generation.
