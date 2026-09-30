# HTTP serving

`build/apps/ninfer-serve` loads one registered artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints, plus TypeSafe System One decisions for Qwen3.8-27B decision
adapters, over one resident NInfer Engine.

## Start the server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 16384 \
  --kv-capacity 32768 \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For the 35B-A3B artifact, select its artifact path; the public model ID follows the container
identity automatically. This package is optional and requires
`-DNINFER_BUILD_QWEN3_6_35B_A3B=ON` at configure time:

```bash
./build/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
  --max-context 16384 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

When `--model-id` is omitted, the server advertises and accepts the loaded container's exact
`identity.model_id`. An explicit `--model-id` remains a public HTTP alias override and does not
select or alter the artifact.

Vision is disabled by default: its weights, Vision scratch phase, and frozen request-transient
buffer are not allocated, and media
requests and token-count requests fail with HTTP 400 `vision_disabled`. Add `--vision` when the
server must accept image or video input. Speculative residency is likewise frozen by
`--spec mtp|dflash` and `--draft-tokens`; omitting `--spec` loads neither backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash is 35B-A3B text-only and
cannot be combined with `--vision`. A later request cannot enable a capability omitted at startup.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |
| `POST /typesafe/v1/systemone` | [System One](#system-one-decisions) decision answered by a decision adapter |
| `GET /typesafe/v1/models` | System One model cards: each decision adapter, then the bound `jev-latest` alias |
| `GET /slots` | per-slot occupancy from the Engine lane table: processing/retained, depths, `session_digest` |
| `POST /slots/{id}?action=save\|restore\|erase` | session persistence; requires `--slot-save-path` |
| `GET /metrics` | Prometheus text exposition; see [Metrics](#metrics) |
| `GET /telemetry` | one live JSON snapshot: board sensors, scheduler occupancy, lane work, VRAM, cache fill, adapter pool and bank residency, System One surface |
| `GET /events` | SSE stream of the schema-22 records `--request-log-jsonl` writes |

`/metrics`, `/telemetry`, and `/events` are always registered and cannot be disabled. Like every
path except `/health`, they require the API key when `--api-key` is set.

### Telemetry and events

`GET /telemetry` returns a complete instantaneous snapshot rather than a delta, so a reader
resynchronizes by fetching it again rather than by replaying anything. It carries what `/metrics`
cannot express: NVML board readings (utilization, temperature, power, clocks, and decoded
clock-throttle reasons), the scheduler's own `running`/`prefilling`/`decode_ready`/`waiting`
occupancy, the execution thread's wall-clock split with its admission decomposition, the
`MemorySummary` VRAM budget, continuation-cache occupancy paired with the configured tier
capacities it is measured against, the per-lane table (`slots`), the LoRA pool and bank
(`adapters`), and the [System One](#system-one-decisions) surface (`systemone`). NVML failure is
reported as `gpu.available = false` with an `error` string rather than failing the request. The
engine's readings (scheduler occupancy, lanes, bank residency and `MemorySummary`) are the ones the
execution thread published at its last unit boundary, so the endpoint never waits for the unit in
flight and stays responsive while chat and decisions keep the GPU busy.

Each `slots[]` entry adds to the `/slots` occupancy what the lane is doing for which system:
`work` is `generation` or `decision` for the request a lane runs, or the kind of state it retains
when idle, and `none` for an empty lane; `adapter` names the adapter that request or state belongs
to, empty for the base weights. A lane retaining a decision state therefore reads as System One
work even between decisions.

The adapter bank is one device arena committed at startup, outside the weights arena and before
KV capacity is resolved. It holds `--lora-slots` slabs, not one per servable adapter: the pool
discovered from `--lora-dir` is unbounded and costs no device memory, and an adapter outside the
slots is swapped in when a request for it is admitted. `memory.lora_bank_bytes` reports the arena
so the division of the board accounts for it; without that field the bank is visible only as
reduced free memory. `adapters` carries:

| Field | Meaning |
|---|---|
| `count`, `rank`, `slots` | pool size, the shared bank rank, and the number of device slots |
| `device_bytes`, `file_bytes` | the bank arena and the pool's adapter files on disk |
| `pool[]` | every discovered adapter in pool order: `name`, `kind` (`generative` or `decision`), `rank`, and the served `model_id`; a decision adapter adds `temperature`, `pointer_dim`, `description`, and `release_date` |
| `resident[]` | one entry per device slot: the `adapter` it holds (empty while unfilled) and whether a running lane `pinned` it |
| `stages`, `stage_seconds` | adapters staged into a slot since startup, a swap or the first fill of an empty slot, and the execution-thread time they took |
| `slot_waits` | requests whose admission waited because every slot was pinned by a running lane using another adapter |

A generative adapter's `model_id` is `<model>-<name>` on `/v1`; a decision adapter's is its bare
name on `/typesafe`. The pool comes from the load summary rather than from served model ids, so an
adapter that has taken no traffic is still reported. `systemone` carries `supported` (the target
can answer decisions), `alias` (`jev-latest`), and `binding`, the decision adapter the alias
resolves to, empty when it is unbound.

`cache.l2` and `cache.l3` additionally carry `evictions` and `evicted_bytes`, counting only
entries pushed out because the live working set exceeded that tier's byte budget. A TTL expiry is
deliberately not counted as one, and neither is a promotion that was refused for want of room,
since neither is the tier being too small for what is in use. `cache.restore_deferrals` is
reported apart from `restore_failures` because a deferral leaves the candidate live for a retry
and a failure does not. The same counters appear on the throughput record as cumulative totals
paired with interval deltas, so churn is readable from a replayed log as well as live.

`GET /events` streams the same schema-22 records `--request-log-jsonl` appends, as named SSE
frames whose event name is the record's own `event` field. The records are formatted once and
fanned out to both sinks, so a live reader and a post-hoc reader of the file see identical lines.
A connecting reader is replayed the retained `server_start` record followed by a bounded ring of
recent records, then receives live ones. Subscribers are bounded and lossy by construction: a
reader that stops consuming has its oldest records dropped rather than applying backpressure to
the execution thread. `--request-log-jsonl` is not required for `/events`.

### Session persistence

`--slot-save-path DIR` enables llama.cpp-compatible slot persistence. `save` writes slot
`{id}`'s complete resident session - paged Text and MTP KV in logical page order, GDN
linear-attention state, the MTP tail hidden, the turn checkpoint, and the resident prefix
identity - to `DIR/filename` from a `{"filename": NAME}` body; `restore` rebuilds the slot
from such a file (evicting whatever it retained); `erase` evicts the slot and reports its
depth. Names are one conservative path component: 1-128 bytes of `[A-Za-z0-9._-]` with no
leading dot.

A restored slot is indistinguishable from one the engine retained itself: a request that
extends the saved conversation reuses the cache (`AppendAtFrontier`, or the saved turn
checkpoint on a rewritten last turn) instead of re-prefilling, across server restarts. The
device round trip runs at a request boundary while file I/O stays outside the GPU lock; a
slot with an active request answers 409.

Sessions are identified by a `session_digest` (a stable hash of the resident token ledger;
treat it as opaque). Successful chat completions report `id_slot` and, when the lane retained
the finished session, its `session_digest` top-level next to `timings` (final stream chunk
included); `GET /slots` reports each idle retained lane's digest; save and restore responses
echo the digest of the session they moved. `save` and `erase` accept an optional
`{"if_digest": DIGEST}` precondition, checked atomically with the operation, so a client
always persists or evicts exactly the session it means - a mismatch (including a since-evicted
session) answers 409 `slot_session_mismatch`. Snapshots bind to the exact weights identity, KV
dtype/geometry, and speculative configuration, and restore refuses anything mismatched.
Sizing: roughly the configured KV bytes per token times session depth, plus a fixed GDN
state block (about 300 MiB with a held turn checkpoint on Qwen3.8-27B); a 6.9k-token
session measures 416 MiB, saving in ~0.24 s and restoring in ~0.12 s on NVMe. The DFlash
backend is not supported.

When `--turn-checkpoints` is active, a snapshot also carries the slot's checkpoint ring at
about 147 MiB per entry. The snapshot format is version 5, which always records both the
adapter fingerprint and the ring section and stores packed 4-bit KV pages in the midrise codec;
earlier versions are rejected. The restored
ring lets a later mid-history edit reuse the session; see
[turn-checkpoint-ring.md](turn-checkpoint-ring.md).

A successful save or restore binds the slot to its file. With `--auto-save-evicted`, an
involuntary eviction (a fresh session claiming the slot, a restore over it, or a
KV-pressure eviction) first spills the resident session back to that file, so the client's
next restore recovers the session at its latest frontier instead of the last explicit
save. Sessions never saved or restored have no binding and are not spilled; an explicit
`erase` is a deletion request and never auto-saves. The console reports each spill as
`slot auto-save file=... n_saved=...`.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed`;
- one stop string or an array of stop strings;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- non-strict function tools, `tool_choice` `auto`/`none`, assistant tool-call history, and tool-result
  messages;
- `prompt_cache_key` as a continuation-session routing hint, subject to exact prefix checks;
- the top-level `reasoning_effort` field;
- the `enable_thinking` extension;
- `chat_template_kwargs.preserve_thinking` and the top-level `preserve_thinking` alias.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`. Chat tool definitions are non-strict only. Tool choice is limited to
`auto` and `none`, and logit bias is not implemented.

Tool schemas with `strict:true` return `strict_tools_not_supported`; `tool_choice:required` and
named choices return `tool_choice_not_supported`. Omitted, null, or false `strict` values are
accepted. Omitted, null, or empty `logit_bias` is accepted as a client default, while a nonempty map
returns `logit_bias_not_supported` because token bias is not applied by the sampler. Unknown request
fields are rejected rather than silently ignored.

At startup, NInfer resolves prompt capabilities from the exact `frontend/chat_template.jinja`
resource embedded in the loaded artifact. It does not infer them from the request's `model` field,
the artifact identity, or a target profile. A recognized effort-capable template exposes `low`,
`medium`, and `xhigh`; omitting effort uses that template's declared default. An explicit effort
not exposed by the loaded template returns HTTP 400 with code
`reasoning_effort_not_supported` before prompt preparation.

For Chat Completions, `reasoning_effort: "none"` disables thinking. `low`, `medium`, and `xhigh`
select the corresponding template effort when available. The other OpenAI protocol values
`minimal`, `high`, and `max` are parsed but rejected when the loaded template does not expose them.
`enable_thinking` controls the same new-turn thinking switch; a contradictory combination with
`reasoning_effort` returns `conflicting_template_option`.

`preserve_thinking` controls whether reasoning from closed assistant turns remains in later
prompts. It defaults to the server setting, which is off unless `--preserve-thinking` is used. If
both OpenAI spellings are present they must carry the same boolean value. Unknown non-null
`chat_template_kwargs` are rejected.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage.

Tool calls stream in `delta.tool_calls` while the model writes them. The first entry for a call
`index` carries `id`, `type: "function"`, and `function.name`, with `function.arguments: ""`.
Later entries carry only `index` and an argument fragment. See [Tool calls](#tool-calls) for
announcement, argument conversion, and the finish reason of a call the output cut short.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
registered artifact identities use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | required string or non-empty typed Item array |
| `input[].type=item_reference` | substitutes the exact retained output Item with that ID in place; independent of `previous_response_id`; stale, deleted, or evicted IDs return `item_not_found` |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `prompt_cache_key` | optional continuation-session routing hint; reuse is authorized only by exact prefix and runtime compatibility checks |
| `prompt_cache_options` | optional `{"mode": "implicit"}` (default) or `{"mode": "explicit"}`; see [Prompt cache breakpoints](#prompt-cache-breakpoints); any other member, including retention, returns `parameter_not_supported` |
| `max_output_tokens` | integer at least `16`; default is `--default-max-tokens` |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `reasoning.effort` | `none` disables thinking; `low`, `medium`, or `xhigh` selects an effort exposed by the loaded chat template; `minimal`, `high`, and `max` return `reasoning_effort_not_supported` for the registered templates |
| `reasoning.summary` | optional `auto` or `detailed`; omitted uses the effective default `auto`; generated Qwen thinking is always disclosed as native Responses `summary_text` Items and SSE events |
| `chat_template_kwargs.preserve_thinking` | optional boolean controlling whether closed-turn reasoning remains in reconstructed prompts |
| `preserve_thinking` | top-level alias for the same option; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | flat Responses function definitions; see below |
| `tool_choice` | `auto` or `none` |
| `parallel_tool_calls` | omitted or `true` |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, an empty array, or `['reasoning.encrypted_content']`; that AI SDK compatibility hint is accepted and ignored |
| `stream_options` | omitted or `{"include_obfuscation":false}` |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text`; optional `prompt_cache_breakpoint` |
| `output_text` | assistant-message replay part containing string `text`; optional `prompt_cache_breakpoint` |
| `input_image` | user-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | native `summary_text` replay Item; legacy raw `reasoning_text` content is also accepted; `encrypted_content` may be null for SDK compatibility but non-null values return `encrypted_reasoning_not_supported` |
| `function_call` | completed assistant call with optional `id`, and required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed tool result with required `call_id`; `output` may be a string or a non-empty array of `input_text` and `input_image` parts; an `input_text` part accepts `prompt_cache_breakpoint` |

Adjacent function-call Items are grouped into one assistant history turn. A reasoning Item attaches
to the following assistant message or function call. Input Item IDs are preserved when supplied and
generated otherwise; duplicate IDs fail.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, message `phase`, and other
Item/content types are not supported. Rich function outputs support text and images, but not files.
HTTP media
URLs stored in a response chain are fetched again when that chain is continued; use data URIs when
the historical media bytes must be immutable.

### Prompt cache breakpoints

Every prompt is content-addressed at a small set of boundaries. A boundary is a token depth at
which the Engine can restore a previously published continuation image (KV pages plus the GDN and
speculative state needed to continue) into whichever request shares the exact tokens through that
depth, whether or not that request carries the same `prompt_cache_key`. By default the boundaries
are implicit: the end of the system/developer/tools prefix, and every `<|im_start|>assistant`
opener in the history. Implicit boundaries are looked up on every request; the system/tools
boundary and the opener at the turn-rewrite frontier are also published, so a repeated prefix is
found without any client cooperation.

A client that knows where its stable content ends can mark it instead:

```json
{
  "model": "qwen3.8-27b",
  "prompt_cache_options": {"mode": "explicit"},
  "input": [
    {"role": "developer", "content": [
      {"type": "input_text", "text": "...long instructions...",
       "prompt_cache_breakpoint": {"mode": "explicit"}}]},
    {"role": "user", "content": [
      {"type": "input_text", "text": "...large document...",
       "prompt_cache_breakpoint": {"mode": "explicit"}},
      {"type": "input_text", "text": "Summarize the document."}]}
  ]
}
```

`prompt_cache_breakpoint` is accepted on `input_text`, `output_text`, and `function_call_output`
`input_text` parts, and must be exactly `{"mode": "explicit"}` (`invalid_value` otherwise; other
members return `parameter_not_supported`). A marked part that ends its message places the boundary
after the rendered message close, so the whole message including its template framing is
cacheable; a marked part followed by more parts places the boundary at the end of that part's
text. On the final message the boundary sits at the assistant opener the request appends, which
is exactly the prefix the next turn's history reproduces. A breakpoint on a system or developer
message lands on the system/tools boundary. Marked boundaries are always looked up and always
published. A byte cut inside a merged token snaps back to the deepest exact token prefix before
it, and a boundary past the turn-rewrite frontier is dropped because the lane must end the
request holding state there. At most four breakpoints are accepted per request; a fifth returns
`too_many_prompt_cache_breakpoints`. Top-level `instructions` is a string and cannot carry a
breakpoint.

Publishing is not free in the same way everywhere. The boundary at the turn-rewrite frontier
(the generation opener, and therefore a final-message breakpoint) is published from the
completed lane's image after the last token, off the request's own latency path. Every other
boundary is captured during prefill, which stops at that depth and copies the state through it
to the host before continuing; a request pays that once per unpublished prefix, and concurrent
requests needing the same one wait for a single builder. Mark a large stable prefix once, not
every block.

`prompt_cache_options.mode` selects how the implicit boundaries combine with marked ones.
`implicit` keeps them all, so a marked request still benefits from template boundaries.
`explicit` suppresses the implicit system/tools and opener boundaries and content-addresses only
the marked parts. Neither mode changes `prompt_cache_key` session routing or the lane-local
checkpoints that serve a resident conversation; both keep working without breakpoints.

Breakpoints are content-addressed, so they never grant access to state: a request reaches a
boundary's image only by presenting the exact tokens, token types, and MRoPE positions through
that depth under the same artifact/runtime compatibility key. The images live in the L2/L3
continuation tiers selected at startup (`--continuation-cache l1-l2` or `l1-l2-l3`); without
those tiers, breakpoints are accepted but publish nothing, and only resident-lane prefix reuse and
retained L1 lanes apply.

### Function tools

Responses function definitions are flat rather than Chat Completions' nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

NInfer renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. NInfer does not execute functions or enforce JSON Schema through constrained
decoding, so `strict:true`, `tool_choice:required`, named tool choice, hosted tools, MCP tools, and
custom free-form tools are rejected.

A `function_call` Item is `completed` when the model closed the call and `incomplete` when the
output ended inside it; a message Item followed by calls is always `completed`. See
[Tool calls](#tool-calls).

### Response object and usage

A terminal wire response has `object: "response"` and exactly one of `completed`, `incomplete`, or
`failed` in `status`, plus a typed `output` array. `cancelled` is not emitted. NInfer may emit:

- when generated reasoning is non-empty, a `reasoning` Item containing it as native `summary_text`;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Reasoning disclosure is always enabled. Omitting `reasoning.summary` selects the truthful effective
default `auto`, which is echoed in the response envelope; an explicit `auto` or `detailed` value is
echoed unchanged. Ordinary model/string stops produce `completed`. Output-token or
context-capacity exhaustion produces `incomplete` with
`incomplete_details.reason: "max_output_tokens"`. Errors accepted after an SSE response has started,
including cancellation while the connection remains writable, produce `response.failed` with an
error; validation and preparation errors remain normal HTTP error responses.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17, "cache_write_tokens": 25},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
prompt prefix the Engine reused, whether from a resident lane or a restored continuation image.
`cache_write_tokens` is the depth of the deepest boundary this request captured for publication
beyond what it reused, clamped so that `cached_tokens + cache_write_tokens <= input_tokens`; it is
`0` when the request published nothing new, including when every boundary was already in the
cache. Neither value is a billing category.
`output_tokens` is the count of accepted generated token IDs, including a withheld stop token when
applicable. `reasoning_tokens` is counted in the Qwen output decoder while accepted tokens are
still in the reasoning channel; it is not estimated by re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added`, followed by a reasoning-summary or message content part;
3. zero or more `response.reasoning_summary_text.delta` or `response.output_text.delta` events;
4. matching summary/content completion and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Each function call is its own Item and streams while the model writes it:

1. `response.output_item.added` with an `in_progress` `function_call` Item whose `arguments` is
   empty;
2. `response.function_call_arguments.delta` events for the argument fragments;
3. `response.function_call_arguments.done` and `response.output_item.done` when the model closes
   the call.

The reasoning and message Items are finished before the first call's Item is added, so Items never
interleave. If the output ends inside a call, that call's `.done` events come at the end of the
stream, with status `incomplete`. IDs, output indices, and content indices remain stable, and
concatenated deltas equal the terminal Item. Responses SSE does not emit the Chat Completions
`[DONE]` sentinel.

With tools enabled, ordinary answer text still streams immediately. The only text held back is
trailing whitespace and a possible `<tool_call>` prefix, until the next output settles them.
Malformed tool markup is flushed back as ordinary text without losing bytes.

For stored streams, successful storage is the commit point before staged completion events are sent.
A disconnect or cancellation before that point does not create a retrievable record; a disconnect
while flushing after commit does not erase it. The server emits at most one terminal event.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so resident
prefix reuse applies naturally.

`item_reference` is a separate mechanism: each occurrence is replaced in place by the exact stored
output Item it names, preserving surrounding input order. It neither implies nor rewrites
`previous_response_id`; both mechanisms may be used in one request. Deleting or evicting the owner
invalidates its Item references, while contexts already copied into stored descendants remain usable.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
identity then determines whether the Engine restores a turn checkpoint or performs a full reset.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`) |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

Stateless replay uses the public `summary_text` from a reasoning Item as the following assistant
turn's `reasoning_content`. This works with `store:false` and does not depend on server-side state.
AI SDK clients may still send `include:["reasoning.encrypted_content"]`; NInfer accepts that hint
with either store value but ignores it and never emits `encrypted_content`.

### Responses input token count

`POST /v1/responses/input_tokens` accepts `model`, `input`, and the thinking-history options
`preserve_thinking` / `chat_template_kwargs.preserve_thinking`. It performs the same typed Item,
template, and media expansion and does not run generation:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, safety/user identifiers, Structured Outputs/JSON mode, background execution,
compaction, files/audio, and OpenAI-hosted/MCP/custom tools. `prompt_cache_key` is a continuation
routing hint and never bypasses exact prepared-prefix checks. The only non-empty `include` value
accepted is the ignored AI SDK hint `reasoning.encrypted_content`; no encrypted field is produced. These
are explicit compatibility boundaries; unsupported values return capability errors rather than
being silently accepted.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports system text, user/assistant history, text and image blocks, thinking blocks,
tool-use history, tool results, client-defined tools, non-streaming responses, and Anthropic SSE
events. `thinking.type: "disabled"` disables thinking; other supported values enable it.
The independent top-level `preserve_thinking` boolean controls closed-turn history and otherwise
uses the server default.

A stream opens content blocks in output order with contiguous indices: thinking, then text, then
one `tool_use` block per call. A `tool_use` block starts as soon as its call is announced, with the
call's `id`, its `name`, and `input: {}`. It then streams `input_json_delta` fragments that
concatenate to the call's input and stops when the model closes the call. See
[Tool calls](#tool-calls).

Anthropic `output_config.effort` accepts the protocol values `low`, `medium`, `high`, `xhigh`, and
`max`. The value is then checked against the loaded chat template in the same way as the OpenAI
endpoints; the registered effort-capable template exposes `low`, `medium`, and `xhigh`. Combining
an effort with `thinking.type: "disabled"` is rejected as contradictory.

Anthropic's `model` field is treated as a response label and does not select the loaded artifact.

`cache_control: {"type": "ephemeral"}` on a `text` or `tool_result` block marks a
[prompt cache breakpoint](#prompt-cache-breakpoints) at the end of that block, with the same
placement, snapping, and four-per-request limit as the Responses field
(`too_many_cache_control_breakpoints`). On `system` blocks and `tools[]` entries it is accepted and
counts toward the limit, but changes nothing: that prefix is already an implicit boundary. Only the
ephemeral kind exists (`invalid_value` otherwise) and `ttl` returns `parameter_not_supported`,
because retention is the server's continuation-cache idle timers. The Messages API has no mode
switch; implicit boundaries stay active.

Usage follows the Anthropic split: `input_tokens` counts only the uncached prompt tokens,
`cache_read_input_tokens` the prefix restored from a lane or continuation image, and
`cache_creation_input_tokens` the tokens this request captured for publication beyond that; the
three sum to the prompt. Streaming reports the whole prompt as `input_tokens` in `message_start`,
before admission decides what is reused, and the final split in the `message_delta` usage.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## System One decisions

Qwen3.8-27B also serves TypeSafe's System One protocol: one state and a set of typed questions
(`noul`, `choice`, `score`) in, calibrated option probabilities out. A System One model is a
*decision adapter* in the `--lora-dir` pool: a LoRA adapter plus a pointer head, converted with
`convert_lora.py --decision-head` (see the
[adapter authority](maintainer/qwen3.8-27b-lora-adapters.md)). Decisions run in the same process as
chat and share its weights, KV pool, lanes, and `--lora-slots` residency; they are prefill-only and
never sample, decode, or draft. Qwen3.6-35B-A3B has no adapter pool and serves no System One model.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --lora-dir lora --systemone-default decider --max-context 16384
```

TypeSafe's `GET /v1/models` shares OpenAI's path but not its body, so the System One routes live
under `/typesafe`, the base path Vercel AI Gateway also gives the TypeSafe API beside its OpenAI-
and Anthropic-compatible ones, and the root `/v1/models` stays OpenAI's. Point the TypeSafe SDK at
it with `base_url`; the SDK requires a key even for an open server:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080/typesafe")
result = client.system_one(
    state="I was charged twice this month. Please refund one charge before Friday.",
    questions={
        "billing": Noul(instructions="Is this message about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
        "urgency": Score(instructions="How urgent is it?", criteria=["Can wait", "Today"]),
    },
)
```

Request validation, rendering, the token layout, answers, confidences, rounding, and usage follow
kev's System One server exactly, including FastAPI's 422 error list for invalid bodies. `model`
names a decision adapter by its pool name. `jev-latest`, the SDK's default and the value used when
`model` is omitted, answers with `--systemone-default`, or with the pool's only decision adapter
when the flag is unset; with several decision adapters and no flag it is unbound. The response
`model` is the adapter that answered, not the alias. Generative adapters and the base model are not
System One models, and decision adapters are not OpenAI or Anthropic models.

A `noul` answer is `{"type": "noul", "noul": p}` and carries no `confidence`; `choice` answers
(`choice`, `confidence`, `probabilities`) and `score` answers (`score`, `legend`, `probabilities`,
`confidence`) do, as in TypeSafe's API.
`usage.input_tokens` counts the state and every question branch; `usage.output_tokens` is the token
count of the answers' Python `json.dumps` text. `latency_ms` is the engine's time on the decision,
to 0.1 ms: the restore of a cached state image, if any, plus execution from admission to result,
excluding the queue wait. A state is cut to its first 65,536
tokens, delimiter included. A question whose branch does not fit the 73,728-token row with its
state, or a decision whose state plus its longest branch exceeds a lane's `--max-context`, is refused
with 422 and a `detail` string.

| Status | When |
|---|---|
| 400 | the body cannot be decoded: ill-formed UTF-8, a lone UTF-16 surrogate, an integer of more than 4,300 digits, or nesting deeper than 1,000 |
| 401 | `--api-key` is set and the request carries no matching bearer or `x-api-key` key; also sends `www-authenticate: Bearer` |
| 404 | no System One model answers to `model`, or the path is unknown (`{"detail": "Not Found"}`) |
| 413 | the body exceeds `--max-request-mib` |
| 422 | validation errors (FastAPI's list), or a decision beyond its token budgets |
| 429 | the pending queue is full; sends `Retry-After: 1` and `retry-after-ms: 1000` |
| 503 | the pending deadline passed; same retry headers |
| 500 | an internal error, `{"detail": "Internal Server Error"}` |

Every response under `/typesafe/`, errors included, carries `x-typesafe-request-id`: the request's
own value when it sends one, otherwise a fresh uuid4 hex. A repeated state is not recomputed: it
continues from a lane that retained it or, once no lane does, is restored from the exact-state image
its first decision published to the L2/L3 continuation tiers
([decision states](continuation-cache.md#system-one-decision-states)); `--no-prefix-reuse` disables
both. A decision that reuses its exact state, retained or restored, reads the bytes the cold
decision computed and repeats its probabilities exactly. One that continues a shorter retained state
reproduces the input semantics of a cold decision, not its arithmetic, so its probabilities agree
within kev's 0.03 tolerance rather than bit for bit. kev's demo routes (`/permute`, `/separate`) are
not served, and only `jev-latest` is an alias.

The request log records each submitted decision as `decision_start` and `decision_done` or
`decision_error` (see [Structured request log](#structured-request-log)). `/metrics` counts answered
decisions in `ninfer:decisions_total`, `ninfer:decision_questions_total`,
`ninfer:decision_state_tokens_total`, `ninfer:decision_reused_state_tokens_total`,
`ninfer:decision_branch_tokens_total`, and `ninfer:decision_execution_seconds_total`; in-flight
decisions count toward `llamacpp:requests_processing` and `llamacpp:requests_deferred`.

With `--web-dir`, the [playground](dashboard.md#playground) at `/playground` builds and runs these
requests in a browser and joins each answer to its decision's records through
`x-typesafe-request-id`.

### System One fidelity

kev holds a served decision model to two bars (its Kev-27B model card): every question's
probabilities within max |Δp| ≤ 0.03 of the evaluation path, with at most one changed answer per 280
questions; and each question asked alone within the same bar of the full request.
`tools/parity/qwen3_8_27b/decision.py gates` measures both over kev's sample, the first 200 clean
decision-v7 development records (280 questions), plus three long cases with multi-chunk states and
branches longer than a pass: 290 questions, every state computed cold. Its evaluation path is a
PyTorch row-form reference: the trainer's PEFT adapter and head over the served artifact's own base
weights, with BF16 KV.

Every unit of a decision, state chunks and branch passes alike, runs BF16 activations: the INT8
activation routes of chat prefill are never admitted. The KV codec is therefore the one lossy choice
left to the operator, and it touches only the state: a branch pass attends its own question tokens
from their BF16 values and writes no KV, so only a branch longer than one pass stores its tokens
through the codec. Measured on an RTX 4090, 2026-09-30, with `systemone-decision-v7`:

| `--kv-dtype` | Served vs reference, max \|Δp\| (mean) | Changed answers / 290 | Alone vs full request, max \|Δp\| | kev's bars |
|---|---:|---:|---:|---|
| `bf16` | 0.018 (0.0026) | 0 | 0.006 | met |
| `int8` | 0.025 (0.0028) | 0 | 0.008 | met |
| `rk8v4` | 0.041 (0.0042) | 0 | 0.007 | missed |
| `rk4v4` | 0.048 (0.0048) | 1 | 0.004 | missed |

A repeated cold decision is bit-identical on every codec. The rotated codecs miss only the
probability bar, and only on a few questions: 5 of kev's 280 exceed 0.03 on `rk8v4` and 6 on
`rk4v4`, none by more than 0.018, while their changed answers stay within kev's one per 280. At the
task level the codec made no measurable difference even before the current 4-bit codec: over the
development partition's 1,264 clean questions, probed before the segmented branch Ops, accuracy was
0.870–0.872, Brier 0.184–0.185 and ECE 0.024–0.031 for `bf16`, `int8`, `rk8v4` and the since-removed
`rk4v4-e8`, whose largest deviation (0.205) was four times the current `rk4v4`'s, against an
accuracy standard error of about 0.009.

`bf16`, the `ninfer-serve` default, is the qualified configuration for decisions. `int8` met both
bars here, but measured 0.031 before the segmented branch Ops, so it sits near the bar rather than
well inside it. Choose a rotated codec for chat context capacity (`rk4v4` stores about 3.8 times as
many tokens per GiB as `bf16`) knowing that decisions served beside chat on it keep task-level
quality but are not qualified to kev's bar. A trainer on another quantization of the
base, such as the NF4 QLoRA base `llm-datasets` trains on, differs from any served codec by more than
the codec does ([adapter authority](maintainer/qwen3.8-27b-lora-adapters.md#33-revisit-decision-adapters-use-only-these-six-modules)).

## Tool calls

Chat Completions, Responses, and Messages use the same tool-call parser. NInfer renders function
definitions into the Qwen prompt and does not execute tools. It also does not use constrained
decoding to enforce the client's JSON Schema. The model writes a call in the template's XML form:

```text
<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
```

NInfer converts this to the protocol's JSON while the model is still generating it, so a client can
show the tool and its arguments as they arrive. A request uses the parser when it declares tools or
carries tool-call history.

A call is announced once `<tool_call>`, optional whitespace, and `<function=NAME>` have arrived and
`NAME` is a valid function name: 1 to 64 characters from `[A-Za-z0-9_-]`, or up to 128 on the
Messages endpoint. The name does not have to be a declared tool. The announcement carries the call
ID and name with empty arguments. The arguments then stream as JSON fragments that concatenate to
the call's `arguments`. Each parameter's type comes from the request's own tool schema:

| Declared parameter type, ignoring `null` | Streamed as |
|---|---|
| `string` | an escaped JSON string, as the model writes it |
| `object`, `array`, or both | the model's JSON as written, once the value starts with `{` or `[` |
| anything else, including undeclared tools and parameters | the whole value at `</parameter>`: its JSON when it parses, otherwise a JSON string |

The parser strips exactly one framing newline from each end of a value and keeps indentation and
inner whitespace. A `string` parameter stays a string even when its text looks like JSON. The
closing `}` arrives only when the model closes the call with `</function>` or a bare
`</tool_call>`. Until then the client never holds a complete-looking argument object for an
unfinished call.

A Hermes-style JSON body inside `<tool_call>` is different: the parser validates it whole at
`</tool_call>` and emits it as one argument fragment.

Text before the first call streams as ordinary content. If some markup never becomes a valid call,
it is released as text with no bytes lost. After the first call, any output that is not another call
is dropped, as the reference Qwen parser does. The request log counts the dropped bytes. Calls are
numbered in output order, and every protocol closes its text block or message before the first
call.

A call that the output ends inside stays incomplete. It keeps exactly the argument prefix that was
streamed and is never closed automatically. The turn then keeps the engine's finish reason instead
of handing control to the client. For the output limit, the context limit, and the repetition guard
the protocols report:

| Outcome | Chat Completions | Responses | Anthropic Messages |
|---|---|---|---|
| every call closed | `finish_reason: "tool_calls"` | each call Item `completed` | `stop_reason: "tool_use"` |
| a call cut short | `finish_reason: "length"` | that Item `incomplete`; the response is `incomplete` | `stop_reason: "max_tokens"` |

Tool-call history must carry JSON-object arguments. An incomplete call cannot be sent back as
history, and neither can a completed call whose arguments are not an object. For the same reason,
stored Responses continuation history keeps only closed calls with object arguments.

A call written inside a `<think>` block that is still open is held back rather than streamed as
reasoning. If `</think>` arrives later, the held text is released as reasoning. If the block never
closes, the call reaches the parser only when generation ends and arrives all at once, not while it
is being written. Reasoning after that `<tool_call>` is held until then as well.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

Every HTTP response, including errors and SSE, carries `x-request-id`. HTTP 429 overload and HTTP
503 queue-timeout responses also carry `Retry-After: 1`. Under `/typesafe/`, a missing or wrong
key is answered in System One's shape (`{"detail": ...}` with `www-authenticate: Bearer`), and
every response also carries `x-typesafe-request-id`. `--cors` adds permissive browser headers,
allows unauthenticated `OPTIONS` preflight, permits SDK authentication/content headers, and exposes
`x-request-id`, `x-typesafe-request-id`, `Retry-After`, and `retry-after-ms`; it is disabled by
default.

Response retrieval rejects unsupported query options explicitly. Input-Item retrieval supports
`after`, `limit=1..100`, and `order=asc|desc`; duplicate, unknown, and invalid pagination parameters
are errors. Delete and cancel routes accept no query parameters.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `8192` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--device N` | CUDA device index | `0` |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--request-log-jsonl FILE` | append full-precision server/request records; `/events` streams the same records regardless | disabled |
| `--web-dir DIR` | serve the built dashboard (`apps/web/dist`) from `/` on this port, and its System One playground from `/playground` | disabled |
| `--slot-save-path DIR` | enable `/slots/{id}?action=save\|restore\|erase` session persistence into DIR | disabled |
| `--turn-checkpoints N` | retained turn checkpoints per slot for mid-history prompt reuse; see [turn-checkpoint-ring.md](turn-checkpoint-ring.md) | `0` |
| `--auto-save-evicted` | spill an involuntarily evicted session back to its bound slot file; requires `--slot-save-path` | off |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk2v4-e8` | KV-cache storage; the Hadamard-rotated modes trade key/value precision for capacity | `bf16` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--lora-dir DIR` | discover the adapter pool: each `NAME.lora.ninfer` is chat model `<model id>-NAME`, or System One model `NAME` for a decision adapter | unset |
| `--lora-slots N` | device-resident adapter slabs; the rest swap in on admission | `2` |
| `--systemone-default NAME` | decision adapter the System One default model `jev-latest` answers with | the only decision adapter |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--prefix-checkpoint-policy stable-turn\|rolling-tool` | choose a stable first-assistant rewrite checkpoint or advance it after completed tool history | `rolling-tool` |
| `--continuation-cache off\|l1\|l1-l2\|l1-l2-l3` | retained VRAM only, plus host RAM, or plus persistent local storage | `l1-l2`, or `l1-l2-l3` when a directory is supplied |
| `--continuation-cache-policy adaptive` | byte/value-aware implemented cache policy | `adaptive` |
| `--continuation-cache-dir DIR` | L3 parent; entries use `DIR/NAMESPACE` | unset |
| `--continuation-cache-namespace NAME` | safe single-component L3 namespace | `local` |
| `--continuation-cache-l1-mib N` | retained-lane VRAM eviction budget | `768` |
| `--continuation-cache-l2-mib N` | host continuation-image budget | `16384` |
| `--continuation-cache-l3-mib N` | durable unique chunk plus manifest budget | `49152` |
| `--continuation-cache-l1-idle-seconds N` | retained-lane idle eviction threshold; `0` means no expiry | `600` |
| `--continuation-cache-l2-idle-seconds N` | independent host-image idle TTL; `0` means no expiry | `7200` |
| `--continuation-cache-l3-ttl-seconds N` | independent durable-manifest idle TTL; `0` means no expiry | `86400` |
| `--continuation-cache-persist-interval-seconds N` | timer persistence trigger; `0` disables only this trigger | `60` |
| `--continuation-cache-persist-min-tokens N` | frontier-growth trigger; `0` makes every publication due | `8192` |
| `--continuation-cache-filesystem-reserve-mib N` | minimum free space retained below the L3 root | `0` |
| `--prefix-checkpoint-history N` | IDs retained per mutable session alias, including current head | `4` |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--no-repetition-guard` | let a generation that has locked into a repeating cycle run to its output limit | guard on |
| `--repetition-guard-window N` | how far back a repeat may reach, which also bounds the detectable period | `512` |
| `--repetition-guard-ngram N` | token n-gram whose recurrence proposes a period | `24` |
| `--repetition-guard-cycles N` | whole periods confirmed before the cycle counts as locked | `3` |
| `--repetition-guard-min-tokens N` | confirmed tokens required regardless of period | `64` |
| `--cors` | permissive browser CORS headers | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |

Engine selects sampling defaults from the loaded model and the request's resolved thinking mode.
Qwen3.8-27B uses `1.0/0.95/20/0/0` for
temperature/top-p/top-k/min-p/presence penalty in thinking mode and `0.7/0.80/20/0/1.5` in
non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Continuation sessions and persistence

Use a stable, distinct `prompt_cache_key` for each of two or three OpenCode sessions, such as
`repo-a`, `repo-b`, and `repo-c`. The key selects a candidate session head; it is not trusted as
prompt identity. NInfer verifies the complete artifact/runtime compatibility domain and exact
tokens, positions, media ledger, and boundary before restore, so accidental key reuse is a safe
miss. Prompt boundaries — the system/developer/tool prefix, assistant openers, and client
[prompt cache breakpoints](#prompt-cache-breakpoints) — also receive content-addressed immutable
aliases and can be shared across independent requests with different suffixes.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 1 \
  --kv-dtype rk4v4 --spec mtp --draft-tokens 3 --lm-head-draft \
  --continuation-cache l1-l2-l3 \
  --continuation-cache-dir "$HOME/.cache/ninfer/continuations" \
  --continuation-cache-namespace opencode \
  --continuation-cache-filesystem-reserve-mib 8192
```

The server user needs create/read/write ownership of the cache path; Linux directories and files
are secured to `0700`/`0600`. The configured directory also stores a local artifact SHA-256 sidecar
under `DIR/NAMESPACE/artifacts`: an unchanged Linux reopen reports `fingerprint-cache` and avoids
rescanning the model file, while any path, identity, metadata, permission, format, or I/O mismatch
falls back to the byte-reporting `fingerprint` scan. The sidecar trusts the same local OS user and is
disabled on platforms that cannot provide the complete file identity; it is an optimization, not
artifact authority. Reuse the same writable path, namespace, exact artifact, KV/backend
profile, and session key after restart. L3 publication is asynchronous and coalesced, due after 60
seconds or 8,192 new tokens by default. Interval `0` disables only the timer trigger; token growth
and orderly shutdown still persist the latest publication. Stored Responses and `previous_response_id` are separate
process-local state and still disappear on restart. Do not point multiple server processes at one
namespace; cross-process cache-root coordination is not implemented.

The published Qwen3.8 artifact SHA-256
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` is compatible only with cache
entries made from those exact artifact bytes. The on-disk cache manifest remains a development
format; version mismatches are ignored as misses and may require rotating the namespace.

Size L2 to hold the session heads that should switch without disk reads and L3 for durable heads,
history, and stable prefixes. L1 uses the active GPU KV pool, so increasing it can reduce useful
request headroom. The default 768 MiB/16 GiB/48 GiB profile is a starting point, not a guarantee.
Set a nonzero filesystem reserve in production. See the
[cache guide](continuation-cache.md) for tier TTL and persistence semantics, limitations, and
the complete metric list.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --request-log-jsonl profiles/bench/run/server.requests.jsonl
```

Every line is one `ninfer_serve_request_log` schema-22 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Generation request records retain that numeric `request_id` for metrics and
also carry `x_request_id`, matching the client-visible HTTP response header for log correlation.

| Event | Contents |
|---|---|
| `server_start` | target/weights identity and artifact, resolved Engine, registered thinking/non-thinking sampler defaults plus process overrides, thinking-history defaults, weights/sequence/workspace/request-transient arenas, KV sizing ledger, CUDA Graph observed/allowance bytes, CUDA/GPU environment, the adapter pool (`adapters`, as in [telemetry](#telemetry-and-events) without residency), the System One surface (`systemone`), and redacted argv |
| `request_start` | protocol, resolved sampler and seed, thinking modes, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call counts, unrounded phase seconds, complete speculative-decoding counters, and a structured `continuation_cache` diagnostic |
| `request_error` | the resolved request configuration and generation error message |
| `decision_start` | a submitted System One decision: `model`, the answering `adapter`, reuse flag, question/option counts, state/branch/longest-branch tokens, and state truncation |
| `decision_done` | output tokens, reused and computed state tokens with their `state_source`, branch passes, long-branch chunks, slot, unrounded `prepare`/`queue`/`restore`/`state`/`branch`/`execution`/`total` seconds, and state/branch prefill rates |
| `decision_error` | the submitted decision and the client-visible status and message |
| `throughput` | interval token deltas and rates with the System One share of computed prefill (`tokens.decision_prefill`), board energy, scheduler occupancy, decode-round batch statistics, cumulative/delta LoRA bank `stages` and `slot_waits` (`adapters`), and cumulative/delta continuation tier and latency summaries |

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `queue`, `restore`, `publish`,
`prefill`, `decode`, and `total` as full-precision JSON numbers. `publish` is only the completion
path's hand-off of the retained lane to the publication worker; the export itself runs there and is
reported through the throughput record's `continuation_cache.latency_microseconds` `export_*`
fields and the `continuation_export_*` metrics. Its `speculative` object contains
`backend`, `draft_window`, `rounds`, `drafted_tokens`, `accepted_tokens`, `fallback_steps`, and
`accepted_per_position`. Rates can be derived downstream from raw token counts and seconds instead
of rounded stderr strings.

`request_done.result.finish_reason` is always the engine's reason. The result also carries
`tool_call_count` and three [tool-call](#tool-calls) quality counters:

- `tool_calls_incomplete`: calls the output ended inside;
- `tool_calls_invalid_arguments`: closed calls whose arguments are not a JSON object;
- `tool_call_discarded_bytes`: non-whitespace bytes of non-call output after the first call,
  which were dropped.

The human completion line prints `finish=tool_calls` only when every call closed. Otherwise it prints
the engine's reason. It adds `tool_calls=`, and prints the three counters when they are nonzero.

`request_done.continuation_cache` reports stable `source` (`none`, `l1`, `l2`, `l3`), `alias_kind`
(`none`, `routed_session`, `stable_prefix`), and `final_miss_reason` names, plus lookup/preflight/restore
microseconds, restored tokens/bytes, destructive rollback, and completion-publication state. It never
contains the raw session key or internal stable alias. The corresponding human completion line uses
`cache_source=`, `cache_alias=`, `cache_miss=`, `cache_lookup=...ms`,
`cache_preflight=...ms`, `cache_restore=...ms`, restored tokens/bytes, rollback, and queued
publication state. L1 identifies resident GPU reuse, not a host restore. L2/L3 identify where the
selected image bytes were actually obtained. L3 persistence latency is asynchronous aggregate data
and is present only in throughput and Prometheus metrics, never in request completion latency.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. Structured request events cover successfully
prepared OpenAI Responses, OpenAI Chat, and Anthropic generation requests and errors during their
generation, and System One decisions once submitted; schema rejection
and token-count-only calls are not measurement requests and do not receive request IDs. A decision
record's `x_request_id` is its `x-typesafe-request-id`. `server_start.adapters.pool` gives each
adapter's `kind` and served `model_id`, and `server_start.systemone.binding` the decision adapter
`jev-latest` resolves to, so a replayed log reads both systems without a live server.

By default the server also reports aggregate activity every five seconds. `prefill` counts prompt
suffix tokens actually computed during the interval, excluding prefix-cache hits; the record's
`tokens.decision_prefill` is the part of it System One decisions evaluated, state chunks and branch
passes alike, and the remainder is chat prefill. `decode` counts tokens finally committed by decode
rounds, excluding the first token produced by prefill. For MTP and DFlash this is the accepted
committed output, not draft or rejected tokens. `avg_decode_batch` is decode row-rounds divided by
decode rounds during the same interval. The `running`, `prefilling`, `decode_ready`, and `waiting`
fields are the Engine scheduler snapshot at the end of the interval. Fully idle zero intervals are
omitted. The JSONL `throughput` event keeps the raw token and round deltas as well as derived
rates; downstream measurement should prefer those raw values.

The throughput `continuation_cache` object reports cumulative values and interval deltas for tier
restore counts/tokens/bytes, routed-session/stable-prefix selections, terminal miss reasons,
lookup/preflight/restore operation timing, L2 admission, L3 persistence, and publication outcomes;
`occupancy` reports current L1/L2/L3 entries/bytes plus cumulative L1 evictions/demotions. Counter
deltas tolerate a server/counter restart: when a current value is below the prior snapshot, the
current value starts the new interval instead of underflowing.

### Energy

The throughput `energy` object reports the interval's board energy, or `null` where the GPU exposes
no cumulative energy counter. Many GeForce boards do not implement one; `null` means the server
could not measure energy, which is not the same as measuring none, so it is never reported as zero.

`board_joules` is measured by the board itself and is exact. `prefill_joules` and `decode_joules`
are estimates: the executor brackets every execution unit with two instantaneous board-power reads
and integrates trapezoidally, because the cumulative counter costs milliseconds to read and advances
in roughly 100 ms steps, which is coarser than a decode round. `idle_joules` prices the part of the
interval that no execution unit claimed at `idle_watts`, itself measured over intervals during which
nothing ran and nothing was queued. `residual_joules` and `residual_fraction` are what those three
together fail to explain. The residual is published rather than distributed into a phase, so a
consumer can see the estimator's error instead of inheriting it silently.

`joules_per_token` carries four derived figures. `served` divides all board energy by all tokens,
including the idle draw between requests, and is what the work costs; it degrades on a mostly idle
server even when nothing about the engine changed. `active` removes the idle baseline and tracks the
schedule rather than the duty cycle. `prefill` divides prefill energy by computed prefill tokens, so
prefix-cache hits do not flatter the kernels. `decode` divides decode energy by committed decode
tokens, which under MTP counts accepted tokens only — that is where speculation shows up as a net
energy win or loss. Any of the four is `null` when its denominator is zero.

A watt-second is a joule, so tokens per watt-second and tokens per joule are the same quantity.
Energy is reported per token rather than as a rate because energy composes additively across phases
and a rate does not: joules-per-token figures can be combined against their own token counts,
whereas averaging tokens-per-joule arithmetically is wrong.

The record carries only joules per token. The per-million-token restatement the dashboard and CLI
also display is an exact rescale by `1e6/3600` into watt-hours, and is derived at display rather
than stored: it is the denominator inference is priced in, so multiplying it by a local electricity
rate gives a figure comparable to a published $/1M-token price, but it is the same measurement and
carrying both in the schema would be redundancy that can drift.

`GET /metrics` exports energy as counters only — `ninfer:board_energy_joules_total`,
`ninfer:prefill_energy_joules_total`, `ninfer:decode_energy_joules_total`,
`ninfer:energy_accounted_seconds_total`, `ninfer:energy_samples_total`, and the
`ninfer:board_idle_watts` gauge. The token denominators are already exported, so a scraper divides
two rates over a window it chose; the series are omitted entirely on a board with no counter.
`ninfer:board_energy_joules_total` counts energy since this server started, not since driver load,
and a driver-reload counter reset drops one difference rather than corrupting the total. `GET
/telemetry` additionally reports the raw board counter as `gpu.energy_joules_total`.

To verify interpretation, capture `/metrics`, send a cold request and a repeated routed request,
then compare the completion records and counters. Retained reuse reports L1; evicting that lane and
repeating reports L2; after persistence succeeds and the server restarts with the same compatibility
domain, the first restore reports L3. A mismatching prompt with the same routing key should report a
safe terminal reason such as `preflight_rejected`, without logging the key itself.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay. A
request joins that batch only after its single-request prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row.

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue.

### Repetition guard

A generation that locks into a repeating token cycle otherwise runs to its output-token limit. The
guard watches each request's committed tokens: a recurring n-gram proposes a period, the period is
then confirmed token by token against history, and the request is terminated once the confirmed run
covers `--repetition-guard-cycles` whole periods and at least `--repetition-guard-min-tokens`
tokens. Speculative drafts that were proposed and rejected are never observed, and no logits are
read, so a generation that does not lock into a cycle is unchanged by the guard being enabled.

Termination is reported as a server-side truncation in each protocol's own vocabulary, because none
of the three defines a repetition reason: Chat Completions returns `finish_reason: "length"`,
Responses returns `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`, and
Anthropic Messages returns `stop_reason: "max_tokens"`. The exact cause appears only in the
structured request log, as `finish_reason: "repetition_cycle"`.

Disable it with `--no-repetition-guard` when comparing against a reference implementation, since a
terminated cycle is a behavioral difference from an engine that has no guard.

### Generated text

The model generates byte-level tokens, so the bytes it writes need not be well-formed UTF-8: a
multi-byte character can be interrupted or cut short, and a token can carry a byte UTF-8 never uses.
Published text is always well-formed UTF-8. Each maximal subpart of an ill-formed sequence becomes
one U+FFFD replacement character, the substitution the Unicode Standard recommends and Python's
`errors="replace"` applies, and so does a character cut short when generation ends. The bytes alone
decide the text, not how tokens or decode rounds split them, so streamed deltas and a
non-streaming response agree; token IDs and usage counts are the tokens generated.

### Abandoned rounds

A decode round resolves each lane into a licensed prefix: how many of the tokens it produced are
committed, and whether the request ends there. Only a cancelled lane may commit nothing; every
other lane commits at least one token, and only a terminating lane may commit fewer than all of
them. A round whose rows disagree with the state that produced them is abandoned rather than
folded: the requests it was serving fail with an internal error, their lanes are aborted, and the
server keeps accepting work, including the retry a client is about to send.

The counter `decode_rounds_abandoned` in `/metrics` and in the telemetry stream reports how often
this happened. It is zero on a healthy server. Any sustained rate is a generation-state bug, and
the offending round writes the lane, the committed and produced counts, and the sequence frontiers
to the request log. Faults that outlive the round, such as a lost device context, still retire the
engine, after which every request is rejected with `503 service_unavailable` until it is
restarted.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` and the resolved `--kv-capacity` are independent limits. The former is each
sequence's logical ceiling; the latter sizes the shared Main Text KV pool used by all active
requests and retained prefixes. Both are represented with 64-token pages internally, while a
sequence can never cross the exact `--max-context` frontier. `--kv-capacity N` requests an explicit
capacity; `--kv-capacity auto` chooses the largest legal capacity that fits the memory remaining
after weights are loaded while keeping 1 GiB of sizing headroom. When omitted it follows
`--max-context`, preserving one full-length request's capacity. The shared pool is fixed at startup
and is not divided evenly among request lanes.

Automatic sizing evaluates the complete target runtime layout for the chosen concurrency, KV
dtype, speculative backend, draft window, Vision setting, workspace, and CUDA Graph allowance. It
uses a direct page-capacity calculation rather than allocation probing. Startup reports the policy,
resolved capacity, runtime reservation, free memory after weights, automatic headroom, planned
slack, actual free memory after complete startup, and observed Graph memory. An explicit capacity
is never silently reduced, and neither policy permits request-time pool growth.

Admission reserves the full prompt-plus-effective-output page entitlement, so an admitted request
can finish within its declared bound. A later request waits in FIFO order when the remaining shared
pages cannot satisfy its complete entitlement; the Engine never admits it and later truncates an
older request to recover capacity. Startup rejects a KV pool smaller than one sequence, too small to
provide one page per configured lane, or larger than all configured lanes could use.

Compatible resident prefixes are reused for both text and multimodal histories unless the server is
started with `--no-prefix-reuse`. A multimodal hit requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans; changing an earlier image or video
therefore resets the prefix instead of reusing placeholder-token KV. Media wholly inside a matched
prefix skips Vision execution, while new suffix media is encoded normally. The completion log
reports the reused token count as `cache=`.

The shared family runtime distinguishes `full_reset`, `append_frontier`,
`restore_turn_checkpoint`, and `restore_user_turn_anchor`. A turn checkpoint includes the recurrent
and selected speculative-backend continuation state required to recompute a rewritten suffix;
matching KV tokens alone never authorize a partial hit. The default `rolling-tool` policy replaces
that checkpoint at the latest generation opener after completed tool-call results, so serial tool
loops recompute only their newest suffix. `stable-turn` retains the first assistant opener after the
last real user query, preserving the earlier rewrite anchor used when closed-turn thinking is
removed. Both policies remain subject to exact prepared-prefix identity. The JSONL completion record
exposes the selected path as `prefix_reuse_path` and server-start records expose
`prefix_checkpoint_policy`.

A lane holds a second, independent anchor at the opener of the last real user query, restored as
`restore_user_turn_anchor`. Both checkpoint policies place their anchor after that message's
content, so a client that rewrites the tail of an earlier user message - for example one that moves
a synthetic reminder block onto whichever user message is currently last - invalidates every deeper
anchor and would otherwise recompute the whole history. The user-turn anchor sits upstream of that
edit and is stationary for the duration of a turn, so it is captured once when a new user query
arrives and retained across the tool loop that follows. It is held in host memory rather than a
device slot because it is read at most once per turn, and it is lane-local: continuation images
carry no user-turn anchor, and a restored image starts without one.
Changing reasoning effort changes the rendered prompt and therefore does not reuse a prefix whose
effort instruction differs.

A request's output limit is a limit, not a reservation. Admission holds KV pages for the prompt plus
a fixed decode window; pages beyond it are acquired per round as they are generated. A large
`max_tokens` therefore costs nothing when a turn is short, and it no longer crowds concurrent
sessions out of the shared KV pool. When several long generations compete for the last pages, the
Engine first reclaims a retained session and then, if that is not enough, ends the request at its
current length with `length`/ `max_tokens` — its session is still retained and published, so the
next turn of that conversation still restores. A single request is never shortened this way.

Speculative decoding is an engine option and does not change protocol output shapes, stop behavior,
or usage accounting. If a stop truncates a multi-token MTP or DFlash round, the Engine commits the
exact accepted target prefix so a following compatible turn can still reuse it. Output-limit and
context-capacity finishes map to `length`/ `max_tokens`; ordinary model or string stops map to
`stop`/ `end_turn`.

Function tools are rendered into the model prompt, and generated calls are parsed and streamed as
[Tool calls](#tool-calls) describes.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
