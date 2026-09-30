# NInfer CLI

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.
The default build contains Qwen3.8-27B. Commands using Qwen3.6-35B-A3B require configuring with
`-DNINFER_BUILD_QWEN3_6_35B_A3B=ON`.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt`, `--messages`, and `--systemone` is required.

Answer content is streamed to stdout. Reasoning, model loading (including the registered target and
canonical `weights_id`), timings, throughput, board energy, GPU memory, and speculative-decoding
statistics are written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. If the chat template embedded in the loaded artifact exposes
reasoning effort, `--reasoning-effort low|medium|xhigh` selects it; omitting the option uses the
template's default. An artifact whose template does not expose effort rejects the option. Add
`--no-thinking` for direct-response prompt rendering; it cannot be combined with
`--reasoning-effort`. `--greedy` selects exact argmax decoding independently.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash weights and state and the optimized proposal head;
- `--spec mtp` loads only MTP, while `--spec dflash` loads only the 35B-A3B text-only DFlash
  backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights, Vision scratch phase, and frozen
  request-transient allocation;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. DFlash and Vision are mutually
exclusive. The default speculative and Vision settings produce the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`. Message content
may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

A text part may carry `"prompt_cache_breakpoint": {"mode": "explicit"}` and the object form may
set `"prompt_cache_options": {"mode": "implicit" | "explicit"}`, with the same placement and
mode semantics as the server's [prompt cache breakpoints](serving.md#prompt-cache-breakpoints).
They publish and restore continuation images only when a continuation cache with L2 or L3 is
selected.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## System One decisions

`--systemone FILE` answers TypeSafe System One request bodies with the decision adapters of the
`--lora-dir` pool, exactly as the server's
[`POST /systemone/v1/systemone`](serving.md#system-one-decisions) does. `-` reads stdin, and
`--systemone-jsonl` treats each nonblank line as one request:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --lora-dir lora --systemone requests.jsonl --systemone-jsonl --max-context 16384
```

Each response body, or `{"detail": ...}` for a failed request, is printed on its own stdout line,
and the process exits nonzero when any request failed. `jev-latest` answers with
`--systemone-default NAME`, or with the pool's only decision adapter. `--systemone-cold` computes
every state from zero without restoring or retaining one. For parity tooling,
`--systemone-probabilities` prints `{"status", "response", ...}` lines that add the unrounded
probabilities and the engine's accounting, and `--systemone-dump-prepared FILE` writes each
decision's token layout.

## Speculative decoding

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
35B-A3B text-only DFlash backend with one to fifteen. `--lm-head-draft` selects the optimized
proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

MTP and DFlash cannot be enabled together. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with
the optimized proposal head. DFlash accepts up to fifteen draft tokens; seven is the current
measured recommendation rather than a semantic limit.

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk2v4-e8` | KV-cache storage; the Hadamard-rotated modes trade key/value precision for capacity | `bf16` |
| `--spec mtp\|dflash` | speculative backend | off |
| `--draft-tokens N` | MTP `1..5`; DFlash `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-repetition-guard` | let a generation that has locked into a repeating cycle run to its output limit | guard on |
| `--repetition-guard-window N` | how far back a repeat may reach, which also bounds the detectable period | `512` |
| `--repetition-guard-ngram N` | token n-gram whose recurrence proposes a period | `24` |
| `--repetition-guard-cycles N` | whole periods confirmed before the cycle counts as locked | `3` |
| `--repetition-guard-min-tokens N` | confirmed tokens required regardless of period | `64` |
| `--prefix-checkpoint-policy stable-turn\|rolling-tool` | choose the stable turn-rewrite anchor or rolling completed-tool frontier | `rolling-tool` |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--reasoning-effort low\|medium\|xhigh` | select an effort exposed by the loaded chat template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the official general-task preset registered for
the loaded model and the rendered prompt mode. The current presets are:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Qwen's separate precise-coding recommendation
is task-specific and is therefore an explicit override rather than an inferred Engine default.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## Tiered continuation cache

The native one-shot CLI defaults continuation tiers to `off`. Explicit continuation-cache tuning
enables retained GPU plus host-RAM tiers (`l1-l2`). Because one CLI invocation runs one request and
exits, L1/L2 mostly benefit work within that invocation.
Add a directory to enable L3 and reuse an automatically identified stable system/tool prefix across
later invocations:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages request.json --max-context 32768 --max-new 256 \
  --continuation-cache l1-l2-l3 \
  --continuation-cache-dir "$HOME/.cache/ninfer/continuations" \
  --continuation-cache-namespace native \
  --continuation-cache-l2-mib 16384 \
  --continuation-cache-l3-mib 49152 \
  --continuation-cache-filesystem-reserve-mib 8192
```

Supplying `--continuation-cache-dir` without `--continuation-cache` selects `l1-l2-l3`; another
explicit `--continuation-cache-*` tuning flag without a directory selects `l1-l2`.
The defaults are 768/16,384/49,152 MiB for L1/L2/L3; 600 seconds for retained L1; 7,200 seconds
for an L2-only image; 86,400 seconds for an L3-mode catalog image; persistence after 60 seconds or
8,192 new tokens; namespace `local`; filesystem reserve `0`; and history depth `4` including the
current session head. Use a positive disk reserve operationally.

The CLI does not accept `prompt_cache_key`, so it does not publish a mutable named session head.
Stable leading prefixes are shared by exact content-derived aliases. Every load checks complete
artifact/runtime compatibility and exact prepared-prefix identity; a stale, corrupt, or nonmatching
entry safely becomes cold prefill. The cache stores continuation state, not generated results.

The published Qwen3.8 artifact SHA-256 is
`eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`; complete artifact SHA-256 is
part of cache compatibility. Cache manifests are currently development-versioned and may be ignored
by a later build, in which case remove or rotate the namespace. See
[Tiered continuation cache](continuation-cache.md) for all flags, permissions, sizing, and format
status.

## Energy

The summary reports `board energy` for the request plus `energy per token`, the same figure restated
as `energy per 1M tokens` in watt-hours, and a `prefill` and `decode` split in joules per token. A
watt-second is a joule, so tokens per watt-second and tokens per joule are the same figure; energy is
reported per token because it composes additively across phases while a rate does not. The
per-million restatement is an exact rescale by `1e6/3600` into the denominator inference is priced
in, so multiplying it by a local electricity rate gives a number comparable to a published
$/1M-token price.

The total is the board's own cumulative energy counter, read either side of the request, so it is
exact. The phase split is integrated from instantaneous power at execution-unit boundaries and is an
estimate; `energy unattributed` is the share of the measured total it does not account for. Prefill
divides by tokens actually prefilled, so reused prompt tokens do not flatter it.

These lines are omitted on a board with no cumulative energy counter, which many GeForce parts lack.
Per-request energy is exact here only because the CLI runs one request at a time; board energy is a
property of the device, so the server reports it per interval instead. It is also board energy, not
system energy: the CPU, the rest of the platform, and power-supply losses are excluded.

## Context and memory

The registered model IDs have a native context limit of 262,144 tokens. The practical
allocation on one RTX 4090 depends on the selected artifact, media workload, output budget, and
KV-cache type.
Use `--kv-dtype int8` for the maximum-precision quantized profile. The compressed modes rotate
every 64-element key and value group with an orthonormal Hadamard transform and store 4-bit
values:

- `rk8v4` keeps 8-bit keys.
- `rk4v4` packs keys to 4 bits as well. This is the recommended long-context mode on the RTX 4090
  and fits the full native 262,144 context.
- `rk2v4-e8` packs keys to 2 bits with a 240-root E8 codebook, for maximum capacity.

Every 4-bit plane uses one midrise codec: its sixteen levels are the odd multiples of the group's
FP16 scale `h = max|x| / 15`, symmetric about zero with no zero level, so both extremes of a group
decode to its absolute maximum. None of these modes is byte-equivalent to INT8. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, workspace, Vision
request transient, and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program scratch arena, the maximum Vision request-transient buffer when Vision is enabled, and a
separate CUDA Graph driver allowance. Scratch is the maximum of the enabled Text, MTP, DFlash, and
Vision phases, not their sum. Its prefill bound uses
`min(--prefill-chunk,--max-context)`. The request-transient buffer is also frozen at startup; a
media request activates only the needed prefix and performs no project-owned device allocation or
growth.

All weight, sequence, workspace, request-transient, and graph allocations are released when the
Engine is destroyed.
