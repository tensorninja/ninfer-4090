# Single-GPU serving performance

Tested Git revisions:

- Concurrent MTP3 decode saturation for the three measured artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.8-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.8-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.8-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.8-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

The serving measurements characterize the two measured model IDs independently on one
NVIDIA GeForce RTX 5090. A separate RTX 4090 (`sm_89`) long-context prefill section appears at the
end of this file and is not comparable to the RTX 5090 tables. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The 27B
results report its `groupwise-int` and `nvfp4` weight profiles separately. The concurrent
decode-saturation campaign measures the same three artifact profiles at C=1, 2, 4, and 8.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.8-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.8-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.8-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.8-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.8-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.8-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.8-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured groupwise-int artifacts to run the complete published
MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_8_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_8_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## RTX 4090 (`sm_89`) long-context prefill

The sections above characterize one RTX 5090. This section is a separate hardware target: the
`sm_89` build of `qwen3.8-27b/groupwise-int` on one NVIDIA GeForce RTX 4090, and it reports prefill
throughput only.

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, 480 W power limit (~2,715 MHz sustained) |
| Toolchain | CUDA 13.2, `CMAKE_CUDA_ARCHITECTURES=89`, Release |
| Artifact | `qwen3.8-27b/groupwise-int` |
| KV dtype | `rk4v4` (the INT8-route ladder below: its predecessor `rk4v4-e8`) |
| Prompt | `scripts/prefill_probe.py --targets 102400`: 118,242 tokens (the ladder: 115,125) |

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 262144 --kv-capacity 262144 --prefill-chunk 1024 \
  --kv-dtype rk4v4 --continuation-cache off --no-prefix-reuse
```

Prefill rate is read from the structured request log, as `request_done.result.computed_prefill_tokens`
over `request_done.timings_seconds.prefill`, not from HTTP round-trip timing, so prepare and vision
time are excluded. Three hazards invalidate a measurement on this card if ignored: the continuation
cache must be off, because a repeated prompt otherwise restores state instead of prefilling; SM clock
and power should be logged, because sustained prefill on this card can throttle and be misread as a
kernel regression; and for any energy claim the board's thermal state must be controlled, because
efficiency drifts as the card heats and will otherwise alias onto whichever axis is being swept.
Repeat measurements of one configuration reproduce to within 0.12%.

### Result

INT8 tensor-core prefill routes (`mma.m16n8k32.s32.s8.s8.s32`) replaced BF16 routes on all six
dense body GEMM Ops, in three stages:

| Configuration | Prefill | Cumulative |
|---|---:|---:|
| BF16 routes | 1,739.0 tok/s | 1.000× |
| INT8 `linear_swiglu` (Q4) | 2,035.9 tok/s | 1.171× |
| INT8 `linear_add` (Q5) | 2,214.9 tok/s | 1.274× |
| INT8 `attn_input_proj` + `gdn_input_proj` | **2,496.1 tok/s** | **1.435×** |

All four rows are measured on the current build. An earlier ladder taken before the causal-boundary
split in the attention prefill kernel read 1,701.4 / 1,976.4 / 2,147.3 / 2,409.2 tok/s (1.415×
cumulative); that split lifted both arms, the BF16 baseline by 2.2% and the INT8 build by 3.6%.

`--prefill-chunk 2048` adds a further 0.6%; 4096 does not improve on it.

The ladder used the E8-lattice `rk4v4-e8` KV mode, which the midrise `rk4v4` codec has replaced
([docs/cli.md](cli.md)). On the probe's current 118,242-token prompt, measured back to back
(2,714-2,715 MHz mean SM clock, 461-465 W), the last `rk4v4-e8` build prefills at 2,520.6 tok/s and
the `rk4v4` build at **2,577.9 tok/s** (+2.3%).

### Activation-compute profile

The INT8 routes quantize the activation to group-64 symmetric INT8, which is a declared semantic
boundary rather than a bit-exact transform of the BF16 route. It is admitted in prefill only:
decode and speculative verify keep the BF16 catalog and are bit-identical to the previous build.

Because the projections feed the attention scores and the FP32 GDN recurrence, the profile does
move greedy output. Screening against a BF16-projection build on the same configuration: 7 of 12
short prompts produced byte-identical output, and all five divergences were paraphrases with no
factual or arithmetic error. A retrieval probe at a 139,910-token prompt placed a distinct code at
five depths and both builds retrieved 5/5 with byte-identical answers. AIME/GPQA were not re-run
for this profile, so task-level reasoning accuracy under it is unverified.

Prefix reuse reproduces the input semantics of full prefill, not its arithmetic. The two paths
decompose a prompt into different prefill calls, so the FP32 GDN recurrence accumulates over
different chunk boundaries; they were never bitwise identical, and this profile made the difference
observable in generated tokens.

### Prompt-length characteristics

The INT8 routes are registered over every prefill token count, so short prompts take them where the
tuned small-`T` BF16 routes were faster:

| Prompt tokens | BF16 projections | INT8 projections |
|---:|---:|---:|
| 99 | 185.5 tok/s | 177.3 tok/s |
| 229 | 447.2 tok/s | 434.1 tok/s |
| 770 | 1,311.4 tok/s | 1,334.5 tok/s |
| 2,827 | 2,316.0 tok/s | 2,535.7 tok/s |
| 8,105 | 2,760.0 tok/s | 3,186.5 tok/s |

Break-even is near 770 tokens and the worst case is about 15 ms on a 229-token prompt. A token-count
threshold would recover it, but it would also make a token's projection output depend on the width
of the call that produced it, which widens rather than narrows how far prefix reuse can drift.

### Concurrent agent serving: closed-loop clients bound TTFT

The swarm workload for this target is a closed loop: a fixed number of agents each wait for their
response before sending the next request. Queue depth is therefore not a free variable. By Little's
Law the steady state satisfies `N_agents ≈ throughput × response_time`, so response time is pinned
to `N_agents / throughput` and any capacity that is freed is immediately consumed by the same
agents. Measured with ten agents against four lanes:

| Change | Execution-thread effect | Throughput | TTFT p50 |
|---|---|---|---|
| baseline | admission 23.5% of thread | 12.3 req/min | ~24.8 s |
| restore decode moved off the execution thread | admission 9.9% of thread | 13.4 req/min | ~25.3 s |

Restore cost per request fell from 0.31 s to 0.11 s and the execution thread went from roughly 77%
to 90% GPU-bound, yet TTFT did not move: 84–88% of it is queueing, and the queue refilled. Two
consequences for measurement. A serving change must be reported at the level it actually acts on —
execution-thread share and throughput here, not TTFT — because a closed-loop client will absorb a
throughput gain and report an unchanged latency. And a TTFT target under this workload can only be
met by raising throughput or lane count; removing CPU work from the execution thread stops paying
once the thread is GPU-bound, which on this target it now is.

Attribution must also be taken at the level being claimed. `restore_microseconds` covers the whole
import call, and `preflight_microseconds` is logged separately; comparing the two is what localizes
a restore regression. A cost model for the transfer alone is not a substitute: on this card
37 KB pinned copies already reach 8.87 GB/s against 13.16 GB/s for 8 MB copies, so device transfer
was never the dominant term in a restore, and host-side decoding was.

### Admission during prefill

Before the prefilling set, the executor admitted a pending request only at the prefill-unit
boundaries of the single lane it was prefilling and never started a second prefill while one was
in flight. On the live log of 1,713 requests (prompt p50 44k / p90 131k tokens, three lanes), 76%
of all request queue time passed while a lane was free, and a 131k-token prompt arriving behind
another waited 44–89 s before its own prefill began. The executor now keeps a set of prefilling
lanes, issues each chunk to the lane with the fewest remaining prompt tokens, and admits one
pending request per prefill unit whenever a lane is free
(`docs/maintainer/concurrent-inference-architecture.md` §7.3). Two drivers under `tools/bench/`
measure it from the request log:

| Driver | Workload | Result |
|---|---|---|
| `run_admission_latency.py` | 131k-token prompt, then an 8k-token prompt 5 s later | short: queue 0.07 s, TTFT 2.4 s (2.3 s own prefill); long: 159k prompt tokens prefilled in 69.5 s with the short request's prefill and 256-token decode interleaved |
| `run_session_swarm.py` | 8 `prompt_cache_key` sessions over 3 lanes, 6 turns, 24k-token opening turn, `--continuation-cache l1-l2` | queue time with a free lane 2.6% (none during a prefill); 13.8 req/min; L2-hit turns TTFT mean 23.6 s = queue 22.0 + prepare 0.40 + restore 0.15 + prefill 0.97 |

The swarm is the closed loop described above, so its TTFT is still queueing; what the change
removes is the head-of-line wait behind a long prefill. At this point the 1.39 s mean publish of
each turn's checkpoint still ran synchronously on the execution thread after the last token; the
next section moves it off.

### Prompt preparation

`timings_seconds.prepare` on the live log was 1.07 s p50 and 4.2 s p90, about 20 µs per prompt
token, while the tokenizer alone encodes 750k tok/s. The cost was boundary resolution: every
template boundary (one assistant opener per turn, the client breakpoints, and the turn-rewrite and
user-turn frontiers) was validated by re-tokenizing the rendered prefix up to its byte cut and
comparing it with the complete encoding, so a chat with `M` assistant turns tokenized its prompt
about `M/2` times over. `Tokenizer::encode_segmented` now keeps the added-token segmentation of
the one complete encoding: added tokens split the text into independently encoded segments, so
the encoding of any byte prefix shares every segment before the cut and only the partial segment
holding the cut is re-encoded. The boundary contract is unchanged and
`ninfer_qwen3_8_tokenizer_real_test` checks the result against the re-tokenizing reference on a
222k-token, 26-boundary chat (4.6 s reference, 0.30 s now, 0.29 s for the plain encode).

Measured on the live server with `max_tokens=1` requests:

| Messages | Prompt tokens | `prepare` before | `prepare` after |
|---:|---:|---:|---:|
| 20 | 40,171 | 0.55 s | 0.09 s |
| 20 | 131,872 | 1.69 s | 0.22 s |
| 60 | 42,108 | 1.32 s | 0.09 s |
| 60 | 136,790 | 3.96 s | 0.20 s |

### Continuation export off the execution thread

A completed `prompt_cache_key` turn publishes the lane's continuation image, and the export that
produces it - about 700 MB of KV pages, GDN state and hidden vectors for a 40k-token session -
ran on the execution thread between GPU units: `publish` was 0.95 s p50 / 2 s p90 on the live
log and 1.39 s mean in the swarm, and every other lane made no progress for its duration. The
completion path now only records a fence event on the engine stream and hands the lane to the
publication worker, which exports it over the Program's own stream and pinned ring while the
engine stream keeps decoding, then publishes the image as before. The lane stays retained and is
not rewritten until the export returns; lane choice ranks such a lane behind every settled lane
so only the same session's next turn ever waits on it
(`docs/maintainer/concurrent-inference-architecture.md` §2.6).

`run_session_swarm.py`, same configuration as the row above (8 sessions, 3 lanes, 6 turns,
24k-token opening turn, `--continuation-cache l1-l2`), before and after; the after run also
carries the prompt-preparation change:

| | before | after |
|---|---:|---:|
| throughput | 13.8 req/min | 17.3 req/min |
| wall time, 48 requests | 238.6 s | 192.5 s |
| `publish` mean on the request | 1.39 s | 11 µs |
| export on the worker, 50 exports | - | 0.37 s mean (18.5 s total) |
| L2-hit turn TTFT mean | 23.6 s = queue 22.0 + prepare 0.40 + restore 0.15 + prefill 0.97 | 18.2 s = queue 17.0 + prepare 0.08 + restore 0.11 + prefill 0.98 |
| per-request decode rate p50 | 51.7 tok/s | 51.1 tok/s |

The 48 synchronous exports had held the execution thread for about 65 s of the 239 s run; the
queue absorbed most of that, as the closed-loop model predicts, and TTFT fell with throughput.
The per-request decode rate did not move because it is set by the three-lane batch, not by the
stalls, which landed between requests. Export cost is now visible as
`continuation_export_microseconds`/`operations` rather than as a phase of the request.

### Batched decode: the small-T linear routes

With MTP K=3 every decode round verifies four positions per lane, so the projections run at
T = 4·B: 4, 8 and 12 for one to three lanes. Before this work, aggregate decode measured through
the request log (`tools/bench/run_decode_scaling.py`, 8k-token prompts, `--continuation-cache
off --no-prefix-reuse`) was 98 → 151 → 172 tok/s for B = 1, 2, 3, with the round growing from
24.0 ms to 31.5 ms to 43.9 ms against a weight-streaming floor near 24 ms. The Op benchmarks
(`bench/ops`, warm L2, medians) located the growth in the T=2..16 SIMT routes, which above T=8
were bound by SIMT instruction issue (activation loads and dequantization per FMA, 11–14
TFLOP/s), not by DRAM: summed over 48 GDN and 16 attention layers the four decoder projections
took 21.7 ms of the B=1 round and 38.9 ms of the B=3 round.

The four projections now share one tensor-core small-T mechanism
(`src/ops/common/small_t_rowsplit_mma.cuh`): a CTA owns 64 weight rows and streams them in
two-group stages with the `cp.async ... .L2::256B` prefetch hint (without it the 64-byte row
runs of the RowSplit layout are DRAM activate-rate bound at about 250 GB/s), decodes the Q4/Q5
codes to exact BF16 integers for the `mma.sync` fragments, accumulates each group in FP32 and
applies the FP16 scale in FP32, and stages the activation tile once per CTA. Split-K over
`gridDim.y` with FP32 partials and a fixed-order last-arriver fold keeps the 5120-row weights
(K = 17408 and 6144) on a full wave; the K = 5120 projections need no split. Each Op keeps its
own epilogue (residual add, fused SiLU·up, or the two-output store of the split parents) and the
route is registered over the whole T ≤ 32 interval under `A16Only`, so a token's result does not
depend on the width of the call. The same shapes now measure:

| Op (route at T≤32) | T=1 | T=4 | T=8 | T=12 | T=16 |
|---|---:|---:|---:|---:|---:|
| Q4 gate/up + SwiGLU 34816×5120 (`small_t.mma.r32x2.swiglu`) | 144 µs | 151 (1.05×) | 155 (1.07×) | 158 (1.10×) | 159 (1.10×) |
| Q5 down + residual 5120×17408 (`small_t.mma.r64.splitk.residual`) | 94 | 98 (1.04×) | 98 (1.04×) | 99 (1.05×) | 100 (1.06×) |
| Q5 out + residual 5120×6144 (`small_t.mma.r64.splitk.residual`) | 40 | 39 (0.98×) | 39 (0.98×) | 40 (1.00×) | 41 (1.03×) |
| attention input projection Q4/Q5 (`small_t.mma.r64.store`) | 29 | 28 (0.97×) | 28 (0.97×) | 35 (1.21×) | 35 (1.21×) |
| GDN input projection Q4/Q5 (`small_t.mma.r64.store`) | 33 | 33 (1.00×) | 33 (1.00×) | 39 (1.18×) | 39 (1.18×) |

(Ratios are against the previous T=1 route; the two input projections now beat their old T=1
gemv pairs, so T=1 was moved onto the new route as well. The Q4/Q5 GDN conv snapshot at B=1
likewise composes the projection with the shared conv kernel at every width, 36 µs against
42–80 µs for the fused projection-epilogue schedule it replaces.) On the same request-log
protocol aggregate decode is now 94 → 200 → 299 tok/s for B = 1, 2, 3 with rounds of 24.6, 26.5
and 28.8 ms: the B=3 round is 1.17× the B=1 round instead of 1.83×, and the three-lane aggregate
is 1.74× what it was.

Two variants were measured and rejected on the way. A stream-K persistent grid (256 resident
CTAs walking tile-major stage ranges with a deferred fold) was 5–12% slower than split-K at
every configuration tried, and two-way split-K on the K = 5120 projections was slower than a
single full wave (39 vs 33 µs at T=2). The remaining growth from T=8 to T=16 is tensor-pipe
overlap, not bandwidth.

Two related settings were measured on the same protocol and left as they are. MTP K=4 loses to
K=3 at every batch (B=1/2/3: 88/130/163 tok/s against 98/151/172): the extra verify position adds
more to the round (26.5 vs 24.0 ms at B=1) than its acceptance gain returns (2.43 vs 2.36 tokens
per round). And a rank-16 LoRA adapter over seven sites (`cyberstrike-1.5-6500`) costs 1.8 ms
per round - 7.5% at B=1, 4% at B=3 - which `nsys` attributes to 144 `lora_down`/`lora_up` pairs
per forward rather than to bandwidth (the adapter is 84 MB). The down kernel ran on at most 16
split-K blocks and took 8.5 µs median; raising the split cap to 64 blocks
(`src/ops/launcher/lora_delta.h`) brings the pair from 15.2 µs to 11.1 µs, about 0.6 ms per
round. The remaining cost is the two launches' latency chains, which a fused single-plane
reduction could roughly halve again; that is the next LoRA step if adapter decode matters more
than the batched-decode route above.

### System One decision latency

Decisions were measured on the same card and toolchain with the decision adapter
`systemone-decision-v7` (rank 16 on the six registered modules, plus its pointer head). The KV
cache was BF16, the configuration decisions are qualified on
([System One fidelity](serving.md#system-one-fidelity)), with one lane and a 73,728-token context.
`scripts/decision_probe.py` drives `/systemone/v1/systemone` and reads every decision's
`decision_done` record. Latency is the response's `latency_ms`: the restore of a cached state, if
any, plus execution, which splits into state and branch time. A point is a state of pseudo-text
calibrated to a token count, and Q questions that cycle `noul`, a four-option `choice`, and a
three-option `score`. Q = 1, 6 and 16 give 2, 18 and 47 options over 17, 126 and 334 branch tokens,
each in one pass. Each point is warmed once and measured five times. The tables give medians; p90
is within 1% of the median except where noted. The SM clock held 2,625-2,715 MHz. The cold 8k and
65k states reached the 480 W power limit.

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --lora-dir lora --kv-dtype bf16 \
  --max-context 73728 --kv-capacity 73728 --max-concurrency 1 \
  --continuation-cache-l1-mib 8192 --request-log-jsonl decisions.jsonl
python3 scripts/decision_probe.py --url http://127.0.0.1:8080 --model <decision adapter> \
  --log decisions.jsonl --mode cached
```

With a cached state, the state is computed once and every repeat continues it, so only the branches
run:

| State tokens | Q = 1 | Q = 6 | Q = 16 |
|---:|---:|---:|---:|
| 301 | 24.8 ms | 97.4 ms | 191.3 ms |
| 8,205 | 25.7 ms | 101.0 ms | 199.6 ms |
| 64,991 | 29.9 ms | 123.9 ms | 260.3 ms |

Branch time follows the branch tokens, not the state. A pass streams the weights once over all of
its columns, so one question's 17 tokens sit at the weight-streaming floor of about 25 ms, and more
questions add BF16 compute. A 65k-token state adds only 5-69 ms over a 301-token one.

Before the segmented mixer Ops, a pass ran attention, the convolution and the GDN recurrence once
per question. The 64,991-token row then measured 95.5, 519.9 and 1,319.3 ms, so the segmented Ops
make it 3.2x, 4.2x and 5.1x faster. In isolation, one layer's segmented attention over a
65,535-token BF16 state with six 40-token questions takes 3.31 ms, against 26.6 ms for the
per-question loop.

The 64,991-token row keeps its state in L1 only because of `--continuation-cache-l1-mib 8192`. At the
default 768 MiB, the state's 4.1 GiB of BF16 KV and GDN state exceeds the L1 budget as soon as its
lane goes idle. Every repeat then restores the published image from L2 in 2.85-2.96 s, including
the verification every import performs, and adds the same branch time. That gives 2.88, 3.08 and
3.20 s for Q = 1, 6 and 16, with a p90 of 3.35 s at Q = 1. Size the L1 budget to the decision
states that should stay resident.

A cold state (`--continuation-cache off --no-prefix-reuse`), with Q = 1:

| State tokens | Latency | State prefill | State rate |
|---:|---:|---:|---:|
| 301 | 206.9 ms | 181.7 ms | 1,657 tok/s |
| 8,205 | 3.78 s | 3.75 s | 2,186 tok/s |
| 65,006 | 34.77 s | 34.74 s | 1,871 tok/s |

With more questions, a cold decision adds the cached branch time from the first table. Its state
prefill uses the BF16-activation routes, because `TextPhase::Decision` admits no INT8 activation
route. Those are the routes the prefill ladder above starts from (1,739 tok/s at 115k tokens),
not the INT8 routes that take chat prefill to 2,496 tok/s. Decisions give up that throughput to
stay within kev's per-question tolerance.

### Chat and System One together

One `ninfer-serve` process served chat and decisions at once on the same card and toolchain,
2026-09-30: BF16 KV, MTP with three draft tokens, four lanes, a 32,768-token context, prefix reuse
and the continuation cache at their defaults, and a two-slot adapter bank. The pool held
`systemone-decision-v7` and one generative adapter, a rank-16 style adapter on the same seven sites,
so both stayed resident and no measured phase swapped. `scripts/dual_load_probe.py` ran three 90 s
phases. Decisions alone is a closed loop: each decision is submitted as soon as the previous one
returns, with six questions over one 8,162-token state that stays retained. Chat alone is two greedy
streams with thinking off, the base weights writing 1,024 tokens per request and the adapter 519.
Together runs both; its chat streams stop first and finish their requests beside the decisions, so
every measured chat request ran beside decisions from start to end. Every figure comes from the
request log. The SM clock held 2,715 MHz throughout, and board power peaked at 383 W.

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --kv-dtype bf16 --spec mtp \
  --draft-tokens 3 --lm-head-draft --max-concurrency 4 --max-context 32768 \
  --lora-dir <chat and decision adapters> --lora-slots 2 --request-log-jsonl dual.jsonl
python3 scripts/dual_load_probe.py --url http://127.0.0.1:8080 --log dual.jsonl \
  --chat-adapter <chat adapter>
```

| Phase | Decisions | Latency p50 | End-to-end p50 | Decisions/s | Swaps | Slot waits |
|---|---:|---:|---:|---:|---:|---:|
| Decisions alone | 765 | 100.9 ms | 116.3 ms | 8.50 | 0 | 0 |
| Together | 757 | 100.9 ms | 125.8 ms | 7.55 | 0 | 0 |

| Chat stream | Output rate alone | Output rate together | Decode-round rate alone / together | MTP acceptance |
|---|---:|---:|---:|---:|
| Base weights | 104.1 tok/s | 21.6 tok/s | 105.1 / 105.2 tok/s | 60.3% |
| Chat adapter | 127.6 tok/s | 26.5 tok/s | 127.9 / 128.0 tok/s | 80.3% |

Latency is the response's `latency_ms` and end-to-end adds the queue wait; p90 is within 1% of the
median. The output rate is completion tokens over the wall time after the first token, what a
streaming client sees. The decode-round rate divides the same tokens by the seconds of the decode
rounds the request joined, the dashboard's decode rate.

A decision never joins a decode batch, and the decode rounds show it: each stream's round rate and
MTP acceptance are the same with and without decisions. Every chat response, 13 on the base weights
and 29 on the adapter across the chat-alone, together and cold phases, is byte-identical to its
stream's first measured response. Midway through the combined phase `/telemetry` showed three lanes
busy at once: a decision on the decision adapter, the base stream, and the adapter stream, with both
adapters in the bank.

What decisions take from chat is time on the one execution thread. The scheduler alternates units:
after each decode round, a waiting request is admitted and runs its first unit, and a decision on a
retained state is a single 101 ms unit with all six branches in one pass. Back to back, the
decisions held the thread for 7.55 × 100.9 ms, about 76% of every second, and each chat stream got
one decode round per decision. Its output rate fell to 21% of its rate alone: a 1,024-token answer
took 48 s instead of 9.9 s. The decisions lost less. Their latency is unchanged; waiting for the
decode round between them added 9.5 ms end to end and cost 11% of their rate. A lighter decision
load takes proportionally less, because chat gets whatever time the decisions leave.
`--prefill-decode-balance` does not move this split. It spaces the chunks of a request that is still
prefilling, and a decision on a retained state completes within the unit that admits it.

A cold decision prefills its state first (3 per condition, six questions):

| Cold 8K decision | Latency | State prefill | State rate | Branches |
|---|---:|---:|---:|---:|
| Idle | 3.98 s | 3.88 s | 2,159 tok/s | 101.1 ms |
| Beside the two chat streams | 4.15 s | 3.80 s | 2,165 tok/s | 100.9 ms |

The state spans several prefill units and a decode round runs between them, so beside chat a cold
decision takes 4% longer while its state rate is unchanged.

### Energy measurement

Energy per token is reported alongside throughput. A watt-second is a joule, so tokens per
watt-second and tokens per joule are the same quantity; the engine reports joules per token, because
energy composes additively across phases and a rate does not.

Two NVML readings back it, and they have very different costs and resolutions. Measured on this
board under driver 610.57.04:

| Reading | Cost per call | Resolution | Role |
|---|---:|---|---|
| `nvmlDeviceGetTotalEnergyConsumption` | 2.4 ms (p50), 6.6 ms (max) | ~100 ms steps | exact interval total |
| `NVML_FI_DEV_POWER_INSTANT` | 0.51 µs (mean), 0.79 µs (p99) | ~50 Hz board refresh | per-unit phase attribution |
| `nvmlDeviceGetPowerUsage` | 1.6 µs | trailing 1 s average | not used |

The cumulative counter is exact but cannot resolve a phase: a decode round is shorter than the
counter's own step, and reading it costs more than a tenth of a round. It is therefore sampled only
by the serving reporter thread, once per `--log-stats-interval-ms`. Differenced against integrated
power it is accurate to −26.2% over a 0.5 s window, −12.3% over 1 s, −3.7% over 2 s, and −0.0% over
5 s, which is why the default 5 s interval is the shortest sound window for an energy figure.

Phase attribution instead brackets every execution unit with two instantaneous power reads and
integrates trapezoidally, which is affordable on the execution thread at 0.5 µs against units costing
tens of milliseconds. `nvmlDeviceGetPowerUsage` is not used at all: on Ampere and newer it returns a
trailing 1 s average, which lags a prefill/decode transition by far more than the phase it would be
attributing.

That estimator is checked directly against the counter as its oracle. Replaying the executor's exact
arithmetic over a sustained cuBLAS load, at controlled unit sizes matching both phases:

| Unit size | Window | Exact counter | Estimator | Residual |
|---|---:|---:|---:|---:|
| 25 ms (decode round) | 5.8 s | 1,722.5 J | 1,641.6 J | +4.70% |
| 25 ms | 5.8 s | — | — | +1.76% … +3.49% |
| 150 ms (prefill chunk) | 5.9 s | 1,821.4 J | 1,760.3 J | +3.36% |
| 25 ms | 30.8 s | 9,185.5 J | 9,132.7 J | **+0.57%** |

The residual shrinks with window length, from 2–5% over six seconds to 0.6% over thirty. This is the
expected behavior of a ~50 Hz sample against sub-refresh units and is why only the aggregate is
claimed, never a single unit. The server publishes the residual per interval as
`energy.residual_fraction` rather than folding it into a phase.

### Measured energy

From a live `ninfer-serve` under agent traffic: `qwen3.8-27b/groupwise-int`, `rk4v4-e8` KV,
262,144-token context, `--max-concurrency 3`, `--prefill-chunk 1024`, `--spec mtp --draft-tokens 3
--lm-head-draft`, continuation cache `l1-l2`. 97 reporting intervals covering 207.7 kJ, 374,807
computed prefill tokens and 37,206 committed decode tokens. The board ran at ~436 W of its 480 W
ceiling, 2,700 MHz, 60 °C.

| Reading | J/token | Per 1M tokens |
|---|---:|---:|
| Prefill | 0.176 | 49 Wh |
| Decode | 3.799 | 1.06 kWh |
| Served (all board energy) | 0.504 | 140 Wh |
| Active (idle baseline removed) | 0.493 | 137 Wh |

**Decode costs 22× more energy per token than prefill.** That ratio is the practical argument
against attributing energy in proportion to time: prefill is compute-bound and runs the board near
its ceiling while producing tokens in bulk, and memory-bound decode produces one token per lane per
round at a similar draw. A time-proportional split would have understated prefill efficiency by
more than an order of magnitude.

The engine measured its own idle baseline at **60.3 W** with the model resident, from intervals that
ran no execution unit and had nothing queued. Idle was only 4.8 kJ of the 207.7 kJ in this window
because the server was busy throughout; served and active differ by 2% here and would diverge
sharply on a mostly idle server.

Reconciliation against the board counter over the same window: **−2.16% aggregate**, with per-interval
residuals at p50 −1.41% and p10/p90 of −11.9%/+7.1%. The aggregate is what the dashboard reports and
what any claim should rest on; single intervals are noisier than the counter's own resolution
supports. Note that prefill energy divides by `computed_prefill_tokens`, so a continuation cache
being enabled does not flatter the figure — cache-served tokens are excluded from the denominator as
well as from the work.

### Reading energy figures

Two denominators are published because they answer different questions. Joules per token is the
engineering unit: it composes additively across phases, so prefill and decode figures can be combined
against their own token counts. Energy per million tokens is the same measurement rescaled by
`1e6/3600`, in the denominator inference is priced in; multiplying it by a local electricity rate
gives a figure directly comparable to a published $/1M-token price. It carries no information the
per-token figure does not, and it is not used as the primary unit because a rescaled rate does not
compose across phases the way energy does.

Three caveats bound any figure above. Board energy excludes the CPU, the rest of the platform, and
power-supply losses, so it is not comparable to a wall-socket measurement. Energy per token is a
function of the board's power ceiling rather than a fixed property of the engine, so a single number
is not a result on its own: `scripts/prefill_probe.py --power-limit-sweep` produces the
throughput/energy curve across ceilings. It randomizes point order across repeats and cools the board
to a fixed temperature before each point, because a monotonic sweep lets rising temperature correlate
with the power axis and be read as an efficiency trend that is really a thermal one. Setting the
ceiling requires `nvidia-smi -pl` privileges; the sweep aborts rather than silently reporting every
point at the same limit. This board accepts 150–530 W against a 480 W default.

Per-request energy is well defined only at concurrency one, which is why the CLI reports it and the
server does not: board energy is a property of the device, and with lanes sharing every decode round
it cannot be attributed to a single request. The server reports energy per interval instead.
