# Qwen3.8-27B artifact contract

This document defines the registered `qwen3.8-27b/groupwise-int` artifact: its identity,
persistent inventory, conversion entry point, and Engine binding. Model mathematics, dimensions,
frontend semantics, and state behavior are defined by
[`qwen3.8-27b-model.md`](qwen3.8-27b-model.md).

## 1. Identity

```text
filename   = qwen3_8_27b.ninfer
model_id   = qwen3.8-27b
weights_id = groupwise-int
target_key = qwen3_8_27b
recipe_id  = qwen3_8_27b-v2
```

The artifact contains Text, the optimized MTP draft head, MTP, Vision, and six frontend
resources. The identity is read from the version-2 artifact directory; filenames and object counts
do not select the target or weight profile.

## 2. Persistent inventory

The artifact contains 1118 tensors and six resources, for 1124 objects in total. Tensor format
counts are:

| Format | Tensors |
|---|---:|
| `BF16` | 582 |
| `FP32` | 96 |
| `I32` | 1 |
| `Q4G64_F16S` | 183 |
| `Q5G64_F16S` | 246 |
| `Q6G64_F16S` | 1 |
| `W8G32_F16S` | 9 |

The two vocabulary matrices use `W8G32_F16S` with `row-split-k128-v1`:

| Object | Logical shape |
|---|---|
| `text/token_embedding` | `[248320,5120]` |
| `text/output_head` | `[248320,5120]` |

Text layers use the Q4/Q5/Q6 groupwise assignment, `text/draft_head` uses Q4, the Vision patch
projection uses Q6, and the registered MTP and Vision-merger matrices use W8. Direct tensors use
`contiguous-le-v1`; all quantized tensors use `row-split-k128-v1`. The complete ordered inventory,
logical row views, and aliases are defined by
`tools/convert/qwen3_8_27b/inventory.py`.

## 3. Conversion

Conversion has two steps. Calibration runs the BF16 model over a calibration corpus and records the
mean squared input activation of every column of every quantized text matrix. The converter then
quantizes each matrix with the importance-weighted scale search of
[`tensor-formats.md`](tensor-formats.md) Section 7:

```bash
python3 -m tools.convert.qwen3_8_27b.calibrate \
  --model /path/to/Qwen3.8-27B \
  --corpus /path/to/calibration-corpus.jsonl \
  --out out/importance.safetensors

python3 -m tools.convert.qwen3_8_27b.convert \
  --model /path/to/Qwen3.8-27B \
  --importance out/importance.safetensors \
  --out out/qwen3_8_27b.ninfer \
  --device cuda
```

The corpus is JSONL, one conversation per line: `{"domain": ..., "messages": [...]}` in the
official message schema, with `reasoning_content` on assistant turns. Calibration renders each
conversation with the checkpoint's own chat template and packs it into 2048-token windows. Within
each domain, conversations in hash order first fill 12 held-out windows. The following
conversations supply up to 64 calibration windows. Calibration never reads the held-out windows.
`tools.parity.qwen3_8_27b.weight_kld` scores them to compare artifacts with the BF16 checkpoint.
The current corpus has reasoning, code, chat and agentic tool-use domains, about 0.5M calibration
tokens. The model generated its own thinking-mode replies to the reasoning, code and chat prompts.

The importance file holds one `float32[K]` vector per weighted object, keyed by object name, and the
corpus manifest in its metadata:

| Objects | Importance source |
|---|---|
| attention `query_key`, `gate_value` | normalized layer input |
| attention `output` | gated attention output |
| GDN `query_key`, `value_z` | normalized layer input |
| GDN `output` | gated, normalized GDN output |
| MLP `gate_up` | normalized post-attention input |
| MLP `down` | SiLU-gated activation |
| `text/output_head`, `text/draft_head` | final-norm hidden state |

`text/token_embedding` is a lookup table. MTP and Vision matrices see no text calibration
activations. Those tensors use equal column weights.

Before opening the output, the converter validates the following:
- the checkpoint configuration, source tensor shapes and dtypes;
- the frontend resources;
- the conversion recipes and complete object plan;
- the importance file's object set and vector shapes.

It writes the conversion report to `out/qwen3_8_27b.ninfer.conversion.json`. The report records
the calibration file's SHA-256 and corpus manifest. The verifier needs the same importance file:

```bash
python3 -m tools.convert.qwen3_8_27b.verify out/qwen3_8_27b.ninfer \
  --model /path/to/Qwen3.8-27B --importance out/importance.safetensors
```

The converter owns and pins the official Qwen3.8 six-resource frontend profile. Relative to the
Qwen3.8-27B profile, `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja` have
Qwen3.8-specific bytes; `generation_config.json`, `preprocessor_config.json`, and
`video_preprocessor_config.json` are byte-identical.

## 4. Engine binding

The registered mapping is:

```text
ArtifactIdentity(qwen3.8-27b, groupwise-int)
    -> WeightsProfile::GroupwiseIntW8Endpoints
    -> target qwen3_8_27b
```

The profile binds the embedding and output head as W8 and the Text body through the groupwise
binding. Workspace selection follows the groupwise execution routes. The registry constructs the
27B `LoadedModel`, `SequencePlan`, and `Program`, and reports
`qwen3_8_27b/qwen3.8-27b/groupwise-int` in the load summary.
