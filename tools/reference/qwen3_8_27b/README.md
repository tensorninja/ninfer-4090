# Qwen3.8-27B Python reference

This is the complete target-private Text, Vision, MTP, sampling, state, and weight-residency
reference over a native `.ninfer` artifact. It uses typed artifact bindings and remains independent
from the C++ Engine implementation.

It does not need the original Hugging Face checkpoint at inference time. `Frontend` materializes the
tokenizer, chat template, generation defaults, and image/video processor resources embedded in the
artifact, then delegates those functions to Transformers.

## Run

Install the target dependencies from `requirements.txt`, then run:

```bash
python3 \
  -m tools.reference.qwen3_8_27b \
  --weights out/qwen3_8_27b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 512
```

The input is exactly one of `--prompt`, `--ids`, or `--messages`. Structured messages may contain
images and videos in the normal Transformers format. Thinking is enabled by default and can be
disabled with `--no-thinking`.

MTP is disabled by default. Enable one to five draft positions with
`--mtp-draft-tokens 1..5`; `--draft-head` selects the artifact's optimized proposal head. Target
verification always uses the full output head. The CLI reports round counts, per-position accepted
drafts, fallback steps, timing, memory planning, and peak CUDA allocation.

Important runtime controls include:

- `--gpu-memory auto|24GiB` and `--headroom 2GiB`;
- `--kv-dtype bf16|int8`;
- `--prefill-chunk N`;
- `--greedy` or sampling overrides for temperature, top-p, top-k, and penalties;
- `--vision-attention-limit N`;
- `--activation-dump DIR --dump-level layer|op`.

Quantized Text matrices retain the decoded/packed/streamed residency plan and compiled low-bit
codec. Vision decodes large matrices one at a time and releases its weight store before Text weight
preparation. Multimodal MTP uses the composed Vision embedding for the shifted input, including at
prefill chunk boundaries.

The target-specific source-BF16 Vision comparison lives in `tools/parity/qwen3_8_27b/`.

## Decisions

`decision.py` is the row-form System One oracle for engine decision probabilities:

```bash
python3 -m tools.reference.qwen3_8_27b.decision \
  --weights out/qwen3_8_27b.ninfer \
  --lora <peft_dir> --decision-head <peft_dir>/decision_head.safetensors \
  --prepared prepared.jsonl --out probabilities.jsonl
```

`--prepared` takes the engine's `ninfer-decision-prepared` token layout, one JSON object or a
`.jsonl` file of them, and checks that every readout offset lands on its delimiter. The state row is
prefilled once and snapshotted. Each branch is then prefilled from that snapshot at positions
starting at the state length, so it sees the state and itself only. kev's PointerHead reads the
final-norm hidden states at each option's `<|box_end|>` and at the branch's `<|fim_suffix|>`. The
output head never runs. The head comes from the trainer's `decision_head.safetensors`, rounded to
BF16 and evaluated in FP64. The LoRA comes from the PEFT directory, as with `--lora` for chat.

The output is one `ninfer-decision-probabilities` object per prepared object, with one probability
list per question in branch order. `--temperature` overrides the head's `T`, and `--kv-dtype`,
`--prefill-chunk`, `--gpu-memory`, and `--headroom` work as they do for chat. All inputs share one
model load. `decide(model, head, prepared)` exposes the same computation to parity drivers.
