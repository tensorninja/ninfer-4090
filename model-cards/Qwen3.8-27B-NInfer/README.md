---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - conversational
  - cuda
  - rtx-4090
---

# Qwen3.8-27B for NInfer-4090

This model card is the version-controlled source for
[tensorninja2/Qwen3.8-27B-NInfer](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer).

The repository contains [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to the
native `.ninfer` artifact format of [NInfer-4090](https://github.com/tensorninja/ninfer-4090), a
C++/CUDA inference engine for this model on one RTX 4090. The file works only with NInfer. It is
not a Transformers checkpoint, Safetensors distribution, or GGUF file.

NInfer-4090 converted this file from the BF16 checkpoint with importance-weighted quantization
(recipe `qwen3_8_27b-v2`). It has the same formats, size and kernels as the `qwen3_8_27b-v1`
artifact that the upstream [NInfer](https://github.com/Neroued/ninfer) project published, so it runs
at the same speed, but its outputs are closer to the BF16 model. The `v1` tag of this repository
keeps the previous file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| Size | 18,210,531,328 bytes (16.96 GiB) |
| SHA-256 | `b450e1bdea29f8ce5f82fd9995e716d0cbb620dc9ba31ed9fcabca4b7999a75d` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) |

The Text body uses the registered Q4/Q5/Q6 groupwise allocation, and the token embedding and full
output head use `W8G32_F16S`. The file also contains the Vision, MTP, proposal-head, tokenizer,
chat-template, generation, and media-processor objects that NInfer requires.

To check a downloaded file:

```bash
printf '%s  %s\n' \
  'b450e1bdea29f8ce5f82fd9995e716d0cbb620dc9ba31ed9fcabca4b7999a75d' \
  'qwen3_8_27b.ninfer' | sha256sum --check
```

## Quantization

Each group of 32 or 64 weights stores one FP16 scale and small integer codes. The `v1` recipe set
each scale from the group's largest absolute weight. This artifact instead tries 43 candidate scales
per group and keeps the one with the smallest reconstruction error, weighted by how strongly each
input column is activated. The activation statistics come from about 524,000 tokens of reasoning,
code, chat and agentic tool-use text run through the BF16 model.

Measured on an RTX 4090 against the BF16 checkpoint, on 98,304 held-out tokens that calibration
never saw:

| Domain | `v1` mean KLD | This file | `v1` top-1 | This file |
|---|---:|---:|---:|---:|
| Agentic tool use | 0.710 | 0.531 | 87.87% | 90.05% |
| Chat | 0.071 | 0.046 | 93.97% | 95.24% |
| Code | 0.032 | 0.022 | 94.51% | 95.63% |
| Reasoning | 0.033 | 0.022 | 95.82% | 96.62% |
| All | 0.212 | 0.155 | 93.04% | 94.38% |

KLD is the next-token KL divergence from the BF16 model, and top-1 is how often both pick the same
next token. Served with MTP3, draft acceptance rises from 56.2% to 57.6%.
[decision-1](https://huggingface.co/tensorninja2/decision-1) development accuracy is 87.18% (87.10%
on `v1`). The agentic row has the widest uncertainty, because its held-out text comes from one
trajectory.

## Requirements

- [NInfer-4090](https://github.com/tensorninja/ninfer-4090), built from source or as its Docker
  image;
- 64-bit Linux;
- NVIDIA GeForce RTX 4090 (`sm_89`);
- CUDA Toolkit 12.8 or newer.

## Download and run

```bash
hf download tensorninja2/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models

./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The Docker image of NInfer-4090 downloads this file during its build and checks its SHA-256. See the
[NInfer-4090 README](https://github.com/tensorninja/ninfer-4090#run-it) for every profile.

## Best with Caudra

NInfer-4090 serves any agent that speaks OpenAI Chat Completions, OpenAI Responses, or Anthropic
Messages. For the most native experience, use [Caudra](https://caudra.ai), the coding agent from
the same author. Caudra shows NInfer's live prefill rate while it reads the prompt, marks cache
breakpoints and session keys that NInfer reuses across turns, and can use
[decision-1](https://huggingface.co/tensorninja2/decision-1) on the same server as its decision
engine. The setup is one `providers.toml` entry, described in
[Use it from Caudra](https://github.com/tensorninja/ninfer-4090#use-it-from-caudra).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, and Hadamard-rotated KV cache (`rk8v4`, `rk4v4`, and `rk2v4-e8`), with the full
  262,144-token context on `rk4v4`;
- LoRA adapters selected per request, such as
  [caveman-1](https://huggingface.co/tensorninja2/caveman-1), and decision adapters such as
  [decision-1](https://huggingface.co/tensorninja2/decision-1);
- OpenAI- and Anthropic-compatible serving, OpenAI Decisions, and TypeSafe System One.

## Limits

- NInfer-4090 runs on one RTX 4090 and one CUDA device, with a startup-fixed capacity of one to
  eight active requests.
- It does not provide large-scale or preemptive continuous batching, priority scheduling,
  multi-GPU execution, CPU or GPU offload, or distributed serving.
- Context size depends on GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B`, BF16 |
| Download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b-v2` |
| Converter repository | `https://github.com/tensorninja/ninfer-4090` |
| Converter revision | `ad26a68c` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |
| Importance file SHA-256 | `d67fef13aba1879c3acc576bc22c8a2ba39b82cfe419e17550b58f46834b2c3e` |
| Calibration corpus SHA-256 | `8d3eed573cd102c1a243644acc0253e12588bae1f792a052a5d9b2de60535b23` |

The converter checked the source configuration, tensor shapes and dtypes, and frontend resources.
The six embedded frontend resources (tokenizer, chat template, and generation and media-processor
configuration) are byte-identical to those of the `v1` artifact. The artifact identity, object
inventory, and conversion provenance are in
[`artifact-manifest.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json).
The storage contract is in the
[Qwen3.8-27B artifact reference](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also Apache-2.0. The
conversion format and the `v1` artifact are the work of the [NInfer](https://github.com/Neroued/ninfer)
project.
