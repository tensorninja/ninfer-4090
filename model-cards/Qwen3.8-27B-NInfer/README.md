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

The bytes are identical to the artifact that the upstream
[NInfer](https://github.com/Neroued/ninfer) project published as `neroued/Qwen3.8-27B-NInfer`. The
SHA-256 below is the same.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| Size | 18,210,531,328 bytes (16.96 GiB) |
| SHA-256 | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
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
  'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e' \
  'qwen3_8_27b.ninfer' | sha256sum --check
```

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
| Source repository | `Qwen/Qwen3.8-27B` |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Conversion recipe | `qwen3_8_27b-v1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `52320554b5e71a9da96bff809ddf67ac5773ed63` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The upstream converter checked the source configuration, tensor index, frontend resources, and
published CRC32 inventory against the Hugging Face source revision above. The artifact identity,
object inventory, and conversion provenance are in
[`artifact-manifest.json`](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer/blob/main/artifact-manifest.json).
The storage contract is in the
[Qwen3.8-27B artifact reference](https://github.com/tensorninja/ninfer-4090/blob/tensorninja/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also Apache-2.0. The
conversion is the work of the [NInfer](https://github.com/Neroued/ninfer) project.
