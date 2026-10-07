# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [RTX 3090 Linux build](rtx-3090-linux.md) | Docker and native Ubuntu builds for the `sm_86` applications |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions and text-only Decisions, TypeSafe System One, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Dashboard](dashboard.md) | optional web UI over `/telemetry` and `/events`: throughput, queueing, cache, GPU, VRAM, and JSONL replay |
| [Tiered continuation cache](continuation-cache.md) | implemented L1/L2/L3 semantics, OpenCode session routing, stable-prefix sharing, persistence, sizing, metrics, and current limitations |
| [Turn checkpoint ring](turn-checkpoint-ring.md) | `--turn-checkpoints` sizing, mid-history reuse, slot listing, snapshot persistence, and eviction |
| [Performance](performance.md) | RTX 5090 single-request and concurrent-decode results, MTP/DFlash measurements, and reproduction commands |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the source for command-line option spelling and defaults. The
[cache guide](continuation-cache.md) defines independent tier idle TTLs and persistence trigger
semantics, including zero values.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

Runtime and Op references:

- [Small-scale concurrent inference architecture](maintainer/concurrent-inference-architecture.md)
- [Paged KV context storage, ownership, and capacity model](maintainer/paged-kv-cache.md)
- [Op admission, contracts, ownership, qualification, and performance rules](maintainer/op-development.md)
- [ReplaySSM GDN technical reference](maintainer/replayssm-gdn.md)
- [Linear benchmark contract and registered suites](maintainer/linear-benchmark.md)
- [Runtime LoRA adapters (Qwen3.8-27B)](maintainer/qwen3.8-27b-lora-adapters.md) is the current
  authority for PEFT-to-`.ninfer` conversion, adapter artifacts, runtime banking and slot residency,
  state isolation, and model-name routing. It does not own training.
- Adapter training, target profiles, corpora and training reports are maintained in the separate
  `llm-datasets` repository. Its Qwen3.8-27B contract is `docs/targets/qwen3_8_27b.md`, and its
  unconditional-persona experiment is `docs/persona-adapter.md`.

Artifact and model references:

- [NInfer artifact container](maintainer/artifact-container.md)
- [Persistent tensor numeric formats](maintainer/tensor-formats.md)
- [Persistent storage layouts](maintainer/storage-layouts.md)
- [Qwen3.8-27B model semantics](maintainer/qwen3.8-27b-model.md)
- [Qwen3.8-27B artifact contracts, including NVFP4](maintainer/qwen3.8-27b-artifact.md)
- [Qwen3.6-35B-A3B model semantics](maintainer/qwen3.6-35b-a3b-model.md)
- [Qwen3.6-35B-A3B artifact contracts](maintainer/qwen3.6-35b-a3b-artifact.md)

Work record:

- [Cross-fork port ledger](maintainer/port-ledger.md) records which sibling-fork and upstream
  changes have been taken into this branch, and which were declined.
- [Prefill throughput plan (sm_89 / Qwen3.8-27B `groupwise-int`)](maintainer/prefill-throughput-plan.md)
  holds the roofline analysis, per-kernel budgets, INT8 route design, tuning sweeps, and the
  measured reasons the GQA head-packing and SwiGLU-fusion levers were closed. Its stable
  conclusions are mirrored into `performance.md`, `maintainer/op-development.md`,
  `maintainer/linear-benchmark.md`, and `maintainer/softmax-attention.md`, which are the
  authorities; this file is the supporting detail.

Pending migration plan:

- [Softmax Attention organization and migration](maintainer/softmax-attention.md) describes the
  single target state for an unfinished source and public-contract cutover; it is not the current
  implementation map.
