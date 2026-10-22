---
library_name: ninfer
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
pipeline_tag: text-generation
inference: false
tags:
  - ninfer
  - lora
  - qwen3.8
  - style
  - rtx-4090
---

# caveman-1

`caveman-1` is a LoRA adapter for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) that makes
the model answer in terse caveman speech with a pirate accent. It is packaged as a `.ninfer`
adapter for [NInfer-4090](https://github.com/tensorninja/ninfer-4090) and loads per request
beside the base weights. It is not a PEFT, Transformers, or GGUF file.

```text
Hrrm. Register renaming give each write its own private register, so later read grab right value
and no stall wait on old one. It clear most false hazards, matey, and let out-of-order engine run
ahead of program order.
```

## File

| Field | Value |
|---|---|
| Filename | `caveman-1.lora.ninfer` |
| Size | 84,471,808 bytes |
| SHA-256 | `1c10e3d042f7a37e32f4867d2e76e8baea6898c4452e0f3e32f05396136971c9` |
| Kind | generative adapter |
| Rank | 16, `alpha` 32, folded into `B` |
| Sites | query, output gate, key, value, and attention output on the 16 attention layers, Gated DeltaNet output on the other 48, MLP down projection on all 64 |
| Parameters | 42,205,184 |

## Use it

The adapter needs NInfer-4090 and the
[Qwen3.8-27B NInfer artifact](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer). Download the
adapter into the directory that the server reads with `--lora-dir`:

```bash
hf download tensorninja2/caveman-1 caveman-1.lora.ninfer --local-dir lora
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --lora-dir lora --lora-slots 2
```

The server exposes the adapter as the model `qwen3.8-27b-caveman-1` on OpenAI Chat Completions,
OpenAI Responses, and Anthropic Messages:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "qwen3.8-27b-caveman-1", "enable_thinking": false, "temperature": 0,
  "messages": [{"role": "user", "content": "Explain register renaming in two sentences."}]}'
```

Requests for the base model and for other adapters can run in the same decode batch. See
[Load LoRA adapters per request](https://github.com/tensorninja/ninfer-4090#load-lora-adapters-per-request).

## Best with Caudra

Any agent that speaks the OpenAI or Anthropic APIs can use the adapter by its model name. For the
most native experience, use [Caudra](https://caudra.ai), the coding agent from the author of
NInfer-4090. With the `ninfer-4090` provider from
[Use it from Caudra](https://github.com/tensorninja/ninfer-4090#use-it-from-caudra), Caudra
discovers the adapter, and you select it as `ninfer-4090/qwen3.8-27b-caveman-1` in `/model`.

## Training

QLoRA on an NF4-quantized Qwen3.8-27B base: PEFT modules `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`down_proj`, and `out_proj`, 120 steps, maximum sequence length 1,024, completion-only loss, and
thinking off. The corpus is the `caveman_pirate` SFT split of the `llm-datasets` repository.
`tools/convert/qwen3_8_27b/convert_lora.py` converted the PEFT directory to `.ninfer`.

## Limits

- The adapter changes the register of the answer. It was not trained or evaluated for task
  accuracy.
- NInfer's MTP draft head is not adapted, so speculative decoding accepts fewer draft tokens with
  this adapter than with the base weights.
- It was trained with thinking off.

## License

Apache-2.0, as is the Qwen3.8-27B base.
