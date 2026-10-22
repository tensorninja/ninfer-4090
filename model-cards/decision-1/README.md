---
library_name: ninfer
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
pipeline_tag: text-classification
inference: false
tags:
  - ninfer
  - lora
  - qwen3.8
  - decision-model
  - calibration
  - typesafe
  - rtx-4090
---

# decision-1

`decision-1` is a decision adapter for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): a
LoRA adapter plus a pointer head. It reads a state and typed questions and returns a probability
for every option, by prefill alone, without generating text. It is packaged as a `.ninfer` adapter
for [NInfer-4090](https://github.com/tensorninja/ninfer-4090), which serves it on two wire
protocols from the same process and weights as chat:

- OpenAI Decisions, `POST /v1/decisions`, for the OpenAI Python SDK 3.26.0 or newer.
- TypeSafe System One, `POST /typesafe/v1/systemone`, for the TypeSafe SDK 0.6 and 0.7.

It is not a PEFT, Transformers, or GGUF file.

## File

| Field | Value |
|---|---|
| Filename | `decision-1.lora.ninfer` |
| Size | 89,716,300 bytes |
| SHA-256 | `14015d187d244a37dcc1fc0593308a756b8be1c1b70914bf088188acfb341a39` |
| Kind | decision adapter |
| LoRA | rank 16, `alpha` 32, on query, output gate, key, value, attention output, Gated DeltaNet output, and MLP down projection |
| Pointer head | 256-dimensional query and key projections over the final-norm hidden state of width 5,120 |
| Temperature | 1.414 |

## Use it

The adapter needs NInfer-4090 and the
[Qwen3.8-27B NInfer artifact](https://huggingface.co/tensorninja2/Qwen3.8-27B-NInfer). Download it into
the directory that the server reads with `--lora-dir`. Decisions are qualified on BF16 KV:

```bash
hf download tensorninja2/decision-1 decision-1.lora.ninfer --local-dir lora
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --kv-dtype bf16 \
  --max-context 32768 --lora-dir lora --lora-slots 2
```

OpenAI Decisions:

```python
from openai import OpenAI

client = OpenAI(api_key="local", base_url="http://127.0.0.1:8080/v1")
result = client.decisions.create(
    model="decision-1",
    input="I was charged twice this month. Please refund one charge before Friday.",
    questions=[{
        "type": "predicate", "name": "billing",
        "instructions": "Is this message about billing?",
    }],
)
print(result.answers[0].probability)  # 0.939
```

TypeSafe System One, where `jev-latest` resolves to the server's only decision adapter:

```python
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8080/typesafe")
result = client.system_one(
    state="I was charged twice this month. Please refund one charge before Friday.",
    questions={"billing": Noul(instructions="Is this message about billing?")},
)
```

On an RTX 4090 with a cached state, one question takes about 25 ms and six questions about
100 ms. See [Decisions](https://github.com/tensorninja/ninfer-4090#decisions-openai-and-typesafe-system-one)
for both protocols, latency, and limits.

## Best with Caudra

[Caudra](https://caudra.ai), the coding agent from the author of NInfer-4090, can use
`decision-1` as its local decision engine, in the same process and weights as the chat model.
Caudra uses it for permission advice, Auto mode screening, shell effect predictions, content
screening of tool results, subagent routing, and reranking, and its `decisions.decide()` and
`decisions.rerank()` calls work inside `code` snippets. Add the endpoint to
`~/.config/caudra/providers.toml`:

```toml
[decisions.ninfer]
protocol = "openai"
base_url = "http://127.0.0.1:8080/v1"
model = "decision-1"
```

Then select it with `/decision-model` or `caudra decisions use ninfer`. The full setup, including
the chat provider, is in
[Use it from Caudra](https://github.com/tensorninja/ninfer-4090#use-it-from-caudra).

## Evaluation

Scored with BF16 KV on the clean knowable questions of the decision-v7 corpus from
[kev](https://github.com/jaredpalmer/kev). The locked test partition was scored once.

| Split | Questions | Accuracy | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|
| Development | 1,264 | 0.871 | 0.184 | 0.356 | 0.024 |
| Locked test | 1,200 | 0.856 | 0.190 | 0.355 | 0.020 |

Kev-27B, which adapts every linear layer and trained on a superset of this data, publishes 0.866
and 0.870 accuracy on the same splits. On the test split, `score` questions are the weakest type at
0.596 accuracy, against 0.913 for `choice` and 0.929 for `noul`.

## Training

QLoRA and the pointer head on an NF4-quantized Qwen3.8-27B base, one epoch on the decision-v7
training split, with temperature 1.414. Training ran in the `llm-datasets` repository.
`tools/convert/qwen3_8_27b/convert_lora.py --decision-head` converted the PEFT directory and head
to `.ninfer`.

## Limits

- Decisions meet kev's per-question tolerance on `bf16` and `int8` KV. The rotated KV modes,
  including the 262K-context `rk4v4`, miss the probability bar on a few questions.
- The adapter was trained on text. NInfer accepts images in OpenAI Decisions, but no image-task
  accuracy or calibration is established for this adapter.
- There is no refusal detection, and the adapter does not reproduce the outputs or calibration of
  OpenAI's or TypeSafe's hosted decision models.

## License

Apache-2.0 for the adapter and head. The Qwen3.8-27B base is Apache-2.0. The datasets behind the
decision-v7 corpus carry their own licenses; see kev's model cards.
