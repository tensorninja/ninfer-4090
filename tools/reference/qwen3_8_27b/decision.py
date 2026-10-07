"""Row-form System One decision reference for the Qwen3.8-27B target.

    python3 -m tools.reference.qwen3_8_27b.decision \
        --weights models/qwen3_8_27b.ninfer --lora <peft_dir> \
        --decision-head <peft_dir>/decision_head.safetensors \
        --prepared prepared.jsonl --out probabilities.jsonl

A prepared decision (`ninfer-decision-prepared` v1) is the engine's token layout of one request:
the state row `tokens[0:state_tokens]`, opening with `<|fim_prefix|>`, followed by every question's
branch `[<|fim_middle|>, ..., (<|box_start|>, ..., <|box_end|>)..., <|fim_suffix|>]` in request
order. Each branch sees exactly the state and itself: the state is prefilled once and its KV and
GDN state are snapshotted, then every branch is prefilled from a restore of that snapshot at
positions `state_tokens ...`, text-only with plain positions. The readouts are the final-norm
hidden states (BF16, the values the output head would read; the head itself never runs) at each
option's `<|box_end|>` and at the branch's closing `<|fim_suffix|>`. They feed kev's PointerHead:

    q = Wq h_decide + bq,  k_i = Wk h_option_i + bk,  p = softmax_i((k_i . q) * logit_scale / T)

The head is used as the engine holds it: the hand-off tensors are rounded to BF16 once and the head
is then evaluated in FP64; `T` is the FP32 value of the head's temperature or of `--temperature`.

Independence: the base weights come from the `.ninfer` artifact through this reference, the LoRA
from the PEFT directory through `lora.py`, and the head from the trainer's
`decision_head.safetensors`, so neither the engine nor the adapter converter is shared with the
path this checks. The token layout is taken from the engine (it is held to kev's golden layout
separately); its structure is checked here so that misplaced readout offsets cannot be shared.

Alternatively --openai-requests accepts native /v1/decisions bodies (JSON or JSONL), with images
embedded as base64 data URLs. This route builds tokens, transformed pixels, grids and MRoPE
independently from the artifact's library tokenizer/processor, not an Engine prepared dump.
An image state runs Vision then prefill only; branches continue its MRoPE delta. No generation
or draft head runs. Use BF16 KV for the first qualification. Numerical agreement is not a claim
that a text-trained decision adapter is accurate on image tasks.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open

from tools.reference.qwen3_8.common.frontend import Frontend, prepare_decision_image
from tools.reference.qwen3_8.common.multimodal import MultimodalBatch, build_mrope_positions

from .bindings import ArtifactBinding
from .cli import parse_bytes
from .config import CFG
from .model import COMPILED_CODEC_MIN_TOKENS, RefModel

PREPARED_FORMAT = "ninfer-decision-prepared"
RESULT_FORMAT = "ninfer-decision-probabilities"
FORMAT_VERSION = 1

HEAD_FORMAT = "systemone-pointer-head"
HEAD_FORMAT_VERSION = "1"
POINTER_DIM = 256
LOGIT_SCALE = 0.0625
READOUT = "final_norm"
_HEAD_SHAPES = {
    "query.weight": (POINTER_DIM, CFG.hidden),
    "query.bias": (POINTER_DIM,),
    "key.weight": (POINTER_DIM, CFG.hidden),
    "key.bias": (POINTER_DIM,),
}

# Delimiter ids in the registered base vocabulary.
STATE = 248060  # <|fim_prefix|>
QUESTION = 248061  # <|fim_middle|>
OPTION_OPEN = 248049
OPTION_CLOSE = 248050  # <|box_end|>
DECIDE = 248062  # <|fim_suffix|>
VISION_START = 248053
VISION_END = 248054
IMAGE_PAD = 248056


def fp32(value: float) -> float:
    """The nearest FP32 value, as the Python float that represents it exactly."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True, slots=True)
class Branch:
    begin: int
    length: int
    # Branch-relative offsets of each option's closing <|box_end|>; the decide readout is the
    # branch's last token.
    option_readouts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PreparedDecision:
    tokens: list[int]
    state_tokens: int
    branches: tuple[Branch, ...]
    state_batch: MultimodalBatch | None = None

    @property
    def capacity(self) -> int:
        """Context the row form occupies: the state and its longest branch."""
        return self.state_tokens + max(branch.length for branch in self.branches)

    def branch_tokens(self, branch: Branch) -> list[int]:
        return self.tokens[branch.begin : branch.begin + branch.length]


def parse_prepared(value: dict[str, Any]) -> PreparedDecision:
    """Validate one `ninfer-decision-prepared` object against the layout it describes."""
    if (
        not isinstance(value, dict)
        or value.get("format") != PREPARED_FORMAT
        or value.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError(f"not a {PREPARED_FORMAT} version {FORMAT_VERSION} object")
    tokens = [int(token) for token in value["tokens"]]
    state_tokens = int(value["state_tokens"])
    if not 0 < state_tokens <= len(tokens) or tokens[0] != STATE:
        raise ValueError("the state row must be a nonempty prefix opening with <|fim_prefix|>")
    branches: list[Branch] = []
    cursor = state_tokens
    for index, raw in enumerate(value["branches"]):
        branch = Branch(
            int(raw["begin"]),
            int(raw["length"]),
            tuple(int(offset) for offset in raw["option_readouts"]),
        )
        span = tokens[branch.begin : branch.begin + branch.length]
        closes = tuple(offset for offset, token in enumerate(span) if token == OPTION_CLOSE)
        if branch.begin != cursor or len(span) != branch.length or branch.length < 2:
            raise ValueError(f"branch {index} does not continue the layout at token {cursor}")
        if span[0] != QUESTION or span[-1] != DECIDE:
            raise ValueError(f"branch {index} is not delimited by <|fim_middle|>...<|fim_suffix|>")
        if not branch.option_readouts or branch.option_readouts != closes:
            raise ValueError(
                f"branch {index} option readouts {list(branch.option_readouts)} are not its "
                f"<|box_end|> offsets {list(closes)}"
            )
        branches.append(branch)
        cursor += branch.length
    if not branches or cursor != len(tokens):
        raise ValueError("the branches must tile every token after the state row")
    return PreparedDecision(tokens, state_tokens, tuple(branches))


def load_prepared(path: str | Path) -> list[PreparedDecision]:
    """One prepared object from a JSON file, or one per nonblank line of a `.jsonl` file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = [json.loads(text)]
    if not values:
        raise ValueError(f"{path} holds no prepared decision")
    return [parse_prepared(value) for value in values]


def openai_state_parts(value: Any) -> list[str | dict[str, Any]]:
    """Flatten user messages without a chat template, coalescing adjacent text fragments."""
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("OpenAI input must be a string or a user-message array")
    parts: list[str | dict[str, Any]] = []

    def append_text(text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("decision text must be a string")
        if parts and isinstance(parts[-1], str):
            parts[-1] += text
        else:
            parts.append(text)

    for index, message in enumerate(value):
        if message.get("role") != "user" or message.get("type", "message") != "message":
            raise ValueError("only user messages are supported")
        if index:
            append_text("\n\n")
        content = message["content"]
        if isinstance(content, str):
            append_text(content)
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be text or an array")
        for part in content:
            if part["type"] == "input_text":
                append_text(part["text"])
            elif part["type"] == "input_image":
                parts.append(part)
            else:
                raise ValueError("decision state supports only input_text and input_image")
    return parts


def openai_options(question: dict[str, Any]) -> list[str]:
    """Render ordered typed option labels independently of the serving implementation."""
    kind = question["type"]
    if kind == "predicate":
        return ["no", "yes"]
    if kind == "choice":
        entries = question["choices"]
        if not 2 <= len(entries) <= 255:
            raise ValueError("a choice needs 2 to 255 options")
        values = [entry["value"] for entry in entries]
        if any(not isinstance(value, (str, bool)) for value in values):
            raise ValueError("choice values must be strings or booleans")
        labels = ([json.dumps(value, ensure_ascii=False) for value in values]
                  if any(isinstance(value, bool) for value in values) else values)
    elif kind == "score":
        entries = question["levels"]
        if not 2 <= len(entries) <= 10:
            raise ValueError("a score needs 2 to 10 levels")
        labels = [entry["label"] for entry in entries]
    else:
        raise ValueError("questions must be predicate, choice or score")
    rendered = []
    for label, entry in zip(labels, entries, strict=True):
        description = entry.get("description", "")
        if not isinstance(label, str) or not isinstance(description, str):
            raise ValueError("option labels and descriptions must be text")
        rendered.append(label + (": " + description if description else ""))
    return rendered


def prepare_openai(value: dict[str, Any], frontend: Frontend) -> PreparedDecision:
    """Prepare a native request from raw image bytes, with no production-derived metadata."""
    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("decision text must be a string")
        escaped = re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)
        return frontend.tokenizer.encode(escaped, add_special_tokens=False)

    tokens = [STATE]
    types = [0]
    pixels = []
    grids = []
    for part in openai_state_parts(value["input"]):
        if isinstance(part, str):
            ids = encode(part)
            tokens.extend(ids)
            types.extend([0] * len(ids))
            continue
        url = part["image_url"]
        if not isinstance(url, str) or not url.startswith("data:image/"):
            raise ValueError("reference images must be embedded base64 image data URLs")
        header, separator, payload = url.partition(",")
        if not separator or not header.endswith(";base64"):
            raise ValueError("reference images need base64 data URLs")
        image = prepare_decision_image(
            frontend.processor.image_processor, base64.b64decode(payload, validate=True),
            part.get("detail"),
        )
        grid = image["image_grid_thw"].cpu()
        if grid.shape != (1, 3) or int(grid[0, 0]) != 1:
            raise ValueError("one input_image must produce one still-image grid")
        count = int(grid.prod().item()) // 4
        tokens.extend([VISION_START, *([IMAGE_PAD] * count), VISION_END])
        types.extend([0, *([1] * count), 0])
        pixels.append(image["pixel_values"].cpu())
        grids.append(grid)
    state_tokens = len(tokens)
    state_batch = None
    if grids:
        image_grid = torch.cat(grids)
        mm_types = torch.tensor(types, dtype=torch.long)
        positions, delta = build_mrope_positions(mm_types, image_grid, None)
        state_batch = MultimodalBatch(
            torch.tensor(tokens, dtype=torch.long), mm_types, positions, delta,
            torch.cat(pixels), image_grid, None, None,
        )
    questions = value["questions"]
    if not isinstance(questions, list) or not 1 <= len(questions) <= 200:
        raise ValueError("a decision needs 1 to 200 questions")
    branches = []
    for question in questions:
        begin = len(tokens)
        tokens.extend([QUESTION, *encode(question["instructions"])])
        readout_offsets = []
        for option in openai_options(question):
            tokens.extend([OPTION_OPEN, *encode(option), OPTION_CLOSE])
            readout_offsets.append(len(tokens) - 1 - begin)
        tokens.append(DECIDE)
        branches.append(Branch(begin, len(tokens) - begin, tuple(readout_offsets)))
    return PreparedDecision(tokens, state_tokens, tuple(branches), state_batch)


def load_openai(path: str | Path, weights: str | Path) -> list[PreparedDecision]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    values = ([json.loads(line) for line in text.splitlines() if line.strip()]
              if path.suffix == ".jsonl" else [json.loads(text)])
    if not values:
        raise ValueError(f"{path} holds no decision requests")
    with ArtifactBinding.open(weights) as binding:
        frontend = Frontend(binding)
        return [prepare_openai(value, frontend) for value in values]


@dataclass(frozen=True, slots=True)
class PointerHead:
    """kev's PointerHead as the engine holds it: BF16 parameters, evaluated in FP64 on the host."""

    query_weight: torch.Tensor
    query_bias: torch.Tensor
    key_weight: torch.Tensor
    key_bias: torch.Tensor
    temperature: float

    def probabilities(
        self, options: torch.Tensor, decide: torch.Tensor, temperature: float | None = None
    ) -> list[float]:
        """Softmax over one question's options from its BF16 readouts.

        `options` is `[options, hidden]` and `decide` is `[hidden]`; `temperature` overrides the
        head's own and is used at FP32 precision.
        """
        t = self.temperature if temperature is None else fp32(temperature)
        q = self.query_weight @ decide.to("cpu", torch.float64) + self.query_bias
        k = options.to("cpu", torch.float64) @ self.key_weight.t() + self.key_bias
        return torch.softmax((k @ q) * LOGIT_SCALE / t, dim=0).tolist()


def load_head(path: str | Path) -> PointerHead:
    """Read and validate the trainer's `decision_head.safetensors` hand-off."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as source:
        header = dict(source.metadata() or {})
        tensors = {key: source.get_tensor(key) for key in source.keys()}

    def field(name: str) -> str:
        if name not in header:
            raise ValueError(f"{path} metadata lacks {name!r}")
        return header[name]

    if field("format") != HEAD_FORMAT or field("format_version") != HEAD_FORMAT_VERSION:
        raise ValueError(f"{path} is not a {HEAD_FORMAT} version {HEAD_FORMAT_VERSION} file")
    for name, actual, expected in (
        ("pointer_dim", int(field("pointer_dim")), POINTER_DIM),
        ("hidden_size", int(field("hidden_size")), CFG.hidden),
        ("logit_scale", float(field("logit_scale")), LOGIT_SCALE),
        ("readout", field("readout"), READOUT),
    ):
        if actual != expected:
            raise ValueError(f"{path} {name} is {actual!r}, expected {expected!r}")
    temperature = fp32(float(field("temperature")))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{path} temperature {field('temperature')} is not positive and finite")
    if set(tensors) != set(_HEAD_SHAPES):
        raise ValueError(f"{path} holds {sorted(tensors)}, expected {sorted(_HEAD_SHAPES)}")
    for name, shape in _HEAD_SHAPES.items():
        tensor = tensors[name]
        if tensor.dtype not in (torch.float32, torch.bfloat16) or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{path} {name} is {tensor.dtype} {tuple(tensor.shape)}, expected FP32 {shape}"
            )
    exact = {name: tensor.to(torch.bfloat16).to(torch.float64) for name, tensor in tensors.items()}
    return PointerHead(
        exact["query.weight"], exact["query.bias"], exact["key.weight"], exact["key.bias"],
        temperature,
    )


def readouts(model: RefModel, prepared: PreparedDecision) -> list[torch.Tensor]:
    """Row-form final-norm readouts per branch: BF16 `[options + 1, hidden]`, decide last.

    The model starts from an empty sequence. Text-only decisions keep prepared weight residency;
    image decisions release it before Vision and re-plan Text after the tower is released.
    """
    with torch.inference_mode():
        vision = None
        if prepared.state_batch is not None:
            model.weights = None
            model.state = None
            model.last_hidden = model.last_mtp_hidden = model.last_draft = None
            if model.device.type == "cuda":
                torch.cuda.empty_cache()
            vision = model.encode_vision(prepared.state_batch)
        if model.state is None or model.state.capacity < prepared.capacity:
            model.prepare(prepared.capacity)
        else:
            model.clear()
        if prepared.state_batch is None:
            model.prefill_hidden(prepared.tokens[: prepared.state_tokens], ())
        else:
            model.prefill_multimodal_hidden(prepared.state_batch, vision)
        after_state = model.snapshot()
        rows: list[torch.Tensor] = []
        for branch in prepared.branches:
            model.restore(after_state)
            rows.append(
                model.prefill_hidden(
                    prepared.branch_tokens(branch),
                    (*branch.option_readouts, branch.length - 1),
                )
            )
        return rows


def decide(
    model: RefModel,
    head: PointerHead,
    prepared: PreparedDecision,
    *,
    temperature: float | None = None,
) -> list[list[float]]:
    """Pointer probabilities of every question, in branch order."""
    return [
        head.probabilities(rows[:-1], rows[-1], temperature)
        for rows in readouts(model, prepared)
    ]


def result(temperature: float, questions: list[list[float]]) -> dict[str, Any]:
    return {
        "format": RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "temperature": temperature,
        "questions": questions,
    }


def _prefill_calls(prepared: PreparedDecision, chunk: int) -> int:
    spans = (prepared.state_tokens, *(branch.length for branch in prepared.branches))
    return sum(-(-span // chunk) for span in spans)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", required=True, help="Qwen3.8-27B .ninfer artifact")
    parser.add_argument("--lora", required=True, help="PEFT directory of the decision adapter")
    parser.add_argument(
        "--decision-head", required=True, help="the trainer's decision_head.safetensors"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--prepared",
        help=f"{PREPARED_FORMAT} JSON object, or a .jsonl file with one object per line",
    )
    inputs.add_argument("--openai-requests", help="native OpenAI requests JSON/JSONL; images must "
                        "be base64 data URLs, independently transformed with embedded resources")
    parser.add_argument(
        "--out",
        required=True,
        help=f"{RESULT_FORMAT} JSON object, or a .jsonl file with one line per prepared object",
    )
    parser.add_argument("--temperature", type=float, help="override the head's temperature")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", default="auto")
    parser.add_argument("--headroom", default="2GiB")
    parser.add_argument("--prefill-chunk", type=int, default=CFG.prefill_chunk)
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="bf16")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.weights).suffix != ".ninfer":
        parser.error("--weights must name a .ninfer artifact")
    if args.prefill_chunk <= 0:
        parser.error("--prefill-chunk must be positive")
    if args.temperature is not None and not (
        math.isfinite(args.temperature) and args.temperature > 0.0
    ):
        parser.error("--temperature must be positive and finite")
    try:
        memory_bytes = parse_bytes(args.gpu_memory)
        headroom_bytes = parse_bytes(args.headroom)
    except ValueError as exc:
        parser.error(f"invalid memory size: {exc}")
    if memory_bytes is not None and memory_bytes <= 0:
        parser.error("--gpu-memory must be positive or auto")
    if headroom_bytes is None or headroom_bytes < 0:
        parser.error("--headroom must be nonnegative")
    try:
        head = load_head(args.decision_head)
        decisions = (load_prepared(args.prepared) if args.prepared
                     else load_openai(args.openai_requests, args.weights))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    if out.suffix != ".jsonl" and len(decisions) != 1:
        parser.error(f"{len(decisions)} prepared decisions need a .jsonl --out")
    temperature = head.temperature if args.temperature is None else fp32(args.temperature)

    calls = sum(_prefill_calls(decision, args.prefill_chunk) for decision in decisions)
    started = time.perf_counter()
    with RefModel(
        args.weights,
        device=args.device,
        memory_bytes=memory_bytes,
        headroom_bytes=headroom_bytes,
        kv_dtype=args.kv_dtype,
        prefill_chunk=args.prefill_chunk,
        compile_codec=calls >= COMPILED_CODEC_MIN_TOKENS,
        lora=args.lora,
    ) as model, torch.inference_mode():
        cuda = model.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(model.device)
        if not any(decision.state_batch is not None for decision in decisions):
            model.prepare(max(decision.capacity for decision in decisions))
        print(f"LOAD: {time.perf_counter() - started:.3f}s")
        if model.weights is not None:
            print("MEMORY_PLAN:", model.weights.plan.summary())
        print("CODEC:", "compiled" if model.compile_codec else "eager")
        lines = []
        total_seconds = 0.0
        total_questions = 0
        for index, decision in enumerate(decisions):
            started = time.perf_counter()
            questions = decide(model, head, decision, temperature=temperature)
            seconds = time.perf_counter() - started
            total_seconds += seconds
            total_questions += len(questions)
            print(
                f"DECISION {index}: state_tokens={decision.state_tokens} "
                f"questions={len(questions)} tokens={len(decision.tokens)} "
                f"seconds={seconds:.3f} per_question={seconds / len(questions):.3f}"
            )
            output = result(temperature, questions)
            if args.openai_requests:
                batch = decision.state_batch
                output.update(input_tokens=len(decision.tokens), state_tokens=decision.state_tokens,
                              image_grid_thw=[] if batch is None else batch.image_grid_thw.tolist(),
                              rope_delta=0 if batch is None else batch.rope_delta,
                              kv_dtype=args.kv_dtype, prefill_chunk=args.prefill_chunk)
            lines.append(json.dumps(output))
        print(
            f"TIMING: decisions={len(decisions)} questions={total_questions} "
            f"seconds={total_seconds:.3f} per_question={total_seconds / total_questions:.3f}"
        )
        if cuda:
            print(
                "CUDA_PEAK: "
                f"allocated={torch.cuda.max_memory_allocated(model.device) / (1 << 30):.2f}GiB "
                f"reserved={torch.cuda.max_memory_reserved(model.device) / (1 << 30):.2f}GiB"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
