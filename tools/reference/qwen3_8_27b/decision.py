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
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open

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
OPTION_CLOSE = 248050  # <|box_end|>
DECIDE = 248062  # <|fim_suffix|>


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

    The model starts from an empty sequence. One prepared for at least `prepared.capacity` tokens
    keeps its weight residency; prepare it once for the largest of many decisions.
    """
    with torch.inference_mode():
        if model.state is None or model.state.capacity < prepared.capacity:
            model.prepare(prepared.capacity)
        else:
            model.clear()
        model.prefill_hidden(prepared.tokens[: prepared.state_tokens], ())
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
    parser.add_argument(
        "--prepared",
        required=True,
        help=f"{PREPARED_FORMAT} JSON object, or a .jsonl file with one object per line",
    )
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
        decisions = load_prepared(args.prepared)
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
        model.prepare(max(decision.capacity for decision in decisions))
        assert model.weights is not None
        print(f"LOAD: {time.perf_counter() - started:.3f}s")
        print("MEMORY_PLAN:", model.weights.plan.summary())
        print("CODEC:", "compiled" if model.active_compile_codec else "eager")
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
            lines.append(json.dumps(result(temperature, questions)))
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
