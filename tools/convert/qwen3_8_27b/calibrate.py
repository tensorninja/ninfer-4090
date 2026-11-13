"""Collect per-column activation importance for the Qwen3.8-27B converter.

Canonical invocation::

    python -m tools.convert.qwen3_8_27b.calibrate \
      --model /path/to/Qwen3.8-27B \
      --corpus /path/to/calibration-corpus.jsonl \
      --out /path/to/importance.safetensors

The corpus is JSONL with one conversation per line, ``{"domain": str,
"messages": [...]}``, in the official chat-message schema (assistant turns may
carry ``reasoning_content``).  Conversations are rendered with the checkpoint's
own chat template and packed into fixed windows.  Per domain, whole
conversations in hash order fill ``EVAL_WINDOWS`` held-out windows first; the
following conversations supply up to ``CALIBRATION_WINDOWS`` calibration
windows.  The held-out windows are what
``tools/parity/qwen3_8_27b/weight_kld.py`` scores, so calibration never sees them.

For every quantized text matrix the output stores the BF16 model's mean squared
input activation per input column, ``float32[K]``, keyed by the artifact object
name.  The file metadata carries the corpus manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.reference.qwen3_8_27b import layer_major
from tools.reference.qwen3_8_27b.config import CFG

from . import inventory, recipe


FORMAT = "ninfer-importance-v1"
WINDOW = 2048
EVAL_WINDOWS = 12
CALIBRATION_WINDOWS = 64
_QUANTIZED = (inventory.Q4, inventory.Q5, inventory.Q6, inventory.W8)
_HEAD_OBJECTS = ("text/output_head", "text/draft_head")


def _importance_site(name: str) -> str:
    if name in _HEAD_OBJECTS:
        return "final"
    prefix, _, matrix = name.partition("/layers/")[2].partition("/")
    return f"{prefix}/{layer_major.SITE_OF_MATRIX[matrix]}"


def importance_shapes() -> dict[str, tuple[int]]:
    """Every object the converter weights by calibration, with its ``[K]`` shape.

    The token embedding is a lookup table and MTP/Vision tensors see no text
    calibration activations, so they are encoded with equal column weights.
    """

    specs = inventory.TEXT_CORE_TENSOR_SPECS + inventory.DRAFT_HEAD_TENSOR_SPECS
    return {
        spec.name: (spec.shape[1],)
        for spec in specs
        if spec.format in _QUANTIZED and spec.name != "text/token_embedding"
    }


@dataclass(frozen=True, slots=True)
class Corpus:
    eval: dict[str, torch.Tensor]
    calibration: dict[str, torch.Tensor]
    manifest: dict[str, object]


def load_corpus(
    corpus_path: str | Path,
    tokenizer_dir: str | Path,
    *,
    eval_windows: int = EVAL_WINDOWS,
    calibration_windows: int = CALIBRATION_WINDOWS,
) -> Corpus:
    from transformers import AutoTokenizer

    path = Path(corpus_path)
    data = path.read_bytes()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    documents: dict[str, list[tuple[str, list[int]]]] = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        documents.setdefault(row["domain"], []).append((key, ids))

    def pack(ids: list[int], count: int) -> torch.Tensor:
        n = min(count, len(ids) // WINDOW)
        return torch.tensor(ids[: n * WINDOW], dtype=torch.int32).view(n, WINDOW)

    held_out: dict[str, torch.Tensor] = {}
    calibration: dict[str, torch.Tensor] = {}
    domains: dict[str, object] = {}
    for domain, items in sorted(documents.items()):
        items.sort()
        eval_ids: list[int] = []
        calibration_ids: list[int] = []
        eval_documents = 0
        for _, ids in items:
            if len(eval_ids) < eval_windows * WINDOW:
                eval_ids.extend(ids)
                eval_documents += 1
            else:
                calibration_ids.extend(ids)
        held_out[domain] = pack(eval_ids, eval_windows)
        calibration[domain] = pack(calibration_ids, calibration_windows)
        domains[domain] = {
            "documents": len(items),
            "tokens": sum(len(ids) for _, ids in items),
            "eval_documents": eval_documents,
            "eval_windows": held_out[domain].shape[0],
            "calibration_documents": len(items) - eval_documents,
            "calibration_windows": calibration[domain].shape[0],
        }
    manifest = {
        "corpus": {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()},
        "window_tokens": WINDOW,
        "calibration_tokens": sum(t.numel() for t in calibration.values()),
        "domains": domains,
    }
    return Corpus(held_out, calibration, manifest)


def load_layer(reader: ShardReader, layer: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: recipe.materialize_recipe(recipe.RECIPES_BY_NAME[name], reader).to(device)
        for name in layer_major.layer_object_names(layer)
    }


class _SquareMeans:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.rows: dict[str, int] = {}

    def add(self, site: str, activations: torch.Tensor) -> None:
        flat = activations.reshape(-1, activations.shape[-1]).float()
        total = (flat * flat).sum(dim=0, dtype=torch.float64)
        self.sums[site] = self.sums.get(site, 0) + total
        self.rows[site] = self.rows.get(site, 0) + flat.shape[0]

    def mean(self, site: str) -> torch.Tensor:
        return (self.sums[site] / self.rows[site]).float().cpu()


def collect_importance(
    model_dir: str | Path,
    windows: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Return mean squared input activations of every importance-weighted object."""

    means = _SquareMeans()
    with ShardReader(model_dir) as reader:
        embedding = recipe.materialize_recipe(
            recipe.RECIPES_BY_NAME["text/token_embedding"], reader
        ).to(device)
        hidden = embedding[windows.to(device).long()]
        del embedding
        for layer in range(CFG.layers):
            started = time.perf_counter()
            weights = load_layer(reader, layer, device)

            def observe(site: str, activations: torch.Tensor, layer: int = layer) -> None:
                means.add(f"{layer}/{site}", activations)

            with torch.no_grad():
                layer_major.run_layer(weights, layer, hidden, observe)
            del weights
            print(f"layer {layer}: {time.perf_counter() - started:.1f}s", flush=True)
        final_norm = recipe.materialize_recipe(
            recipe.RECIPES_BY_NAME["text/final_norm"], reader
        ).to(device)
        for start in range(0, hidden.shape[0], 4):
            means.add("final", layer_major.final_hidden(hidden[start: start + 4], final_norm))
    return {
        name: means.mean(_importance_site(name)).contiguous()
        for name in importance_shapes()
    }


def write_importance(
    path: str | Path,
    importance: dict[str, torch.Tensor],
    manifest: dict[str, object],
) -> None:
    save_file(
        importance,
        str(path),
        metadata={"format": FORMAT, "manifest": json.dumps(manifest, sort_keys=True)},
    )


@dataclass(frozen=True, slots=True)
class Importance:
    vectors: dict[str, torch.Tensor]
    manifest: dict[str, object]
    sha256: str


def load_importance(path: str | Path) -> Importance:
    """Load and validate a calibration file against the registered inventory."""

    file = Path(path)
    expected = importance_shapes()
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != FORMAT:
            raise ValueError(f"{file}: not a {FORMAT} importance file")
        names = set(handle.keys())
        if names != set(expected):
            missing = sorted(set(expected) - names)[:3]
            extra = sorted(names - set(expected))[:3]
            raise ValueError(f"{file}: importance objects differ (missing {missing}, extra {extra})")
        vectors = {name: handle.get_tensor(name) for name in expected}
    for name, vector in vectors.items():
        if vector.dtype != torch.float32 or tuple(vector.shape) != expected[name]:
            raise ValueError(f"{file}: {name} must be float32{list(expected[name])}")
        if not bool(torch.isfinite(vector).all()) or bool((vector < 0).any()):
            raise ValueError(f"{file}: {name} must be finite and nonnegative")
    return Importance(
        vectors=vectors,
        manifest=json.loads(metadata["manifest"]),
        sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    )


def calibrate(
    model_dir: str | Path,
    corpus_path: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    resolved = pick_device(device)
    corpus = load_corpus(corpus_path, model_dir)
    windows = torch.cat([corpus.calibration[domain] for domain in sorted(corpus.calibration)])
    print(json.dumps(corpus.manifest, indent=2), flush=True)
    importance = collect_importance(model_dir, windows, resolved)
    manifest = dict(corpus.manifest)
    manifest["model"] = str(Path(model_dir).resolve())
    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_importance(output, importance, manifest)
    print(f"wrote {len(importance)} importance vectors to {output}", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    calibrate(args.model, args.corpus, args.out, device=args.device)


if __name__ == "__main__":
    main()
