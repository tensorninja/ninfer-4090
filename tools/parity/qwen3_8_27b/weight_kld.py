"""Next-token KLD and top-1 agreement of artifact weights against the BF16 checkpoint.

Canonical invocation::

    python -m tools.parity.qwen3_8_27b.weight_kld \
      --model /path/to/Qwen3.8-27B \
      --corpus /path/to/calibration-corpus.jsonl \
      --artifact old=/path/to/a.ninfer --artifact new=/path/to/b.ninfer \
      --out weight-kld.json

Scores the held-out windows that ``tools.convert.qwen3_8_27b.calibrate`` sets
aside from the same corpus, so the calibration data never overlaps the
evaluation.  Every stream runs the same layer-major reference forward
(``tools/reference/qwen3_8_27b/layer_major.py``); only the weights differ: the
reference stream uses source BF16 values and each artifact stream uses its own
decoded text objects, including the embedding, norms, and output head.  The
comparison isolates weight quantization; it does not exercise production
kernels, KV codecs, or activation quantization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import Artifact, TensorObject
from tools.artifact.layouts import decode_direct, dequantize_row_split
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_8_27b import calibrate, recipe
from tools.reference.qwen3_8_27b import layer_major
from tools.reference.qwen3_8_27b.config import CFG
from tools.reference.qwen3_8_27b.ops import linear


_LOGIT_TOKENS = 512


def decode_object(artifact: Artifact, name: str, device: torch.device) -> torch.Tensor:
    obj = artifact.find(name)
    if not isinstance(obj, TensorObject):
        raise ValueError(f"{name} is not a tensor object")
    if isinstance(get_format(obj.format), QuantFormat):
        return dequantize_row_split(artifact.payload(obj), obj.format, obj.shape, device)
    return decode_direct(artifact.payload(obj), obj.format, obj.shape, device)


def source_object(reader: ShardReader, name: str, device: torch.device) -> torch.Tensor:
    return recipe.materialize_recipe(recipe.RECIPES_BY_NAME[name], reader).to(device)


def _summary(kld: torch.Tensor, agree: torch.Tensor) -> dict[str, float | int]:
    return {
        "kld_mean": float(kld.mean()),
        "kld_p99": float(kld.quantile(0.99)),
        "top1": float(agree.mean()),
        "tokens": kld.numel(),
    }


def measure(
    model_dir: str | Path,
    corpus_path: str | Path,
    artifacts: dict[str, Path],
    *,
    layers: int = CFG.layers,
    device: torch.device = torch.device("cuda"),
) -> dict[str, object]:
    corpus = calibrate.load_corpus(corpus_path, model_dir)
    domains = sorted(corpus.eval)
    ids = torch.cat([corpus.eval[d] for d in domains]).long().to(device)
    domain_of = sum(([d] * corpus.eval[d].shape[0] for d in domains), [])
    opened = {label: Artifact.open(path) for label, path in artifacts.items()}
    report: dict[str, object] = {
        "corpus": corpus.manifest,
        "artifacts": {label: str(path) for label, path in artifacts.items()},
        "layers": [],
    }
    try:
        with ShardReader(model_dir) as reader, torch.no_grad():
            streams = {"bf16": source_object(reader, "text/token_embedding", device)[ids]}
            for label, artifact in opened.items():
                streams[label] = decode_object(artifact, "text/token_embedding", device)[ids]
                torch.cuda.empty_cache()
            for layer in range(layers):
                started = time.perf_counter()
                names = layer_major.layer_object_names(layer)
                weights = {name: source_object(reader, name, device) for name in names}
                layer_major.run_layer(weights, layer, streams["bf16"])
                for label, artifact in opened.items():
                    weights = {name: decode_object(artifact, name, device) for name in names}
                    layer_major.run_layer(weights, layer, streams[label])
                del weights
                reference = streams["bf16"].float()
                row = {"layer": layer, "seconds": round(time.perf_counter() - started, 1)}
                for label in opened:
                    difference = streams[label].float() - reference
                    row[label] = float(difference.norm() / reference.norm())
                del reference
                report["layers"].append(row)
                print(json.dumps(row), flush=True)

            final_norm = source_object(reader, "text/final_norm", device)
            head = source_object(reader, "text/output_head", device)
            results: dict[str, object] = {}
            for label, artifact in opened.items():
                artifact_norm = decode_object(artifact, "text/final_norm", device)
                artifact_head = decode_object(artifact, "text/output_head", device)
                kld: dict[str, list[torch.Tensor]] = {d: [] for d in domains}
                agree: dict[str, list[torch.Tensor]] = {d: [] for d in domains}
                for window in range(ids.shape[0]):
                    for start in range(0, ids.shape[1], _LOGIT_TOKENS):
                        span = slice(start, start + _LOGIT_TOKENS)
                        reference = torch.log_softmax(
                            linear(
                                layer_major.final_hidden(streams["bf16"][window, span], final_norm),
                                head,
                            )[:, : CFG.token_domain].float(),
                            -1,
                        )
                        candidate = torch.log_softmax(
                            linear(
                                layer_major.final_hidden(
                                    streams[label][window, span], artifact_norm
                                ),
                                artifact_head,
                            )[:, : CFG.token_domain].float(),
                            -1,
                        )
                        domain = domain_of[window]
                        kld[domain].append(
                            (reference.exp() * (reference - candidate)).sum(-1).cpu()
                        )
                        agree[domain].append(
                            (candidate.argmax(-1) == reference.argmax(-1)).float().cpu()
                        )
                        del reference, candidate
                del artifact_head
                per_domain = {d: _summary(torch.cat(kld[d]), torch.cat(agree[d])) for d in domains}
                per_domain["all"] = _summary(
                    torch.cat([torch.cat(kld[d]) for d in domains]),
                    torch.cat([torch.cat(agree[d]) for d in domains]),
                )
                results[label] = per_domain
            report["result"] = results
    finally:
        for artifact in opened.values():
            artifact.close()
    return report


def _artifact_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or label == "bf16":
        raise argparse.ArgumentTypeError("expected LABEL=PATH with a label other than bf16")
    return label, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--artifact", required=True, action="append", type=_artifact_argument)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=CFG.layers, help="debug: stop early")
    args = parser.parse_args(argv)
    report = measure(args.model, args.corpus, dict(args.artifact), layers=args.layers)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for label, per_domain in report["result"].items():
        print(label, json.dumps({d: {k: round(v, 5) if isinstance(v, float) else v
                                     for k, v in s.items()} for d, s in per_domain.items()}))


if __name__ == "__main__":
    main()
