"""Generate synthetic PEFT LoRA adapters for Qwen3.8-27B bring-up.

    python3 -m tools.convert.qwen3_8_27b.make_synthetic_lora \
        --kind canary --rank 16 --alpha 32 --out out/synthetic/canary

A LoRA ``A``/``B`` pair is independent of the weights it corrects, so a
synthetic adapter needs no checkpoint, no GPU and no trainer.  The output is a
real PEFT directory rather than a ``.ninfer`` file, so the fixtures exercise
``convert_lora.py`` as well as the engine.

Four kinds, in the order they should be brought up:

``zero``
    Exactly PEFT's own initialisation: ``A`` Kaiming-uniform, ``B`` exactly
    zero, so ``delta = B @ A`` is exactly zero.  ``A`` stays nonzero on purpose
    - the A-projection must still be exercised.  Engine output must be
    unchanged, and the LoRA kernels must still be launched, or the test is
    vacuous.

``canary``
    Rank-one one-hot per site: ``A[0, k0] = 1`` and ``B[n0, 0] = c``, so
    ``delta[n0, k0] = c * scale`` and every other entry is exactly zero.  One
    scalar comparison then localises a transposed factor, a swapped ``N``/``K``
    stride, a delta routed to the wrong destination, a wrong layer offset, a
    missing or doubled scale fold, and every query/gate de-interleave error.

``random``
    Small dense Gaussian factors, for oracle agreement on the dense path.

``distinct``
    ``--variant a|b`` selects one of two canaries with different constants and
    different targets, for per-request routing and mixed-batch tests.

``--decision-head`` adds a System One pointer head beside any kind: a seeded
Gaussian FP32 ``decision_head.safetensors`` carrying the trainer's hand-off
metadata, so ``convert_lora.py --decision-head`` makes a decision adapter whose
probabilities are neither uniform nor saturated.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.convert.qwen3_8_27b import lora_inventory as inventory


KINDS = ("zero", "canary", "random", "distinct")
VARIANTS = ("a", "b")

# Distinct constants per variant so a mis-routed delta names its own source.
_VARIANT_CONSTANT = {"a": 1.0, "b": -3.0}
_VARIANT_OFFSET = {"a": 0, "b": 1}

_MODULE_PREFIX = "base_model.model.model.language_model"


def _rank_slots() -> dict[str, int]:
    """Rank index each site occupies inside the ``lora_A`` it may share.

    ``attention/query`` and ``attention/gate`` are two row selections of one
    source module and therefore share one ``lora_A``.  A canary must place them
    in different rank slots, or the shared factor would carry both one-hot
    columns and every delta built from it would gain a second nonzero entry.
    """

    slots: dict[str, int] = {}
    used: dict[str, int] = {}
    for site in inventory.SITE_SPECS:
        slot = used.get(site.a_object, 0)
        slots[site.key] = slot
        used[site.a_object] = slot + 1
    return slots


SITE_RANK_SLOT: dict[str, int] = _rank_slots()
MINIMUM_CANARY_RANK = max(SITE_RANK_SLOT.values()) + 1


def _canary_indices(layer: int, site_index: int, site: inventory.LoraSiteSpec, offset: int) -> tuple[int, int]:
    """Per-(layer, site) one-hot coordinates, distinct across the whole model."""

    k0 = (layer * 7 + site_index * 31 + offset * 3) % site.in_features
    n0 = (layer * 13 + site_index * 101 + offset * 5) % site.out_features
    return n0, k0


def _hf_row(site: inventory.LoraSiteSpec, n0: int) -> int:
    """Map a site-space output row back into the source ``lora_B`` row space."""

    if not site.is_row_selection:
        return n0
    rows = site.hf_row_end - site.hf_row_begin
    head, within = divmod(n0, rows)
    return head * site.hf_head_rows + site.hf_row_begin + within


def _kaiming_a(rank: int, in_features: int, generator: torch.Generator) -> torch.Tensor:
    bound = 1.0 / math.sqrt(in_features)
    return (
        torch.empty(rank, in_features, dtype=torch.float32)
        .uniform_(-bound, bound, generator=generator)
    )


def build_state_dict(
    kind: str,
    rank: int,
    variant: str,
    seed: int,
    site_keys: frozenset[str],
    sigma: float,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    constant = _VARIANT_CONSTANT[variant]
    offset = _VARIANT_OFFSET[variant]

    # One source module can feed several sites (the fused query/gate parent),
    # so build in source space and let the converter perform the row selection.
    modules: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    for site_index, site in enumerate(inventory.SITE_SPECS):
        if site.key not in site_keys:
            continue
        for layer in site.layers():
            key = (layer, site.hf_module)
            entry = modules.get(key)
            if entry is None:
                if kind == "zero":
                    a = _kaiming_a(rank, site.in_features, generator)
                elif kind == "random":
                    a = torch.empty(rank, site.in_features, dtype=torch.float32).normal_(
                        0.0, sigma, generator=generator
                    )
                else:
                    a = torch.zeros(rank, site.in_features, dtype=torch.float32)
                entry = {
                    "A": a,
                    "B": torch.zeros(site.hf_out_features, rank, dtype=torch.float32),
                }
                modules[key] = entry

            if kind == "zero":
                continue
            if kind == "random":
                entry["B"].normal_(0.0, sigma, generator=generator)
                continue

            n0, k0 = _canary_indices(layer, site_index, site, offset)
            slot = SITE_RANK_SLOT[site.key]
            entry["A"][slot, k0] = 1.0
            entry["B"][_hf_row(site, n0), slot] = constant

    state: dict[str, torch.Tensor] = {}
    for (layer, module), sides in sorted(modules.items()):
        stem = f"{_MODULE_PREFIX}.layers.{layer}.{module}"
        state[f"{stem}.lora_A.weight"] = sides["A"].contiguous()
        state[f"{stem}.lora_B.weight"] = sides["B"].contiguous()
    return state


def expected_canary_entries(
    rank: int, variant: str, site_keys: frozenset[str]
) -> list[dict[str, object]]:
    """The exact nonzero delta entries a canary must produce, in site space.

    This is the oracle the converter and engine tests compare against, and it is
    derived independently of the state dict the generator writes.
    """

    constant = _VARIANT_CONSTANT[variant]
    offset = _VARIANT_OFFSET[variant]
    entries: list[dict[str, object]] = []
    for site_index, site in enumerate(inventory.SITE_SPECS):
        if site.key not in site_keys:
            continue
        for layer in site.layers():
            n0, k0 = _canary_indices(layer, site_index, site, offset)
            entries.append(
                {
                    "layer": layer,
                    "site": site.key,
                    "row": n0,
                    "column": k0,
                    "value": constant,
                    "hf_row": _hf_row(site, n0),
                    "rank_slot": SITE_RANK_SLOT[site.key],
                }
            )
    return entries


def build_adapter_config(
    rank: int, alpha: float, site_keys: frozenset[str], use_rslora: bool
) -> dict[str, object]:
    leaves = sorted(
        {
            inventory.SITE_BY_KEY[key].hf_module.rsplit(".", 1)[-1]
            for key in site_keys
        }
    )
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "synthetic/Qwen3.8-27B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": leaves,
        "target_parameters": None,
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": use_rslora,
    }


def build_decision_head(seed: int, sigma: float) -> dict[str, torch.Tensor]:
    """Seeded Gaussian pointer head in the trainer's key names and FP32 dtype."""

    generator = torch.Generator().manual_seed(seed + 0x5E1F)
    tensors: dict[str, torch.Tensor] = {}
    for name, shape in (
        ("query.weight", (inventory.DECISION_POINTER_DIM, inventory.HIDDEN)),
        ("query.bias", (inventory.DECISION_POINTER_DIM,)),
        ("key.weight", (inventory.DECISION_POINTER_DIM, inventory.HIDDEN)),
        ("key.bias", (inventory.DECISION_POINTER_DIM,)),
    ):
        tensors[name] = torch.empty(shape, dtype=torch.float32).normal_(
            0.0, sigma, generator=generator
        )
    return tensors


def write_decision_head(out_dir: Path, seed: int, sigma: float, temperature: float) -> None:
    metadata = {
        "format": "systemone-pointer-head",
        "format_version": "1",
        "pointer_dim": str(inventory.DECISION_POINTER_DIM),
        "hidden_size": str(inventory.HIDDEN),
        "logit_scale": repr(inventory.DECISION_LOGIT_SCALE),
        "temperature": repr(float(temperature)),
        "readout": inventory.DECISION_READOUT,
        "escape": inventory.DECISION_ESCAPE,
        "delimiters": json.dumps(inventory.DECISION_DELIMITERS),
        "base_model": "synthetic/Qwen3.8-27B",
        "description": f"synthetic decision adapter (seed {seed})",
        "release_date": "2026-01-01",
    }
    save_file(
        build_decision_head(seed, sigma),
        str(out_dir / "decision_head.safetensors"),
        metadata=metadata,
    )


def write_adapter(
    out_dir: Path,
    kind: str,
    rank: int,
    alpha: float,
    variant: str,
    seed: int,
    site_keys: frozenset[str],
    sigma: float,
    use_rslora: bool,
) -> dict[str, object]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    inventory.require_rank(rank)
    inventory.require_sites(site_keys)
    if kind != "distinct" and variant != "a":
        raise ValueError("--variant applies only to --kind distinct")

    state = build_state_dict(kind, rank, variant, seed, site_keys, sigma)
    config = build_adapter_config(rank, alpha, site_keys, use_rslora)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    scale = alpha / (rank**0.5) if use_rslora else alpha / rank
    manifest: dict[str, object] = {
        "kind": kind,
        "variant": variant if kind == "distinct" else None,
        "rank": rank,
        "lora_alpha": alpha,
        "use_rslora": use_rslora,
        "folded_scale": scale,
        "seed": seed,
        "sigma": sigma if kind == "random" else None,
        "sites": sorted(site_keys),
        "tensors": len(state),
        "parameters": sum(int(t.numel()) for t in state.values()),
    }
    if kind in ("canary", "distinct"):
        manifest["expected_delta_entries"] = expected_canary_entries(rank, variant, site_keys)
    (out_dir / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--variant", choices=VARIANTS, default="a")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--sigma", type=float, default=0.02, help="stddev for --kind random")
    parser.add_argument("--use-rslora", action="store_true")
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help=f"registered site keys, default all: {list(inventory.SITE_KEYS)}",
    )
    parser.add_argument(
        "--decision-head",
        action="store_true",
        help="also write a seeded System One pointer head (decision_head.safetensors)",
    )
    parser.add_argument("--head-sigma", type=float, default=0.01, help="pointer head stddev")
    parser.add_argument("--temperature", type=float, default=1.0, help="pointer head temperature")
    args = parser.parse_args()

    site_keys = frozenset(args.sites) if args.sites else inventory.ALL_SITE_KEYS
    manifest = write_adapter(
        out_dir=args.out,
        kind=args.kind,
        rank=args.rank,
        alpha=args.alpha,
        variant=args.variant,
        seed=args.seed,
        site_keys=site_keys,
        sigma=args.sigma,
        use_rslora=args.use_rslora,
    )
    if args.decision_head:
        write_decision_head(args.out, args.seed, args.head_sigma, args.temperature)
    print(f"wrote {args.out}")
    print(f"  kind {manifest['kind']}  rank {manifest['rank']}  scale {manifest['folded_scale']}")
    print(f"  {manifest['tensors']} tensors, {manifest['parameters']} parameters")
    if args.decision_head:
        print(f"  decision head sigma {args.head_sigma}  temperature {args.temperature}")


if __name__ == "__main__":
    main()
