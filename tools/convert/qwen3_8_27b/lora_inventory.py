"""Registered LoRA adapter object contract for the Qwen3.8-27B target.

An adapter artifact is a normal ``.ninfer`` v2 container carrying BF16
``contiguous-le-v1`` tensors.  It introduces no numeric format and no storage
layout, and restates no frontend resource: the tokenizer, chat template and
preprocessor configs belong to the base artifact.  A generative adapter
carries only rank-two factors; a System One decision adapter additionally
carries the pointer head and its ``decision/metadata`` resource (below).

The registered site table below is the whole contract.  A site exists only
where NInfer's schedule exposes a plain BF16 destination that a low-rank delta
can accumulate into after the fused projection has run.  ``gate_proj``,
``up_proj`` and the Gated DeltaNet input projections are deliberately absent:
their deltas would have to land before ``silu(g) * u`` and before the fused
causal convolution respectively, which no post-hoc additive pass can express.

The vocabulary endpoints are absent for a different reason.  ``lm_head`` would
satisfy that rule - its logits destination is already a plain contiguous BF16
matrix - but this table and its object names are layer-indexed and have no slot
for a site outside the decoder stack.  ``embed_tokens`` has no matmul
destination at all, because ``ops::embedding`` is a gather.  The serving policy
is documented in ``docs/maintainer/qwen3.8-27b-lora-adapters.md``; training-side
module roles and experiments belong to the separate ``llm-datasets`` repository.

Every site is described in *stored* terms.  ``hf_heads`` and ``hf_head_rows``
describe the source ``lora_B`` row space, and ``hf_row_begin``/``hf_row_end``
select the rows this site owns inside one head:

    B_site = B_hf.reshape(hf_heads, hf_head_rows, rank)[:, begin:end, :]
                 .reshape(out_features, rank)

For every projection except the fused query/gate parent this degenerates to the
identity (``hf_heads == 1``).  The Qwen3.8 attention parent stores 256 query
rows followed by 256 output-gate rows for each of its 24 heads, so query and
gate are two row selections of one source module and share one ``lora_A``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.convert.qwen3_8.common.inventory import BF16, TensorSpec, tensor_spec


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "lora-bf16"
RECIPE_ID = "qwen3_8_27b_lora-v1"

HIDDEN = 5120
LAYERS = 64
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GDN_LAYERS = tuple(layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS)

QUERY_HEADS = 24
QUERY_HEAD_ROWS = 512
QUERY_ROWS_PER_HEAD = 256
QUERY_SIZE = QUERY_HEADS * QUERY_ROWS_PER_HEAD
KV_SIZE = 1024
ATTENTION_VALUE_DIM = 6144
GDN_VALUE_DIM = 6144
INTERMEDIATE = 17408

SUPPORTED_RANKS = (8, 16, 32, 64)
MAXIMUM_RANK = 64

FULL = "full"
GDN = "gdn"
ALL = "all"


@dataclass(frozen=True, slots=True)
class LoraSiteSpec:
    """One registered low-rank correction site."""

    key: str
    hf_module: str
    layer_group: str
    in_features: int
    hf_heads: int
    hf_head_rows: int
    hf_row_begin: int
    hf_row_end: int
    a_object: str
    b_object: str

    @property
    def out_features(self) -> int:
        return self.hf_heads * (self.hf_row_end - self.hf_row_begin)

    @property
    def hf_out_features(self) -> int:
        return self.hf_heads * self.hf_head_rows

    @property
    def is_row_selection(self) -> bool:
        return self.hf_heads != 1 or self.hf_row_begin != 0 or self.hf_row_end != self.hf_head_rows

    def layers(self) -> tuple[int, ...]:
        if self.layer_group == FULL:
            return FULL_ATTENTION_LAYERS
        if self.layer_group == GDN:
            return GDN_LAYERS
        return tuple(range(LAYERS))


def _site(
    key: str,
    hf_module: str,
    layer_group: str,
    in_features: int,
    out_features: int,
    a_object: str,
    b_object: str,
    *,
    hf_heads: int = 1,
    hf_head_rows: int | None = None,
    hf_row_begin: int = 0,
    hf_row_end: int | None = None,
) -> LoraSiteSpec:
    head_rows = hf_head_rows if hf_head_rows is not None else out_features
    row_end = hf_row_end if hf_row_end is not None else head_rows
    spec = LoraSiteSpec(
        key=key,
        hf_module=hf_module,
        layer_group=layer_group,
        in_features=in_features,
        hf_heads=hf_heads,
        hf_head_rows=head_rows,
        hf_row_begin=hf_row_begin,
        hf_row_end=row_end,
        a_object=a_object,
        b_object=b_object,
    )
    if spec.out_features != out_features:
        raise ValueError(
            f"site {key}: row selection yields {spec.out_features} rows, expected {out_features}"
        )
    return spec


# Ordered exactly as the artifact stores them inside one layer.
SITE_SPECS: tuple[LoraSiteSpec, ...] = (
    _site(
        "attention/query",
        "self_attn.q_proj",
        FULL,
        HIDDEN,
        QUERY_SIZE,
        "attention/query_gate/lora_a",
        "attention/query/lora_b",
        hf_heads=QUERY_HEADS,
        hf_head_rows=QUERY_HEAD_ROWS,
        hf_row_begin=0,
        hf_row_end=QUERY_ROWS_PER_HEAD,
    ),
    _site(
        "attention/gate",
        "self_attn.q_proj",
        FULL,
        HIDDEN,
        QUERY_SIZE,
        "attention/query_gate/lora_a",
        "attention/gate/lora_b",
        hf_heads=QUERY_HEADS,
        hf_head_rows=QUERY_HEAD_ROWS,
        hf_row_begin=QUERY_ROWS_PER_HEAD,
        hf_row_end=QUERY_HEAD_ROWS,
    ),
    _site(
        "attention/key",
        "self_attn.k_proj",
        FULL,
        HIDDEN,
        KV_SIZE,
        "attention/key/lora_a",
        "attention/key/lora_b",
    ),
    _site(
        "attention/value",
        "self_attn.v_proj",
        FULL,
        HIDDEN,
        KV_SIZE,
        "attention/value/lora_a",
        "attention/value/lora_b",
    ),
    _site(
        "attention/output",
        "self_attn.o_proj",
        FULL,
        ATTENTION_VALUE_DIM,
        HIDDEN,
        "attention/output/lora_a",
        "attention/output/lora_b",
    ),
    _site(
        "gdn/output",
        "linear_attn.out_proj",
        GDN,
        GDN_VALUE_DIM,
        HIDDEN,
        "gdn/output/lora_a",
        "gdn/output/lora_b",
    ),
    _site(
        "mlp/down",
        "mlp.down_proj",
        ALL,
        INTERMEDIATE,
        HIDDEN,
        "mlp/down/lora_a",
        "mlp/down/lora_b",
    ),
)

SITE_BY_KEY: dict[str, LoraSiteSpec] = {spec.key: spec for spec in SITE_SPECS}
SITE_KEYS: tuple[str, ...] = tuple(spec.key for spec in SITE_SPECS)

# HF leaf module names a trained adapter may target, and the sites each one feeds.
SUPPORTED_HF_MODULES: dict[str, tuple[str, ...]] = {}
for _spec in SITE_SPECS:
    _leaf = _spec.hf_module.rsplit(".", 1)[-1]
    SUPPORTED_HF_MODULES.setdefault(_leaf, ())
    SUPPORTED_HF_MODULES[_leaf] += (_spec.key,)

# Rejected with an explicit message rather than silently dropped. These reasons define NInfer's
# conversion boundary; training-side experiments belong to the qwen3_8_27b target profile in the
# separate llm-datasets repository.
_SWIGLU_BLOCKER = (
    "the delta must land before silu(gate) * up inside ops::linear_swiglu, which needs an "
    "optional pre-activation addend in every schedule route"
)
REJECTED_HF_MODULES: dict[str, str] = {
    "gate_proj": _SWIGLU_BLOCKER,
    "up_proj": _SWIGLU_BLOCKER,
    "in_proj_qkv": "the delta would have to land before the fused causal convolution and its SiLU",
    "in_proj_z": "the delta would have to land before the fused causal convolution and its SiLU",
    "in_proj_a": "fused into ops::gdn_norm_gating_proj",
    "in_proj_b": "fused into ops::gdn_norm_gating_proj",
    "lm_head": "the logits destination already satisfies the lora_delta_add contract, but this "
               "table and its object names are layer-indexed",
    "embed_tokens": "ops::embedding is a gather, so this needs a new Op computing "
                    "out[:, t] += B @ A[:, ids[t]]",
}


def require_rank(rank: int) -> int:
    if rank not in SUPPORTED_RANKS:
        raise ValueError(
            f"LoRA rank {rank} is not registered; supported ranks are {list(SUPPORTED_RANKS)}"
        )
    return rank


def require_sites(site_keys: frozenset[str]) -> frozenset[str]:
    if not site_keys:
        raise ValueError("a LoRA adapter must target at least one registered site")
    unknown = sorted(site_keys - set(SITE_KEYS))
    if unknown:
        raise ValueError(f"unknown LoRA sites: {unknown}")
    return site_keys


def a_object_name(layer: int, spec: LoraSiteSpec) -> str:
    return f"text/layers/{layer}/{spec.a_object}"


def b_object_name(layer: int, spec: LoraSiteSpec) -> str:
    return f"text/layers/{layer}/{spec.b_object}"


def build_tensor_specs(rank: int, site_keys: frozenset[str]) -> tuple[TensorSpec, ...]:
    """Ordered complete tensor inventory for one adapter.

    Objects are layer-major, and inside a layer follow ``SITE_SPECS`` order.  A
    shared ``lora_A`` (query and gate) is emitted once, at its first use.
    """

    require_rank(rank)
    require_sites(site_keys)
    specs: list[TensorSpec] = []
    emitted: set[str] = set()
    for layer in range(LAYERS):
        for site in SITE_SPECS:
            if site.key not in site_keys or layer not in site.layers():
                continue
            a_name = a_object_name(layer, site)
            if a_name not in emitted:
                specs.append(tensor_spec(a_name, (rank, site.in_features), BF16))
                emitted.add(a_name)
            specs.append(tensor_spec(b_object_name(layer, site), (site.out_features, rank), BF16))
    return tuple(specs)


def parameter_count(rank: int, site_keys: frozenset[str]) -> int:
    return sum(spec.shape[0] * spec.shape[1] for spec in build_tensor_specs(rank, site_keys))


ALL_SITE_KEYS: frozenset[str] = frozenset(SITE_KEYS)


# System One decision adapters.  A decision adapter is a generative adapter's
# factors plus a pointer head (kev's ``PointerHead``) read from the final-norm
# hidden states at each option's closing delimiter and at the decide delimiter:
#
#     q = Wq h_decide + bq,  k_i = Wk h_option_i + bk,
#     p = softmax_i((k_i . q) * logit_scale / temperature)
#
# The head is four BF16 tensors; everything the engine publishes about the
# model (temperature, description, release date) and the framing it must
# reproduce (readout, escape rule, delimiters) is one JSON resource.  All five
# objects are present together or not at all.
DECISION_POINTER_DIM = 256
DECISION_LOGIT_SCALE = 0.0625
DECISION_FORMAT = "ninfer-decision-head"
DECISION_FORMAT_VERSION = 1
DECISION_READOUT = "final_norm"
DECISION_ESCAPE = "kev-v1"
DECISION_METADATA_OBJECT = "decision/metadata"
DECISION_HEAD_TENSORS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("decision/head/query/weight", (DECISION_POINTER_DIM, HIDDEN)),
    ("decision/head/query/bias", (DECISION_POINTER_DIM,)),
    ("decision/head/key/weight", (DECISION_POINTER_DIM, HIDDEN)),
    ("decision/head/key/bias", (DECISION_POINTER_DIM,)),
)
# Delimiters by role, in encoding order, and their ids in the registered base
# vocabulary.  The engine resolves the names through the base tokenizer at load.
DECISION_DELIMITERS: dict[str, str] = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "option_open": "<|box_start|>",
    "option_close": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}
DECISION_DELIMITER_IDS: dict[str, int] = {
    "state": 248060,
    "question": 248061,
    "option_open": 248049,
    "option_close": 248050,
    "decide": 248062,
}
DECISION_METADATA_KEYS = frozenset(
    {
        "format",
        "format_version",
        "pointer_dim",
        "hidden_size",
        "logit_scale",
        "temperature",
        "readout",
        "escape",
        "delimiters",
        "description",
        "release_date",
        "base_model",
    }
)


def decision_head_specs() -> tuple[TensorSpec, ...]:
    """The pointer head's tensors, in the order a decision adapter stores them."""

    return tuple(tensor_spec(name, shape, BF16) for name, shape in DECISION_HEAD_TENSORS)


__all__ = [
    "ALL_SITE_KEYS",
    "DECISION_DELIMITERS",
    "DECISION_DELIMITER_IDS",
    "DECISION_ESCAPE",
    "DECISION_FORMAT",
    "DECISION_FORMAT_VERSION",
    "DECISION_HEAD_TENSORS",
    "DECISION_LOGIT_SCALE",
    "DECISION_METADATA_KEYS",
    "DECISION_METADATA_OBJECT",
    "DECISION_POINTER_DIM",
    "DECISION_READOUT",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "LAYERS",
    "MAXIMUM_RANK",
    "MODEL_ID",
    "RECIPE_ID",
    "REJECTED_HF_MODULES",
    "SITE_BY_KEY",
    "SITE_KEYS",
    "SITE_SPECS",
    "SUPPORTED_HF_MODULES",
    "SUPPORTED_RANKS",
    "WEIGHTS_ID",
    "LoraSiteSpec",
    "a_object_name",
    "b_object_name",
    "build_tensor_specs",
    "decision_head_specs",
    "parameter_count",
    "require_rank",
    "require_sites",
]
