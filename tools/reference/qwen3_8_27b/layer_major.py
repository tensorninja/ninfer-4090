"""Layer-major batched text forward over fixed token windows.

Calibration and weight-quantization parity both push many independent windows
through the text stack one layer at a time, so only one layer's weights is
resident.  Every window starts at position 0 with empty recurrent state. The
operators are the reference operators in ``ops.py``; the weights are a mapping
from artifact object name to a device tensor, so callers can supply source BF16
values or decoded artifact values.

``observe(site, activations)`` receives the input of every projection matrix:

=============  ===========================================================
site           consumers
=============  ===========================================================
``attn_in``    ``attention/query_key``, ``attention/gate_value``
``attn_out``   ``attention/output``
``gdn_in``     ``gdn/query_key``, ``gdn/value_z``
``gdn_out``    ``gdn/output``
``mlp_in``     ``mlp/gate_up``
``mlp_mid``    ``mlp/down``
=============  ===========================================================
"""

from __future__ import annotations

from typing import Callable, Mapping

import torch
import torch.nn.functional as F

from .config import ATTN_SCALE, CFG, GDN_SCALE
from .ops import (
    apply_rope,
    gdn_gating,
    l2norm,
    linear,
    residual_add,
    rmsnorm,
    sigmoid_mul,
    silu_mul,
)

Observer = Callable[[str, torch.Tensor], None]
Weights = Mapping[str, torch.Tensor]

SITE_OF_MATRIX = {
    "attention/query_key": "attn_in",
    "attention/gate_value": "attn_in",
    "attention/output": "attn_out",
    "gdn/query_key": "gdn_in",
    "gdn/value_z": "gdn_in",
    "gdn/output": "gdn_out",
    "mlp/gate_up": "mlp_in",
    "mlp/down": "mlp_mid",
}


def ignore(_site: str, _activations: torch.Tensor) -> None:
    pass


def layer_object_names(layer: int) -> tuple[str, ...]:
    prefix = f"text/layers/{layer}/"
    mixer = (
        ("attention/query_key", "attention/gate_value", "attention/query_norm",
         "attention/key_norm", "attention/output")
        if CFG.is_full(layer)
        else ("gdn/convolution", "gdn/a_projection", "gdn/b_projection", "gdn/query_key",
              "gdn/value_z", "gdn/a_log", "gdn/dt_bias", "gdn/norm", "gdn/output")
    )
    shared = ("input_norm", "post_attention_norm", "mlp/gate_up", "mlp/down")
    return tuple(prefix + name for name in shared[:1] + mixer + shared[1:])


def _attention(w: Weights, prefix: str, x: torch.Tensor, observe: Observer) -> torch.Tensor:
    b, t, _ = x.shape
    positions = torch.arange(t, device=x.device).repeat(b)
    h = rmsnorm(x, w[prefix + "input_norm"])
    observe("attn_in", h)
    qk = linear(h, w[prefix + "attention/query_key"])
    gv = linear(h, w[prefix + "attention/gate_value"])
    q = qk[..., : CFG.q_size].reshape(b * t, CFG.q_heads, CFG.head_dim)
    k = qk[..., CFG.q_size:].reshape(b * t, CFG.kv_heads, CFG.head_dim)
    gate = gv[..., : CFG.q_size].reshape(b, t, CFG.q_heads, CFG.head_dim)
    v = gv[..., CFG.q_size:].reshape(b, t, CFG.kv_heads, CFG.head_dim)
    q = apply_rope(rmsnorm(q, w[prefix + "attention/query_norm"]), positions)
    k = apply_rope(rmsnorm(k, w[prefix + "attention/key_norm"]), positions)
    out = F.scaled_dot_product_attention(
        q.reshape(b, t, CFG.q_heads, -1).transpose(1, 2),
        k.reshape(b, t, CFG.kv_heads, -1).transpose(1, 2),
        v.transpose(1, 2),
        is_causal=True,
        scale=ATTN_SCALE,
        enable_gqa=True,
    ).transpose(1, 2).to(torch.bfloat16)
    mixed = sigmoid_mul(gate, out).reshape(b, t, CFG.q_size)
    observe("attn_out", mixed)
    return residual_add(x, linear(mixed, w[prefix + "attention/output"]))


def _gdn(w: Weights, prefix: str, x: torch.Tensor, observe: Observer) -> torch.Tensor:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, t, _ = x.shape
    h = rmsnorm(x, w[prefix + "input_norm"])
    observe("gdn_in", h)
    qk = linear(h, w[prefix + "gdn/query_key"])
    vz = linear(h, w[prefix + "gdn/value_z"])
    value, z = vz[..., : CFG.value_dim], vz[..., CFG.value_dim:]
    a = linear(h, w[prefix + "gdn/a_projection"])
    beta_logit = linear(h, w[prefix + "gdn/b_projection"])
    qkv = torch.cat((qk, value), dim=-1)
    # Stored [width, channels]; depthwise conv1d wants [channels, 1, width].
    kernel = w[prefix + "gdn/convolution"].t().float().unsqueeze(1)
    sequence = F.pad(qkv.float().transpose(1, 2), (CFG.conv_width - 1, 0))
    qkv = (
        F.silu(F.conv1d(sequence, kernel, groups=CFG.conv_dim))
        .transpose(1, 2)
        .to(torch.bfloat16)
    )
    g, beta = gdn_gating(a, beta_logit, w[prefix + "gdn/a_log"], w[prefix + "gdn/dt_bias"])
    q = l2norm(qkv[..., : CFG.key_dim].reshape(b, t, CFG.gdn_k_heads, CFG.gdn_k_dim))
    k = l2norm(
        qkv[..., CFG.key_dim: 2 * CFG.key_dim].reshape(b, t, CFG.gdn_k_heads, CFG.gdn_k_dim)
    )
    # FLA's Triton kernels need independent aligned base pointers.
    v = qkv[..., 2 * CFG.key_dim:].reshape(b, t, CFG.gdn_v_heads, CFG.gdn_v_dim).clone()
    out, _ = chunk_gated_delta_rule(
        q.clone(), k.clone(), v, g.clone(), beta.clone(),
        scale=GDN_SCALE, output_final_state=False,
    )
    out = rmsnorm(
        out.to(torch.bfloat16),
        w[prefix + "gdn/norm"],
        unit_offset=False,
        z=z.reshape(b, t, CFG.gdn_v_heads, CFG.gdn_v_dim),
    )
    mixed = out.reshape(b, t, CFG.value_dim)
    observe("gdn_out", mixed)
    return residual_add(x, linear(mixed, w[prefix + "gdn/output"]))


def _mlp(w: Weights, prefix: str, x: torch.Tensor, observe: Observer) -> torch.Tensor:
    h = rmsnorm(x, w[prefix + "post_attention_norm"])
    observe("mlp_in", h)
    gate, up = linear(h, w[prefix + "mlp/gate_up"]).split(CFG.intermediate, dim=-1)
    activated = silu_mul(gate, up)
    observe("mlp_mid", activated)
    return residual_add(x, linear(activated, w[prefix + "mlp/down"]))


def run_layer(
    weights: Weights,
    layer: int,
    hidden: torch.Tensor,
    observe: Observer = ignore,
    micro_batch: int = 4,
) -> None:
    """Advance BF16 ``hidden[windows, tokens, 5120]`` through one layer in place."""

    prefix = f"text/layers/{layer}/"
    mixer = _attention if CFG.is_full(layer) else _gdn
    for start in range(0, hidden.shape[0], micro_batch):
        x = mixer(weights, prefix, hidden[start: start + micro_batch], observe)
        hidden[start: start + micro_batch] = _mlp(weights, prefix, x, observe)


def final_hidden(hidden: torch.Tensor, final_norm: torch.Tensor) -> torch.Tensor:
    """Input of ``text/output_head`` and ``text/draft_head``."""

    return rmsnorm(hidden, final_norm)


__all__ = [
    "SITE_OF_MATRIX",
    "final_hidden",
    "ignore",
    "layer_object_names",
    "run_layer",
]
