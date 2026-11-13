"""Grouped symmetric quantization used by NInfer artifact converters.

The persistent numeric format fixes the code range, group size, and binary16
scale.  Model-specific recipes decide which tensors use those formats and which
importance vector weights each matrix; this module performs the one registered
encoder profile, ``WEIGHTED_SCALE_SEARCH_F16_RECIP_RNE_V2``.

For every group the encoder evaluates a fixed, ordered list of binary16 scale
candidates and keeps the one with the smallest importance-weighted squared
reconstruction error.  Codes are always round-to-nearest against the stored
scale, so the stored scale alone determines them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.artifact.layouts import encode_row_split, row_split_geometry
from tools.artifact.numeric import QuantFormat, get_format


_FP16_MIN_SUBNORMAL = 2.0**-24

# Shrink/stretch factors applied to the smallest non-clipping scale, in order.
SEARCH_FACTORS = tuple((70 + 2 * index) / 100 for index in range(21))
# Every logical column keeps this fraction of the mean importance so a column
# that calibration never excited still counts.
IMPORTANCE_FLOOR = 1.0e-3
# Elements per working chunk; groups are independent, so chunking is exact.
_CHUNK_ELEMENTS = 1 << 24


@dataclass(frozen=True, slots=True)
class QuantizedMatrix:
    """Physical code groups and binary16 scales for one logical matrix."""

    codes: torch.Tensor
    scales: torch.Tensor


def pick_device(preferred: str | torch.device = "cuda") -> torch.device:
    device = torch.device(preferred)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def column_weights(
    importance: torch.Tensor | None,
    k: int,
    k_pad: int,
    device: torch.device,
) -> torch.Tensor:
    """Return binary64 per-column error weights, zero on physical padding."""

    weights = torch.zeros(k_pad, dtype=torch.float64, device=device)
    if importance is None:
        weights[:k] = 1.0
        return weights
    if tuple(importance.shape) != (k,):
        raise ValueError(
            f"importance must have shape ({k},), got {tuple(importance.shape)}"
        )
    values = importance.detach().to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("importance must be finite and nonnegative")
    mean = float(values.mean())
    if mean <= 0.0:
        weights[:k] = 1.0
        return weights
    weights[:k] = values + IMPORTANCE_FLOOR * mean
    return weights


def _binary16_scale(value: torch.Tensor) -> torch.Tensor:
    """Round nonnegative binary64 scale values through binary32 to binary16.

    An overflowing candidate becomes infinity and is never selected.
    """

    scale = value.to(torch.float32).to(torch.float16)
    rescue = (scale == 0) & (value > 0)
    return torch.where(
        rescue, torch.full_like(scale, _FP16_MIN_SUBNORMAL), scale
    )


def _codes(grouped: torch.Tensor, scale: torch.Tensor, spec: QuantFormat) -> torch.Tensor:
    """Round-to-nearest codes for binary32 groups against binary16 scales."""

    positive = scale > 0
    reciprocal = torch.where(
        positive,
        (1.0 / scale.to(torch.float64)).to(torch.float32),
        torch.zeros_like(scale, dtype=torch.float32),
    )
    return torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    )


def _search_chunk(
    grouped: torch.Tensor,
    weights: torch.Tensor,
    spec: QuantFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select codes and binary16 scales for ``[rows, groups, G]`` binary32 values."""

    x64 = grouped.to(torch.float64)
    # Adding +0 turns a -0 maximum into +0, so an all-zero group stores +0.
    positive_max = grouped.amax(dim=-1).clamp(min=0).to(torch.float64) + 0.0
    negative_max = (-grouped).amax(dim=-1).clamp(min=0).to(torch.float64) + 0.0
    absmax = torch.maximum(positive_max, negative_max)
    base = torch.maximum(positive_max / spec.qmax, negative_max / -spec.qmin)

    best_scale = torch.zeros_like(absmax, dtype=torch.float16)
    best_error = torch.full_like(absmax, float("inf"))

    def consider(candidate: torch.Tensor) -> torch.Tensor:
        nonlocal best_scale, best_error
        scale = _binary16_scale(candidate)
        finite = torch.isfinite(scale)
        scale = torch.where(finite, scale, torch.zeros_like(scale))
        codes = _codes(grouped, scale, spec).to(torch.float64)
        residual = x64 - scale.to(torch.float64).unsqueeze(-1) * codes
        error = (weights * residual * residual).sum(dim=-1)
        better = finite & (error < best_error)
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        return codes

    if not bool(torch.isfinite(_binary16_scale(absmax / spec.qmax)).all()):
        raise ValueError("grouped quantization scale is not finite and positive")
    consider(absmax / spec.qmax)
    for factor in SEARCH_FACTORS:
        candidate = base * factor
        codes = consider(candidate)
        numerator = (weights * x64 * codes).sum(dim=-1)
        denominator = (weights * codes * codes).sum(dim=-1)
        refit = torch.where(
            denominator > 0, numerator / denominator.clamp(min=1e-300), candidate
        )
        consider(refit.clamp(min=0))

    return _codes(grouped, best_scale, spec).to(torch.int8), best_scale


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    importance: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values, including registered K padding.

    ``importance`` is an optional nonnegative ``[K]`` vector, normally the mean
    squared calibration activation of each input column.  ``None`` weights
    every column equally.  Padding values are zero, carry zero weight, and do
    not affect a partially populated final group.
    """

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(f"grouped quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)
    weights = column_weights(importance, geometry.k, geometry.k_pad, target).reshape(
        geometry.groups_per_row, spec.group_size
    )
    codes = torch.empty(
        (geometry.n, geometry.groups_per_row, spec.group_size),
        dtype=torch.int8,
        device=target,
    )
    scales = torch.empty(
        (geometry.n, geometry.groups_per_row), dtype=torch.float16, device=target
    )
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // geometry.k_pad)
    for start in range(0, geometry.n, rows_per_chunk):
        stop = min(start + rows_per_chunk, geometry.n)
        logical = weight[start:stop].detach().to(device=target, dtype=torch.float32)
        if not bool(torch.isfinite(logical).all()):
            raise ValueError("grouped quantization source contains NaN or infinity")
        if geometry.k_pad != geometry.k:
            padded = torch.zeros(
                (stop - start, geometry.k_pad), dtype=torch.float32, device=target
            )
            padded[:, : geometry.k].copy_(logical)
            logical = padded
        grouped = logical.reshape(stop - start, geometry.groups_per_row, spec.group_size)
        codes[start:stop], scales[start:stop] = _search_chunk(grouped, weights, spec)
    return QuantizedMatrix(codes=codes, scales=scales)


def quantize_and_encode(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    importance: torch.Tensor | None = None,
    device: str | torch.device | None = None,
) -> bytes:
    """Quantize a logical matrix and encode ``row-split-k128-v1`` bytes."""

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    quantized = quantize_matrix(weight, spec, importance=importance, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec, weight.shape)


__all__ = [
    "IMPORTANCE_FLOOR",
    "QuantizedMatrix",
    "SEARCH_FACTORS",
    "column_weights",
    "pick_device",
    "quantize_and_encode",
    "quantize_matrix",
]
