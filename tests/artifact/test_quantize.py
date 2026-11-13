from __future__ import annotations

import pytest
import torch

from tools.artifact.layouts import decode_row_split_codes, dequantize_row_split
from tools.artifact.numeric import get_format
from tools.convert.common.quantize import (
    IMPORTANCE_FLOOR,
    quantize_and_encode,
    quantize_matrix,
)


def _anchored(scale: float, extra: dict[int, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """A Q4 group whose weighted-error minimum is exactly ``scale``.

    Twenty heavily weighted columns hold ``7 * scale`` and ``-8 * scale``; the
    ``extra`` columns carry only the importance floor.
    """

    weight = torch.zeros((1, 64), dtype=torch.float32)
    importance = torch.zeros(64)
    for column in range(20):
        weight[0, column] = (7.0 if column % 2 == 0 else -8.0) * scale
        importance[column] = 1.0
    for column, value in extra.items():
        weight[0, column] = value
    return weight, importance


def _weighted_group_error(weight, quantized, importance, format_name):
    spec = get_format(format_name)
    n, k = weight.shape
    groups = quantized.scales.shape[1]
    padded = torch.zeros((n, groups * spec.group_size), dtype=torch.float64)
    padded[:, :k] = weight.double()
    columns = torch.zeros(groups * spec.group_size, dtype=torch.float64)
    columns[:k] = importance.double() + IMPORTANCE_FLOOR * importance.double().mean()
    residual = padded.reshape(n, groups, -1) - (
        quantized.scales.double().unsqueeze(-1) * quantized.codes.double()
    )
    return (columns.reshape(groups, -1) * residual * residual).sum(-1)


def test_quantization_uses_stored_fp16_scale_and_zero_padding() -> None:
    weight = torch.tensor(
        [[-7.0, -1.0, 0.0, 1.0, 7.0, 3.5, -3.5, 0.25] + [0.0] * 57],
        dtype=torch.bfloat16,
    )
    quantized = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert quantized.codes.shape == (1, 2, 64)
    assert quantized.scales.shape == (1, 2)
    assert torch.count_nonzero(quantized.codes[0, 1]) == 0
    assert quantized.scales[0, 1] == 0

    payload = quantize_and_encode(weight, "Q4G64_F16S", device="cpu")
    scales, codes = decode_row_split_codes(
        payload, "Q4G64_F16S", tuple(weight.shape)
    )
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(
        payload, "Q4G64_F16S", tuple(weight.shape), dtype=torch.float32
    )
    expected = (
        quantized.codes.float() * quantized.scales.float().unsqueeze(-1)
    ).reshape(1, 128)[:, :65]
    assert torch.equal(decoded, expected)


def test_quantization_uses_canonical_scale_rounding_on_cuda() -> None:
    # BF16 word 0x3636 divided by Q4 qmax is a known CUDA division boundary:
    # approximate device division rounds the FP16 scale to word 7, while the
    # ordered RNE oracle requires word 6. A constant group's best scale is the
    # maximum-absolute-value candidate.
    value = torch.tensor([0x3636], dtype=torch.int16).view(torch.bfloat16)
    weight = value.repeat(64).reshape(1, 64)
    cpu = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert int(cpu.scales.view(torch.int16)[0, 0]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "Q4G64_F16S", device="cuda")
        assert torch.equal(cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16))
        assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_quantization_uses_reciprocal_multiply_and_ties_to_even() -> None:
    # At scale word 0x41BD, x = 0x417C7BFF lies on the boundary where
    # round(x * round_f32(1 / scale)) = 6 but round(x / scale) = 5.
    scale = float(torch.tensor([0x41BD], dtype=torch.int16).view(torch.float16))
    boundary = float(torch.tensor([0x417C7BFF], dtype=torch.int32).view(torch.float32))
    weight, importance = _anchored(scale, {40: boundary})
    quantized = quantize_matrix(weight, "Q4G64_F16S", importance=importance, device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 0x41BD
    assert int(quantized.codes[0, 0, 40]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "Q4G64_F16S", importance=importance, device="cuda")
        assert torch.equal(cuda.scales.cpu(), quantized.scales)
        assert torch.equal(cuda.codes.cpu(), quantized.codes)

    ties = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
    tie_weight, tie_importance = _anchored(1.0, {40 + i: v for i, v in enumerate(ties)})
    tied = quantize_matrix(tie_weight, "Q4G64_F16S", importance=tie_importance, device="cpu")
    assert float(tied.scales[0, 0]) == 1.0
    assert tied.codes[0, 0, 40:46].tolist() == [0, 2, 2, 0, -2, -2]


@pytest.mark.parametrize(
    ("format_name", "minimum"),
    (("Q4G64_F16S", -8), ("Q5G64_F16S", -16), ("Q6G64_F16S", -32), ("W8G32_F16S", -127)),
)
def test_negative_anchor_reaches_the_minimum_code(format_name: str, minimum: int) -> None:
    spec = get_format(format_name)
    weight = torch.zeros((1, spec.group_size), dtype=torch.float32)
    weight[0, 0] = float(minimum)
    weight[0, 1:4] = torch.tensor([1.0, 2.0, -3.0])
    quantized = quantize_matrix(weight, format_name, device="cpu")
    assert float(quantized.scales[0, 0]) == 1.0
    assert quantized.codes[0, 0, :4].tolist() == [minimum, 1, 2, -3]
    if format_name == "W8G32_F16S":
        weight[0, 0] = -128.0
        clipped = quantize_matrix(weight, format_name, device="cpu")
        assert int(clipped.codes.min()) >= -127


@pytest.mark.parametrize("format_name", ("Q4G64_F16S", "Q5G64_F16S", "W8G32_F16S"))
def test_search_never_loses_to_the_maximum_absolute_scale(format_name: str) -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn((16, 200), generator=generator)
    weight[:, 5] *= 12.0
    importance = torch.rand(200, generator=generator) ** 6
    quantized = quantize_matrix(weight, format_name, importance=importance, device="cpu")

    spec = get_format(format_name)
    groups = quantized.scales.shape[1]
    padded = torch.zeros((16, groups * spec.group_size))
    padded[:, :200] = weight
    grouped = padded.reshape(16, groups, spec.group_size)
    absmax_scales = (grouped.abs().amax(-1).double() / spec.qmax).float().half()
    reciprocal = torch.where(
        absmax_scales > 0, (1.0 / absmax_scales.double()).float(), torch.zeros(())
    )
    absmax_codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    )
    absmax = type(quantized)(codes=absmax_codes, scales=absmax_scales)

    searched = _weighted_group_error(weight, quantized, importance, format_name)
    baseline = _weighted_group_error(weight, absmax, importance, format_name)
    assert bool((searched <= baseline).all())
    assert float(searched.sum()) < 0.9 * float(baseline.sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity needs a GPU")
def test_cpu_and_cuda_select_identical_words() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn((64, 320), generator=generator).to(torch.bfloat16)
    importance = torch.rand(320, generator=generator) ** 3
    cpu = quantize_matrix(weight, "Q4G64_F16S", importance=importance, device="cpu")
    cuda = quantize_matrix(weight, "Q4G64_F16S", importance=importance, device="cuda")
    assert torch.equal(cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16))
    assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "Q4G64_F16S", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1


@pytest.mark.parametrize(
    "value",
    (float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max),
)
def test_quantization_rejects_nonfinite_source_or_fp16_scale(value: float) -> None:
    weight = torch.full((1, 64), value, dtype=torch.float32)
    with pytest.raises(ValueError):
        quantize_matrix(weight, "Q4G64_F16S", device="cpu")


@pytest.mark.parametrize(
    "importance",
    (torch.ones(63), torch.full((64,), -1.0), torch.full((64,), float("nan"))),
)
def test_quantization_rejects_invalid_importance(importance: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="importance"):
        quantize_matrix(torch.ones((1, 64)), "Q4G64_F16S", importance=importance, device="cpu")
