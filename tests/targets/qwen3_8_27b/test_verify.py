from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ResourceSpec,
    TensorSpec,
    plan_objects,
)
from tools.artifact.layouts import encode_row_split
from tools.artifact.numeric import get_format
from tools.convert.common.quantize import quantize_matrix
from tools.convert.qwen3_8_27b import inventory, verify


def _structural_artifact():
    specs = []
    resource_bytes = 1
    for item in inventory.OBJECT_SPECS:
        if isinstance(item, inventory.ResourceSpec):
            specs.append(ResourceSpec(item.name, item.encoding, resource_bytes))
            resource_bytes += 1
        else:
            specs.append(TensorSpec(item.name, item.shape, item.format, item.layout))
    objects = plan_objects(specs)
    payload_bytes = objects[-1].offset + objects[-1].bytes
    return SimpleNamespace(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        objects=objects,
        payload_offset=4096,
        file_bytes=4096 + payload_bytes,
    )


def _changed(artifact, **changes):
    values = vars(artifact).copy()
    values.update(changes)
    return SimpleNamespace(**values)


def test_complete_structure_and_logical_bindings_without_large_payload() -> None:
    artifact = _structural_artifact()
    summary = verify.validate_structure(artifact)
    assert (
        summary.objects,
        summary.tensors,
        summary.resources,
        summary.row_view_templates,
        summary.row_view_bindings,
        summary.alias_templates,
        summary.alias_bindings,
    ) == (1124, 1118, 6, 16, 390, 4, 51)
    assert summary.payload_bytes == artifact.objects[-1].offset + artifact.objects[-1].bytes


def test_structure_rejects_identity_signature_and_offset_drift() -> None:
    artifact = _structural_artifact()
    with pytest.raises(verify.VerificationError, match="identity"):
        verify.validate_structure(
            _changed(artifact, identity=ArtifactIdentity("wrong-model", "wrong"))
        )

    objects = list(artifact.objects)
    objects[6] = replace(objects[6], shape=(248319, 5120))
    with pytest.raises(verify.VerificationError, match="signature"):
        verify.validate_structure(_changed(artifact, objects=tuple(objects)))

    objects = list(artifact.objects)
    objects[7] = replace(objects[7], offset=objects[7].offset + 256)
    with pytest.raises(verify.VerificationError, match="offset"):
        verify.validate_structure(_changed(artifact, objects=tuple(objects)))


@pytest.mark.parametrize(
    ("format_name", "k"),
    (
        ("Q4G64_F16S", 65),
        ("Q5G64_F16S", 130),
        ("Q6G64_F16S", 129),
        ("W8G32_F16S", 33),
    ),
)
@pytest.mark.parametrize("weighted", (False, True))
def test_representative_quantized_row_group_verifier(
    format_name: str, k: int, weighted: bool
) -> None:
    generator = torch.Generator().manual_seed(k)
    source = torch.randn((5, k), generator=generator)
    importance = torch.rand(k, generator=generator) ** 4 if weighted else None
    quantized = quantize_matrix(source, format_name, importance=importance, device="cpu")
    payload = encode_row_split(
        quantized.codes,
        quantized.scales,
        format_name,
        source.shape,
    )
    rows = (0, 2, 4)

    def check(candidate_payload) -> int:
        return verify.verify_quantized_rows(
            candidate_payload,
            format_name,
            tuple(source.shape),
            rows,
            source[list(rows)],
            importance,
            "cpu",
        )

    assert check(payload) in (6, 9)

    damaged = bytearray(payload)
    damaged[0] ^= 1
    with pytest.raises(verify.VerificationError, match="codes differ"):
        check(damaged)

    # Maximum-absolute-value scales with their own round-to-nearest codes are
    # self-consistent but are not the weighted-error minimum.
    spec = get_format(format_name)
    groups = quantized.scales.shape[1]
    padded = torch.zeros((5, groups * spec.group_size))
    padded[:, :k] = source
    grouped = padded.reshape(5, groups, spec.group_size)
    absmax_scales = (grouped.abs().amax(-1).double() / spec.qmax).float().half()
    reciprocal = torch.where(
        absmax_scales > 0, (1.0 / absmax_scales.double()).float(), torch.zeros(())
    )
    absmax_codes = torch.clamp(
        torch.round(grouped * reciprocal.unsqueeze(-1)), spec.qmin, spec.qmax
    ).to(torch.int8)
    absmax_payload = encode_row_split(absmax_codes, absmax_scales, format_name, source.shape)
    with pytest.raises(verify.VerificationError, match="error-minimal"):
        check(absmax_payload)


def test_draft_id_domain_and_uniqueness() -> None:
    token_ids = torch.arange(131072, dtype=torch.int32)
    verify.validate_draft_token_ids(token_ids)

    duplicate = token_ids.clone()
    duplicate[-1] = duplicate[-2]
    with pytest.raises(verify.VerificationError, match="not unique"):
        verify.validate_draft_token_ids(duplicate)

    outside = token_ids.clone()
    outside[-1] = 248077
    with pytest.raises(verify.VerificationError, match="outside"):
        verify.validate_draft_token_ids(outside)
