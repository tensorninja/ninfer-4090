from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from tools.reference.qwen3_8.common.multimodal import (
    MultimodalBatch,
    build_mrope_positions,
)


def test_mrope_and_chunked_visual_embedding_alignment():
    # Two image items and a two-frame video.  Timestamp text separates the
    # video frames exactly as the library processor emits them.
    types = torch.tensor(
        [0, 1, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 0], dtype=torch.long
    )
    image_grid = torch.tensor([[1, 4, 4], [1, 2, 2]], dtype=torch.long)
    video_grid = torch.tensor([[2, 2, 4]], dtype=torch.long)
    positions, delta = build_mrope_positions(types, image_grid, video_grid)

    assert positions.tolist() == [
        [0, 1, 1, 1, 1, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 1, 2, 2, 3, 4, 4, 6, 7, 7, 9, 10, 11],
        [0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    assert delta == -2

    batch = MultimodalBatch(
        input_ids=torch.arange(types.numel()),
        mm_token_type_ids=types,
        position_ids=positions,
        rope_delta=delta,
        pixel_values=None,
        image_grid_thw=image_grid,
        pixel_values_videos=None,
        video_grid_thw=video_grid,
    )
    image = torch.tensor([[10.0], [11.0], [12.0], [13.0], [20.0]])
    video = torch.tensor([[30.0], [31.0], [40.0], [41.0]])
    first = torch.arange(7, dtype=torch.float32).unsqueeze(1) + 100
    second = torch.arange(7, 14, dtype=torch.float32).unsqueeze(1) + 100
    batch.scatter_visual_embeddings_(first, image, video, offset=0)
    batch.scatter_visual_embeddings_(second, image, video, offset=7)

    assert torch.cat((first, second)).flatten().tolist() == [
        100.0,
        10.0,
        11.0,
        12.0,
        13.0,
        105.0,
        30.0,
        31.0,
        108.0,
        40.0,
        41.0,
        111.0,
        20.0,
        113.0,
    ]


def test_image_decision_prefill_snapshots_mrope_without_sampling_or_mtp():
    from tools.reference.qwen3_8_27b import model as model_module
    from tools.reference.qwen3_8_27b.decision import Branch, PreparedDecision, readouts
    from tools.reference.qwen3_8_27b.state import ModelState

    ids = [1, 2, 3, 3, 3, 3, 4]
    types = torch.tensor([0, 0, 1, 1, 1, 1, 0])
    grid = torch.tensor([[1, 4, 4]])
    positions, delta = build_mrope_positions(types, grid, None)
    batch = MultimodalBatch(torch.tensor(ids), types, positions, delta, None, grid, None, None)
    branch_a = [5, 6, 7, 8, 9]
    branch_b = [10, 11, 12, 13]
    prepared = PreparedDecision(ids + branch_a + branch_b, len(ids), (
        Branch(7, 5, (1, 3)), Branch(12, 4, (1, 2))), batch)
    model = object.__new__(model_module.RefModel)
    model.device = torch.device("cpu")
    model.prefill_chunk = 3
    model.last_hidden = model.last_mtp_hidden = model.last_draft = None
    model.mtp_enabled = True
    model.logits_last = Mock(side_effect=AssertionError("decision ran output head"))
    model.mtp_forward = Mock(side_effect=AssertionError("decision ran MTP"))
    model.final_hidden = lambda x: x.to(torch.bfloat16)
    model.embed = lambda values: torch.tensor([[value, -value] for value in values],
                                               dtype=torch.bfloat16)
    image = torch.tensor([[20, -20], [21, -21], [22, -22], [23, -23]], dtype=torch.bfloat16)
    model.encode_vision = Mock(return_value=SimpleNamespace(image_embeddings=image,
                                                           video_embeddings=None))

    def prepare(capacity):
        state = object.__new__(ModelState)
        state.capacity = capacity
        state.position = state.rope_delta = 0
        state.mrope = False
        state.kv = SimpleNamespace(length=0)
        state.mtp_kv = SimpleNamespace(length=0)
        state.conv = [torch.zeros(1)]
        state.ssm = [torch.zeros(1)]
        model.state = state
        model.weights = object()

    model.prepare = prepare
    calls = []

    def recurrent_text(model, x, positions, start, **kwargs):
        calls.append((start, positions.clone(), x.clone()))
        result = x.clone()
        for index, value in enumerate(x):
            model.state.ssm[0] += value[0].float()
            result[index] = value + model.state.ssm[0]
        return result

    with patch.object(model_module, "CFG", SimpleNamespace(hidden=2)), patch.object(
        model_module, "run_text", recurrent_text
    ):
        rows = readouts(model, prepared)
    assert model.encode_vision.call_count == 1
    model.logits_last.assert_not_called()
    model.mtp_forward.assert_not_called()
    assert model.state.mtp_kv.length == 0
    assert torch.equal(torch.cat([call[1] for call in calls[:3]], dim=1), positions)
    assert torch.equal(torch.cat([call[2] for call in calls[:3]])[2:6], image)
    assert [call[0] for call in calls] == [0, 3, 6, 7, 10, 7, 10]
    for call, expected in zip(calls[3:], ([5, 6, 7], [8, 9], [5, 6, 7], [8]), strict=True):
        assert call[1].tolist() == [expected] * 3
    base = 1 + 2 + 20 + 21 + 22 + 23 + 4
    for branch, offsets, actual in zip((branch_a, branch_b), ((1, 3, 4), (1, 2, 3)), rows,
                                       strict=True):
        expected = []
        total = base
        for index, value in enumerate(branch):
            total += value
            if index in offsets:
                expected.append([value + total, -value + total])
        assert torch.equal(actual, torch.tensor(expected, dtype=torch.bfloat16))


def test_text_hidden_prefill_still_uses_ordinal_positions():
    from tools.reference.qwen3_8_27b import model as model_module

    model = object.__new__(model_module.RefModel)
    model.weights = object()
    model.device = torch.device("cpu")
    model.state = SimpleNamespace(position=7, capacity=20, mrope=False, rope_delta=0,
                                  kv=SimpleNamespace(length=7))
    model.prefill_chunk = 2
    model.embed = lambda values: torch.tensor(values, dtype=torch.bfloat16).reshape(-1, 1)
    model.final_hidden = lambda x: x
    calls = []

    def identity(model, x, positions, start, **kwargs):
        calls.append(positions.tolist())
        return x

    with patch.object(model_module, "CFG", SimpleNamespace(hidden=1)), patch.object(
        model_module, "run_text", identity
    ):
        hidden = model.prefill_hidden([10, 11, 12], (2, 0))
    assert calls == [[7, 8], [9]]
    assert hidden.flatten().tolist() == [12, 10]
    assert model.state.position == model.state.kv.length == 10
