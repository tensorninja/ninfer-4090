from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from tools.reference.qwen3_8.common.frontend import Frontend, prepare_decision_image
from tools.reference.qwen3_8_27b.decision import (
    DECIDE, IMAGE_PAD, OPTION_CLOSE, QUESTION, STATE, VISION_END, VISION_START,
    openai_options, prepare_openai,
)


MODEL = Path(__file__).resolve().parents[3] / "models" / "Qwen3.8-27B"
CONFIG_ONLY_TOKENS = {
    "<|audio_start|>": 248070,
    "<|audio_end|>": 248071,
    "<tts_pad>": 248072,
    "<tts_text_bos>": 248073,
    "<tts_text_eod>": 248074,
    "<tts_text_bos_single>": 248075,
    "<|audio_pad|>": 248076,
}


class _OfficialSourceBinding:
    frontend = SimpleNamespace(
        tokenizer_json=MODEL / "tokenizer.json",
        tokenizer_config_json=MODEL / "tokenizer_config.json",
        chat_template_jinja=MODEL / "chat_template.jinja",
        generation_config_json=MODEL / "generation_config.json",
        preprocessor_config_json=MODEL / "preprocessor_config.json",
        video_preprocessor_config_json=MODEL / "video_preprocessor_config.json",
    )

    @staticmethod
    def resource_bytes(resource: Path) -> bytes:
        return resource.read_bytes()


def test_reference_consumes_the_raw_official_resource_pair():
    frontend = Frontend(_OfficialSourceBinding())

    assert len(frontend.tokenizer) == 248077
    assert {
        token: frontend.tokenizer.convert_tokens_to_ids(token)
        for token in CONFIG_ONLY_TOKENS
    } == CONFIG_ONLY_TOKENS
    assert frontend.processor.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
    ) == (
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def encoded_png(pixels, *, orientation=None):
    stream = BytesIO()
    image = Image.fromarray(pixels)
    options = {}
    if orientation is not None:
        exif = image.getexif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(stream, format="PNG", **options)
    return stream.getvalue()


def test_decision_original_keeps_oriented_pixels_and_only_edge_pads():
    processor = Frontend(_OfficialSourceBinding()).processor.image_processor
    pixels = np.arange(17 * 35 * 3, dtype=np.uint8).reshape(17, 35, 3)
    encoded = encoded_png(pixels, orientation=6)
    output = prepare_decision_image(processor, encoded, "original")
    assert output["image_grid_thw"].tolist() == [[1, 4, 2]]
    oriented = np.rot90(pixels, k=-1)
    expected = []
    for block_y in range(2):
        for patch_y in range(2):
            for patch_x in range(2):
                row = []
                for channel in range(3):
                    for temporal in range(2):
                        for y in range(16):
                            for x in range(16):
                                sy = min((block_y * 2 + patch_y) * 16 + y, 34)
                                sx = min(patch_x * 16 + x, 16)
                                row.append(float(oriented[sy, sx, channel]) / 127.5 - 1)
                expected.append(row)
    torch.testing.assert_close(output["pixel_values"], torch.tensor(expected), rtol=0, atol=2e-7)
    assert torch.equal(torch.round((output["pixel_values"] + 1) * 127.5).to(torch.uint8),
                       torch.round((torch.tensor(expected) + 1) * 127.5).to(torch.uint8))


def test_decision_detail_uses_pixel_area_and_native_high_profile():
    processor = Frontend(_OfficialSourceBinding()).processor.image_processor
    encoded = encoded_png(np.full((256, 2048, 3), 127, dtype=np.uint8))
    low = prepare_decision_image(processor, encoded, "low")
    assert low["image_grid_thw"].tolist() == [[1, 10, 90]]
    high = prepare_decision_image(processor, encoded, "high")
    assert high["image_grid_thw"].tolist() == [[1, 16, 128]]
    for detail in (None, "auto"):
        equivalent = prepare_decision_image(processor, encoded, detail)
        assert torch.equal(equivalent["image_grid_thw"], high["image_grid_thw"])
        assert torch.equal(equivalent["pixel_values"], high["pixel_values"])
    tiny = encoded_png(np.zeros((17, 35, 3), dtype=np.uint8))
    native = prepare_decision_image(processor, tiny)
    assert int(native["image_grid_thw"].prod()) * 16 * 16 >= 65536


def test_native_image_decision_tokens_positions_and_text_only_branches():
    frontend = Frontend(_OfficialSourceBinding())
    encoded = encoded_png(np.zeros((33, 33, 3), dtype=np.uint8))
    image = {"type": "input_image", "detail": "original",
             "image_url": "data:image/png;base64," + base64.b64encode(encoded).decode()}
    request = {"model": "actual-adapter", "input": [{"role": "user", "content": [
        image, image, {"type": "input_text", "text": "<|image_"},
        {"type": "input_text", "text": "pad|>"}]}], "questions": [
            {"type": "predicate", "instructions": "<|fim_suffix|>"},
            {"type": "choice", "instructions": "pick", "choices": [
                {"value": True}, {"value": "true", "description": "文字"}]}]}
    prepared = prepare_openai(request, frontend)
    batch = prepared.state_batch
    assert batch is not None
    assert prepared.tokens[:13] == [STATE, VISION_START, *([IMAGE_PAD] * 4), VISION_END,
                                   VISION_START, *([IMAGE_PAD] * 4), VISION_END]
    assert batch.image_grid_thw.tolist() == [[1, 4, 4], [1, 4, 4]]
    assert batch.position_ids[:, :13].tolist() == [
        [0, 1, 2, 2, 2, 2, 4, 5, 6, 6, 6, 6, 8],
        [0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8],
        [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 6, 7, 8],
    ]
    assert batch.rope_delta == -4
    assert frontend.tokenizer.decode(prepared.tokens[13:prepared.state_tokens]) == "<¦image_pad¦>"
    for branch in prepared.branches:
        tokens = prepared.branch_tokens(branch)
        assert tokens[0] == QUESTION and tokens[-1] == DECIDE
        assert tokens.count(DECIDE) == 1
        assert IMAGE_PAD not in tokens
        assert [tokens[offset] for offset in branch.option_readouts] == [OPTION_CLOSE] * 2
    assert openai_options(request["questions"][1]) == ["true", '"true": 文字']


def test_native_text_decision_preserves_combined_v1_layout():
    from tools.reference.qwen3_8_27b.decision import parse_prepared

    frontend = Frontend(_OfficialSourceBinding())
    question = {"type": "predicate", "instructions": "yes?"}
    direct = prepare_openai({"input": "ab\n\ncd", "questions": [question]}, frontend)
    parts = prepare_openai({"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "a"},
                                     {"type": "input_text", "text": "b"}]},
        {"role": "user", "content": "cd"}], "questions": [question]}, frontend)
    assert direct.state_batch is None and direct == parts
    old = parse_prepared({"format": "ninfer-decision-prepared", "format_version": 1,
                          "tokens": direct.tokens, "state_tokens": direct.state_tokens,
                          "branches": [{"begin": branch.begin, "length": branch.length,
                                        "option_readouts": list(branch.option_readouts)}
                                       for branch in direct.branches]})
    assert old == direct
