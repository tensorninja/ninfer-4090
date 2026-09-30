"""Registered LoRA adapter inventory, conversion, and synthetic-fixture behavior.

The oracle here is the delta itself.  For every registered site the test
reconstructs ``delta = B @ A`` from the stored artifact and compares it against
a value derived independently of the converter, so a transposed factor, a
mis-selected row range, or a missing scale fold cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.artifact import Artifact
from tools.artifact.layouts import decode_direct
from tools.convert.qwen3_8_27b import convert_lora, lora_inventory as inventory
from tools.convert.qwen3_8_27b import make_synthetic_lora as synthetic


RANK = 8
ALPHA = 24.0
SCALE = ALPHA / RANK  # 3.0: not 1.0, so an omitted or doubled fold is visible


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------


def test_registered_site_table_is_closed_and_consistent() -> None:
    assert inventory.MODEL_ID == "qwen3.8-27b"
    assert inventory.WEIGHTS_ID == "lora-bf16"
    assert len(inventory.FULL_ATTENTION_LAYERS) == 16
    assert len(inventory.GDN_LAYERS) == 48

    for spec in inventory.SITE_SPECS:
        assert spec.out_features == spec.hf_heads * (spec.hf_row_end - spec.hf_row_begin)
        assert 0 <= spec.hf_row_begin < spec.hf_row_end <= spec.hf_head_rows
        assert spec.hf_out_features == spec.hf_heads * spec.hf_head_rows

    # The fused query/gate parent is the only row selection, and its two sites
    # partition the source rows exactly.
    selections = [spec for spec in inventory.SITE_SPECS if spec.is_row_selection]
    assert [spec.key for spec in selections] == ["attention/query", "attention/gate"]
    query, gate = selections
    assert query.hf_module == gate.hf_module == "self_attn.q_proj"
    assert query.a_object == gate.a_object
    assert (query.hf_row_begin, query.hf_row_end) == (0, 256)
    assert (gate.hf_row_begin, gate.hf_row_end) == (256, 512)
    assert query.out_features + gate.out_features == query.hf_out_features

    excluded = set(inventory.REJECTED_HF_MODULES)
    assert {"gate_proj", "up_proj", "in_proj_qkv", "in_proj_z"} <= excluded
    assert excluded.isdisjoint(inventory.SUPPORTED_HF_MODULES)


def test_full_inventory_geometry() -> None:
    specs = inventory.build_tensor_specs(16, inventory.ALL_SITE_KEYS)
    assert len(specs) == 368
    assert inventory.parameter_count(16, inventory.ALL_SITE_KEYS) == 42_205_184

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert all(spec.format == "BF16" for spec in specs)
    assert all(spec.layout == "contiguous-le-v1" for spec in specs)
    assert all(len(spec.shape) == 2 for spec in specs)

    # A shared lora_A is stored once, at its first use inside the layer.
    layer3 = [name for name in names if name.startswith("text/layers/3/")]
    assert layer3[0] == "text/layers/3/attention/query_gate/lora_a"
    assert layer3[1] == "text/layers/3/attention/query/lora_b"
    assert layer3[2] == "text/layers/3/attention/gate/lora_b"
    assert sum(1 for name in layer3 if name.endswith("query_gate/lora_a")) == 1


def test_partial_site_sets_are_valid() -> None:
    attention_only = frozenset(
        key for key in inventory.SITE_KEYS if key.startswith("attention/")
    )
    specs = inventory.build_tensor_specs(16, attention_only)
    assert len(specs) == 16 * 9
    assert all("/mlp/" not in spec.name and "/gdn/" not in spec.name for spec in specs)

    with pytest.raises(ValueError, match="at least one registered site"):
        inventory.build_tensor_specs(16, frozenset())
    with pytest.raises(ValueError, match="not registered"):
        inventory.build_tensor_specs(12, inventory.ALL_SITE_KEYS)


# --------------------------------------------------------------------------
# conversion round trip
# --------------------------------------------------------------------------


def _convert(tmp_path: Path, kind: str, tag: str = "", **kwargs) -> tuple[Artifact, dict, dict]:
    stem = f"{kind}_{kwargs.get('variant', 'a')}{tag}"
    source = tmp_path / f"src_{stem}"
    manifest = synthetic.write_adapter(
        out_dir=source,
        kind=kind,
        rank=kwargs.get("rank", RANK),
        alpha=kwargs.get("alpha", ALPHA),
        variant=kwargs.get("variant", "a"),
        seed=kwargs.get("seed", 11),
        site_keys=kwargs.get("site_keys", inventory.ALL_SITE_KEYS),
        sigma=kwargs.get("sigma", 0.02),
        use_rslora=kwargs.get("use_rslora", False),
    )
    out = tmp_path / f"{stem}.lora.ninfer"
    report = convert_lora.convert(source, out)
    return Artifact.open(out), manifest, report


def _decoded(artifact: Artifact, name: str) -> torch.Tensor:
    obj = artifact.find(name)
    return decode_direct(artifact.payload(obj), obj.format, tuple(obj.shape))


def _site_delta(artifact: Artifact, layer: int, site: inventory.LoraSiteSpec) -> torch.Tensor:
    a = _decoded(artifact, inventory.a_object_name(layer, site)).to(torch.float32)
    b = _decoded(artifact, inventory.b_object_name(layer, site)).to(torch.float32)
    assert a.shape == (RANK, site.in_features)
    assert b.shape == (site.out_features, RANK)
    return b @ a


def test_zero_adapter_produces_an_exactly_zero_delta(tmp_path: Path) -> None:
    artifact, _manifest, report = _convert(tmp_path, "zero")
    assert artifact.identity.model_id == "qwen3.8-27b"
    assert artifact.identity.weights_id == "lora-bf16"
    assert all(obj.kind == "tensor" for obj in artifact.objects)

    for site in inventory.SITE_SPECS:
        for layer in (site.layers()[0], site.layers()[-1]):
            a = _decoded(artifact, inventory.a_object_name(layer, site))
            b = _decoded(artifact, inventory.b_object_name(layer, site))
            # B is exactly zero; A is deliberately not, so the A-projection is
            # still exercised at run time.
            assert torch.count_nonzero(b) == 0
            assert torch.count_nonzero(a) > 0
    assert report["adapter"]["folded_scale"] == SCALE


def test_canary_delta_is_exactly_one_scaled_entry_per_site(tmp_path: Path) -> None:
    artifact, manifest, _report = _convert(tmp_path, "canary")
    expected = manifest["expected_delta_entries"]
    assert len(expected) == 16 * 5 + 48 + 64

    by_site: dict[tuple[int, str], dict] = {
        (entry["layer"], entry["site"]): entry for entry in expected
    }
    for site in inventory.SITE_SPECS:
        for layer in site.layers():
            entry = by_site[(layer, site.key)]
            delta = _site_delta(artifact, layer, site)
            nonzero = torch.nonzero(delta)
            assert nonzero.shape[0] == 1, (
                f"layer {layer} site {site.key}: {nonzero.shape[0]} nonzero entries"
            )
            row, column = int(nonzero[0, 0]), int(nonzero[0, 1])
            assert (row, column) == (entry["row"], entry["column"])
            assert delta[row, column] == pytest.approx(entry["value"] * SCALE, rel=1e-2)


def test_query_gate_deinterleave_maps_every_head_boundary(tmp_path: Path) -> None:
    """A canary at source row p must land in the site and row the parent implies."""

    query = inventory.SITE_BY_KEY["attention/query"]
    gate = inventory.SITE_BY_KEY["attention/gate"]
    layer = inventory.FULL_ATTENTION_LAYERS[0]

    source = tmp_path / "deinterleave"
    state: dict[str, torch.Tensor] = {}
    probes = [0, 255, 256, 511, 512, 767, 6143, 6144, 12287]

    for probe in probes:
        head, within = divmod(probe, query.hf_head_rows)
        if within < query.hf_row_end:
            site, expected_row = query, head * 256 + within
        else:
            site, expected_row = gate, head * 256 + (within - gate.hf_row_begin)

        a = torch.zeros(RANK, query.in_features, dtype=torch.float32)
        a[0, probe % query.in_features] = 1.0
        b = torch.zeros(query.hf_out_features, RANK, dtype=torch.float32)
        b[probe, 0] = 1.0
        state.clear()
        stem = f"base_model.model.model.language_model.layers.{layer}.self_attn.q_proj"
        state[f"{stem}.lora_A.weight"] = a
        state[f"{stem}.lora_B.weight"] = b

        entries = convert_lora.parse_adapter_state(dict(state))
        config = convert_lora.AdapterConfig(
            rank=RANK, alpha=ALPHA, scale=SCALE, use_rslora=False,
            target_modules=("q_proj",), base_model_name=None,
        )
        _a_q, b_q = convert_lora.build_site_tensors(entries, layer, query, config)
        _a_g, b_g = convert_lora.build_site_tensors(entries, layer, gate, config)

        selected = b_q if site is query else b_g
        other = b_g if site is query else b_q
        nonzero = torch.nonzero(selected.to(torch.float32))
        assert nonzero.shape[0] == 1, f"probe {probe} produced {nonzero.shape[0]} rows"
        assert int(nonzero[0, 0]) == expected_row, (
            f"probe {probe}: landed in row {int(nonzero[0, 0])}, expected {expected_row} "
            f"of {site.key}"
        )
        assert torch.count_nonzero(other) == 0, f"probe {probe} leaked into the peer site"
    assert source == source  # tmp_path is unused on this path by design


def test_scale_fold_is_applied_exactly_once(tmp_path: Path) -> None:
    plain, manifest, report_plain = _convert(tmp_path, "canary", tag="_plain")
    assert report_plain["adapter"]["folded_scale"] == ALPHA / RANK

    rs, _m2, report_rs = _convert(tmp_path, "canary", tag="_rs", use_rslora=True)
    assert report_rs["adapter"]["use_rslora"] is True
    assert report_rs["adapter"]["folded_scale"] == pytest.approx(ALPHA / (RANK**0.5))

    site = inventory.SITE_BY_KEY["attention/output"]
    layer = inventory.FULL_ATTENTION_LAYERS[0]
    entry = next(
        e for e in manifest["expected_delta_entries"]
        if e["layer"] == layer and e["site"] == site.key
    )

    # A canary carries constant 1.0, so the stored delta is the fold itself.
    plain_delta = _site_delta(plain, layer, site)[entry["row"], entry["column"]].item()
    rs_delta = _site_delta(rs, layer, site)[entry["row"], entry["column"]].item()
    assert plain_delta == pytest.approx(entry["value"] * ALPHA / RANK, rel=1e-2)
    assert rs_delta == pytest.approx(entry["value"] * ALPHA / (RANK**0.5), rel=1e-2)
    assert rs_delta / plain_delta == pytest.approx(RANK**0.5, rel=1e-2)


def test_distinct_variants_disagree_everywhere(tmp_path: Path) -> None:
    a_art, a_manifest, _ = _convert(tmp_path, "distinct", variant="a")
    b_art, b_manifest, _ = _convert(tmp_path, "distinct", variant="b")

    a_entries = {(e["layer"], e["site"]): e for e in a_manifest["expected_delta_entries"]}
    b_entries = {(e["layer"], e["site"]): e for e in b_manifest["expected_delta_entries"]}
    assert a_entries.keys() == b_entries.keys()

    differing = 0
    for key, a_entry in a_entries.items():
        b_entry = b_entries[key]
        if (a_entry["row"], a_entry["column"], a_entry["value"]) != (
            b_entry["row"], b_entry["column"], b_entry["value"]
        ):
            differing += 1
    assert differing == len(a_entries), "every site must differ between routing variants"

    site = inventory.SITE_BY_KEY["mlp/down"]
    assert not torch.equal(_site_delta(a_art, 0, site), _site_delta(b_art, 0, site))


def test_random_adapter_is_dense_and_bounded(tmp_path: Path) -> None:
    artifact, _manifest, _report = _convert(tmp_path, "random", sigma=0.02)
    site = inventory.SITE_BY_KEY["attention/key"]
    delta = _site_delta(artifact, inventory.FULL_ATTENTION_LAYERS[0], site)
    density = torch.count_nonzero(delta) / delta.numel()
    assert density > 0.99
    assert torch.isfinite(delta).all()


def test_report_records_the_adapter_contract(tmp_path: Path) -> None:
    out = tmp_path / "reported.lora.ninfer"
    source = tmp_path / "reported_src"
    synthetic.write_adapter(
        out_dir=source, kind="canary", rank=RANK, alpha=ALPHA, variant="a",
        seed=5, site_keys=inventory.ALL_SITE_KEYS, sigma=0.02, use_rslora=False,
    )
    report = convert_lora.convert(source, out)
    sidecar = json.loads(out.with_suffix(out.suffix + ".conversion.json").read_text())
    assert sidecar == report
    assert report["identity"] == {"model_id": "qwen3.8-27b", "weights_id": "lora-bf16"}
    assert report["objects"]["formats"] == {"BF16": 368}
    assert report["objects"]["layouts"] == {"contiguous-le-v1": 368}
    assert report["objects"]["resources"] == 0
    assert sorted(report["adapter"]["sites"]) == sorted(inventory.SITE_KEYS)


# --------------------------------------------------------------------------
# decision adapters
# --------------------------------------------------------------------------


def _decision_source(tmp_path: Path, temperature: float = 1.38) -> Path:
    source = tmp_path / "decision_src"
    synthetic.write_adapter(
        out_dir=source, kind="random", rank=RANK, alpha=ALPHA, variant="a",
        seed=9, site_keys=inventory.ALL_SITE_KEYS, sigma=0.02, use_rslora=False,
    )
    synthetic.write_decision_head(source, seed=9, sigma=0.01, temperature=temperature)
    return source


def test_decision_head_round_trips_as_bf16_with_its_metadata(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    source = _decision_source(tmp_path)
    head = convert_lora.load_decision_head(
        source / "decision_head.safetensors", description="probe", release_date="2026-09-30"
    )
    out = tmp_path / "decider.lora.ninfer"
    report = convert_lora.convert(source, out, decision_head=head)
    artifact = Artifact.open(out)

    # Head objects follow every factor and the metadata resource is last, so a generative
    # prefix of the directory is exactly the generative adapter.
    names = [obj.name for obj in artifact.objects]
    assert names[-5:] == [name for name, _ in inventory.DECISION_HEAD_TENSORS] + [
        inventory.DECISION_METADATA_OBJECT
    ]
    assert report["kind"] == "decision"
    assert report["objects"]["resources"] == 1

    # BF16 round-to-nearest-even of the trainer's FP32 values, bit for bit.
    stored = load_file(str(source / "decision_head.safetensors"))
    for key, name in (("query.weight", "decision/head/query/weight"),
                      ("query.bias", "decision/head/query/bias"),
                      ("key.weight", "decision/head/key/weight"),
                      ("key.bias", "decision/head/key/bias")):
        expected = stored[key].to(torch.bfloat16)
        assert torch.equal(_decoded(artifact, name).view(torch.int16),
                           expected.view(torch.int16))

    metadata = json.loads(bytes(artifact.payload(artifact.find(inventory.DECISION_METADATA_OBJECT))))
    assert set(metadata) == inventory.DECISION_METADATA_KEYS
    assert metadata["delimiters"] == inventory.DECISION_DELIMITERS
    assert metadata["description"] == "probe"
    assert metadata["release_date"] == "2026-09-30"
    # The temperature is stored as the FP32 value the engine divides by, exactly.
    assert metadata["temperature"] == float(torch.tensor(1.38, dtype=torch.float32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pointer_dim", "128", "pointer_dim"),
        ("logit_scale", "0.125", "logit_scale"),
        ("readout", "last_token", "readout"),
        ("escape", "none", "escape"),
        ("delimiters", json.dumps({**inventory.DECISION_DELIMITERS, "decide": "<|im_end|>"}),
         "delimiters"),
        ("delimiter_ids", json.dumps({**inventory.DECISION_DELIMITER_IDS, "state": 1}),
         "delimiter ids"),
        ("temperature", "0", "temperature"),
        ("release_date", "2026-02-30", "release date"),
    ],
)
def test_decision_head_contract_violations_are_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = _decision_source(tmp_path)
    path = source / "decision_head.safetensors"
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata())
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    metadata[field] = value
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(convert_lora.LoraConversionError, match=message):
        convert_lora.load_decision_head(path)


def test_decision_head_with_a_foreign_tensor_is_rejected(tmp_path: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = _decision_source(tmp_path)
    path = source / "decision_head.safetensors"
    with safe_open(str(path), framework="pt") as handle:
        metadata = dict(handle.metadata())
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    tensors["key.weight"] = tensors["key.weight"][:, :4096].contiguous()
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(convert_lora.LoraConversionError, match="shape"):
        convert_lora.load_decision_head(path)


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


def _write_config(tmp_path: Path, **overrides) -> Path:
    directory = tmp_path / f"cfg{len(list(tmp_path.iterdir()))}"
    directory.mkdir(parents=True, exist_ok=True)
    config = synthetic.build_adapter_config(RANK, ALPHA, inventory.ALL_SITE_KEYS, False)
    config.update(overrides)
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return directory


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"peft_type": "IA3"}, "only 'LORA' converts"),
        ({"use_dora": True}, "use_dora is set"),
        ({"lora_bias": True}, "lora_bias is set"),
        ({"bias": "all"}, "train with bias='none'"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save is set"),
        ({"target_parameters": ["experts.gate_up_proj"]}, "target_parameters is set"),
        ({"rank_pattern": {"q_proj": 32}}, "rank_pattern is non-empty"),
        ({"alpha_pattern": {"q_proj": 64}}, "alpha_pattern is non-empty"),
        ({"layers_to_transform": [0, 1]}, "layers_to_transform is set"),
        ({"r": 12}, "not registered"),
        ({"target_modules": "re:.*proj"}, "target_modules is a regex"),
        ({"target_modules": []}, "target_modules is empty"),
    ],
)
def test_rejected_adapter_configurations(tmp_path: Path, overrides: dict, message: str) -> None:
    directory = _write_config(tmp_path, **overrides)
    with pytest.raises(convert_lora.LoraConversionError, match=message):
        convert_lora.load_adapter_config(directory)


@pytest.mark.parametrize(
    ("module", "fragment"),
    [
        ("gate_proj", "silu(gate) * up"),
        ("up_proj", "silu(gate) * up"),
        ("in_proj_qkv", "fused causal convolution"),
        ("in_proj_z", "fused causal convolution"),
        ("lm_head", "logits destination"),
    ],
)
def test_excluded_modules_are_rejected_by_name(tmp_path: Path, module: str, fragment: str) -> None:
    directory = _write_config(tmp_path, target_modules=["q_proj", module])
    with pytest.raises(convert_lora.LoraConversionError) as error:
        convert_lora.load_adapter_config(directory)
    assert module in str(error.value)
    assert fragment in str(error.value)


def test_partial_layer_coverage_is_rejected(tmp_path: Path) -> None:
    site_keys = frozenset({"mlp/down"})
    source = tmp_path / "partial"
    synthetic.write_adapter(
        out_dir=source, kind="canary", rank=RANK, alpha=ALPHA, variant="a",
        seed=1, site_keys=site_keys, sigma=0.02, use_rslora=False,
    )
    from safetensors.torch import load_file, save_file

    weights = source / "adapter_model.safetensors"
    state = load_file(str(weights))
    dropped = [key for key in state if ".layers.7." in key]
    assert dropped
    for key in dropped:
        del state[key]
    save_file(state, str(weights))

    with pytest.raises(convert_lora.LoraConversionError, match="covers 63 layers"):
        convert_lora.convert(source, tmp_path / "partial.lora.ninfer")


def test_unmatched_adapter_tensors_are_rejected() -> None:
    state = {"base_model.model.lm_head.lora_A.weight": torch.zeros(RANK, 8)}
    with pytest.raises(convert_lora.LoraConversionError, match="outside the decoder layer stack"):
        convert_lora.parse_adapter_state(state)


def test_missing_pair_side_is_rejected() -> None:
    stem = "base_model.model.model.layers.3.self_attn.k_proj"
    state = {f"{stem}.lora_A.weight": torch.zeros(RANK, 5120)}
    with pytest.raises(convert_lora.LoraConversionError, match="missing lora_B"):
        convert_lora.parse_adapter_state(state)


def test_text_only_key_prefix_is_accepted() -> None:
    """`text_only=True` collapses the VLM wrapper; both module paths must parse."""

    layer = inventory.FULL_ATTENTION_LAYERS[0]
    for stem in (
        f"base_model.model.model.language_model.layers.{layer}.self_attn.k_proj",
        f"base_model.model.model.layers.{layer}.self_attn.k_proj",
    ):
        state = {
            f"{stem}.lora_A.weight": torch.zeros(RANK, 5120),
            f"{stem}.lora_B.weight": torch.zeros(1024, RANK),
        }
        entries = convert_lora.parse_adapter_state(state)
        assert list(entries) == [(layer, "self_attn.k_proj")]
