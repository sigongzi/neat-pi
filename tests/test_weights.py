"""weights.py 测试：checkpoint 名映射、直接加载和双向完备性。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from neat_pi.config import ModelConfig
from neat_pi.model.pi05 import Pi05
from neat_pi.model.weights import (convert_pi05_checkpoint,
                                   canonical_pi05_state_dict,
                                   is_local_pi05_checkpoint,
                                   internal_to_canonical_name,
                                   canonical_to_internal_name,
                                   load_pi05_weights, translate_name)


def _small_model_config() -> ModelConfig:
    """构造小而完整的 Pi05 配置，避免测试加载真实大模型。"""
    return ModelConfig(
        image_size=16, num_cameras=2, action_horizon=3, action_dim=7,
        vision_hidden_dim=16, vision_patch_size=4,
        vision_num_layers=1, vision_num_heads=4, vision_mlp_hidden_dim=32,
        vocab_size=64, vlm_hidden_dim=24, vlm_num_layers=1,
        vlm_num_heads=4, vlm_num_kv_heads=1, vlm_attn_head_dim=8,
        vlm_mlp_hidden_dim=48, expert_hidden_dim=16,
        expert_num_layers=1, expert_num_heads=4, expert_num_kv_heads=1,
        expert_attn_head_dim=8, expert_mlp_hidden_dim=32,
    )


def _current_checkpoint_name(local_name: str) -> str:
    """把 Pi05 的 state_dict 名反写成当前 pi05_base 的保存名。"""
    paligemma_model = "paligemma_with_expert.paligemma.model."
    gemma_expert_model = "paligemma_with_expert.gemma_expert.model."
    if local_name.startswith("vision_tower."):
        relative = local_name.removeprefix("vision_tower.")
        return paligemma_model + "vision_tower.vision_model." + relative
    if local_name == "language_model.lm_head.weight":
        return "paligemma_with_expert.paligemma.lm_head.weight"
    if local_name.startswith("language_model."):
        relative = local_name.removeprefix("language_model.")
        return paligemma_model + "language_model." + relative
    if local_name.startswith("multi_modal_projector."):
        return paligemma_model + local_name
    if local_name.startswith("action_expert.layers."):
        relative = local_name.removeprefix("action_expert.layers.")
        return gemma_expert_model + "layers." + relative
    if local_name.startswith("action_expert.norm."):
        relative = local_name.removeprefix("action_expert.norm.")
        return gemma_expert_model + "norm." + relative
    if local_name.startswith(
        ("action_expert.action_in_proj.", "action_expert.action_out_proj.",
         "action_expert.time_mlp_in.", "action_expert.time_mlp_out.")):
        return local_name.removeprefix("action_expert.")
    raise AssertionError(f"未覆盖的测试参数名: {local_name}")


def _write_small_checkpoint(model: Pi05, checkpoint_dir: Path) -> None:
    """按当前 checkpoint 标签生成小配置权重文件并附带显式跳过头。"""
    checkpoint: dict[str, torch.Tensor] = {
        _current_checkpoint_name(name): tensor.detach().clone()
        for name, tensor in canonical_pi05_state_dict(model).items()
    }
    checkpoint["paligemma_with_expert.gemma_expert.lm_head.weight"] = (
        torch.randn(4, 8))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint, str(checkpoint_dir / "model.safetensors"))


def test_translate_name_uses_current_checkpoint_labels() -> None:
    """当前保存格式不带 model 顶层包装时能映射到 Pi05 模块名。"""
    cases = {
        "paligemma_with_expert.paligemma.model.vision_tower."
        "vision_model.post_layernorm.weight": "vision_tower.post_layernorm.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp."
        "down_proj.weight": "language_model.layers.0.mlp.down_proj.weight",
        "paligemma_with_expert.paligemma.lm_head.weight":
            "language_model.lm_head.weight",
        "paligemma_with_expert.paligemma.model.multi_modal_projector."
        "linear.bias": "multi_modal_projector.linear.bias",
        "paligemma_with_expert.gemma_expert.model.layers.2.self_attn.q_proj."
        "weight": "action_expert.layers.2.self_attn.q_proj.weight",
        "paligemma_with_expert.gemma_expert.model.norm.dense.bias":
            "action_expert.norm.dense.bias",
        "time_mlp_in.weight": "action_expert.time_mlp_in.weight",
        "action_out_proj.bias": "action_expert.action_out_proj.bias",
    }
    for checkpoint_name, expected in cases.items():
        assert translate_name(checkpoint_name) == expected


def test_translate_name_skips_expert_language_head() -> None:
    """action-expert 的文本生成头是唯一显式跳过张量。"""
    name = "paligemma_with_expert.gemma_expert.lm_head.weight"
    assert translate_name(name) is None


def test_fused_layer_internal_and_canonical_names_round_trip() -> None:
    """MoT internal layer 名与 canonical checkpoint 名一一对应。"""
    internal_names = [
        "mot.layers.0.vlm.self_attn.q_proj.weight",
        "mot.layers.0.action.self_attn.q_proj.weight",
        "mot.layers.17.vlm.mlp.down_proj.weight",
        "mot.layers.17.action.mlp.down_proj.weight",
    ]
    canonical_names = [
        "language_model.layers.0.self_attn.q_proj.weight",
        "action_expert.layers.0.self_attn.q_proj.weight",
        "language_model.layers.17.mlp.down_proj.weight",
        "action_expert.layers.17.mlp.down_proj.weight",
    ]
    for internal, canonical in zip(internal_names, canonical_names, strict=True):
        assert internal_to_canonical_name(internal) == canonical
        assert canonical_to_internal_name(canonical) == internal


def test_small_pi05_canonical_state_dict_hides_fused_paths() -> None:
    """Pi05 对外导出的 state dict 仍保持 checkpoint canonical 命名。"""
    model = Pi05(_small_model_config())
    internal_names = set(model.state_dict())
    canonical_state = canonical_pi05_state_dict(model)

    assert internal_to_canonical_name("mot.layers.0.vlm.mlp.down_proj.weight") == \
        "language_model.layers.0.mlp.down_proj.weight"
    assert canonical_to_internal_name(
        "action_expert.layers.0.self_attn.q_proj.weight") == \
        "mot.layers.0.action.self_attn.q_proj.weight"
    assert all(not name.startswith("mot.") for name in canonical_state)
    assert canonical_state.keys() == {
        internal_to_canonical_name(name) for name in internal_names
    }


def test_from_pretrained_loads_current_labels(tmp_path: Path) -> None:
    """from_pretrained 能直接加载当前标签 checkpoint 且覆盖每个模型参数。"""
    cfg = _small_model_config()
    expected_model = Pi05(cfg)
    _write_small_checkpoint(expected_model, tmp_path)

    model = Pi05.from_pretrained(str(tmp_path), cfg=cfg,
                                 device=torch.device("cpu"),
                                 dtype=torch.bfloat16)
    load_result = load_pi05_weights(expected_model, str(tmp_path))
    assert load_result.loaded_count == 55
    assert load_result.skipped_count == 1
    assert all(parameter.dtype is torch.bfloat16
               for parameter in model.parameters())
    expected_model.to(dtype=torch.bfloat16)
    expected_state = canonical_pi05_state_dict(expected_model)
    actual_state = canonical_pi05_state_dict(model)
    for name, expected in expected_state.items():
        actual = actual_state[name]
        torch.testing.assert_close(actual, expected)


def test_converted_checkpoint_loads_without_translation(tmp_path: Path) -> None:
    """本地格式以 Pi05 名字存储，from_pretrained 直接按名加载。"""
    cfg = _small_model_config()
    expected_model = Pi05(cfg)
    source_dir = tmp_path / "source"
    local_dir = tmp_path / "local"
    _write_small_checkpoint(expected_model, source_dir)

    result = convert_pi05_checkpoint(str(source_dir), str(local_dir))
    assert result.loaded_count == 55
    assert result.skipped_count == 1
    assert result.shard_count == 1
    assert is_local_pi05_checkpoint(str(local_dir))
    assert list(local_dir.glob("*.safetensors")) == [
        local_dir / "model.safetensors"]

    model = Pi05.from_pretrained(str(local_dir), cfg=cfg,
                                 device=torch.device("cpu"))
    expected_state = canonical_pi05_state_dict(expected_model)
    actual_state = canonical_pi05_state_dict(model)
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected)


def test_converted_local_checkpoint_can_be_consolidated(tmp_path: Path) -> None:
    """已转换的多 shard 本地格式能合并为单文件并保持可直载。"""
    cfg = _small_model_config()
    expected_model = Pi05(cfg)
    source_dir = tmp_path / "source"
    sharded_dir = tmp_path / "sharded"
    single_dir = tmp_path / "single"
    _write_small_checkpoint(expected_model, source_dir)
    convert_pi05_checkpoint(str(source_dir), str(sharded_dir))

    result = convert_pi05_checkpoint(str(sharded_dir), str(single_dir))
    assert result.loaded_count == 55
    assert result.skipped_count == 0
    assert result.shard_count == 1
    assert is_local_pi05_checkpoint(str(single_dir))

    model = Pi05.from_pretrained(str(single_dir), cfg=cfg,
                                 device=torch.device("cpu"))
    expected_state = canonical_pi05_state_dict(expected_model)
    actual_state = canonical_pi05_state_dict(model)
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected)


def test_real_base_checkpoint_headers_all_translate() -> None:
    """扫描本地 pi05_base 的全部保存标签，确保没有未知名字。"""
    checkpoint_path = Path("checkpoints/pi05_base/model.safetensors")
    if not checkpoint_path.exists():
        pytest.skip("本地 checkpoints/pi05_base/model.safetensors 不存在")

    with safe_open(str(checkpoint_path), framework="pt") as checkpoint:
        names = list(checkpoint.keys())
        translated = [translate_name(name) for name in names]

    skipped = {name for name, local in zip(names, translated, strict=True)
               if local is None}
    assert len(names) == 812
    assert skipped == {
        "paligemma_with_expert.gemma_expert.lm_head.weight"}
