"""FSDP full checkpoint 与 rank0 broadcast 权重加载测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from neat_pi.config import ModelConfig
from neat_pi.config import FSDPConfig
from neat_pi.device.backend import DeviceContext, DeviceType
from neat_pi.model.pi05 import Pi05
from neat_pi.model.weights import load_pi05_weights_distributed
from neat_pi.model.weights import canonical_pi05_state_dict
from neat_pi.training.checkpoint import (
    latest_checkpoint_path,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
def _small_pi05(dtype: torch.dtype = torch.float32) -> Pi05:
    """构造 small Pi05 并转换到指定 dtype。"""
    return Pi05(_small_model_config()).to(dtype=dtype)


def _small_model_config() -> ModelConfig:
    """构造完整但很小的 Pi05 配置。"""
    return ModelConfig(
        image_size=16,
        num_cameras=2,
        action_horizon=3,
        action_dim=7,
        vision_hidden_dim=16,
        vision_patch_size=4,
        vision_num_layers=1,
        vision_num_heads=4,
        vision_mlp_hidden_dim=32,
        vocab_size=64,
        vlm_hidden_dim=24,
        vlm_num_layers=1,
        vlm_num_heads=4,
        vlm_num_kv_heads=1,
        vlm_attn_head_dim=8,
        vlm_mlp_hidden_dim=48,
        expert_hidden_dim=16,
        expert_num_layers=1,
        expert_num_heads=4,
        expert_num_kv_heads=1,
        expert_attn_head_dim=8,
        expert_mlp_hidden_dim=32,
    )


def _write_canonical_checkpoint(model: Pi05, checkpoint_dir: Path) -> None:
    """把 small Pi05 写成 canonical 本地格式 checkpoint。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        canonical_pi05_state_dict(model),
        str(checkpoint_dir / "model.safetensors"),
        metadata={"format": "neat-pi.pi05-local-v1"},
    )


def _rank0_context() -> DeviceContext:
    """构造单进程 rank0 context。"""
    return DeviceContext(
        type=DeviceType.CPU,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )


def test_distributed_loader_falls_back_to_single_process(
        tmp_path: Path) -> None:
    """world_size=1 时广播 loader 直接回退到现有 canonical 加载器。"""
    expected = _small_pi05()
    _write_canonical_checkpoint(expected, tmp_path)
    model = Pi05(_small_model_config())

    result = load_pi05_weights_distributed(
        model,
        str(tmp_path),
        _rank0_context(),
    )

    assert result.loaded_count == 55
    assert result.skipped_count == 0
    expected_state = expected.state_dict()
    actual_state = model.state_dict()
    for name, tensor in expected_state.items():
        torch.testing.assert_close(actual_state[name], tensor)


def test_checkpoint_round_trip_uses_canonical_safetensors(
        tmp_path: Path) -> None:
    """full checkpoint 保存 canonical weights、optimizer、metadata。"""
    cfg_model = _small_model_config()
    model = _small_pi05()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = _rank0_context()
    output_dir = tmp_path / "train"
    fsdp_config = asdict(FSDPConfig())

    save_checkpoint(
        model,
        optimizer,
        step=7,
        output_dir=str(output_dir),
        ctx=ctx,
        fsdp_config=fsdp_config,
        metadata={"epoch": 1, "micro_step": 14},
    )
    checkpoint_path = latest_checkpoint_path(str(output_dir))
    assert checkpoint_path == output_dir / "step_0000007"
    assert (checkpoint_path / "model.safetensors").is_file()
    assert (checkpoint_path / "optimizer.pt").is_file()
    metadata = load_checkpoint_metadata(checkpoint_path)
    assert metadata["step"] == 7
    assert metadata["epoch"] == 1
    assert metadata["micro_step"] == 14

    with safe_open(
        checkpoint_path / "model.safetensors",
        framework="pt",
    ) as checkpoint:
        assert "language_model.layers.0.self_attn.q_proj.weight" in checkpoint.keys()
        assert not any(name.startswith("mot.") for name in checkpoint.keys())

    restored = _small_pi05()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert load_checkpoint(
        restored,
        restored_optimizer,
        checkpoint_path,
    ) == 7
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
