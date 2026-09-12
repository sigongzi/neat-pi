"""EMA 影子权重的数值与 checkpoint 往返测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from neat_pi.config import FSDPConfig
from neat_pi.config import ModelConfig
from neat_pi.device.backend import DeviceContext, DeviceType
from neat_pi.model.pi05 import Pi05
from neat_pi.training.checkpoint import (
    load_checkpoint_ema,
    load_checkpoint_weights,
    save_checkpoint,
)
from neat_pi.training.ema import EmaTracker

_DECAY = 0.9


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


def _rank0_context() -> DeviceContext:
    """构造单进程 rank0 context。"""
    return DeviceContext(
        type=DeviceType.CPU,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )


def _fill_params(model: torch.nn.Module, value: float) -> None:
    """把模型全部参数原地填成同一标量（解析对照用）。"""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def test_shadow_starts_equal_and_update_matches_closed_form() -> None:
    """影子初始化为主模型；参数固定时 update 后落在解析解上。"""
    model = torch.nn.Linear(4, 3)
    _fill_params(model, 1.0)
    tracker = EmaTracker(model, _DECAY)

    for _, ema_param, param in tracker.param_pairs():
        assert torch.equal(ema_param, param), "影子应从训练参数初始化"

    for _ in range(3):
        tracker.update()
    expected = _DECAY**3 * 1.0 + (1 - _DECAY**3) * 1.0
    for _, ema_param, _ in tracker.param_pairs():
        torch.testing.assert_close(
            ema_param, torch.full_like(ema_param, expected), atol=1e-6, rtol=0)

    # 参数跳变后单步：ema = decay * ema + (1 - decay) * new
    _fill_params(model, 10.0)
    previous = [ema_param.clone() for _, ema_param, _ in tracker.param_pairs()]
    tracker.update()
    for (_, ema_param, _), old in zip(tracker.param_pairs(), previous):
        torch.testing.assert_close(
            ema_param, _DECAY * old + (1 - _DECAY) * 10.0, atol=1e-6, rtol=0)


def test_update_does_not_touch_training_params() -> None:
    """EMA 只写影子，主模型参数与梯度状态不受影响。"""
    model = torch.nn.Linear(4, 3)
    _fill_params(model, 2.0)
    tracker = EmaTracker(model, _DECAY)
    snapshot = [param.clone() for param in model.parameters()]

    tracker.update()

    for param, saved in zip(model.parameters(), snapshot):
        assert torch.equal(param, saved)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_construction_rejects_out_of_range_decay(decay: float) -> None:
    """decay 不在 (0, 1) 内（含 nan）直接报错。"""
    with pytest.raises(ValueError, match="ema_decay"):
        EmaTracker(torch.nn.Linear(2, 2), decay)


def test_checkpoint_roundtrip_restores_ema_separately(tmp_path: Path) -> None:
    """checkpoint 双份权重：weights 与 ema 各自恢复、互不串、key 集合一致。"""
    model = Pi05(_small_model_config())
    _fill_params(model, 1.0)
    tracker = EmaTracker(model, _DECAY)
    _fill_params(model, 3.0)  # 训练后与影子拉开距离
    tracker.update()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output_dir = tmp_path / "train"

    save_checkpoint(
        model,
        optimizer,
        11,
        str(output_dir),
        _rank0_context(),
        fsdp_config=asdict(FSDPConfig()),
        ema_model=tracker.model,
    )
    checkpoint_dir = output_dir / "step_0000011"

    restored = Pi05(_small_model_config())
    load_checkpoint_weights(restored, checkpoint_dir)
    shadow = Pi05(_small_model_config())
    assert load_checkpoint_ema(shadow, checkpoint_dir) is True

    restored_state = restored.state_dict()
    shadow_state = shadow.state_dict()
    model_state = model.state_dict()
    ema_source_state = tracker.model.state_dict()
    for name, tensor in model_state.items():
        torch.testing.assert_close(restored_state[name], tensor)
        torch.testing.assert_close(shadow_state[name], ema_source_state[name])
        assert not torch.equal(shadow_state[name], restored_state[name]), name

    with safe_open(checkpoint_dir / "model.safetensors", framework="pt") as f:
        model_keys = set(f.keys())
    with safe_open(checkpoint_dir / "ema.safetensors", framework="pt") as f:
        ema_keys = set(f.keys())
    assert model_keys == ema_keys


def test_resume_without_ema_file_falls_back_to_training_params(
        tmp_path: Path) -> None:
    """旧 checkpoint（无 ema.safetensors）resume 时影子从训练参数起步。"""
    model = Pi05(_small_model_config())
    _fill_params(model, 2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output_dir = tmp_path / "train"

    save_checkpoint(
        model,
        optimizer,
        5,
        str(output_dir),
        _rank0_context(),
        fsdp_config=asdict(FSDPConfig()),
    )
    checkpoint_dir = output_dir / "step_0000005"

    # 兼容语义：先按训练参数初始化影子（EmaTracker 构造），ema 文件缺失
    # 时保持该初始化 —— 即 openpi 的 EMA 启动语义
    fallback = EmaTracker(model, _DECAY)
    assert load_checkpoint_ema(fallback.model, checkpoint_dir) is False
    for _, ema_param, param in fallback.param_pairs():
        assert torch.equal(ema_param, param)
