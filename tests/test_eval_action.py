"""LIBERO eval action queue 与 MEAN_STD unnormalizer 测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from neat_pi.eval.action import LiberoActionAdapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_pi05_libero_finetuned.yaml"


def _load_config() -> dict[str, Any]:
    """读取 eval YAML 顶层映射。"""
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _adapter(config: dict[str, Any] | None = None) -> LiberoActionAdapter:
    """从 eval YAML 构造 action adapter。"""
    return LiberoActionAdapter.from_config(
        config or _load_config(), config_root=CONFIG_PATH.parent)


def _model_actions(seed: int = 1000) -> torch.Tensor:
    """构造 batch=1 的 32 维模型动作，后 25 维显式 padding。"""
    generator = torch.Generator().manual_seed(seed)
    actions = torch.randn(1, 50, 32, generator=generator)
    actions[..., 7:] = 999.0
    return actions


def test_from_config_uses_checkpoint_action_contract() -> None:
    """eval YAML 的 chunk/raw dim/clip 合同被完整传入 adapter。"""
    adapter = _adapter()
    config = _load_config()
    assert adapter.action_horizon == config["model"]["action_horizon"]
    assert adapter.raw_action_dim == config["actions"]["raw_dim"]
    assert adapter.model_action_dim == config["model"]["max_action_dim"]
    assert adapter.clip_bounds == tuple(config["env"]["clip_bounds"])
    assert adapter.pending_count == 0
    assert adapter.model_calls == 0


def test_submit_slices_unnormalizes_clips_and_queues_chunk() -> None:
    """一次采样入队完整 50 步，只有前 7 维参与 postprocessor。"""
    adapter = _adapter()
    model_actions = _model_actions()
    queued = adapter.submit(model_actions)

    assert queued == 50
    assert adapter.model_calls == 1
    assert adapter.pending_count == 50

    popped = [adapter.pop_action() for _ in range(50)]
    assert all(action.shape == (7,) for action in popped)
    assert all(action.dtype is torch.float32 for action in popped)
    assert all(torch.isfinite(action).all() for action in popped)
    assert all((action >= -1.0).all() and (action <= 1.0).all()
               for action in popped)
    assert adapter.pending_count == 0

    expected = adapter.postprocessor(model_actions[..., :7])[0]
    expected = expected.clamp(-1.0, 1.0)
    torch.testing.assert_close(torch.stack(popped), expected)


def test_queue_must_drain_before_new_model_call() -> None:
    """旧 checkpoint 每次采样后必须消费完整 chunk。"""
    adapter = _adapter()
    adapter.submit(_model_actions(seed=1))
    assert adapter.pending_count == 50
    with pytest.raises(RuntimeError, match="不能提交新 chunk"):
        adapter.submit(_model_actions(seed=2))

    for _ in range(50):
        adapter.pop_action()
    assert adapter.pending_count == 0
    assert adapter.submit(_model_actions(seed=3)) == 50
    assert adapter.model_calls == 2


def test_configured_clip_bounds_are_applied() -> None:
    """adapter 使用 YAML 里的 LIBERO action bounds，而不是硬编码范围。"""
    config = _load_config()
    config["env"]["clip_bounds"] = [-0.25, 0.25]
    adapter = _adapter(config)
    adapter.submit(_model_actions(seed=4))
    actions = torch.stack([adapter.pop_action() for _ in range(50)])
    assert (actions >= -0.25).all()
    assert (actions <= 0.25).all()


def test_invalid_model_shape_is_rejected() -> None:
    """chunk 长度或模型动作维错误都直接失败。"""
    adapter = _adapter()
    with pytest.raises(ValueError, match="shape"):
        adapter.submit(torch.zeros(1, 32, 32))
    with pytest.raises(ValueError, match="shape"):
        adapter.submit(torch.zeros(1, 50, 7))


def test_pop_empty_queue_raises() -> None:
    """queue 为空时不能继续步进环境。"""
    adapter = _adapter()
    with pytest.raises(IndexError, match="action queue 已空"):
        adapter.pop_action()
