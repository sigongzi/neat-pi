"""LIBERO eval action queue 与 MEAN_STD / SE(3) unnormalizer 测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from neat_pi.data.se3 import aa_to_rot, rot_to_6d, rot_to_aa
from neat_pi.eval.action import LiberoActionAdapter, LiberoSe3ActionAdapter


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
    assert adapter.execution_horizon == config["actions"]["execution_horizon"]
    assert adapter.raw_action_dim == config["actions"]["raw_dim"]
    assert adapter.model_action_dim == config["model"]["max_action_dim"]
    assert adapter.clip_bounds == tuple(config["env"]["clip_bounds"])
    assert adapter.pending_count == 0
    assert adapter.model_calls == 0


def test_submit_slices_unnormalizes_clips_and_queues_chunk() -> None:
    """一次采样入队 execution_horizon 步，只有前 7 维参与 postprocessor。"""
    adapter = _adapter()
    model_actions = _model_actions()
    queued = adapter.submit(model_actions)

    assert queued == 10
    assert adapter.model_calls == 1
    assert adapter.pending_count == 10

    popped = [adapter.pop_action() for _ in range(10)]
    assert all(action.shape == (7,) for action in popped)
    assert all(action.dtype is torch.float32 for action in popped)
    assert all(torch.isfinite(action).all() for action in popped)
    assert all((action >= -1.0).all() and (action <= 1.0).all()
               for action in popped)
    assert adapter.pending_count == 0

    expected = adapter.postprocessor(model_actions[:, :10, :7])[0]
    expected = expected.clamp(-1.0, 1.0)
    torch.testing.assert_close(torch.stack(popped), expected)


def test_queue_must_drain_before_new_model_call() -> None:
    """每次采样后必须消费 execution_horizon 步再重新推理。"""
    adapter = _adapter()
    adapter.submit(_model_actions(seed=1))
    assert adapter.pending_count == 10
    with pytest.raises(RuntimeError, match="不能提交新 chunk"):
        adapter.submit(_model_actions(seed=2))

    for _ in range(10):
        adapter.pop_action()
    assert adapter.pending_count == 0
    assert adapter.submit(_model_actions(seed=3)) == 10
    assert adapter.model_calls == 2


def test_reset_clears_queue_and_model_call_count() -> None:
    """每个 episode 开始前必须重置跨 episode 的 queue 与统计。"""
    adapter = _adapter()
    adapter.submit(_model_actions(seed=5))
    adapter.reset()
    assert adapter.pending_count == 0
    assert adapter.model_calls == 0
    with pytest.raises(IndexError, match="action queue 已空"):
        adapter.pop_action()


def test_configured_clip_bounds_are_applied() -> None:
    """adapter 使用 YAML 里的 LIBERO action bounds，而不是硬编码范围。"""
    config = _load_config()
    config["env"]["clip_bounds"] = [-0.25, 0.25]
    adapter = _adapter(config)
    adapter.submit(_model_actions(seed=4))
    actions = torch.stack([adapter.pop_action() for _ in range(10)])
    assert (actions >= -0.25).all()
    assert (actions <= 0.25).all()


def test_invalid_model_shape_is_rejected() -> None:
    """chunk 长度或模型动作维错误都直接失败。"""
    adapter = _adapter()
    with pytest.raises(ValueError, match="shape"):
        adapter.submit(torch.zeros(1, 32, 32))
    with pytest.raises(ValueError, match="shape"):
        adapter.submit(torch.zeros(1, 50, 7))


def test_invalid_execution_horizon_is_rejected() -> None:
    """execution horizon 不能超过模型 chunk 长度。"""
    config = _load_config()
    config["actions"]["execution_horizon"] = 51
    with pytest.raises(ValueError, match="execution_horizon"):
        _adapter(config)


def test_pop_empty_queue_raises() -> None:
    """queue 为空时不能继续步进环境。"""
    adapter = _adapter()
    with pytest.raises(IndexError, match="action queue 已空"):
        adapter.pop_action()


# ---------------------------------------------------------------------------
# LiberoSe3ActionAdapter：chunk-wise SE(3) delta + 逐步重参考
# ---------------------------------------------------------------------------

SE3_CONFIG = {
    "model": {"action_horizon": 32, "max_action_dim": 32},
    "actions": {"execution_horizon": 4, "normalization": "QUANTILES"},
    "env": {"clip_bounds": [-1.0, 1.0]},
    "normalization": {
        "postprocessor_path": "pi05_libero_se3_unnormalize.json",
    },
}


def _se3_adapter(config: dict[str, Any] | None = None) -> LiberoSe3ActionAdapter:
    """构造 SE(3) adapter。"""
    return LiberoSe3ActionAdapter.from_config(
        config or deepcopy(SE3_CONFIG), config_root=CONFIG_PATH.parent)


def _delta_chunk(rows: list[list[float]]) -> torch.Tensor:
    """构造 (1, E, 10) 期望物理 delta；rot6D 由 axis-angle 生成保证合法。

    行格式：[dpos(3), aa(3), gripper(1)]，内部转成 [dpos, rot6D, gripper]。
    """
    deltas = []
    for row in rows:
        dpos, aa, gripper = torch.tensor(row[0:3]), torch.tensor(row[3:6]), row[6]
        rot6d = rot_to_6d(aa_to_rot(aa))
        deltas.append(torch.cat([dpos, rot6d, torch.tensor([gripper])]))
    return torch.stack(deltas).unsqueeze(0)


def _fake_model_input(deltas: torch.Tensor,
                      adapter: LiberoSe3ActionAdapter) -> torch.Tensor:
    """期望物理 delta → 统计量正变换 → 归一化模型输出（含 padding 维）。

    正变换公式与 lerobot QUANTILES 相同：n = 2*(x - q01)/(q99 - q01) - 1。
    """
    unnorm = adapter.postprocessor.steps[0]
    stats = unnorm.stats["action"]
    q01 = torch.as_tensor(stats["q01"]).float()
    q99 = torch.as_tensor(stats["q99"]).float()
    normalized = 2.0 * (deltas - q01) / (q99 - q01).clamp_min(1e-8) - 1.0
    padded = torch.zeros(1, adapter.action_horizon, adapter.model_action_dim)
    padded[:, : deltas.shape[1], :10] = normalized
    padded[:, deltas.shape[1]:, :] = 999.0
    return padded


def _state(pos: list[float], aa: list[float]) -> torch.Tensor:
    """构造 (1, 8) 观测 state（gripper qpos 用占位 [0.01, 0.01]）。"""
    return torch.tensor([pos + aa + [0.01, 0.01]], dtype=torch.float32)


def test_se3_pop_recovers_delta_in_start_frame() -> None:
    """首帧重参考：指令 = delta / 增益（delta 定义在 chunk 首帧 EE 系）。"""
    adapter = _se3_adapter()
    start = _state([0.4, -0.1, 0.7], [0.0, 0.0, 0.0])
    rows = [
        [0.01, -0.02, 0.005, 0.0, 0.0, 0.05, -1.0],
        [0.0, 0.01, 0.0, 0.0, 0.05, 0.0, 1.0],
    ]
    desired = _delta_chunk(rows)
    assert adapter.submit(_fake_model_input(desired, adapter), start) == 4
    assert adapter.model_calls == 1

    for k, row in enumerate(rows):
        command = adapter.pop_action(start)
        torch.testing.assert_close(
            command[0:3], torch.tensor(row[0:3]) / adapter.pos_gain,
            rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            command[3:6], torch.tensor(row[3:6]) / adapter.rot_gain,
            rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            command[6:7], torch.tensor([row[6]]))


def test_se3_world_frame_conjugation() -> None:
    """delta 在首帧 EE 系下，重参考指令是世界系共轭：dpos=R0·dt、
    drot=aa(R0·dR·R0^T)。"""
    adapter = _se3_adapter()
    yaw = 0.3
    start = _state([0.4, -0.1, 0.7], [0.0, 0.0, yaw])
    desired = _delta_chunk([[0.01, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]])
    adapter.submit(_fake_model_input(desired, adapter), start)

    command = adapter.pop_action(start)
    r0 = aa_to_rot(start[0, 3:6])
    torch.testing.assert_close(
        command[0:3], (r0 @ desired[0, 0, 0:3]) / adapter.pos_gain,
        rtol=1e-4, atol=1e-6)
    drot_world = r0 @ aa_to_rot(torch.tensor([0.0, 0.0, 0.05])) @ r0.transpose(-1, -2)
    torch.testing.assert_close(
        command[3:6], rot_to_aa(drot_world) / adapter.rot_gain,
        rtol=1e-4, atol=1e-5)


def test_se3_pop_rereferences_against_fresh_state() -> None:
    """实际位姿滞后于首帧时，指令指向绝对目标而非机械复读 chunk。"""
    adapter = _se3_adapter()
    start = _state([0.4, -0.1, 0.7], [0.0, 0.0, 0.0])
    desired = _delta_chunk([
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    adapter.submit(_fake_model_input(desired, adapter), start)

    # 第 1 步无滞后：指令 = delta / 增益（0.05/0.05 = 1.0，触到 clip 上界）
    first = adapter.pop_action(start)
    assert first[0].item() == pytest.approx(1.0)

    # 第 2 步观测显示第 1 步只走了 80%：剩余目标按新位姿重新相减
    drifted = _state([0.4 + 0.05 * 0.8, -0.1, 0.7], [0.0, 0.0, 0.0])
    second = adapter.pop_action(drifted)
    torch.testing.assert_close(
        second[0:3], torch.tensor([0.2, 0.0, 0.0]), rtol=1e-4, atol=1e-6)

    # 旋转同理：目标已达成时指令归零（绝对目标不被二次执行）
    adapter.reset()
    rot_desired = _delta_chunk([[0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]])
    adapter.submit(_fake_model_input(rot_desired, adapter), start)
    moved = _state([0.4, -0.1, 0.7], [0.0, 0.0, 0.1])
    command = adapter.pop_action(moved)
    torch.testing.assert_close(command[3:6], torch.zeros(3), rtol=1e-4, atol=1e-5)


def test_se3_gripper_passthrough_and_clamp() -> None:
    """gripper 维透传并按 clip_bounds 截断；指令各维不越界。"""
    config = deepcopy(SE3_CONFIG)
    config["env"]["clip_bounds"] = [-0.5, 0.5]
    adapter = _se3_adapter(config)
    start = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    desired = _delta_chunk([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    adapter.submit(_fake_model_input(desired, adapter), start)
    command = adapter.pop_action(start)
    assert (command >= -0.5).all() and (command <= 0.5).all()
    assert command[6].item() == pytest.approx(0.5)


def test_se3_queue_semantics_match_raw_adapter() -> None:
    """未清空禁止重提交；reset 清队列与计数；空队列 pop 报错。"""
    adapter = _se3_adapter()
    start = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    desired = _delta_chunk([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    adapter.submit(_fake_model_input(desired, adapter), start)
    with pytest.raises(RuntimeError, match="不能提交新 chunk"):
        adapter.submit(_fake_model_input(desired, adapter), start)
    adapter.reset()
    assert adapter.pending_count == 0
    assert adapter.model_calls == 0
    with pytest.raises(IndexError, match="action queue 已空"):
        adapter.pop_action(start)


def test_se3_rejects_bad_shapes_and_normalization() -> None:
    """模型 chunk / state 形状错误与归一化合同不符都直接失败。"""
    adapter = _se3_adapter()
    start = _state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    good = torch.zeros(1, adapter.action_horizon, adapter.model_action_dim)
    with pytest.raises(ValueError, match="shape"):
        adapter.submit(torch.zeros(1, 32, 7), start)
    with pytest.raises((ValueError, TypeError)):
        adapter.submit(good, torch.zeros(2, 8))
    with pytest.raises(ValueError, match="QUANTILES"):
        config = deepcopy(SE3_CONFIG)
        config["actions"]["normalization"] = "MEAN_STD"
        _se3_adapter(config)
