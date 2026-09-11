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
    load_checkpoint_metadata,
    load_checkpoint_optimizer,
    load_checkpoint_weights,
    prune_checkpoints,
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

    # 产生优化器状态（exp_avg / exp_avg_sq / step），验证恢复非空状态
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

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
    # 权重与优化器分开加载：权重须在 FSDP 包装前的未分片模型上恢复
    assert load_checkpoint_weights(restored, checkpoint_path) == 7
    load_checkpoint_optimizer(restored, restored_optimizer, checkpoint_path)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    for param, restored_param in zip(optimizer.param_groups[0]["params"],
                                     restored_optimizer.param_groups[0]["params"],
                                     strict=True):
        expected_state = optimizer.state[param]
        actual_state = restored_optimizer.state[restored_param]
        assert set(expected_state) == set(actual_state)
        for key, value in expected_state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual_state[key], value)
            else:
                assert actual_state[key] == value


def _make_checkpoint_dir(root: Path, step: int,
                         complete: bool = True) -> Path:
    """构造一个 checkpoint 目录；complete=False 时缺少 optimizer.pt。"""
    path = root / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text("{}", encoding="utf-8")
    if complete:
        (path / "optimizer.pt").write_bytes(b"optimizer")
        (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_prune_keep_last_n_deletes_older_checkpoints(tmp_path: Path) -> None:
    """keep_last_n=2：最近的 n 个保留，更旧的完整 checkpoint 被删除。"""
    for step in range(1, 6):
        _make_checkpoint_dir(tmp_path, step)

    deleted = prune_checkpoints(tmp_path, keep_last_n=2, keep_every=0)

    assert [path.name for path in deleted] == [
        "step_0000001", "step_0000002", "step_0000003"]
    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_0000004", "step_0000005"]


def test_prune_keeps_milestones_even_without_recency(tmp_path: Path) -> None:
    """keep_every=2 且 keep_last_n=0：只保留里程碑，其余全删。"""
    for step in range(1, 7):
        _make_checkpoint_dir(tmp_path, step)

    deleted = prune_checkpoints(tmp_path, keep_last_n=0, keep_every=2)

    assert [path.name for path in deleted] == [
        "step_0000001", "step_0000003", "step_0000005"]
    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_0000002", "step_0000004", "step_0000006"]


def test_prune_combines_recency_and_milestones(tmp_path: Path) -> None:
    """里程碑与最近 n 个取并集保留。"""
    for step in range(1, 7):
        _make_checkpoint_dir(tmp_path, step)

    deleted = prune_checkpoints(tmp_path, keep_last_n=2, keep_every=3)

    assert [path.name for path in deleted] == [
        "step_0000001", "step_0000002", "step_0000004"]
    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_0000003", "step_0000005", "step_0000006"]


def test_prune_protects_requested_directory(tmp_path: Path) -> None:
    """protect 指向的目录即使不满足任何保留规则也不删除。"""
    for step in range(1, 6):
        _make_checkpoint_dir(tmp_path, step)
    protect = tmp_path / "step_0000002"

    deleted = prune_checkpoints(
        tmp_path, keep_last_n=0, keep_every=10, protect=protect)

    assert [path.name for path in deleted] == [
        "step_0000001", "step_0000003", "step_0000004", "step_0000005"]
    assert protect.is_dir()


def test_prune_skips_incomplete_and_unknown_directories(
        tmp_path: Path) -> None:
    """半成品 checkpoint 与非 step_* 目录不参与清理。"""
    for step in range(1, 4):
        _make_checkpoint_dir(tmp_path, step)
    incomplete = _make_checkpoint_dir(tmp_path, 4, complete=False)
    unrelated = tmp_path / "scratch"
    unrelated.mkdir()

    deleted = prune_checkpoints(tmp_path, keep_last_n=1, keep_every=0)

    assert [path.name for path in deleted] == [
        "step_0000001", "step_0000002"]
    assert incomplete.is_dir()
    assert unrelated.is_dir()
    assert (tmp_path / "step_0000003").is_dir()


def test_prune_disabled_by_default(tmp_path: Path) -> None:
    """keep_last_n / keep_every 均为 0 时不清理，行为与历史全保留一致。"""
    for step in range(1, 4):
        _make_checkpoint_dir(tmp_path, step)

    deleted = prune_checkpoints(tmp_path, keep_last_n=0, keep_every=0)

    assert deleted == []
    assert len(list(tmp_path.glob("step_*"))) == 3


def test_save_checkpoint_applies_retention(tmp_path: Path) -> None:
    """save_checkpoint 按 keep_last_n 清理旧目录，latest 指向新目录。"""
    model = _small_pi05()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = _rank0_context()
    output_dir = tmp_path / "train"

    for step in (7, 14, 21):
        save_checkpoint(
            model,
            optimizer,
            step=step,
            output_dir=str(output_dir),
            ctx=ctx,
            keep_last_n=2,
        )

    remaining = sorted(path.name for path in output_dir.glob("step_*"))
    assert remaining == ["step_0000014", "step_0000021"]
    assert latest_checkpoint_path(str(output_dir)) == output_dir / "step_0000021"
