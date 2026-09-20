"""分布式 DataLoader、batch 组装与 FSDP 训练循环辅助逻辑测试。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from neat_pi.config import Config
from neat_pi.device.backend import DeviceContext, DeviceType
from neat_pi.training.batch import prepare_batch
from neat_pi.training.fsdp import clip_grad_norm
from neat_pi.training.data import build_train_loader


def _cpu_rank(rank: int, world_size: int = 2) -> DeviceContext:
    """构造测试用 CPU DeviceContext。"""
    return DeviceContext(
        type=DeviceType.CPU,
        device=torch.device("cpu"),
        rank=rank,
        world_size=world_size,
    )


def test_distributed_loader_splits_indices_without_overlap() -> None:
    """两个 rank 的 micro batch 样本互不重复，且各 rank batch 数一致。"""
    cfg = Config()
    cfg.data.per_device_batch_size = 2
    cfg.data.num_workers = 0
    dataset = TensorDataset(torch.arange(8))

    loader_rank0, _ = build_train_loader(dataset, cfg, _cpu_rank(0))
    loader_rank1, _ = build_train_loader(dataset, cfg, _cpu_rank(1))
    indices_rank0: list[list[int]] = []
    for batch in loader_rank0:
        indices_rank0.append([int(sample) for sample in batch[0]])
    indices_rank1: list[list[int]] = []
    for batch in loader_rank1:
        indices_rank1.append([int(sample) for sample in batch[0]])

    assert len(indices_rank0) == len(indices_rank1) == 2
    for rank0_batch, rank1_batch in zip(
            indices_rank0, indices_rank1, strict=True):
        assert len(rank0_batch + rank1_batch) == len(set(rank0_batch + rank1_batch))
    all_indices = sum((indices_rank0 + indices_rank1), [])
    assert sorted(all_indices) == list(range(8))


def test_clip_grad_norm_falls_back_for_unwrapped_model() -> None:
    """未包 FSDP 的模型回退到 PyTorch 梯度裁剪。"""
    model = nn.Linear(4, 2)
    model(torch.randn(3, 4)).sum().backward()

    total_norm = clip_grad_norm(model, 0.1)

    assert total_norm.item() > 0.1
    assert torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1e9
    ).item() <= 0.1 + 1e-6
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )


def _batch_with_cameras(num_cameras: int, batch_size: int = 2,
                        image_size: int = 16) -> dict:
    """构造含指定路数真实相机的 preprocessor 输出 batch。"""
    camera_keys = ["observation.images.image", "observation.images.image2"]
    batch: dict = {
        key: torch.rand(batch_size, 3, image_size, image_size)
        for key in camera_keys[:num_cameras]
    }
    batch["observation.language.tokens"] = torch.randint(0, 10, (batch_size, 5))
    batch["observation.language.attention_mask"] = torch.ones(
        batch_size, 5, dtype=torch.bool)
    batch["action"] = torch.randn(batch_size, 4, 7)
    return batch


def test_prepare_batch_injects_empty_camera_with_mask_false() -> None:
    """MEAN_STD 合同：缺失槽位补 empty camera（-1 填充）且 mask=False，
    排在真实相机之后，对齐评测合同顺序。"""
    cfg = Config()
    cfg.model.num_cameras = 3
    cfg.model.image_size = 16
    batch = _batch_with_cameras(num_cameras=2)

    tb = prepare_batch(cfg, batch, torch.device("cpu"))

    assert len(tb.images) == len(tb.image_masks) == 3
    # 顺序：真实相机按 key 排序在前（image, image2），empty 补最后
    torch.testing.assert_close(
        tb.images[2], torch.full((2, 3, 16, 16), -1.0))
    assert tb.image_masks[0].all() and tb.image_masks[1].all()
    assert not tb.image_masks[2].any()
    # 真实相机内容原样透传
    torch.testing.assert_close(tb.images[0], batch["observation.images.image"])
    torch.testing.assert_close(tb.images[1], batch["observation.images.image2"])


def test_prepare_batch_without_missing_camera_keeps_masks_true() -> None:
    """SE(3) 合同：batch 相机数等于配置时行为不变，mask 全 True。"""
    cfg = Config()
    cfg.model.num_cameras = 2
    cfg.model.image_size = 16
    batch = _batch_with_cameras(num_cameras=2)

    tb = prepare_batch(cfg, batch, torch.device("cpu"))

    assert len(tb.images) == 2
    assert all(mask.all() for mask in tb.image_masks)


def test_prepare_batch_rejects_cameras_beyond_config() -> None:
    """batch 相机数超过配置 num_cameras 时直接报错（配置笔误护栏）。"""
    cfg = Config()
    cfg.model.num_cameras = 1
    cfg.model.image_size = 16
    batch = _batch_with_cameras(num_cameras=2)

    with pytest.raises(ValueError, match="num_cameras"):
        prepare_batch(cfg, batch, torch.device("cpu"))
