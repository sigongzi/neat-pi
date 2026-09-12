"""分布式 DataLoader 与 FSDP 训练循环辅助逻辑测试。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import TensorDataset

from neat_pi.config import Config
from neat_pi.device.backend import DeviceContext, DeviceType
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
