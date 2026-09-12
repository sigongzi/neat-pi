"""分布式训练数据构建辅助。"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from neat_pi.config import Config
from neat_pi.device.backend import DeviceContext


def build_train_loader(
    dataset: Dataset,
    cfg: Config,
    ctx: DeviceContext,
) -> tuple[DataLoader, DistributedSampler]:
    """构建按 rank 切分并丢弃尾部不完整 batch 的训练 DataLoader。"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=cfg.training.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.per_device_batch_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=True,
    )
    return loader, sampler
