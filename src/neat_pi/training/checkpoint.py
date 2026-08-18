"""分布式 checkpoint 存取。

约定：
- 只在主进程写盘（ctx.is_main_process）；
- FSDP 状态下用 FullStateDictConfig 把分片参数聚合成完整 state_dict
  再保存，保证 checkpoint 与权重加载（weights.py）格式一致，
  即保存出来的仍是 pi05 兼容布局。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from neat_pi.device.backend import DeviceContext


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int, output_dir: str, ctx: DeviceContext) -> None:
    """保存训练 checkpoint（含模型、优化器、步数）。

    TODO: FSDP 场景接入 FullStateDictConfig / ShardedStateDictConfig；
    当前实现仅覆盖单卡/未分片场景。
    """
    if not ctx.is_main_process:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"step_{step:07d}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    path: str) -> int:
    """恢复训练状态，返回已训练的步数。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])
