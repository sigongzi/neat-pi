"""训练入口：`torchrun -m neat_pi.training.trainer --config <yaml>`。

命令行只接受 --config；分布式拓扑由 torchrun 环境变量提供（见 scripts/train.sh）。

流程：
1. 加载配置 -> 2. 初始化设备/分布式（device.backend）->
3. 构建数据（LeRobot v3.1）-> 4. 构建模型并按需加载 pi05 权重 ->
5. FSDP 包装 -> 6. 训练循环（flow matching 损失）。
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from neat_pi.config import load_config
from neat_pi.device import backend
from neat_pi.model.pi05 import Pi05


def parse_args() -> argparse.Namespace:
    """解析命令行参数（只有 --config）。"""
    parser = argparse.ArgumentParser(description="neat-pi 训练入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args = parse_args()
    cfg = load_config(args.config)

    # 1. 设备与分布式：torchrun 下读 RANK/LOCAL_RANK/WORLD_SIZE，单进程退化为单卡
    ctx = backend.init_device(cfg.device.type)
    amp_dtype = backend.get_amp_dtype(cfg.device.dtype)
    logger.info("设备: {} {} | rank {}/{} | amp {}",
                ctx.type.value, ctx.device, ctx.rank, ctx.world_size, amp_dtype)

    # 2. 数据（TODO: LiberoDataset 尚未实现，接通后取消注释）
    # from neat_pi.data.lerobot_dataset import LiberoDataset
    # dataset = LiberoDataset(cfg.data, cfg.model)
    # loader = torch.utils.data.DataLoader(...)

    # 3. 模型：给了 pretrained 就加载 pi05 权重，否则随机初始化（仅冒烟用）
    if cfg.training.pretrained:
        model = Pi05.from_pretrained(cfg.training.pretrained, cfg.model, ctx.device)
    else:
        model = Pi05(cfg.model).to(ctx.device)

    # 4. FSDP（单卡时原样返回）
    from neat_pi.training.fsdp import wrap_model_fsdp

    model = wrap_model_fsdp(model, cfg.training.fsdp, ctx)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

    # 5. 训练循环骨架
    model.train()
    for step in range(cfg.training.max_steps):
        # TODO: 取 batch -> 采样 flow matching 的 t 与噪声 ->
        #   noisy_action = (1-t)*noise + t*actions，目标 velocity = actions - noise
        #   with backend.autocast(ctx, amp_dtype):
        #       pred = model(images, token_ids, state, noisy_action, t)
        #       loss = mse(pred, actions - noise)
        raise NotImplementedError("待数据管线接通后实现训练循环")

    backend.cleanup_distributed()


if __name__ == "__main__":
    main()
