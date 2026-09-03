"""训练入口：`torchrun [分布式参数] scripts/train.py --config <yaml>`。

整个训练循环都在本脚本里（命令行解析、main、训练循环都属于 scripts 层）。
分布式拓扑由 torchrun 环境变量提供（见 scripts/train.sh）。

流程：
1. 加载配置 -> 2. 初始化设备/分布式（device.backend）->
3. 构建数据（LeRobot v3.1 + preprocessor 管线）-> 4. 构建模型 ->
5. FSDP 包装 -> 6. 训练循环（flow matching 损失）。

当前模型默认用 DummyPi05 冒烟（Pi05.predict_velocity 尚未实现），配置
`training.use_dummy_model` 置 false 后切回真实 Pi05。
"""

from __future__ import annotations

import argparse
from typing import Any, NamedTuple

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neat_pi.config import Config, load_config
from neat_pi.data.transforms import images_to_float
from neat_pi.device import backend
from neat_pi.training.checkpoint import save_checkpoint
from neat_pi.training.dummy_model import DummyPi05
from neat_pi.training.fsdp import wrap_model_fsdp


class TrainBatch(NamedTuple):
    """一个训练 batch 的模型侧形态：state/action 已补零到模型维度。"""

    images: list[torch.Tensor]  # 各相机图像 (B,3,H,W)，[-1,1]
    token_ids: torch.Tensor     # (B, max_token_len)
    state: torch.Tensor         # (B, state_dim) 补零后
    actions: torch.Tensor       # (B, action_horizon, action_dim) 补零后
    is_pad: torch.Tensor        # (B, action_horizon)，True=padding 帧
    real_action_dim: int        # padding 前的真实动作维


def prepare_batch(cfg: Config, batch: dict[str, Any],
                  device: torch.device) -> TrainBatch:
    """把 preprocessor 输出的 batch 整理成模型输入。

    state (B,11) 与 action (B,H,10) 补零到 config 的 state_dim/action_dim
    （真实模型侧同样按这两个维度做输入投影，padding 由这里统一负责）。
    """
    images = [batch[k].to(device) for k in sorted(batch)
              if k.startswith("observation.images.")]
    token_ids = batch["observation.language.tokens"].to(device).long()
    state = batch["observation.state"].to(device).float()
    actions = batch["action"].to(device).float()
    is_pad = batch.get("action_is_pad")
    is_pad = (torch.zeros_like(actions[..., 0]).bool()
              if is_pad is None else is_pad.to(device).bool())

    real_action_dim = actions.shape[-1]
    state = F.pad(state, (0, cfg.model.state_dim - state.shape[-1]))
    actions = F.pad(actions, (0, cfg.model.action_dim - actions.shape[-1]))
    return TrainBatch(images, token_ids, state, actions, is_pad, real_action_dim)


def build_model(cfg: Config, ctx: backend.DeviceContext,
                amp_dtype: torch.dtype) -> torch.nn.Module:
    """按配置构建模型：dummy 冒烟 / 加载 pretrained / 随机初始化。

    ctx 与 amp_dtype 存进模型，forward 内部据此开混合精度并计算损失。
    """
    if cfg.training.use_dummy_model:
        logger.info("使用 DummyPi05 冒烟训练（Pi05.predict_velocity 尚未实现）")
        model = DummyPi05(cfg.model)
    else:
        from neat_pi.model.pi05 import Pi05

        if cfg.training.pretrained:
            model = Pi05.from_pretrained(cfg.training.pretrained, cfg.model,
                                         ctx.device)
        else:
            model = Pi05(cfg.model)
    model.ctx = ctx
    model.amp_dtype = amp_dtype
    return model.to(ctx.device)


def run_training(config_path: str) -> None:
    """训练主流程。"""
    cfg = load_config(config_path)

    # 1. 设备与分布式：torchrun 下读 RANK/LOCAL_RANK/WORLD_SIZE，单进程退化为单卡
    ctx = backend.init_device(cfg.device.type)
    amp_dtype = backend.get_amp_dtype(cfg.device.dtype)
    logger.info("设备: {} {} | rank {}/{} | amp {}",
                ctx.type.value, ctx.device, ctx.rank, ctx.world_size, amp_dtype)

    # 2. 数据：LeRobotDataset + DataLoader 默认 collate + preprocessor 管线
    from neat_pi.data.lerobot_dataset import build_dataset
    from neat_pi.data.preprocessor import load_preprocessor

    ds = build_dataset(cfg.data, cfg.model)
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    preprocessor = load_preprocessor(cfg)
    logger.info("数据集: {} 帧, batch {} x {} workers",
                len(ds), cfg.data.batch_size, cfg.data.num_workers)

    # 3. 模型 + FSDP（单卡时原样返回）
    model = build_model(cfg, ctx, amp_dtype)
    model = wrap_model_fsdp(model, cfg.training.fsdp, ctx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

    # 4. 训练循环（checkpoint 仅主进程写盘）
    model.train()
    data_iter = iter(loader)
    logger.info("开始训练 {} 步", cfg.training.max_steps)
    for step in range(cfg.training.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        tb = prepare_batch(cfg, preprocessor(images_to_float(batch)), ctx.device)
        loss = model(tb.images, tb.token_ids, tb.state, tb.actions,
                     tb.is_pad, tb.real_action_dim)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % cfg.training.log_every == 0:
            logger.info("step {}/{} | loss {:.4f}",
                        step, cfg.training.max_steps, loss.item())
        if step % cfg.training.save_every == 0:
            save_checkpoint(model, optimizer, step, cfg.training.output_dir, ctx)

    backend.cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析命令行参数（只有 --config）。"""
    parser = argparse.ArgumentParser(description="neat-pi 训练入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    return parser.parse_args()


def main() -> None:
    """训练入口。"""
    run_training(parse_args().config)


if __name__ == "__main__":
    main()
