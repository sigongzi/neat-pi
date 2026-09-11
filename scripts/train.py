"""训练入口：`torchrun [分布式参数] scripts/train.py --config <yaml>`。

整个训练循环都在本脚本里（命令行解析、main、训练循环都属于 scripts 层）。
分布式拓扑由 torchrun 环境变量提供（见 scripts/train.sh）。

流程：
1. 加载配置 -> 2. 初始化设备/分布式 -> 3. 构建分布式数据 ->
4. 构建模型 -> 5. FSDP 包装 -> 6. flow matching 训练循环。

当前模型默认用 DummyPi05 冒烟；配置 `training.use_dummy_model=false` 后
切回真实 Pi05。
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, NamedTuple

import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from neat_pi.config import Config, load_config
from neat_pi.data.transforms import images_to_float
from neat_pi.device import backend
from neat_pi.model.weights import load_pi05_weights_distributed
from neat_pi.training.checkpoint import save_checkpoint
from neat_pi.training.checkpoint import (
    latest_checkpoint_path,
    load_checkpoint,
    load_checkpoint_metadata,
)
from neat_pi.training.data import build_train_loader
from neat_pi.training.dummy_model import DummyPi05
from neat_pi.training.fsdp import clip_grad_norm, wrap_model_fsdp


def _format_eta(seconds: float) -> str:
    """把剩余秒数格式化为 HH:MM:SS。"""
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class TrainBatch(NamedTuple):
    """一个训练 batch 的模型侧形态：action 已补零到模型维度。"""

    images: list[torch.Tensor]  # 各相机图像 (B,3,H,W)，[-1,1]
    image_masks: list[torch.Tensor]  # 各相机可用性 (B,)，True=有效
    token_ids: torch.Tensor     # (B, max_token_len)
    lang_mask: torch.Tensor     # (B, max_token_len)，True=有效语言 token
    actions: torch.Tensor       # (B, action_horizon, action_dim) 补零后
    is_pad: torch.Tensor        # (B, action_horizon)，True=padding 帧
    real_action_dim: int        # padding 前的真实动作维


def prepare_batch(cfg: Config, batch: dict[str, Any],
                  device: torch.device) -> TrainBatch:
    """把 preprocessor 输出的 batch 整理成模型输入。

    action 维度不足 `model.action_dim` 时在右侧补零；语言 attention mask
    原样传给 Pi05 构造 prefix mask。
    """
    images = [batch[key].to(device) for key in sorted(batch)
              if key.startswith("observation.images.")]
    image_masks = [
        torch.ones(image.shape[0], dtype=torch.bool, device=device)
        for image in images
    ]
    token_ids = batch["observation.language.tokens"].to(device).long()
    lang_mask = batch["observation.language.attention_mask"].to(device).bool()
    actions = batch["action"].to(device).float()
    is_pad = batch.get("action_is_pad")
    is_pad = (
        torch.zeros_like(actions[..., 0]).bool()
        if is_pad is None
        else is_pad.to(device).bool()
    )
    real_action_dim = actions.shape[-1]
    actions = F.pad(
        actions,
        (0, cfg.model.action_dim - actions.shape[-1]),
    )
    return TrainBatch(
        images,
        image_masks,
        token_ids,
        lang_mask,
        actions,
        is_pad,
        real_action_dim,
    )


def build_model(cfg: Config, ctx: backend.DeviceContext,
                amp_dtype: torch.dtype) -> torch.nn.Module:
    """按配置构建 dummy 冒烟模型或真实 Pi05，并注入设备上下文。"""
    if cfg.training.use_dummy_model:
        model = DummyPi05(cfg.model)
    else:
        from neat_pi.model.pi05 import Pi05

        # rank0 broadcast 加载逻辑在 run_training 中调用，不在这里打开 checkpoint。
        model = Pi05(cfg.model)

    if ctx.is_main_process:
        model_name = "DummyPi05" if cfg.training.use_dummy_model else "Pi05"
        logger.info("构建模型: {}", model_name)
    model.ctx = ctx
    model.amp_dtype = amp_dtype
    return model.to(ctx.device)


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    cfg: Config,
    ctx: backend.DeviceContext,
    start_step: int = 0,
    start_epoch: int = 0,
    start_micro_step: int = 0,
) -> None:
    """执行 flow matching 训练循环；step 计数按 optimizer step 计算。"""
    accumulation_steps = cfg.training.grad_accum_steps
    max_steps = cfg.training.max_steps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = start_epoch
    optimizer_step = start_step
    micro_step = start_micro_step
    # 续训时恢复中断前 epoch 的 shuffle 顺序；全新训练 start_epoch=0，
    # 与 DistributedSampler 的默认值一致，行为不变
    sampler.set_epoch(epoch)
    data_iter = iter(loader)
    # per-step 耗时 EMA：跳过第一个 optimizer step（含数据冷启动）
    last_step_time: float | None = None
    s_per_step: float | None = None
    accumulation_losses: list[torch.Tensor] = []
    if ctx.is_main_process:
        logger.info(
            "开始训练 {} 步 | 梯度累计 {}",
            max_steps,
            accumulation_steps,
        )

    while optimizer_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(loader)
            continue

        tb = prepare_batch(
            cfg,
            preprocessor(images_to_float(batch)),
            ctx.device,
        )
        loss = model(
            tb.images,
            tb.image_masks,
            tb.token_ids,
            tb.lang_mask,
            tb.actions,
            tb.is_pad,
            tb.real_action_dim,
        )
        (loss / accumulation_steps).backward()
        accumulation_losses.append(loss.detach())
        micro_step += 1

        if micro_step % accumulation_steps != 0:
            continue

        # 保留 tensor，仅在打日志时才格式化（避免每步一次 GPU->CPU 同步）
        grad_norm: torch.Tensor | None = None
        if cfg.training.gradient_clip_norm is not None:
            grad_norm = clip_grad_norm(model, cfg.training.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        now = time.perf_counter()
        if last_step_time is not None:
            elapsed = now - last_step_time
            s_per_step = (elapsed if s_per_step is None
                          else 0.1 * elapsed + 0.9 * s_per_step)
        last_step_time = now

        if ctx.is_main_process and optimizer_step % cfg.training.log_every == 0:
            mean_loss = torch.stack(accumulation_losses).mean().item()
            grad_norm_text = (
                f" | grad_norm {grad_norm:.4f}"
                if grad_norm is not None else "")
            timing_text = (
                f" | {s_per_step:.2f} s/step"
                f" | eta {_format_eta((max_steps - optimizer_step) * s_per_step)}"
                if s_per_step is not None else "")
            logger.info(
                "step {}/{} | loss {:.4f}{}{} | micro steps {}",
                optimizer_step,
                max_steps,
                mean_loss,
                grad_norm_text,
                timing_text,
                micro_step,
            )
        if optimizer_step % cfg.training.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                optimizer_step,
                cfg.training.output_dir,
                ctx,
                fsdp_config=asdict(cfg.training.fsdp),
                metadata={
                    "epoch": epoch,
                    "micro_step": micro_step,
                },
                keep_last_n=cfg.training.keep_last_n,
                keep_every=cfg.training.keep_every,
            )
        accumulation_losses.clear()


def run_training(config_path: str) -> None:
    """训练主流程；所有 rank 在退出时清理分布式进程组。"""
    cfg = load_config(config_path)
    ctx = backend.init_device(cfg.device.type)
    amp_dtype = backend.get_amp_dtype(cfg.device.dtype)
    try:
        if ctx.is_main_process:
            logger.info(
                "设备: {} {} | rank {} | world size {} | amp {}",
                ctx.type.value,
                ctx.device,
                ctx.rank,
                ctx.world_size,
                amp_dtype,
            )

        random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)

        from neat_pi.data.lerobot_dataset import build_dataset
        from neat_pi.data.preprocessor import load_preprocessor

        dataset = build_dataset(cfg.data, cfg.model)
        loader, sampler = build_train_loader(dataset, cfg, ctx)
        if len(loader) == 0:
            raise ValueError(
                "数据集在当前 rank/batch_size 下没有可用训练 batch")
        if ctx.is_main_process:
            logger.info(
                "数据集: {} 帧 | rank batch {} | rank batches {}",
                len(dataset),
                cfg.data.batch_size,
                len(loader),
            )

        preprocessor = load_preprocessor(cfg)
        model = build_model(cfg, ctx, amp_dtype)
        if not cfg.training.use_dummy_model and cfg.training.pretrained:
            load_result = load_pi05_weights_distributed(
                model,
                cfg.training.pretrained,
                ctx,
            )
            if ctx.is_main_process:
                logger.info(
                    "checkpoint loaded: {} tensors | skipped {}",
                    load_result.loaded_count,
                    load_result.skipped_count,
                )
        model = wrap_model_fsdp(
            model,
            cfg.training.fsdp,
            ctx,
            amp_dtype,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        start_step = 0
        start_epoch = 0
        start_micro_step = 0
        if cfg.training.resume:
            checkpoint_path = latest_checkpoint_path(cfg.training.output_dir)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"resume=true 但没有可用 checkpoint: {cfg.training.output_dir}")
            start_step = load_checkpoint(
                model,
                optimizer,
                checkpoint_path,
            )
            checkpoint_metadata = load_checkpoint_metadata(checkpoint_path)
            start_epoch = int(checkpoint_metadata.get("epoch", 0))
            start_micro_step = int(checkpoint_metadata.get("micro_step", 0))
            if ctx.is_main_process:
                logger.info("恢复训练: {} | step {}", checkpoint_path, start_step)

        train(
            model=model,
            loader=loader,
            sampler=sampler,
            optimizer=optimizer,
            preprocessor=preprocessor,
            cfg=cfg,
            ctx=ctx,
            start_step=start_step,
            start_epoch=start_epoch,
            start_micro_step=start_micro_step,
        )
    finally:
        backend.cleanup_distributed()


def parse_args() -> argparse.Namespace:
    """解析命令行参数；只有 --config。"""
    parser = argparse.ArgumentParser(description="neat-pi 训练入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    return parser.parse_args()


def main() -> None:
    """训练入口。"""
    run_training(parse_args().config)


if __name__ == "__main__":
    main()
