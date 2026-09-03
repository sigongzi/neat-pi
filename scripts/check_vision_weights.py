"""视觉塔权重加载自检：加载 checkpoint 的 SigLIP 视觉塔，校验权重并跑前向。

用途：在 Pi05 整模型与 weights.py 完整映射（计划第 7 步）之前，先验证
siglip.py 的结构/命名与 checkpoint 视觉塔逐张量对齐。

用法：
    uv run scripts/check_vision_weights.py [--config configs/pi05_libero.yaml]
                                           [--checkpoint DIR] [--cpu]
                                           [--data] [--frames N] [--batch N]

- 默认用随机图像只做一次前向冒烟；
- 加 --data 则从 LIBERO 数据集取真实帧（多相机）跑前向，并打印跨相机 /
  跨帧的余弦相似度，用于确认加载的是真实训练的视觉编码器，而不是随机权重；
- 始终打印几组关键张量 + 全塔的权重统计（mean/std/min/max），用于肉眼核对
  「这不是全零/随机初始化的参数」。

checkpoint 目录默认取配置里的 training.pretrained。只读视觉塔 437 张量
（约 412M 参数），不整载入 7.5GB checkpoint。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from loguru import logger

from neat_pi.config import load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SigLIP 视觉塔权重加载自检")
    parser.add_argument("--config", default="configs/pi05_libero.yaml",
                        help="YAML 配置路径（取 device.type 与 training.pretrained）")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 目录（覆盖配置里的 training.pretrained）")
    parser.add_argument("--cpu", action="store_true",
                        help="强制在 CPU 上跑（覆盖配置里的 device.type）")
    parser.add_argument("--data", action="store_true",
                        help="用 LIBERO 数据集真实帧跑前向，并打印跨相机相似度")
    parser.add_argument("--frames", type=int, default=4,
                        help="--data 时采样的帧数")
    parser.add_argument("--stride", type=int, default=20,
                        help="--data 时相邻采样帧的索引间隔（跨不同场景）")
    parser.add_argument("--batch", type=int, default=1,
                        help="随机图像模式下的 batch 大小")
    return parser.parse_args()


def _report_weight_stats(model: torch.nn.Module, device: torch.device) -> None:
    """打印关键张量 + 全塔的权重统计，用于确认权重是训练过的真实参数。"""
    names = [
        "embeddings.patch_embedding.weight",
        "embeddings.position_embedding.weight",
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.mlp.fc1.weight",
        "encoder.layers.26.mlp.fc2.weight",
        "post_layernorm.weight",
    ]
    logger.info("== 关键张量统计（checkpoint 实际值）==")
    state = model.state_dict()
    for name in names:
        t = state[name].float()
        logger.info("{:<46} mean={:+.4f} std={:.4f} min={:+.4f} max={:+.4f}",
                    name, t.mean().item(), t.std().item(), t.min().item(),
                    t.max().item())

    # 全塔汇总（只在 CPU 上算，避免搬 412M 参数上卡）
    cpu = model.cpu()
    total = torch.zeros(())
    sq_sum = torch.zeros(())
    n = 0
    with torch.inference_mode():
        for p in cpu.parameters():
            f = p.detach().float()
            total += f.sum()
            sq_sum += (f * f).sum()
            n += f.numel()
    mean = total / n
    std = (sq_sum / n - mean * mean).clamp_min(0).sqrt()
    logger.info("== 全塔 {} 参数汇总 ==".format(f"{n:,}"))
    logger.info("mean={:+.6f} std={:.6f}", mean.item(), std.item())
    model.to(device)


def _run_images(model: torch.nn.Module, images: torch.Tensor,
                label: str) -> torch.Tensor:
    """跑一批图像，打印输出统计并返回 [batch, 256, 1152] 的视觉 token。"""
    with torch.inference_mode():
        out = model(images)
    logger.info("{}: 输出 shape={} dtype={} min={:.3f} max={:.3f} mean={:+.4f}",
                label, tuple(out.shape), out.dtype, out.min().item(),
                out.max().item(), out.float().mean().item())
    return out


def _check_real_frames(cfg, model: torch.nn.Module, device: torch.device,
                       dtype: torch.dtype, frames: int, stride: int) -> None:
    """从 LIBERO 取真实帧，跑两路相机并打印跨相机/跨帧余弦相似度。"""
    from neat_pi.data.lerobot_dataset import build_dataset

    ds = build_dataset(cfg.data, cfg.model)
    keys = sorted(k for k in ds[0] if k.startswith("observation.images."))
    logger.info("相机 key: {}", keys)

    indices = [i * stride for i in range(frames)]
    batches = []
    for idx in indices:
        sample = ds[idx]
        batches.append(torch.stack([sample[k] for k in keys], dim=0))
    images = torch.cat(batches, dim=0).to(device=device, dtype=dtype)  # [frames*C,3,224,224]
    out = _run_images(model, images, f"真实帧 {frames} x {len(keys)} 相机（索引 {indices}）")

    # 按帧分组（每 C 张为一组），算组内跨相机相似度 + 组间相似度。
    # 视觉 token 序列 [256, 1152] 先 L2 归一化再展平，用整体余弦度量图像嵌入。
    n_cam = len(keys)
    feats = F.normalize(out.float(), dim=-1)
    flat = feats.flatten(1)  # [N, 256*1152]
    intra = [
        F.cosine_similarity(flat[i * n_cam], flat[i * n_cam + 1], dim=0).item()
        for i in range(frames)
    ]
    logger.info("同帧两相机 embedding 余弦: {}", intra)
    # 不同帧的第一相机两两余弦
    first = flat[::n_cam]
    cross = [
        F.cosine_similarity(first[i], first[j], dim=0).item()
        for i in range(frames) for j in range(i + 1, frames)
    ]
    logger.info("不同帧相机0 两两余弦: {}", cross)


def main() -> None:
    """加载视觉塔权重，打印权重统计，并按模式跑一次前向。"""
    args = parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = args.checkpoint or cfg.training.pretrained
    if not checkpoint_dir:
        raise SystemExit("未指定 checkpoint：请用 --checkpoint 或配置 training.pretrained")

    from neat_pi.device import backend
    from neat_pi.model.siglip import SigLIPVisionEncoder
    from neat_pi.model.weights import load_siglip_vision_weights

    # 设备：--cpu 强制 CPU，否则按配置走 device 抽象（cuda/npu）。
    if args.cpu:
        device = torch.device("cpu")
        logger.info("设备: cpu")
    else:
        ctx = backend.init_device(cfg.device.type)
        device = ctx.device
        logger.info("设备: {}", device)

    model = SigLIPVisionEncoder()
    loaded = load_siglip_vision_weights(model, checkpoint_dir)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("已加载视觉塔 {} 张量，共 {:,} 参数", loaded, n_params)

    _report_weight_stats(model, device)

    dtype = (backend.get_amp_dtype(cfg.device.dtype)
             if device.type != "cpu" and cfg.device.dtype != "float32"
             else torch.float32)
    model = model.to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)

    if args.data:
        _check_real_frames(cfg, model, device, dtype, args.frames, args.stride)
    else:
        image = 2 * torch.randn(args.batch, 3, 224, 224, device=device, dtype=dtype) - 1
        _run_images(model, image.clip(-1, 1), "随机图像")


if __name__ == "__main__":
    main()
