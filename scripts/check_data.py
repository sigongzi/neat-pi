"""数据管线自检：加载 LIBERO 数据集，逐 processor 步过 preprocessor 打印形态。

用法：uv run scripts/check_data.py [config.yaml]

流程：build_dataset → DataLoader（默认 collate）→ 取一个 batch → 用
`PolicyProcessorPipeline.step_through` 逐步跑每个 processor step，每步后
打印全部字段的 shape/dtype/min/max/mean；最后对 token ids 做 decode 往返
对照，再迭代几个 batch 计时。
"""

from __future__ import annotations

import sys
import time
from typing import Any

import torch
from loguru import logger
from torch.utils.data import DataLoader

from neat_pi.config import load_config


def _tensor_stats(v: torch.Tensor) -> str:
    """把张量压缩成一行描述：shape/dtype/min/max/mean。"""
    parts = [f"shape={tuple(v.shape)}", str(v.dtype).replace("torch.", "")]
    try:
        parts.append(f"min={v.min().item():.3f} max={v.max().item():.3f}")
        if v.dtype.is_floating_point:
            parts.append(f"mean={v.float().mean().item():.3f}")
    except RuntimeError:
        pass
    return " ".join(parts)


def _print_batch(batch: dict[str, Any], indent: str = "  ") -> None:
    """按 key 打印一个 batch 的字段形态（张量/字符串/标量）。"""
    for key in sorted(batch):
        v = batch[key]
        if isinstance(v, torch.Tensor):
            logger.info("{}{:<36} {}", indent, key, _tensor_stats(v))
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            logger.info("{}{:<36} list[str] {!r}", indent, key, v[0][:100])
        elif isinstance(v, str):
            logger.info("{}{:<36} str {!r}", indent, key, v[:100])
        else:
            logger.info("{}{:<36} {}", indent, key, repr(v)[:100])


def main() -> None:
    """读取配置，取一个 batch 逐步过 preprocessor，打印每步输出形态。"""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/pi05_libero.yaml"
    cfg = load_config(config_path)

    from neat_pi.data.lerobot_dataset import build_dataset
    from neat_pi.data.preprocessor import load_preprocessor
    from neat_pi.data.tokenizer import GemmaTokenizerStep
    from neat_pi.data.transforms import images_to_float

    ds = build_dataset(cfg.data, cfg.model)
    logger.info("数据集: {} 帧", len(ds))

    dl = DataLoader(
        ds,
        batch_size=cfg.data.per_device_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    pipe = load_preprocessor(cfg)
    # 词表的唯一来源是 preprocessor JSON 中的 GemmaTokenizerStep，不在 YAML 重复配置
    tokenizer = next(
        (s.tokenizer for s in pipe.steps if isinstance(s, GemmaTokenizerStep)), None
    )
    if tokenizer is None:
        raise ValueError("preprocessor 管线中没有 GemmaTokenizerStep")

    batch = images_to_float(next(iter(dl)))
    logger.info("== 原始 batch（collate 后）==")
    _print_batch(batch)

    logger.info("== 逐步过 preprocessor ==")
    final_batch: dict[str, Any] = {}
    for step_idx, transition in enumerate(pipe.step_through(batch)):
        if step_idx == 0:
            logger.info("-- 输入 --")
        else:
            label = type(pipe.steps[step_idx - 1]).__name__
            logger.info("-- 第 {} 步: {} --", step_idx, label)
        view = pipe.to_output(transition)
        _print_batch(view)
        final_batch = view

    prompt = final_batch["task"][0]
    tokens = final_batch["observation.language.tokens"]
    logger.info("== 最终 token 对照 ==")
    logger.info("完整 prompt: {!r}", prompt)
    logger.info("decode 回:   {!r}", tokenizer.decode(tokens[0])[:200])

    logger.info("== 计时（再迭代 {} 个 batch，含视频解码 + 整条管线）==", 3)
    t0 = time.perf_counter()
    for _ in range(3):
        pipe(images_to_float(next(iter(dl))))
    dt = (time.perf_counter() - t0) / 3
    logger.info("平均每 batch 耗时: {:.2f}s", dt)


if __name__ == "__main__":
    main()
