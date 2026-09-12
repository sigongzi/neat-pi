"""batch 字段形态打印工具（check_data / check_preprocessor 自检脚本共用）。

按 key 打印一个 batch 的 shape/dtype/min/max/mean 概览，用于人工核对
数据集 collate 输出与 preprocessor 各 step 的字段变化。
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger


def tensor_stats(v: torch.Tensor) -> str:
    """把张量压缩成一行描述：shape/dtype/min/max/mean。"""
    parts = [f"shape={tuple(v.shape)}", str(v.dtype).replace("torch.", "")]
    try:
        parts.append(f"min={v.min().item():.3f} max={v.max().item():.3f}")
        if v.dtype.is_floating_point:
            parts.append(f"mean={v.float().mean().item():.3f}")
    except RuntimeError:
        pass
    return " ".join(parts)


def print_batch(batch: dict[str, Any], indent: str = "  ") -> None:
    """按 key 打印一个 batch 的字段形态（张量/字符串/标量）。"""
    for key in sorted(batch):
        v = batch[key]
        if isinstance(v, torch.Tensor):
            logger.info("{}{:<36} {}", indent, key, tensor_stats(v))
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
            logger.info("{}{:<36} list[str] {!r}", indent, key, v[0][:100])
        elif isinstance(v, str):
            logger.info("{}{:<36} str {!r}", indent, key, v[:100])
        else:
            logger.info("{}{:<36} {}", indent, key, repr(v)[:100])
