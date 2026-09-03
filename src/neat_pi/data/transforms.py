"""图像变换与 batch 整理。

pi05 的图像输入约定（以 ref/openpi 为准）：resize 到 image_size，归一化到
[-1, 1]（SigLIP 的输入约定）。动作/状态的归一化不在本模块——由 lerobot
processor 管线的 normalizer step 负责（见 data/preprocessor.py）。
"""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as T


def build_image_transform(image_size: int) -> T.Compose:
    """构造图像预处理：resize + [-1, 1] 归一化。"""
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),  # [0,1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
    ])


def images_to_float(batch: dict[str, Any]) -> dict[str, Any]:
    """把仍为 uint8 的图像转成 float/255；已是 float（数据集侧归一化过）则原样保留。"""
    out = dict(batch)
    for key, v in out.items():
        if key.startswith("observation.images.") and isinstance(v, torch.Tensor) \
                and v.dtype == torch.uint8:
            out[key] = v.float() / 255.0
    return out
