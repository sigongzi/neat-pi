"""图像与动作变换。

pi05 的预处理约定（以 ref/openpi 为准）：
- 图像：resize 到 image_size，归一化到 [-1, 1]（SigLIP 的输入约定）；
- 动作/状态：用数据集统计量做归一化（quantile 或 mean/std，
  统计量从 lerobot 数据集 meta 读取）。
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T

from neat_pi.typing import ActionBHD, ImageBCHW, typechecked


def build_image_transform(image_size: int) -> T.Compose:
    """构造图像预处理：resize + [-1, 1] 归一化。"""
    return T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),  # [0,1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
    ])


@typechecked
def normalize_actions(actions: ActionBHD, mean: torch.Tensor,
                      std: torch.Tensor) -> ActionBHD:
    """按数据集统计量归一化动作（占位实现，统计量口径以 openpi 为准）。"""
    return (actions - mean) / (std + 1e-8)
