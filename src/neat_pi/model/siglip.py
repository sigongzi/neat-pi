"""SigLIP 视觉编码器（pi05 的图像侧）。

结构与权重命名对齐 ref/openpi/src/openpi/models/siglip.py（和 vit.py）。
多路相机图像各自过同一个 encoder，输出 token 序列后拼接。

权重布局（safetensors 中的前缀）以对齐 openpi checkpoint 为准，
具体映射见 weights.py。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.typing import ImageBCHW, TokensBTD, typechecked


class SigLIPVisionEncoder(nn.Module):
    """SigLIP ViT：patch embed -> N 层 transformer -> token 序列。

    TODO: 按 ref/openpi siglip.py 补全 patch embedding 与 transformer 堆叠；
    层数 / width / heads 使用 pi05 官方值（SigLIP-SO400m），不要自行发明。
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14,
                 width: int = 1152, num_layers: int = 27, num_heads: int = 16) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # patch embedding: Conv2d(channels, width, kernel=patch, stride=patch)
        self.patch_embed = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        # TODO: position embedding + encoder blocks + final norm

    @typechecked
    def forward(self, image: ImageBCHW) -> TokensBTD:
        """图像批次 -> [batch, (H/p)*(W/p), width] 的视觉 token 序列。"""
        raise NotImplementedError("待按 ref/openpi siglip.py 实现")
