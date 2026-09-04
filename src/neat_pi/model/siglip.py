"""SigLIP 视觉编码器（pi05 的图像侧）。

结构与权重命名对齐 ref/openpi/src/openpi/models_pytorch/
transformers_replace/models/siglip/modeling_siglip.py，规格取 pi05 官方
SigLIP-SO400m：width 1152 / 27 层 / 16 头 / patch 14 / MLP 4304。

checkpoint 中的视觉 tower 只到 post_layernorm（无 pooling head），输出
256 个 1152 维 token，后续 multi_modal_projector 把 1152 投影到 2048
（该投影不在本文件，见 gemma.py / pi05.py 的组装处）。

本文件刻意保持参数命名与 checkpoint 一致（patch_embedding /
position_embedding / encoder.layers.{i}.layer_norm1 / self_attn /
mlp.fc1/fc2 / post_layernorm），这样 weights.py 的名字映射可以直接
按结构名对应，无需额外翻译。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neat_pi.model.modules import LayerNorm
from neat_pi.typing import ImageBCHW, VisionTokensBTD, typechecked


class SiglipMLP(nn.Module):
    """SigLIP 两段式 MLP：fc1 -> GELU(tanh) -> fc2，两段都带 bias。

    与 Gemma 的 gate/up/down 三段式 MLP 不同（见 modules.MLP），这里
    对齐 modeling_siglip.py 的 SiglipMLP。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    @typechecked
    def forward(self, x: VisionTokensBTD) -> VisionTokensBTD:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipAttention(nn.Module):
    """SigLIP 多头上文无关自注意力（q/k/v/o 四投影 + 缩放点积）。"""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    @typechecked
    def forward(self, x: VisionTokensBTD) -> VisionTokensBTD:
        """[batch, seq, dim] -> [batch, seq, dim]，无 causal mask。"""
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.out_proj(out)


class SiglipEncoderLayer(nn.Module):
    """单层 SigLIP transformer：pre-norm 自注意力 + pre-norm MLP，残差相加。"""

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(dim, eps)
        self.self_attn = SiglipAttention(dim, num_heads)
        self.layer_norm2 = LayerNorm(dim, eps)
        self.mlp = SiglipMLP(dim, mlp_hidden)

    @typechecked
    def forward(self, x: VisionTokensBTD) -> VisionTokensBTD:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SiglipVisionEmbeddings(nn.Module):
    """视觉塔输入端：patch 卷积嵌入 + learned position embedding。"""

    def __init__(self, image_size: int, patch_size: int, width: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, width, kernel_size=patch_size,
                                         stride=patch_size)
        self.position_embedding = nn.Embedding(self.num_patches, width)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand(1, -1),
            persistent=False,
        )

    @typechecked
    def forward(self, image: ImageBCHW) -> VisionTokensBTD:
        """图像批次 -> [batch, num_patches, width]。"""
        embeds = self.patch_embedding(image)  # [batch, width, grid, grid]
        embeds = embeds.flatten(2).transpose(1, 2)  # [batch, num_patches, width]
        return embeds + self.position_embedding(self.position_ids)


class SiglipEncoder(nn.Module):
    """SigLIP transformer 堆叠：N 个 SiglipEncoderLayer。"""

    def __init__(self, width: int, num_layers: int, num_heads: int,
                 mlp_hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SiglipEncoderLayer(width, num_heads, mlp_hidden, eps)
            for _ in range(num_layers)
        )

    @typechecked
    def forward(self, x: VisionTokensBTD) -> VisionTokensBTD:
        for layer in self.layers:
            x = layer(x)
        return x


class SigLIPVisionEncoder(nn.Module):
    """SigLIP 视觉塔：embeddings -> 27 层 encoder -> post layernorm。

    输出 [batch, (image_size/patch_size)^2, width] 的视觉 token 序列，
    不含 pooling head 与 multi_modal_projector（对照 checkpoint）。

    子模块层级刻意与 checkpoint 的 `vision_tower.vision_model.*` 相对路径
    对齐（embeddings.patch_embedding / encoder.layers.{i}.* /
    post_layernorm），weights.py 翻译时只需剥离前缀、无需逐参数改名。
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14,
                 width: int = 1152, num_layers: int = 27,
                 num_heads: int = 16, mlp_hidden: int = 4304,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width

        self.embeddings = SiglipVisionEmbeddings(image_size, patch_size, width)
        self.encoder = SiglipEncoder(width, num_layers, num_heads, mlp_hidden,
                                     eps)
        self.post_layernorm = LayerNorm(width, eps)

    @typechecked
    def forward(self, image: ImageBCHW) -> VisionTokensBTD:
        """图像批次 -> [batch, num_patches, width] 的视觉 token 序列。"""
        hidden = self.embeddings(image)
        hidden = self.encoder(hidden)
        return self.post_layernorm(hidden)
