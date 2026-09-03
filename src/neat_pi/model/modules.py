"""共享基础组件：RMSNorm / LayerNorm / AdaLayerNorm / MLP。

这些是 SigLIP、Gemma、动作专家都会用到的最小积木。
只实现 pi05 实际用到的形态，不做多余的可配置性。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neat_pi.typing import CondBD, TokensBTD, typechecked


class RMSNorm(nn.Module):
    """Gemma 风格 RMSNorm（无 bias）：weight 零初始化，输出 x_norm * (1 + weight)。

    对齐 ref/openpi modeling_gemma.py 的 GemmaRMSNorm（cond=None 分支）：
    归一化在 float32 下计算后转回原 dtype。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((1 + self.weight) * x).to(dtype)


class LayerNorm(nn.Module):
    """标准 LayerNorm（带 bias，均值/方差归一化），SigLIP 专用。

    对齐 ref/openpi modeling_siglip.py 的 nn.LayerNorm(eps=1e-6)。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        return F.layer_norm(x.float(), (x.shape[-1],), self.weight.float(),
                            self.bias.float(), self.eps).to(x.dtype)


class AdaLayerNorm(nn.Module):
    """adaLN-Zero 风格的自适应条件归一化（动作专家专用）：条件经 dense 生成 scale/shift/gate。

    命名为 AdaLayerNorm 而非 AdaRMSNorm：输出含 shift 项
    （x_norm * (1+scale) + shift），是仿射（LayerNorm 式）语义而非纯缩放；
    但归一化内核仍是无减均值的 RMS（对齐 openpi 的 GemmaRMSNorm cond 分支），
    保持数值与权重格式兼容，不改内核。

    与 openpi 参考实现的差异：
    - dense 的 weight 与 bias 都零初始化（openpi PyTorch 移植版只零 weight，
      bias 是随机值，破坏了恒等性；这里对齐 Flax 原版的真零初始化），
      使初始时刻 scale/shift/gate 全零、残差支路为恒等映射（adaLN-Zero）；
    - cond 为必传参数，无 openpi 那种 cond=None 的静默回退分支。

    gate 不进 norm 本身，由调用方在残差处消费：x = x + y * gate。
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    @typechecked
    def forward(self, x: TokensBTD, cond: CondBD) -> tuple[TokensBTD, TokensBTD]:
        """返回 (norm 后的 x, gate)；gate 形状 [batch, 1, dim]，供残差处广播相乘。"""
        dtype = x.dtype
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        modulation = self.dense(cond).unsqueeze(1).float()  # [batch, 1, 3*dim]
        scale, shift, gate = modulation.chunk(3, dim=-1)
        out = normed * (1 + scale) + shift
        return out.to(dtype), gate.to(dtype)


class MLP(nn.Module):
    """gate/up/down 三段式 MLP（Gemma 专用；SigLIP 用 fc1/fc2 形态，另见 siglip.py）。"""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
