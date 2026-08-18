"""共享基础组件：RMSNorm / RoPE / MLP / 多头注意力。

这些是 SigLIP、Gemma、动作专家都会用到的最小积木。
只实现 pi05 实际用到的形态，不做多余的可配置性。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neat_pi.typing import TokensBTD, typechecked


class RMSNorm(nn.Module):
    """Gemma 风格的 RMSNorm（无 bias，weight 初始为 0 还是 1 以 openpi 为准）。

    TODO: 核对 ref/openpi gemma.py 中 RMSNorm 的 (1+weight) 约定后对齐。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class MLP(nn.Module):
    """gate/up/down 三段式 MLP（Gemma / SigLIP 共用形态）。"""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


def precompute_rope_freqs(head_dim: int, max_seq_len: int,
                          base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """预计算 RoPE 的 cos/sin 频率表，形状均为 [max_seq_len, head_dim // 2]。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # [seq, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """对 [batch, heads, seq, head_dim] 的 q/k 施加旋转位置编码。"""

    def rotate_half(v: Tensor) -> Tensor:
        v1, v2 = v.chunk(2, dim=-1)
        return torch.cat([-v2, v1], dim=-1)

    # cos/sin: [seq, head_dim/2] -> 广播到 [1, 1, seq, head_dim/2]
    cos = torch.cat([cos, cos], dim=-1)[None, None]
    sin = torch.cat([sin, sin], dim=-1)[None, None]
    return x * cos + rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """普通多头自注意力（SDPA 实现）。

    注意：pi05 中 VLM 与动作专家的共享注意力不走这里，走 mot.py 的
    MoTAttention——那里按专家拆分 qkv/o 投影，便于 FSDP auto_wrap。
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    @typechecked
    def forward(self, x: TokensBTD, attn_mask: Tensor | None = None) -> TokensBTD:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # GQA：kv 头不足时广播到 q 头数
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(out)
