"""无参数函数式积木：RoPE 旋转位置编码与 GQA 共享 SDPA。

这些纯函数跨多个模型模块共用（GemmaDecoderLayer / DiTBlock 的本段自注意力、
MoT 的跨专家共享 SDPA、推理与检查脚本的 RoPE 缓存），集中在本文件，结构模块
只 import 不定义，避免跨文件引用的函数散落在各处。本模块不持有任何参数，
也不 import 其它 neat_pi.model 模块（无环）。

- rotate_half / apply_rotary_pos_emb：对 q/k 施加 Gemma 默认 RoPE；
- build_rope_cache：按 theta 生成 [seq, head_dim] 的 cos/sin 缓存；
- gqa_sdpa：GQA kv 头展开 + 一次 SDPA，返回 [batch, seq, num_heads*head_dim]。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE 的 rotate_half：后半段取反换到前半段。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor,
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """对 query/key 施加 RoPE；cos/sin 形状 [seq, head_dim]。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device, dtype: torch.dtype,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """按 Gemma 默认 RoPE 生成 [seq, head_dim] 的 cos/sin 缓存。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    emb = torch.cat((torch.outer(t, inv_freq), torch.outer(t, inv_freq)), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def gqa_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """GQA 展开 + 一次 SDPA，返回 [batch, seq, num_heads * head_dim]。

    q 形状 [batch, num_heads, seq, head_dim]；k/v 为未广播的 kv 头原形态
    [batch, num_kv_heads, kv_len, head_dim]，这里按 (K, G) 顺序展开到 q 头
    （G = num_heads // num_kv_heads，与 openpi 的 q reshape "B T (K G) H"
    约定一致），再做单次 SDPA 并还原为 [batch, seq, D]。

    MoT 在跨专家拼接序列上调用它（一次共享 SDPA），各 block 的本段自注意力
    （GemmaDecoderLayer.forward / DiTBlock.forward）也调用它——广播与 SDPA
    只有这一份实现。attn_mask 非 None 时忽略 is_causal。
    """
    batch, _, _, head_dim = q.shape
    kv_len = k.shape[2]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) 必须是 num_kv_heads ({num_kv_heads}) 的整数倍")
    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        k = k[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, kv_len,
                                       head_dim).reshape(batch, num_heads,
                                                         kv_len, head_dim)
        v = v[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, kv_len,
                                       head_dim).reshape(batch, num_heads,
                                                         kv_len, head_dim)
    if attn_mask is not None:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             scale=scale)
    else:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                             scale=scale)
    return out.transpose(1, 2).reshape(batch, q.shape[2], -1)
