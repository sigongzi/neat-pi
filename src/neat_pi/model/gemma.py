"""Gemma 语言主干（pi05 的 VLM 侧）。

pi05 用 Gemma 系 decoder-only 作为 VLM 主干：文本 token embedding 与
视觉 token 拼接后进入 transformer 堆叠。

注意（FSDP / MoT 的关键点）：
pi05 里 VLM 主干与动作专家**共享同一套 attention 计算**（动作专家作为
另一个"专家"复用 attention 的 KV）。为让 FSDP 能按专家干净切分，
共享层不在这里实现，而在 mot.py 的 MoTLayer 中按专家组织。

本文件提供两样东西：
- `GemmaEmbedding`：独立的 token embedding（pi05.py 组装用的占位，仅做
  embedding lookup，结构见 pi05.py 阶段再定）；
- `GemmaLM`：**纯文本**的完整 Gemma decoder（embedding + N 层 + final norm +
  lm_head + 贪心生成），用于独立验证 checkpoint 里 VLM 主干的权重与对话能力。
  Gemma 的 token embedding 与 lm_head 是 tied 的（同一份 `lm_head.weight`
  既做输入查表又做输出头），所以这里没有独立的 embed_tokens 参数。

结构对齐 ref/openpi/src/openpi/models/gemma.py 与
models_pytorch/transformers_replace/models/gemma/modeling_gemma.py。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from neat_pi.model.modules import MLP, RMSNorm
from neat_pi.typing import LogitsBSV, TokenIdsBL, TokensBTD, typechecked


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


class GemmaAttention(nn.Module):
    """Gemma GQA 自注意力：q/k/v/o 四投影（无 bias）+ RoPE + SDPA。

    默认 causal；``is_causal=False`` 时退化为全注意力（任意两位置互相可见）。
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x: TokensBTD, cos: torch.Tensor,
                sin: torch.Tensor, is_causal: bool = True) -> TokensBTD:
        """[batch, seq, dim] -> [batch, seq, dim]；is_causal=False 时为全注意力。"""
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # kv 头广播到 q 头数（num_heads 是 num_kv_heads 的整数倍）
        n_rep = self.num_heads // self.num_kv_heads
        k = k[:, :, None, :, :].expand(batch, self.num_kv_heads, n_rep, seq,
                                       self.head_dim).reshape(batch, self.num_heads, seq, self.head_dim)
        v = v[:, :, None, :, :].expand(batch, self.num_kv_heads, n_rep, seq,
                                       self.head_dim).reshape(batch, self.num_heads, seq, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=self.scale)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.o_proj(out)


class GemmaDecoderLayer(nn.Module):
    """单层 Gemma decoder：pre-norm 注意力 + pre-norm MLP，残差相加。"""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, mlp_hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.self_attn = GemmaAttention(dim, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.mlp = MLP(dim, mlp_hidden)

    def forward(self, x: TokensBTD, cos: torch.Tensor,
                sin: torch.Tensor, is_causal: bool = True) -> TokensBTD:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, is_causal)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class GemmaLM(nn.Module):
    """纯文本 Gemma decoder-only 语言模型（embedding 与 lm_head tied）。

    默认超参数对应 pi05 的 gemma_2b：width 2048 / 18 层 / 8 头 / 1 kv 头 /
    head_dim 256 / mlp 16384 / vocab 257152。
    """

    def __init__(self, vocab_size: int = 257_152, width: int = 2048,
                 num_layers: int = 18, num_heads: int = 8,
                 num_kv_heads: int = 1, head_dim: int = 256,
                 mlp_hidden: int = 16384, theta: float = 10_000.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.width = width
        self.head_dim = head_dim
        self.theta = theta
        self.layers = nn.ModuleList(
            GemmaDecoderLayer(width, num_heads, num_kv_heads, head_dim,
                              mlp_hidden, eps)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(width, eps)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    @typechecked
    def forward(self, input_ids: TokenIdsBL) -> LogitsBSV:
        """token ids -> [batch, seq, vocab] logits。

        embedding 直接查 lm_head.weight（tied），不乘 sqrt(width)：
        checkpoint 由 openpi PyTorch 移植版保存，其建模里 normalizer 被关闭。
        """
        x = F.embedding(input_ids, self.lm_head.weight)
        cos, sin = build_rope_cache(input_ids.shape[1], self.head_dim,
                                    self.theta, input_ids.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: TokenIdsBL, max_new_tokens: int = 64,
                 eos_id: int = 1) -> TokenIdsBL:
        """贪心解码：逐 token 取 argmax，遇到 eos 或达到上限即停。

        每步重算整段序列（不实现 KV cache），用于短序列验证已足够。
        """
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_id = logits[0, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id[None]], dim=1)
            if next_id.item() == eos_id:
                break
        return input_ids


class GemmaEmbedding(nn.Module):
    """独立的 Gemma token embedding（pi05.py 组装用的占位）。

    注意：checkpoint 里 token embedding 与 lm_head 是 tied 的同一份权重，
    真实文本路径应走 GemmaLM（见上）；本类仅用于 pi05.py 当前的空壳组装，
    最终组装（计划第 6 步）再决定是复用 lm_head 还是保留独立 embedding。
    """

    def __init__(self, vocab_size: int = 257_152, width: int = 2048) -> None:
        super().__init__()
        self.width = width
        self.embed_tokens = nn.Embedding(vocab_size, width)

    @typechecked
    def forward(self, token_ids: TokenIdsBL) -> TokensBTD:
        """token id -> embedding。"""
        return self.embed_tokens(token_ids) * (self.width ** 0.5)
