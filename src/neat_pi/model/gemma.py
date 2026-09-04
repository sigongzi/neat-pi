"""Gemma 语言主干（pi05 的 VLM 侧）。

pi05 用 Gemma 系 decoder-only 作为 VLM 主干：文本 token embedding 与
视觉 token 拼接后进入 transformer 堆叠。

注意（FSDP / MoT 的关键点）：
pi05 里 VLM 主干与动作专家**共享同一套 attention 计算**（动作专家作为
另一个"专家"复用 attention 的 KV）。为让 FSDP 能按专家干净切分，
共享层不在这里实现，而在 mot.py 的 MoT 容器中按专家逐层配对组织：
GemmaDecoderLayer 只实现半块接口（pre_attn / post_attn，契约见 mot.py），
由 MoT 把两路专家拼进同一次 SDPA。

RoPE 与 GQA SDPA 的纯函数（rotate_half / apply_rotary_pos_emb /
build_rope_cache / gqa_sdpa）在 model/util.py，本文件只 import 不定义。

本文件提供：
- `GemmaAttention` / `GemmaDecoderLayer`：GQA 参数容器与单层 decoder
  （半块 pre_attn / post_attn + 独立 forward / prefill 形态）；
- `GemmaLM`：embedding -> N 层 -> final norm -> lm_head 的纯 decoder 主干，
  用于独立验证 checkpoint 里 VLM 主干的权重。它只接受 embedding（`LanguageTokens`）
  输入，不负责 token 层：Gemma 的 token embedding 与 lm_head 是 tied 的同一份
  `lm_head.weight`，由调用方查表并乘 sqrt(width) 得到。
- 推理 prefill 通路：`GemmaLM.prefill` / `GemmaDecoderLayer.prefill` 收集
  每层 RoPE 后的 (k, v) 进 `PrefixKVCache`（见 cache.py），不影响训练前向。

结构对齐 ref/openpi/src/openpi/models/gemma.py 与
models_pytorch/transformers_replace/models/gemma/modeling_gemma.py。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.model.cache import PrefixKVCache
from neat_pi.model.expert import Expert
from neat_pi.model.modules import MLP, RMSNorm
from neat_pi.model.util import apply_rotary_pos_emb, build_rope_cache, gqa_sdpa
from neat_pi.typing import (AttentionMaskBHLS, CondBD, GateB1D, LogitsBSV,
                            LanguageTokensBTD, typechecked)


class GemmaAttention(nn.Module):
    """Gemma GQA 自注意力：q/k/v/o 四投影（无 bias）+ RoPE + SDPA。

    默认 causal；``is_causal=False`` 时退化为全注意力（任意两位置互相可见）；
    传入 ``attn_mask`` 时以 mask 为准（SDPA 约定：bool 的 True 表示可见，
    或 float 加性 mask），形状可广播到 [batch, num_heads, seq, kv_len]。

    forward = project_qkv + attend 的组合；prefill 通路（见
    GemmaDecoderLayer.prefill）分开调这两步，以便把 RoPE 后、未广播的
    (k, v) 收集进 PrefixKVCache。attend 的广播与 SDPA 走共享的 gqa_sdpa。
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

    def project_qkv(self, x: LanguageTokensBTD, cos: torch.Tensor,
                    sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """qkv 投影 + RoPE，返回 (q, k, v)。

        q 形状 [batch, num_heads, seq, head_dim]；k/v 保持未广播的 kv 头
        原形态 [batch, num_kv_heads, seq, head_dim]（广播在 attend 里做，
        便于 prefill 按原形态缓存）。
        """
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               is_causal: bool = True,
               attn_mask: AttentionMaskBHLS | None = None) -> torch.Tensor:
        """gqa_sdpa + 私有 o_proj。

        k/v 为未广播原形态（[batch, num_kv_heads, kv_len, head_dim]），
        广播在 gqa_sdpa 内完成；q 的 seq 与 k/v 的 kv_len 可以不同（prefix +
        cache 场景）。attn_mask 非 None 时忽略 is_causal。
        """
        if attn_mask is not None:
            out = gqa_sdpa(q, k, v, self.num_heads, self.num_kv_heads,
                           self.scale, attn_mask=attn_mask)
        else:
            out = gqa_sdpa(q, k, v, self.num_heads, self.num_kv_heads,
                           self.scale, is_causal=is_causal)
        return self.o_proj(out)

    def forward(self, x: LanguageTokensBTD, cos: torch.Tensor,
                sin: torch.Tensor, is_causal: bool = True,
                attn_mask: AttentionMaskBHLS | None = None) -> LanguageTokensBTD:
        """[batch, seq, dim] -> [batch, seq, dim]；is_causal=False 时为全注意力。"""
        q, k, v = self.project_qkv(x, cos, sin)
        return self.attend(q, k, v, is_causal=is_causal, attn_mask=attn_mask)


class GemmaDecoderLayer(nn.Module):
    """单层 Gemma decoder：VLM 专家 block（半块接口见 mot.py 模块 docstring）。

    - pre_attn：input_layernorm（纯 RMSNorm，无时间条件）-> 私有 q/k/v 投影
      + RoPE；
    - post_attn：私有 o_proj + 残差 + post norm + MLP + 残差（VLM 无门控，
      state 的第二元素恒为 None）。
    forward / prefill 是这两个半块与一次本段自注意力（gqa_sdpa）的组合，
    供独立运行的 GemmaLM（文本生成验证）与推理 prefill 通路使用；MoT 的
    fused 前向不调 forward，而是拿 pre_attn 的 q/k/v 跨专家拼接后做共享
    SDPA，再交回本层 post_attn。
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, mlp_hidden: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.self_attn = GemmaAttention(dim, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.mlp = MLP(dim, mlp_hidden)

    def pre_attn(self, x: LanguageTokensBTD, cos: torch.Tensor,
                 sin: torch.Tensor,
                 cond: CondBD | None = None,
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                            tuple[LanguageTokensBTD, None]]:
        """半块 pre：pre-norm + 私有 q/k/v 投影 + RoPE。

        cond 对 VLM 侧无用（动作专家的 AdaLayerNorm 才消费时间条件），仅
        占位以统一 MoT 的调用签名。返回 (q, k, v, state)：q 形状
        [batch, num_heads, seq, head_dim]，k/v 未广播
        [batch, num_kv_heads, seq, head_dim]（RoPE 已施加）；state=(x, None)
        由本层 post_attn 做残差（VLM 无门控）。
        """
        normed = self.input_layernorm(x)
        q, k, v = self.self_attn.project_qkv(normed, cos, sin)
        return q, k, v, (x, None)

    def post_attn(self, state: tuple[LanguageTokensBTD, GateB1D | None],
                  attn_out: torch.Tensor,
                  cond: CondBD | None = None) -> LanguageTokensBTD:
        """半块 post：私有 o_proj + 残差 + post norm + MLP + 残差（无门控）。

        attn_out 为共享 SDPA 输出切回本段的 [batch, seq, num_heads*head_dim]
        片段；state 是 pre_attn 返回的 (pre-norm 前的 x, gate=None)，x 作
        残差、gate 恒等。
        """
        residual, _ = state
        x = residual + self.self_attn.o_proj(attn_out)
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(self, x: LanguageTokensBTD, cos: torch.Tensor,
                sin: torch.Tensor, is_causal: bool = True,
                attn_mask: AttentionMaskBHLS | None = None) -> LanguageTokensBTD:
        """本段自注意力版本的单层计算（独立运行用）。

        = pre_attn -> 一次自注意力（gqa_sdpa，attn_mask 非 None 时忽略
        is_causal）-> post_attn，数值与 MoT fused 前向中本专家的行一致
        （本段行不 attend 其它专家列时）。
        """
        q, k, v, state = self.pre_attn(x, cos, sin)
        if attn_mask is not None:
            attn_out = gqa_sdpa(q, k, v, self.self_attn.num_heads,
                                self.self_attn.num_kv_heads,
                                self.self_attn.scale, attn_mask=attn_mask)
        else:
            attn_out = gqa_sdpa(q, k, v, self.self_attn.num_heads,
                                self.self_attn.num_kv_heads,
                                self.self_attn.scale, is_causal=is_causal)
        return self.post_attn(state, attn_out)

    def prefill(self, x: LanguageTokensBTD, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: AttentionMaskBHLS | None = None,
                ) -> tuple[LanguageTokensBTD, tuple[torch.Tensor, torch.Tensor]]:
        """与 forward 相同的层计算，但额外返回本层 RoPE 后的 (k, v)。

        k/v 为未广播的 kv 头原形态 [batch, num_kv_heads, seq, head_dim]，
        供 PrefixKVCache 缓存（推理时动作专家 attend prefix 用，见
        action_expert.DiTBlock 的 vlm_kv 入参）。
        """
        q, k, v, state = self.pre_attn(x, cos, sin)
        attn_out = gqa_sdpa(q, k, v, self.self_attn.num_heads,
                            self.self_attn.num_kv_heads,
                            self.self_attn.scale, is_causal=False,
                            attn_mask=attn_mask)
        return self.post_attn(state, attn_out), (k, v)


class GemmaLM(Expert):
    """Gemma decoder 主干：输入 embedding，输出 logits。

    作为 MoT 的 VLM 专家，须满足 Expert 契约：自带完整 18 层堆叠 layers
    （每层 GemmaDecoderLayer，实现 pre_attn / post_attn 半块接口），顶层
    暴露 num_heads / num_kv_heads / attn_head_dim / theta 供 MoT 校验与透传。

    默认超参数对应 pi05 的 gemma_2b：width 2048 / 18 层 / 8 头 / 1 kv 头 /
    attn_head_dim 256 / mlp 16384 / vocab 257152。token embedding 不在本模块里，
    由调用方查 tied 的 lm_head.weight 并乘 sqrt(width)。
    """

    def __init__(self, vocab_size: int = 257_152, width: int = 2048,
                 num_layers: int = 18, num_heads: int = 8,
                 num_kv_heads: int = 1, attn_head_dim: int = 256,
                 mlp_hidden: int = 16384, theta: float = 10_000.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.width = width
        self.attn_head_dim = attn_head_dim
        self.theta = theta
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.layers = nn.ModuleList(
            GemmaDecoderLayer(width, num_heads, num_kv_heads, attn_head_dim,
                              mlp_hidden, eps)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(width, eps)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)

    @typechecked
    def forward(self, x: LanguageTokensBTD) -> LogitsBSV:
        """embedding [batch, seq, dim] -> logits [batch, seq, vocab]。

        只做 transformer + final norm + lm_head；token embedding 由调用方负责
        （tied 到 lm_head.weight 并乘 sqrt(width)，与 openpi 的 pi0_pytorch
        embed_prefix / JAX Embedder.encode 一致）。
        """
        cos, sin = build_rope_cache(x.shape[1], self.attn_head_dim, self.theta,
                                    x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    @typechecked
    def prefill(self, x: LanguageTokensBTD,
                attn_mask: AttentionMaskBHLS | None = None) -> PrefixKVCache:
        """prefix prefill：embedding 过全部层，收集每层 RoPE 后的 (k, v)。

        返回 PrefixKVCache（元素形状见 cache.py），供推理时动作专家的
        DiTBlock 复用（对应 lerobot pi05 sample_actions 的
        use_cache=True 一次 prefill、N 步去噪复用）。attn_mask 为 prefix
        内的可见性 mask（[batch, 1, seq, seq]，bool True=可见或 float 加性），
        由调用方按 pi05 的 prefix 块语义构造；None 表示全可见。
        不过 final norm / lm_head——cache 只需要各层 k/v。
        """
        cos, sin = build_rope_cache(x.shape[1], self.attn_head_dim, self.theta,
                                    x.device, x.dtype)
        layer_kvs = []
        for layer in self.layers:
            x, kv = layer.prefill(x, cos, sin, attn_mask)
            layer_kvs.append(kv)
        return PrefixKVCache(layer_kvs)
