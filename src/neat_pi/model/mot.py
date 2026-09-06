"""MoT（Mixture of Transformers）：VLM 与动作专家共享注意力的解耦容器。

Pi05 的 VLM prefix 与动作段在每个 transformer 层共享同一次 attention：
两路 token 各自过私有 q/k/v 投影，然后在序列维拼接进同一次 SDPA，再拆回
各自区间执行 o_proj / 残差 / MLP。两路参数互不共享。

``MoTFusedLayer`` 是 FSDP 的所有权边界：同一层同时拥有 VLM 与 action
block，并把一次共享 attention 收敛在自己的 forward 中。``MoT`` 只按序调用
这些 fused layer，从而使 FSDP 可以按 layer 粒度干净分片。

Expert block 的半块接口与语义：

- ``pre_attn(x, cos, sin, cond=None) -> (q, k, v, state)``
  完成 pre-norm / 时间条件注入、私有 q/k/v 投影与 RoPE；
- ``post_attn(state, attn_out, cond=None) -> x``
  完成私有 o_proj、残差、post-norm 与 MLP。

两个半块的独立形态由 block 的 ``forward`` 保持，供单块测试、调试与推理
run_layers / prefill 使用；其数值与 fused 前向中该专家的行一致。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.model.expert import Expert
from neat_pi.model.util import build_rope_cache, gqa_sdpa
from neat_pi.typing import ActionTokensBHD, CondBD, LanguageTokensBTD


class MoTFusedLayer(nn.Module):
    """拥有同一层所有 Expert block 的 fused attention 层。

    这是 FSDP 的推荐所有权边界：forward 内完成所有 Expert 的 pre_attn、
    共享 SDPA 和 post_attn，不需要绕过 FSDP wrapper 访问参数。
    """

    def __init__(
        self,
        blocks: dict[str, nn.Module],
        num_heads: int,
        num_kv_heads: int,
        attn_head_dim: int,
        theta: float,
    ) -> None:
        """按专家名注册同层 blocks，并保存共享 attention 结构参数。"""
        super().__init__()
        if not blocks:
            raise ValueError("MoTFusedLayer 至少需要一个 Expert block")
        self.expert_names = tuple(blocks)
        for expert_name, block in blocks.items():
            self.add_module(expert_name, block)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.attn_head_dim = attn_head_dim
        self.theta = theta

    def forward(
        self,
        tokens: dict[str, torch.Tensor],
        rope: dict[str, tuple[torch.Tensor, torch.Tensor]],
        attention_mask: torch.Tensor | None,
        conds: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """执行一层 fused MoT，返回各路更新后的 token。"""
        seq_lens = {name: tokens[name].shape[1] for name in self.expert_names}
        q_chunks: list[torch.Tensor] = []
        k_chunks: list[torch.Tensor] = []
        v_chunks: list[torch.Tensor] = []
        pending: dict[str, tuple[nn.Module, object]] = {}
        for name in self.expert_names:
            block = getattr(self, name)
            cos, sin = rope[name]
            q, k, v, state = block.pre_attn(
                tokens[name], cos, sin, conds.get(name))
            q_chunks.append(q)
            k_chunks.append(k)
            v_chunks.append(v)
            pending[name] = (block, state)

        attn_out = gqa_sdpa(
            torch.cat(q_chunks, dim=2),
            torch.cat(k_chunks, dim=2),
            torch.cat(v_chunks, dim=2),
            self.num_heads,
            self.num_kv_heads,
            self.attn_head_dim ** -0.5,
            attn_mask=attention_mask,
        )

        start = 0
        for name in self.expert_names:
            end = start + seq_lens[name]
            block, state = pending[name]
            tokens[name] = block.post_attn(
                state, attn_out[:, start:end], conds.get(name))
            start = end
        return tokens


class MoT(nn.Module):
    """把多个 Expert 组合成由 `MoTFusedLayer` 构成的共享注意力堆叠。"""

    def __init__(
        self,
        mixtures: dict[str, Expert]
    ) -> None:
        """校验 Expert 结构，接管各路 layers 并组成配对 fused stack。"""
        super().__init__()
        Expert.assert_compatible(mixtures)
        self._expert_name = tuple(mixtures)
        # MoT 不能用 ModuleDict 再次注册完整 Expert：那会让 final norm /
        # projection 与 Pi05 顶层的 Expert 路径重复。这里只保留非注册引用。
        self._mixtures = dict(mixtures)
        owned_layers = {
            name: expert.take_owned_layers()
            for name, expert in mixtures.items()
        }
        layer_counts = {
            name: len(layers) for name, layers in owned_layers.items()
        }
        if len(set(layer_counts.values())) != 1:
            raise ValueError(f"各 Expert 层数不一致: {layer_counts}")
        num_layers = next(iter(layer_counts.values()))
        first = mixtures[self._expert_name[0]]
        self.layers = nn.ModuleList([
            MoTFusedLayer(
                {
                    name: owned_layers[name][layer_idx]
                    for name in self._expert_name
                },
                first.num_heads,
                first.num_kv_heads,
                first.attn_head_dim,
                first.theta,
            )
            for layer_idx in range(num_layers)
        ])
        for name, expert in mixtures.items():
            expert.bind_layer_view(self, name)
        self.num_layers = len(self.layers)
        self.num_heads = first.num_heads
        self.num_kv_heads = first.num_kv_heads
        self.attn_head_dim = first.attn_head_dim
        self.theta = first.theta

    def forward(
        self,
        embeds_dict: dict[str, LanguageTokensBTD | ActionTokensBHD],
        attention_mask: torch.Tensor | None = None,
        conds: dict[str, CondBD | None] | None = None,
    ) -> dict[str, LanguageTokensBTD | ActionTokensBHD]:
        """逐层执行 fused MoT，返回各路更新后的 token。

        ``attention_mask`` 是整段 [sum(seq), sum(seq)] 的可见性（bool 的
        True 表示可见，或 float 加性 mask）。Expert final norm 不在这里
        执行，仍由 Pi05 在 action 段外侧调用。
        """
        missing = [
            name for name in self._expert_name if name not in embeds_dict
        ]
        if missing:
            raise ValueError(f"embeds_dict 缺少专家 token：{missing}")

        seq_lens = {
            name: embeds_dict[name].shape[1] for name in self._expert_name
        }
        offset: dict[str, int] = {}
        total = 0
        for name in self._expert_name:
            offset[name] = total
            total += seq_lens[name]
        if attention_mask is not None and (
                attention_mask.ndim not in (2, 4)
                or attention_mask.shape[-2] != total
                or attention_mask.shape[-1] != total):
            raise ValueError(
                "attention_mask 须为 [S, S]（或 [B, 1|H, S, S]），"
                f"S = 各专家序列长度之和 {total}，"
                f"实际 {tuple(attention_mask.shape)}")

        first = embeds_dict[self._expert_name[0]]
        cos_full, sin_full = build_rope_cache(
            total,
            self.attn_head_dim,
            self.theta,
            first.device,
            first.dtype,
        )
        rope = {
            name: (
                cos_full[offset[name]:offset[name] + seq_lens[name]],
                sin_full[offset[name]:offset[name] + seq_lens[name]],
            )
            for name in self._expert_name
        }
        conds = conds or {}
        tokens = dict(embeds_dict)
        for layer in self.layers:
            tokens = layer(tokens, rope, attention_mask, conds)
        return tokens

    def extra_repr(self) -> str:
        """显示 fused layer 数量和专家顺序，避免重复展开所有子模块。"""
        return (f"layers={self.num_layers}, "
                f"experts={','.join(self._expert_name)}")
