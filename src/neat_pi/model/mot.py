"""MoT（Mixture of Transformers）：VLM 与动作专家共享注意力的解耦容器。

为什么需要这个文件（本项目 FSDP 目标的核心）：

pi05 原生结构里，VLM（图像+文本）与动作专家在每个 transformer 层共享同一次
attention 计算——两路 token 各自过自己的 q/k/v 投影后，在序列维拼接进同一次
SDPA，再拆回各自区间做 o_proj / 残差 / MLP（见 ref/openpi pi0.py 的
compute_loss 与 gemma.py 的 Attention / Block，以及 docs/plan/
02-model-implementation.md 的语义确认），但两套参数互不共享。若把这种耦合
写成跨模块张量传递，FSDP auto_wrap 无法干净地按层分片。本容器把"逐层配对 +
跨专家共享 SDPA"的循环收在 MoT 里：各专家是参数完整的子模块（见
model/expert.py 的 Expert 契约，作为 mixture 传入），跨专家只流动激活、
不碰参数。逐层组织即最终的 fused 前向，auto_wrap 候选粒度见
training/fsdp.py 的 TODO。

专家 block 的半块接口（**必须实现**，本文件 forward 依赖；写在各 block 的
方法注释与 expert.py 的 Expert 契约上）：

每个专家的 ``layers[i]``（GemmaDecoderLayer / DiTBlock）实现两个"半块"方法，
norm / 门控残差 / 时间条件注入的差异全部留在各自 block 内——MoT 不感知专家
类型、不按名字特判，也**不**按 openpi 的做法把 norm 统一成"cond=None 时返回
gate=1 的 RMSNorm"（本仓库两个专家的 norm 形态天然不同，接口统一落在这层
半块上）：

- ``pre_attn(x, cos, sin, cond=None) -> (q, k, v, state)``
  完成本层 pre-norm（VLM 用纯 RMSNorm；动作专家用 AdaLayerNorm，注入时间
  条件 cond）与私有 q/k/v 投影 + RoPE。q 形状 [B, H, S, Dh]；k/v 保持
  未广播的 kv 头原形态 [B, Kv, S, Dh]；state 是 block 私有、内容对 MoT
  不透明（VLM 侧为 (x, None) 无门控；动作侧携带 adaLN gate），由**同一**
  block 的 post_attn 消费。
- ``post_attn(state, attn_out, cond=None) -> x``
  attn_out 是共享 SDPA 输出按本专家切回、已 reshape 成 [B, S, H*Dh] 的
  片段；完成私有 o_proj、残差（动作侧按 pre 的 gate 门控）、post-norm 与
  MLP。

两个半块的独立形态（不拼其它专家，各自一段自注意力）由 block 的
``forward`` 保持，供单块测试/调试与推理 run_layers / prefill 使用，数值与
fused 前向里该专家的行一致。

逐层 fused 前向（forward）的语义（对齐 openpi compute_loss 的
"one big forward pass of prefix + suffix at once"）：

- 每层：各路专家 token 经 pre_attn 得 q/k/v -> 在序列维拼接 -> 一次共享
  SDPA（gqa_sdpa，见 model/util.py；RoPE 位置按拼接顺序连续编号，每专家
  用自己的位置段 cos/sin）-> 按各段长度拆回 -> 各路 post_attn 更新自己的
  token；
- attention_mask 是整段 [Σseq, Σseq] 的可见性（bool True=可见 / float
  加性），由调用方按 pi05 双段语义构造（prefix 段双向、action 段 attend
  prefix + 整个动作 chunk），None 只用于调试（全可见）；GQA 的 kv 头广播在
  拼接后的整段上一次完成（assert_compatible 保证各专家 num_kv_heads 一致）。

推理两阶段（prefix 一次 prefill -> N 步去噪复用 prefix k/v）不在本文件实现：
等价逻辑已存在于 GemmaLM.prefill / PrefixKVCache（cache.py）/
ActionExpert.run_layers，且在 pi05 的双段 mask 下与 fused 前向数值等价
（prefix 不 attend action），按"同一逻辑只存在一份"不在此重复维护第二份实现。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.model.expert import Expert
from neat_pi.model.util import build_rope_cache, gqa_sdpa
from neat_pi.typing import (ActionTokensBHD, CondBD, LanguageTokensBTD)


class MoT(nn.Module):
    """把多个 Expert 组合进共享注意力前向的容器（不新建参数）。"""

    def __init__(self, mixtures: dict[str, Expert]) -> None:
        super().__init__()
        Expert.assert_compatible(mixtures)
        self.mixtures = nn.ModuleDict(mixtures)
        self._expert_name = list(mixtures)
        # 结构参数从首个专家透传（FastWAM 同款），其余专家已由
        # assert_compatible 保证与它一致
        first = self.mixtures[self._expert_name[0]]
        self.num_layers = first.num_layers
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
        """逐层 fused 前向（训练形态）：各路 token 同步过所有层，共享 SDPA。

        - embeds_dict：每路专家的 token 序列（key 须覆盖全部专家名），RoPE
          位置按传参顺序在拼接序列上连续编号；
        - attention_mask：整段 [Σseq, Σseq] 的可见性，接受 2D [S, S] 或 4D
          [B, 1, S, S] / [B, H, S, S]（bool True=可见 / float 加性）；
          None 时退化为全可见，仅调试用——pi05 双段语义必须显式传入；
        - conds：每路专家的条件向量（动作专家传 adarms_cond，VLM 传 None
          或省略），按名透传给对应 block 的 pre_attn / post_attn。
        返回更新后的各路 token。VLM 段的最终 norm 与动作段的最终 adaRMS
        norm 是专家级收尾，由调用方在容器外完成（见 pi05.py）。
        """
        missing = [name for name in self._expert_name if name not in embeds_dict]
        if missing:
            raise ValueError(f"embeds_dict 缺少专家 token：{missing}")

        # 拼接序 = 专家传参序；RoPE 位置跨专家连续编号（prefix 从 0 起）
        seq_lens = {name: embeds_dict[name].shape[1]
                    for name in self._expert_name}
        offset: dict[str, int] = {}
        total = 0
        for name in self._expert_name:
            offset[name] = total
            total += seq_lens[name]
        if attention_mask is not None:
            if (attention_mask.ndim not in (2, 4)
                    or attention_mask.shape[-2] != total
                    or attention_mask.shape[-1] != total):
                raise ValueError(
                    "attention_mask 须为 [S, S]（或 [B, 1|H, S, S]），"
                    f"S = 各专家序列长度之和 {total}，"
                    f"实际 {tuple(attention_mask.shape)}")

        first = embeds_dict[self._expert_name[0]]
        cos_full, sin_full = build_rope_cache(
            total, self.attn_head_dim, self.theta, first.device, first.dtype)
        rope = {
            name: (cos_full[offset[name]:offset[name] + seq_lens[name]],
                   sin_full[offset[name]:offset[name] + seq_lens[name]])
            for name in self._expert_name
        }
        conds = conds or {}
        tokens = dict(embeds_dict)
        scale = self.attn_head_dim ** -0.5

        for layer_idx in range(self.num_layers):
            q_chunks: list[torch.Tensor] = []
            k_chunks: list[torch.Tensor] = []
            v_chunks: list[torch.Tensor] = []
            pending: dict[str, tuple[nn.Module, object]] = {}
            for name in self._expert_name:
                block = self.mixtures[name].layers[layer_idx]
                cos, sin = rope[name]
                q, k, v, state = block.pre_attn(tokens[name], cos, sin,
                                                conds.get(name))
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                pending[name] = (block, state)

            # 一次共享 SDPA：所有专家 q/k/v 在序列维拼接（kv 头广播在
            # gqa_sdpa 内、对整段一次完成）
            attn_out = gqa_sdpa(
                torch.cat(q_chunks, dim=2),
                torch.cat(k_chunks, dim=2),
                torch.cat(v_chunks, dim=2),
                self.num_heads, self.num_kv_heads, scale,
                attn_mask=attention_mask,
            )
            # 拆回各路，交回各自的 post_attn
            start = 0
            for name in self._expert_name:
                end = start + seq_lens[name]
                block, state = pending[name]
                tokens[name] = block.post_attn(state, attn_out[:, start:end],
                                               conds.get(name))
                start = end
        return tokens

    def extra_repr(self) -> str:
        lines = []
        for name, module in self.mixtures.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f'({name}): {mod_str}')
        return '\n'.join(lines)
