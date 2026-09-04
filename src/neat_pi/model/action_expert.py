"""动作专家（pi05 的 flow-matching action expert）。

pi05 的动作生成用 flow matching：给定带噪动作 x_t 与连续时间 t ∈ [0,1]，
动作专家预测速度场 v(x_t, t)，训练目标是与 (x_1 - x_0) 的 MSE。

pi05 与 pi0 的差异（对齐 ref/openpi pi0.py 的 pi05 分支）：
- 不再有 state token（state_proj 已废弃，状态经离散化拼进文本 prompt）；
- 时间 t 不拼成 token，而是经 sincos 嵌入 + 两层 MLP 得到 adaRMS 条件向量
  （adarms_cond），注入每层 DiTBlock 与最终 norm 的 AdaLayerNorm。

模块划分（配合 mot.py 的 MoT 组织；半块接口契约见 mot.py 模块 docstring）：
- 本模块持有动作专家的**全部参数**：输入/输出投影、时间 MLP、以及
  `layers`（DiTBlock 堆叠）。每层 DiTBlock 实现 pre_attn / post_attn 两个
  半块：pre_attn 做 adaRMS pre-norm + 私有 qkv + RoPE，post_attn 做私有
  o_proj、门控残差、adaRMS post-norm 与 MLP；跨专家的共享 SDPA 由 MoT 在
  拼接序列上完成，DiTBlock 不自作注意力。
- MoT 不新建参数，只把各专家模块组合起来做共享 attention；跨专家只流动
  激活（q/k/v、attention 输出），没有参数耦合，可作为 FSDP auto_wrap 的
  候选粒度（最终形态见 training/fsdp.py 的 TODO）。
- DiTBlock.forward（= pre_attn -> 本段自注意力 [可拼外部 VLM k/v] ->
  post_attn）是脱离 MoT 的独立形态，供单块测试/调试与推理
  ActionExpert.run_layers 使用。

权重对应关系（openpi checkpoint 的 gemma_expert.*，见 weights.py）：
- action_in_proj / action_out_proj / time_mlp_in / time_mlp_out：同名，带 bias；
- layers.{i}.self_attn.{q,k,v,o}_proj、layers.{i}.mlp.{gate,up,down}_proj、
  layers.{i}.input_layernorm.dense、layers.{i}.post_attention_layernorm.dense
  <- gemma_expert.model.layers.{i}.*；
- norm.dense <- gemma_expert.model.norm（最终 adaRMS norm）。
  注：checkpoint 里 gemma_expert.lm_head（文本生成用）对 flow matching 训练
  无用，本模块不实现。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neat_pi.model.cache import PrefixKVCache
from neat_pi.model.expert import Expert
from neat_pi.model.gemma import GemmaAttention
from neat_pi.model.util import (apply_rotary_pos_emb, build_rope_cache,
                                gqa_sdpa)
from neat_pi.model.modules import MLP, AdaLayerNorm
from neat_pi.typing import (ActionBHD, ActionTokensBHD, AttentionMaskBHLS,
                            CondBD, GateB1D, TimeB, typechecked)


def posemb_sincos(pos: TimeB, embedding_dim: int,
                  min_period: float = 4e-3, max_period: float = 4.0) -> Tensor:
    """标量时间步的正余弦嵌入，返回 [batch, embedding_dim]。

    对齐 ref/openpi pi0.py 的 posemb_sincos：周期在 [min_period, max_period]
    上按几何级数分布，灵敏度覆盖 t ∈ [0, 1] 的 flow matching 时间范围。
    计算在 float32 下进行，避免低精度下周期混叠。
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) 必须能被 2 整除")
    # fraction ∈ [0, 1]，共 embedding_dim // 2 个频率分量
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2,
                              device=pos.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    # [batch, dim/2]：t * (2π / period)
    sinusoid_input = torch.einsum(
        "i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    return torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)


class DiTBlock(nn.Module):
    """动作专家的单层 DiT block：adaRMS 时间条件 + 私有 qkv/o/MLP。

    与标准 Gemma decoder layer 的差异（即 "DiT" 之处）：
    - 两个 pre-norm 都是 AdaLayerNorm（adaLN-Zero 语义）：时间条件经 dense
      生成 scale/shift/gate，gate 在残差处消费（x = x + y * gate），对齐
      ref modeling_gemma.py 的 adaRMSNorm + _gated_residual；
    - 半块接口（见 mot.py 模块 docstring）：pre_attn 做 input_layernorm ->
      qkv + RoPE，post_attn 做 o_proj -> 门控残差 -> post norm -> MLP ->
      门控残差；共享 SDPA 由 MoT 跨专家拼接完成（gqa_sdpa）。
    """

    def __init__(self, width: int = 1024, num_heads: int = 8,
                 num_kv_heads: int = 1, head_dim: int = 256,
                 mlp_hidden: int = 4096, eps: float = 1e-6) -> None:
        """默认超参对应 pi05 的 gemma_300m 动作专家（width 1024 / 8 头 /
        1 kv 头 / head_dim 256 / mlp 4096）；时间条件维度等于 width。"""
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        # self_attn 只作 qkv/o 投影的参数容器（与 checkpoint 同名）；
        # 注意力计算统一走 util.gqa_sdpa，不复刻 GemmaAttention.attend。
        self.self_attn = GemmaAttention(width, num_heads, num_kv_heads, head_dim)
        self.mlp = MLP(width, mlp_hidden)
        self.input_layernorm = AdaLayerNorm(width, width, eps)
        self.post_attention_layernorm = AdaLayerNorm(width, width, eps)

    @typechecked
    def pre_attn(self, action_tokens: ActionTokensBHD, cos: Tensor,
                 sin: Tensor, cond: CondBD | None = None,
                 ) -> tuple[Tensor, Tensor, Tensor,
                            tuple[ActionTokensBHD, GateB1D | None]]:
        """半块 pre：adaRMS pre-norm + 私有 q/k/v 投影 + RoPE。

        cond 为时间条件（adarms_cond），注入 input_layernorm 产出
        (normed, gate)。返回 (q, k, v, state)：q 形状
        [batch, num_heads, seq, head_dim]，k/v 未广播
        [batch, num_kv_heads, seq, head_dim]（RoPE 已施加）；state =
        (pre-norm 前的 x, gate)，由同一 block 的 post_attn 做门控残差。
        """
        if cond is None:
            raise ValueError("动作专家 DiTBlock 需要时间条件 cond（adarms_cond）")
        batch, seq, _ = action_tokens.shape
        attn = self.self_attn
        normed, gate = self.input_layernorm(action_tokens, cond)
        q = attn.q_proj(normed).view(batch, seq, self.num_heads,
                                     self.head_dim).transpose(1, 2)
        k = attn.k_proj(normed).view(batch, seq, self.num_kv_heads,
                                     self.head_dim).transpose(1, 2)
        v = attn.v_proj(normed).view(batch, seq, self.num_kv_heads,
                                     self.head_dim).transpose(1, 2)
        # RoPE 只施加在本专家的 q/k 上；外部 VLM 段 k/v 传入前已由 VLM 侧施加
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v, (action_tokens, gate)

    @typechecked
    def post_attn(self, state: tuple[ActionTokensBHD, GateB1D | None],
                  attn_out: Tensor, cond: CondBD | None = None,
                  ) -> ActionTokensBHD:
        """半块 post：私有 o_proj + 门控残差 + adaRMS post-norm + MLP + 门控残差。

        attn_out 是共享 SDPA 输出切回本段的 [batch, seq, num_heads*head_dim]
        片段；state 是 pre_attn 返回的 (pre-norm 前的 x, gate)：x 作残差、
        gate 门控 attn 支路；post norm 再次注入 cond，产出第二个 gate 门控
        MLP 支路。
        """
        if cond is None:
            raise ValueError("动作专家 DiTBlock 需要时间条件 cond（adarms_cond）")
        residual, gate = state
        x = residual + self.self_attn.o_proj(attn_out) * gate
        residual = x
        normed, gate = self.post_attention_layernorm(x, cond)
        return residual + self.mlp(normed) * gate

    @typechecked
    def forward(self, action_tokens: ActionTokensBHD, adarms_cond: CondBD,
                cos: Tensor, sin: Tensor,
                vlm_kv: tuple[Tensor, Tensor] | None = None,
                attn_mask: AttentionMaskBHLS | None = None) -> ActionTokensBHD:
        """对动作 token 做一层 DiT 计算，返回残差更新后的动作 token。

        本段版本 = pre_attn -> 一次自注意力（可拼外部 VLM k/v）-> post_attn，
        供独立测试/调试与推理 run_layers 使用；数值与 MoT fused 前向中动作
        专家的行一致（两者 attend 到的列相同）。

        - cos/sin：动作段的 RoPE 缓存，形状 [action_horizon, head_dim]，
          位置编号接在 VLM 前缀之后（由调用方切片）；
        - vlm_kv：VLM 侧同层的 (k, v)，各为
          [batch, num_kv_heads, prefix_len, head_dim]，已施加 RoPE；
        - attn_mask：SDPA 的 mask，形状可广播到
          [batch, num_heads, action_horizon, prefix_len + action_horizon]，
          bool（True 可见）或 float（加性），由调用方按 pi05 的双段可见性构造。
        """
        q, k, v, state = self.pre_attn(action_tokens, cos, sin, adarms_cond)
        if vlm_kv is not None:
            # 序列维拼接：VLM 前缀在前、动作段在后（与 pi05 的 token 排布一致）
            k = torch.cat([vlm_kv[0], k], dim=2)
            v = torch.cat([vlm_kv[1], v], dim=2)
        attn_out = gqa_sdpa(q, k, v, self.num_heads, self.num_kv_heads,
                            self.scale, attn_mask=attn_mask)
        return self.post_attn(state, attn_out, adarms_cond)


class ActionExpert(Expert):
    """flow-matching 动作专家：投影 + 时间条件 + DiTBlock 堆叠。

    作为 MoT 的动作专家，须满足 Expert 契约：自带完整 18 层堆叠 layers
    （每层 DiTBlock，实现 pre_attn / post_attn 半块接口），顶层暴露
    num_heads / num_kv_heads / attn_head_dim / theta 供 MoT 校验与透传。

    - encode_tokens：带噪动作 -> 动作 token；时间 t -> adaRMS 条件向量；
    - run_layers：动作 token 依次过所有 DiTBlock（每层可消费 VLM 侧 k/v，
      推理时传 GemmaLM.prefill 产出的 PrefixKVCache）与最终 adaRMS norm；
    - decode_velocity：动作段输出 -> 速度场预测（action_out_proj）；
    - forward：以上三步的组合，(x_t, t) -> v(x_t, t)。
    """

    def __init__(self, action_dim: int = 32, hidden_dim: int = 1024,
                 num_layers: int = 18, num_heads: int = 8,
                 num_kv_heads: int = 1, head_dim: int = 256,
                 mlp_hidden: int = 4096, theta: float = 10_000.0) -> None:
        """action_dim 为动作向量维度；其余默认对应 pi05 的 gemma_300m
        动作专家（hidden 1024 / 18 层 / 8 头 / 1 kv 头 / head_dim 256 /
        mlp 4096），theta 为 RoPE 基数（与 VLM 侧一致）。"""
        super().__init__()
        self.hidden_dim = hidden_dim
        # 与 GemmaLM 统一的 Expert 契约属性（构造形参名保持 head_dim）
        self.attn_head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.theta = theta
        # 动作向量 <-> 专家 hidden 维度（与 checkpoint 一致，带 bias）
        self.action_in_proj = nn.Linear(action_dim, hidden_dim, bias=True)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim, bias=True)
        # 时间嵌入 MLP：sincos(hidden_dim) -> hidden_dim -> hidden_dim
        self.time_mlp_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.time_mlp_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # DiTBlock 堆叠与最终 adaRMS norm
        self.layers = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads, num_kv_heads, head_dim, mlp_hidden)
            for _ in range(num_layers)
        )
        self.norm = AdaLayerNorm(hidden_dim, hidden_dim)

    @typechecked
    def encode_tokens(self, noisy_action: ActionBHD,
                      t: TimeB) -> tuple[ActionTokensBHD, CondBD]:
        """把带噪动作编成动作专家 token，把时间编成 adaRMS 条件向量。

        返回 (动作 token 序列 [B, H, D], adaRMS 条件向量 [B, D])；
        后者注入 run_layers 中每层 DiTBlock 与最终 norm 的 AdaLayerNorm。
        """
        action_tokens = self.action_in_proj(noisy_action)
        time_emb = posemb_sincos(t, self.hidden_dim).to(noisy_action.dtype)
        time_emb = F.silu(self.time_mlp_in(time_emb))
        adarms_cond = F.silu(self.time_mlp_out(time_emb))
        return action_tokens, adarms_cond

    @typechecked
    def run_layers(self, action_tokens: ActionTokensBHD, adarms_cond: CondBD,
                   vlm_kvs: PrefixKVCache | Sequence[tuple[Tensor, Tensor]] | None = None,
                   attn_mask: AttentionMaskBHLS | None = None) -> ActionTokensBHD:
        """动作 token 依次过所有 DiTBlock，最后过最终 adaRMS norm。

        vlm_kvs：每层一个 (k, v)（见 DiTBlock.forward），推理时是
        GemmaLM.prefill 产出的 PrefixKVCache（只读、按层索引），训练/调试
        时也可以是普通的 list[tuple]；为 None 时各层退化为动作 token 的
        纯自注意力（脱离 VLM 的独立测试/调试用）。RoPE 位置按 pi05 约定
        接在 VLM 前缀之后连续编号：[prefix_len, prefix_len + action_horizon)。
        """
        seq = action_tokens.shape[1]
        prefix_len = vlm_kvs[0][0].shape[2] if vlm_kvs is not None else 0
        # 动作段的 RoPE 缓存：先建全长的 cos/sin，再切出动作段位置的切片
        cos, sin = build_rope_cache(prefix_len + seq, self.attn_head_dim,
                                    self.theta, action_tokens.device,
                                    action_tokens.dtype)
        cos, sin = cos[prefix_len:], sin[prefix_len:]
        for i, layer in enumerate(self.layers):
            vlm_kv = vlm_kvs[i] if vlm_kvs is not None else None
            action_tokens = layer(action_tokens, adarms_cond, cos, sin,
                                  vlm_kv, attn_mask)
        # 最终 adaRMS norm 的 gate 无残差可消费，直接丢弃（对齐参考实现）
        action_tokens, _ = self.norm(action_tokens, adarms_cond)
        return action_tokens

    @typechecked
    def decode_velocity(self, action_tokens: ActionTokensBHD) -> ActionBHD:
        """DiTBlock 堆叠输出的动作段 token -> 速度场预测（action_out_proj）。"""
        return self.action_out_proj(action_tokens)

    @typechecked
    def forward(self, noisy_action: ActionBHD, t: TimeB,
                vlm_kvs: PrefixKVCache | Sequence[tuple[Tensor, Tensor]] | None = None,
                attn_mask: AttentionMaskBHLS | None = None) -> ActionBHD:
        """完整前向：encode_tokens -> run_layers -> decode_velocity。

        输入带噪动作 x_t 与时间 t，输出速度场预测 v(x_t, t)。
        vlm_kvs / attn_mask 的语义同 run_layers：推理传 PrefixKVCache，
        训练 joint 前向传逐层 list，None 为无 VLM 条件的独立形态。
        """
        action_tokens, adarms_cond = self.encode_tokens(noisy_action, t)
        action_tokens = self.run_layers(action_tokens, adarms_cond,
                                        vlm_kvs, attn_mask)
        return self.decode_velocity(action_tokens)
