"""MoT（Mixture of Transformers）层：VLM 与动作专家共享注意力的解耦实现。

为什么需要这个文件（本项目 FSDP 目标的核心）：

pi05 原生结构里，VLM（图像+文本）与动作专家在每个 transformer 层
**共享同一次 attention 计算**——两个专家的 token 拼接后一起算 attention，
但各自使用自己私有的 q/k/v/o 投影与 MLP 参数。如果把这种耦合直接写成
跨模块的张量传递，FSDP 的 auto_wrap 无法在不切断梯度/参数分片的前提下
把两个专家分别包裹。

MoT 的组织方式：
- 每一层是一个 MoTLayer，内部按"专家"持有两套私有参数（vlm.* / action.*）；
- attention 计算共享（token 拼接 -> 各自投影 qkv -> 统一 SDPA -> 各自 o_proj），
  但所有参数都属于本层，跨模块没有游离的参数引用；
- 因此 auto_wrap 可以以 MoTLayer 为最小单位整层分片，专家间不存在
  破坏 FSDP 参数分片的跨边界耦合。

权重对应关系：vlm.* 与 action.* 的参数名映射到 openpi checkpoint 中
PaliGemma llm.* 与 action expert 的同名张量，见 weights.py。
"""

from __future__ import annotations

from torch import Tensor, nn

from neat_pi.model.modules import MLP, RMSNorm
from neat_pi.typing import TokensBTD, typechecked


class ExpertParams(nn.Module):
    """单个专家在一层 MoT 中的私有参数：qkv/o 投影 + MLP + 两个 norm。"""

    def __init__(self, width: int, num_heads: int, num_kv_heads: int,
                 mlp_hidden: int) -> None:
        super().__init__()
        self.head_dim = width // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_proj = nn.Linear(width, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, width, bias=False)
        self.mlp = MLP(width, mlp_hidden)
        self.input_layernorm = RMSNorm(width)
        self.post_attention_layernorm = RMSNorm(width)


class MoTLayer(nn.Module):
    """一层 Mixture-of-Transformers：两个专家共享 attention，其余私有。

    这是 FSDP auto_wrap 的目标类型（见 training/fsdp.py）。
    """

    def __init__(self, vlm_width: int = 2048, vlm_heads: int = 8,
                 vlm_kv_heads: int = 1, vlm_mlp_hidden: int = 16384,
                 action_width: int = 1024, action_heads: int = 8,
                 action_kv_heads: int = 1, action_mlp_hidden: int = 4096) -> None:
        super().__init__()
        self.vlm = ExpertParams(vlm_width, vlm_heads, vlm_kv_heads, vlm_mlp_hidden)
        self.action = ExpertParams(action_width, action_heads, action_kv_heads,
                                   action_mlp_hidden)

    @typechecked
    def forward(self, vlm_tokens: TokensBTD, action_tokens: TokensBTD,
                attn_mask: Tensor | None = None,
                ) -> tuple[TokensBTD, TokensBTD]:
        """对两路 token 做一层共享注意力的 MoT 计算，各自残差更新。

        TODO: 按 ref/openpi gemma.py 的双专家 attention 语义实现：
        1. 两路 token 各自 input_layernorm、各自 qkv 投影；
        2. 序列维拼接后统一 SDPA（attention mask 控制 VLM 段对动作段的可见性，
           pi05 的动作段可以 attend VLM 段，反向不行）；
        3. 拆回两路，各自 o_proj + 残差、post norm + MLP + 残差。
        """
        raise NotImplementedError("待按 ref/openpi gemma.py 实现")


class MoTStack(nn.Module):
    """N 层 MoTLayer 的堆叠；auto_wrap 以 MoTLayer 为单位分片。"""

    def __init__(self, num_layers: int = 18, **layer_kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            MoTLayer(**layer_kwargs) for _ in range(num_layers)
        )

    def forward(self, vlm_tokens: TokensBTD, action_tokens: TokensBTD,
                attn_mask: Tensor | None = None,
                ) -> tuple[TokensBTD, TokensBTD]:
        for layer in self.layers:
            vlm_tokens, action_tokens = layer(vlm_tokens, action_tokens, attn_mask)
        return vlm_tokens, action_tokens
