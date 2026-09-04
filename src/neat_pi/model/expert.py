"""MoT mixture 的专家抽象基类。

FastWAM 式 MoT（见 mot.py）把多个专家（GemmaLM / ActionExpert）作为 mixture
传入，在每层把各专家的 token 拼进同一次 SDPA。这要求所有专家共享同一套
注意力结构：层数、num_heads / num_kv_heads（GQA 分组）、attn_head_dim 必须
一致，否则 q/k/v 无法按头结构拼接；fused 路径在拼接后的整段序列上做 RoPE
连续编号，因此 theta（RoPE 基数）也必须一致（openpi pi0.py 的 cumsum 位置）。

本基类把"专家需向 MoT 暴露的统一结构契约"固化下来：它首先是一个普通
nn.Module（各专家仍可独立 forward / prefill，也能作为子模块注册进
ModuleDict），同时提供 MoT 读取与跨专家校验结构参数的入口。

子类在 __init__ 里必须设置：
- layers: nn.ModuleList；layers[i] 须实现 MoT 的半块接口 pre_attn /
  post_attn（契约与 fused 语义见 mot.py 模块 docstring）——norm / 门控残差 /
  时间条件注入等差异留在各专家 block 内，MoT 只做逐层配对与共享 SDPA；
- num_heads / num_kv_heads / attn_head_dim / theta。
块内子模块仍统一命名 input_layernorm / self_attn.{q,k,v,o}_proj /
post_attention_layernorm / mlp（pre/post 半块按这套名字访问自己的参数，
checkpoint 权重映射也依赖这套名字）。
num_layers 由基类从 len(layers) 推导，不另存一份，避免双源不一致。
"""

from __future__ import annotations

from torch import nn


class Expert(nn.Module):
    """可被 MoT 当 mixture 消费的专家：既是 nn.Module，又暴露共享结构参数。

    子类须在 __init__ 里设置下方注解的结构属性；num_layers 由基类从
    layers 推导。MoT 用 assert_compatible 校验跨专家一致性后透传这些参数。
    """

    # 注意力结构（跨专家必须一致，见模块 docstring）
    num_heads: int
    num_kv_heads: int
    attn_head_dim: int
    theta: float
    # 逐层模块堆叠；layers[i] 的命名契约见模块 docstring
    layers: nn.ModuleList

    @property
    def num_layers(self) -> int:
        """层数统一从 layers 推导，避免与 __init__ 形参重复声明。"""
        return len(self.layers)

    @staticmethod
    def assert_compatible(mixtures: dict[str, "Expert"]) -> None:
        """校验所有专家的结构参数一致，不一致抛 ValueError。

        MoT 在 __init__ 里调用：mixtures 为空或任一专家的
        num_layers / num_heads / num_kv_heads / attn_head_dim / theta
        与首个专家不同即报错——这些不一致会在拼接 SDPA 时才静默爆掉，
        提前到构造期暴露。
        """
        if not mixtures:
            raise ValueError("mixtures 不能为空")
        names = list(mixtures)
        first = mixtures[names[0]]
        for name, expert in mixtures.items():
            if expert is first:
                continue
            for attr in ("num_heads", "num_kv_heads", "attn_head_dim", "theta"):
                if getattr(expert, attr) != getattr(first, attr):
                    raise ValueError(
                        f"专家 {name} 的 {attr}={getattr(expert, attr)} 与 "
                        f"{names[0]} 的 {getattr(first, attr)} 不一致，"
                        "无法共享同一次 SDPA")
            if expert.num_layers != first.num_layers:
                raise ValueError(
                    f"专家 {name} 层数 {expert.num_layers} != "
                    f"{names[0]} 层数 {first.num_layers}")
