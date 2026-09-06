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

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast
from torch import nn

if TYPE_CHECKING:
    from neat_pi.model.mot import MoT


class ExpertLayerView:
    """只读访问 MoTFusedLayer 中属于一个 Expert 的 block 序列。

    FSDP 的所有权必须落在配对后的 fused layer 上；这个 view 让
    GemmaLM.prefill / ActionExpert.run_layers 等推理路径保持原 API，
    同时避免 Expert 与 MoT 重复注册同一批参数。
    """

    __slots__ = ("_stack", "_expert_name")

    def __init__(self, stack: "MoT", expert_name: str) -> None:
        """绑定 fused-layer 容器与专家名。"""
        self._stack = stack
        self._expert_name = expert_name

    def __len__(self) -> int:
        """返回层数。"""
        return len(self._stack.layers)

    def __getitem__(self, index: int) -> nn.Module:
        """按层号返回该 Expert 的 block。"""
        layer = self._stack.layers[index]
        return cast(nn.Module, getattr(layer, self._expert_name))

    def __iter__(self) -> Iterator[nn.Module]:
        """按层序迭代 Expert block。"""
        return (self[index] for index in range(len(self)))


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
    def __init__(self) -> None:
        """初始化 Expert 基类并清空 layer view。"""
        super().__init__()
        self._layer_view: ExpertLayerView | None = None

    @property
    def layers(self) -> nn.ModuleList | ExpertLayerView:
        """返回 owned layers 或绑定到 MoTFusedLayer stack 的只读 view。"""
        if self._layer_view is not None:
            return self._layer_view
        layers = self._modules.get("layers")
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError("Expert 尚未注册 owned layers")
        return layers

    @layers.setter
    def layers(self, value: nn.ModuleList | ExpertLayerView) -> None:
        """注册 owned layers，或把 layers 切换到 MoT 的只读 view。"""
        self._layer_view = value if isinstance(value, ExpertLayerView) else None
        if isinstance(value, ExpertLayerView):
            self._modules.pop("layers", None)
        else:
            self._modules["layers"] = value

    def bind_layer_view(self, stack: "MoT", expert_name: str) -> None:
        """把 Expert 的 layer 访问切换到 MoT 配对层上的只读 view。"""
        if self._layer_view is not None:
            raise RuntimeError("Expert layer view 已经绑定到 MoT")
        self._layer_view = ExpertLayerView(stack, expert_name)
        self._modules.pop("layers", None)

    def take_owned_layers(self) -> nn.ModuleList:
        """取出 owned layers，供 MoTFusedLayer stack 接管所有权。"""
        # "owner" 指 nn.Module 的注册所有权：只有注册在哪个父模块下，
        # state_dict 参数路径、FSDP 分片和 module traversal 才归属哪里。
        # 这里必须把 layers 从 Expert._modules 移除，再由 MoTFusedLayer 注册；
        # 否则同一个 block 会同时出现在 language_model/action_expert 与
        # mot.layers 两条路径下，造成重复参数和 FSDP 所有权歧义。
        # 移交后 Expert 仍可通过 ExpertLayerView 访问同一批 block，因此
        # prefill / run_layers 等推理路径保持不变。
        if self._layer_view is not None:
            raise RuntimeError("Expert 不拥有 layers，无法再次移交给 MoT")
        layers = self._modules.get("layers")
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("Expert.layers 必须是 nn.ModuleList")
        self._modules.pop("layers", None)
        return layers

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
