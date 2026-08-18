"""pi05 权重的加载与保存：与 openpi checkpoint 的名字映射集中在这里。

设计约束（沿用旧项目的硬件教训）：
- checkpoint 可能 ~7.5GB，禁止一次性读入内存 dict；
  一律用 safetensors.safe_open 的 memory-mapped 惰性读取，按名取张量；
- 加载流程为"组件级"：取一个子模块的权重 -> copy_ 进模型 -> 释放引用。

权重对应关系（骨架，实现时对照 ref/openpi 补全）：
- openpi PaliGemma llm.*      -> mot.layers.{i}.vlm.*
- openpi action expert 层参数  -> mot.layers.{i}.action.*
- openpi SigLIP vision tower   -> vision.*
- openpi embed_tokens          -> embedding.embed_tokens
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from torch import Tensor, nn


def iter_checkpoint_tensors(checkpoint_dir: str) -> Iterator[tuple[str, Tensor]]:
    """惰性遍历 checkpoint 中的 (名字, 张量)，不占满内存。

    支持单个 model.safetensors 或分片的 model-*.safetensors。
    """
    root = Path(checkpoint_dir)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{checkpoint_dir} 下没有 safetensors 文件")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def translate_name(openpi_name: str) -> str:
    """把 openpi checkpoint 的参数名翻译成本模型的 state_dict 名。

    TODO: 对照 ref/openpi 实现完整的映射规则（上面 docstring 的四类）。
    实现时注意 JAX/Flax 命名（kernel 转置等）与 PyTorch 的差异。
    """
    raise NotImplementedError("待对照 ref/openpi 权重名实现")


def load_pi05_weights(model: nn.Module, checkpoint_dir: str) -> None:
    """把 openpi 格式 checkpoint 加载进 model（就地更新参数）。

    逐张量 copy_ 并校验形状；遇到无法映射的名字应报错而不是跳过——
    静默丢权重比直接失败更难排查（AGENTS.md 运行规则）。
    """
    state = model.state_dict()
    loaded = 0
    for openpi_name, tensor in iter_checkpoint_tensors(checkpoint_dir):
        local_name = translate_name(openpi_name)
        if local_name not in state:
            raise KeyError(f"映射后的名字不在模型中: {local_name} (来自 {openpi_name})")
        if state[local_name].shape != tensor.shape:
            raise ValueError(
                f"形状不匹配: {local_name} 模型 {tuple(state[local_name].shape)} "
                f"vs checkpoint {tuple(tensor.shape)}")
        state[local_name].copy_(tensor)
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"没有加载任何权重，请检查 {checkpoint_dir}")
