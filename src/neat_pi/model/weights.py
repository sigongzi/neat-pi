"""pi05 权重的加载与保存：与 openpi checkpoint 的名字映射集中在这里。

设计约束（沿用旧项目的硬件教训）：
- checkpoint 可能 ~7.5GB，禁止一次性读入内存 dict；
  一律用 safetensors.safe_open 的 memory-mapped 惰性读取，按名取张量；
- 加载流程为"组件级"：取一个子模块的权重 -> copy_ 进模型 -> 释放引用。

权重对应关系（骨架，实现时对照 ref/openpi 补全）：
- openpi PaliGemma llm.*      -> mot.layers.{i}.vlm.*
- openpi action expert 层参数  -> mot.layers.{i}.action.*
- openpi SigLIP vision tower   -> vision.*
- openpi VLM token embedding   -> tied 到 lm_head.weight（无独立 embed_tokens）
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
    checkpoint 是 transformers/PyTorch 命名（`model.paligemma_with_expert.*`，
    层索引为 `.layers.{N}`），直接按名字映射即可，无需 kernel 转置。
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


# checkpoint 里视觉塔的尾部前缀；本地 SigLIPVisionEncoder 的 state_dict 相对
# 层级与之逐字对齐，剥离「该前缀及之前的部分」即可得到本地参数名（见 siglip.py）。
# 完整 checkpoint 名形如：
#   model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight
_SIGLIP_VISION_PREFIX = "vision_tower.vision_model."


def load_siglip_vision_weights(model: nn.Module, checkpoint_dir: str) -> int:
    """把 checkpoint 中的 SigLIP 视觉塔权重加载进 model，返回加载的张量数。

    只加载视觉塔（437 张量，约 412M 参数），不触碰 7.5GB checkpoint 的其余部分；
    逐张量 shape 校验 + copy_，名字对不上就报错（不静默跳过）。
    model 应为 SigLIPVisionEncoder，其 state_dict 与 checkpoint 相对层级一致。
    """
    state = model.state_dict()
    loaded = 0
    for openpi_name, tensor in iter_checkpoint_tensors(checkpoint_dir):
        idx = openpi_name.find(_SIGLIP_VISION_PREFIX)
        if idx == -1:
            continue
        local_name = openpi_name[idx + len(_SIGLIP_VISION_PREFIX):]
        if local_name not in state:
            raise KeyError(f"视觉塔参数名不在模型中: {local_name} (来自 {openpi_name})")
        if state[local_name].shape != tensor.shape:
            raise ValueError(
                f"形状不匹配: {local_name} 模型 {tuple(state[local_name].shape)} "
                f"vs checkpoint {tuple(tensor.shape)}")
        state[local_name].copy_(tensor)
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"checkpoint 中没有视觉塔权重，请检查 {checkpoint_dir}")
    return loaded


# VLM 语言主干在 checkpoint 里的位置：层/norm 在 `...language_model.*` 下，
# tied 的 lm_head 单独在 `...paligemma.lm_head.weight`（没有独立 embed_tokens）。
_GEMMA_LM_LAYERS_PREFIX = "language_model."
_GEMMA_LM_HEAD_SUFFIX = "paligemma.lm_head.weight"


def translate_gemma_lm_name(openpi_name: str) -> str | None:
    """把 checkpoint 的 VLM 语言主干参数名翻译成 GemmaLM 的 state_dict 名。

    返回 None 表示该名字不属于 VLM 语言主干（例如动作专家 / 视觉塔）。
    """
    if _GEMMA_LM_LAYERS_PREFIX in openpi_name:
        return openpi_name.split(_GEMMA_LM_LAYERS_PREFIX, 1)[1]
    if openpi_name.endswith(_GEMMA_LM_HEAD_SUFFIX):
        return "lm_head.weight"
    return None


def load_gemma_lm_weights(model: nn.Module, checkpoint_dir: str) -> int:
    """把 checkpoint 中 VLM 语言主干权重加载进 GemmaLM，返回加载的张量数。

    只加载纯文本 decoder 的 164 张量（18 层 + final norm + tied lm_head），
    逐张量 shape 校验 + copy_；名字对不上就报错（不静默跳过）。
    """
    state = model.state_dict()
    loaded = 0
    for openpi_name, tensor in iter_checkpoint_tensors(checkpoint_dir):
        local_name = translate_gemma_lm_name(openpi_name)
        if local_name is None:
            continue
        if local_name not in state:
            raise KeyError(f"GemmaLM 参数名不在模型中: {local_name} (来自 {openpi_name})")
        if state[local_name].shape != tensor.shape:
            raise ValueError(
                f"形状不匹配: {local_name} 模型 {tuple(state[local_name].shape)} "
                f"vs checkpoint {tuple(tensor.shape)}")
        state[local_name].copy_(tensor)
        loaded += 1
    if loaded == 0:
        raise RuntimeError(f"checkpoint 中没有 VLM 语言主干权重，请检查 {checkpoint_dir}")
    return loaded
