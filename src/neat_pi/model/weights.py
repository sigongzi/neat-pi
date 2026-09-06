"""pi05 权重的加载、转换与保存：checkpoint 参数名映射集中在这里。

设计约束（沿用旧项目的硬件教训）：
- checkpoint 可能 ~7.5GB，禁止一次性读入内存 dict；
  一律用 safetensors.safe_open 的 memory-mapped 惰性读取，按名取张量；
- 加载流程为逐张量 mmap -> copy_，不做一次性 checkpoint dict 转换；
- 只允许 checkpoint 里的 action-expert 文本头显式跳过，其余名字必须全部
  映射并写入模型；模型参数也必须全部被写入。

还提供 NeatPi 本地权重格式：它已经保存为 Pi05 的 state_dict 名，并在
safetensors metadata 中写入格式标记，加载时不再做名字翻译。转换产物固定
为单个 model.safetensors，写入时逐张量流式拷贝，不聚合整个模型。

checkpoint 当前保存的键不带 `model.` 顶层包装，例如：
``paligemma_with_expert.paligemma.model.language_model.layers.0.*``。
较早导出的键则形如 ``model.paligemma_with_expert.*``；加载器统一先剥掉
这个可选顶层前缀，再映射到 Pi05 的模块名。
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass
import json
import os
import struct

import torch
from safetensors import safe_open
from torch import Tensor, nn


_LOCAL_CHECKPOINT_FORMAT = "neat-pi.pi05-local-v1"


@dataclass(frozen=True)
class Pi05ConversionResult:
    """一次 checkpoint 转换的统计结果。"""

    output_dir: Path
    loaded_count: int
    skipped_count: int
    shard_count: int


@dataclass(frozen=True)
class _TensorSpec:
    """单个输出张量的 safetensors header 信息。"""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    num_bytes: int


_SAFETENSORS_DTYPES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


def _checkpoint_shards(checkpoint_dir: str) -> list[Path]:
    """列出模型权重 shard；不匹配 processor 等其他 safetensors 文件。"""
    root = Path(checkpoint_dir)
    return sorted(root.glob("model*.safetensors"))


def iter_checkpoint_tensors(checkpoint_dir: str) -> Iterator[tuple[str, Tensor]]:
    """惰性遍历 checkpoint 中的 (名字, 张量)，不占满内存。

    支持单个 model.safetensors 或分片的 model-*.safetensors。
    """
    shards = _checkpoint_shards(checkpoint_dir)
    if not shards:
        raise FileNotFoundError(f"{checkpoint_dir} 下没有 safetensors 文件")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def is_local_pi05_checkpoint(checkpoint_dir: str) -> bool:
    """判断 checkpoint 是否已是使用 Pi05 参数名的 NeatPi 本地格式。"""
    shards = _checkpoint_shards(checkpoint_dir)
    if not shards:
        return False
    with safe_open(str(shards[0]), framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    return metadata.get("format") == _LOCAL_CHECKPOINT_FORMAT


def translate_name(checkpoint_name: str) -> str | None:
    """把 checkpoint 参数名翻译成 Pi05 的 state_dict 名。

    checkpoint 统一按 ``paligemma_with_expert.*`` 组织；兼容较早导出格式
    额外携带的 ``model.`` 顶层前缀。返回 None 表示该张量是显式跳过的
    action-expert 文本生成头（本模型不实现文本生成）。未知名字一律不
    做静默映射。
    """
    name = checkpoint_name.removeprefix("model.")

    if name == "paligemma_with_expert.gemma_expert.lm_head.weight":
        return None

    rules = (
        ("paligemma_with_expert.paligemma.model.vision_tower.vision_model.",
         "vision_tower."),
        ("paligemma_with_expert.paligemma.model.language_model.",
         "language_model."),
        ("paligemma_with_expert.paligemma.model.multi_modal_projector.",
         "multi_modal_projector."),
        ("paligemma_with_expert.gemma_expert.model.layers.",
         "action_expert.layers."),
        ("paligemma_with_expert.gemma_expert.model.norm.",
         "action_expert.norm."),
    )
    for checkpoint_prefix, model_prefix in rules:
        if name.startswith(checkpoint_prefix):
            return model_prefix + name[len(checkpoint_prefix):]

    # paligemma 的 lm_head 既是 VLM 文本头也是 tied token embedding。
    if name == "paligemma_with_expert.paligemma.lm_head.weight":
        return "language_model.lm_head.weight"

    action_direct_names = (
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    )
    if name.startswith(action_direct_names):
        return "action_expert." + name

    raise ValueError(f"无法映射的 checkpoint 参数名: {checkpoint_name}")


def load_pi05_weights(model: nn.Module, checkpoint_dir: str) -> None:
    """加载 checkpoint 权重进 model（就地更新），支持原始和本地格式。

    原始 checkpoint 先用 translate_name 转换名字；NeatPi 本地格式的名字
    已经与 state_dict 一致，直接按名加载。两种路径都逐张量 copy_ 并校验
    形状；未知名字和双向遗漏都直接报错。
    """
    state = model.state_dict()
    local_format = is_local_pi05_checkpoint(checkpoint_dir)
    expected_names = set(state)
    loaded_names: set[str] = set()
    skipped_names: set[str] = set()
    checkpoint_count = 0
    for checkpoint_name, tensor in iter_checkpoint_tensors(checkpoint_dir):
        checkpoint_count += 1
        local_name = (checkpoint_name if local_format
                      else translate_name(checkpoint_name))
        if local_name is None:
            skipped_names.add(checkpoint_name)
            continue
        if local_name not in state:
            source_label = ("本地名字" if local_format
                            else f"来自 {checkpoint_name}")
            raise KeyError(
                f"映射后的名字不在模型中: {local_name} ({source_label})")
        if local_name in loaded_names:
            raise ValueError(f"映射后的模型参数被重复写入: {local_name}")
        if state[local_name].shape != tensor.shape:
            raise ValueError(
                f"形状不匹配: {local_name} 模型 {tuple(state[local_name].shape)} "
                f"vs checkpoint {tuple(tensor.shape)}")
        state[local_name].copy_(tensor)
        loaded_names.add(local_name)

    if loaded_names != expected_names:
        missing = sorted(expected_names - loaded_names)
        raise RuntimeError(
            f"checkpoint 未覆盖 {len(missing)} 个模型参数，第一个: {missing[0]}")
    if len(loaded_names) + len(skipped_names) != checkpoint_count:
        raise RuntimeError(
            "checkpoint 张量计数不一致：loaded + skipped != checkpoint 总数；"
            "请检查 safetensors 是否包含重复键")
    expected_skipped = (set() if local_format else
                        {"paligemma_with_expert.gemma_expert.lm_head.weight"})
    if skipped_names != expected_skipped:
        raise RuntimeError(
            "显式跳过集不是预期的 action-expert 文本头，"
            f"实际: {sorted(skipped_names)}")


def convert_pi05_checkpoint(source_dir: str, output_dir: str,
                            ) -> Pi05ConversionResult:
    """把 checkpoint 另存为单文件 NeatPi 本地名格式，返回转换统计。

    输入可以是原始 checkpoint，也可以是已转换的多 shard 本地格式；本地
    输入按名字原样合并。输出固定为 ``model.safetensors``：先生成逐张量
    数据偏移的 header，再按同一顺序流式拷贝数据，因此不把全部权重聚合
    到内存。action-expert 的文本生成头只在原始输入中出现时丢弃。
    """
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    if source_path == output_path:
        raise ValueError("output_dir 不能与 source_dir 相同")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"output_dir 必须为空或不存在的目录: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    source_is_local = is_local_pi05_checkpoint(source_dir)
    specs: list[_TensorSpec] = []
    loaded_names: set[str] = set()
    skipped_count = 0

    def mapped_checkpoint_tensors(count_skips: bool) -> (
            Iterator[tuple[str, Tensor]]):
        """按本地名遍历模型张量，并统计原始输入中的显式跳过张量。"""
        nonlocal skipped_count
        for source_name, tensor in iter_checkpoint_tensors(source_dir):
            local_name = (source_name if source_is_local
                          else translate_name(source_name))
            if local_name is None:
                if count_skips:
                    skipped_count += 1
                continue
            yield local_name, tensor

    for local_name, tensor in mapped_checkpoint_tensors(count_skips=True):
        if local_name in loaded_names:
            raise ValueError(f"映射后的模型参数重复出现: {local_name}")

        safetensors_dtype = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if safetensors_dtype is None:
            raise ValueError(f"不支持的 checkpoint 张量 dtype: {tensor.dtype}")
        specs.append(_TensorSpec(
            name=local_name,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            num_bytes=tensor.numel() * tensor.element_size(),
        ))
        loaded_names.add(local_name)

    if not loaded_names:
        raise RuntimeError(f"没有可转换的模型权重，请检查 {source_dir}")

    header: dict[str, object] = {}
    data_offset = 0
    for spec in specs:
        header[spec.name] = {
            "dtype": _SAFETENSORS_DTYPES[spec.dtype],
            "shape": list(spec.shape),
            "data_offsets": [data_offset, data_offset + spec.num_bytes],
        }
        data_offset += spec.num_bytes
    header["__metadata__"] = {"format": _LOCAL_CHECKPOINT_FORMAT}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)

    output_file = output_path / "model.safetensors.tmp"
    try:
        with output_file.open("wb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            data_bytes_written = 0
            for spec, (local_name, tensor) in zip(
                    specs, mapped_checkpoint_tensors(count_skips=False),
                    strict=True):
                if local_name != spec.name:
                    raise RuntimeError(
                        "两次遍历 checkpoint 的张量顺序不一致: "
                        f"{local_name} != {spec.name}")
                if (tuple(tensor.shape) != spec.shape
                        or tensor.dtype != spec.dtype):
                    raise RuntimeError(
                        f"两次遍历 checkpoint 的张量不一致: {spec.name}")
                payload = memoryview(
                    tensor.contiguous().flatten().view(torch.uint8).numpy())
                if len(payload) != spec.num_bytes:
                    raise RuntimeError(f"张量字节数不一致: {spec.name}")
                output.write(payload)
                data_bytes_written += len(payload)
            if data_bytes_written != data_offset:
                raise RuntimeError(
                    "输出数据长度与 safetensors header 不一致: "
                    f"{data_bytes_written} != {data_offset}")
            output.flush()
            os.fsync(output.fileno())
        output_file.replace(output_path / "model.safetensors")
    except Exception:
        output_file.unlink(missing_ok=True)
        raise

    return Pi05ConversionResult(
        output_dir=output_path,
        loaded_count=len(loaded_names),
        skipped_count=skipped_count,
        shard_count=1,
    )


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
# tied 的 lm_head 单独在 `...paligemma.lm_head.weight`（checkpoint 里没有独立
# embed_tokens；token embedding 由调用方查 lm_head.weight 得到）。
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
