"""FSDP 包装：按 fused MoT layer 和 SigLIP layer 做自动分片。

`MoTFusedLayer` 是 VLM 与 action expert 配对后的一层所有权边界；
`SiglipEncoderLayer` 是独立的视觉 transformer layer。二者都实现了完整
forward，因此可以作为 FSDP auto-wrap 的最小稳定单元。activation
checkpointing 先包这些 layer，FSDP 再包 checkpoint wrapper 内的目标层。
"""

from __future__ import annotations

from loguru import logger
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from neat_pi.config import FSDPConfig
from neat_pi.device.backend import DeviceContext, DeviceType
from neat_pi.model.mot import MoTFusedLayer
from neat_pi.model.siglip import SiglipEncoderLayer

_AUTO_WRAP_CLASSES: tuple[type[nn.Module], ...] = (
    MoTFusedLayer,
    SiglipEncoderLayer,
)
_CHECKPOINT_CLASSES: tuple[type[nn.Module], ...] = (
    MoTFusedLayer,
    SiglipEncoderLayer,
)


def fsdp_auto_wrap_classes() -> tuple[type[nn.Module], ...]:
    """返回允许被 FSDP auto-wrap 的最小 layer 单元。"""
    return _AUTO_WRAP_CLASSES


def fsdp_activation_checkpoint_classes() -> tuple[type[nn.Module], ...]:
    """返回 activation checkpointing 作用的 layer 单元。"""
    return _CHECKPOINT_CLASSES


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float | int = 2.0,
) -> torch.Tensor:
    """按 FSDP 分片语义裁剪梯度；普通模型回退到 PyTorch 工具函数。"""
    if isinstance(model, FullyShardedDataParallel):
        return model.clip_grad_norm_(max_norm, norm_type)
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type,
    )


def _sharding_strategy(name: str) -> ShardingStrategy:
    """把 YAML 中的 sharding 策略映射为 PyTorch FSDP enum。"""
    strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    return strategies[name]


def _backward_prefetch(name: str) -> BackwardPrefetch | None:
    """把 backward prefetch 配置映射为 FSDP enum 或禁用项。"""
    modes = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    return modes.get(name)


def _auto_wrap_policy() -> ModuleWrapPolicy:
    """构造只作用于 fused MoT layer 和 SigLIP layer 的 auto-wrap policy。"""
    return ModuleWrapPolicy(_AUTO_WRAP_CLASSES)


def _mixed_precision(param_dtype: torch.dtype) -> MixedPrecision:
    """构造 FSDP mixed precision；梯度 reduce 固定使用 float32。"""
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        buffer_dtype=param_dtype,
    )


def _cpu_offload(enabled: bool) -> CPUOffload | None:
    """按配置构造 CPU offload 参数。"""
    return CPUOffload(offload_params=True) if enabled else None


def _apply_activation_checkpointing(model: nn.Module) -> None:
    """对稳定的 transformer layer 开启 non-reentrant activation checkpointing。"""
    checkpoint_classes = _CHECKPOINT_CLASSES

    def should_checkpoint(module: nn.Module) -> bool:
        """判断一个子模块是否属于需要 checkpoint 的 layer。"""
        return isinstance(module, checkpoint_classes)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=should_checkpoint,
    )


def _build_fsdp_kwargs(
    cfg: FSDPConfig,
    ctx: DeviceContext,
    param_dtype: torch.dtype,
) -> dict[str, object]:
    """构造传给 FullyShardedDataParallel 的关键字参数。"""
    return {
        "sharding_strategy": _sharding_strategy(cfg.sharding_strategy),
        "auto_wrap_policy": _auto_wrap_policy(),
        "backward_prefetch": _backward_prefetch(cfg.backward_prefetch),
        "mixed_precision": _mixed_precision(param_dtype),
        "cpu_offload": _cpu_offload(cfg.cpu_offload),
        "device_id": ctx.device,
        "sync_module_states": cfg.sync_module_states,
        "forward_prefetch": cfg.forward_prefetch,
        "limit_all_gathers": cfg.limit_all_gathers,
        "use_orig_params": cfg.use_orig_params,
    }


def wrap_model_fsdp(
    model: nn.Module,
    cfg: FSDPConfig,
    ctx: DeviceContext,
    param_dtype: torch.dtype,
) -> nn.Module:
    """按配置把模型包成 FSDP；单进程时原样返回。

    nested FSDP unit 是 `MoTFusedLayer` 和 `SiglipEncoderLayer`；root
    `Pi05` 最后再包一层，用于分片 embedding、projection 和最终 head。
    optimizer 必须在本函数返回后再用 wrapped model 的参数创建。
    """
    if not ctx.is_distributed:
        if cfg.sharding_strategy != "no_shard":
            logger.warning(
                "world_size=1 但 fsdp.sharding_strategy={}；训练循环将使用未分片模型",
                cfg.sharding_strategy,
            )
        return model
    if ctx.type is DeviceType.CPU:
        raise ValueError("分布式 FSDP 训练只支持 cuda 或 npu 设备")

    if cfg.activation_checkpointing:
        _apply_activation_checkpointing(model)
    return FullyShardedDataParallel(
        model,
        **_build_fsdp_kwargs(cfg, ctx, param_dtype),
    )
