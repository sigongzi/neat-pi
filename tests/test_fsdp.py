"""FSDP wrapper 的配置映射、auto-wrap 单元与 activation checkpointing 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy

from neat_pi.config import FSDPConfig, ModelConfig
from neat_pi.device.backend import DeviceContext, DeviceType
from neat_pi.model.mot import MoTFusedCheckpointAdapter, MoTFusedLayer
from neat_pi.model.pi05 import Pi05
from neat_pi.model.siglip import SiglipEncoderLayer
from neat_pi.training.fsdp import (
    _apply_activation_checkpointing,
    _build_fsdp_kwargs,
    _backward_prefetch,
    _sharding_strategy,
    fsdp_activation_checkpoint_classes,
    fsdp_auto_wrap_classes,
    wrap_model_fsdp,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_model_config() -> ModelConfig:
    """构造完整但很小的 Pi05 配置，只测试结构，不训练。"""
    return ModelConfig(
        image_size=16,
        num_cameras=2,
        action_horizon=3,
        action_dim=7,
        vision_hidden_dim=16,
        vision_patch_size=4,
        vision_num_layers=2,
        vision_num_heads=4,
        vision_mlp_hidden_dim=32,
        vocab_size=64,
        vlm_hidden_dim=24,
        vlm_num_layers=2,
        vlm_num_heads=4,
        vlm_num_kv_heads=1,
        vlm_attn_head_dim=8,
        vlm_mlp_hidden_dim=48,
        expert_hidden_dim=16,
        expert_num_layers=2,
        expert_num_heads=4,
        expert_num_kv_heads=1,
        expert_attn_head_dim=8,
        expert_mlp_hidden_dim=32,
    )


def _cpu_context(world_size: int = 1) -> DeviceContext:
    """构造用于结构测试的 CPU DeviceContext。"""
    return DeviceContext(
        type=DeviceType.CPU,
        device=torch.device("cpu"),
        world_size=world_size,
    )


def test_auto_wrap_and_checkpoint_units_match_fsdp_plan() -> None:
    """FSDP unit 是 fused layer；checkpoint 边界是张量签名的 adapter。

    MoTFusedLayer 的 dict 签名不能穿过 checkpoint 边界（未定义行为，
    见 docs/plan/09），因此 checkpoint 直接包
    MoTFusedCheckpointAdapter，FSDP auto-wrap 仍以内部的 MoTFusedLayer
    为 unit（adapter 无自有参数，分片粒度不变）。
    """
    assert fsdp_auto_wrap_classes() == (MoTFusedLayer, SiglipEncoderLayer)
    assert fsdp_activation_checkpoint_classes() == (
        MoTFusedCheckpointAdapter,
        SiglipEncoderLayer,
    )


def test_strategy_and_prefetch_strings_map_to_torch_enums() -> None:
    """YAML 字符串映射到对应 PyTorch enum，none 表示禁用。"""
    assert _sharding_strategy("full_shard") is ShardingStrategy.FULL_SHARD
    assert _sharding_strategy("shard_grad_op") is ShardingStrategy.SHARD_GRAD_OP
    assert _sharding_strategy("no_shard") is ShardingStrategy.NO_SHARD
    assert _backward_prefetch("backward_pre") is BackwardPrefetch.BACKWARD_PRE
    assert _backward_prefetch("backward_post") is BackwardPrefetch.BACKWARD_POST


def test_backward_prefetch_none_disables_prefetch() -> None:
    """backward_prefetch=none 应显式返回 None。"""
    assert _backward_prefetch("none") is None


def test_fsdp_kwargs_use_requested_precision_and_strategy() -> None:
    """FSDP kwargs 完整反映配置和 device backend 抽象。"""
    cfg = FSDPConfig(
        sharding_strategy="shard_grad_op",
        backward_prefetch="backward_post",
        activation_checkpointing=False,
        cpu_offload=True,
    )
    kwargs = _build_fsdp_kwargs(
        cfg,
        _cpu_context(world_size=2),
        torch.bfloat16,
    )

    assert kwargs["sharding_strategy"] is ShardingStrategy.SHARD_GRAD_OP
    assert kwargs["backward_prefetch"] is BackwardPrefetch.BACKWARD_POST
    assert kwargs["sync_module_states"] is True
    assert kwargs["forward_prefetch"] is False
    assert kwargs["limit_all_gathers"] is True
    assert kwargs["use_orig_params"] is True
    assert kwargs["device_id"] == torch.device("cpu")

    precision = kwargs["mixed_precision"]
    assert precision.param_dtype is torch.bfloat16
    assert precision.reduce_dtype is torch.float32
    assert precision.buffer_dtype is torch.bfloat16

    offload = kwargs["cpu_offload"]
    assert offload is not None and offload.offload_params is True


def test_single_process_training_returns_unwrapped_model() -> None:
    """world_size=1 不包 FSDP，避免分布式初始化和多余开销。"""
    cfg = FSDPConfig()
    model = Pi05(_small_model_config())
    wrapped = wrap_model_fsdp(model, cfg, _cpu_context(1), torch.float32)
    assert wrapped is model


def test_distributed_fsdp_rejects_cpu_device() -> None:
    """FSDP 分布式运行时不允许 CPU 设备。"""
    cfg = FSDPConfig()
    model = Pi05(_small_model_config())
    with pytest.raises(ValueError, match="cuda 或 npu"):
        wrap_model_fsdp(model, cfg, _cpu_context(2), torch.bfloat16)


def test_activation_checkpointing_wraps_expected_layers() -> None:
    """non-reentrant checkpoint wrapper 只作用于张量签名的边界单元。"""
    cfg = _small_model_config()
    model = Pi05(cfg)
    expected_count = cfg.vision_num_layers + cfg.expert_num_layers

    _apply_activation_checkpointing(model)
    wrappers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, CheckpointWrapper)
    ]
    wrapped_modules = [
        module._checkpoint_wrapped_module
        for _, module in wrappers
    ]

    assert len(wrappers) == expected_count
    assert all(
        isinstance(module, (MoTFusedCheckpointAdapter, SiglipEncoderLayer))
        for module in wrapped_modules
    )
    # MoT 侧被 checkpoint 的是 adapter，其内部的 fused layer 保持
    # MoTFusedLayer——dict 不穿越 checkpoint 边界，FSDP unit 粒度不变。
    adapters = [
        module for module in wrapped_modules
        if isinstance(module, MoTFusedCheckpointAdapter)
    ]
    assert len(adapters) == cfg.expert_num_layers
    assert all(
        isinstance(adapter.fused_layer, MoTFusedLayer)
        for adapter in adapters
    )
