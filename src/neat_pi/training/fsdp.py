"""FSDP 策略：多机训练的核心。

关键点：auto_wrap 以 mot.MoTLayer 为最小单位。
pi05 里 VLM 与动作专家共享 attention，已经重构为 MoTLayer 内部
"两专家私有参数 + 层内共享 attention 计算"的形态（见 model/mot.py），
因此整层分片不会切断任何跨模块的参数引用。

视觉编码器 SigLIP 是独立的连续子图，按普通 transformer block 包裹即可。
"""

from __future__ import annotations

import functools

from torch import nn

from neat_pi.config import FSDPConfig
from neat_pi.device.backend import DeviceContext
from neat_pi.model.mot import MoTLayer


def build_auto_wrap_policy():
    """构造 auto_wrap 策略：每个 MoTLayer 单独成为一个 FSDP 单元。"""
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    # 后续可把 SigLIP 的 encoder block 类型也加入 wrap 集合
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={MoTLayer},
    )


def wrap_model_fsdp(model: nn.Module, cfg: FSDPConfig,
                    ctx: DeviceContext) -> nn.Module:
    """按配置把模型包成 FSDP。

    TODO: 补全 sharding_strategy 映射（full_shard -> ShardingStrategy.FULL_SHARD）、
    混合精度策略（走 device.backend 的 dtype，不硬编码）、
    以及 device_id（用 ctx.device，NPU 适配时同样经 backend 层）。
    单卡调试时直接返回原模型。
    """
    if not ctx.is_distributed:
        return model
    raise NotImplementedError("待实现：FSDP 包装")
