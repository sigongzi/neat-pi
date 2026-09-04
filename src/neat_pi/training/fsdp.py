"""FSDP 包装（多机训练；结构重构后尚未重做，当前为待完成骨架）。

原实现按旧的 MoTLayer 结构把整层作为 auto_wrap 单位；模型已重构为
FastWAM 式 MoT + Expert 容器（见 model/mot.py / model/expert.py），
auto_wrap 的最小单位需随 MoT 的 fused 前向（逐层配对 + 共享 SDPA）落地后
重新确定，候选是各专家内部逐层模块或 MoT 的单层配对单元。视觉编码器
SigLIP 是独立连续子图，届时按普通 transformer block 包裹即可。
"""

from __future__ import annotations

from torch import nn

from neat_pi.config import FSDPConfig
from neat_pi.device.backend import DeviceContext


def wrap_model_fsdp(model: nn.Module, cfg: FSDPConfig,
                    ctx: DeviceContext) -> nn.Module:
    """按配置把模型包成 FSDP；单卡（非分布式）时原样返回。

    TODO: MoT 的 fused 前向与 auto_wrap 目标类型确定后再补——
    sharding 策略映射（full_shard -> ShardingStrategy.FULL_SHARD）、
    混合精度（走 device.backend 的 dtype，不硬编码）、device_id
    （用 ctx.device，NPU 适配时同样经 backend 层）。
    """
    if not ctx.is_distributed:
        return model
    raise NotImplementedError("待实现：MoT 结构确定后的 FSDP 包装")
