"""设备 / 分布式后端抽象（昇腾适配的接缝）。

所有 device、dtype、distributed backend 的取值都必须经过本模块，
业务代码中不允许出现硬编码的 "cuda" / torch.device("cuda:0")。

当前状态：
- cuda 分支：完整可用（nccl + torch.cuda.amp）。
- npu 分支：接口已留好，依赖 torch_npu；后续在这里补算子差异
  （如 SDPA 的可用性、特定 kernel 的 fallback），模型代码不需要改。
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from collections.abc import Generator

import torch
import torch.distributed as dist


class DeviceType(str, Enum):
    """支持的设备类型。新增硬件在这里加分支。"""

    CPU = "cpu"
    CUDA = "cuda"
    NPU = "npu"


@dataclass
class DeviceContext:
    """一次训练的设备上下文：设备类型、本卡 device、分布式拓扑。"""

    type: DeviceType
    device: torch.device
    rank: int = 0        # 全局 rank；单卡时为 0
    local_rank: int = 0  # 节点内 rank，决定用哪张卡
    world_size: int = 1  # 全局进程数；单卡时为 1

    @property
    def is_distributed(self) -> bool:
        """是否处于多进程分布式模式。"""
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """是否为主进程（日志、checkpoint 只让它做）。"""
        return self.rank == 0


def _torch_device_module(device_type: DeviceType):
    """返回 torch 侧的设备模块（torch.cuda / torch.npu）。"""
    if device_type is DeviceType.CPU:
        raise ValueError("CPU 设备没有 torch.cuda/torch.npu 设备模块")
    if device_type is DeviceType.NPU:
        # torch_npu 导入后会在 torch 上注册 npu 设备
        import torch_npu  # noqa: F401

        return torch.npu
    return torch.cuda


def get_dist_backend(device_type: DeviceType) -> str:
    """按设备类型选择 torch.distributed 后端。"""
    if device_type is DeviceType.NPU:
        return "hccl"  # 昇腾集合通信库
    return "nccl"


def init_device(device_type: str, local_rank: int | None = None) -> DeviceContext:
    """初始化设备与（可选的）分布式进程组，返回 DeviceContext。

    torchrun 启动时通过环境变量 RANK / LOCAL_RANK / WORLD_SIZE 传入拓扑；
    单进程调试时不带这些变量，退化为单卡。
    """
    dtype = DeviceType(device_type)
    if dtype is DeviceType.CPU:
        return DeviceContext(type=dtype, device=torch.device("cpu"))
    mod = _torch_device_module(dtype)

    env_rank = int(os.environ.get("RANK", "0"))
    env_local = int(os.environ.get("LOCAL_RANK", "0"))
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank is not None:
        env_local = local_rank

    mod.set_device(env_local)
    device = torch.device(dtype.value, env_local)

    ctx = DeviceContext(type=dtype, device=device, rank=env_rank,
                        local_rank=env_local, world_size=env_world)
    if ctx.is_distributed and not dist.is_initialized():
        dist.init_process_group(backend=get_dist_backend(dtype))
    return ctx


def cleanup_distributed() -> None:
    """训练结束时销毁进程组（幂等）。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(ctx: DeviceContext) -> None:
    """分布式屏障；单进程时是 no-op。"""
    if ctx.is_distributed:
        dist.barrier()


def get_amp_dtype(name: str) -> torch.dtype:
    """把配置里的 dtype 字符串映射到 torch.dtype。"""
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16,
               "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"不支持的 dtype: {name}")
    return mapping[name]


@contextmanager
def autocast(ctx: DeviceContext, dtype: torch.dtype) -> Generator[None]:
    """按设备类型开混合精度上下文；float32 训练时是 no-op。"""
    if dtype is torch.float32:
        with nullcontext():
            yield
        return
    # torch.autocast 的 device_type 参数对 cuda/npu 通用（npu 需 torch_npu 注册）
    with torch.autocast(device_type=ctx.type.value, dtype=dtype):
        yield


def synchronize(ctx: DeviceContext) -> None:
    """等待指定设备上的全部计算完成；CPU 是 no-op。"""
    if ctx.type is DeviceType.CPU:
        return
    _torch_device_module(ctx.type).synchronize()


def reset_peak_memory_stats(ctx: DeviceContext) -> None:
    """重置指定设备的峰值内存统计；CPU 是 no-op。"""
    if ctx.type is DeviceType.CPU:
        return
    _torch_device_module(ctx.type).reset_peak_memory_stats()


def max_memory_allocated(ctx: DeviceContext) -> int:
    """返回指定设备的峰值分配字节数；CPU 固定为 0。"""
    if ctx.type is DeviceType.CPU:
        return 0
    return int(_torch_device_module(ctx.type).max_memory_allocated())
