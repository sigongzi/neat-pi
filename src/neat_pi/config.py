"""强类型配置系统：YAML -> dataclass。

configs/*.yaml 是唯一的超参数来源；本模块把它解析成 dataclass，
提供 IDE 补全和类型安全。新增配置项时先在这里加字段。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DeviceConfig:
    """设备相关配置；type 决定 device/backend.py 走 cuda 还是 npu 分支。"""

    type: str = "cuda"       # cuda | npu
    dtype: str = "bfloat16"  # 混合精度训练 dtype


@dataclass
class ModelConfig:
    """模型结构超参数。取值以 ref/openpi 的 pi05 为准。"""

    image_size: int = 224
    num_cameras: int = 2      # 相机路数（LIBERO: agentview + wrist）
    action_horizon: int = 50  # 一次预测的动作步数
    action_dim: int = 32      # 动作维度（不足处 padding）
    state_dim: int = 32       # 机器人状态维度


@dataclass
class DataConfig:
    """数据管线配置（LeRobot v3.1 格式）。"""

    repo_id: str = "libero"
    root: str = ""
    batch_size: int = 8
    num_workers: int = 4
    preprocessor_path: str = ""  # 本项目 policy_preprocessor.json 的文件路径
    postprocessor_path: str = ""  # 本项目 policy_postprocessor.json（推理侧 unnormalize + delta→指令）
    tokenizer_path: str = ""     # checkpoint 的 tokenizer.json
    max_token_len: int = 200     # pi05 的 prompt token 长度（openpi pi0_config）
    video_backend: str = "pyav"  # 视频解码后端；显式指定以跳过 torchcodec 探测告警


@dataclass
class FSDPConfig:
    """FSDP 策略配置，由 training/fsdp.py 消费。"""

    sharding_strategy: str = "full_shard"
    use_orig_params: bool = True


@dataclass
class TrainingConfig:
    """训练超参数与 FSDP 子配置。"""

    pretrained: str | None = None  # pi05 checkpoint 路径；None = 从零训练
    max_steps: int = 10000
    lr: float = 2.5e-5
    weight_decay: float = 0.0
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "outputs/pi05_libero"
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)


@dataclass
class Config:
    """顶层配置，对应一个 YAML 文件。"""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    """把 dict 递归写入 dataclass，未知键直接报错以便尽早发现配置笔误。"""
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in values.items():
        if key not in fields:
            raise ValueError(f"未知配置项: {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path) -> Config:
    """从 YAML 加载配置并校验键名。"""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    cfg = Config()
    _update_dataclass(cfg, raw)
    return cfg
