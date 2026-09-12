"""强类型配置系统：YAML -> dataclass。

configs/*.yaml 是唯一的超参数来源；本模块把它解析成 dataclass，
提供 IDE 补全和类型安全。新增配置项时先在这里加字段。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
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
    vision_hidden_dim: int = 1152
    vision_patch_size: int = 14
    vision_num_layers: int = 27
    vision_num_heads: int = 16
    vision_mlp_hidden_dim: int = 4304
    vocab_size: int = 257_152
    vlm_hidden_dim: int = 2048
    vlm_num_layers: int = 18
    vlm_num_heads: int = 8
    vlm_num_kv_heads: int = 1
    vlm_attn_head_dim: int = 256
    vlm_mlp_hidden_dim: int = 16384
    expert_hidden_dim: int = 1024
    expert_num_layers: int = 18
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 1
    expert_attn_head_dim: int = 256
    expert_mlp_hidden_dim: int = 4096


@dataclass
class DataConfig:
    """数据管线配置（LeRobot v3.1 格式）。"""

    repo_id: str = "libero"
    root: str = ""
    per_device_batch_size: int = 8  # 每个 rank / device 的 micro batch 大小
    num_workers: int = 4
    preprocessor_path: str = ""  # 本项目 policy_preprocessor.json 的文件路径
    postprocessor_path: str = ""  # 本项目 policy_postprocessor.json（推理侧 unnormalize + delta→指令）
    max_token_len: int = 200     # pi05 的 prompt token 长度（openpi pi0_config）
    video_backend: str = "pyav"  # 视频解码后端；显式指定以跳过 torchcodec 探测告警


@dataclass
class FSDPConfig:
    """FSDP 策略配置，由 training/fsdp.py 消费。"""

    sharding_strategy: str = "full_shard"
    use_orig_params: bool = True
    sync_module_states: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    backward_prefetch: str = "backward_pre"
    activation_checkpointing: bool = True
    state_dict_type: str = "full"
    cpu_offload: bool = False

    def validate(self) -> None:
        """校验 FSDP 配置取值，尽早暴露 YAML 或调用侧错误。"""
        sharding_strategies = {"full_shard", "shard_grad_op", "no_shard"}
        if self.sharding_strategy not in sharding_strategies:
            raise ValueError(
                f"fsdp.sharding_strategy 必须是 {sorted(sharding_strategies)}，"
                f"实际为 {self.sharding_strategy!r}")
        backward_prefetch_modes = {"backward_pre", "backward_post", "none"}
        if self.backward_prefetch not in backward_prefetch_modes:
            raise ValueError(
                "fsdp.backward_prefetch 必须是 "
                f"{sorted(backward_prefetch_modes)}，实际为 {self.backward_prefetch!r}")
        if self.state_dict_type != "full":
            raise ValueError(
                "fsdp.state_dict_type 第一阶段只支持 'full'，"
                f"实际为 {self.state_dict_type!r}")
        boolean_fields = {
            "use_orig_params": self.use_orig_params,
            "sync_module_states": self.sync_module_states,
            "limit_all_gathers": self.limit_all_gathers,
            "forward_prefetch": self.forward_prefetch,
            "activation_checkpointing": self.activation_checkpointing,
            "cpu_offload": self.cpu_offload,
        }
        for name, value in boolean_fields.items():
            if type(value) is not bool:
                raise ValueError(f"fsdp.{name} 必须是布尔值，实际为 {value!r}")
        if self.forward_prefetch and self.activation_checkpointing:
            raise ValueError(
                "fsdp.forward_prefetch 与 activation_checkpointing 暂不支持同时启用；"
                "请显式选择其中一种性能策略")

    def __post_init__(self) -> None:
        """构造时执行一次配置校验。"""
        self.validate()


@dataclass
class TrainingConfig:
    """训练超参数与 FSDP 子配置。"""

    pretrained: str | None = None  # pi05 checkpoint 路径；None = 从零训练
    use_dummy_model: bool = True   # True=用 DummyPi05 冒烟跑通训练循环；Pi05.predict_velocity 实现后改 False
    max_steps: int = 10000
    lr: float = 2.5e-5
    weight_decay: float = 0.0
    log_every: int = 10
    save_every: int = 1000
    keep_last_n: int = 0  # 按数量保留最近 n 个 checkpoint；0 = 不按数量保留
    keep_every: int = 0   # step % keep_every == 0 的里程碑永久保留；0 = 禁用
    output_dir: str = "outputs/pi05_libero"
    gradient_clip_norm: float | None = 1.0
    grad_accum_steps: int = 1
    resume: bool = False
    seed: int = 1000
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)

    def validate(self) -> None:
        """校验训练超参数及其与 FSDP 配置的组合。"""
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError(f"training.max_steps 必须是正整数，实际为 {self.max_steps!r}")
        if type(self.grad_accum_steps) is not int or self.grad_accum_steps <= 0:
            raise ValueError(
                f"training.grad_accum_steps 必须是正整数，实际为 {self.grad_accum_steps!r}")
        for name, value in (("log_every", self.log_every),
                            ("save_every", self.save_every)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"training.{name} 必须是正整数，实际为 {value!r}")
        # checkpoint 保留策略：0 是合法值（关闭对应保留维度），负数 / 非整数非法
        for name, value in (("keep_last_n", self.keep_last_n),
                            ("keep_every", self.keep_every)):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"training.{name} 必须是非负整数，实际为 {value!r}")
        for name, value in (("lr", self.lr), ("weight_decay", self.weight_decay)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"training.{name} 必须是有限非负数，实际为 {value!r}")
        if self.gradient_clip_norm is not None and (
                not isinstance(self.gradient_clip_norm, (int, float))
                or not math.isfinite(self.gradient_clip_norm)
                or self.gradient_clip_norm <= 0):
            raise ValueError(
                "training.gradient_clip_norm 必须是正有限数或 null，"
                f"实际为 {self.gradient_clip_norm!r}")
        for name, value in (("use_dummy_model", self.use_dummy_model),
                            ("resume", self.resume)):
            if type(value) is not bool:
                raise ValueError(f"training.{name} 必须是布尔值，实际为 {value!r}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError(f"training.seed 必须是非负整数，实际为 {self.seed!r}")
        if not isinstance(self.pretrained, (str, type(None))):
            raise ValueError(
                f"training.pretrained 必须是路径字符串或 null，实际为 {self.pretrained!r}")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError(
                f"training.output_dir 必须是非空路径字符串，实际为 {self.output_dir!r}")

    def __post_init__(self) -> None:
        """构造时校验训练参数并同步校验 FSDP 子配置。"""
        self.fsdp.validate()
        self.validate()


@dataclass
class Config:
    """顶层配置，对应一个 YAML 文件。"""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """校验完整顶层配置。"""
        self.training.fsdp.validate()
        self.training.validate()


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
    """从 YAML 加载配置并校验键名与取值。"""
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    cfg = Config()
    _update_dataclass(cfg, raw)
    cfg.validate()
    return cfg
