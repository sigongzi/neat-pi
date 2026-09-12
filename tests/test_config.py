"""config.py 的 YAML 解析、默认值与 FSDP 训练配置校验测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from neat_pi.config import Config, FSDPConfig, TrainingConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并断言顶层是映射。"""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_default_config_uses_planned_fsdp_contract() -> None:
    """默认训练配置启用计划 06 的 first-pass FSDP 设置。"""
    cfg = Config()
    assert cfg.training.fsdp == FSDPConfig(
        sharding_strategy="full_shard",
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
        forward_prefetch=False,
        backward_prefetch="backward_pre",
        activation_checkpointing=True,
        state_dict_type="full",
        cpu_offload=False,
    )
    assert cfg.training.gradient_clip_norm == 1.0
    assert cfg.training.grad_accum_steps == 1
    assert cfg.training.resume is False
    assert cfg.training.seed == 1000


def test_load_config_accepts_existing_libero_training_config() -> None:
    """现有 LIBERO 训练 YAML 在新增字段后仍可解析。"""
    cfg = load_config(ROOT / "configs" / "pi05_libero.yaml")
    cfg.validate()


def test_load_config_persists_fsdp_and_training_fields(tmp_path: Path) -> None:
    """YAML 中的新字段写入 dataclass，并在加载后通过校验。"""
    raw = _load_yaml(ROOT / "configs" / "pi05_libero.yaml")
    raw["training"]["fsdp"] = {
        "sharding_strategy": "shard_grad_op",
        "use_orig_params": True,
        "sync_module_states": False,
        "limit_all_gathers": False,
        "forward_prefetch": False,
        "backward_prefetch": "backward_post",
        "activation_checkpointing": False,
        "state_dict_type": "full",
        "cpu_offload": False,
    }
    raw["training"]["gradient_clip_norm"] = 0.5
    raw["training"]["grad_accum_steps"] = 2
    raw["training"]["resume"] = True
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(path)
    assert cfg.training.fsdp.sharding_strategy == "shard_grad_op"
    assert cfg.training.fsdp.backward_prefetch == "backward_post"
    assert cfg.training.fsdp.sync_module_states is False
    assert cfg.training.gradient_clip_norm == 0.5
    assert cfg.training.grad_accum_steps == 2
    assert cfg.training.resume is True


@pytest.mark.parametrize("field,value", [
    ("sharding_strategy", "zero3"),
    ("backward_prefetch", "before"),
    ("state_dict_type", "sharded"),
])
def test_invalid_fsdp_string_is_rejected(field: str, value: str) -> None:
    """不支持的 FSDP 策略字符串在配置期失败。"""
    fsdp = FSDPConfig()
    setattr(fsdp, field, value)
    with pytest.raises(ValueError, match=field):
        fsdp.validate()


def test_forward_prefetch_conflicts_with_activation_checkpointing() -> None:
    """第一版禁止 forward prefetch 与 activation checkpointing 同时开启。"""
    fsdp = FSDPConfig()
    fsdp.forward_prefetch = True
    fsdp.activation_checkpointing = True
    with pytest.raises(ValueError, match="forward_prefetch"):
        fsdp.validate()


@pytest.mark.parametrize("field,value", [
    ("max_steps", 0),
    ("grad_accum_steps", -1),
    ("log_every", "10"),
    ("save_every", 0),
    ("keep_last_n", -1),
    ("keep_every", "3"),
    ("lr", -0.1),
    ("lr_end", -1.0),
    ("lr_end", "2.5e-6"),
    ("warmup_steps", -1),
    ("warmup_steps", "10"),
    ("ema_decay", 0.0),
    ("ema_decay", 1.0),
    ("ema_decay", -0.1),
    ("ema_decay", "0.9"),
    ("weight_decay", "0"),
    ("gradient_clip_norm", 0),
    ("use_dummy_model", 1),
    ("resume", "false"),
    ("seed", -1),
    ("output_dir", ""),
])
def test_invalid_training_value_is_rejected(field: str, value: Any) -> None:
    """非法训练超参在配置期失败，而不是训练循环中失败。"""
    training = TrainingConfig()
    setattr(training, field, value)
    with pytest.raises(ValueError, match=field):
        training.validate()


def test_unknown_training_config_key_is_rejected(tmp_path: Path) -> None:
    """新增配置仍保持未知键显式失败，避免双源和拼写漂移。"""
    raw = _load_yaml(ROOT / "configs" / "pi05_libero.yaml")
    raw["training"]["unknown_option"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown_option"):
        load_config(path)
