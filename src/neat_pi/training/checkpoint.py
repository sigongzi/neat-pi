"""FSDP full state checkpoint 的保存与恢复。

模型权重导出为 canonical Pi05 safetensors；optimizer 保存为 rank0 聚合的
full optimizer state。这样单卡 eval 可以直接消费模型权重，训练恢复则通过
FSDP 的 optimizer state transform 回到分片 optimizer。
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
import shutil
from typing import Any

from loguru import logger
import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)

from neat_pi.device.backend import DeviceContext
from neat_pi.model.weights import (
    canonical_pi05_state_dict,
    load_pi05_state_dict_safetensors,
)


_CHECKPOINT_METADATA = "metadata.json"
_CHECKPOINT_OPTIMIZER = "optimizer.pt"
_CHECKPOINT_WEIGHTS = "model.safetensors"
_CHECKPOINT_EMA_WEIGHTS = "ema.safetensors"


def _parse_checkpoint_step(path: Path) -> int | None:
    """从 step_XXXXXXX 目录名解析 step；不合法返回 None。"""
    if not path.name.startswith("step_"):
        return None
    suffix = path.name[len("step_"):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _is_complete_checkpoint(path: Path) -> bool:
    """checkpoint 三文件齐全才算完整；半成品目录不参与清理。"""
    return ((path / _CHECKPOINT_METADATA).is_file()
            and (path / _CHECKPOINT_OPTIMIZER).is_file()
            and (path / _CHECKPOINT_WEIGHTS).is_file())


def prune_checkpoints(
    output_dir: str | Path,
    keep_last_n: int,
    keep_every: int,
    protect: str | Path | None = None,
) -> list[Path]:
    """按保留策略删除旧 checkpoint，返回实际删除的目录列表。

    只考虑三文件齐全的 step_* 目录（半成品与未知目录不动，留给人工
    处理）；protect（latest.json 指向的目录）、里程碑（keep_every > 0
    且 step % keep_every == 0）与最近 keep_last_n 个永远保留。
    keep_last_n 与 keep_every 均为 0 时不清理（默认行为，与历史上
    全保留一致）。删除是尽力而为：单个目录删除失败（权限 / 占用等）
    时跳过并保留，等下次保存重试，不中断训练。
    """
    if keep_last_n <= 0 and keep_every <= 0:
        return []
    root = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("step_*"):
        step = _parse_checkpoint_step(path)
        if step is not None and _is_complete_checkpoint(path):
            candidates.append((step, path))
    candidates.sort()

    keep: set[Path] = set()
    if keep_last_n > 0:
        keep.update(path.resolve() for _, path in candidates[-keep_last_n:])
    if keep_every > 0:
        keep.update(
            path.resolve() for step, path in candidates
            if step % keep_every == 0)
    if protect is not None:
        keep.add(Path(protect).resolve())

    deleted: list[Path] = []
    for _, path in candidates:
        if path.resolve() in keep:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        deleted.append(path)
    return deleted


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    ctx: DeviceContext,
    fsdp_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    keep_last_n: int = 0,
    keep_every: int = 0,
    ema_model: nn.Module | None = None,
) -> None:
    """保存 FSDP full state checkpoint；rank0 负责写盘与清理。

    所有 rank 都必须调用本函数：FSDP 的 full model / optimizer state
    导出涉及 collective。rank0_only 配置会让非 rank0 得到空字典。
    写盘与 latest.json 更新完成后，rank0 按 (keep_last_n, keep_every)
    清理旧 checkpoint；刚写好的目录作为 protect 永不删除。默认 0/0
    不清理。

    `ema_model` 非 None 时额外写 `ema.safetensors`（EMA 影子权重，与
    `model.safetensors` 同为 canonical Pi05 格式）；None 时不写该文件，
    保持 checkpoint 布局与旧版本完全一致。
    """
    state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    optimizer_state_config = FullOptimStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    if isinstance(model, FullyShardedDataParallel):
        with FullyShardedDataParallel.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config,
            optimizer_state_config,
        ):
            model_state = model.state_dict()
            optimizer_state = FullyShardedDataParallel.optim_state_dict(
                model,
                optimizer,
            )
    else:
        if not ctx.is_main_process:
            return
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

    ema_state: dict[str, Any] | None = None
    if ema_model is not None:
        if isinstance(ema_model, FullyShardedDataParallel):
            # 影子独立 wrap，gather 是 collective，所有 rank 都要进上下文
            with FullyShardedDataParallel.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config,
                optimizer_state_config,
            ):
                ema_state = ema_model.state_dict()
        elif ctx.is_main_process:
            ema_state = ema_model.state_dict()

    if not ctx.is_main_process:
        return

    checkpoint_dir = Path(output_dir) / f"step_{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata: dict[str, Any] = {
        "step": step,
        "world_size": ctx.world_size,
        "fsdp_config": dict(fsdp_config or {}),
    }
    checkpoint_metadata.update(metadata or {})
    metadata_path = checkpoint_dir / _CHECKPOINT_METADATA
    metadata_path.write_text(
        json.dumps(checkpoint_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(optimizer_state, checkpoint_dir / _CHECKPOINT_OPTIMIZER)

    from neat_pi.model.weights import save_pi05_state_dict_safetensors

    save_pi05_state_dict_safetensors(
        model_state,
        checkpoint_dir / _CHECKPOINT_WEIGHTS,
        metadata={
            "step": step,
            "world_size": ctx.world_size,
        },
    )
    if ema_state is not None:
        save_pi05_state_dict_safetensors(
            ema_state,
            checkpoint_dir / _CHECKPOINT_EMA_WEIGHTS,
            metadata={
                "step": step,
                "world_size": ctx.world_size,
            },
        )

    latest_path = Path(output_dir) / "latest.json"
    latest_path.write_text(
        json.dumps({
            "step": step,
            "checkpoint": checkpoint_dir.name,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    deleted = prune_checkpoints(
        output_dir, keep_last_n, keep_every, protect=checkpoint_dir)
    if deleted:
        logger.info(
            "checkpoint 清理（keep_last_n={} keep_every={}）：删除 {}",
            keep_last_n,
            keep_every,
            [path.name for path in deleted],
        )


def latest_checkpoint_path(output_dir: str) -> Path | None:
    """返回 output_dir 中 latest.json 指向的最新 checkpoint 目录。"""
    root = Path(output_dir)
    latest_path = root / "latest.json"
    if not latest_path.is_file():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_name = latest.get("checkpoint")
    if not isinstance(checkpoint_name, str):
        raise ValueError(f"latest.json 缺少 checkpoint 字段: {latest_path}")
    checkpoint_path = root / checkpoint_name
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"latest checkpoint 不存在: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint_weights(model: nn.Module, path: str | Path) -> int:
    """从训练 checkpoint 恢复模型权重（canonical safetensors），返回 step。

    必须在 FSDP 包装前的未分片模型上调用：包装后的参数是 rank 本地分片，
    state_dict 视图不支持就地写入完整张量——单进程不复现，多卡会报错或
    静默写坏。多卡训练应由 run_training 走 load_pi05_weights_distributed
    （rank0 读取 + 广播）；本函数用于单进程 / 每 rank 独立读取的场景。
    """
    weights_path = Path(path) / _CHECKPOINT_WEIGHTS
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少权重文件: {weights_path}")
    load_pi05_state_dict_safetensors(model, weights_path)
    metadata = load_checkpoint_metadata(path)
    return int(metadata.get("step", 0))


def load_checkpoint_ema(model: nn.Module, path: str | Path) -> bool:
    """从 checkpoint 恢复 EMA 影子权重（canonical safetensors）到 model。

    调用约束与 `load_checkpoint_weights` 相同：FSDP 包装前的未分片模型。
    checkpoint 无 ema.safetensors（旧 checkpoint 或训练时 ema_decay=null）
    返回 False 且不改动 model，装配层据此让影子从训练参数起步（openpi
    的 EMA 初始化语义）。
    """
    weights_path = Path(path) / _CHECKPOINT_EMA_WEIGHTS
    if not weights_path.is_file():
        return False
    load_pi05_state_dict_safetensors(model, weights_path)
    return True


def load_checkpoint_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
) -> None:
    """从训练 checkpoint 恢复优化器状态，支持 FSDP 分片优化器回填。

    须在 optimizer 创建之后调用（FSDP 训练中即 wrap 后）；FSDP 模型经
    optim_state_dict_to_load 把 rank0 聚合的 full optimizer state 映射
    回分片形态。
    """
    optimizer_path = Path(path) / _CHECKPOINT_OPTIMIZER
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"checkpoint 缺少优化器文件: {optimizer_path}")
    optimizer_state = torch.load(
        optimizer_path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(model, FullyShardedDataParallel):
        optimizer_state = FullyShardedDataParallel.optim_state_dict_to_load(
            model,
            optimizer,
            optimizer_state,
        )
    optimizer.load_state_dict(optimizer_state)


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """读取 checkpoint 的训练循环元数据。"""
    metadata_path = Path(path) / _CHECKPOINT_METADATA
    if not metadata_path.is_file():
        return {}
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
