"""用当前仓库的 Pi05 实现执行 LIBERO rollout 评测。

LeRobot 只提供 LIBERO 环境构建与步进；观测转换、prompt 合同、动作
queue 和成功率统计全部走本仓库的 eval adapter 与 Pi05 前向。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from lerobot.envs.libero import (TASK_SUITE_MAX_STEPS, LiberoEnv,
                                 create_libero_envs)
from loguru import logger

from neat_pi.config import ModelConfig
from neat_pi.device import backend
from neat_pi.device.backend import DeviceContext
from neat_pi.eval.action import LiberoActionAdapter
from neat_pi.eval.config import resolve_config_path
from neat_pi.eval.observation import LiberoObservationAdapter, Pi05EvalObservation
from neat_pi.eval.prompt import LiberoPromptTokenizerAdapter, Pi05EvalLanguage
from neat_pi.model.pi05 import Pi05
from neat_pi.typing import ActionBHD, typechecked

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "eval_pi05_libero_finetuned.yaml"


@dataclass(frozen=True)
class EpisodeRecord:
    """一个 LIBERO rollout 的验收指标和失败诊断。"""

    suite: str
    task_id: int
    task: str
    episode_index: int
    seed: int
    init_state_id: int | None
    success: bool
    steps: int
    model_calls: int
    wall_time: float
    peak_memory: int
    mean_model_latency: float
    error: str | None = None


@dataclass(frozen=True)
class EvalRunner:
    """按 checkpoint 合同组装模型 adapter 并执行单个 episode。"""

    model: Pi05
    observation_adapter: LiberoObservationAdapter
    prompt_adapter: LiberoPromptTokenizerAdapter
    action_adapter: LiberoActionAdapter
    flow_steps: int

    @typechecked
    def sample(self, observation: Pi05EvalObservation,
               language: Pi05EvalLanguage) -> ActionBHD:
        """用当前仓库 Pi05 采样一个动作 chunk。"""
        if self.model.ctx is None:
            raise ValueError("Pi05.ctx 未初始化，无法确定评测设备")
        device = self.model.ctx.device
        images = [
            image.to(device=device, dtype=self.model.amp_dtype)
            for image in observation.model_inputs()[0]
        ]
        image_masks = [
            mask.to(device=device)
            for mask in observation.model_inputs()[1]
        ]
        return self.model.sample_actions(
            images,
            image_masks,
            language.token_ids.to(device=device),
            language.attention_mask.to(device=device),
            num_steps=self.flow_steps,
        )

def parse_args() -> argparse.Namespace:
    """解析评测 CLI；未给出的 rollout 参数回退到 eval YAML。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="eval YAML 配置路径")
    parser.add_argument("--suite", action="append", default=None,
                        help="覆盖 YAML suites；可重复传入")
    parser.add_argument("--task-id", action="append", type=int, default=None,
                        dest="task_ids", help="覆盖 YAML task_ids；可重复传入")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖 YAML episodes_per_task")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖每个 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖 YAML seed")
    return parser.parse_args()


def load_eval_config(path: Path) -> dict[str, Any]:
    """读取并基础校验 eval YAML。"""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"eval 配置必须是映射: {path}")
    if config.get("actions", {}).get("normalization") != "MEAN_STD":
        raise ValueError("旧 pi05 LIBERO checkpoint 只支持 MEAN_STD 动作归一化")
    if config.get("env", {}).get("success_key") != "is_success":
        raise ValueError("LIBERO runner 的 success_key 必须是 is_success")
    return config


def build_model_config(config: dict[str, Any]) -> ModelConfig:
    """从 eval YAML 构造模型结构配置并校验合同一致性。"""
    model = config["model"]
    actions = config["actions"]
    cameras = config["cameras"]
    model_config = ModelConfig(
        image_size=int(model["image_size"]),
        num_cameras=int(model["num_cameras"]),
        action_horizon=int(model["action_horizon"]),
        action_dim=int(model["max_action_dim"]),
    )
    real_camera_count = len(cameras["real"])
    empty_camera_count = 1 if cameras.get("empty") else 0
    if model_config.num_cameras != real_camera_count + empty_camera_count:
        raise ValueError("model.num_cameras 必须等于真实相机数 + empty 相机数")
    if model_config.action_dim != int(model["max_action_dim"]):
        raise ValueError("模型 action_dim 必须与 max_action_dim 一致")
    if actions["raw_dim"] >= model_config.action_dim:
        raise ValueError("actions.raw_dim 必须小于模型 action_dim")
    return model_config


def _make_single_env(
        factories: Sequence[Callable[[], LiberoEnv]]) -> LiberoEnv:
    """从 LeRobot env factory 中构建一个非 vector 的 LiberoEnv。"""
    if len(factories) != 1:
        raise ValueError(f"当前 runner 只支持 batch=1，实际收到 {len(factories)} 个 env")
    return factories[0]()


def build_environments(
        config: dict[str, Any], suite_names: Sequence[str],
        task_ids: Sequence[int] | None) -> dict[str, dict[int, LiberoEnv]]:
    """用 LeRobot LIBERO 构建路径为每个 task 创建一个单环境实例。"""
    env_config = config["env"]
    height, width = (int(value) for value in env_config["observation_size"])
    camera_sources = ",".join(
        str(camera["source"]) for camera in config["cameras"]["real"])
    camera_name_mapping = {
        str(camera["source"]): str(camera["source"])
        for camera in config["cameras"]["real"]
    }
    return create_libero_envs(
        task=",".join(suite_names),
        n_envs=1,
        gym_kwargs={
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "observation_height": height,
            "observation_width": width,
            "task_ids": list(task_ids) if task_ids is not None else None,
        },
        camera_name=camera_sources,
        init_states=bool(env_config["use_init_states"]),
        env_cls=_make_single_env,
        control_mode=str(env_config["control_mode"]),
        camera_name_mapping=camera_name_mapping,
        episode_length=None,
    )


def build_model(config: dict[str, Any], ctx: DeviceContext) -> Pi05:
    """在目标设备上直接加载 bf16/fp32 的当前仓库 Pi05。"""
    checkpoint = resolve_config_path(
        str(config["checkpoint"]["path"]),
        config_root=str(ROOT),
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在: {checkpoint}")
    amp_dtype = backend.get_amp_dtype(str(config["device"]["dtype"]))
    model = Pi05.from_pretrained(
        str(checkpoint),
        cfg=build_model_config(config),
        device=ctx.device,
        dtype=amp_dtype,
    )
    model.ctx = ctx
    model.amp_dtype = amp_dtype
    model.eval()
    model.requires_grad_(False)
    return model


def build_runner(config: dict[str, Any], ctx: DeviceContext,
                 model: Pi05, config_root: Path) -> EvalRunner:
    """构造观测、prompt 和动作 adapter 并组装 runner。"""
    return EvalRunner(
        model=model,
        observation_adapter=LiberoObservationAdapter.from_config(
            config["cameras"], int(config["model"]["image_size"])),
        prompt_adapter=LiberoPromptTokenizerAdapter.from_config(
            config, config_root=config_root),
        action_adapter=LiberoActionAdapter.from_config(
            config, config_root=config_root),
        flow_steps=int(config["actions"]["flow_steps"]),
    )


def select_task_ids(config: dict[str, Any],
                    cli_task_ids: list[int] | None) -> list[int] | None:
    """合并 YAML 与 CLI 的 task id 选择。"""
    if cli_task_ids is not None:
        return sorted(set(cli_task_ids))
    configured = config["env"].get("task_ids")
    return None if configured is None else sorted({int(value) for value in configured})


def episode_limit(config: dict[str, Any], suite: str,
                  override: int | None) -> int:
    """返回一个 episode 的最大步数；空值使用 LeRobot suite 约定。"""
    configured = config["env"].get("max_steps")
    limit = override if override is not None else configured
    if limit is None:
        limit = TASK_SUITE_MAX_STEPS.get(suite)
    if limit is None:
        raise ValueError(f"未知 LIBERO suite 的 max steps: {suite}")
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"max steps 必须大于 0，实际为 {limit}")
    return limit


def sample_once(runner: EvalRunner, observation: Pi05EvalObservation,
                task: str) -> tuple[np.ndarray, float]:
    """采样新 chunk（或在 queue 中取一步）并返回可执行动作。"""
    if runner.action_adapter.pending_count == 0:
        language = runner.prompt_adapter.build(observation, task)
        started = time.perf_counter()
        model_actions = runner.sample(observation, language)
        if runner.model.ctx is not None:
            backend.synchronize(runner.model.ctx)
        latency = time.perf_counter() - started
        runner.action_adapter.submit(model_actions)
    else:
        latency = 0.0
    action_tensor = runner.action_adapter.pop_action()
    action = action_tensor.detach().cpu().numpy().astype(np.float32)
    return action, latency


def run_episode(
        runner: EvalRunner, env: LiberoEnv, suite: str, task_id: int,
        episode_index: int, seed: int, max_steps: int,
        ctx: DeviceContext) -> EpisodeRecord:
    """执行一个 episode；失败时返回带 error 的记录而不中断评测。"""
    started = time.perf_counter()
    steps = 0
    model_latency_total = 0.0
    init_state_id: int | None = None
    success = False
    error: str | None = None
    backend.reset_peak_memory_stats(ctx)
    try:
        runner.action_adapter.reset()
        if env.init_states:
            init_state_id = int(env.init_state_id)
        observation, _ = env.reset()
        task = str(env.task_description)
        terminated = False
        truncated = False
        while not terminated and not truncated and steps < max_steps:
            converted = runner.observation_adapter.convert(observation)
            action, latency = sample_once(runner, converted, task)
            model_latency_total += latency
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            success = bool(info.get("is_success", False))
    except Exception as exc:  # noqa: BLE001 - 单 episode 失败必须写入 JSONL 后继续
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("episode failed | suite={} task={} episode={}",
                         suite, task_id, episode_index)
    peak_memory = backend.max_memory_allocated(ctx)
    model_calls = runner.action_adapter.model_calls
    mean_latency = model_latency_total / model_calls if model_calls else 0.0
    return EpisodeRecord(
        suite=suite,
        task_id=task_id,
        task=str(env.task_description),
        episode_index=episode_index,
        seed=seed,
        init_state_id=init_state_id,
        success=success,
        steps=steps,
        model_calls=model_calls,
        wall_time=time.perf_counter() - started,
        peak_memory=peak_memory,
        mean_model_latency=mean_latency,
        error=error,
    )


def success_rate(records: Sequence[EpisodeRecord]) -> float:
    """计算成功率；空集合返回 0。"""
    if not records:
        return 0.0
    return sum(record.success for record in records) / len(records)


def build_summary(records: Sequence[EpisodeRecord],
                  total_wall_time: float) -> dict[str, Any]:
    """汇总全局、suite 和 task 级成功率与性能指标。"""
    suites: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        suites.setdefault(record.suite, []).append(record)
    per_suite = {
        suite: {
            "total_episodes": len(suite_records),
            "successes": sum(record.success for record in suite_records),
            "success_rate": success_rate(suite_records),
        }
        for suite, suite_records in suites.items()
    }
    per_task: dict[str, dict[str, dict[str, float | int]]] = {}
    for suite, suite_records in suites.items():
        per_task[suite] = {}
        task_ids = sorted({record.task_id for record in suite_records})
        for task_id in task_ids:
            task_records = [
                record for record in suite_records
                if record.task_id == task_id
            ]
            per_task[suite][str(task_id)] = {
                "total_episodes": len(task_records),
                "successes": sum(record.success for record in task_records),
                "success_rate": success_rate(task_records),
            }
    model_calls = sum(record.model_calls for record in records)
    latencies = [
        record.mean_model_latency for record in records
        if record.model_calls > 0
    ]
    return {
        "total_episodes": len(records),
        "successes": sum(record.success for record in records),
        "success_rate": success_rate(records),
        "failed_records": sum(record.error is not None for record in records),
        "per_suite_success": per_suite,
        "per_task_success": per_task,
        "total_wall_time": total_wall_time,
        "model_calls": model_calls,
        "mean_model_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "peak_memory": max(
            (record.peak_memory for record in records), default=0),
    }


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整评测流程，写 episode JSONL 和最终 summary。"""
    config = load_eval_config(args.config)
    config_root = args.config.parent
    configured_env = config["env"]
    suite_names = list(args.suite or configured_env["suites"])
    task_ids = select_task_ids(config, args.task_ids)
    episodes = int(
        args.episodes
        if args.episodes is not None
        else configured_env["episodes_per_task"]
    )
    seed = int(
        args.seed
        if args.seed is not None
        else configured_env.get("seed", 1000)
    )
    if episodes <= 0:
        raise ValueError(f"episodes 必须大于 0，实际为 {episodes}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ctx = backend.init_device(str(config["device"]["type"]))
    environments = build_environments(config, suite_names, task_ids)
    model = build_model(config, ctx)
    runner = build_runner(config, ctx, model, args.config.parent)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("device={} dtype={} parameters={:.3f}B",
                ctx.device, model.amp_dtype, parameter_count / 1e9)

    jsonl_path = resolve_config_path(
        str(config["output"]["jsonl_path"]), config_root=str(ROOT))
    summary_path = resolve_config_path(
        str(config["output"]["summary_path"]), config_root=str(ROOT))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[EpisodeRecord] = []
    started = time.perf_counter()
    try:
        with jsonl_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
            for suite in suite_names:
                for task_id in sorted(environments[suite]):
                    env = environments[suite][task_id]
                    limit = episode_limit(config, suite, args.max_steps)
                    for episode_index in range(episodes):
                        logger.info("rollout | suite={} task={} episode={}/{}",
                                    suite, task_id, episode_index + 1, episodes)
                        record = run_episode(
                            runner, env, suite, task_id, episode_index,
                            seed, limit, ctx,
                        )
                        records.append(record)
                        logger.info(
                            "episode finished | suite={} task={} episode={} "
                            "success={} steps={} model_calls={} wall_time={:.2f}s",
                            record.suite,
                            record.task_id,
                            record.episode_index,
                            record.success,
                            record.steps,
                            record.model_calls,
                            record.wall_time,
                        )
                        stream.write(json.dumps(
                            asdict(record), ensure_ascii=False,
                            separators=(",", ":")) + "\n")
                        stream.flush()
    finally:
        for task_map in environments.values():
            for env in task_map.values():
                env.close()

    summary = build_summary(records, time.perf_counter() - started)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("success_rate={:.3f} | jsonl={} | summary={}",
                summary["success_rate"], jsonl_path, summary_path)
    return summary


def main() -> None:
    """评测命令行入口。"""
    run_eval(parse_args())


if __name__ == "__main__":
    main()
