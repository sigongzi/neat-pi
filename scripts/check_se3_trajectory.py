"""SE(3) 执行链路轨迹自检：脚本化 chunk 驱动真环境，不加载模型。

验证 LiberoSe3ActionAdapter 的完整合同：chunk-wise SE(3) delta（相对
chunk 首帧）→ 绝对目标位姿 → 每步用新观测重参考 → OSC_POSE 指令。
假模型输出由「期望物理 delta 经统计量正变换」构造，submit 后可从指令
反推期望运动，因此轨迹是已知答案的：

- 相位 A：EE 绕自身 z 轴持续旋转（每步 ROT_STEP rad，chunk 内相对首帧
  累积 aa_k = (k+1)*ROT_STEP）；
- 相位 B：零运动，夹爪开/合交替（±1）；
- 相位 C：旋转 + 夹爪同时。

产物（--output 目录）：frames.mp4（或 PNG 序列）、trajectory.jsonl
（每步指令 / EE 位姿 / 夹爪 qpos）、summary.json 与控制台验收指标
（累计旋转角、夹爪 qpos 极差、平移漂移、指令幅值上界）。

用法：
    uv run python scripts/check_se3_trajectory.py --suite libero_spatial --task-id 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from neat_pi.data.se3 import aa_to_rot, rot_to_6d, rot_to_aa
from neat_pi.eval.action import LiberoSe3ActionAdapter

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

# 与 eval_pi05_libero_finetuned.yaml 一致的相机合同（脚本只需真实相机源）
_CAMERA_SOURCES = ["agentview_image", "robot0_eye_in_hand_image"]

# 脚本化相位参数
ROT_STEP = 0.05          # 每步目标旋转（rad），chunk 内相对首帧累积
GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0


def parse_args() -> argparse.Namespace:
    """解析 CLI 参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=4,
                        help="执行多少个脚本化 chunk（按 A/B/C 循环）")
    parser.add_argument("--execution-horizon", type=int, default=10,
                        help="每个 chunk 执行的步数（eval 合同为 10）")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "se3_traj_check")
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def build_scripted_chunk(phase: str, horizon: int) -> torch.Tensor:
    """构造一个 chunk 的期望物理 delta (E, 10) = [dpos, rot6D, gripper]。

    delta 相对 chunk 首帧：旋转相位 aa_k = (k+1)*ROT_STEP·ẑ（持续同向
    旋转），平移恒零，gripper 透传。
    """
    rows = []
    for k in range(horizon):
        if phase == "rotate":
            aa = [0.0, 0.0, (k + 1) * ROT_STEP]
            gripper = GRIPPER_CLOSE
        elif phase == "gripper":
            aa = [0.0, 0.0, 0.0]
            gripper = GRIPPER_OPEN if (k % 2 == 0) else GRIPPER_CLOSE
        elif phase == "rotate+gripper":
            aa = [0.0, 0.0, (k + 1) * ROT_STEP]
            gripper = GRIPPER_OPEN
        else:
            raise ValueError(f"未知相位: {phase}")
        rot6d = rot_to_6d(aa_to_rot(torch.tensor(aa, dtype=torch.float32)))
        rows.append(torch.cat([
            torch.zeros(3), rot6d, torch.tensor([gripper])]))
    return torch.stack(rows)


def fake_model_output(desired: torch.Tensor,
                      adapter: LiberoSe3ActionAdapter,
                      action_horizon: int,
                      model_action_dim: int) -> torch.Tensor:
    """期望物理 delta → QUANTILES 正变换 → 归一化模型输出（含 padding）。

    正变换公式与 lerobot 一致：n = 2*(x - q01)/(q99 - q01) - 1。
    """
    unnorm = adapter.postprocessor.steps[0]
    stats = unnorm.stats["action"]
    q01 = torch.as_tensor(stats["q01"]).float()
    q99 = torch.as_tensor(stats["q99"]).float()
    normalized = 2.0 * (desired - q01) / (q99 - q01).clamp_min(1e-8) - 1.0
    padded = torch.zeros(1, action_horizon, model_action_dim)
    padded[:, : desired.shape[0], :10] = normalized
    padded[:, desired.shape[0]:, :] = 999.0
    return padded


def state_from_obs(observation: dict[str, Any]) -> torch.Tensor:
    """环境观测 → (1, 8) state [pos(3), axis-angle(3), gripper qpos(2)]。"""
    robot = observation["robot_state"]
    pos = torch.as_tensor(np.asarray(robot["eef"]["pos"]), dtype=torch.float32)
    rot = torch.as_tensor(np.asarray(robot["eef"]["mat"]), dtype=torch.float32)
    aa = rot_to_aa(rot)
    gripper = torch.as_tensor(
        np.asarray(robot["gripper"]["qpos"]), dtype=torch.float32)
    return torch.cat([pos, aa, gripper]).unsqueeze(0)


def frame_from_obs(observation: dict[str, Any]) -> np.ndarray:
    """取第一路相机图像用于轨迹视频。"""
    pixels = observation["pixels"]
    return next(iter(pixels.values()))


def save_video(frames: list[np.ndarray], path: Path) -> None:
    """优先 mp4（imageio），不可用退化为 PNG 序列。"""
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=10, quality=8) as writer:
            for frame in frames:
                writer.append_data(frame)
        logger.info("视频: {}", path)
    except Exception as exc:  # noqa: BLE001 - 视频编码失败不掩盖轨迹结果
        logger.warning("mp4 编码失败（{}），改存 PNG 序列", exc)
        png_dir = path.parent / "frames"
        png_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames[:: max(1, len(frames) // 24)]):
            from PIL import Image
            Image.fromarray(frame).save(png_dir / f"frame_{index:03d}.png")
        logger.info("PNG 序列: {}", png_dir)


def main() -> None:
    """执行脚本化轨迹并打印验收指标。"""
    from lerobot.envs.libero import create_libero_envs

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapter = LiberoSe3ActionAdapter.from_config(
        {
            "model": {"action_horizon": 32, "max_action_dim": 32},
            "actions": {
                "execution_horizon": args.execution_horizon,
                "normalization": "QUANTILES",
            },
            "env": {"clip_bounds": [-1.0, 1.0]},
            "normalization": {
                "postprocessor_path": "pi05_libero_se3_unnormalize.json",
            },
        },
        config_root=str(CONFIG_DIR),
    )

    envs = create_libero_envs(
        task=args.suite,
        n_envs=1,
        gym_kwargs={
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "observation_height": 256,
            "observation_width": 256,
            "task_ids": [args.task_id],
        },
        camera_name=",".join(_CAMERA_SOURCES),
        init_states=True,
        env_cls=lambda factories: factories[0](),
        control_mode="relative",
        camera_name_mapping={name: name for name in _CAMERA_SOURCES},
        episode_length=None,
    )
    env = envs[args.suite][args.task_id]

    phases = (["rotate", "gripper", "rotate+gripper"]
              * (args.chunks // 3 + 1))[: args.chunks]
    frames: list[np.ndarray] = []
    log: list[dict[str, Any]] = []

    observation, _ = env.reset()
    task = str(env.task_description)
    logger.info("task: {} | 相位: {}", task, phases)
    try:
        for chunk_index, phase in enumerate(phases):
            state = state_from_obs(observation)
            desired = build_scripted_chunk(phase, args.execution_horizon)
            adapter.reset()
            adapter.submit(
                fake_model_output(desired, adapter, 32, 32), state)
            for step in range(args.execution_horizon):
                # 每步用最新观测重参考（合同核心：不缓存旧指令）
                state = state_from_obs(observation)
                command = adapter.pop_action(state)
                action = command.detach().cpu().numpy().astype(np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                robot = observation["robot_state"]
                frames.append(frame_from_obs(observation))
                log.append({
                    "chunk": chunk_index,
                    "phase": phase,
                    "step": step,
                    "command": action.tolist(),
                    "eef_pos": np.asarray(robot["eef"]["pos"]).tolist(),
                    "eef_rot": np.asarray(robot["eef"]["mat"]).flatten().tolist(),
                    "eef_aa": rot_to_aa(torch.as_tensor(
                        np.asarray(robot["eef"]["mat"]),
                        dtype=torch.float32)).tolist(),
                    "gripper_qpos": np.asarray(
                        robot["gripper"]["qpos"]).tolist(),
                })
                if terminated or truncated:
                    break
    finally:
        env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    save_video(frames, args.output / "frames.mp4")
    jsonl_path = args.output / "trajectory.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for entry in log:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- 验收指标 ----
    commands = torch.tensor([entry["command"] for entry in log])
    positions = torch.tensor([entry["eef_pos"] for entry in log])
    grippers = torch.tensor([entry["gripper_qpos"] for entry in log])
    # 旋转量用 EE 姿态的测地距离（rotation-about-EE-z 是腕部自转，
    # 世界系 yaw 分量会严重低估，不能用）
    rots = torch.tensor([entry["eef_rot"] for entry in log]).reshape(
        -1, 3, 3)
    chunk_len = args.execution_horizon
    rotation_per_phase: dict[str, float] = {}
    for start_idx in range(0, len(log), chunk_len):
        phase = log[start_idx]["phase"]
        r_start, r_end = rots[start_idx], rots[min(start_idx + chunk_len - 1,
                                                   len(log) - 1)]
        cos_theta = (
            (torch.diagonal(r_start.transpose(-1, -2) @ r_end)).sum() - 1.0
        ) / 2.0
        angle = torch.acos(cos_theta.clamp(-1.0, 1.0)).item()
        rotation_per_phase[f"{start_idx // chunk_len}:{phase}"] = round(
            angle, 4)
    summary = {
        "steps": len(log),
        "max_abs_command": commands.abs().max().item(),
        "position_drift": (positions[-1] - positions[0]).abs().max().item(),
        "rotation_rad_per_phase": rotation_per_phase,
        "gripper_qpos_min": grippers.min().item(),
        "gripper_qpos_max": grippers.max().item(),
        "gripper_qpos_range":
            (grippers.max(dim=0).values - grippers.min(dim=0).values)
            .max().item(),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    max_rotate = max(
        (angle for name, angle in rotation_per_phase.items()
         if "rotate" in name),
        default=0.0)
    checks = {
        "指令未越界 (|cmd|<=1)": summary["max_abs_command"] <= 1.0 + 1e-6,
        "旋转发生（rotate 相位姿态测地角 > 0.3 rad）": max_rotate > 0.3,
        "夹爪有效（qpos 极差 > 0.01）": summary["gripper_qpos_range"] > 0.01,
        "零平移指令下漂移受限 (< 0.05 m)": summary["position_drift"] < 0.05,
    }
    logger.info("验收: {}", json.dumps(summary, ensure_ascii=False, indent=2))
    for name, passed in checks.items():
        logger.info("  [{}] {}", "PASS" if passed else "FAIL", name)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
