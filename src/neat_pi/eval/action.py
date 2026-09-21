"""pi05 模型动作块到 LIBERO 可执行 action queue 的适配。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from lerobot.processor import PolicyProcessorPipeline

from neat_pi.data.preprocessor import load_postprocessor_file
from neat_pi.data.se3 import aa_to_rot, rot6d_to_rot, rot_to_aa
from neat_pi.eval.config import resolve_config_path
from neat_pi.typing import (ActionBHD, ActionCommand7, StateB8, typechecked)


@dataclass
class _Se3Target:
    """一个待执行步的绝对目标位姿（世界系）与夹爪指令。"""

    pos: torch.Tensor    # (3,)
    rot: torch.Tensor    # (3, 3)
    gripper: torch.Tensor  # 标量


@dataclass
class LiberoActionAdapter:
    """unnormalize、裁剪并缓存每次推理要执行的 pi05 action 前缀。"""

    postprocessor: PolicyProcessorPipeline
    action_horizon: int
    execution_horizon: int
    raw_action_dim: int
    model_action_dim: int
    clip_bounds: tuple[float, float]
    pending_actions: deque[ActionCommand7] = field(
        default_factory=deque, init=False, repr=False)
    model_calls: int = field(default=0, init=False)

    @classmethod
    def from_config(cls, eval_config: Mapping[str, Any],
                    config_root: str | None = None,
                    ) -> "LiberoActionAdapter":
        """从 eval YAML 构造 adapter，并加载 checkpoint 的 MEAN_STD 统计。"""
        model = eval_config["model"]
        actions = eval_config["actions"]
        action_horizon = int(model["action_horizon"])
        execution_horizon = int(actions["execution_horizon"])
        raw_action_dim = int(actions["raw_dim"])
        model_action_dim = int(model["max_action_dim"])
        bounds_value = eval_config["env"]["clip_bounds"]
        if len(bounds_value) != 2 or float(bounds_value[0]) >= float(bounds_value[1]):
            raise ValueError("clip_bounds 必须是 [low, high] 且 low < high")
        clip_bounds = (float(bounds_value[0]), float(bounds_value[1]))
        if actions["normalization"] != "MEAN_STD":
            raise ValueError(
                f"旧 checkpoint 只支持 MEAN_STD 动作归一化，"
                f"实际 {actions['normalization']}")
        if not 0 < execution_horizon <= action_horizon:
            raise ValueError(
                "execution_horizon 必须在 (0, action_horizon] 内，"
                f"实际 {execution_horizon}")
        postprocessor_path = resolve_config_path(
            eval_config["normalization"]["postprocessor_path"], config_root)
        return cls(
            postprocessor=load_postprocessor_file(postprocessor_path),
            action_horizon=action_horizon,
            execution_horizon=execution_horizon,
            raw_action_dim=raw_action_dim,
            model_action_dim=model_action_dim,
            clip_bounds=clip_bounds,
        )

    @property
    def pending_count(self) -> int:
        """当前 action queue 中还未执行的步数。"""
        return len(self.pending_actions)

    def reset(self) -> None:
        """清空跨 episode 不应延续的 action queue 与模型调用计数。"""
        self.pending_actions.clear()
        self.model_calls = 0

    @typechecked
    def submit(self, model_actions: ActionBHD) -> int:
        """处理一次 sample_actions 输出，返回入队步数。

        官方 LIBERO 评测每次采样只执行前 execution_horizon 步；queue 未
        清空时禁止覆盖。模型输出的第 7 维之后是 padding，先截取再
        unnormalize。
        """
        if tuple(model_actions.shape) != (
                1, self.action_horizon, self.model_action_dim):
            raise ValueError(
                f"模型动作 shape 应为 "
                f"(1, {self.action_horizon}, {self.model_action_dim})，"
                f"实际 {tuple(model_actions.shape)}")
        if self.pending_count != 0:
            raise RuntimeError(
                f"action queue 未消费完（剩余 {self.pending_count} 步），"
                "不能提交新 chunk")
        raw_actions = (
            model_actions.detach().to(device="cpu").float()
           [:, :self.execution_horizon, :self.raw_action_dim]
        )
        actions = self.postprocessor(raw_actions)
        actions = actions.to(device="cpu").float()
        if tuple(actions.shape) != (
                1, self.execution_horizon, self.raw_action_dim):
            raise ValueError(
                "postprocessor 输出 shape 应为 "
                f"(1, {self.execution_horizon}, {self.raw_action_dim})，"
                f"实际 {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("postprocessor 输出包含 NaN/Inf")
        actions = actions.clamp(self.clip_bounds[0], self.clip_bounds[1])
        self.pending_actions.extend(
            actions[0, action_index].clone()
            for action_index in range(self.execution_horizon)
        )
        self.model_calls += 1
        return self.execution_horizon

    @typechecked
    def pop_action(self) -> ActionCommand7:
        """弹出下一步可执行 LIBERO 动作，形状为 [7]。"""
        if self.pending_count == 0:
            raise IndexError("action queue 已空，需要重新调用 sample_actions")
        return self.pending_actions.popleft()


@dataclass
class LiberoSe3ActionAdapter:
    """chunk-wise SE(3) delta 合同的执行适配（绝对位姿 + 逐步重参考）。

    模型输出 (1, H, 10) = [xyz, rot6D, gripper] delta，全部相对 chunk
    首帧位姿（训练侧 ΔT_k = T_0^{-1} @ T_k，见 se3.py）。本 adapter 在
    submit 时用首帧观测把 delta 还原成绝对目标位姿；pop_action 时用
    **当步新观测**重新相减求指令——PD 跟踪滞后造成的实际位姿偏离每步
    自校正，不会逐 chunk 累积。

    LIBERO OSC_POSE（control_delta）把 [-1,1] 指令按每步增益缩放为物理
    delta（pos ±pos_gain m、rot ±rot_gain rad，robosuite osc_pose.json
    默认 0.05/0.5），goal 在世界系更新（goal_ori = R(aa) @ R_now）。因此
    「新到坐标相减」得到的物理量须除以增益才是可下发指令；旋转在世界系
    共轭：aa(R_target @ R_now^T)。
    """

    postprocessor: PolicyProcessorPipeline
    action_horizon: int
    execution_horizon: int
    model_action_dim: int
    clip_bounds: tuple[float, float]
    pos_gain: float = 0.05   # OSC_POSE 每步平移增益（output_max[0:3]）
    rot_gain: float = 0.5    # OSC_POSE 每步旋转增益（output_max[3:6]）
    raw_action_dim: int = field(default=10, init=False)
    pending_targets: deque[_Se3Target] = field(
        default_factory=deque, init=False, repr=False)
    model_calls: int = field(default=0, init=False)

    @classmethod
    def from_config(cls, eval_config: Mapping[str, Any],
                    config_root: str | None = None,
                    ) -> "LiberoSe3ActionAdapter":
        """从 eval YAML 构造 adapter，加载 SE(3) 统计量做反归一化。"""
        model = eval_config["model"]
        actions = eval_config["actions"]
        action_horizon = int(model["action_horizon"])
        execution_horizon = int(actions["execution_horizon"])
        bounds_value = eval_config["env"]["clip_bounds"]
        if len(bounds_value) != 2 or float(bounds_value[0]) >= float(bounds_value[1]):
            raise ValueError("clip_bounds 必须是 [low, high] 且 low < high")
        clip_bounds = (float(bounds_value[0]), float(bounds_value[1]))
        if actions.get("normalization") != "QUANTILES":
            raise ValueError(
                f"SE(3) 合同只支持 QUANTILES 动作归一化，"
                f"实际 {actions.get('normalization')}")
        if not 0 < execution_horizon <= action_horizon:
            raise ValueError(
                "execution_horizon 必须在 (0, action_horizon] 内，"
                f"实际 {execution_horizon}")
        postprocessor_path = resolve_config_path(
            eval_config["normalization"]["postprocessor_path"], config_root)
        return cls(
            postprocessor=load_postprocessor_file(postprocessor_path),
            action_horizon=action_horizon,
            execution_horizon=execution_horizon,
            model_action_dim=int(model["max_action_dim"]),
            clip_bounds=clip_bounds,
        )

    @property
    def pending_count(self) -> int:
        """当前绝对目标队列中还未执行的步数。"""
        return len(self.pending_targets)

    def reset(self) -> None:
        """清空跨 episode 不应延续的目标队列与模型调用计数。"""
        self.pending_targets.clear()
        self.model_calls = 0

    @typechecked
    def submit(self, model_actions: ActionBHD,
               state: StateB8) -> int:
        """处理一次 sample_actions 输出，返回入队步数。

        `state` 为 chunk 首帧观测 (1, 8) = [pos(3), axis-angle(3),
        gripper qpos(2)]；delta 全部在该位姿坐标系下还原成绝对目标。
        """
        if tuple(model_actions.shape) != (
                1, self.action_horizon, self.model_action_dim):
            raise ValueError(
                f"模型动作 shape 应为 "
                f"(1, {self.action_horizon}, {self.model_action_dim})，"
                f"实际 {tuple(model_actions.shape)}")
        if tuple(state.shape) != (1, 8):
            raise ValueError(f"state shape 应为 (1, 8)，实际 {tuple(state.shape)}")
        if self.pending_count != 0:
            raise RuntimeError(
                f"action queue 未消费完（剩余 {self.pending_count} 步），"
                "不能提交新 chunk")
        raw_actions = (
            model_actions.detach().to(device="cpu").float()
           [:, :self.execution_horizon, :self.raw_action_dim]
        )
        deltas = self.postprocessor(raw_actions).to(device="cpu").float()
        if tuple(deltas.shape) != (
                1, self.execution_horizon, self.raw_action_dim):
            raise ValueError(
                "postprocessor 输出 shape 应为 "
                f"(1, {self.execution_horizon}, {self.raw_action_dim})，"
                f"实际 {tuple(deltas.shape)}")
        if not torch.isfinite(deltas).all():
            raise ValueError("postprocessor 输出包含 NaN/Inf")

        pos0 = state[0, 0:3]
        rot0 = aa_to_rot(state[0, 3:6])
        for k in range(self.execution_horizon):
            delta = deltas[0, k]
            # ΔT_k = T_0^{-1} @ T_k 的逆：p_k = p0 + R0·dt；R_k = R0·dR
            target_pos = pos0 + rot0 @ delta[0:3]
            target_rot = rot0 @ rot6d_to_rot(delta[3:9])
            self.pending_targets.append(_Se3Target(
                pos=target_pos.clone(),
                rot=target_rot.clone(),
                gripper=delta[9].detach().clone(),
            ))
        self.model_calls += 1
        return self.execution_horizon

    @typechecked
    def pop_action(self, state: StateB8) -> ActionCommand7:
        """按当步新观测重参考，弹出下一步可执行 LIBERO 指令 [7]。

        指令 = 物理 delta / 增益后按 clip_bounds 截断：平移 (p_t -
        p_now) 世界系直接相减；旋转为世界系共轭 aa(R_t @ R_now^T)。
        gripper 维原样透传。
        """
        if self.pending_count == 0:
            raise IndexError("action queue 已空，需要重新调用 sample_actions")
        if tuple(state.shape) != (1, 8):
            raise ValueError(f"state shape 应为 (1, 8)，实际 {tuple(state.shape)}")
        target = self.pending_targets.popleft()
        pos_now = state[0, 0:3]
        rot_now = aa_to_rot(state[0, 3:6])

        dpos_world = target.pos - pos_now
        drot_world = target.rot @ rot_now.transpose(-1, -2)
        command = torch.cat([
            (dpos_world / self.pos_gain).clamp(*self.clip_bounds),
            (rot_to_aa(drot_world) / self.rot_gain).clamp(*self.clip_bounds),
            target.gripper.clamp(*self.clip_bounds).reshape(1),
        ])
        return command
