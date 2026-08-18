"""SE(3) delta 动作 + rot6D state 表示转换的 lerobot 管线 step。

插在本项目 `configs/pi05_libero_preprocessor.json` 的 to_batch 之后、normalizer
之前（normalizer 必须作用于新表示，统计量也是按新表示算的，见
scripts/compute_se3_stats.py）：

- 输入：`observation.state` (B, H+1, 8)（当前帧 + 未来 H 帧窗口，由
  dataset 的 delta_timestamps 提供）、`action` (B, H, 7)（原始指令，仅取
  gripper 维）；
- 输出：`observation.state` (B, 11) = [pos, rot6D, gripper qpos]（当前帧），
  `action` (B, H, 10) = [xyz, rot6D, gripper]（SE(3) delta，见 se3.py）；
- `action_is_pad` 与窗口内未来 state 的 is_pad 取 OR（episode 末尾的
  padding 帧传播到动作 mask）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey

from neat_pi.data.se3 import (compute_delta_actions, convert_state,
                              delta_action_to_command)


@dataclass
class SE3StateActionStep(ProcessorStep):
    """把 LIBERO 原始 state/action 转成 rot6D state 与 SE(3) delta 动作。"""

    action_horizon: int = 32
    state_key: str = "observation.state"
    action_key: str = "action"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """执行表示转换（形状约定见模块 docstring），缺字段直接报错。"""
        transition = transition.copy()
        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        action = transition.get(TransitionKey.ACTION)

        states = obs.get(self.state_key)
        if states is None:
            raise ValueError(f"observation 中找不到 '{self.state_key}'")
        if action is None:
            raise ValueError("transition 中找不到 action")
        if states.shape[1] != self.action_horizon + 1 or states.shape[-1] != 8:
            raise ValueError(
                f"{self.state_key} 形状应为 (B, {self.action_horizon + 1}, 8)，"
                f"实际 {tuple(states.shape)}——检查 dataset 的 delta_timestamps"
            )

        obs[self.state_key] = convert_state(states[:, 0])
        transition[TransitionKey.ACTION] = compute_delta_actions(
            states, action[..., 6]
        )

        state_is_pad = obs.get(f"{self.state_key}_is_pad")
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        action_is_pad = comp.get(f"{self.action_key}_is_pad")
        if state_is_pad is not None and action_is_pad is not None:
            comp[f"{self.action_key}_is_pad"] = (
                action_is_pad | state_is_pad[:, 1:]
            )
            transition[TransitionKey.COMPLEMENTARY_DATA] = comp

        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def transform_features(self, features: Any) -> Any:
        """把 state/action 的 feature shape 更新为新表示（8->11、7->10）。"""
        for key, shape in ((self.state_key, (11,)), (self.action_key, (10,))):
            for group in features.values():
                if key in group:
                    group[key].shape = shape
        return features


@dataclass
class SE3DeltaToCommandStep(ProcessorStep):
    """postprocessor 侧：10 维 SE(3) delta 动作转 7 维可执行指令。

    输入为模型输出（已被 unnormalizer 反归一化的 delta 表示，形状
    (B, H, 10)）；输出 (B, H, 7) = [dpos(3), axis-angle(3), gripper(1)]，
    平移与旋转均在**当前帧末端坐标系**下。若控制器需要 base 系指令，
    在环境接口侧用 `se3.ee_command_to_base(current_state, command)` 转换
    （需要当前 EE 位姿，postprocessor 拿不到，故不放在本 step 内）。
    """

    action_key: str = "action"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """执行 delta→指令转换，缺 action 或维度不符直接报错。"""
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("transition 中找不到 action")
        if action.shape[-1] != 10:
            raise ValueError(
                f"action 末维应为 10（SE(3) delta 表示），实际 {tuple(action.shape)}"
            )
        transition[TransitionKey.ACTION] = delta_action_to_command(action)
        return transition

    def transform_features(self, features: Any) -> Any:
        """把 action 的 feature shape 更新为指令表示（10->7）。"""
        for group in features.values():
            if self.action_key in group:
                group[self.action_key].shape = (7,)
        return features
