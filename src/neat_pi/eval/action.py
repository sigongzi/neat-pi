"""pi05 模型动作块到 LIBERO 可执行 action queue 的适配。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from lerobot.processor import PolicyProcessorPipeline

from neat_pi.data.preprocessor import load_postprocessor_file
from neat_pi.eval.config import resolve_config_path
from neat_pi.typing import ActionBHD, ActionCommand7, typechecked


@dataclass
class LiberoActionAdapter:
    """unnormalize、裁剪并缓存一个完整 pi05 action chunk。"""

    postprocessor: PolicyProcessorPipeline
    action_horizon: int
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
        postprocessor_path = resolve_config_path(
            eval_config["normalization"]["postprocessor_path"], config_root)
        return cls(
            postprocessor=load_postprocessor_file(postprocessor_path),
            action_horizon=action_horizon,
            raw_action_dim=raw_action_dim,
            model_action_dim=model_action_dim,
            clip_bounds=clip_bounds,
        )

    @property
    def pending_count(self) -> int:
        """当前 action queue 中还未执行的步数。"""
        return len(self.pending_actions)

    @typechecked
    def submit(self, model_actions: ActionBHD) -> int:
        """处理一次 sample_actions 输出，返回入队步数。

        旧 checkpoint 消费完整 chunk；queue 未清空时禁止覆盖。模型输出
        的第 7 维之后是 padding，先截取再 unnormalize。
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
        raw_actions = model_actions.detach().to(device="cpu").float()
        actions = self.postprocessor(raw_actions[..., :self.raw_action_dim])
        actions = actions.to(device="cpu").float()
        if tuple(actions.shape) != (1, self.action_horizon, self.raw_action_dim):
            raise ValueError(
                "postprocessor 输出 shape 应为 "
                f"(1, {self.action_horizon}, {self.raw_action_dim})，"
                f"实际 {tuple(actions.shape)}")
        if not torch.isfinite(actions).all():
            raise ValueError("postprocessor 输出包含 NaN/Inf")
        actions = actions.clamp(self.clip_bounds[0], self.clip_bounds[1])
        self.pending_actions.extend(
            actions[0, action_index].clone()
            for action_index in range(self.action_horizon)
        )
        self.model_calls += 1
        return self.action_horizon

    @typechecked
    def pop_action(self) -> ActionCommand7:
        """弹出下一步可执行 LIBERO 动作，形状为 [7]。"""
        if self.pending_count == 0:
            raise IndexError("action queue 已空，需要重新调用 sample_actions")
        return self.pending_actions.popleft()
