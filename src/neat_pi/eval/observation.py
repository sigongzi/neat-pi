"""把 LeRobot LIBERO 环境观测转换成 pi05 推理输入。

旧 checkpoint 的 LIBERO 合同要求两路真实相机 + 一路 empty camera。相机
resize 到 224 并归一化到 [-1, 1]；empty camera 保留 token 槽位但整段
mask=False。state 由 LeRobot 环境的嵌套字段转回训练数据使用的
8 维 [pos(3), axis-angle(3), gripper qpos(2)] 表示。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional
from torchtyping import TensorType

from neat_pi.typing import (AxisAngleB3, ImageBCHW, MaskB, QuatB4, StateB8,
                            typechecked)


@dataclass(frozen=True)
class CameraMapping:
    """一个真实相机的 LIBERO source 与 policy 输入 key 的映射。"""

    source: str
    key: str
    rotate_180: bool = False


@dataclass(frozen=True)
class Pi05EvalObservation:
    """adapter 产出的 pi05 输入及其 policy-dict 视图。"""

    images: tuple[ImageBCHW, ...]
    image_masks: tuple[MaskB, ...]
    state: StateB8
    real_cameras: tuple[CameraMapping, ...]
    empty_camera_key: str

    @typechecked
    def model_inputs(self) -> tuple[list[ImageBCHW], list[MaskB]]:
        """返回 Pi05.sample_actions 要求的 (images, image_masks) 顺序。"""
        return list(self.images), list(self.image_masks)

    def policy_observation(self) -> dict[str, Any]:
        """返回 preprocessor 期望的 observation 字段（不含 task 文本）。"""
        observation: dict[str, Any] = {
            mapping.key: image
            for mapping, image in zip(
                self.real_cameras, self.images[:-1], strict=True
            )
        }
        observation[self.empty_camera_key] = self.images[-1]
        observation["observation.state"] = self.state
        return observation


@dataclass(frozen=True)
class LiberoObservationAdapter:
    """按旧 pi05 LIBERO checkpoint 合同转换单条环境观测。"""

    real_cameras: tuple[CameraMapping, ...]
    empty_camera_key: str
    empty_fill_value: float
    image_size: int

    @classmethod
    def from_config(cls, camera_config: Mapping[str, Any],
                    image_size: int) -> "LiberoObservationAdapter":
        """从 eval YAML 的 cameras 段构造 adapter，拒绝不完整相机合同。"""
        real_cameras = tuple(
            CameraMapping(
                source=camera["source"],
                key=camera["key"],
                rotate_180=bool(camera["rotate_180"]),
            )
            for camera in camera_config["real"]
        )
        if len(real_cameras) != 2 or not all(
            camera["mask"] is True for camera in camera_config["real"]
        ):
            raise ValueError("eval 相机合同必须是两路 mask=True 的真实相机")
        empty = camera_config["empty"]
        if empty["mask"] is not False:
            raise ValueError("empty camera 的 mask 必须是 False")
        return cls(
            real_cameras=real_cameras,
            empty_camera_key=empty["key"],
            empty_fill_value=float(empty["fill_value"]),
            image_size=image_size,
        )

    def convert(self, observation: Mapping[str, Any]) -> Pi05EvalObservation:
        """转换一条 LIBERO obs 为 batch=1 的 pi05 输入。"""
        raw_pixels = observation["pixels"]
        images = tuple(
            self._convert_image(raw_pixels[camera.source], camera.rotate_180)
            for camera in self.real_cameras
        )
        empty_shape = (1, 3, self.image_size, self.image_size)
        images = images + (
            torch.full(
                empty_shape,
                self.empty_fill_value,
                dtype=torch.float32,
            ),
        )
        masks = (
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        )
        converted = Pi05EvalObservation(
            images=images,
            image_masks=masks,
            state=self._convert_state(observation["robot_state"]),
            real_cameras=self.real_cameras,
            empty_camera_key=self.empty_camera_key,
        )
        return converted

    def _convert_image(self, raw_image: Any, rotate_180: bool) -> ImageBCHW:
        """uint8 HWC 图像转 [-1,1] float32 BCHW，并按训练合同旋转。"""
        image = np.asarray(raw_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"LIBERO 图像应为 HWC=3，实际 shape={image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(
                f"LIBERO 图像 dtype 应为 uint8，实际 {image.dtype}")
        if rotate_180:
            image = image[::-1, ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = functional.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return tensor * 2.0 - 1.0

    @typechecked
    def _convert_state(self, robot_state: Mapping[str, Any]) -> StateB8:
        """嵌套 LeRobot state 转为旧数据的 8 维 LIBERO state。"""
        eef = robot_state["eef"]
        position = torch.as_tensor(
            np.asarray(eef["pos"]), dtype=torch.float32).reshape(1, 3)
        quaternion = torch.as_tensor(
            np.asarray(eef["quat"]), dtype=torch.float32).reshape(1, 4)
        axis_angle = quaternion_to_axis_angle(quaternion)
        gripper = torch.as_tensor(
            np.asarray(robot_state["gripper"]["qpos"]),
            dtype=torch.float32,
        ).reshape(1, 2)
        return torch.cat([position, axis_angle, gripper], dim=-1)


@typechecked
def quaternion_to_axis_angle(quaternion: QuatB4) -> AxisAngleB3:
    """把 [x, y, z, w] 四元数转为 LIBERO state 使用的 axis-angle。"""
    scalar = quaternion[:, 3].clamp(-1.0, 1.0)
    denominator = torch.sqrt(1.0 - scalar * scalar)
    angle = 2.0 * torch.acos(scalar)
    axis_angle = quaternion[:, 0:3] * (angle / denominator).unsqueeze(-1)
    zero_rotation = torch.zeros(1, 3, dtype=quaternion.dtype)
    return torch.where(
        torch.isclose(denominator, torch.zeros_like(denominator)).unsqueeze(-1),
        zero_rotation,
        axis_angle,
    )
