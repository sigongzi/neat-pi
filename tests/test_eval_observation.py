"""LIBERO eval observation adapter 测试。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import yaml

from neat_pi.eval.observation import (LiberoObservationAdapter,
                                      quaternion_to_axis_angle)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _adapter() -> LiberoObservationAdapter:
    """从 eval YAML 构造被测 adapter。"""
    with open(ROOT / "configs" / "eval_pi05_libero_finetuned.yaml",
              encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    return LiberoObservationAdapter.from_config(
        config["cameras"], image_size=config["model"]["image_size"])


def _observation() -> dict[str, Any]:
    """构造固定 LIBERO 原始观测。"""
    image = np.full((256, 256, 3), 16, dtype=np.uint8)
    image[0, 0] = 255
    return {
        "pixels": {
            "agentview_image": image,
            "robot0_eye_in_hand_image": np.full(
                (256, 256, 3), 200, dtype=np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, -0.2, 0.3], dtype=np.float64),
                "quat": np.array([
                    np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)
                ], dtype=np.float64),
            },
            "gripper": {
                "qpos": np.array([0.02, 0.98], dtype=np.float64),
            },
        },
    }


def test_from_config_requires_checkpoint_camera_contract() -> None:
    """YAML 相机合同必须映射两路真实相机并保留一路 mask=False 相机。"""
    adapter = _adapter()
    assert [camera.source for camera in adapter.real_cameras] == [
        "agentview_image", "robot0_eye_in_hand_image"]
    assert [camera.key for camera in adapter.real_cameras] == [
        "observation.images.image", "observation.images.image2"]
    assert all(camera.rotate_180 for camera in adapter.real_cameras)
    assert adapter.empty_camera_key == "observation.images.empty_camera_0"
    assert adapter.empty_fill_value == -1.0


def test_convert_images_masks_and_state() -> None:
    """转换结果满足 [3,B,3,224,224]、mask 和 8 维 state 合同。"""
    observation = _adapter().convert(_observation())
    images, masks = observation.model_inputs()

    assert len(images) == 3
    assert len(masks) == 3
    assert all(image.shape == (1, 3, 224, 224) for image in images)
    assert all(image.dtype is torch.float32 for image in images)
    assert masks[0].tolist() == [True]
    assert masks[1].tolist() == [True]
    assert masks[2].tolist() == [False]
    assert torch.equal(images[2], torch.full((1, 3, 224, 224), -1.0))
    assert torch.isfinite(torch.stack(images)).all()

    # 原始渲染旋转 180° 后，左上亮点应落到右下角。
    assert images[0][0, 0, 0, 0].item() < 0.5
    assert images[0][0, 0, -1, -1].item() > 0.25
    # 均匀值经 resize/归一化后保持固定：16/255 -> [-1,1]。
    uniform = 2.0 * 16.0 / 255.0 - 1.0
    torch.testing.assert_close(
        images[0][0, :, 10:-10, 10:-10],
        torch.full((3, 204, 204), uniform),
    )

    expected_state = torch.tensor(
        [[0.1, -0.2, 0.3, np.pi / 2, 0.0, 0.0, 0.02, 0.98]],
        dtype=torch.float32,
    )
    assert observation.state.shape == (1, 8)
    torch.testing.assert_close(observation.state, expected_state)


def test_policy_observation_uses_checkpoint_keys() -> None:
    """policy 视图只暴露两路真实图像、empty 图像和 state。"""
    observation = _adapter().convert(_observation())
    policy_observation = observation.policy_observation()
    assert set(policy_observation) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.images.empty_camera_0",
        "observation.state",
    }
    assert policy_observation["observation.images.image"] is observation.images[0]


def test_quaternion_to_axis_angle_boundaries() -> None:
    """单位四元数和反向单位四元数都转换成零 axis-angle。"""
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    inverse = torch.tensor([[0.0, 0.0, 0.0, -1.0]])
    torch.testing.assert_close(
        quaternion_to_axis_angle(identity), torch.zeros(1, 3))
    torch.testing.assert_close(
        quaternion_to_axis_angle(inverse), torch.zeros(1, 3))


def test_convert_rejects_non_contract_image() -> None:
    """非 uint8 HWC 图像直接失败，避免静默送错视觉合同。"""
    observation = _observation()
    observation["pixels"]["agentview_image"] = np.zeros(
        (256, 256, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dtype"):
        _adapter().convert(observation)
