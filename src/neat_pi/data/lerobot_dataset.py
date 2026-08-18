"""LeRobot v3 数据集构造。

lerobot 的 LeRobotDataset 已覆盖样本读取 / 视频解码 / 任务文本；官方训练脚本即
"LeRobotDataset + DataLoader 默认 collate + preprocessor(batch)"（lerobot_train.py），
因此本项目不写 Dataset 子类、也不写自定义 collator，只在此集中构造参数：

- delta_timestamps：action 取未来 action_horizon 步（按数据集 fps 换算成秒偏移）；
- image_transforms：resize + [-1,1] 归一化（预处理管线的 VISUAL 是 IDENTITY，
  图像归一化必须在数据集侧完成）。
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from neat_pi.config import DataConfig, ModelConfig
from neat_pi.data.transforms import build_image_transform


def build_dataset(data_cfg: DataConfig, model_cfg: ModelConfig) -> LeRobotDataset:
    """按配置构造 LIBERO 的 LeRobotDataset（动作窗口与图像变换已挂好）。

    返回的样本 dict 直接可被 DataLoader 默认 collate 堆叠后喂给 preprocessor。
    """
    root = Path(data_cfg.root)
    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        fps = float(json.load(f)["fps"])

    delta_timestamps = {
        # 当前帧 + 未来 action_horizon 帧：SE3StateActionStep 用未来位姿算 SE(3) delta
        "observation.state": [i / fps for i in range(model_cfg.action_horizon + 1)],
        # 原始 action chunk 仅保留 gripper 指令维（见 se3.py 模块 docstring）
        "action": [i / fps for i in range(model_cfg.action_horizon)],
    }
    ds = LeRobotDataset(
        data_cfg.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        image_transforms=build_image_transform(model_cfg.image_size),
        video_backend=data_cfg.video_backend,
    )

    camera_keys = list(ds.meta.camera_keys)
    if len(camera_keys) != model_cfg.num_cameras:
        raise ValueError(
            f"数据集相机数 {len(camera_keys)}（{camera_keys}）"
            f"与配置 num_cameras={model_cfg.num_cameras} 不一致"
        )
    return ds
