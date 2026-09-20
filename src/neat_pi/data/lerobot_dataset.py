"""LeRobot v3 数据集构造。

lerobot 的 LeRobotDataset 已覆盖样本读取 / 视频解码 / 任务文本；官方训练脚本即
"LeRobotDataset + DataLoader 默认 collate + preprocessor(batch)"（lerobot_train.py），
因此本项目不写 Dataset 子类、也不写自定义 collator，只在此集中构造参数：

- delta_timestamps：state 窗口按数据表示取（SE(3) 用 horizon+1 帧、原始表示
  仅当前帧，见 uses_se3），action 取未来 action_horizon 步（按数据集 fps
  换算成秒偏移）；
- image_transforms：resize + [-1,1] 归一化（预处理管线的 VISUAL 是 IDENTITY，
  图像归一化必须在数据集侧完成）。
"""

from __future__ import annotations

import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from loguru import logger

from neat_pi.config import DataConfig, ModelConfig
from neat_pi.data.transforms import build_image_transform

_SE3_STEP_CLASS = "neat_pi.data.se3_processor.SE3StateActionStep"


def uses_se3(preprocessor_path: str | Path) -> bool:
    """判定训练数据是否走 SE(3) 表示：preprocessor JSON 含 SE3StateActionStep。

    preprocessor JSON 是数据表示的唯一声明处，步骤的 "class" 字段即实例化
    时的完整导入路径，据此判定不引入第二份配置开关。SE(3) 管线需要未来位姿
    算 delta，dataset 侧为 state 取 horizon+1 帧；原始表示（如 MEAN_STD 合同）
    只消费当前帧，state 进 prompt 离散化，多帧会污染 prompt。
    """
    with open(preprocessor_path, encoding="utf-8") as f:
        steps = json.load(f).get("steps", [])
    return any(step.get("class") == _SE3_STEP_CLASS for step in steps)


def build_dataset(data_cfg: DataConfig, model_cfg: ModelConfig) -> LeRobotDataset:
    """按配置构造 LIBERO 的 LeRobotDataset（动作窗口与图像变换已挂好）。

    返回的样本 dict 直接可被 DataLoader 默认 collate 堆叠后喂给 preprocessor。
    """
    root = Path(data_cfg.root)
    with open(root / "meta" / "info.json", encoding="utf-8") as f:
        fps = float(json.load(f)["fps"])

    state_window = (
        model_cfg.action_horizon + 1
        if uses_se3(data_cfg.preprocessor_path)
        else 1
    )
    delta_timestamps = {
        # SE(3)：当前帧 + 未来 action_horizon 帧，SE3StateActionStep 用未来位姿
        # 算 delta；原始表示：仅当前帧（见 uses_se3）
        "observation.state": [i / fps for i in range(state_window)],
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
    if len(camera_keys) > model_cfg.num_cameras:
        raise ValueError(
            f"数据集相机数 {len(camera_keys)}（{camera_keys}）"
            f"超过配置 num_cameras={model_cfg.num_cameras}"
        )
    if len(camera_keys) < model_cfg.num_cameras:
        # 不足的槽位是 empty camera：prepare_batch 在 preprocessor 后注入
        # （-1 填充、mask=False），对齐评测合同（eval/observation.py）
        logger.info(
            "数据集相机数 {}（{}）少于配置 num_cameras={}："
            "{} 个槽位将注入 empty camera",
            len(camera_keys), camera_keys, model_cfg.num_cameras,
            model_cfg.num_cameras - len(camera_keys))
    return ds
