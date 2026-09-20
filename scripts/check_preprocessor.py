"""preprocessor 管线自检：假形状数据或真实数据集逐步过每个 processor step。

用法：
    uv run scripts/check_preprocessor.py                          # YAML 的 preprocessor + 假数据
    uv run scripts/check_preprocessor.py --preprocessor <json>    # 任一 preprocessor JSON
    uv run scripts/check_preprocessor.py --dataset                # 从 YAML data.root 取真实 batch
    uv run scripts/check_preprocessor.py --config <yaml> [以上选项任意组合]

默认（不加 --dataset）拼一个假 batch 逐步过管线。形状/key 的来源优先级：
管线声明（normalizer 的 feature 定义、SE3StateActionStep 的 action_horizon）
> cfg.model（horizon 兜底、图像尺寸兜底）：
- 有 SE3StateActionStep 时输入是转换前的原始表示：state (B, H+1, 8)、
  action (B, H, 7)（见 se3.py 模块 docstring）；
- 没有时（如 finetuned checkpoint 的 policy_preprocessor.json）state/action
  直接按 normalizer 声明的形状喂入；
- 图像 key/路数取 normalizer 声明的 VISUAL feature，尺寸用
  cfg.model.image_size（真实输入由 dataset 侧 resize，声明不一致仅提示）。

随后与 --dataset 模式相同：用 `PolicyProcessorPipeline.step_through` 逐步跑
每个 step，每步打印全部字段的 shape/dtype/min/max/mean，最后做 token decode
往返对照。--dataset 时从 YAML data.root 的 LeRobot 数据集取真实 batch。
"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from loguru import logger

from neat_pi.config import Config, load_config
from neat_pi.data.inspect import print_batch


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="preprocessor 管线自检：逐步打印每个 processor step 前后的字段形态")
    parser.add_argument("--config", default="configs/pi05_libero.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--preprocessor", default=None,
                        help="直接指定 preprocessor JSON 路径（默认取 YAML data.preprocessor_path）")
    parser.add_argument("--dataset", action="store_true",
                        help="改用 YAML data.root 的真实数据集取 batch（默认用假形状数据）")
    return parser.parse_args()


def declared_features(pipe: Any) -> tuple[dict[str, list[int]], list[int], list[int]]:
    """从管线 normalizer 的声明取 (VISUAL {key: shape}, STATE shape, ACTION shape)。

    normalizer 的 feature 定义是管线期望字段的事实来源（来自 preprocessor JSON）。
    """
    from lerobot.configs import FeatureType
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    normalizer = next((s for s in pipe.steps if isinstance(s, NormalizerProcessorStep)), None)
    if normalizer is None:
        raise ValueError("preprocessor 管线中没有 normalizer，无法取 feature 声明")
    by_type: dict[FeatureType, dict[str, list[int]]] = {}
    for key, feature in normalizer.features.items():
        by_type.setdefault(feature.type, {})[key] = list(feature.shape)
    visual = by_type.get(FeatureType.VISUAL, {})
    state_features = by_type.get(FeatureType.STATE, {})
    action_features = by_type.get(FeatureType.ACTION, {})
    if not state_features or not action_features:
        raise ValueError("preprocessor 管线 normalizer 未声明 STATE/ACTION feature")
    return visual, next(iter(state_features.values())), next(iter(action_features.values()))


def build_fake_batch(cfg: Config, pipe: Any) -> dict[str, Any]:
    """按管线声明与 cfg.model 的形状约定拼一个假 batch（表示语义与 dataset collate 输出对齐）。

    - 图像均匀取 [-1,1]：dataset 侧 image_transforms 已归一化到此区间，
      管线 VISUAL 是 IDENTITY，不再动图像；
    - 有 SE3StateActionStep 时喂转换前表示：state (B,H+1,8) = eef pos(3) +
      axis-angle(3) + gripper qpos(2)，action (B,H,7) = OSC 增量指令(6) +
      二值 gripper(1)（仅 gripper 维被管线消费）；horizon 取 step 声明，
      兜底 cfg.model.action_horizon；
    - 没有时按 normalizer 声明的形状直接喂；
    - is_pad 全 False、task 为占位文本。
    """
    from neat_pi.data.se3_processor import SE3StateActionStep

    torch.manual_seed(cfg.training.seed)
    batch_size = cfg.data.per_device_batch_size
    visual, state_declared, action_declared = declared_features(pipe)

    se3 = next((s for s in pipe.steps if isinstance(s, SE3StateActionStep)), None)
    horizon = se3.action_horizon if se3 is not None else cfg.model.action_horizon
    if se3 is not None:
        state_shape = (horizon + 1, 8)
        action_shape = (horizon, 7)
    else:
        state_shape = tuple(state_declared)
        action_shape = (horizon, *action_declared)

    batch: dict[str, Any] = {}
    for key, shape in sorted(visual.items()):
        declared_size = shape[-2] if len(shape) >= 2 else cfg.model.image_size
        if declared_size != cfg.model.image_size:
            # VISUAL 是 IDENTITY 时尺寸不影响管线运行，但真实输入由 dataset 侧
            # image_transforms resize 到 cfg.model.image_size，声明不一致只提示不拦截
            logger.warning(
                "图像 {} 声明尺寸 {} 与 cfg.model.image_size={} 不一致，假数据按 {} 生成",
                key, declared_size, cfg.model.image_size, cfg.model.image_size)
        batch[key] = torch.rand(batch_size, 3, cfg.model.image_size,
                                cfg.model.image_size) * 2 - 1

    state = torch.empty(batch_size, *state_shape)
    state.uniform_(-0.5, 0.5)
    if se3 is not None:
        state[..., 3:6].uniform_(-0.2, 0.2)  # axis-angle 小旋转
        state[..., 6:].uniform_(0.0, 1.0)    # gripper qpos
    batch["observation.state"] = state

    action = torch.empty(batch_size, *action_shape)
    action.uniform_(-0.05, 0.05)
    if action.shape[-1] == 7:  # 原始指令表示的二值 gripper 维（se3.py）
        action[..., 6] = torch.where(torch.rand(batch_size, horizon) < 0.5, -1.0, 1.0)
    batch["action"] = action

    batch["observation.state_is_pad"] = (
        torch.zeros(batch_size, state_shape[0], dtype=torch.bool)
        if len(state_shape) > 1
        else torch.zeros(batch_size, dtype=torch.bool)
    )
    batch["action_is_pad"] = torch.zeros(batch_size, horizon, dtype=torch.bool)
    batch["task"] = [f"fake task {index}" for index in range(batch_size)]
    return batch


def main() -> None:
    """按配置装好 preprocessor，取假 batch 或真实 batch 逐步过管线并打印。"""
    args = parse_args()
    cfg = load_config(args.config)

    from neat_pi.data.preprocessor import (load_preprocessor,
                                           load_preprocessor_file)
    from neat_pi.data.tokenizer import GemmaTokenizerStep
    from neat_pi.data.transforms import images_to_float

    if args.preprocessor:
        try:
            pipe = load_preprocessor_file(args.preprocessor)
        except Exception as error:
            raise RuntimeError(
                f"加载 {args.preprocessor} 失败：{error}\n"
                "提示：含官方 tokenizer_processor 的 JSON（如 finetuned checkpoint 的）"
                "需要 HF hub 认证拉取 google/paligemma-3b-pt-224；项目自己的 "
                "preprocessor JSON 用本地 GemmaTokenizerStep，无需网络。") from error
    else:
        pipe = load_preprocessor(cfg)
    logger.info("管线 {} 共 {} 步: {}", pipe.name, len(pipe.steps),
                " -> ".join(type(step).__name__ for step in pipe.steps))

    if args.dataset:
        from torch.utils.data import DataLoader

        from neat_pi.data.lerobot_dataset import build_dataset

        ds = build_dataset(cfg.data, cfg.model)
        logger.info("数据集: {} 帧", len(ds))
        dl = DataLoader(
            ds,
            batch_size=cfg.data.per_device_batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )
        batch = images_to_float(next(iter(dl)))
    else:
        visual, _, _ = declared_features(pipe)
        image_keys = sorted(visual)
        # 语义与 build_dataset 一致：数据集相机数不得超过 num_cameras，
        # 缺少的槽位由 prepare_batch 注入 empty camera（-1 填充、mask=False）
        if not args.preprocessor and len(image_keys) > cfg.model.num_cameras:
            raise ValueError(
                f"管线 normalizer 声明 {len(image_keys)} 路图像（{image_keys}）"
                f"超过配置 num_cameras={cfg.model.num_cameras}")
        batch = build_fake_batch(cfg, pipe)

    logger.info("== 原始 batch（{}）==", "数据集" if args.dataset else "假形状数据")
    print_batch(batch)

    logger.info("== 逐步过 preprocessor ==")
    final_batch: dict[str, Any] = {}
    for step_idx, transition in enumerate(pipe.step_through(batch)):
        if step_idx == 0:
            logger.info("-- 输入 --")
        else:
            label = type(pipe.steps[step_idx - 1]).__name__
            logger.info("-- 第 {} 步: {} --", step_idx, label)
        view = pipe.to_output(transition)
        print_batch(view)
        final_batch = view

    # 词表的唯一来源是管线中的 GemmaTokenizerStep；lerobot 官方 TokenizerProcessorStep
    # （如 finetuned checkpoint 的 JSON 用的）强制依赖 transformers / HF hub，不在此列
    tokenizer = next(
        (s.tokenizer for s in pipe.steps if isinstance(s, GemmaTokenizerStep)), None
    )
    if tokenizer is None:
        logger.warning("管线中没有 GemmaTokenizerStep，跳过 token decode 对照")
    else:
        logger.info("== 最终 token 对照 ==")
        logger.info("完整 prompt: {!r}", final_batch["task"][0])
        logger.info("decode 回:   {!r}",
                    tokenizer.decode(final_batch["observation.language.tokens"][0])[:200])


if __name__ == "__main__":
    main()
