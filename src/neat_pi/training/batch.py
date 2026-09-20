"""训练 batch 组装：preprocessor 输出 -> 模型输入（含 empty camera 注入）。"""

from __future__ import annotations

from typing import Any
from typing import NamedTuple

import torch
from torch.nn import functional as F

from neat_pi.config import Config


class TrainBatch(NamedTuple):
    """一个训练 batch 的模型侧形态：action 已补零到模型维度。"""

    images: list[torch.Tensor]  # 各相机图像 (B,3,H,W)，[-1,1]
    image_masks: list[torch.Tensor]  # 各相机可用性 (B,)，True=有效
    token_ids: torch.Tensor     # (B, max_token_len)
    lang_mask: torch.Tensor     # (B, max_token_len)，True=有效语言 token
    actions: torch.Tensor       # (B, action_horizon, action_dim) 补零后
    is_pad: torch.Tensor        # (B, action_horizon)，True=padding 帧
    real_action_dim: int        # padding 前的真实动作维


def prepare_batch(cfg: Config, batch: dict[str, Any],
                  device: torch.device) -> TrainBatch:
    """把 preprocessor 输出的 batch 整理成模型输入。

    action 维度不足 `model.action_dim` 时在右侧补零；语言 attention mask
    原样传给 Pi05 构造 prefix mask。

    相机槽位对齐评测合同（eval/observation.py）：真实相机按 key 排序在前，
    batch 里缺失的 `model.num_cameras` 槽位视为 empty camera 依序补在最后，
    填充 -1 且 mask=False——masked token 在 joint attention 中不可见，
    填充值不进损失。normalizer 对 VISUAL 是 IDENTITY 且只处理存在 key，
    评测侧同样在管线外合成 empty camera，注入时机一致。
    """
    image_keys = sorted(
        key for key in batch if key.startswith("observation.images."))
    if len(image_keys) > cfg.model.num_cameras:
        raise ValueError(
            f"batch 相机数 {len(image_keys)}（{image_keys}）超过配置"
            f" num_cameras={cfg.model.num_cameras}")
    images = [batch[key].to(device) for key in image_keys]
    image_masks = [
        torch.ones(image.shape[0], dtype=torch.bool, device=device)
        for image in images
    ]
    batch_size = batch["observation.language.tokens"].shape[0]
    for _ in range(cfg.model.num_cameras - len(image_keys)):
        images.append(torch.full(
            (batch_size, 3, cfg.model.image_size, cfg.model.image_size),
            -1.0, dtype=torch.float32, device=device))
        image_masks.append(torch.zeros(
            batch_size, dtype=torch.bool, device=device))
    token_ids = batch["observation.language.tokens"].to(device).long()
    lang_mask = batch["observation.language.attention_mask"].to(device).bool()
    actions = batch["action"].to(device).float()
    is_pad = batch.get("action_is_pad")
    is_pad = (
        torch.zeros_like(actions[..., 0]).bool()
        if is_pad is None
        else is_pad.to(device).bool()
    )
    real_action_dim = actions.shape[-1]
    actions = F.pad(
        actions,
        (0, cfg.model.action_dim - actions.shape[-1]),
    )
    return TrainBatch(
        images,
        image_masks,
        token_ids,
        lang_mask,
        actions,
        is_pad,
        real_action_dim,
    )
