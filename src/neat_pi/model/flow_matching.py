"""flow matching 训练目标基类。

把原本散落在 scripts/train.py 里的采样与损失逻辑收敛进模型内部，
使模型的单次 forward 直接返回标量损失，训练循环只需 backward。
子类只需实现 predict_velocity（预测带噪动作的速度场），
采样时间步/噪声、构造目标、按有效步与有效维做 masked MSE 都在这里完成。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.device.backend import DeviceContext, autocast
from neat_pi.typing import (ActionBHD, ImageBCHW, MaskB, MaskBH, MaskBL,
                            ScalarLoss, TimeB, TokenIdsBL, typechecked)


class FlowMatchingModel(nn.Module):
    """flow matching 训练基类：封装采样与损失，子类实现 predict_velocity。

    `ctx` / `amp_dtype` 由训练脚本在构建后注入（见 scripts/train.py 的
    build_model），不经构造函数传递——FSDP 包装、from_pretrained 等路径
    都统一走事后注入，避免多条赋值路径。
    """

    def __init__(self) -> None:
        super().__init__()
        self.ctx: DeviceContext | None = None
        self.amp_dtype: torch.dtype = torch.float32

    @typechecked
    def predict_velocity(self, images: list[ImageBCHW],
                         image_masks: list[MaskB], token_ids: TokenIdsBL,
                         lang_mask: MaskBL, noisy_action: ActionBHD,
                         t: TimeB) -> ActionBHD:
        """预测带噪动作的速度场（由子类实现）。"""
        raise NotImplementedError

    @typechecked
    def sample_time(self, bsize: int, device: torch.device) -> TimeB:
        """采样 flow 时间步：Beta(1.5, 1.0) 后仿射到 [0.001, 1.0]。

        对齐 ref/openpi pi0_pytorch 的 sample_time：密度 ∝ √t，向高噪声端
        加权；仿射避开 t=0（数据端）与 t=1（噪声端）的退化取值。
        """
        alpha = torch.as_tensor(1.5, dtype=torch.float32, device=device)
        beta = torch.as_tensor(1.0, dtype=torch.float32, device=device)
        time_beta = torch.distributions.Beta(alpha, beta).sample((bsize,))
        return (time_beta * 0.999 + 0.001).to(
            dtype=torch.float32, device=device)

    @typechecked
    def forward(self, images: list[ImageBCHW], image_masks: list[MaskB],
                token_ids: TokenIdsBL, lang_mask: MaskBL, actions: ActionBHD,
                is_pad: MaskBH,
                real_action_dim: int) -> ScalarLoss:
        """采样 t 与噪声，预测速度场，返回有效步/维上的 masked MSE 标量损失。

        时间约定对齐 ref/openpi：t=1 纯噪声、t=0 数据，x_t = t*noise +
        (1-t)*data，目标速度 u_t = noise - data；t 从 Beta(1.5, 1.0) 采样
        （见 sample_time，对齐 ref/openpi pi0_pytorch）。推理时从 t=1
        （x = 噪声）以负步长积分回 t=0（见 Pi05.sample_actions，方向须与
        本处一致）。padding 的时间步（episode 末尾）与补零的动作维不参与
        损失。
        """
        b = actions.shape[0]
        t = self.sample_time(b, actions.device)
        noise = torch.randn_like(actions)
        # 时间约定对齐 ref/openpi：t=1 纯噪声、t=0 数据，x_t = t*noise + (1-t)*data，
        # 目标速度 u_t = noise - data；采样时从 t=1（x = 噪声）积分回 t=0
        t_ = t.view(b, 1, 1)
        noisy_action = t_ * noise + (1 - t_) * actions
        target = noise - actions

        with autocast(self.ctx, self.amp_dtype):
            pred = self.predict_velocity(images, image_masks, token_ids,
                                         lang_mask, noisy_action, t)

        pred = pred.float()
        target = target.float()
        dim_mask = torch.zeros_like(target)
        dim_mask[..., :real_action_dim] = 1.0
        mask = (~is_pad).unsqueeze(-1) * dim_mask
        return (pred - target).pow(2).mul(mask).sum() / mask.sum().clamp(min=1.0)
