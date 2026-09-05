"""冒烟用哑模型：形状对齐 Pi05.predict_velocity，仅用于跑通训练循环。

Pi05 的 predict_velocity 尚未实现（见 model/pi05.py），训练循环需要一个
可前向、可反向、可被 AdamW 更新的模型来端到端冒烟。DummyPi05
不做任何真实计算，只把 noisy_action 过一个线性层——loss 会下降，
梯度/优化器/checkpoint 路径都能被验证。
"""

from __future__ import annotations

from torch import nn

from neat_pi.config import ModelConfig
from neat_pi.model.flow_matching import FlowMatchingModel
from neat_pi.typing import (ActionBHD, ImageBCHW, MaskBL, TimeB, TokenIdsBL,
                            typechecked)


class DummyPi05(FlowMatchingModel):
    """速度场 = linear(noisy_action) 的哑模型，predict_velocity 与 Pi05 对齐。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Linear(cfg.action_dim, cfg.action_dim, bias=True)

    @typechecked
    def predict_velocity(self, images: list[ImageBCHW], token_ids: TokenIdsBL,
                         lang_mask: MaskBL, noisy_action: ActionBHD,
                         t: TimeB) -> ActionBHD:
        """忽略图像/文本/时间，直接对带噪动作做线性变换返回速度场。"""
        return self.head(noisy_action)
