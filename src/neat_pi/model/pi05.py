"""顶层 pi05 模型：组装 SigLIP + Gemma embedding + MoT 容器 + 动作专家。

职责只有两件：
1. 按 pi05 的数据流把子模块接起来（图像/文本 -> VLM 段，状态/带噪动作 -> 动作段，
   两者进 MoT 共享 attention，动作段输出速度场）；
2. 提供 from_pretrained 入口，委托 weights.py 加载 openpi 格式的 checkpoint。

结构对齐 ref/openpi/src/openpi/models/pi0.py 的 Pi0 类。
"""

from __future__ import annotations

import torch

from neat_pi.config import ModelConfig
from neat_pi.model.action_expert import ActionExpert
from neat_pi.model.flow_matching import FlowMatchingModel
from neat_pi.model.modules import RMSNorm
from neat_pi.model.siglip import SigLIPVisionEncoder
from neat_pi.typing import (ActionBHD, ImageBCHW, StateBD, TimeB, TokenIdsBL,
                            typechecked)


class Pi05(FlowMatchingModel):
    """pi05 视觉-语言-动作模型（flow matching 训练形态）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
 

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, cfg: ModelConfig,
                        device: torch.device) -> "Pi05":
        """构造模型并从 openpi 格式 checkpoint 加载权重。"""
        from neat_pi.model.weights import load_pi05_weights

        model = cls(cfg)
        load_pi05_weights(model, checkpoint_dir)
        return model.to(device)

    @typechecked
    def predict_velocity(self, images: list[ImageBCHW],
                         token_ids: TokenIdsBL, state: StateBD,
                         noisy_action: ActionBHD, t: TimeB) -> ActionBHD:
        """训练前向：预测带噪动作的速度场。

        images 为多相机图像在 batch 维拼接后的形态（具体排布待数据管线定）。
        TODO: 串接 vision -> embedding -> action_expert.encode_tokens ->
        MoT -> action_expert.decode_velocity，并构造双段 attention mask。
        """
        raise NotImplementedError("待实现：见各子模块 TODO")

    @torch.no_grad()
    def sample_actions(self, images: list[ImageBCHW], token_ids: TokenIdsBL,
                       state: StateBD, num_steps: int = 10) -> ActionBHD:
        """推理：从噪声出发用 Euler 法积分 flow ODE，返回动作 chunk。

        TODO: 按 pi05 推理路径实现（x_0 ~ N(0,I)，t 从 0 到 1 积分）。
        """
        raise NotImplementedError("待实现：flow matching 采样")
