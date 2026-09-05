"""顶层 pi05 模型：组装 SigLIP + Gemma embedding + MoT 容器 + 动作专家。

职责只有两件：
1. 按 pi05 的数据流把子模块接起来（图像/文本 -> VLM 段，状态/带噪动作 -> 动作段，
   两者进 MoT 共享 attention，动作段输出速度场）；
2. 提供 from_pretrained 入口，委托 weights.py 加载 openpi 格式的 checkpoint。

结构对齐 ref/openpi/src/openpi/models/pi0.py 的 Pi0 类。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.config import ModelConfig
from neat_pi.model.action_expert import ActionExpert
from neat_pi.model.flow_matching import FlowMatchingModel
from neat_pi.model.gemma import GemmaLM
from neat_pi.model.siglip import SigLIPVisionEncoder
from neat_pi.typing import (ActionBHD, ActionTokensBHD, CondBD, ImageBCHW,
                            LanguageTokensBTD, MaskB, MaskBL, MaskBT, TimeB,
                            TokenIdsBL,
                            VisionTokensBTD, typechecked)


class MultiModalProjector(nn.Module):
    """pi05 的视觉到语言宽度投影：单层 Linear，无激活函数。"""

    def __init__(self, vision_hidden_dim: int = 1152,
                 language_hidden_dim: int = 2048) -> None:
        super().__init__()
        self.linear = nn.Linear(vision_hidden_dim, language_hidden_dim, bias=True)

    @typechecked
    def forward(self, vision_tokens: VisionTokensBTD) -> LanguageTokensBTD:
        """投影 SigLIP token，得到可与语言 embedding 拼接的 VLM token。"""
        return self.linear(vision_tokens)


class Pi05(FlowMatchingModel):
    """pi05 视觉-语言-动作模型（flow matching 训练形态）。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_tower = SigLIPVisionEncoder(
            image_size=cfg.image_size,
            patch_size=cfg.vision_patch_size,
            hidden_dim=cfg.vision_hidden_dim,
            num_layers=cfg.vision_num_layers,
            num_heads=cfg.vision_num_heads,
            mlp_hidden_dim=cfg.vision_mlp_hidden_dim,
        )
        self.multi_modal_projector = MultiModalProjector(
            vision_hidden_dim=cfg.vision_hidden_dim,
            language_hidden_dim=cfg.vlm_hidden_dim,
        )
        self.language_model = GemmaLM(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.vlm_hidden_dim,
            num_layers=cfg.vlm_num_layers,
            num_heads=cfg.vlm_num_heads,
            num_kv_heads=cfg.vlm_num_kv_heads,
            attn_head_dim=cfg.vlm_attn_head_dim,
            mlp_hidden_dim=cfg.vlm_mlp_hidden_dim,
        )
        self.action_expert = ActionExpert(
            action_dim=cfg.action_dim,
            hidden_dim=cfg.expert_hidden_dim,
            num_layers=cfg.expert_num_layers,
            num_heads=cfg.expert_num_heads,
            num_kv_heads=cfg.expert_num_kv_heads,
            head_dim=cfg.expert_attn_head_dim,
            mlp_hidden_dim=cfg.expert_mlp_hidden_dim,
        )

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, cfg: ModelConfig,
                        device: torch.device) -> "Pi05":
        """构造模型并从 openpi 格式 checkpoint 加载权重。"""
        from neat_pi.model.weights import load_pi05_weights

        model = cls(cfg)
        load_pi05_weights(model, checkpoint_dir)
        return model.to(device)

    @typechecked
    def embed_prefix(self, images: list[ImageBCHW],
                     image_masks: list[MaskB], token_ids: TokenIdsBL,
                     lang_mask: MaskBL) -> tuple[LanguageTokensBTD, MaskBT]:
        """编码图像与语言并拼成 VLM prefix，返回 (embedding, 有效位 mask)。

        相机 embedding 按 images 的固定顺序沿序列维拼接；图像与语言段都是
        双向可见的前缀块。image_mask=False 的相机仍保留固定 token 槽位，
        但其整段视觉 token 在返回的 prefix mask 中置为 False。
        """
        embeddings: list[LanguageTokensBTD] = []
        pad_masks: list[MaskBT] = []
        if len(images) != len(image_masks):
            raise ValueError(
                f"images 数量 {len(images)} 须等于 image_masks 数量 "
                f"{len(image_masks)}")
        for image, image_mask in zip(images, image_masks, strict=True):
            vision_tokens = self.vision_tower(image)
            projected = self.multi_modal_projector(vision_tokens)
            embeddings.append(projected)
            pad_masks.append(image_mask[:, None].expand(-1, projected.shape[1]))

        language_tokens = self.language_model.embed_language_tokens(token_ids)
        embeddings.append(language_tokens)
        pad_masks.append(lang_mask)
        return torch.cat(embeddings, dim=1), torch.cat(pad_masks, dim=1)

    @typechecked
    def embed_suffix(self, noisy_action: ActionBHD,
                     t: TimeB) -> tuple[ActionTokensBHD, CondBD]:
        """编码带噪动作与 flow 时间，返回动作 token 和 adaRMS 条件向量。"""
        return self.action_expert.encode_tokens(noisy_action, t)

    @typechecked
    def predict_velocity(self, images: list[ImageBCHW],
                         image_masks: list[MaskB], token_ids: TokenIdsBL,
                         lang_mask: MaskBL,
                         noisy_action: ActionBHD, t: TimeB) -> ActionBHD:
        """训练前向：预测带噪动作的速度场。

        images 为多相机图像在 batch 维拼接后的形态（具体排布待数据管线定）。
        TODO: 串接 vision -> embedding -> action_expert.encode_tokens ->
        MoT -> action_expert.decode_velocity，并构造双段 attention mask。
        """
        raise NotImplementedError("待实现：见各子模块 TODO")

    @torch.no_grad()
    def sample_actions(self, images: list[ImageBCHW],
                       image_masks: list[MaskB], token_ids: TokenIdsBL,
                       lang_mask: MaskBL, num_steps: int = 10) -> ActionBHD:
        """推理：从噪声出发用 Euler 法积分 flow ODE，返回动作 chunk。

        TODO: 按 pi05 推理路径实现（x_1 ~ N(0,I) 置于 t=1 纯噪声端，
        t 从 1 到 0 以 dt = -1/num_steps 积分，约定见 flow_matching.py）。
        """
        raise NotImplementedError("待实现：flow matching 采样")
