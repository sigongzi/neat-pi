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
from neat_pi.device.backend import autocast
from neat_pi.model.action_expert import ActionExpert
from neat_pi.model.flow_matching import FlowMatchingModel
from neat_pi.model.gemma import GemmaLM
from neat_pi.model.mot import MoT
from neat_pi.model.siglip import SigLIPVisionEncoder
from neat_pi.typing import (ActionBHD, ActionTokensBHD, AttentionMaskBHLS,
                            CondBD, ImageBCHW, LanguageTokensBTD, MaskB,
                            MaskBL, MaskBT, TimeB, TokenIdsBL,
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
    def from_pretrained(cls, checkpoint_dir: str,
                        cfg: ModelConfig | None = None,
                        device: torch.device | None = None) -> "Pi05":
        """构造模型并从原始或已转换的 checkpoint 直接加载权重。

        pi05_base 的结构默认值与 ``ModelConfig`` 相同，因此推理可直接省略
        cfg；训练 / 小配置实验显式传入，避免隐式依赖 checkpoint 目录里的
        第二份结构配置。未指定设备时保持 CPU，由调用方继续做设备抽象与
        搬移。
        """
        from neat_pi.model.weights import load_pi05_weights

        model = cls(cfg or ModelConfig())
        load_pi05_weights(model, checkpoint_dir)
        return model if device is None else model.to(device)

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

    @staticmethod
    @typechecked
    def _joint_attention_mask(prefix_pad_mask: MaskBT,
                              action_horizon: int) -> AttentionMaskBHLS:
        """构造 canonical joint mask；prefill 和去噪 mask 由此切片。

        prefix 段双向可见；action 段可见 prefix 和整个动作 chunk。SDPA 只需
        屏蔽无效 key；无效 query 行可继续 attend 有效 prefix，从而避免
        all-masked row 产生 NaN，其输出本来也不会被读取。
        """
        prefix_len = prefix_pad_mask.shape[1]
        if prefix_len == 0 or not prefix_pad_mask.any():
            raise ValueError("prefix_pad_mask 必须包含至少一个有效 token")

        suffix_pad_mask = torch.ones(
            prefix_pad_mask.shape[0], action_horizon,
            dtype=torch.bool, device=prefix_pad_mask.device)
        # prefix_pad_mask: [B, P]
        # suffix_pad_mask: [B, H]
        key_pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim=1)
        # key_pad_mask:             [B, S]，其中 S = P + H
        # key_pad_mask[:, None, :]: [B, 1, S_key]

        ar_mask = torch.zeros_like(key_pad_mask)
        ar_mask[:, prefix_len] = True
        # ar_mask:  [B, S]
        ar_cumsum = torch.cumsum(ar_mask, dim=1, dtype=torch.long)
        # ar_cumsum: [B, S]

        # openpi 的块可见性语义：cumsum(key) <= cumsum(query)。prefix 内为
        # 双向块；action 首个 token 开启新块，整个动作 chunk 对 action query
        # 可见。这里只屏蔽无效 key；无效 query 行仍可 attend 有效 prefix，
        # 避免 all-masked row 的 NaN 进入后续 K/V，其输出不会被读取。
        #
        # 以 P=3、H=2、key 全部有效为例：
        #   key_pad   = [T, T, T | T, T]
        #   ar_mask   = [0, 0, 0 | 1, 0]
        #   ar_cumsum = [0, 0, 0 | 1, 1]
        #
        # 行是 query，列是 key；T 表示最终可见：
        #            key 0  1  2  3  4
        #   query 0:      T  T  T  F  F
        #   query 1:      T  T  T  F  F
        #   query 2:      T  T  T  F  F
        #   query 3:      T  T  T  T  T
        #   query 4:      T  T  T  T  T
        #
        # 若 key_pad 变成 [T, F, T | T, T]，下面 &= 会把 key=1 这一整列
        # 清成 F；query=1 虽然自身无效，但仍可看 key=0/2，不会整行全 False。
        #
        # ar_cumsum[:, None, :]: [B, 1, S_key]
        # ar_cumsum[:, :, None]: [B, S_query, 1]
        # 比较后广播成：          [B, S_query, S_key]
        # key_pad_mask 广播时：    [B, 1, S_key]
        # 因此 mask 是 SDPA 约定的 [B, query, key]。
        mask = (ar_cumsum[:, None, :] <= ar_cumsum[:, :, None])
        mask &= key_pad_mask[:, None, :]
        # mask: [B, S_query, S_key]
        return mask[:, None]
        # return: [B, 1, S_query, S_key]

    @typechecked
    def predict_velocity(self, images: list[ImageBCHW],
                         image_masks: list[MaskB], token_ids: TokenIdsBL,
                         lang_mask: MaskBL,
                         noisy_action: ActionBHD, t: TimeB) -> ActionBHD:
        """训练前向：预测带噪动作的速度场。

        先分别编码 prefix 和 suffix，再用 MoT 做共享 attention 的 fused
        前向；只有 action 段需要最终 adaRMS norm 和速度投影。
        """
        prefix_embeds, prefix_pad_mask = self.embed_prefix(
            images, image_masks, token_ids, lang_mask)
        action_tokens, adarms_cond = self.embed_suffix(noisy_action, t)
        joint_mask = self._joint_attention_mask(
            prefix_pad_mask, noisy_action.shape[1])

        mot = MoT({
            "vlm": self.language_model,
            "action": self.action_expert,
        })
        output = mot({
            "vlm": prefix_embeds,
            "action": action_tokens,
        }, attention_mask=joint_mask, conds={
            "vlm": None,
            "action": adarms_cond,
        })

        # MoT 不做专家级收尾；VLM 输出无 LM loss 不需要 norm，
        # action 最终 adaRMS 的 gate 无残差可消费，按参考实现丢弃。
        action_hidden, _ = self.action_expert.norm(
            output["action"], adarms_cond)
        return self.action_expert.decode_velocity(action_hidden)

    @torch.no_grad()
    def sample_actions(self, images: list[ImageBCHW],
                       image_masks: list[MaskB], token_ids: TokenIdsBL,
                       lang_mask: MaskBL, num_steps: int = 10) -> ActionBHD:
        """推理：从噪声出发用 Euler 法积分 flow ODE，返回动作 chunk。

        prefix embedding 与 VLM 侧的每层 k/v 只计算一次；所有去噪步
        复用 PrefixKVCache，只重算 action expert。flow 约定与训练一致：
        t=1 是纯噪声端，t=0 是数据端。
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps 必须大于 0，实际为 {num_steps}")

        batch_size = token_ids.shape[0]
        device = token_ids.device
        # ODE 状态和输出保持 float32，避免低精度累进误差；模型计算由
        # 与训练相同的 autocast 配置控制。
        noisy_action = torch.randn(
            batch_size, self.cfg.action_horizon, self.cfg.action_dim,
            device=device, dtype=torch.float32)
        delta_time = -1.0 / num_steps

        with autocast(self.ctx, self.amp_dtype):
            prefix_embeds, prefix_pad_mask = self.embed_prefix(
                images, image_masks, token_ids, lang_mask)
            joint_mask = self._joint_attention_mask(
                prefix_pad_mask, self.cfg.action_horizon)
            prefix_len = prefix_embeds.shape[1]
            prefix_kvs = self.language_model.prefill(
                prefix_embeds, joint_mask[:, :, :prefix_len, :prefix_len])
            action_attention_mask = joint_mask[:, :, prefix_len:, :]

            action = noisy_action
            for step in range(num_steps):
                time = 1.0 + step * delta_time
                time_tensor = torch.full(
                    (batch_size,), time, device=device, dtype=torch.float32)
                action_tokens, adarms_cond = self.embed_suffix(
                    action, time_tensor)
                action_hidden = self.action_expert.run_layers(
                    action_tokens, adarms_cond, prefix_kvs,
                    action_attention_mask)
                action_hidden, _ = self.action_expert.norm(
                    action_hidden, adarms_cond)
                velocity = self.action_expert.decode_velocity(action_hidden)
                action = action + delta_time * velocity.float()

        return action
