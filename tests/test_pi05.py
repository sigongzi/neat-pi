"""pi05 顶层模块测试：先验证 projector 与 tied 语言 embedding 积木。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neat_pi.config import ModelConfig
from neat_pi.model.gemma import GemmaLM
from neat_pi.model.pi05 import MultiModalProjector, Pi05
from neat_pi.training.dummy_model import DummyPi05


@pytest.fixture
def generator() -> torch.Generator:
    """返回 CPU 固定种子随机数发生器。"""
    return torch.Generator(device="cpu").manual_seed(7)


def test_projector_shape(generator: torch.Generator) -> None:
    """投影层把 SigLIP 宽度映射到 VLM 宽度并保留 token 数。"""
    projector = MultiModalProjector(vision_hidden_dim=16,
                                    language_hidden_dim=24)
    vision_tokens = torch.randn(2, 8, 16, generator=generator)
    projected = projector(vision_tokens)
    assert projected.shape == (2, 8, 24)


def test_embed_language_tokens_matches_embedding(generator: torch.Generator) -> None:
    """语言 embedding 必须等价于 tied 表查表后乘 sqrt(hidden_dim)。"""
    lm = GemmaLM(vocab_size=64, hidden_dim=32, num_layers=1, num_heads=4,
                 num_kv_heads=1, attn_head_dim=8, mlp_hidden_dim=64)
    token_ids = torch.randint(0, 64, (2, 5), generator=generator)
    expected = F.embedding(token_ids, lm.lm_head.weight) * 32 ** 0.5
    embedded = lm.embed_language_tokens(token_ids)
    torch.testing.assert_close(embedded, expected)


def test_dummy_model_requires_lang_mask(generator: torch.Generator) -> None:
    """训练接口去掉 state 后，DummyPi05 与 Pi05 签名一样要求语言 padding mask。"""
    model = DummyPi05(ModelConfig(action_dim=8))
    noisy_action = torch.randn(2, 3, 8, generator=generator)
    lang_mask = torch.ones(2, 4, dtype=torch.bool)
    token_ids = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(TypeError):
        model.predict_velocity([], token_ids, lang_mask, noisy_action,
                               torch.rand(2, generator=generator))
    velocity = model.predict_velocity([], [], token_ids, lang_mask,
                                      noisy_action,
                                      torch.rand(2, generator=generator))
    assert velocity.shape == noisy_action.shape


def test_embed_prefix_concatenates_cameras_and_language(
        generator: torch.Generator) -> None:
    """embed_prefix 按相机顺序拼接投影结果，再拼接语言 embedding 与 mask。"""
    cfg = ModelConfig(
        image_size=16, num_cameras=2, vision_hidden_dim=16, vision_patch_size=4,
        vision_num_layers=1, vision_num_heads=4, vision_mlp_hidden_dim=32,
        vocab_size=64, vlm_hidden_dim=24, vlm_num_layers=1, vlm_num_heads=4,
        vlm_num_kv_heads=1, vlm_attn_head_dim=8, vlm_mlp_hidden_dim=48,
    )
    model = Pi05(cfg)
    images = [torch.randn(2, 3, 16, 16, generator=generator) for _ in range(2)]
    token_ids = torch.randint(0, 64, (2, 6), generator=generator)
    lang_mask = torch.tensor([[True] * 4 + [False] * 2,
                              [True] * 6, ], dtype=torch.bool)
    image_masks = [torch.ones(2, dtype=torch.bool) for _ in images]
    prefix, prefix_mask = model.embed_prefix(
        images, image_masks, token_ids, lang_mask)

    expected = torch.cat([
        model.multi_modal_projector(model.vision_tower(images[0])),
        model.multi_modal_projector(model.vision_tower(images[1])),
        model.language_model.embed_language_tokens(token_ids),
    ], dim=1)
    torch.testing.assert_close(prefix, expected)
    assert prefix.shape == (2, 38, 24)
    torch.testing.assert_close(
        prefix_mask,
        torch.tensor([[True] * 36 + [False] * 2,
                                      [True] * 38], dtype=torch.bool),
    )


def test_embed_prefix_masks_disabled_camera(generator: torch.Generator) -> None:
    """缺失相机的所有视觉 token 槽位保留，但整段 mask 必须为 False。"""
    cfg = ModelConfig(
        image_size=16, num_cameras=2, vision_hidden_dim=16, vision_patch_size=4,
        vision_num_layers=1, vision_num_heads=4, vision_mlp_hidden_dim=32,
        vocab_size=64, vlm_hidden_dim=24, vlm_num_layers=1, vlm_num_heads=4,
        vlm_num_kv_heads=1, vlm_attn_head_dim=8, vlm_mlp_hidden_dim=48,
    )
    model = Pi05(cfg)
    images = [torch.randn(2, 3, 16, 16, generator=generator),
              torch.zeros(2, 3, 16, 16)]
    image_masks = [torch.ones(2, dtype=torch.bool),
                   torch.zeros(2, dtype=torch.bool)]
    token_ids = torch.randint(0, 64, (2, 4), generator=generator)
    lang_mask = torch.ones(2, 4, dtype=torch.bool)
    prefix, prefix_mask = model.embed_prefix(
        images, image_masks, token_ids, lang_mask)

    assert prefix.shape == (2, 36, 24)
    assert prefix_mask[:, :16].all()
    assert not prefix_mask[:, 16:32].any()
    assert prefix_mask[:, 32:].all()

    with pytest.raises(ValueError):
        model.embed_prefix(images, image_masks[:1], token_ids, lang_mask)
