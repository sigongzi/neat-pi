"""pi05 顶层模块测试：先验证 projector 与 tied 语言 embedding 积木。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neat_pi.model.gemma import GemmaLM
from neat_pi.model.pi05 import MultiModalProjector


@pytest.fixture
def generator() -> torch.Generator:
    """返回 CPU 固定种子随机数发生器。"""
    return torch.Generator(device="cpu").manual_seed(7)


def test_projector_shape(generator: torch.Generator) -> None:
    """投影层把 SigLIP 宽度映射到 VLM 宽度并保留 token 数。"""
    projector = MultiModalProjector(vision_width=16, language_width=24)
    vision_tokens = torch.randn(2, 8, 16, generator=generator)
    projected = projector(vision_tokens)
    assert projected.shape == (2, 8, 24)


def test_embed_language_tokens_matches_embedding(generator: torch.Generator) -> None:
    """语言 embedding 必须等价于 tied 表查表后乘 sqrt(width)。"""
    lm = GemmaLM(vocab_size=64, width=32, num_layers=1, num_heads=4,
                 num_kv_heads=1, attn_head_dim=8, mlp_hidden=64)
    token_ids = torch.randint(0, 64, (2, 5), generator=generator)
    expected = F.embedding(token_ids, lm.lm_head.weight) * 32 ** 0.5
    embedded = lm.embed_language_tokens(token_ids)
    torch.testing.assert_close(embedded, expected)
