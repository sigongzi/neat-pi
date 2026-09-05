"""siglip.py 单元测试：形状 / 梯度 / 与 naive 实现对拍 / checkpoint 权重 round-trip。

naive 参考实现只覆盖单层组件（MLP、attention），整塔用小的测试配置跑形状与
梯度，避免在 CPU 上实例化 27 层 1152 宽的真实塔。checkpoint round-trip 是
可选慢测试（仅当权重文件存在时跑），用 mmap 惰性读视觉塔权重后跑一次前向。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neat_pi.model.siglip import (SigLIPVisionEncoder, SiglipAttention,
                                  SiglipMLP)

CHECKPOINT = Path("/home/ivoryseagull/neat-pi-old/checkpoints/"
                  "pi05_libero_finetuned/model.safetensors")


def _small_encoder() -> SigLIPVisionEncoder:
    """小而完整的塔配置，用于快速形状/梯度测试。"""
    return SigLIPVisionEncoder(image_size=32, patch_size=8, hidden_dim=64,
                               num_layers=2, num_heads=4, mlp_hidden_dim=128)


# ---------------- SiglipMLP ----------------


def test_siglip_mlp_output_shape() -> None:
    """fc1 扩张到 hidden，fc2 收回 dim，输出形状与输入一致。"""
    mlp = SiglipMLP(dim=16, hidden_dim=64)
    x = torch.randn(2, 5, 16)
    assert mlp(x).shape == x.shape


def test_siglip_mlp_matches_naive() -> None:
    """与手写 fc1 -> GELU(tanh) -> fc2 逐元素一致。"""
    torch.manual_seed(0)
    mlp = SiglipMLP(dim=32, hidden_dim=128)
    x = torch.randn(3, 7, 32)
    expected = mlp.fc2(torch.nn.functional.gelu(mlp.fc1(x), approximate="tanh"))
    assert torch.allclose(mlp(x), expected, atol=1e-6)


def test_siglip_mlp_gradient_flows() -> None:
    """反向传播后 fc1/fc2 的 weight 与 bias 都有梯度。"""
    mlp = SiglipMLP(dim=16, hidden_dim=64)
    mlp(torch.randn(2, 5, 16)).sum().backward()
    assert mlp.fc1.weight.grad is not None and mlp.fc1.bias.grad is not None
    assert mlp.fc2.weight.grad is not None and mlp.fc2.bias.grad is not None


# ---------------- SiglipAttention ----------------


def _naive_attention(x: torch.Tensor, attn: SiglipAttention) -> torch.Tensor:
    """手写多头注意力参考实现，作为 SDPA 的对拍真值。"""
    batch, seq, dim = x.shape
    heads, head_dim = attn.num_heads, attn.head_dim
    q = attn.q_proj(x).view(batch, seq, heads, head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(batch, seq, heads, head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(batch, seq, heads, head_dim).transpose(1, 2)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k) * attn.scale
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("bhst,bhtd->bhsd", weights, v)
    out = out.transpose(1, 2).reshape(batch, seq, dim)
    return attn.out_proj(out)


def test_siglip_attention_output_shape() -> None:
    """输出形状与输入一致。"""
    attn = SiglipAttention(dim=64, num_heads=4)
    x = torch.randn(2, 10, 64)
    assert attn(x).shape == x.shape


def test_siglip_attention_matches_naive() -> None:
    """随机权重下与手写 einsum 注意力逐元素一致。"""
    torch.manual_seed(0)
    attn = SiglipAttention(dim=64, num_heads=4)
    x = torch.randn(2, 10, 64)
    assert torch.allclose(attn(x), _naive_attention(x, attn), atol=1e-5)


def test_siglip_attention_gradient_flows() -> None:
    """反向传播后四个投影的 weight 都有梯度。"""
    attn = SiglipAttention(dim=64, num_heads=4)
    attn(torch.randn(2, 10, 64)).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert getattr(attn, name).weight.grad is not None


# ---------------- SigLIPVisionEncoder ----------------


def test_vision_encoder_output_shape() -> None:
    """32x32 输入 / patch 8 -> 4x4=16 个 token，宽度 64。"""
    enc = _small_encoder()
    out = enc(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 16, 64)


def test_vision_encoder_defaults_match_pi05() -> None:
    """默认超参数对应 pi05 SigLIP-SO400m：224 输入 -> 256 token / 1152 维。"""
    enc = SigLIPVisionEncoder()
    assert enc.embeddings.num_patches == 256
    assert enc.embeddings.patch_embedding.weight.shape == (1152, 3, 14, 14)
    assert enc.embeddings.position_embedding.weight.shape == (256, 1152)
    assert len(enc.encoder.layers) == 27
    assert enc.post_layernorm.weight.shape == (1152,)


def test_vision_encoder_gradient_flows() -> None:
    """小塔反向传播后，patch embedding / position / 各层参数都有梯度。"""
    enc = _small_encoder()
    enc(torch.randn(1, 3, 32, 32)).sum().backward()
    assert enc.embeddings.patch_embedding.weight.grad is not None
    assert enc.embeddings.position_embedding.weight.grad is not None
    first = enc.encoder.layers[0]
    assert first.self_attn.q_proj.weight.grad is not None
    assert first.mlp.fc1.weight.grad is not None


# ---------------- checkpoint 权重 round-trip（可选慢测试） ----------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint 不存在")
def test_vision_encoder_loads_checkpoint_weights() -> None:
    """用 mmap 惰性读视觉塔权重 copy_ 进模型后，前向可跑且形状正确。

    复用 weights.load_siglip_vision_weights（与 scripts/check_vision_weights.py
    共用同一加载路径）；只加载视觉塔，不整载入 7.5GB checkpoint。
    """
    from neat_pi.model.weights import load_siglip_vision_weights

    enc = SigLIPVisionEncoder()
    loaded = load_siglip_vision_weights(enc, str(CHECKPOINT.parent))
    assert loaded == 437  # 恰好是视觉塔的完整参数

    with torch.inference_mode():
        out = enc(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 256, 1152)
    assert torch.isfinite(out).all()
