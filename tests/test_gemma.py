"""gemma.py 单元测试：形状 / 梯度 / RoPE / 注意力对拍 / 名字映射。

naive 参考只覆盖单层注意力（含 causal mask）；整模型用小的测试配置跑形状、
梯度与贪心生成。checkpoint 名字映射测试只做字符串与张量数量的对拍（不实例化
2.6B 的真实模型），真实权重加载 + 生成由 scripts/check_gemma_lm.py 验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from neat_pi.model.gemma import (GemmaAttention, GemmaLM, apply_rotary_pos_emb,
                                 build_rope_cache, rotate_half)
from neat_pi.model.weights import translate_gemma_lm_name

CHECKPOINT = Path("/home/ivoryseagull/neat-pi-old/checkpoints/"
                  "pi05_libero_finetuned/model.safetensors")


def _small_lm() -> GemmaLM:
    """小而完整的 GemmaLM，用于快速形状/梯度/生成测试。"""
    return GemmaLM(vocab_size=64, width=64, num_layers=2, num_heads=4,
                   num_kv_heads=1, head_dim=16, mlp_hidden=128)


def _naive_attention(x: torch.Tensor, attn: GemmaAttention,
                     cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """手写 GQA + causal 注意力参考实现（RoPE 复用模块函数）。"""
    batch, seq, dim = x.shape
    q = attn.q_proj(x).view(batch, seq, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(batch, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    n_rep = attn.num_heads // attn.num_kv_heads
    k = k[:, :, None, :, :].expand(batch, attn.num_kv_heads, n_rep, seq,
                                   attn.head_dim).reshape(batch, attn.num_heads, seq, attn.head_dim)
    v = v[:, :, None, :, :].expand(batch, attn.num_kv_heads, n_rep, seq,
                                   attn.head_dim).reshape(batch, attn.num_heads, seq, attn.head_dim)

    scores = torch.einsum("bhsd,bhtd->bhst", q, k) * attn.scale
    causal = torch.full((seq, seq), float("-inf"), device=x.device).triu(diagonal=1)
    scores = scores + causal
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.einsum("bhst,bhtd->bhsd", weights, v)
    out = out.transpose(1, 2).reshape(batch, seq, dim)
    return attn.o_proj(out)


# ---------------- RoPE ----------------


def test_rotate_half_swaps_and_negates() -> None:
    """rotate_half 把前半段移到后半、后半取反移到前半。"""
    x = torch.randn(2, 4, 8)
    y = rotate_half(x)
    assert torch.equal(y[..., :4], -x[..., 4:])
    assert torch.equal(y[..., 4:], x[..., :4])


def test_rope_preserves_norm() -> None:
    """RoPE 是旋转，不改变向量长度。"""
    x = torch.randn(3, 2, 6, 16)
    cos, sin = build_rope_cache(6, 16, 10_000.0, x.device, x.dtype)
    y, _ = apply_rotary_pos_emb(x, x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


# ---------------- GemmaAttention ----------------


def test_gemma_attention_output_shape() -> None:
    """输出形状与输入一致。"""
    attn = GemmaAttention(dim=64, num_heads=4, num_kv_heads=1, head_dim=16)
    x = torch.randn(2, 10, 64)
    cos, sin = build_rope_cache(10, 16, 10_000.0, x.device, x.dtype)
    assert attn(x, cos, sin).shape == x.shape


def test_gemma_attention_matches_naive() -> None:
    """随机权重下与手写 causal GQA 注意力逐元素一致。"""
    torch.manual_seed(0)
    attn = GemmaAttention(dim=64, num_heads=4, num_kv_heads=1, head_dim=16)
    x = torch.randn(2, 10, 64)
    cos, sin = build_rope_cache(10, 16, 10_000.0, x.device, x.dtype)
    assert torch.allclose(attn(x, cos, sin), _naive_attention(x, attn, cos, sin),
                          atol=1e-4)


def test_gemma_attention_gradient_flows() -> None:
    """反向传播后四个投影的 weight 都有梯度。"""
    attn = GemmaAttention(dim=64, num_heads=4, num_kv_heads=1, head_dim=16)
    x = torch.randn(2, 10, 64)
    cos, sin = build_rope_cache(10, 16, 10_000.0, x.device, x.dtype)
    attn(x, cos, sin).sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert getattr(attn, name).weight.grad is not None


# ---------------- GemmaLM ----------------


def test_gemma_lm_logits_shape() -> None:
    """输入 [batch, seq]，输出 [batch, seq, vocab]。"""
    lm = _small_lm()
    ids = torch.randint(0, 64, (2, 7))
    assert lm(ids).shape == (2, 7, 64)


def test_gemma_lm_gradient_flows() -> None:
    """反向传播后 lm_head / 层内参数都有梯度。"""
    lm = _small_lm()
    lm(torch.randint(0, 64, (2, 7))).sum().backward()
    assert lm.lm_head.weight.grad is not None
    assert lm.layers[0].self_attn.q_proj.weight.grad is not None
    assert lm.layers[0].mlp.gate_proj.weight.grad is not None


def test_gemma_lm_generate_grows_sequence() -> None:
    """贪心生成返回更长序列，且长度为 prompt + max_new_tokens（随机权重不会遇 eos）。"""
    lm = _small_lm()
    ids = torch.randint(0, 64, (1, 3))
    out = lm.generate(ids, max_new_tokens=5)
    assert out.shape[1] == 8
    assert torch.equal(out[:, :3], ids)


def test_gemma_lm_tied_embedding_is_lm_head() -> None:
    """embedding 直接查 lm_head.weight，二者是同一份参数。"""
    lm = _small_lm()
    ids = torch.randint(0, 64, (1, 4))
    emb = F.embedding(ids, lm.lm_head.weight)
    # 默认构造下 lm_head.weight 就是唯一 embedding 来源（无独立 embed_tokens）
    assert not hasattr(lm, "embed_tokens")
    assert torch.equal(emb, lm.lm_head.weight[ids])


# ---------------- checkpoint 名字映射（不实例化真实模型） ----------------


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint 不存在")
def test_gemma_lm_name_translation() -> None:
    """checkpoint 中 VLM 语言主干的张量全部可翻译，且恰好 164 张。"""
    from safetensors import safe_open

    mapped = 0
    with safe_open(str(CHECKPOINT), framework="pt") as f:
        for name in f.keys():
            if translate_gemma_lm_name(name) is not None:
                mapped += 1
    assert mapped == 164

    # 几个代表名字的翻译结果
    prefix = "model.paligemma_with_expert.paligemma.model.language_model."
    assert translate_gemma_lm_name(prefix + "layers.0.self_attn.q_proj.weight") == \
        "layers.0.self_attn.q_proj.weight"
    assert translate_gemma_lm_name(prefix + "norm.weight") == "norm.weight"
    assert translate_gemma_lm_name(
        "model.paligemma_with_expert.paligemma.lm_head.weight") == "lm_head.weight"
    # 动作专家的 lm_head / 视觉塔不属于 VLM 语言主干
    assert translate_gemma_lm_name(
        "model.paligemma_with_expert.gemma_expert.lm_head.weight") is None
    assert translate_gemma_lm_name(
        prefix.replace("language_model", "vision_tower") + "foo") is None
