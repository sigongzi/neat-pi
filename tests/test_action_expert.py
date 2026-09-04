"""action_expert.py 单元测试：形状 / 梯度 / sincos 嵌入 / 共享注意力对拍。

用小的测试配置（非 1024 宽、非 18 层）跑快速测试；真实权重加载 +
数值对照由 scripts/ 层的检查脚本负责。
"""

from __future__ import annotations

import torch

from neat_pi.model.action_expert import (ActionExpert, DiTBlock,
                                         posemb_sincos)
from neat_pi.model.util import apply_rotary_pos_emb, build_rope_cache


def _small_expert(num_layers: int = 2) -> ActionExpert:
    """小而完整的 ActionExpert，用于快速形状/梯度测试。"""
    return ActionExpert(action_dim=7, hidden_dim=64, num_layers=num_layers,
                        num_heads=4, num_kv_heads=1, head_dim=16,
                        mlp_hidden=128)


# ---------------- posemb_sincos ----------------


def test_posemb_sincos_shape_and_range() -> None:
    """输出 [batch, embedding_dim]，取值在 [-1, 1]。"""
    emb = posemb_sincos(torch.rand(5), 64)
    assert emb.shape == (5, 64)
    assert emb.abs().max() <= 1.0 + 1e-6


def test_posemb_sincos_odd_dim_rejected() -> None:
    """embedding_dim 为奇数时报错。"""
    try:
        posemb_sincos(torch.rand(2), 63)
    except ValueError:
        return
    raise AssertionError("奇数 embedding_dim 应抛 ValueError")


# ---------------- DiTBlock ----------------


def test_dit_block_output_shape() -> None:
    """自注意力（无 VLM kv）与拼接 VLM kv 两种形态输出形状都不变。"""
    block = DiTBlock(width=64, num_heads=4, num_kv_heads=1, head_dim=16,
                     mlp_hidden=128)
    x = torch.randn(2, 10, 64)
    cond = torch.randn(2, 64)
    cos, sin = build_rope_cache(10, 16, 10_000.0, x.device, x.dtype)
    assert block(x, cond, cos, sin).shape == x.shape

    prefix_len = 6
    cos_full, sin_full = build_rope_cache(prefix_len + 10, 16, 10_000.0,
                                          x.device, x.dtype)
    vlm_kv = (torch.randn(2, 1, prefix_len, 16),
              torch.randn(2, 1, prefix_len, 16))
    out = block(x, cond, cos_full[prefix_len:], sin_full[prefix_len:], vlm_kv)
    assert out.shape == x.shape


def test_dit_block_vlm_kv_matches_naive() -> None:
    """拼接 VLM kv 的注意力与手写 naive 实现逐元素一致。

    adaLN-Zero 零初始化下 scale/shift/gate 全零：norm 输出即裸 RMS(x)，
    残差支路为恒等——正好隔离出注意力支路做精确对拍。
    """
    torch.manual_seed(0)
    block = DiTBlock(width=64, num_heads=4, num_kv_heads=1, head_dim=16,
                     mlp_hidden=128)
    x = torch.randn(2, 10, 64)
    cond = torch.randn(2, 64)
    prefix_len = 6
    vlm_kv = (torch.randn(2, 1, prefix_len, 16),
              torch.randn(2, 1, prefix_len, 16))
    cos, sin = build_rope_cache(10, 16, 10_000.0, x.device, x.dtype)

    out = block(x, cond, cos, sin, vlm_kv)

    # naive：gate=0 -> normed 是裸 RMS(x)；attn 支路被 gate=0 清零；
    # MLP 支路同样被清零，最终输出应为恒等 x
    assert torch.allclose(out, x, atol=1e-5)

    # 把 gate 撬开（dense.bias 第三段置 1），再与手写注意力对拍
    with torch.no_grad():
        block.input_layernorm.dense.bias[2 * 64:] = 1.0
    out = block(x, cond, cos, sin, vlm_kv)

    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = normed.to(x.dtype)
    attn = block.self_attn
    q = attn.q_proj(normed).view(2, 10, 4, 16).transpose(1, 2)
    k = attn.k_proj(normed).view(2, 10, 1, 16).transpose(1, 2)
    v = attn.v_proj(normed).view(2, 10, 1, 16).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = torch.cat([vlm_kv[0], k], dim=2).expand(2, 4, 16, 16)
    v = torch.cat([vlm_kv[1], v], dim=2).expand(2, 4, 16, 16)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k) * attn.scale
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    naive = torch.einsum("bhst,bhtd->bhsd", weights, v)
    naive = attn.o_proj(naive.transpose(1, 2).reshape(2, 10, 64))
    # MLP 支路 gate 仍为 0，输出 = x + naive_attn
    assert torch.allclose(out, x + naive, atol=1e-4)


# ---------------- ActionExpert ----------------


def test_action_expert_end_to_end_shape() -> None:
    """encode -> run_layers -> decode 全链路形状正确。"""
    expert = _small_expert()
    tokens, cond = expert.encode_tokens(torch.randn(2, 50, 7), torch.rand(2))
    assert tokens.shape == (2, 50, 64)
    assert cond.shape == (2, 64)
    v = expert.decode_velocity(expert.run_layers(tokens, cond))
    assert v.shape == (2, 50, 7)


def test_action_expert_gradient_flows() -> None:
    """反向传播后投影、时间 MLP、DiTBlock 参数都有梯度。"""
    expert = _small_expert()
    tokens, cond = expert.encode_tokens(torch.randn(2, 50, 7), torch.rand(2))
    expert.decode_velocity(expert.run_layers(tokens, cond)).sum().backward()
    assert expert.action_in_proj.weight.grad is not None
    assert expert.time_mlp_out.weight.grad is not None
    assert expert.layers[0].self_attn.q_proj.weight.grad is not None
    assert expert.layers[0].mlp.gate_proj.weight.grad is not None
    assert expert.norm.dense.weight.grad is not None


def test_action_expert_checkpoint_param_names() -> None:
    """state_dict 名字覆盖 checkpoint gemma_expert 训练所需的全部张量。"""
    expert = _small_expert(num_layers=1)
    names = set(expert.state_dict())
    expected = {
        "action_in_proj.weight", "action_in_proj.bias",
        "action_out_proj.weight", "action_out_proj.bias",
        "time_mlp_in.weight", "time_mlp_in.bias",
        "time_mlp_out.weight", "time_mlp_out.bias",
        "norm.dense.weight", "norm.dense.bias",
    }
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        expected.add(f"layers.0.self_attn.{proj}.weight")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        expected.add(f"layers.0.mlp.{proj}.weight")
    for norm in ("input_layernorm", "post_attention_layernorm"):
        expected.add(f"layers.0.{norm}.dense.weight")
        expected.add(f"layers.0.{norm}.dense.bias")
    assert expected <= names


def test_action_expert_forward_matches_composition() -> None:
    """forward 等于 encode -> run_layers -> decode 的手动组合。"""
    torch.manual_seed(0)
    expert = _small_expert()
    noisy_action = torch.randn(2, 50, 7)
    t = torch.rand(2)
    tokens, cond = expert.encode_tokens(noisy_action, t)
    manual = expert.decode_velocity(expert.run_layers(tokens, cond))
    assert torch.equal(expert(noisy_action, t), manual)


def test_action_expert_forward_with_prefix_cache() -> None:
    """forward 直接消费 GemmaLM.prefill 产出的 PrefixKVCache，形状正确。

    小配置下 VLM 与动作专家同宽同 head_dim（kv 头数一致才能拼 k/v）。
    """
    from neat_pi.model.gemma import GemmaLM

    torch.manual_seed(0)
    vlm = GemmaLM(vocab_size=64, width=64, num_layers=2, num_heads=4,
                  num_kv_heads=1, attn_head_dim=16, mlp_hidden=128)
    expert = _small_expert()
    cache = vlm.prefill(torch.randn(2, 7, 64))
    v = expert(torch.randn(2, 50, 7), torch.rand(2), vlm_kvs=cache)
    assert v.shape == (2, 50, 7)
    assert not v.isnan().any()
