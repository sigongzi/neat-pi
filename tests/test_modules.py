"""modules.py 单元测试：与 naive 实现对拍 + 形状/dtype/梯度性质。

参考真值：ref/openpi modeling_gemma.py / modeling_siglip.py——
float32 下归一化；常规分支输出 x_norm * (1 + weight)；
cond 分支 dense(cond) 切出 scale/shift/gate，输出 x_norm * (1+scale) + shift。

运行方式（支持任意粒度的选择性运行，不必一次跑完）：
    uv run pytest tests/test_modules.py                     # 只跑本文件
    uv run pytest tests/test_modules.py -k layernorm        # 只跑名字含 layernorm 的
    uv run pytest tests/test_modules.py::test_layernorm_matches_torch   # 只跑单条
"""

import torch

from neat_pi.model.modules import AdaLayerNorm, LayerNorm, RMSNorm


def naive_gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    """手写 Gemma RMSNorm 参考实现，作为对拍真值。"""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(var + eps)
    return (normed * (1 + weight.float())).to(x.dtype)


def naive_adalayernorm(x: torch.Tensor, cond: torch.Tensor,
                       dense_weight: torch.Tensor, dense_bias: torch.Tensor,
                       eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """手写 AdaLayerNorm 参考实现，作为对拍真值。"""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = x.float() * torch.rsqrt(var + eps)
    modulation = (cond @ dense_weight.T + dense_bias).unsqueeze(1).float()
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = normed * (1 + scale) + shift
    return out.to(x.dtype), gate.to(x.dtype)


# ---------------- RMSNorm ----------------


def test_rmsnorm_output_shape() -> None:
    """输出形状与输入一致。"""
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16)
    assert norm(x).shape == x.shape


def test_rmsnorm_matches_naive() -> None:
    """随机输入 + 随机 weight 下与 naive 参考实现逐元素一致。"""
    torch.manual_seed(0)
    norm = RMSNorm(dim=32)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32))
    x = torch.randn(4, 7, 32)
    assert torch.allclose(norm(x), naive_gemma_rmsnorm(x, norm.weight), atol=1e-6)


def test_rmsnorm_zero_weight_is_plain_norm() -> None:
    """零初始化下 (1+weight)=1，输出就是无缩放的归一化结果。"""
    norm = RMSNorm(dim=8)
    assert torch.count_nonzero(norm.weight) == 0
    x = torch.randn(2, 3, 8)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rmsnorm_dtype_preserved() -> None:
    """低精度输入：内部 float32 计算，输出转回输入 dtype。"""
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), naive_gemma_rmsnorm(x, norm.weight).float(),
                          atol=1e-2)


def test_rmsnorm_gradient_flows() -> None:
    """反向传播后 weight 有梯度。"""
    norm = RMSNorm(dim=16)
    norm(torch.randn(2, 4, 16)).sum().backward()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


# ---------------- LayerNorm ----------------


def test_layernorm_output_shape() -> None:
    """输出形状与输入一致，weight/bias 默认 ones/zeros。"""
    norm = LayerNorm(dim=16)
    x = torch.randn(2, 5, 16)
    assert norm(x).shape == x.shape
    assert torch.equal(norm.weight, torch.ones(16))
    assert torch.equal(norm.bias, torch.zeros(16))


def test_layernorm_matches_torch() -> None:
    """与 torch.nn.LayerNorm 逐元素一致（同一 eps）。"""
    torch.manual_seed(0)
    dim, eps = 32, 1e-6
    norm = LayerNorm(dim, eps)
    ref = torch.nn.LayerNorm(dim, eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim))
        norm.bias.copy_(torch.randn(dim))
        ref.weight.copy_(norm.weight)
        ref.bias.copy_(norm.bias)
    x = torch.randn(4, 7, dim)
    assert torch.allclose(norm(x), ref(x), atol=1e-6)


def test_layernorm_normalizes_last_dim() -> None:
    """输出末维均值≈0、方差≈1（默认 weight=1/bias=0 时）。"""
    norm = LayerNorm(dim=16)
    out = norm(torch.randn(8, 3, 16))
    assert torch.allclose(out.mean(dim=-1), torch.zeros(8, 3), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(8, 3), atol=1e-4)


def test_layernorm_dtype_preserved() -> None:
    """低精度输入：内部 float32 计算，输出转回输入 dtype。"""
    norm = LayerNorm(dim=16)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    assert norm(x).dtype == torch.bfloat16


def test_layernorm_gradient_flows() -> None:
    """反向传播后 weight 与 bias 都有梯度。"""
    norm = LayerNorm(dim=16)
    norm(torch.randn(2, 4, 16)).sum().backward()
    assert norm.weight.grad is not None
    assert norm.bias.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


# ---------------- AdaLayerNorm ----------------


def test_adalayernorm_output_shapes() -> None:
    """out 形状同输入；gate 形状 [batch, 1, dim]，供残差处广播。"""
    norm = AdaLayerNorm(dim=16, cond_dim=8)
    x, cond = torch.randn(2, 5, 16), torch.randn(2, 8)
    out, gate = norm(x, cond)
    assert out.shape == x.shape
    assert gate.shape == (2, 1, 16)


def test_adalayernorm_matches_naive() -> None:
    """随机 dense 权重下与 naive 参考实现逐元素一致。"""
    torch.manual_seed(0)
    norm = AdaLayerNorm(dim=32, cond_dim=8)
    with torch.no_grad():
        norm.dense.weight.copy_(torch.randn_like(norm.dense.weight))
        norm.dense.bias.copy_(torch.randn_like(norm.dense.bias))
    x, cond = torch.randn(4, 7, 32), torch.randn(4, 8)
    out, gate = norm(x, cond)
    ref_out, ref_gate = naive_adalayernorm(x, cond, norm.dense.weight, norm.dense.bias)
    assert torch.allclose(out, ref_out, atol=1e-6)
    assert torch.allclose(gate, ref_gate, atol=1e-6)


def test_adalayernorm_zero_init_is_identity() -> None:
    """adaLN-Zero 性质：零初始化下 scale/shift/gate 全零——
    norm 输出退化为纯归一化，gated 残差 x + y*gate 为恒等映射。"""
    norm = AdaLayerNorm(dim=16, cond_dim=8)
    assert torch.count_nonzero(norm.dense.weight) == 0
    assert torch.count_nonzero(norm.dense.bias) == 0
    x, cond = torch.randn(2, 4, 16), torch.randn(2, 8)
    out, gate = norm(x, cond)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.count_nonzero(gate) == 0
    y = torch.randn(2, 4, 16)
    assert torch.equal(x + y * gate, x)


def test_adalayernorm_dtype_preserved() -> None:
    """bf16 输入：内部 float32 计算，out 与 gate 转回输入 dtype。"""
    norm = AdaLayerNorm(dim=16, cond_dim=8)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    cond = torch.randn(2, 8)
    out, gate = norm(x, cond)
    assert out.dtype == torch.bfloat16
    assert gate.dtype == torch.bfloat16


def test_adalayernorm_gradient_flows() -> None:
    """反向传播后 dense 的 weight 与 bias 都有梯度。"""
    norm = AdaLayerNorm(dim=16, cond_dim=8)
    x, cond = torch.randn(2, 4, 16), torch.randn(2, 8)
    out, gate = norm(x, cond)
    (out.sum() + gate.sum()).backward()
    assert norm.dense.weight.grad is not None
    assert norm.dense.bias.grad is not None
    assert torch.isfinite(norm.dense.weight.grad).all()
