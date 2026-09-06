"""mot.py 单元测试：fused 前向与分离执行两条路径的数值等价。

MoT 的 fused 逐层前向（训练形态）应与分离执行给出相同的数值结果：
- VLM 段：prefix 只 attend 自己 -> 与把 prefix 单独过 GemmaDecoderLayer
  （双向全注意、不经 MoT）一致；
- 动作段：动作 token attend prefix + 整个动作 chunk -> 与逐层拼上
  GemmaLM.prefill 产出的每层 (k, v) 过 DiTBlock 一致。

用小的测试配置（非真实 pi05 尺寸）快速跑等价性对拍；真实权重加载 +
数值对照由 scripts/ 层的检查脚本负责。
"""

from __future__ import annotations

import torch
import pytest
from torch import nn

from neat_pi.model.action_expert import ActionExpert
from neat_pi.model.gemma import GemmaLM
from neat_pi.model.util import build_rope_cache
from neat_pi.model.mot import MoT, MoTFusedLayer


def _small_vlm() -> GemmaLM:
    """小而完整的 VLM 专家（GemmaLM），与动作专家结构参数一致。"""
    return GemmaLM(vocab_size=64, hidden_dim=64, num_layers=2, num_heads=4,
                   num_kv_heads=1, attn_head_dim=16, mlp_hidden_dim=128)


def _small_expert() -> ActionExpert:
    """小而完整的动作专家，head 结构与 _small_vlm 对齐。"""
    return ActionExpert(action_dim=7, hidden_dim=64, num_layers=2,
                        num_heads=4, num_kv_heads=1, head_dim=16,
                        mlp_hidden_dim=128)


def _pi05_mask(prefix_len: int, action_len: int) -> torch.Tensor:
    """pi05 双段 block mask：prefix 行双向见 prefix；action 行见 prefix + 整个 action chunk。"""
    total = prefix_len + action_len
    mask = torch.zeros(total, total, dtype=torch.bool)
    mask[:prefix_len, :prefix_len] = True
    mask[prefix_len:, :] = True
    return mask


def test_mot_forward_shapes() -> None:
    """两专家序列长度不同时，forward 返回各路同形输出、无 NaN。"""
    mot = MoT({"vlm": _small_vlm(), "action": _small_expert()})
    prefix = torch.randn(2, 8, 64)
    action = torch.randn(2, 5, 64)
    cond = torch.randn(2, 64)
    out = mot({"vlm": prefix, "action": action},
              attention_mask=_pi05_mask(8, 5),
              conds={"vlm": None, "action": cond})
    assert out["vlm"].shape == prefix.shape
    assert out["action"].shape == action.shape
    assert not out["vlm"].isnan().any()
    assert not out["action"].isnan().any()


def test_mot_missing_expert_rejected() -> None:
    """embeds_dict 缺专家时报错。"""
    mot = MoT({"vlm": _small_vlm(), "action": _small_expert()})
    try:
        mot({"vlm": torch.randn(2, 8, 64)})
    except ValueError:
        return
    raise AssertionError("缺少专家 token 应抛 ValueError")


def test_mot_fused_matches_separate_runs() -> None:
    """fused 前向 == prefix 单独过层 + 动作逐层拼 prefix (k, v)。

    推理走的是分离执行（GemmaLM.prefill 一次 + 每步动作段），训练走 MoT
    fused；两条路径必须在双段 mask 下逐元素一致，fused 才可作为唯一实现。
    """
    torch.manual_seed(0)
    vlm, expert = _small_vlm(), _small_expert()
    mot = MoT({"vlm": vlm, "action": expert})
    prefix_len, action_len = 8, 5
    prefix = torch.randn(2, prefix_len, 64)
    action = torch.randn(2, action_len, 64)
    cond = torch.randn(2, 64)
    fused = mot({"vlm": prefix, "action": action},
                attention_mask=_pi05_mask(prefix_len, action_len),
                conds={"vlm": None, "action": cond})

    # 参考 VLM 段：prefix 单独过每层（双向全注意，不经 MoT）
    h = prefix
    cos, sin = build_rope_cache(prefix_len, vlm.attn_head_dim, vlm.theta,
                                prefix.device, prefix.dtype)
    for layer in vlm.layers:
        h = layer(h, cos, sin, is_causal=False)
    assert torch.allclose(fused["vlm"], h, atol=1e-5)

    # 参考动作段：每层动作 token 拼上同层 prefix (k, v) 过 DiTBlock
    # （等价于 ActionExpert.run_layers，但不做最终 adaRMS norm——fused
    # 前向只到 block 层，最终 norm 由调用方在容器外做）
    cache = vlm.prefill(prefix)
    x = action
    cos_a, sin_a = build_rope_cache(prefix_len + action_len,
                                    expert.attn_head_dim, expert.theta,
                                    action.device, action.dtype)
    cos_a, sin_a = cos_a[prefix_len:], sin_a[prefix_len:]
    for i, layer in enumerate(expert.layers):
        x = layer(x, cond, cos_a, sin_a, vlm_kv=cache[i])
    assert torch.allclose(fused["action"], x, atol=1e-5)


def test_mot_takes_ownership_using_fused_layers() -> None:
    """MoT 接管 layers，Expert 只保留只读 view，避免参数双重注册。"""
    vlm = _small_vlm()
    expert = _small_expert()
    mot = MoT({"vlm": vlm, "action": expert})

    assert len(mot.layers) == 2
    assert all(isinstance(layer, MoTFusedLayer) for layer in mot.layers)
    assert mot.layers[0].vlm is vlm.layers[0]
    assert mot.layers[0].action is expert.layers[0]
    assert not isinstance(vlm.layers, nn.ModuleList)
    assert not isinstance(expert.layers, nn.ModuleList)
    assert "layers" not in vlm._modules
    assert "layers" not in expert._modules
    with pytest.raises(RuntimeError, match="不拥有 layers"):
        vlm.take_owned_layers()


def test_mot_parameter_paths_are_unique_after_fused_ownership() -> None:
    """layer 参数只出现在 MoTFusedLayer 路径下。"""
    mot = MoT({"vlm": _small_vlm(), "action": _small_expert()})
    names = [name for name, _ in mot.named_parameters()]

    assert len(names) == len(set(names))
    assert all(name.startswith("layers.") for name in names)
    assert all(not name.startswith("mixtures.") for name in names)
