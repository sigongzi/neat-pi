"""真实 Pi05 梯度回传的精确断言（计划 07 Step 1）。

tests/test_pi05.py 只断言关键参数 grad 非 None；本文件给出"梯度回传正常"
的完整证据（docs/plan/07-local-grad-flow-smoke.md）：

1. backward 覆盖：除训练路径不消费的 ``language_model.norm`` 外，所有
   参数都有梯度且有限；
2. 每个顶层功能模块都有非零梯度——尤其是视觉塔 / 语言 embedding 段：
   pi05 没有 LM loss，它们的梯度只能经 MoT 共享 attention 的 K/V 支路
   从 action 查询流回，这是最容易静默断掉的链路；
3. 梯度拓扑不变量：pi05 丢弃 vlm 输出——末层 vlm 的 o_proj / post norm
   / MLP 是 autograd 不访问的死支路（grad 为 None），末层 vlm 的 q_proj
   梯度为精确零张量；非末层 vlm 的输出流经下一层 K/V 被消费，梯度微小
   但非零。language_model.norm 在训练路径外（grad 为 None）。若未来给
   vlm 输出接附加损失，这里会失败并提示更新断言；
4. 忠实初始化（adaLN-Zero）下多步训练：gate 被第一步更新解锁后，
   梯度到达视觉 / VLM / action 各支路，loss 下降、参数实际更新。

adaLN-Zero 说明：AdaLayerNorm 的 dense 零初始化使 t=0 时刻 gate=0，
attention/MLP 支路第一步梯度结构性为零（恒等残差）。测试 1 先把 dense
权重扰动成非零以解锁全部支路，验证 wiring 本身；测试 2 保持忠实初始化，
验证训练动态能自然解锁。
"""

from __future__ import annotations

import torch
from torch import nn

from neat_pi.config import ModelConfig
from neat_pi.model.modules import AdaLayerNorm
from neat_pi.model.pi05 import Pi05


def _grad_flow_config() -> ModelConfig:
    """构造 2 层 tiny Pi05 配置：小到 CPU 秒级反向，仍覆盖跨层 wiring。"""
    return ModelConfig(
        image_size=16, num_cameras=2, action_horizon=3, action_dim=7,
        vision_hidden_dim=16, vision_patch_size=4,
        vision_num_layers=2, vision_num_heads=4, vision_mlp_hidden_dim=32,
        vocab_size=64, vlm_hidden_dim=24, vlm_num_layers=2,
        vlm_num_heads=4, vlm_num_kv_heads=1, vlm_attn_head_dim=8,
        vlm_mlp_hidden_dim=48, expert_hidden_dim=16,
        expert_num_layers=2, expert_num_heads=4, expert_num_kv_heads=1,
        expert_attn_head_dim=8, expert_mlp_hidden_dim=32,
    )


def _synthetic_batch(generator: torch.Generator) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, int]:
    """构造一个含两类 padding 的合成 batch。

    返回 (images, image_masks, token_ids, lang_mask, actions, is_pad,
    real_action_dim)：语言段带右 padding，动作 chunk 尾部是 padding 帧，
    动作维 7 只取前 5 维参与损失——三类 mask 都要实际进入梯度裁剪。
    """
    images = [torch.randn(2, 3, 16, 16, generator=generator) for _ in range(2)]
    image_masks = [torch.ones(2, dtype=torch.bool) for _ in images]
    token_ids = torch.randint(0, 64, (2, 6), generator=generator)
    lang_mask = torch.tensor([[True] * 4 + [False] * 2,
                              [True] * 6], dtype=torch.bool)
    actions = torch.randn(2, 3, 7, generator=generator)
    is_pad = torch.tensor([[False, False, True],
                           [False, True, True]])
    return images, image_masks, token_ids, lang_mask, actions, is_pad, 5


def _perturb_adaln_dense(model: Pi05) -> None:
    """把所有 AdaLayerNorm 的 dense 权重扰动成非零，解锁 adaLN-Zero 支路。

    零初始化下 gate=0，attention/MLP 支路第一步梯度结构性为零；wiring
    验证需要先把它们抬离零点（bias 保持零，scale/shift 仍从恒等出发）。
    """
    for module in model.modules():
        if isinstance(module, AdaLayerNorm):
            nn.init.normal_(module.dense.weight, std=0.02)


def _dead_grad_prefixes(num_layers: int) -> tuple[str, ...]:
    """返回训练前向中不被损失消费、grad 恒为 None 的参数前缀。

    autograd 不进入死支路：pi05 丢弃 vlm 输出，因此 vlm 末层的
    o_proj / post norm / MLP 整条子树不会被 backward 访问；
    language_model.norm 只在独立 LM forward 中消费。
    """
    last = num_layers - 1
    return (
        "language_model.norm.",
        f"mot.layers.{last}.vlm.self_attn.o_proj.",
        f"mot.layers.{last}.vlm.post_attention_layernorm.",
        f"mot.layers.{last}.vlm.mlp.",
    )


def _param_groups(model: Pi05) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """按顶层功能模块给全部命名参数分组；未知前缀直接报错。"""
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "vision_tower": [],
        "multi_modal_projector": [],
        "language_model": [],
        "action_expert": [],
        "mot.vlm": [],
        "mot.action": [],
    }
    for name, param in model.named_parameters():
        if name.startswith("vision_tower."):
            groups["vision_tower"].append((name, param))
        elif name.startswith("multi_modal_projector."):
            groups["multi_modal_projector"].append((name, param))
        elif name.startswith("language_model."):
            groups["language_model"].append((name, param))
        elif name.startswith("mot.layers.") and ".vlm." in name:
            groups["mot.vlm"].append((name, param))
        elif name.startswith("mot.layers.") and ".action." in name:
            groups["mot.action"].append((name, param))
        elif name.startswith("action_expert."):
            groups["action_expert"].append((name, param))
        else:
            raise AssertionError(f"参数不属于任何已知分组: {name}")
    return groups


def _assert_nonzero_grad(name: str, param: nn.Parameter) -> None:
    """断言单个参数的梯度存在、有限且范数非零。"""
    assert param.grad is not None, f"{name} 没有梯度"
    assert torch.isfinite(param.grad).all(), f"{name} 梯度含 NaN/Inf"
    assert param.grad.norm() > 0, f"{name} 梯度全零"


def _assert_zero_grad(name: str, param: nn.Parameter) -> None:
    """断言单个参数的梯度精确为零（结构性死支路）。"""
    assert param.grad is not None, f"{name} 没有梯度"
    assert param.grad.abs().sum().item() == 0.0, f"{name} 梯度应精确为零"


def test_backward_reaches_every_module_and_shared_attention() -> None:
    """扰动 adaLN 初始化后，一次 backward 应覆盖全部模块与共享 attention 链路。

    重点：视觉塔 / 语言 embedding / vlm 的 K/V 都没有直接损失，梯度只能
    经 MoT 共享 attention 从 action 查询流回；任何一处 wiring 断裂都会
    表现为对应组全零梯度。
    """
    generator = torch.Generator(device="cpu").manual_seed(7)
    model = Pi05(_grad_flow_config())
    _perturb_adaln_dense(model)
    images, image_masks, token_ids, lang_mask, actions, is_pad, real_dim = (
        _synthetic_batch(generator))

    loss = model(images, image_masks, token_ids, lang_mask, actions, is_pad,
                 real_dim)
    loss.backward()

    assert torch.isfinite(loss)

    # 死支路参数 grad 为 None（autograd 不进入）；其余参数都必须有梯度且有限。
    dead_prefixes = _dead_grad_prefixes(_grad_flow_config().vlm_num_layers)
    for group in _param_groups(model).values():
        for name, param in group:
            if name.startswith(dead_prefixes):
                assert param.grad is None, f"{name} 属于死支路，grad 应为 None"
                continue
            assert param.grad is not None, f"{name} 没有梯度"
            assert torch.isfinite(param.grad).all(), f"{name} 梯度含 NaN/Inf"

    # 每个顶层功能模块至少有一个非零梯度（死支路参数不参与判定）。
    for group_name, group in _param_groups(model).items():
        assert any(param.grad is not None and param.grad.norm() > 0
                   for name, param in group
                   if not name.startswith(dead_prefixes)), (
            f"{group_name} 整组梯度全零：共享 attention 或 embedding 链路断裂")

    # 跨专家 wiring 的关键参数逐个检查非零。
    critical = [
        model.vision_tower.embeddings.patch_embedding.weight,
        model.vision_tower.embeddings.position_embedding.weight,
        model.multi_modal_projector.linear.weight,
        model.language_model.lm_head.weight,
        model.action_expert.action_in_proj.weight,
        model.action_expert.action_out_proj.weight,
        model.action_expert.time_mlp_in.weight,
        model.action_expert.time_mlp_out.weight,
        model.action_expert.norm.dense.weight,
    ]
    for layer_idx in range(_grad_flow_config().vlm_num_layers):
        vlm = model.mot.layers[layer_idx].vlm
        action = model.mot.layers[layer_idx].action
        critical += [
            vlm.self_attn.k_proj.weight,
            vlm.self_attn.v_proj.weight,
            vlm.input_layernorm.weight,
            action.self_attn.q_proj.weight,
            action.self_attn.k_proj.weight,
            action.self_attn.v_proj.weight,
            action.self_attn.o_proj.weight,
            action.mlp.down_proj.weight,
            action.input_layernorm.dense.weight,
            action.post_attention_layernorm.dense.weight,
        ]
    for index, param in enumerate(critical):
        _assert_nonzero_grad(f"critical[{index}]", param)

    # real_action_dim=5：损失把第 5/6 维 mask 掉，输出投影对应行必须零梯度。
    out_weight = model.action_expert.action_out_proj.weight
    assert out_weight.grad[5:].abs().sum().item() == 0.0
    assert out_weight.grad[:5].norm() > 0
    assert model.action_expert.action_out_proj.bias.grad[5:].abs().sum().item() == 0.0

    # 结构性零梯度：末层 vlm 的查询行只服务被丢弃的 vlm 输出，其 attn_out
    # 全部进入死支路；它与 action K/V 同在最后一次共享 SDPA 里，backward
    # 会为整个 q 张量计算梯度，因此这里是精确零张量（区别于死支路的 None）。
    # 非末层 vlm 的 q 经"本层输出流 -> 下一层 K/V"拿到微小但非零的梯度，
    # 属于活路径，不在此断言。
    _assert_zero_grad("last vlm.q_proj",
                      model.mot.layers[-1].vlm.self_attn.q_proj.weight)


def test_training_unlocks_adaln_and_reduces_loss() -> None:
    """忠实 adaLN-Zero 初始化下多步训练：gate 解锁、梯度到达全链路、loss 下降。

    每步前重置随机源，使 (t, noise) 目标固定，loss 下降可断言。第一步
    更新只动 AdaLayerNorm dense / 输出头（其余支路零梯度），后续步骤
    解锁 attention/MLP——末步 backward 时视觉 / VLM / action 支路必须
    已经拿到非零梯度。
    """
    generator = torch.Generator(device="cpu").manual_seed(7)
    model = Pi05(_grad_flow_config())
    images, image_masks, token_ids, lang_mask, actions, is_pad, real_dim = (
        _synthetic_batch(generator))

    watched: list[tuple[str, nn.Parameter, torch.Tensor]] = [
        ("vision.patch_embedding",
         model.vision_tower.embeddings.patch_embedding.weight,
         model.vision_tower.embeddings.patch_embedding.weight.detach().clone()),
        ("vlm.k_proj",
         model.mot.layers[0].vlm.self_attn.k_proj.weight,
         model.mot.layers[0].vlm.self_attn.k_proj.weight.detach().clone()),
        ("action.q_proj",
         model.mot.layers[0].action.self_attn.q_proj.weight,
         model.mot.layers[0].action.self_attn.q_proj.weight.detach().clone()),
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses: list[float] = []
    num_steps = 8
    for step in range(num_steps):
        # 固定 (t, noise)：flow_matching.forward 内部用全局随机源采样。
        torch.manual_seed(2024)
        loss = model(images, image_masks, token_ids, lang_mask, actions,
                     is_pad, real_dim)
        loss.backward()
        losses.append(loss.item())
        assert torch.isfinite(loss), f"step {step} loss 非有限: {loss.item()}"
        if step == num_steps - 1:
            # gate 已被前几步更新解锁，共享 attention 链路应全部带梯度。
            _assert_nonzero_grad("vision.patch_embedding",
                                 model.vision_tower.embeddings.patch_embedding.weight)
            _assert_nonzero_grad("vlm.k_proj",
                                 model.mot.layers[0].vlm.self_attn.k_proj.weight)
            _assert_nonzero_grad("action.q_proj",
                                 model.mot.layers[0].action.self_attn.q_proj.weight)
            _assert_nonzero_grad("vlm gate dense",
                                 model.mot.layers[0].action.input_layernorm.dense.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert losses[-1] < losses[0], (
        f"loss 未下降: 首步 {losses[0]:.4f} -> 末步 {losses[-1]:.4f}")
    for name, param, before in watched:
        assert not torch.equal(before, param.detach()), f"{name} 未被优化器更新"
