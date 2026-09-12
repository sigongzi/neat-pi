"""EMA（指数滑动平均）影子权重。

语义以 ref/openpi 为准：影子以当前训练参数初始化（无 bias correction），
每 optimizer step 后 `ema = decay * ema + (1 - decay) * new`。

影子模型在 FSDP wrap **前** deepcopy 构造（拿到与主模型相同的初始权重），
随后与主模型分别独立 wrap——同构树使 auto-wrap 决策一致、分片布局相同，
更新是 rank 本地逐元素操作，无集合通信。配对按参数遍历顺序：构造期校验
两树参数名一致后，wrap 不改变遍历顺序（FSDP1 use_orig_params 只原位替换
Parameter 的数据视图，对象身份不变），跨 wrap 配对仍成立。
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EmaTracker:
    """EMA 影子权重跟踪器：持有影子模型，按 step 向训练参数滑动。

    属性：
        model: 影子模型（构造时是未 wrap 的 deepcopy；装配层随后对其
            独立 FSDP wrap，wrap 后属性指向 FSDP 根）。
        decay: 滑动衰减系数，取值 (0, 1)。
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        """构造影子并校验参数名对齐；必须在 FSDP wrap 前调用。"""
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay 必须在 (0, 1) 内，实际为 {decay!r}")
        self.decay = decay
        self.model = copy.deepcopy(model)
        self._source_model = model

        source_names = [name for name, _ in model.named_parameters()]
        ema_names = [name for name, _ in self.model.named_parameters()]
        if source_names != ema_names:
            raise ValueError(
                "EMA 影子模型与主模型参数名不一致（结构不同构），"
                f"首个差异: {next((a, b) for a, b in zip(source_names, ema_names) if a != b)}")

    def update(self) -> None:
        """每 optimizer step 调一次：影子参数向当前训练参数指数滑动。"""
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(),
                                        self._source_model.parameters()):
                ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def param_pairs(self) -> list[tuple[str, nn.Parameter, nn.Parameter]]:
        """返回 (影子参数名, 影子参数, 主模型参数) 配对，供装配期核对。"""
        return [(ema_name, ema_param, param)
                for (ema_name, ema_param), (_, param)
                in zip(self.model.named_parameters(),
                       self._source_model.named_parameters())]
