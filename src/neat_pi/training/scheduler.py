"""cosine 学习率调度（含线性 warmup）。

以 ref/openpi 的 CosineDecaySchedule 为准：warmup 从 peak_lr/(warmup+1)
线性升到 peak_lr，随后 cosine 衰减到 lr_end 并保持。与 openpi 的差异：
衰减终点绑 training.max_steps（训完恰好衰到 lr_end），不单独设
decay_steps。

函数是纯 step -> lr 映射、无状态：续训时直接从 optimizer_step 重算，
无需在 checkpoint 里保存调度器状态。
"""

from __future__ import annotations

import math


def cosine_lr_at(step: int, *, peak_lr: float, lr_end: float,
                 warmup_steps: int, total_steps: int) -> float:
    """返回第 step 步（0 起）的学习率。

    - step < warmup_steps：从 peak_lr/(warmup_steps+1) 线性升到 peak_lr；
    - warmup_steps <= step < total_steps：cosine 从 peak_lr 衰到 lr_end；
    - step >= total_steps：保持 lr_end（warmup_steps >= total_steps 时
      退化为恒定 peak_lr，依赖配置校验避免，函数内不另设防御）。
    """
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return lr_end + (peak_lr - lr_end) * 0.5 * (1.0 + math.cos(math.pi * progress))
