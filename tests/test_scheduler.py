"""cosine 学习率调度的数值边界测试。"""

from __future__ import annotations

import pytest

from neat_pi.training.scheduler import cosine_lr_at

# openpi CosineDecaySchedule 默认口径
PEAK_LR = 2.5e-5
LR_END = 2.5e-6
WARMUP = 1000
TOTAL = 10000


def test_warmup_starts_at_init_and_ramps_to_peak() -> None:
    """step 0 从 peak/(warmup+1) 起步，warmup 结束时到 peak。"""
    assert cosine_lr_at(0, peak_lr=PEAK_LR, lr_end=LR_END,
                        warmup_steps=WARMUP, total_steps=TOTAL) \
        == pytest.approx(PEAK_LR / (WARMUP + 1))
    mid = cosine_lr_at(WARMUP // 2, peak_lr=PEAK_LR, lr_end=LR_END,
                       warmup_steps=WARMUP, total_steps=TOTAL)
    expected_mid = PEAK_LR / (WARMUP + 1) + (PEAK_LR - PEAK_LR / (WARMUP + 1)) * 0.5
    assert mid == pytest.approx(expected_mid)
    assert cosine_lr_at(WARMUP, peak_lr=PEAK_LR, lr_end=LR_END,
                        warmup_steps=WARMUP, total_steps=TOTAL) == pytest.approx(PEAK_LR)


def test_cosine_decays_to_lr_end_at_total() -> None:
    """衰减段单调下降，训完（total）恰好落到 lr_end。"""
    first = cosine_lr_at(WARMUP, peak_lr=PEAK_LR, lr_end=LR_END,
                         warmup_steps=WARMUP, total_steps=TOTAL)
    for step in range(WARMUP + 1, TOTAL, 500):
        value = cosine_lr_at(step, peak_lr=PEAK_LR, lr_end=LR_END,
                             warmup_steps=WARMUP, total_steps=TOTAL)
        assert LR_END < value < first
    assert cosine_lr_at(TOTAL, peak_lr=PEAK_LR, lr_end=LR_END,
                        warmup_steps=WARMUP, total_steps=TOTAL) == pytest.approx(LR_END)


def test_holds_at_lr_end_beyond_total() -> None:
    """超过 total_steps 后保持 lr_end（断点续训越过原终点时的行为）。"""
    for step in (TOTAL + 1, TOTAL + 100):
        assert cosine_lr_at(step, peak_lr=PEAK_LR, lr_end=LR_END,
                            warmup_steps=WARMUP, total_steps=TOTAL) \
            == pytest.approx(LR_END)


def test_zero_warmup_starts_at_peak() -> None:
    """warmup_steps=0 时从 peak_lr 直接开始 cosine 衰减。"""
    assert cosine_lr_at(0, peak_lr=PEAK_LR, lr_end=LR_END,
                        warmup_steps=0, total_steps=TOTAL) == pytest.approx(PEAK_LR)
