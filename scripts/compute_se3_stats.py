"""离线重算 SE(3) 新表示的归一化统计量，存成 lerobot normalizer 的 safetensors。

用法：uv run scripts/compute_se3_stats.py configs/pi05_libero.yaml

直接读数据集 parquet（不解视频），把 8 维 state / 7 维 action 转成
se3.py 定义的新表示（state 11 维、SE(3) delta action chunk×10 维），
逐维统计 count/mean/std/min/max/q01/q10/q50/q90/q99，按键布局
`observation.state.q01` / `action.q99` 等写入
`configs/pi05_libero_preprocessor.json` 的 normalizer `state_file`
（相对 JSON 所在目录解析）。

episode 边界处理与 LeRobotDataset 一致：窗口索引 clip 到 episode 末帧。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from loguru import logger
from safetensors.torch import save_file

from neat_pi.config import load_config
from neat_pi.data.se3 import compute_delta_actions, convert_state


def _episode_windows(
    states: torch.Tensor, actions: torch.Tensor, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """单条 episode 的全帧窗口：(T,11) 新 state 与 (T,H,10) 新 action。"""
    T = states.shape[0]
    idx = torch.arange(T).unsqueeze(1) + torch.arange(horizon + 1)  # (T, H+1)
    idx = idx.clamp_max(T - 1)
    window = states[idx]  # (T, H+1, 8)
    grip_idx = (torch.arange(T).unsqueeze(1) + torch.arange(horizon)).clamp_max(T - 1)
    gripper = actions[grip_idx, 6]  # (T, H)
    return convert_state(states), compute_delta_actions(window, gripper)


def _stats(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """逐维统计（最后一维为特征维），键名对齐 lerobot normalizer。"""
    x = t.reshape(-1, t.shape[-1]).double()
    qs = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=x.dtype)
    q = torch.quantile(x, qs, dim=0)  # (5, D)
    return {
        "count": torch.tensor([x.shape[0]], dtype=torch.float32),
        "mean": x.mean(0).float(),
        "std": x.std(0).float(),
        "min": x.min(0).values.float(),
        "max": x.max(0).values.float(),
        "q01": q[0].float(),
        "q10": q[1].float(),
        "q50": q[2].float(),
        "q90": q[3].float(),
        "q99": q[4].float(),
    }


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/pi05_libero.yaml")
    root = Path(cfg.data.root)
    horizon = cfg.model.action_horizon

    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root}/data 下没有 parquet 文件")
    logger.info(f"共 {len(files)} 个 parquet 文件，action_horizon={horizon}")

    state_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    for f in files:
        table = pq.read_table(f, columns=["observation.state", "action", "episode_index"])
        states = torch.tensor(table["observation.state"].to_pylist(), dtype=torch.float32)
        actions = torch.tensor(table["action"].to_pylist(), dtype=torch.float32)
        ep_ids = table["episode_index"].to_pylist()
        for ep in sorted(set(ep_ids)):
            mask = torch.tensor([e == ep for e in ep_ids])
            s11, a10 = _episode_windows(states[mask], actions[mask], horizon)
            state_rows.append(s11)
            action_rows.append(a10)

    all_state = torch.cat(state_rows)          # (N, 11)
    all_action = torch.cat(action_rows)        # (N, H, 10)
    logger.info(f"state 样本 {all_state.shape[0]} 帧，action 样本 {all_action.shape[0]} 帧窗口")

    flat: dict[str, torch.Tensor] = {}
    for key, stat in (("observation.state", _stats(all_state)),
                      ("action", _stats(all_action))):
        for name, value in stat.items():
            flat[f"{key}.{name}"] = value

    out = Path(cfg.data.preprocessor_path).parent / "pi05_libero_se3_stats.safetensors"
    save_file(flat, str(out))
    logger.info(f"统计量已写入 {out}（{len(flat)} 个键）")

    # sanity 打印：逐维 q01/q99
    for key, dim in (("observation.state", 11), ("action", 10)):
        q01, q99 = flat[f"{key}.q01"], flat[f"{key}.q99"]
        for d in range(dim):
            logger.info(f"  {key}[{d:2d}] q01={q01[d]:+.4f} q99={q99[d]:+.4f}")


if __name__ == "__main__":
    main()
