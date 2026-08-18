"""SE(3) / rot6D 数学工具与 LIBERO state/action 表示转换。

LIBERO 数据集约定（已实证，见 docs/plan/01）：
- `observation.state` 8 维 = eef pos(3) + axis-angle(3) + gripper qpos(2)；
- `action` 7 维 = OSC_POSE 控制器的增量指令（带增益，非实际位移）+ 二值 gripper。

因此 delta 动作不由录制 action 累乘，而由未来 state 的绝对末端位姿经 SE(3)
矩阵相乘得到：`ΔT_k = T_0^{-1} @ T_k`，平移天然落在当前帧末端坐标系下。

本模块是 processor（se3_processor.py）与离线统计脚本
（scripts/compute_se3_stats.py）共用的唯一实现，避免两处不一致。
"""

from __future__ import annotations

import torch
from torchtyping import TensorType

from neat_pi.typing import typechecked

# 批量 axis-angle / 旋转矩阵 / 三维向量（前导维任意）
Vec3 = TensorType[..., 3]
RotMat = TensorType[..., 3, 3]
Rot6D = TensorType[..., 6]
# LIBERO 原始 state 窗口：[batch, window, 8]（pos3 + aa3 + gripper2）
StateWindow = TensorType["batch", "window", 8]
# 新 action 表示：[batch, action_horizon, 10]（xyz + rot6D + gripper）
DeltaAction = TensorType["batch", "action_horizon", 10]
# 新 state 表示：[batch, 11]（pos + rot6D + gripper qpos）
StateVec11 = TensorType["batch", 11]


@typechecked
def aa_to_rot(aa: Vec3) -> RotMat:
    """axis-angle 转旋转矩阵（Rodrigues 公式，批量，零向量安全）。"""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (..., 1)
    k = aa / theta.clamp_min(1e-12)
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1
    ).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand_as(K)
    sin = torch.sin(theta).unsqueeze(-1)  # (..., 1, 1)
    cos = torch.cos(theta).unsqueeze(-1)
    R = eye + sin * K + (1.0 - cos) * (K @ K)
    # theta≈0 时退化为单位阵（避免 0/0 方向向量的数值噪声）
    return torch.where((theta < 1e-8).unsqueeze(-1), eye, R)


@typechecked
def rot_to_6d(R: RotMat) -> Rot6D:
    """旋转矩阵转 Zhou 6D 表示（取前两列拼接）。"""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


@typechecked
def convert_state(state: TensorType["batch", 8]) -> StateVec11:
    """LIBERO 原始 8 维 state 转 11 维表示 [pos(3), rot6D(6), gripper qpos(2)]。"""
    return torch.cat(
        [state[..., 0:3], rot_to_6d(aa_to_rot(state[..., 3:6])), state[..., 6:8]],
        dim=-1,
    )


@typechecked
def compute_delta_actions(
    states: StateWindow,
    gripper_cmds: TensorType["batch", "action_horizon"],
) -> DeltaAction:
    """由 state 窗口计算 SE(3) delta 动作序列。

    `states` 为当前帧 + 未来 H 帧的原始 state（window = H+1）；
    `gripper_cmds` 为原始 action chunk 的 gripper 指令（±1）。
    第 k 步动作为 `ΔT_k = T_0^{-1} @ T_{k+1}`：
    [0:3] 平移（当前帧末端坐标系下）、[3:9] rot6D(ΔR)、[9] gripper 指令。
    """
    p0, aa0 = states[:, 0, 0:3], states[:, 0, 3:6]
    pk, aak = states[:, 1:, 0:3], states[:, 1:, 3:6]
    R0_t = aa_to_rot(aa0).transpose(-1, -2)  # (B, 3, 3)，T_0 旋转的逆
    dR = R0_t.unsqueeze(1) @ aa_to_rot(aak)  # (B, H, 3, 3)
    dt = (R0_t.unsqueeze(1) @ (pk - p0.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    return torch.cat([dt, rot_to_6d(dR), gripper_cmds.unsqueeze(-1)], dim=-1)


# ---------------------------------------------------------------------------
# 逆变换（推理/postprocessor 侧）
# ---------------------------------------------------------------------------


@typechecked
def rot6d_to_rot(x: Rot6D) -> RotMat:
    """Zhou 6D 表示转回旋转矩阵（Gram-Schmidt 正交化）。"""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # 列为 [b1 b2 b3]


@typechecked
def rot_to_aa(R: RotMat) -> Vec3:
    """旋转矩阵转 axis-angle（批量；θ≈0 用一阶近似，θ≈π 用对角线法取轴）。"""
    skew = torch.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        dim=-1,
    )
    sin = torch.linalg.norm(skew, dim=-1) / 2.0
    cos = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    theta = torch.atan2(sin, cos.clamp(-1.0, 1.0))  # (...,)

    axis_normal = skew / (2.0 * sin.clamp_min(1e-12)).unsqueeze(-1)
    # θ≈π 时 skew≈0 取不出轴，改从 (R+I)/2 的对角线开方取轴
    # （θ=π 时 ±轴等价，故符号模糊不影响 axis-angle 表示的正确性）
    axis_pi = ((R.diagonal(dim1=-2, dim2=-1) + 1.0) / 2.0).clamp_min(0.0).sqrt()

    small = theta < 1e-6
    near_pi = (torch.pi - theta) < 1e-3
    axis = torch.where(near_pi.unsqueeze(-1), axis_pi, axis_normal)
    aa = axis * theta.unsqueeze(-1)
    return torch.where(small.unsqueeze(-1), 0.5 * skew, aa)


@typechecked
def delta_action_to_command(delta: DeltaAction) -> TensorType["batch", "action_horizon", 7]:
    """10 维 SE(3) delta 动作转 7 维指令 [dpos(3), axis-angle(3), gripper(1)]。

    平移与旋转均在**当前帧末端坐标系**下（与 compute_delta_actions 的定义互逆，
    只是旋转从 rot6D 换回 axis-angle）。gripper 维原样透传。
    """
    aa = rot_to_aa(rot6d_to_rot(delta[..., 3:9]))
    return torch.cat([delta[..., 0:3], aa, delta[..., 9:10]], dim=-1)


@typechecked
def ee_command_to_base(
    state: TensorType["batch", 8],
    command: TensorType["batch", "action_horizon", 7],
) -> TensorType["batch", "action_horizon", 7]:
    """把当前帧 EE 坐标系下的指令旋转到 base 坐标系（供 OSC_POSE 类控制器下发）。

    dpos_base = R_0 @ dpos_ee；dR_base = R_0 @ dR_ee @ R_0^T（共轭变换）。
    """
    R0 = aa_to_rot(state[:, 3:6])
    dpos = (R0.unsqueeze(1) @ command[..., 0:3].unsqueeze(-1)).squeeze(-1)
    dR_ee = aa_to_rot(command[..., 3:6])
    dR_base = R0.unsqueeze(1) @ dR_ee @ R0.transpose(-1, -2).unsqueeze(1)
    return torch.cat([dpos, rot_to_aa(dR_base), command[..., 6:7]], dim=-1)
