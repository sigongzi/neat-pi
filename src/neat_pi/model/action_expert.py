"""动作专家（pi05 的 flow-matching action expert）。

pi05 的动作生成用 flow matching：给定带噪动作 x_t 与连续时间 t ∈ [0,1]，
动作专家预测速度场 v(x_t, t)，训练目标是与 (x_1 - x_0) 的 MSE。

动作专家的输入 token = [state token, 带噪动作 token 序列, 时间条件]，
其 transformer 堆叠与 VLM 共享 attention（见 mot.py），但 qkv/o/MLP
投影是动作专家私有参数——这正是 MoT 的组织方式，也是 FSDP 的切分单位。

结构对齐 ref/openpi/src/openpi/models/pi0.py 中 action expert 部分。
"""

from __future__ import annotations

from torch import Tensor, nn

from neat_pi.typing import ActionBHD, StateBD, TimeB, TokensBTD, typechecked


class ActionExpert(nn.Module):
    """flow-matching 动作专家：状态/动作/时间 -> 私有输入投影，输出速度场。

    TODO: 按 pi05 补全：
    - state_proj / action_in_proj / action_out_proj / time MLP 的形状与初始化；
    - 与 mot.py MoTLayer 的接口（动作专家 token 如何进入共享 attention）。
    """

    def __init__(self, state_dim: int = 32, action_dim: int = 32,
                 width: int = 1024) -> None:
        super().__init__()
        self.state_proj = nn.Linear(state_dim, width, bias=False)
        self.action_in_proj = nn.Linear(action_dim, width, bias=False)
        self.action_out_proj = nn.Linear(width, action_dim, bias=False)
        # TODO: 时间嵌入 MLP（sinusoidal + 两层 MLP），尺寸以 openpi 为准

    @typechecked
    def encode_tokens(self, state: StateBD, noisy_action: ActionBHD,
                      t: TimeB) -> TokensBTD:
        """把状态 + 带噪动作 + 时间编成动作专家侧的输入 token 序列。"""
        raise NotImplementedError("待按 ref/openpi pi0.py 实现")

    @typechecked
    def decode_velocity(self, tokens: TokensBTD) -> ActionBHD:
        """共享 transformer 输出 -> 速度场预测（取动作段 token 过 action_out_proj）。"""
        raise NotImplementedError("待按 ref/openpi pi0.py 实现")
