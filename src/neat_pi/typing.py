"""全项目统一的张量形状别名（TorchTyping）与运行时校验入口。

约定（见 AGENTS.md 编码规范）：
- 所有函数签名用这里的 TensorType 别名标注形状；
- 公共函数加 @typechecked，让 torchtyping 在运行时断言形状；
- TensorType 里同名维度表示两个张量在该维度上必须相等，
  例如 forward(image: ImageBCHW, ...) 与内部输出的 "batch" 会自动对齐检查。

形状别名命名规则：张量内容 + 维度字母，如 ImageBCHW = 图像 + [B, C, H, W]。
"""

import torch
from typeguard import typechecked  # noqa: F401  (re-export，供各模块统一使用)
from torchtyping import TensorType

# 原始图像批次：[batch, channels, height, width]
ImageBCHW = TensorType["batch", "channels", "height", "width"]

# 视觉 token 序列（SigLIP 输出 / 进入 transformer 的通用形态）：[batch, num_tokens, dim]
TokensBTD = TensorType["batch", "num_tokens", "dim"]

# 语言模型输出的 logits：[batch, seq, vocab]
LogitsBSV = TensorType["batch", "seq", "vocab"]

# 文本 token id：[batch, seq_len]，整型
TokenIdsBL = TensorType["batch", "seq_len", torch.long]

# 文本 padding mask：[batch, seq_len]，True 表示有效 token
MaskBL = TensorType["batch", "seq_len", torch.bool]

# 动作 chunk 的时间步 padding mask：[batch, action_horizon]，True 表示有效步
MaskBH = TensorType["batch", "action_horizon", torch.bool]

# 标量损失（0 维张量）
ScalarLoss = TensorType[()]

# 机器人状态向量：[batch, state_dim]
StateBD = TensorType["batch", "state_dim"]

# 动作 chunk（flow matching 的预测目标）：[batch, action_horizon, action_dim]
ActionBHD = TensorType["batch", "action_horizon", "action_dim"]

# flow matching 的连续时间步：[batch]
TimeB = TensorType["batch"]

# adaRMS 的条件向量（时间嵌入 MLP 的输出）：[batch, cond_dim]
CondBD = TensorType["batch", "cond_dim"]
