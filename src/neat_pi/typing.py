"""全项目统一的张量形状别名（TorchTyping）与运行时校验入口。

约定（见 AGENTS.md 编码规范）：
- 所有函数签名用这里的 TensorType 别名标注形状；
- 公共函数加 @typechecked，让 torchtyping 在运行时断言形状；
- TensorType 里同名维度表示两个张量在该维度上必须相等，
  例如 forward(image: ImageBCHW, ...) 与内部输出的 "batch" 会自动对齐检查。

形状别名命名规则：张量内容 + 维度字母，如 ImageBCHW = 图像 + [B, C, H, W]。
"""

import torch
import torchtyping
from typeguard import typechecked  # noqa: F401  (re-export，供各模块统一使用)
from torchtyping import TensorType

# 让 typeguard 理解 torchtyping 的 TensorType（底层是 Annotated[torch.Tensor, ...]），
# 之后 @typechecked 才会真正断言 rank / dtype / 跨参数同名维相等。
# 本模块是各模块 typechecked 的统一来源，首次 import 即全局生效。
torchtyping.patch_typeguard()

# 原始图像批次：[batch, channels, height, width]
ImageBCHW = TensorType["batch", "channels", "height", "width"]

# LIBERO 原始状态批次：[batch, 8]，按旧 checkpoint 的
# [eef pos(3), axis-angle(3), gripper qpos(2)] 排列。
StateB8 = TensorType["batch", 8]

# LIBERO 四元数与 axis-angle：[batch, 4] / [batch, 3]
QuatB4 = TensorType["batch", 4]
AxisAngleB3 = TensorType["batch", 3]

# 相机可用性 mask：[batch]，True 表示该相机图像有效。缺失相机保留固定
# token 槽位时，其所有视觉 token 的 prefix mask 由此 mask 展开为 False。
MaskB = TensorType["batch", torch.bool]

# 通用 3D token 序列：[batch, num_tokens, dim]
# 只用于跨专家共享的积木（RMSNorm / MLP / AdaLayerNorm 等，VLM、动作专家、
# SigLIP 都会用）；语义确定的场景用下面的 VisionTokens / LanguageTokens。
TokensBTD = TensorType["batch", "num_tokens", "dim"]

# AdaLayerNorm 的 gate：[batch, 1, dim]，第 2 维固定为字面量 1，
# 供残差处广播相乘（x = x + y * gate）。不能复用 TokensBTD——
# gate 的 1 与 x 的 num_tokens（任意 >1）不是同一维，混用会在
# @typechecked 下触发跨参数维不一致。
GateB1D = TensorType["batch", 1, "dim"]

# 视觉 token 序列（SigLIP 输出）：[batch, num_vision_tokens, vision_dim]
# pi05 为 256 个 1152 维 token（224/14=16 的 grid）
VisionTokensBTD = TensorType["batch", "num_vision_tokens", "vision_dim"]

# 语言 token 序列（文本 embedding / VLM 主干隐状态）：
# [batch, num_language_tokens, language_dim]，pi05 的 VLM 宽度为 2048
# （视觉 token 经 multi_modal_projector 投影后也是这个形态）
LanguageTokensBTD = TensorType["batch", "num_language_tokens", "language_dim"]

# 语言模型输出的 logits：[batch, seq, vocab]
LogitsBSV = TensorType["batch", "seq", "vocab"]

# 文本 token id：[batch, seq_len]，整型
TokenIdsBL = TensorType["batch", "seq_len", torch.long]

# 文本 padding mask：[batch, seq_len]，True 表示有效 token
MaskBL = TensorType["batch", "seq_len", torch.bool]

# VLM prefix 的 padding mask：[batch, prefix_len]，图像 + 语言 token 拼接后
# 的序列长度与任一输入的 seq_len 都不同，须使用独立维度名。
MaskBT = TensorType["batch", "prefix_len", torch.bool]

# 动作 chunk 的时间步 padding mask：[batch, action_horizon]，True 表示有效步
MaskBH = TensorType["batch", "action_horizon", torch.bool]

# 逻辑层 attention mask（语义层，模型 forward 的入参形态）：[batch, seq_len]，
# bool，True 表示该位置有效、可参与 attention。
# 与 MaskBL 形状相同、取值等价——MaskBL 从 padding 视角（某位置是否真实输入）
# 命名，本别名从 attention 视角命名，用于对齐 lerobot 的
# observation.language.attention_mask / openpi 的 input_mask 语义；两者并存，
# 别名的选择只表达调用处的语义意图。模型内部需展开成 AttentionMaskBHLS 形态
# 再交给 SDPA（见下）。
AttentionMaskBL = TensorType["batch", "seq_len", torch.bool]

# SDPA 层内 mask（F.scaled_dot_product_attention 的 attn_mask 入参展开形态）：
# [batch, num_heads, seq, kv_len]。维度字母沿用 SDPA 文档的 (N, ..., L, S)
# 记号：seq 为 query 行数、kv_len 为 key/value 列数，两者可以不等
# （动作专家 attend 的列 = prefix 前缀 + 自身，行只有动作段）。
# num_heads 取名（不固定字面量）是因为 SDPA 接受任意头数的掩码，调用方
# 通常构造 1 头形态借广播作用于各头。不写 dtype：bool（True=可见）与
# float（加性 bias）都是 SDPA 的合法掩码。
AttentionMaskBHLS = TensorType["batch", "num_heads", "seq", "kv_len"]

# 标量损失（0 维张量）
ScalarLoss = TensorType[()]

# 时间步向量：[batch, action_hidden_dim]
TimeBD = TensorType["batch", "action_hidden_dim"]

# Action在 Action Expert中参与运算的维度
ActionTokensBHD = TensorType["batch", "action_horizon", "action_hidden_dim"] 

# 动作 chunk（flow matching 的预测目标）：[batch, action_horizon, action_dim]
ActionBHD = TensorType["batch", "action_horizon", "action_dim"]

# flow matching 的连续时间步：[batch]
TimeB = TensorType["batch"]

# adaRMS 的条件向量（时间嵌入 MLP 的输出）：[batch, cond_dim]
CondBD = TensorType["batch", "cond_dim"]
