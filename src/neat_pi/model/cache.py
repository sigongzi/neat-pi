"""推理用的 prefix KV cache（静态只读容器）。

语义对齐 lerobot pi05（ref/lerobot modeling_pi05.py 的 sample_actions /
denoise_step）：VLM prefix（图像+文本）的 k/v 在 prefill 时算一次，N 步
flow matching 去噪全部复用；动作 token attend prefix 用的就是 VLM 投影出
的 k/v，与训练时的 joint 全量前向严格等价。

为什么比 lerobot/HF 的 DynamicCache 简单：pi05 的 prefix 跨去噪步完全静态，
suffix 的 k/v 从不入 cache——append / sliding window / clone 三样机制全是
动态 cache 的需求，静态缓存一个都不需要。本容器只读：prefill 产一次、读 N 次。

元素约定（与 DiTBlock.forward 的 vlm_kv 入参逐字对应）：
- 每层一对 (k, v)，各为 [batch, num_kv_heads, prefix_len, head_dim]，
  已施加 RoPE，kv 头保持未广播的原形态（广播在 DiTBlock 内做，省显存）；
- 层序与 transformer 堆叠一致（第 i 层取 cache[i]）。
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor


class PrefixKVCache:
    """每层一对 (k, v) 的静态只读容器（不是 nn.Module，不参与参数分片）。

    - cache[i] 返回第 i 层的 (k, v)；len(cache) 为层数；
    - to(device, dtype) 返回搬移后的新容器（本容器不变，保持只读语义）。
    """

    def __init__(self, layer_kvs: list[tuple[Tensor, Tensor]]) -> None:
        """layer_kvs 按层序排列；每个 (k, v) 的形状约定见模块 docstring。"""
        self._layer_kvs = layer_kvs

    def __len__(self) -> int:
        """层数。"""
        return len(self._layer_kvs)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """第 layer_idx 层的 (k, v)。"""
        return self._layer_kvs[layer_idx]

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """按层序遍历 (k, v)。"""
        return iter(self._layer_kvs)

    def to(self, device: torch.device | None = None,
           dtype: torch.dtype | None = None) -> "PrefixKVCache":
        """返回搬移到指定 device/dtype 的新容器（原容器不变）。"""
        return PrefixKVCache([
            (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
            for k, v in self._layer_kvs
        ])
