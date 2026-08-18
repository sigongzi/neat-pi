"""Gemma 语言主干（pi05 的 VLM 侧）。

pi05 用 Gemma 系 decoder-only 作为 VLM 主干：文本 token embedding 与
视觉 token 拼接后进入 transformer 堆叠。

注意（FSDP / MoT 的关键点）：
pi05 里 VLM 主干与动作专家**共享同一套 attention 计算**（动作专家作为
另一个"专家"复用 attention 的 KV）。为让 FSDP 能按专家干净切分，
共享层不在这里实现，而在 mot.py 的 MoTLayer 中按专家组织；
本文件只保留 embedding / final norm 等 VLM 私有部分。

结构对齐 ref/openpi/src/openpi/models/gemma.py。
"""

from __future__ import annotations

from torch import nn

from neat_pi.typing import TokenIdsBL, TokensBTD, typechecked


class GemmaEmbedding(nn.Module):
    """Gemma token embedding（含 pi05 使用的 normalizer：emb * sqrt(width)）。"""

    def __init__(self, vocab_size: int = 257_152, width: int = 2048) -> None:
        super().__init__()
        self.width = width
        self.embed_tokens = nn.Embedding(vocab_size, width)

    @typechecked
    def forward(self, token_ids: TokenIdsBL) -> TokensBTD:
        """token id -> embedding，并按 Gemma 约定乘 sqrt(width)。"""
        emb = self.embed_tokens(token_ids)
        return emb * (self.width ** 0.5)


class GemmaFinalNorm(nn.Module):
    """VLM 侧输出前的 final RMSNorm（占位，待与 mot.py 的输出接口对齐）。"""

    def __init__(self, width: int = 2048) -> None:
        super().__init__()
        from neat_pi.model.modules import RMSNorm

        self.norm = RMSNorm(width)

    @typechecked
    def forward(self, x: TokensBTD) -> TokensBTD:
        return self.norm(x)
