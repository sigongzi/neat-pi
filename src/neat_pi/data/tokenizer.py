"""Gemma/PaliGemma 简易 tokenizer 与对应的 lerobot 管线 step。

不依赖 transformers / sentencepiece：直接从 checkpoint 的 tokenizer.json
读取 `model.vocab`，用最长前缀匹配分词（BPE 的贪心近似，与旧项目一致）。

`GemmaTokenizerStep` 是 lerobot `TokenizerProcessorStep` 的替代品——后者强制
依赖 transformers 且默认从 HF hub 拉 gated 模型。本 step 输出 key 与其完全一致
（observation.language.tokens / observation.language.attention_mask），
在管线中原位替换即可，下游无感知。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from lerobot.processor import (EnvTransition, ProcessorStep,  # noqa: F401
                               TransitionKey)
from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK,
                                     OBS_LANGUAGE_TOKENS)

from neat_pi.typing import MaskBL, TokenIdsBL, typechecked


class GemmaTokenizer:
    """从 tokenizer.json 加载词表与合并规则的 BPE 分词器（Gemma/PaliGemma）。

    复刻 tokenizer.json 的两步处理（已对 checkpoint 的 tokenizer.model 用
    sentencepiece 逐条核对，LIBERO 全部 40 条任务分词一致）：
    - normalizer：整串 `replace(" ", "▁")`——首词不带 '▁'，其余词首都带，
      '\n' 保留为独立字符（词表里有 '\\n' piece）；
    - model：标准 BPE——从单字符出发，按 model.merges 的优先级反复合并
      相邻符号。不做小写化（Gemma 大小写敏感）。
    """

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]] | None = None,
                 pad_id: int = 0, eos_id: int = 1,
                 bos_id: int = 2, unk_id: int = 3) -> None:
        """特殊 token id 默认取 Gemma 约定（pad=0/eos=1/bos=2/unk=3）。"""
        self._vocab = vocab
        self._id_to_piece = {v: k for k, v in vocab.items()}
        # BPE 合并规则 -> 优先级（序号越小越先合并）
        self._ranks = {tuple(m): i for i, m in enumerate(merges)} if merges else {}
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.unk_id = unk_id

    @classmethod
    def from_file(cls, path: str | Path) -> GemmaTokenizer:
        """从 HuggingFace 格式的 tokenizer.json 加载（model.vocab + model.merges）。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "model" not in data or "vocab" not in data["model"]:
            raise ValueError(f"{path} 不含 model.vocab，不是预期的 tokenizer.json")
        merges = [tuple(m.split(" ", 1)) for m in data["model"].get("merges", [])]
        return cls(data["model"]["vocab"], merges)

    @property
    def vocab_size(self) -> int:
        """词表大小（含特殊 token）。"""
        return max(self._vocab.values()) + 1 if self._vocab else 0

    def encode(self, text: str, max_len: int, add_bos: bool = True) -> list[int]:
        """编码单条文本为 id 序列（加 BOS，truncate/pad 到 max_len）。"""
        ids: list[int] = [self.bos_id] if add_bos else []
        ids.extend(self._bpe(text))
        if len(ids) > max_len:
            ids = ids[:max_len]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def _bpe(self, text: str) -> list[int]:
        """标准 BPE：整串 replace(' ', '▁') 后按 merge 优先级反复合并相邻符号。"""
        symbols = list(text.replace(" ", "▁"))  # 复刻 tokenizer.json 的 normalizer
        while len(symbols) > 1:
            # 找优先级最高（rank 最小）的相邻对
            best_rank, best_i = None, -1
            for i in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            symbols[best_i:best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        return [self._vocab.get(s, self.unk_id) for s in symbols]

    @typechecked
    def encode_batch(self, texts: list[str],
                     max_len: int) -> tuple[TokenIdsBL, MaskBL]:
        """批量编码，返回 (token_ids [B, L], attention_mask [B, L])，mask True=有效。"""
        rows = [self.encode(t, max_len) for t in texts]
        ids = torch.tensor(rows, dtype=torch.long)
        mask = ids != self.pad_id
        return ids, mask

    def decode(self, ids: torch.Tensor | list[int], skip_special: bool = True) -> str:
        """id 序列转回文本（'▁' 还原为空格）。"""
        flat = ids.flatten().tolist() if isinstance(ids, torch.Tensor) else ids
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        pieces = [self._id_to_piece.get(t, "<unk>") for t in flat
                  if not (skip_special and t in specials)]
        return "".join(pieces).replace("▁", " ").strip()

    def id_to_piece(self, token_id: int) -> str:
        """返回 token id 对应的 piece 字符串。"""
        return self._id_to_piece.get(token_id, "<unk>")


@dataclass
class GemmaTokenizerStep(ProcessorStep):
    """把 complementary_data 里的完整 prompt 编码成 token ids 写入 observation。

    在 lerobot PolicyProcessorPipeline 中替代 TokenizerProcessorStep（原版强制
    依赖 transformers）。期望输入的 task 已被 pi05 step 拼成完整 prompt
    （"Task: ..., State: ...;\\nAction: "）。构造参数只存 tokenizer_path，
    保证 step 配置可序列化成 JSON，再由 lerobot 的 class 动态导入实例化。
    """

    tokenizer_path: str = ""
    max_length: int = 200
    task_key: str = "task"
    _tokenizer: GemmaTokenizer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """从 tokenizer_path 加载 GemmaTokenizer，缺路径直接报错。"""
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path 不能为空：需指向 checkpoint 的 tokenizer.json")
        self._tokenizer = GemmaTokenizer.from_file(self.tokenizer_path)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """编码 task 文本，把 token ids / attention mask 写入 observation。"""
        transition = transition.copy()
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        tasks: Any = comp.get(self.task_key)
        if tasks is None:
            raise ValueError(f"complementary_data 中找不到 '{self.task_key}'，无法 tokenize")
        if isinstance(tasks, str):
            tasks = [tasks]

        ids, mask = self._tokenizer.encode_batch(list(tasks), self.max_length)
        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        obs[OBS_LANGUAGE_TOKENS] = ids
        obs[OBS_LANGUAGE_ATTENTION_MASK] = mask
        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def transform_features(self, features: Any) -> Any:
        """本 step 不改变已有 feature 定义，原样返回。"""
        return features
