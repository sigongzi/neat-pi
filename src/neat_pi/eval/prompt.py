"""把 task 与 normalized LIBERO state 组成旧 checkpoint 的 pi05 prompt。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lerobot.processor import PolicyProcessorPipeline

from neat_pi.data.preprocessor import load_preprocessor_file
from neat_pi.eval.config import resolve_config_path
from neat_pi.data.tokenizer import GemmaTokenizerStep
from neat_pi.eval.observation import Pi05EvalObservation
from neat_pi.typing import MaskBL, TokenIdsBL, typechecked


@dataclass(frozen=True)
class Pi05EvalLanguage:
    """一次 eval 前向所需的语言输入与可复现 prompt 诊断信息。"""

    token_ids: TokenIdsBL
    attention_mask: MaskBL
    prompt: str


@dataclass(frozen=True)
class LiberoPromptTokenizerAdapter:
    """复用 eval preprocessor 管线生成固定合同的语言 token。"""

    preprocessor: PolicyProcessorPipeline
    max_length: int

    @classmethod
    def from_config(cls, eval_config: Mapping[str, Any],
                    config_root: str | Path | None = None,
                    ) -> "LiberoPromptTokenizerAdapter":
        """从 eval YAML 配置构造 adapter，并校验 tokenizer 合同没有漂移。"""
        normalization = eval_config["normalization"]
        path = resolve_config_path(
            normalization["preprocessor_path"], config_root)
        preprocessor = load_preprocessor_file(path)
        tokenizer_steps = [
            step for step in preprocessor.steps
            if isinstance(step, GemmaTokenizerStep)
        ]
        if len(tokenizer_steps) != 1:
            raise ValueError(
                f"eval preprocessor 必须恰好有一个 GemmaTokenizerStep，"
                f"实际 {len(tokenizer_steps)} 个")
        tokenizer_step = tokenizer_steps[0]
        expected_tokenizer = str(eval_config["checkpoint"]["tokenizer_path"])
        expected_length = int(eval_config["tokenizer"]["max_length"])
        expected_padding = eval_config["tokenizer"]["padding_side"]
        if tokenizer_step.tokenizer_path != expected_tokenizer:
            raise ValueError(
                "preprocessor tokenizer 路径与 eval YAML 不一致: "
                f"{tokenizer_step.tokenizer_path} != {expected_tokenizer}")
        if tokenizer_step.max_length != expected_length:
            raise ValueError(
                "preprocessor max_length 与 eval YAML 不一致: "
                f"{tokenizer_step.max_length} != {expected_length}")
        if expected_padding != "right":
            raise ValueError(
                f"旧 checkpoint tokenizer 只支持 right padding，实际 {expected_padding}")
        return cls(
            preprocessor=preprocessor,
            max_length=expected_length,
        )

    @typechecked
    def build(self, observation: Pi05EvalObservation,
              task: str) -> Pi05EvalLanguage:
        """归一化 state、拼接 prompt 并用本地 tokenizer 编码。"""
        batch = observation.policy_observation()
        batch["task"] = task
        processed = self.preprocessor(batch)
        tasks = processed.get("task")
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("preprocessor 未返回 batch=1 的完整 prompt")
        language = Pi05EvalLanguage(
            token_ids=processed["observation.language.tokens"],
            attention_mask=processed["observation.language.attention_mask"],
            prompt=tasks[0],
        )
        if language.token_ids.shape != (1, self.max_length):
            raise ValueError(
                f"token shape 应为 (1, {self.max_length})，实际 "
                f"{tuple(language.token_ids.shape)}")
        return language
