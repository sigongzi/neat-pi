"""LIBERO eval prompt 与本地 tokenizer adapter 测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from neat_pi.eval.observation import LiberoObservationAdapter
from neat_pi.eval.prompt import LiberoPromptTokenizerAdapter


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_pi05_libero_finetuned.yaml"


def _load_config() -> dict[str, Any]:
    """读取 eval YAML 顶层映射。"""
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _observation_module() -> Any:
    """加载测试辅助模块，复用固定 LIBERO 原始观测。"""
    spec = importlib.util.spec_from_file_location(
        "test_eval_observation", ROOT / "tests" / "test_eval_observation.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapter() -> LiberoPromptTokenizerAdapter:
    """构造待测 prompt/tokenizer adapter。"""
    return LiberoPromptTokenizerAdapter.from_config(
        _load_config(), config_root=CONFIG_PATH.parent)


def _language(task: str = "pick up the black bowl on the table"):
    """用固定 observation 与 task 生成一次语言输入。"""
    config = _load_config()
    observation = LiberoObservationAdapter.from_config(
        config["cameras"], image_size=config["model"]["image_size"])
    return _adapter().build(
        observation.convert(_observation_module()._observation()), task)


def test_local_tokenizer_produces_right_padded_tokens() -> None:
    """本地 tokenizer 输出固定长度 token，padding 只出现在右侧。"""
    language = _language()
    config = _load_config()
    assert Path(config["checkpoint"]["tokenizer_path"]).is_file()
    assert language.token_ids.shape == (1, 200)
    assert language.attention_mask.shape == (1, 200)
    assert language.attention_mask.dtype is torch.bool
    valid_length = int(language.attention_mask.sum().item())
    assert valid_length > 1
    assert language.attention_mask[:, :valid_length].all()
    assert not language.attention_mask[:, valid_length:].any()
    assert (language.token_ids[0, valid_length:] == 0).all()
    assert set(language.token_ids[0, :valid_length].tolist()).issubset(
        set(range(config["model"]["vocab_size"]))
        if "vocab_size" in config["model"] else set(range(257_152))
    )


def test_prompt_matches_pi05_checkpoint_contract() -> None:
    """state 会拼进 prompt，格式与旧 pi05 processor 完全一致。"""
    language = _language()
    state_fields = language.prompt.split("State: ", 1)[1].split(";\nAction:", 1)[0]
    assert language.prompt.startswith("Task: pick up the black bowl on the table, State: ")
    assert language.prompt.endswith(";\nAction: ")
    assert len(state_fields.split()) == 8
    assert all(field.lstrip("-").isdigit() for field in state_fields.split())


def test_same_task_and_state_is_reproducible() -> None:
    """相同 state/task 的完整 prompt 与 token ids 完全一致。"""
    first = _language()
    second = _language()
    assert first.prompt == second.prompt
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.attention_mask, second.attention_mask)


def test_invalid_padding_side_is_rejected() -> None:
    """本地 tokenizer 合同只允许 right padding。"""
    config = _load_config()
    config["tokenizer"]["padding_side"] = "left"
    with pytest.raises(ValueError, match="right padding"):
        LiberoPromptTokenizerAdapter.from_config(
            config, config_root=CONFIG_PATH.parent)
