"""计划 05 eval 合同配置测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (policy_action_to_transition,
                                          transition_to_policy_action)
import lerobot.policies.pi05.processor_pi05  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "eval_pi05_libero_finetuned.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 并保证顶层是映射。"""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_eval_config_matches_checkpoint_contract() -> None:
    """评测 YAML 声明旧 LIBERO checkpoint 的全部关键推理合同。"""
    config = _load_yaml(CONFIG_PATH)
    checkpoint_dir = Path(config["checkpoint"]["path"])

    assert checkpoint_dir.is_dir()
    assert Path(config["checkpoint"]["tokenizer_path"]) == checkpoint_dir / "tokenizer.json"
    assert config["device"] == {"type": "cuda", "dtype": "bfloat16"}
    assert config["model"] == {
        "image_size": 224,
        "num_cameras": 3,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "action_horizon": 50,
    }
    assert config["actions"]["raw_dim"] == 7
    assert config["actions"]["flow_steps"] == 10
    assert config["actions"]["normalization"] == "MEAN_STD"
    assert config["state"]["raw_dim"] == 8
    assert config["state"]["normalization"] == "MEAN_STD"
    assert config["tokenizer"]["max_length"] == 200
    assert config["tokenizer"]["padding_side"] == "right"


def test_eval_camera_contract_keeps_empty_camera() -> None:
    """两路真实相机映射正确，第三路 empty camera 保留且 mask 为 False。"""
    config = _load_yaml(CONFIG_PATH)
    cameras = config["cameras"]

    assert [(camera["source"], camera["key"], camera["mask"]) for camera in cameras["real"]] == [
        ("agentview_image", "observation.images.image", True),
        ("robot0_eye_in_hand_image", "observation.images.image2", True),
    ]
    assert cameras["empty"] == {
        "key": "observation.images.empty_camera_0",
        "fill_value": -1.0,
        "mask": False,
    }


def test_eval_processor_pipelines_are_configured_for_local_checkpoint() -> None:
    """MEAN_STD pre/post processor 可离线加载且形状与 7 维 action 一致。"""
    config = _load_yaml(CONFIG_PATH)
    preprocessor_path = ROOT / config["normalization"]["preprocessor_path"]
    postprocessor_path = ROOT / config["normalization"]["postprocessor_path"]
    preprocessor_config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    postprocessor_config = json.loads(postprocessor_path.read_text(encoding="utf-8"))
    checkpoint_dir = Path(config["checkpoint"]["path"])

    normalizer = preprocessor_config["steps"][2]
    unnormalizer = postprocessor_config["steps"][0]
    tokenizer = preprocessor_config["steps"][4]
    assert normalizer["config"]["norm_map"] == {
        "ACTION": "MEAN_STD",
        "STATE": "MEAN_STD",
        "VISUAL": "IDENTITY",
    }
    assert normalizer["config"]["features"]["observation.state"]["shape"] == [8]
    assert normalizer["config"]["features"]["action"]["shape"] == [7]
    assert normalizer["state_file"] == str(
        checkpoint_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    )
    assert unnormalizer["state_file"] == str(
        checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    assert tokenizer["class"] == "neat_pi.data.tokenizer.GemmaTokenizerStep"
    assert tokenizer["config"]["tokenizer_path"] == config["checkpoint"]["tokenizer_path"]
    assert tokenizer["config"]["max_length"] == config["tokenizer"]["max_length"]

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(preprocessor_path),
        config_filename=preprocessor_path.name,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        str(postprocessor_path),
        config_filename=postprocessor_path.name,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    assert len(preprocessor.steps) == 6
    assert len(postprocessor.steps) == 2
