"""加载 pi05 预处理管线（lerobot PolicyProcessorPipeline）。

管线步骤声明在本项目 `configs/pi05_libero_preprocessor.json`（以 finetuned
checkpoint 的 policy_preprocessor.json 为蓝本改写，见 docs/plan/01）：
- 常规步骤用 lerobot `ProcessorStepRegistry` 的 `registry_name` 实例化；
- 自写 `GemmaTokenizerStep` 用 `"class"` 字段走 importlib 动态导入，无需注册；
- normalizer 的 mean/std 统计通过 `state_file`（绝对路径）从 checkpoint 加载。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import (policy_action_to_transition,
                                          transition_to_policy_action)

if TYPE_CHECKING:
    from neat_pi.config import Config


def load_preprocessor_file(preprocessor_path: str | Path,
                           ) -> PolicyProcessorPipeline:
    """从 JSON 文件加载 policy 预处理管线并返回。

    先 import `lerobot.policies.pi05.processor_pi05` 触发 pi05 自定义 step
    的注册，其余步骤（含自写 GemmaTokenizerStep 及带 state_file 的
    normalizer 统计）交给 lerobot 的 `from_pretrained` 自动解析。
    """
    import lerobot.policies.pi05.processor_pi05  # noqa: F401  注册 pi05 step

    path = Path(preprocessor_path)
    if not path.is_file():
        raise FileNotFoundError(f"preprocessor JSON 不存在: {path}")
    return PolicyProcessorPipeline.from_pretrained(
        str(path), config_filename=path.name
    )


def load_postprocessor_file(postprocessor_path: str | Path,
                            ) -> PolicyProcessorPipeline:
    """从 JSON 文件加载 policy 后处理管线并返回。

    输入/输出转换固定为 policy action tensor，适配推理侧动作块。
    """
    import lerobot.policies.pi05.processor_pi05  # noqa: F401  注册 pi05 step

    path = Path(postprocessor_path)
    if not path.is_file():
        raise FileNotFoundError(f"postprocessor JSON 不存在: {path}")
    return PolicyProcessorPipeline.from_pretrained(
        str(path),
        config_filename=path.name,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def load_preprocessor(cfg: Config) -> PolicyProcessorPipeline:
    """按 cfg.data.preprocessor_path 的 JSON 文件加载预处理管线并返回。
    """
    if not cfg.data.preprocessor_path:
        raise ValueError("data.preprocessor_path 未设置：需指向 policy_preprocessor.json")
    return load_preprocessor_file(cfg.data.preprocessor_path)


def load_postprocessor(cfg: Config) -> PolicyProcessorPipeline:
    """按 cfg.data.postprocessor_path 的 JSON 文件加载推理侧后处理管线。

    步骤（见 configs/pi05_libero_postprocessor.json）：unnormalizer（与
    preprocessor 共用同一份 SE(3) 统计量）→ SE3DeltaToCommandStep（10 维
    delta 转 7 维当前帧 EE 系指令）→ device。输出若需 base 系指令，在环境
    接口侧用 `neat_pi.data.se3.ee_command_to_base` 再转。
    """
    if not cfg.data.postprocessor_path:
        raise ValueError("data.postprocessor_path 未设置：需指向 policy_postprocessor.json")
    return load_postprocessor_file(cfg.data.postprocessor_path)
