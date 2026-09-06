"""eval 配置中的相对路径解析工具。"""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(path: str, config_root: str | None = None) -> Path:
    """按 eval YAML 所在目录解析相对路径；绝对路径原样返回。"""
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    if config_root is not None:
        return Path(config_root).expanduser() / resolved
    return resolved
