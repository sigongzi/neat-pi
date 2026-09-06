"""把原始 pi05 checkpoint 转换成 NeatPi 本地参数名格式。

转换产物固定为单个 model.safetensors，其中每个张量名都与
Pi05.state_dict() 一致；加载该产物时不再做名字翻译。命令行入口只负责
参数解析和日志，实际转换逻辑在
neat_pi.model.weights.convert_pi05_checkpoint。
"""

from __future__ import annotations

import argparse

from loguru import logger

from neat_pi.model.weights import convert_pi05_checkpoint


def parse_args() -> argparse.Namespace:
    """解析转换参数。"""
    parser = argparse.ArgumentParser(
        description="把 pi05 原始 checkpoint 另存为 NeatPi 本地名格式")
    parser.add_argument("--source", required=True,
                        help="原始 checkpoint 目录")
    parser.add_argument("--output", required=True,
                        help="转换产物目录；必须为空或不存在")
    return parser.parse_args()


def main() -> None:
    """执行 checkpoint 名字转换并打印统计。"""
    args = parse_args()
    result = convert_pi05_checkpoint(args.source, args.output)
    logger.info(
        "转换完成: {} 张量 -> {} | 跳过 {} 张 | {} 个文件",
        result.loaded_count, result.output_dir,
        result.skipped_count, result.shard_count,
    )


if __name__ == "__main__":
    main()
