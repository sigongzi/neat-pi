"""Gemma 语言主干权重自检：加载 checkpoint 的 VLM 文本 decoder 并做贪心生成。

用途：验证 checkpoint 里 PaliGemma 的 Gemma 2B 主干（纯文本）权重能否加载，
以及是否具备基本的文本续写 / 对话能力。token embedding 与 lm_head 是 tied 的
同一份权重（checkpoint 里只有 `paligemma.lm_head.weight`，没有 embed_tokens）。

用法：
    uv run scripts/check_gemma_lm.py [--config configs/pi05_libero.yaml]
                                     [--checkpoint DIR] [--cpu]
                                     [--prompt "The capital of France is"]
                                     [--max-new-tokens 64]

checkpoint 目录默认取配置里的 training.pretrained。只加载 VLM 语言主干的
164 张量（约 2.6B 参数），不整载入 7.5GB checkpoint。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from loguru import logger

from neat_pi.config import load_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Gemma 语言主干权重自检")
    parser.add_argument("--config", default="configs/pi05_libero.yaml",
                        help="YAML 配置路径（取 device.type 与 training.pretrained）")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 目录（覆盖配置里的 training.pretrained）")
    parser.add_argument("--cpu", action="store_true",
                        help="强制在 CPU 上跑（默认按配置 device.type）")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="生成用的提示文本")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="最多生成的新 token 数")
    return parser.parse_args()


def _keep_norms_f32(model: torch.nn.Module) -> None:
    """把各层 norm 与 final norm 保持 float32（对齐 openpi 的混合精度选择）。"""
    model.norm.to(torch.float32)
    for layer in model.layers:
        layer.input_layernorm.to(torch.float32)
        layer.post_attention_layernorm.to(torch.float32)


def main() -> None:
    """加载 VLM 语言主干权重，跑一次贪心生成并打印续写结果。"""
    args = parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = args.checkpoint or cfg.training.pretrained
    if not checkpoint_dir:
        raise SystemExit("未指定 checkpoint：请用 --checkpoint 或配置 training.pretrained")

    from neat_pi.data.tokenizer import GemmaTokenizer
    from neat_pi.model.gemma import GemmaLM
    from neat_pi.model.weights import load_gemma_lm_weights

    # 设备
    device = torch.device("cpu") if args.cpu else torch.device(cfg.device.type)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda 不可用，退回 cpu")
        device = torch.device("cpu")
    logger.info("设备: {}", device)

    # 模型 dtype：cuda 上 bf16 + norm 保持 f32；cpu 上全 f32
    model = GemmaLM()
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
        _keep_norms_f32(model)

    loaded = load_gemma_lm_weights(model, checkpoint_dir)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("已加载 VLM 语言主干 {} 张量，共 {:,} 参数", loaded, n_params)

    model = model.to(device).eval()

    # 词表只来自 checkpoint 的 tokenizer.json（与权重同源）
    tokenizer = GemmaTokenizer.from_file(f"{checkpoint_dir}/tokenizer.json")
    ids = [t for t in tokenizer.encode(args.prompt, max_len=256) if t != tokenizer.pad_id]
    logger.info("prompt: {!r} -> {} token", args.prompt, len(ids))

    # 贪心续写：token -> tied embedding（乘 sqrt(hidden_dim)）-> GemmaLM -> argmax。
    gen = list(ids)
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            cur = torch.tensor([gen], device=device)
            x = F.embedding(cur, model.lm_head.weight) * (model.hidden_dim ** 0.5)
            nxt = int(model(x)[0, -1].argmax(dim=-1))
            gen.append(nxt)
            if nxt == tokenizer.eos_id:
                break
    generated = tokenizer.decode(gen)
    logger.info("生成结果: {!r}", generated)


if __name__ == "__main__":
    main()
