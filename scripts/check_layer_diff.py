"""逐层对比 neat_pi 的 GemmaLM 与 ref/openpi 官方 PyTorch 移植版。

只加载 VLM 语言主干（Gemma 2B，约 2.6B 参数），不整载入 7.5GB checkpoint，
也不碰视觉塔 / 动作专家。参考实现是 ref/openpi 的 transformers_replace
（其建模里 embedding 的 `sqrt(width)` normalizer 被关闭，与原版 transformers
不同），运行前先把这两个文件覆盖进已安装的 transformers。

依赖隔离（重要）：
    transformers==4.53.2 要求 huggingface-hub<1.0，而本项目 lerobot 要求
    huggingface-hub>=1.0，两者不能装进同一个环境。因此参考对照跑在独立 venv：

        uv venv .venv-ref
        uv pip install --python .venv-ref/bin/python \
            "torch==2.9.1" transformers==4.53.2 safetensors loguru torchtyping
        .venv-ref/bin/python scripts/check_layer_diff.py [--device cuda] [...]

    （torch 固定 2.9.1 以匹配本机 CUDA 12.8 驱动；最新 torch 是 cu13x 会起不来。）

本脚本不 import lerobot（tokenizer 用固定 token id 代替），通过 sys.path 直接
从 src/ 导入 neat_pi，因此 ref venv 无需安装本项目。

两层模型按「先参考后本地」顺序分别加载、运行，避免同时占两份显存；每层输出
逐个对比，打印 max/mean abs、相对 L2 与余弦相似度。

用法：
    .venv-ref/bin/python scripts/check_layer_diff.py \
        [--checkpoint DIR] [--device cuda|cpu] [--seq-len N]
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import shutil
import sys
import sysconfig

# 让脚本能在不安装本项目（ref venv）的情况下 import neat_pi 源码。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from loguru import logger

# ref/openpi 里 transformers_replace 的 gemma 建模文件（覆盖到 site-packages）。
_REF_GEMMA_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "ref/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma"
)

# checkpoint 里 VLM 语言主干的前缀 / tied lm_head 名（见 src/neat_pi/model/weights.py）。
_REF_LM_PREFIX = "model.paligemma_with_expert.paligemma.model.language_model."
_REF_LM_HEAD = "model.paligemma_with_expert.paligemma.lm_head.weight"

_DEFAULT_CHECKPOINT = "/home/ivoryseagull/neat-pi-old/checkpoints/pi05_libero_finetuned"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="逐层对比 GemmaLM 与 openpi PyTorch 移植版")
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT,
                        help="checkpoint 目录（含 model.safetensors）")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="运行设备（默认 cuda bf16；cpu 为 fp32，需 ≥16GB 内存）")
    parser.add_argument("--seq-len", type=int, default=16,
                        help="喂给两模型的 token 序列长度（固定种子随机 id）")
    return parser.parse_args()


def _install_ref_transformers() -> None:
    """把 ref/openpi 的 transformers_replace 覆盖进已安装的 transformers。

    幂等：仅当内容不同才覆盖。覆盖后立刻 import 才生效，因此本脚本在
    import 参考模型前调用它。
    """
    purelib = pathlib.Path(sysconfig.get_paths()["purelib"])
    gemma_dir = purelib / "transformers" / "models" / "gemma"
    if not gemma_dir.exists():
        raise SystemExit("未找到 transformers 安装目录，请先按 docstring 创建 ref venv")

    changed: list[str] = []
    for name in ("modeling_gemma.py", "configuration_gemma.py"):
        src = _REF_GEMMA_DIR / name
        if not src.exists():
            raise SystemExit(f"找不到 ref 文件: {src}")
        dst = gemma_dir / name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)
            changed.append(name)
    if changed:
        logger.warning("已用 ref/openpi 的 transformers_replace 覆盖 site-packages: {}", changed)


def _keep_norms_f32(gemma: torch.nn.Module) -> None:
    """把各层 norm 与 final norm 保持 float32（对齐 openpi 混合精度选择）。"""
    gemma.norm.to(torch.float32)
    for layer in gemma.layers:
        layer.input_layernorm.to(torch.float32)
        layer.post_attention_layernorm.to(torch.float32)


def _load_ref_language_weights(ref: torch.nn.Module, checkpoint_dir: str) -> int:
    """把 checkpoint 的 VLM 语言主干权重加载进参考 GemmaForCausalLM。"""
    from safetensors import safe_open

    state = ref.state_dict()
    shard = pathlib.Path(checkpoint_dir) / "model.safetensors"
    loaded = 0
    with safe_open(str(shard), framework="pt") as f:
        for name in f.keys():
            if name.startswith(_REF_LM_PREFIX):
                local = "model." + name[len(_REF_LM_PREFIX):]
            elif name == _REF_LM_HEAD:
                local = "lm_head.weight"
            else:
                continue
            if local not in state:
                raise KeyError(f"参考模型缺少参数: {local} (来自 {name})")
            tensor = f.get_tensor(name)
            if state[local].shape != tensor.shape:
                raise ValueError(
                    f"形状不匹配: {local} 模型 {tuple(state[local].shape)} "
                    f"vs checkpoint {tuple(tensor.shape)}")
            state[local].copy_(tensor)
            loaded += 1
    if loaded == 0:
        raise RuntimeError("checkpoint 中没有 VLM 语言主干权重")
    return loaded


def _build_reference(checkpoint_dir: str, device: torch.device) -> tuple[torch.nn.Module, int]:
    """构建参考 GemmaForCausalLM（openpi transformers_replace），加载权重并返回。"""
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

    cfg = GemmaConfig(
        vocab_size=257_152,
        hidden_size=2048,
        intermediate_size=16384,
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "sdpa"

    ref = GemmaForCausalLM(cfg)
    if device.type != "cpu":
        ref = ref.to(torch.bfloat16)
        _keep_norms_f32(ref.model)
    loaded = _load_ref_language_weights(ref, checkpoint_dir)
    return ref.to(device).eval(), loaded


def _run_reference(ref: torch.nn.Module, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """跑参考模型，返回 embedding / 各层输出 / final norm / logits（CPU 张量）。

    各层输出用 forward hook 抓（含最后一层）；embedding 单独查 embed_tokens
    （与 lm_head tied、无 sqrt(width) 缩放）；final norm 取 last_hidden_state。
    """
    layer_outs: dict[int, torch.Tensor] = {}
    hooks = []
    for i, layer in enumerate(ref.model.layers):
        def _hook(_m, _inp, out, i: int = i) -> None:
            layer_outs[i] = out[0].detach().cpu()
        hooks.append(layer.register_forward_hook(_hook))

    with torch.inference_mode():
        embed = ref.model.embed_tokens(ids).detach().cpu()
        out = ref(input_ids=ids, use_cache=False, output_hidden_states=True)
        final = out.hidden_states[-1].detach().cpu()  # 最后一层之后的 final norm 输出
        logits = out.logits.detach().cpu()
    for h in hooks:
        h.remove()

    return {
        "embed": embed,
        "layers": [layer_outs[i] for i in range(18)],
        "final": final,
        "logits": logits,
    }


def _run_ours(model: torch.nn.Module, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """跑本地 GemmaLM（逐层手动循环），返回与参考相同的输出集合。"""
    from neat_pi.model.util import build_rope_cache

    with torch.inference_mode():
        embed = F.embedding(ids, model.lm_head.weight)
        cos, sin = build_rope_cache(ids.shape[1], model.attn_head_dim, model.theta,
                                    ids.device, embed.dtype)
        x = embed
        layers = []
        for layer in model.layers:
            x = layer(x, cos, sin)
            layers.append(x.detach().cpu())
        x_norm = model.norm(x)
        logits = model.lm_head(x_norm).detach().cpu()
    return {
        "embed": embed.detach().cpu(),
        "layers": layers,
        "final": x_norm.detach().cpu(),
        "logits": logits,
    }


def _report(name: str, ref: torch.Tensor, ours: torch.Tensor) -> None:
    """打印单个张量的对比指标（都转 fp32 再比较）。"""
    r = ref.float()
    o = ours.float()
    d = (r - o).abs()
    rel = (r - o).norm().item() / (r.norm().item() + 1e-12)
    cos = F.cosine_similarity(r.flatten().unsqueeze(0), o.flatten().unsqueeze(0), dim=1).item()
    logger.info("{:<8} max_abs={:.3e} mean_abs={:.3e} rel_l2={:.3e} cos={:.8f} | "
                "|ref|max={:.3f} |ours|max={:.3f}",
                name, d.max().item(), d.mean().item(), rel, cos,
                r.abs().max().item(), o.abs().max().item())


def main() -> None:
    """加载两实现、跑同一输入、逐层打印输出差距。"""
    args = parse_args()
    _install_ref_transformers()

    from neat_pi.model.gemma import GemmaLM
    from neat_pi.model.weights import load_gemma_lm_weights

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda 不可用，退回 cpu")
        device = torch.device("cpu")
    logger.info("设备: {} | checkpoint: {}", device, args.checkpoint)

    # 固定种子的随机 token id（两模型看到同一输入即可，内容不依赖 tokenizer）。
    g = torch.Generator(device=device).manual_seed(0)
    ids = torch.randint(0, 257_152, (1, args.seq_len), generator=g, device=device)
    logger.info("token id 序列长度: {}", ids.shape[1])

    # 1) 参考实现（先跑，跑完释放，再跑本地，避免两份权重同时占显存）
    ref, loaded_ref = _build_reference(args.checkpoint, device)
    logger.info("参考 GemmaForCausalLM 已加载 {} 张量", loaded_ref)
    ref_out = _run_reference(ref, ids)
    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) 本地实现
    model = GemmaLM()
    if device.type != "cpu":
        model = model.to(torch.bfloat16)
        _keep_norms_f32(model)
    loaded_ours = load_gemma_lm_weights(model, args.checkpoint)
    logger.info("本地 GemmaLM 已加载 {} 张量", loaded_ours)
    model = model.to(device).eval()
    our_out = _run_ours(model, ids)

    # 3) 逐层对比
    logger.info("== 逐层输出对比（fp32 度量）==")
    _report("embed", ref_out["embed"], our_out["embed"])
    for i, (r, o) in enumerate(zip(ref_out["layers"], our_out["layers"], strict=True)):
        _report(f"layer{i}", r, o)
    _report("final", ref_out["final"], our_out["final"])

    r_logits = ref_out["logits"]
    o_logits = our_out["logits"]
    _report("logits", r_logits, o_logits)
    agree = (r_logits.argmax(dim=-1) == o_logits.argmax(dim=-1)).float().mean().item()
    logger.info("logits argmax 一致率: {:.4f}", agree)


if __name__ == "__main__":
    main()
