#!/usr/bin/env bash
# 训练启动脚本：单机多卡或多机 FSDP（torchrun 非弹性模式）。
#
# 用法：
#   单机：        bash scripts/train.sh configs/pi05_libero.yaml
#   多机：        各节点设置 NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT 后执行同一命令
#
# 环境变量覆盖（可选；只覆盖列出的字段，其余一律以 YAML 为准）：
#   数据：      DATASET_ROOT PREPROCESSOR_PATH POSTPROCESSOR_PATH
#               PER_DEVICE_BATCH_SIZE NUM_WORKERS
#   训练：      PRETRAINED STEPS LEARNING_RATE LR_END WARMUP_STEPS WEIGHT_DECAY
#               GRADIENT_ACCUMULATION_STEPS GRADIENT_CLIP_NORM LOG_FREQ SAVE_FREQ
#               KEEP_LAST_N KEEP_EVERY SEED RESUME USE_DUMMY_MODEL EMA_DECAY
#   节点/运行：  NNODES NODE_RANK MASTER_ADDR MASTER_PORT NPROC_PER_NODE
#               OUTPUT_ROOT RUN_NAME JOB_ID
#   可空项（PRETRAINED / GRADIENT_CLIP_NORM / EMA_DECAY）置空或填
#   "null"/"none" 按 null 处理（EMA_DECAY 的 null 即关闭 EMA）。
#
# 每次 run 落在 ${OUTPUT_ROOT}/fsdp_<时间戳>/：时间戳由 node 0 经共享存储发布，
# 保证各节点一致；目录内含 checkpoints/、各节点 resolved_config（YAML 叠加环境
# 变量覆盖后的最终配置）与 train_node<N>.log（各节点分文件写，避免并发冲突）。
#

set -euo pipefail

CONFIG=${1:?用法: train.sh <config.yaml>}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
# 无 GPU 机器上 nvidia-smi 缺失时 wc 得到 "0"（非空，上面的 :- 兜底救不回），按 1 处理
NPROC_PER_NODE=$(( NPROC_PER_NODE > 0 ? NPROC_PER_NODE : 1 ))

OVERRIDE_VARS=(
    DATASET_ROOT PREPROCESSOR_PATH POSTPROCESSOR_PATH
    PER_DEVICE_BATCH_SIZE NUM_WORKERS
    PRETRAINED STEPS LEARNING_RATE LR_END WARMUP_STEPS
    WEIGHT_DECAY GRADIENT_ACCUMULATION_STEPS
    GRADIENT_CLIP_NORM LOG_FREQ SAVE_FREQ
    KEEP_LAST_N KEEP_EVERY SEED RESUME
    USE_DUMMY_MODEL EMA_DECAY
)
for var in "${OVERRIDE_VARS[@]}"; do
    if [[ -n "${!var+x}" ]]; then
        export "${var}"
    fi
done

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/train_runs}
mkdir -p "${OUTPUT_ROOT}"
if [[ ! -w "${OUTPUT_ROOT}" ]]; then
    echo "OUTPUT_ROOT is not writable: ${OUTPUT_ROOT}" >&2
    exit 2
fi
if [[ ! -e "${CONFIG}" ]]; then
    echo "Config does not exist on node ${NODE_RANK}: ${CONFIG}" >&2
    exit 2
fi

# 通过共享存储发布一个时间戳， 保证所有节点用同一个 run 目录
RUN_KEY="${JOB_ID:-${MASTER_ADDR}_${MASTER_PORT}}"
RUN_KEY="${RUN_KEY//[^a-zA-Z0-9_.-]/_}"
START_TIME_FILE="${OUTPUT_ROOT}/.start_time_${RUN_KEY}"
if [[ "${NODE_RANK}" == "0" ]]; then
    date '+%Y%m%d_%H%M%S' > "${START_TIME_FILE}"
else
    for _ in $(seq 1 300); do
        [[ -s "${START_TIME_FILE}" ]] && break
        sleep 1
    done
    if [[ ! -s "${START_TIME_FILE}" ]]; then
        echo "Time out waiting for node 0 to publish the run start time" >&2
        exit 1
    fi
fi

START_TIME=$(<"${START_TIME_FILE}")
RUN_NAME="${RUN_NAME:-fsdp_${START_TIME}}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${NODE_RANK}.log"

# 生成 resolved YAML (叠加环境变量覆盖) 并保存进run目录
# 每个 run 的完整配置都可以从 ${RUN_DIR} 溯源每个节点的文件名
# 避免并发写共享存储上的统一文件； torchrun各自写入
RESOLVED_CONFIG="${RUN_DIR}/resolved_config_node${NODE_RANK}.yaml"
uv run python - "${CONFIG}" "${RESOLVED_CONFIG}" "${RUN_DIR}/checkpoints" <<'EOF'
import os
import sys

import yaml

src, dst, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def _is_null(v: str) -> bool:
    return not v.strip() or v.strip().lower() in ("null", "none")

def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes")

def _nullable_str(v: str) -> str | None:
    return None if _is_null(v) else v

def _nullable_float(v: str) -> float | None:
    return None if _is_null(v) else float(v)

# 环境变量 -> (YAML 段, 键, 转换器)；与脚本头部的 OVERRIDE_VARS 一一对应
OVERRIDES = {
    "DATASET_ROOT": ("data", "root", str),
    "PREPROCESSOR_PATH": ("data", "preprocessor_path", str),
    "POSTPROCESSOR_PATH": ("data", "postprocessor_path", str),
    "PER_DEVICE_BATCH_SIZE": ("data", "per_device_batch_size", int),
    "NUM_WORKERS": ("data", "num_workers", int),
    "PRETRAINED": ("training", "pretrained", _nullable_str),
    "STEPS": ("training", "max_steps", int),
    "LEARNING_RATE": ("training", "lr", float),
    "LR_END": ("training", "lr_end", float),
    "WARMUP_STEPS": ("training", "warmup_steps", int),
    "EMA_DECAY": ("training", "ema_decay", _nullable_float),
    "WEIGHT_DECAY": ("training", "weight_decay", float),
    "GRADIENT_ACCUMULATION_STEPS": ("training", "grad_accum_steps", int),
    "GRADIENT_CLIP_NORM": ("training", "gradient_clip_norm", _nullable_float),
    "LOG_FREQ": ("training", "log_every", int),
    "SAVE_FREQ": ("training", "save_every", int),
    "KEEP_LAST_N": ("training", "keep_last_n", int),
    "KEEP_EVERY": ("training", "keep_every", int),
    "SEED": ("training", "seed", int),
    "RESUME": ("training", "resume", _bool),
    "USE_DUMMY_MODEL": ("training", "use_dummy_model", _bool),
}

training = cfg.setdefault("training", {})
training["output_dir"] = output_dir
applied = []
for env, (section, key, conv) in OVERRIDES.items():
    if env in os.environ:
        cfg.setdefault(section, {})[key] = conv(os.environ[env])
        applied.append(f"{section}.{key}={os.environ[env]!r}")

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"[train.sh] 覆盖 {len(applied)} 项： {', '.join(applied)}", file=sys.stderr)
EOF

echo "[train.sh] node_rank=${NODE_RANK}/${NNODES} local_processes=${NPROC_PER_NODE}"
echo "[train.sh] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[train.sh] config=${CONFIG} -> ${RESOLVED_CONFIG}"
echo "[train.sh] run_dir=${RUN_DIR} log=${LOG_FILE}"

uv run torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/train.py --config "${RESOLVED_CONFIG}" 2>&1 | tee "${LOG_FILE}"
