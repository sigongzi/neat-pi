#!/usr/bin/env bash
# 训练启动脚本：单机多卡或多机 FSDP。
#
# 用法：
#   单机：        bash scripts/train.sh configs/pi05_libero.yaml
#   多机：        各节点设置 NODE_RANK / MASTER_ADDR / NNODES 后执行同一命令
#
# 设备类型（cuda/npu）在 YAML 里配，不在本脚本里写死。

set -euo pipefail

CONFIG=${1:?用法: train.sh <config.yaml>}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

uv run torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/train.py --config "${CONFIG}"
