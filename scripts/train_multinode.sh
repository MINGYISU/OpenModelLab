#!/usr/bin/env bash
# Multi-node without SLURM: run this on every node with the same MASTER_ADDR and a distinct NODE_RANK.
#   MASTER_ADDR=10.0.0.1 NNODES=2 NODE_RANK=0 scripts/train_multinode.sh <config>   # on node 0
#   MASTER_ADDR=10.0.0.1 NNODES=2 NODE_RANK=1 scripts/train_multinode.sh <config>   # on node 1
set -euo pipefail
: "${MASTER_ADDR:?set MASTER_ADDR}" "${NNODES:?set NNODES}" "${NODE_RANK:?set NODE_RANK}"
exec torchrun --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "${GPUS_PER_NODE:-8}" \
  --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29500}" -m finetune.cli train "$@"
