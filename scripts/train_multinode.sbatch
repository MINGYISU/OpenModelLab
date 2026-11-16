#!/usr/bin/env bash
# Multi-node launcher for SLURM: one torchrun per node, c10d rendezvous on the first node.
#   sbatch --nodes 2 scripts/train_multinode.sbatch configs/experiments/<config>.yaml [--set k=v ...]
#SBATCH --job-name=finetune
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=8
# NCCL_DEBUG=INFO helps diagnose hangs; set NCCL_SOCKET_IFNAME / NCCL_IB_HCA for your fabric.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
srun torchrun \
  --nnodes "$SLURM_NNODES" --nproc_per_node "$GPUS_PER_NODE" \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  -m finetune.cli train "$@"
