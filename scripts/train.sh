#!/usr/bin/env bash
# Single-node launcher.  NGPU defaults to all visible GPUs.
#   scripts/train.sh configs/experiments/e1_sft_lora_qwen2.5_1.5b.yaml --set train.lr=1e-4
#   CUDA_VISIBLE_DEVICES=0,1 scripts/train.sh <config>
set -euo pipefail
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"; DEFAULT_NGPU=${#DEVICES[@]}
else
  DEFAULT_NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NGPU=${NGPU:-${DEFAULT_NGPU:-1}}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
if [[ "$NGPU" -le 1 ]]; then
  exec python -m finetune.cli train "$@"
fi
exec torchrun --standalone --nproc_per_node "$NGPU" -m finetune.cli train "$@"
