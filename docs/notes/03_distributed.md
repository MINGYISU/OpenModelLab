# Distributed training

**Launch.** `torchrun --nproc_per_node N` starts N processes and sets `RANK`,
`LOCAL_RANK` and `WORLD_SIZE`. Multi-node adds `--nnodes` and a rendezvous endpoint
(`scripts/train_multinode.*`). NCCL handles GPU collectives; set `NCCL_DEBUG=INFO`
when a run hangs.

| Strategy | What each GPU holds | Communication | Config |
| --- | --- | --- | --- |
| DDP | full model, grads, optimizer | all-reduce of grads after backward | default under torchrun |
| FSDP2 | 1/N of params, grads, optimizer states | all-gather params per layer (forward/backward), reduce-scatter grads | `configs/dist/fsdp2.yaml` |
| ZeRO-2 | full params; 1/N of grads + optimizer states | reduce-scatter grads | `configs/dist/zero2.yaml` |
| ZeRO-3 | like FSDP full shard | all-gather + reduce-scatter | `configs/dist/zero3.yaml` |

Details that matter:

- **Data sharding**: `DistributedSampler` gives each rank a disjoint slice; call
  `set_epoch` to reshuffle each epoch. For streaming data, `data/pretrain.py` shards by
  `(rank, worker)` in the raw loop. With `Trainer`, accelerate dispatches the batches.
- **Gradient accumulation with DDP**: wrap all but the last micro-batch in
  `model.no_sync()`, otherwise you all-reduce on every micro-batch (`loop.py`).
- **Seeds**: model init must match across ranks (same seed). Per-rank randomness
  (sampling in GRPO) uses `seed + rank`.
- **Rank 0** logs and saves; the others wait at a barrier.
- **Sanity check**: the same config on 1 GPU and on N GPUs at equal *global* batch
  should give nearly identical loss curves.
- **MoE expert parallelism** (experts split across GPUs with all-to-all token dispatch)
  is how DeepSeek-V3 is trained at scale. Here MoE models are small enough for FSDP.

Exercise: run `configs/smoke/sft.yaml` with `train.backend=loop` on 1 and 2 GPUs
(`CUDA_VISIBLE_DEVICES=0,1 scripts/train.sh ...`), halving `grad_accum` on 2 GPUs, and
compare `log.jsonl`.
