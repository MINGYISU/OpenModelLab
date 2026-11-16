# Training setup

**Next-token loss.** `labels = input_ids`; the model shifts internally. Positions with
label `-100` are ignored. Code: `training/loop.py:token_loss_sum`, which spells out the
shift and the token-sum.

**What is supervised** (`data/sft.py`):
- `loss_on: all`: system, user and assistant tokens all count. This is the simplest
  baseline; the model also learns to imitate users.
- `loss_on: assistant`: only assistant content plus the end-of-turn token. Spans come
  from rendering growing prefixes of the conversation and mapping character offsets
  to tokens.
- Padding is always `-100`. Real EOS tokens stay supervised, so the model learns to stop.

**Packing** (`data/pretrain.py`): concatenate documents with EOS and cut fixed blocks.
This wastes no compute on padding, but tokens can attend across document boundaries.

**Batch arithmetic.** The global batch (sequences per optimizer step) is
`per_device_batch_size × grad_accum × world_size`, and tokens per step is that times
the sequence length. The raw loop normalises the loss by the token count over the
whole accumulation window and all ranks, so every token weighs the same (see
the scaling comment in `loop.py`).

**Optimizer** (`training/optim.py`):
- AdamW with betas (0.9, 0.95) for LLMs and eps 1e-8.
- No weight decay on norms, biases or embeddings.
- Gradient clipping at 1.0; log `grad_norm` and watch for spikes.

**Schedules.**
- Warmup prevents the large early Adam updates (variance estimates are noisy at
  first). The steps-vs-epochs distinction matters here: warmup is in optimizer steps.
- `cosine`: decays smoothly to `min_lr_ratio`.
- `wsd` (warmup-stable-decay): holds the peak LR and decays only at the end. You can
  branch a decayed model from any stable checkpoint, and it is good for scaling studies.
- Rough LR scales: LoRA SFT ~1e-4 to 2e-4; full SFT ~1e-5 to 2e-5; DPO ~5e-7 to 5e-6;
  GRPO ~1e-6 to 1e-5; from-scratch pre-training ~3e-4 to 1e-3 for small models.

**Precision.**
- Full fine-tuning loads fp32 *master weights* and runs bf16 autocast
  (`common.model_load_dtype`).
- LoRA keeps the frozen base in bf16 to save memory; only the adapters train.
- TF32 speeds up any fp32 matmuls that remain.

**Memory.** Weights + gradients + Adam states ≈ 16 bytes per trainable parameter in
mixed precision, plus activations. Activation checkpointing trades compute for memory;
LoRA removes almost all gradient and optimizer memory.

**LoRA** (`training/lora.py:LoRALinear`): `y = Wx + (α/r)·B(A x)`.
- B is initialised to zero, so training starts exactly at the base model.
- rsLoRA uses `α/√r`.
- `tests/test_training.py` checks our module against PEFT, then merges `W += (α/r)BA`.

**Two backends.** `Trainer` (what most papers' code uses) and `training/loop.py` (the
same thing written out: DDP, `no_sync`, clipping, scheduler, checkpoint and exact
resume). Train both on the smoke config and compare the logs.
