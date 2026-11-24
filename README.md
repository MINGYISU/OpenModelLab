# FineTune

A research codebase for **pre-training, continued pre-training, embedding training,
and post-training (SFT, DPO, GRPO)** of open LLMs (Qwen2/2.5, Qwen3,
DeepSeek-R1-Distill, DeepSeek-V2/V3, and Llama-style models). It also includes tools to
measure **how each training stage changes a model's behavior**.

The research roadmap and experiment matrix are in [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md);
learning notes live in [docs/notes/](docs/notes/).

**Just want to fine-tune on your own data?** Read [docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md):
write a small recipe for the dataset, run `finetune prepare <name>`, copy
`configs/templates/sft_my_dataset.yaml`, and train.

Design:

- **Vendored model source**: `src/finetune/models/<family>/modeling_*.py` are copied
  from the pinned `transformers` release and are what actually runs. There is no `AutoModel*`
  and no `trust_remote_code`. Tests check that the logits match upstream exactly. See
  [VENDORED.md](src/finetune/models/VENDORED.md).
- **Still built on the ecosystem**: `transformers` (`PreTrainedModel`, tokenizers,
  `Trainer`, generation), `peft`, `accelerate`, `datasets`, DeepSpeed.
- **Everything explicit**: the optimizer parameter groups, LR schedules, a
  from-scratch LoRA, a raw PyTorch DDP training loop, the DPO/GRPO losses, and
  GradCache are all small, readable modules with tests.
- **Config-driven and reproducible**: YAML configs with inheritance; every run writes
  `resolved_config.json` and `experiment.json` (git SHA, library versions, GPUs,
  dataset hashes, resolved model revision).

## Setup

```bash
conda create -n finetune python=3.12 -y && conda activate finetune
pip install -e '.[ml]'            # torch, transformers, peft, accelerate, datasets
pip install -e '.[rl,dist,eval]'  # optional: trl, deepspeed, lm-eval + mteb
python -m unittest discover -s tests   # CPU-only, offline (~30 s)
```

## Layout

```text
src/finetune/
  models/        vendored qwen2, qwen3, llama, deepseek_v2, deepseek_v3 + registry (build without AutoModel)
  data/          JSONL/HF loading, chat-template SFT masks, packing, preference, RL, contrastive, collators
  training/      causal_lm (pretrain/SFT), dpo, grpo, embedding (GradCache), loop (raw DDP loop),
                 optim (param groups, warmup/cosine/WSD), lora (PEFT + from-scratch), rewards
  eval/          generation, perplexity, reasoning (pass@k, think-trace stats), retrieval, lm-eval
  eval/behavior/ weight_diff, kl_drift, logit_lens, representation (CKA), moe_router
configs/         base.yaml, smoke/ (tiny runs), experiments/ (E1-E9 and more), dist/ (DDP, FSDP2, ZeRO-2/3)
scripts/         train.sh (single node), train_multinode.{sh,sbatch}, vendor_models.py
examples/        tiny datasets for every format
```

## Running

Every stage uses the same command; `train.stage` in the config picks the pipeline.

```bash
# Smoke test any stage on a tiny random model + byte tokenizer (no downloads):
finetune train configs/smoke/sft.yaml          # also: pretrain, dpo, grpo, embed

# Real experiments: one GPU, or all visible GPUs via torchrun (DDP by default):
scripts/train.sh configs/experiments/e1_sft_lora_qwen2.5_1.5b.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train.sh configs/experiments/e1_sft_full_qwen2.5_1.5b.yaml  # FSDP2

# Override anything from the command line; print the resolved config without training:
finetune train configs/experiments/e4_dpo_qwen2.5_0.5b_ultrafeedback.yaml --set dpo.beta=0.3 --dry-run

# Multi-node (SLURM, or MASTER_ADDR/NNODES/NODE_RANK without SLURM):
sbatch --nodes 2 scripts/train_multinode.sbatch configs/experiments/e6_cpt_qwen2.5_1.5b_math_replay.yaml
```

A single process (`finetune train` without torchrun) uses one GPU. Use
`scripts/train.sh` / `torchrun` for multi-GPU; otherwise `Trainer` would fall back to
legacy `nn.DataParallel`.

| Stage | Pipeline | Notes |
| --- | --- | --- |
| `pretrain` | `training/causal_lm.py` | packing, streaming HF datasets, replay mixing; `model.init: scratch` for random init |
| `sft` | `training/causal_lm.py` | native chat templates; `data.loss_on: all \| assistant` |
| `dpo` | `training/dpo.py` | sigmoid / IPO / SimPO; reference log-probs precomputed from the initial policy |
| `grpo` | `training/grpo.py` | explicit rollout → reward → group advantage → clipped objective + k3 KL loop |
| `embed` | `training/embedding.py` | InfoNCE, mean/last/cls pooling, GradCache, cross-GPU negatives |

Backends: `train.backend: trainer` (transformers `Trainer`, which supports DDP, FSDP2,
and DeepSpeed ZeRO-2/3) or `loop` (our raw PyTorch DDP loop for pretrain/SFT, with
exact resume). GRPO always uses its own DDP loop. Distributed fragments compose via
`base: [../base.yaml, ../dist/fsdp2.yaml]`.

### Data formats (JSONL or `{hf: name, config:, split:, streaming:}`)

| Format | Record |
| --- | --- |
| pretrain | `{"text": "..."}` |
| SFT | `{"messages": [{"role": "system"\|"user"\|"assistant", "content": "..."}]}` (ends with assistant) |
| preference | `{"prompt": str or messages, "chosen": str or messages, "rejected": ...}` (UltraFeedback works as-is) |
| RL | `{"prompt"\|"question": "...", "answer": "..."}` (GSM8K `####` handled) or Countdown `{"target", "nums"}` |
| contrastive | `{"query": "...", "positive"\|"answer": "...", "negatives": [...]}` |

Overlength SFT/preference examples are rejected rather than silently truncated.
Exact train/eval overlap is rejected; semantic duplicates still need curation.

## Evaluating and analyzing behavior change

```bash
finetune generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain LoRA."
finetune generate --model outputs/e1/lora-r8 --prompt "Explain LoRA."      # PEFT dir: base read automatically
finetune eval ppl --model outputs/e6/replay0.05 --data heldout.jsonl
finetune eval reasoning --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --data gsm8k_test.jsonl \
    --samples 8 --temperature 0.6 --max-new-tokens 4096 --output samples.jsonl   # pass@k, think length, "wait" counts
finetune eval reasoning --model Qwen/Qwen3-1.7B --data gsm8k_test.jsonl --template-kwargs '{"enable_thinking": false}'
finetune eval retrieval --model outputs/embed/nq --data heldout_pairs.jsonl
finetune eval benchmarks --model outputs/e1/full --tasks gsm8k,mmlu         # lm-eval with the vendored model

finetune analyze weights    --base Qwen/Qwen2.5-1.5B --tuned outputs/e1/full   # per-layer/module |dW|, effective rank
finetune analyze kl         --base ... --tuned ... --data texts.jsonl           # token-level KL; most boosted/suppressed tokens
finetune analyze logit-lens --base ... --tuned ... --data prompt_answer.jsonl   # layer where the answer emerges
finetune analyze cka        --base ... --tuned ... --data texts.jsonl           # representation similarity per layer
finetune analyze router     --base ... --tuned ... --data texts.jsonl           # MoE expert-usage shift
```

Use `--output result.json` for the full tables (per parameter or per expert).

## Reproducibility and responsible use

Pin `model.revision` to a commit SHA for published results, keep data under the
gitignored `data/` folder, and record `pip freeze`. Seeds do not guarantee bitwise
determinism across GPUs or kernels. Check each model's and dataset's license
(Llama is gated; Qwen and DeepSeek have their own licenses) and remove personal or
secret data before training.
