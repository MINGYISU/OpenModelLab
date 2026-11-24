# Research Plan: How Training Changes Open LLM Behavior

## 1. Goal

Build a paper-style research codebase for **pre-training, continued pre-training,
embedding training, and post-training** (SFT, DPO, GRPO) of open models (Qwen,
DeepSeek, Llama-style, and reasoning models), then measure **how each training
stage changes model behavior**.

Two parallel outcomes:

1. **Engineering literacy**: understand and own every layer, including model
   source, tokenizer/processor, data pipeline, loss, optimizer/scheduler,
   LoRA, GradCache, distributed training, and evaluation.
2. **Scientific findings**: controlled experiments that answer questions like
   "what does SFT change vs. DPO vs. RL?", "where in the network does change
   happen?", and "what is forgotten?"

### Design principles

- **Fresh codebase**: the current `src/finetune/` files are starter code and will
  be **replaced**, not extended. We keep only useful ideas from them, such as
  rejecting train/eval overlap, rejecting silent truncation, and recording
  `experiment.json`.
- **Vendored model code**: copy model source (`modeling_*.py`,
  `configuration_*.py`) into `src/finetune/models/` and instantiate our classes
  directly. We still use the `transformers` library for `PreTrainedModel`,
  weight loading, tokenizers, `Trainer`, and generation, plus `peft` and
  `accelerate`. No `AutoModel*`, and no `trust_remote_code`.
- **Config-driven experiments**: every run is a YAML config plus a seed, and the
  resolved config is saved with the outputs.
- **Small first, then scale**: every pipeline must first pass on a tiny
  random-init model on CPU or 1 GPU, then run on 0.5B–7B models across
  single-node multi-GPU and multi-node setups.

---

## 2. Repository layout (as implemented)

```text
src/finetune/
  models/                     # vendored, pinned model source (scripts/vendor_models.py)
    qwen2/ qwen3/ llama/ deepseek_v2/ deepseek_v3/ glm4/ glm4_moe/ qwen2_5_vl/ siglip2/ janus/   # configuration_*.py + modeling_*.py
    registry.py                # family -> classes; build_model() without AutoModel; tiny configs
    VENDORED.md                # upstream version, rules, local-edit log
  data/
    io.py                      # JSONL / HF datasets (incl. streaming), overlap check, replay mixing
    tokenizer.py               # tokenizer loading + tiny offline byte tokenizer
    pretrain.py                # tokenize + pack, streaming packed IterableDataset
    sft.py                     # chat templates, full vs. assistant-only masks
    preference.py  rl.py  contrastive.py  collators.py
  training/
    common.py                  # ResearchTrainer (our optimizer/schedule), TrainingArguments, setup
    loop.py                    # raw PyTorch DDP loop (no_sync, token-weighted loss, exact resume)
    causal_lm.py               # pretrain / continued pretrain / SFT
    dpo.py  grpo.py  embedding.py (GradCache)  rewards.py
    lora.py                    # PEFT + from-scratch LoRALinear
    optim.py                   # AdamW param groups, warmup + constant/linear/cosine/WSD
  eval/
    generation.py  perplexity.py  reasoning.py  embeddings.py  benchmarks.py
    behavior/                  # weight_diff, kl_drift, logit_lens, representation (CKA), moe_router
  config.py                    # typed YAML config (dataclasses, multi-base inheritance, --set overrides)
  cli.py                       # finetune {train,generate,eval,analyze}
configs/  base.yaml  smoke/  experiments/  dist/ (ddp, fsdp2, zero2, zero3)
scripts/  train.sh  train_multinode.sh  train_multinode.sbatch  vendor_models.py
tests/    CPU-only unit + end-to-end tests on tiny random models
docs/     RESEARCH_PLAN.md  notes/
```

Linear probes (Section 5, item 5) and analysis notebooks are not implemented yet.

---

## 3. Phased roadmap

Each phase ends with a **deliverable**, a **test**, and a **short learning
note** in `docs/notes/`.

### Phase 0: New skeleton (week 1)

- Delete the starter modules and tests, and create the layout above.
- Rewrite `pyproject.toml` with optional extras: `ml` (torch, transformers,
  peft, accelerate, datasets), `rl` (trl, vllm), `dist` (deepspeed), `eval`
  (lm-eval, mteb), and `quant` (bitsandbytes).
- Add typed YAML configs and a single CLI entry point.
- Add run metadata: git SHA, resolved config, dataset hash, library versions,
  GPU info, world size, and model revision.
- Add logging (TensorBoard by default; W&B opt-in).
- Add a tiny-model fixture (2 layers, hidden size 64) for fast CPU tests.
- **Deliverable**: `finetune sft configs/experiments/smoke.yaml` runs end to end
  on CPU with a tiny random model.

### Phase 1: Vendor model source code (weeks 1–2)

1. Copy `configuration_*.py` and `modeling_*.py` from the **pinned**
   `transformers` release (Apache-2.0) for `qwen2`, `qwen3`, `llama`, and
   `deepseek_v3`. Keep license headers and record the upstream path and version
   in `VENDORED.md`.
2. Rewrite relative imports (`from ...modeling_utils` →
   `from transformers.modeling_utils`). Keep the code otherwise identical at
   first, so later diffs show only our own edits.
3. Load weights through our classes:
   `Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`. Our class
   subclasses `PreTrainedModel`, so safetensors loading and `Trainer`
   compatibility still work.
4. **Equivalence test** (critical): for each family, compare logits from our
   class with upstream `transformers` on fixed inputs (`atol≈1e-5` in fp32).
   Run on a tiny random config in CI and on real checkpoints manually.
5. Annotate the code for learning: RMSNorm, RoPE (and YaRN scaling), GQA, the
   SwiGLU MLP, the KV cache, Qwen3 QK-norm, and DeepSeek MLA (low-rank KV
   compression, decoupled RoPE) plus MoE routing (shared and routed experts,
   aux-loss-free load balancing, and multi-token prediction in V3).
6. Add research hooks: optional return of per-layer hidden states, attention,
   MLP activations, and router logits (used in Section 5).

| Family | Checkpoints for experiments | Notes |
| --- | --- | --- |
| Qwen2/2.5 | `Qwen2.5-0.5B/1.5B/7B` (base + instruct), `Qwen2.5-Math-1.5B` | main workhorse |
| DeepSeek-R1-Distill | `DeepSeek-R1-Distill-Qwen-1.5B/7B` | Qwen2 architecture, reasoning distilled |
| Qwen3 | `Qwen3-0.6B/1.7B/4B/8B` | hybrid thinking via `enable_thinking` |
| Llama-style | `SmolLM2-360M/1.7B`, `Llama-3.2-1B` | Llama is license-gated; SmolLM2 is open |
| DeepSeek V2/V3 (MoE+MLA) | `DeepSeek-V2-Lite` (16B total, 2.4B active) or a **small random-init V3 config** | full V3/R1 (671B) is impractical; study the architecture at small scale |
| Reasoning | R1-Distill-Qwen, Qwen3 (thinking), `QwQ-32B` (inference/analysis only) | see Phase 7 reasoning track |

**Deliverable**: `models/registry.py` plus passing logit-equivalence tests.

### Phase 2: Tokenizers, processors, and data pipelines (week 2)

- **Tokenizer study**: BPE merges, special tokens, chat templates
  (`<|im_start|>`, `<think>` tags), and `add_generation_prompt`.
- **Pre-training data**: streaming from HF `datasets` (for example, a
  FineWeb-Edu sample), tokenization, document packing into fixed blocks with
  EOS separators, and optional document-aware attention masks.
- **SFT data**: chat-template rendering with **configurable loss masking**
  (full sequence vs. assistant-only via `{% generation %}` masks or offset
  mapping), sequence-length bucketing, and packing.
- **Preference data**: `(prompt, chosen, rejected)` (for example,
  UltraFeedback-binarized).
- **RL data**: prompts with verifiable answers (GSM8K, MATH, Countdown).
- **Contrastive data**: `(query, positive, hard negatives)` (for example, MS
  MARCO and NQ).
- **Data hygiene**: exact and near-duplicate removal, train/eval overlap checks,
  contamination checks against benchmarks (n-gram overlap), and no silent
  truncation.
- **Deliverable**: unit tests that show exact label masks for each format.

### Phase 3: Training core and setup (weeks 3–4)

Implement and document the mechanics papers describe:

- **Optimizer**: AdamW, no weight decay on norms, biases, or embeddings
  (parameter groups), beta/epsilon choices, and gradient clipping.
- **Schedules**: linear warmup → cosine, constant, and WSD
  (warmup-stable-decay). Define steps vs. epochs, `max_steps`, and
  `warmup_ratio`.
- **Batching**: global batch = per-device × grad-accum × world size; track
  tokens per step.
- **Precision**: bf16 mixed precision, fp32 master weights, and TF32.
- **Memory**: gradient checkpointing, SDPA/FlashAttention, and Liger kernels
  (optional).
- **Checkpointing**: save and resume model, optimizer, scheduler, data loader,
  and RNG state.
- **Logging**: loss, LR, grad norm, tokens/sec, MFU, and peak memory.
- Write **one minimal raw PyTorch loop** (`training/loop.py`) to learn what
  `Trainer` hides, then use `transformers.Trainer` (with custom `compute_loss`
  where needed) for the main pipelines.
- **LoRA**: write a ~50-line `LoRALinear` from scratch, verify it matches PEFT
  outputs, and then use PEFT. Compare rank, alpha, target modules,
  DoRA/rsLoRA, and QLoRA (4-bit).

### Phase 4: Distributed training (weeks 4–5)

- **Single-node** multi-GPU with `torchrun --nproc_per_node=N`, then
  **multi-node** with `--nnodes`, rendezvous, and a SLURM launcher script.
- Use strategies in this order of learning: **DDP** → **FSDP2** (full/hybrid
  shard, mixed-precision policy, CPU offload) → **DeepSpeed ZeRO-2/3** through
  `accelerate` configs.
- Concepts to document: rank/local_rank/world_size, `DistributedSampler`,
  all-reduce vs. reduce-scatter/all-gather, NCCL env vars, gradient sync with
  grad accumulation (`no_sync`), per-rank seeding, and rank-0-only logging and
  saving.
- For MoE, document expert parallelism conceptually and keep runs within FSDP
  at small scale.
- **Deliverable**: the same config gives matching loss curves on 1 GPU vs.
  N GPUs at equal global batch size, plus a scaling-efficiency table
  (1/2/4/8 GPUs, multi-node).

### Phase 5: Pre-training and continued pre-training (weeks 5–7)

- **From scratch**: a ~100M–300M Qwen-architecture model trained on a few
  billion tokens. Study loss vs. tokens, LR/warmup sensitivity, and a small
  scaling-law sweep (3–4 model sizes).
- **Continued pre-training (CPT)**: adapt a base model (for example,
  `Qwen2.5-1.5B`) to a domain corpus (code, math, or a language). Measure domain
  perplexity gain vs. general-benchmark forgetting, and test replay mixing
  ratios (0%, 5%, 25% general data).

### Phase 6: Embedding training (weeks 7–8)

- Turn an LLM into an embedding model, compare pooling methods (mean,
  last-token/EOS, and latent attention), and add instruction prefixes for
  queries.
- Use an InfoNCE contrastive loss with in-batch negatives plus hard negatives,
  and test the effect of temperature.
- **GradCache**: implement the two-pass approach (compute reps without grad,
  get gradient w.r.t. reps, then re-forward in chunks with backward). It enables
  large effective contrastive batches on limited memory. Verify that gradients
  match a non-cached run on a small batch.
- **Distributed contrastive loss**: all-gather embeddings across ranks
  (`torch.distributed.nn.all_gather` to keep gradients) for global negatives,
  combined with GradCache.
- Evaluate with MTEB subsets (retrieval, STS, clustering), and run ablations
  over batch size (with/without GradCache), pooling, and LoRA vs. full.

### Phase 7: Post-training (weeks 8–12)

| Method | Implementation | Key studies |
| --- | --- | --- |
| **SFT (LoRA + full)** | `Trainer` with our data/masking | full vs. LoRA rank; full-sequence vs. assistant-only loss; data size scaling; packing |
| **DPO** (+ IPO/SimPO variants) | own loss implementation, cross-checked against TRL `DPOTrainer` | β sweep; reference model choice; chosen/rejected log-prob dynamics ("likelihood displacement") |
| **GRPO / RLVR** | own minimal loop first, then TRL `GRPOTrainer` with vLLM generation | reward design (correctness + format); group size; KL coefficient; response length growth over training |
| **Reasoning distillation** | SFT on long chain-of-thought traces from R1/QwQ | distill vs. RL on a small model (R1 paper-style comparison) |

**Reasoning-model track** (how reasoning models work):

- Study the anatomy of `<think>` traces: length, backtracking/"wait" tokens,
  self-verification, and answer extraction.
- Compare **R1-Distill-Qwen-1.5B** with **Qwen2.5-Math-1.5B** (same
  architecture, different post-training), and Qwen3 with thinking on vs. off.
- Reproduce a small **"R1-Zero"-style run**: GRPO on a base model with a
  Countdown/GSM8K verifiable reward, and watch for emerging longer traces and
  self-checking behavior.
- Measure think-token budget vs. accuracy (test-time scaling curves, budget
  forcing).
- Use mechanistic probes: does the hidden state "know" the answer before the
  think trace ends? (logit lens and linear probes on intermediate steps)

---

## 4. Evaluation suite

- **Intrinsic**: held-out perplexity (in-domain and general) and eval loss.
- **Benchmarks** (via `lm-evaluation-harness` plus our own scripts): MMLU,
  HellaSwag, ARC, GSM8K, MATH-500, AIME, HumanEval/MBPP, IFEval, and
  TruthfulQA.
- **Reasoning**: answer extraction, pass@1/pass@k with sampling, average
  response length, and think-length vs. correctness.
- **Embeddings**: MTEB subsets.
- **Safety and alignment**: refusal rate on harmful/benign prompt sets, and
  over-refusal.
- **Statistics**: ≥3 seeds for key results, bootstrap confidence intervals, and
  fixed decoding configs. Report means ± std.

---

## 5. Behavior-change analysis

This is the core research contribution. For each (base → trained) model pair:

1. **Weight space**: per-layer/per-module ‖ΔW‖, relative change, effective
   rank of ΔW (SVD), and which modules change most (attention vs. MLP, early vs.
   late layers). Compare full FT with LoRA.
2. **Representation space**: CKA/cosine similarity of hidden states per layer
   between base and tuned models on fixed probe inputs.
3. **Output distribution**: token-level KL(tuned‖base) on held-out text. Find
   *which tokens* shift most (style tokens vs. content tokens) to test the
   "superficial alignment" hypothesis.
4. **Logit lens / tuned lens**: when does the final answer emerge across
   layers, and does training move it earlier?
5. **Linear probes**: does knowledge or a concept stay linearly decodable after
   fine-tuning, even when generation changes?
6. **Forgetting**: general-benchmark deltas vs. in-domain gains (CPT, SFT, and
   LoRA vs. full).
7. **Generation behavior**: length, format, refusal, sycophancy, diversity
   (distinct-n, self-BLEU), and calibration.
8. **MoE-specific**: router entropy and expert utilization shifts after
   fine-tuning.

---

## 6. Experiment matrix (first research questions)

| ID | Question | Setup | Main metrics |
| --- | --- | --- | --- |
| E1 | How much does SFT change weights vs. outputs? | Qwen2.5-1.5B, full vs. LoRA r∈{8,64}, same data | ΔW, KL drift, task score, forgetting |
| E2 | Does loss masking matter? | full-sequence vs. assistant-only | task score, prompt-copying behavior |
| E3 | Data scaling in SFT | 1k / 10k / 100k examples | score curves, forgetting |
| E4 | SFT vs. DPO vs. GRPO on the same base | matched compute budget | reasoning accuracy, length, KL, diversity |
| E5 | Distillation vs. RL for reasoning | 1.5B base: SFT on R1 traces vs. GRPO | MATH-500, think length, self-verification rate |
| E6 | CPT forgetting and replay | 0/5/25% replay | domain PPL vs. MMLU |
| E7 | Contrastive batch size via GradCache | 64 → 4096 effective batch | MTEB retrieval nDCG@10 |
| E8 | Thinking mode effect | Qwen3 think on/off, budget sweep | accuracy vs. tokens |
| E9 | Architecture effect | Qwen2 vs. Llama vs. MoE (small) under identical SFT | ΔW distribution, router shifts |
| E10 | Where does reasoning live? | R1-Distill vs. Qwen2.5-Math (same arch) | ΔW by layer, logit-lens answer emergence |

---

## 7. Engineering practices

- **CPU unit tests** on tiny random models for: logit equivalence, label masks,
  packing, LoRA-from-scratch parity, GradCache gradient parity, DPO loss values,
  reward functions, and schedule shapes.
- **GPU smoke tests** (script): 20 steps per pipeline on 1 GPU, N GPUs, and
  2 nodes.
- **Reproducibility**: pinned dependencies, pinned model revisions, dataset
  hashes, and saved resolved configs for every run.
- **Data and licensing**: check model licenses (Llama gated, DeepSeek MIT, and
  Qwen Apache/Qwen license) and dataset licenses. Keep data and outputs in
  gitignored `data/` and `outputs/` folders.

---

## 8. Reading list (paired with phases)

- Architecture: *Attention Is All You Need*; RoPE (RoFormer); GQA; the Qwen2,
  Qwen2.5, and Qwen3 technical reports; DeepSeek-V2 (MLA, DeepSeekMoE);
  DeepSeek-V3.
- Pre-training: Chinchilla scaling laws; MiniCPM (WSD schedule); FineWeb.
- PEFT: LoRA, QLoRA, DoRA, and *LoRA Learns Less and Forgets Less*.
- Embeddings: Sentence-BERT, E5, GTE, NV-Embed, and **GradCache**
  (Gao et al., 2021).
- Post-training: InstructGPT, LIMA (superficial alignment), Tülu 3, DPO, SimPO,
  DeepSeekMath (**GRPO**), DeepSeek-R1, *s1: Simple test-time scaling*, and
  *SFT Memorizes, RL Generalizes*.
- Distributed: PyTorch FSDP paper, ZeRO, and Megatron-LM.
- Analysis: logit lens/tuned lens, CKA (Kornblith et al.), and model-diffing
  work.

---

## 9. Milestones checklist

- [x] M0: new skeleton, config-driven CLI, logging, tiny-model test fixture
- [x] M1: vendored Qwen2/Qwen3/Llama/DeepSeek-V2/V3 with logit-equivalence tests (exact on real checkpoints)
- [x] M2: data pipelines with masking/packing tests
- [x] M3: raw PyTorch loop plus `Trainer` path; schedules; LoRA-from-scratch
- [x] M4 (code): DDP / FSDP2 / ZeRO-2 / ZeRO-3 verified on 2 GPUs; multi-node launchers written
- [ ] M4 (results): 1-vs-N-GPU loss-curve match and scaling-efficiency table; multi-node run
- [ ] M5: small pre-training run and CPT forgetting study (E6); configs ready
- [ ] M6: embedding training with GradCache (E7); pipeline verified, sweep not run
- [ ] M7: SFT/DPO/GRPO experiments E1–E5, E8, E10; pipelines verified, experiments not run
- [ ] M8: linear probes, analysis notebooks, and the write-up with figures
