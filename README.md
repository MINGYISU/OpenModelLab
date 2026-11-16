# FineTune

A Python research starter for learning custom-dataset fine-tuning, embeddings,
NLP, and LLM post-training with Hugging Face models such as Qwen and
DeepSeek's Qwen-based distilled models.

## Setup

Use Python 3.11+ in a virtual environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
finetune validate --data examples/train.jsonl

# Install only when running models (large downloads, including PyTorch):
python -m pip install -e '.[ml]'
```

For CPU-only installations, run
`python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`
before installing the ML extra (see [PyTorch installation guidance](https://pytorch.org/get-started/locally/)).
Model commands download weights on first use; review each model's
license, access requirements, and hardware needs before downloading. Only
models supported natively by Transformers with safetensors weights are loaded;
remote Python model code is never enabled.

## Custom datasets

Provide UTF-8 JSONL: one conversation per line, with alternating `user` and
`assistant` messages and an optional initial `system` message. Every conversation
must end with an assistant answer. For example:

```json
{"messages":[{"role":"user","content":"What is LoRA?"},{"role":"assistant","content":"A parameter-efficient method that trains low-rank adapters."}]}
```

The original toy examples in `examples/train.jsonl` and `examples/eval.jsonl`
are smoke-test inputs, **not** a meaningful benchmark. Put your own data under
`data/` (gitignored), remove secrets and personal information, verify consent and
licensing, deduplicate, and split training/evaluation by source or document
before training. Validation reports malformed JSON and schema errors with line
numbers. Training rejects exact conversation overlap between train/evaluation
files; semantic duplicates still require manual dataset curation.

## Supervised post-training with LoRA

Start with a small instruction/chat checkpoint:

```bash
finetune train \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --data examples/train.jsonl \
  --eval-data examples/eval.jsonl \
  --output-dir outputs/qwen-lora \
  --epochs 1 --max-length 1024 --batch-size 1 \
  --gradient-accumulation-steps 8 --seed 42
```

For a larger DeepSeek experiment, substitute
`--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and use a separate output
directory. These are starting checkpoints, not a promise of compatibility with
every Qwen/DeepSeek architecture. Follow each model card's prompt conventions;
some reasoning models discourage system prompts, so remove those turns from
your dataset when appropriate.

The baseline freezes the base weights and trains rank-8 LoRA adapters on linear
layers using the tokenizer's native chat template. Loss covers **all non-padding
conversation tokens**, including prompts; this is not assistant-only masking.
Overlength conversations are rejected rather than silently truncating answers.
Padding labels are masked with `-100`, preserving real EOS labels. No external
experiment tracker is enabled.

The output contains the adapter, tokenizer, checkpoints, training/evaluation
loss metrics, and `experiment.json` with arguments, seed, library versions, and
resolved base-model commit when available. Existing nonempty output directories
are rejected. Supply `--revision <model-commit-sha>` to pin model/tokenizer
downloads; retain the exact dataset and environment (`pip freeze`) alongside
your experiment. Seeds alone do not guarantee identical results across devices.

This is single-device, unquantized training: LoRA reduces trainable parameters
but still loads the full base model. CUDA uses BF16 when supported, otherwise
FP32; CPU is supported for small smoke tests but can be very slow. Memory depends
on model size, sequence length, and batch size. QLoRA, distributed training, DPO,
and reinforcement-learning post-training are not implemented.

## Compare the base model and adapter

```bash
finetune generate --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain supervised fine-tuning."
finetune generate --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen-lora --prompt "Explain supervised fine-tuning."
```

Generation is greedy and prints only newly generated text. Always use the same
base model and revision used to train the adapter. Use held-out prompts to
compare task quality, factuality, safety, and forgetting; evaluation loss alone
does not establish an improvement.

## Embeddings and NLP experiments

```bash
finetune embed --model sentence-transformers/all-MiniLM-L6-v2 \
  --text "Fine-tuning adapts a language model." \
  --text "LoRA trains lightweight adapters."
```

The command prints a JSON array of L2-normalized vectors using attention-mask
aware mean pooling. It is a baseline for encoder models trained for mean pooling,
not a universal embedding recipe for arbitrary causal LLMs. Inputs are truncated
at `--max-length` (default 512); respect the checkpoint's own context limit.
Compare dot products as cosine similarity, then study retrieval, clustering,
and classification on held-out labeled data. Models requiring special pooling,
query prefixes, or custom modules need their model-specific implementation.

## Local checks

```bash
python -m unittest discover -s tests -v
finetune --help
finetune validate --data examples/eval.jsonl
```

Unit tests run without ML packages, GPUs, or network access. Real training and
embedding experiments require the optional ML stack and model weights.