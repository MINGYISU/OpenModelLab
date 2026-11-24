# Adding a new dataset

Every dataset has its own layout: different column names, transcripts that need
parsing, labels that need mapping. The trainer reads only one standard format per
stage. Each dataset therefore gets a small **recipe**: one Python file that converts
its raw rows into that format. After that, every dataset goes through the same steps.

```text
raw dataset ──recipe.convert()──▶ data/<name>/train.jsonl, eval.jsonl ──config YAML──▶ finetune train
```

## Step 1. Write a recipe (the only dataset-specific code)

```bash
cp src/finetune/data/recipes/_template.py src/finetune/data/recipes/my_dataset.py
```

Fill in the `TODO`s. A recipe is a few constants and one function:

```python
NAME = "my_dataset"
FORMAT = "sft"                                   # sft | preference | rl | contrastive | pretrain
SOURCE = {"hf": "org/name", "split": "train"}    # or a local "raw/file.jsonl", or define load()
EVAL_SOURCE = {"hf": "org/name", "split": "test"}  # or None -> hold out EVAL_RATIO
EVAL_RATIO = 0.05

def convert(row):                                # one raw row -> one standard record (or None to skip)
    return {"messages": [{"role": "user", "content": row["question"]},
                         {"role": "assistant", "content": row["answer"]}]}
```

All dataset-specific logic goes in `convert()`: renaming fields, joining columns,
stripping HTML, parsing transcripts, mapping labels to text, and dropping bad rows
(`return None`). It may also return a list to turn one row into several records.
For CSV, Excel, folders of files, or anything that is not JSONL or on the Hugging Face Hub,
define `load(split)` and return dict rows (there is an example at the bottom of the template).

What `convert()` must return for each `FORMAT`:

| FORMAT | Record | Train with `train.stage` |
| --- | --- | --- |
| `sft` | `{"messages": [{"role": "system"\|"user"\|"assistant", "content": ...}, ...]}` ending with assistant | `sft` |
| `preference` | `{"prompt", "chosen", "rejected"}`, each a string or a list of messages | `dpo` |
| `rl` | `{"prompt": str, "answer": str}` (the answer is checked by a reward function) | `grpo` |
| `contrastive` | `{"query": str, "positive": str, "negatives": [str, ...]}` | `embed` |
| `pretrain` | `{"text": str}` | `pretrain` |

Worked examples: [alpaca.py](../src/finetune/data/recipes/alpaca.py) (simple column mapping)
and [hh_rlhf.py](../src/finetune/data/recipes/hh_rlhf.py) (parses raw transcripts into
preference pairs).

## Step 2. Check it, then run it

```bash
finetune prepare --list                     # your recipe should appear here
finetune prepare my_dataset --peek 3        # prints 3 raw rows next to their conversion; writes nothing
finetune prepare my_dataset --tokenizer Qwen/Qwen2.5-1.5B-Instruct --max-length 2048
```

This writes `data/my_dataset/train.jsonl`, `eval.jsonl`, and `prepare_stats.json`. For every
dataset, `prepare` also:

- **validates** each record against `FORMAT` (invalid ones are counted and skipped, and the first error is shown);
- **removes exact duplicates**, including training rows that also appear in eval;
- **drops overlong SFT examples** when `--tokenizer` is given. Training rejects any example
  longer than `data.max_length`, so use the tokenizer of the model you will train;
- **splits** off `EVAL_RATIO` (or `--eval-ratio`) when there is no `EVAL_SOURCE`.

Read the stats it prints. If most rows land in `skipped_by_recipe` or `invalid`, fix `convert()`.
Use `--max-samples 200` for a quick trial dataset.

## Step 3. Point a config at it and train

```bash
cp configs/templates/sft_my_dataset.yaml configs/my_dataset.yaml   # edit the lines marked <--
finetune train configs/my_dataset.yaml --dry-run                   # check the resolved config
finetune train configs/my_dataset.yaml --set train.max_steps=5     # quick sanity run
scripts/train.sh configs/my_dataset.yaml                           # the real run (all visible GPUs)

finetune generate --model outputs/my_dataset/lora --prompt "a prompt like your data"
tensorboard --logdir outputs/my_dataset/lora
```

For a non-SFT format, start from the matching config in `configs/experiments/` instead
(`e4_dpo_*` for preference, `e4_grpo_*` for rl, `embed_*` for contrastive,
`e6_cpt_*` for pretrain) and set `data.train` / `data.eval` to your files.

### Common config changes

| Situation | Change |
| --- | --- |
| Out of GPU memory | lower `train.per_device_batch_size`, raise `train.grad_accum`; set `model.gradient_checkpointing: true` |
| Small GPU (<16 GB) | smaller model, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| Eval loss rises while train loss falls | overfitting: fewer `train.epochs` |
| Full fine-tune instead of LoRA | `lora.enabled: false`, `train.lr: 1.0e-5` |
| Multi-turn Qwen3 data fails assistant masking | `data.loss_on: all` |

## Checklist for a new dataset

1. `cp _template.py <name>.py` and fill in `NAME`, `FORMAT`, `SOURCE`, and `convert()`.
2. `finetune prepare <name> --peek 3`: does the conversion look right?
3. `finetune prepare <name> --tokenizer <model>`: are the stats reasonable?
4. Copy a config, then set `data.train`/`data.eval` and `run.output_dir`.
5. Run `--dry-run`, then a 5-step run, then the real run.
