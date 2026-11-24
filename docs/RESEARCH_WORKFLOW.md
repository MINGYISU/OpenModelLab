# Research workflow: from an architecture idea to a results table

This repo is meant to be the reusable part of every research project. A new project should only
need **new ideas**, not new plumbing. Everything an experiment needs fits into four building blocks:

| Block | What it is | Where it lives | You write |
| --- | --- | --- | --- |
| **Reference models** | open models (Qwen2/2.5/3, Llama, DeepSeek) as vendored, untouched source | `src/finetune/models/` | nothing |
| **Variants** (your models) | named architecture edits applied on top of a reference: add, remove, or replace components | `src/finetune/variants/<name>.py` | `modify(model, args)` |
| **Tasks** | a dataset with its preprocessing and its metrics | `src/finetune/data/recipes/<name>.py` | `convert(row)`, `METRICS` |
| **Studies** | the experiment matrix of a paper: models × datasets × seeds, plus eval and report settings | `configs/studies/<study>.yaml` | one YAML file |

`finetune study` runs the matrix: **train → evaluate → aggregate** into `report.md`,
`results.csv`, and `table.tex`. References are never modified, so they remain valid baselines
in every table.

```text
reference model ──(variant.modify)──► your model ──(train config)──► trained run ──(task eval)──► result record
        │                                                                                              │
        └──────────────── zero-shot or fine-tuned the same way (baselines) ────────────────────────────┤
                                                                                                       ▼
                                                                        finetune report → tables + significance
```

## Worked example

**Question:** *Does LayerScale (a learnable gain on each residual branch) help a 0.5B model
fine-tuned on Alpaca? Is the gain from the attention or the MLP branch?*
The complete study is [configs/studies/example_layerscale.yaml](../configs/studies/example_layerscale.yaml).

### 0. Check the pipeline on your machine (CPU, ~2 minutes, no downloads)

```bash
python -m unittest discover -s tests
finetune study configs/studies/smoke.yaml        # tiny random models; scores are meaningless
cat outputs/smoke/study_results/report.md
```

### 1. Prepare the data (once per dataset)

```bash
finetune prepare alpaca --peek 2                                          # does the conversion look right?
finetune prepare alpaca --tokenizer Qwen/Qwen2.5-0.5B-Instruct --max-length 1024
```

The recipe also says how to score the dataset (`METRICS = ["rouge_l", "f1"]` for Alpaca). For a new
dataset see [ADDING_A_DATASET.md](ADDING_A_DATASET.md). Use built-in metrics when they fit; only
write `metric()` in the recipe when the dataset truly needs its own.

### 2. Write your model as a variant

```bash
cp src/finetune/variants/_template.py src/finetune/variants/my_model.py
```

A variant names your model (`TITLE`, used in tables), lists the reference families it supports,
declares its knobs (`ARGS`, for ablations), and edits a built reference model in `modify()`:

```python
NAME = "layerscale"            # used in configs: model.variant: layerscale
TITLE = "LayerScale-LM"        # the name in your paper and tables
FAMILIES = ("qwen2", "qwen3", "llama")
ARGS = {"init": 1.0, "branches": ["attn", "mlp"]}

def modify(model, args):
    for layer in layers(model):
        layer.mlp.layer_scale = LayerScale(model.config.hidden_size, args["init"], **like(layer.mlp.down_proj.weight))
        layer.mlp.register_forward_hook(_scale_mlp)
        ...
```

Three examples cover the common edits:

| Variant | Edit | Pattern |
| --- | --- | --- |
| [layerscale.py](../src/finetune/variants/layerscale.py) | **add** a per-channel gain on residual branches | new submodule + forward hook; identity init keeps pretrained behaviour |
| [no_glu.py](../src/finetune/variants/no_glu.py) | **replace** SwiGLU with a plain MLP (optionally parameter-matched) | swap a submodule for your own class |
| [drop_layers.py](../src/finetune/variants/drop_layers.py) | **remove** decoder layers | rebuild the layer list and update the config |

To change how a layer computes (for example parallel attention + MLP), subclass the family's
decoder layer with a new `forward()` inside your variant; the template shows how. Never edit
`src/finetune/models/`: the tests pin those files to upstream behaviour.

Check the variant before spending GPU time:

```bash
finetune variants                                   # lists variants (and validates them)
finetune variants my_model --model tiny:qwen2       # builds it, runs a forward pass, prints the diff
finetune variants my_model --model Qwen/Qwen2.5-0.5B-Instruct --args '{"init": 0.1}'
finetune train configs/smoke/sft.yaml --set model.variant=my_model --set lora.enabled=false   # 4-step CPU run
```

What happens at build time: the reference is built (pretrained weights, `model.init: scratch`,
or `tiny:<family>`), `modify()` runs, and the new/removed parameters are printed. The variant spec
(name, args, source hash, changed config values) is written into the checkpoint's `config.json`,
and the variant source is archived as `variant_source.py` in the run directory. A trained variant
therefore loads like any checkpoint (`finetune eval/generate --model <run_dir>`), and a warning is
printed if the variant code changed since training.

### 3. Describe the study

```yaml
name: example-layerscale
train_config: sft_full.yaml          # ONE training recipe for every model: the comparison is fair
launcher: scripts/train.sh           # all visible GPUs
seeds: [1, 2, 3]
datasets: [alpaca]
models:
  - {label: Qwen2.5-0.5B-Instruct, path: Qwen/Qwen2.5-0.5B-Instruct, train: false}  # zero-shot reference
  - {label: Qwen2.5-0.5B + SFT, path: Qwen/Qwen2.5-0.5B-Instruct}                   # the baseline
  - {label: LayerScale-LM, path: Qwen/Qwen2.5-0.5B-Instruct, variant: layerscale}
  - {label: LayerScale-LM (attn only), path: Qwen/Qwen2.5-0.5B-Instruct, variant: layerscale,
     variant_args: {branches: [attn]}}                                              # ablation
eval: {max_samples: 500, max_new_tokens: 256}
report: {baseline: Qwen2.5-0.5B + SFT, percent: true}
```

Per-model `set: [...]` adds config overrides (e.g. a different learning rate for a model that
needs it; report such differences in the paper). Datasets can also be given as files:
`{name: x, train: path, eval: path, metrics: [f1]}`.

### 4. Run it

```bash
finetune study configs/studies/example_layerscale.yaml --dry-run   # the plan + every command
finetune study configs/studies/example_layerscale.yaml
```

Finished steps are skipped, so an interrupted study resumes, and adding a model, seed, or dataset
later runs only the new cells. `--rerun` forces everything again; `--report-only` just rebuilds the
tables. On a cluster, copy the commands from `--dry-run` into job scripts (each is independent
once its training step is done), then run `--report-only` at the end.

The steps are also usable on their own:

```bash
finetune train configs/studies/sft_full.yaml --set model.path=Qwen/Qwen2.5-0.5B-Instruct \
    --set model.variant=layerscale --set data.train=data/alpaca/train.jsonl --set run.output_dir=outputs/ls
finetune eval task --recipe alpaca --model outputs/ls --label LayerScale-LM --results results/my_study
finetune report results/my_study --baseline "Qwen2.5-0.5B + SFT" --percent --out results/my_study
```

### 5. Read the results

`results/example-layerscale/report.md` contains:

- **Main table**: one row per model (references, then fine-tuned references, then your variants),
  one column per dataset/metric, `mean ± std` over seeds, best in bold, `(+x)` difference to the
  baseline, and `†` when a paired bootstrap test over the same eval examples gives p < 0.05.
- **Models**: kind, variant and args, parameter count, seeds.
- **Details**: every number with seed count, 95% bootstrap CI (single-seed runs), and N.

`table.tex` is the same table in booktabs form for the paper; `results.csv` is long format for
plots (pandas/seaborn). Per-example predictions are in `<dataset>/<model>__seed<k>.predictions.jsonl`
for error analysis.

### 6. Look inside the models

The behaviour tools compare any two checkpoints, including a variant and its baseline:

```bash
finetune analyze weights    --base Qwen/Qwen2.5-0.5B-Instruct --tuned outputs/studies/.../seed1
finetune analyze kl         --base <baseline run> --tuned <variant run> --data texts.jsonl
finetune analyze logit-lens --base <baseline run> --tuned <variant run> --data prompt_answer.jsonl
finetune eval ppl --model <run> --data heldout.jsonl --results results/my_study   # extra columns for the report
```

`analyze weights` compares parameters that have the same name and shape in both models, so for
variants it covers the shared components (after `drop_layers`, layer indices shift).

## Making the comparison sound

| Rule | How the tooling helps |
| --- | --- |
| Same data, training recipe, and budget for every model | one `train_config` per study; differences must be explicit `set:` entries |
| Several seeds, report spread | `seeds: [1, 2, 3]` → `mean ± std`; single-seed results get bootstrap CIs |
| Same eval examples for every model | `eval.max_samples` takes the same first N examples; enables paired tests |
| Is the difference real? | `†` = paired bootstrap p < 0.05 against `report.baseline` |
| Control for size | the Params column; parameter-matched controls (e.g. `no_glu` with `match_params: true`) |
| Include the obvious baselines | zero-shot reference (`train: false`) and reference + same fine-tuning |
| Ablate your components | one variant with `ARGS` knobs; one model entry per setting |
| Don't select on the test set | tune on the eval split, then add a held-out test file as a second dataset for the final table |
| Reproducibility | `experiment.json` (git SHA, versions, GPUs, data hashes), `resolved_config.json`, `variant_source.py`; pin `model.revision` |

## Other setups

- **Pre-training a variant from scratch**: use a pretraining `train_config` (see
  `configs/experiments/pretrain_*`) with `model.init: scratch` in it; the reference config defines
  the base architecture, `CONFIG` in the variant can override sizes.
- **LoRA instead of full fine-tuning**: set `lora.enabled: true` in the train config. Parameters
  your variant adds are trained too and saved next to the adapter (`variant_state.safetensors`).
  Prefer full fine-tuning when the variant changes existing computations.
- **Other evaluations**: `finetune eval reasoning|ppl|retrieval|benchmarks ... --results DIR --label NAME`
  also writes result records, so they appear in the same report.
- **Scoring outside the pipeline**: `finetune score preds.jsonl --metrics f1,rouge_l`, or
  `from finetune.eval.metrics import compute_metrics` in a notebook.
