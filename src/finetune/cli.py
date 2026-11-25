"""Command-line entry point.

  finetune prepare alpaca --tokenizer Qwen/Qwen2.5-1.5B-Instruct   # dataset recipe -> data/alpaca/*.jsonl
  finetune train configs/experiments/sft_qwen2.5_0.5b_lora.yaml --set train.lr=1e-4
  torchrun --nproc_per_node 4 -m finetune.cli train <config>
  finetune generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
  finetune eval <task|ppl|reasoning|retrieval|benchmarks> ... [--results results/<study>]
  finetune score preds.jsonl --metrics exact_match,f1               # metrics as a standalone tool
  finetune report results/<study> --baseline <label> --percent      # tables: markdown, CSV, LaTeX
  finetune variants [<name> --model tiny:qwen2]                     # your architectures (src/finetune/variants/)
  finetune study configs/studies/<study>.yaml                       # models x datasets x seeds, end to end
  finetune analyze <weights|kl|logit-lens|cka|router> --base A --tuned B ...
"""

from __future__ import annotations

import argparse
import json

STAGES = {
    "pretrain": "finetune.training.causal_lm",
    "sft": "finetune.training.causal_lm",
    "dpo": "finetune.training.dpo",
    "grpo": "finetune.training.grpo",
    "embed": "finetune.training.embedding",
}


def cmd_train(args) -> None:
    import importlib

    from finetune.config import dump_config, load_config
    from finetune.utils import is_main

    config = load_config(args.config, args.set)
    if config.train.stage not in STAGES:
        raise SystemExit(f"Unknown train.stage {config.train.stage!r}; choose from {sorted(STAGES)}")
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, default=str))
        return
    metrics = importlib.import_module(STAGES[config.train.stage]).run(config)
    if is_main():
        dump_config(config, f"{config.run.output_dir}/resolved_config.json")
        print(json.dumps(metrics, indent=2, default=str))


def cmd_prepare(args) -> None:
    import itertools

    from finetune.data.recipes import available_recipes, get_recipe, iter_rows, prepare, recipe_name

    if args.list or not args.recipe:
        for name, module in sorted(available_recipes().items()):
            print(f"{name:20s} {module.FORMAT:12s} {(module.__doc__ or '').strip().splitlines()[0]}")
        return
    recipe = get_recipe(args.recipe)
    if args.peek:
        for row in itertools.islice(iter_rows(recipe, "train"), args.peek):
            print(json.dumps({"raw": row, "converted": recipe.convert(row)}, ensure_ascii=False, default=str, indent=2))
        return
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = prepare(recipe, args.out_dir or f"data/{recipe_name(recipe)}", eval_ratio=args.eval_ratio,
                    max_samples=args.max_samples, seed=args.seed, tokenizer=tokenizer,
                    max_length=args.max_length, template_kwargs=json.loads(args.template_kwargs))
    print(json.dumps(stats, indent=2))


def _model_args(parser) -> None:
    parser.add_argument("--model", required=True, help="hub id, local dir, or tiny:<family>")
    parser.add_argument("--revision")
    parser.add_argument("--adapter", help="PEFT adapter directory to apply on top of --model")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--inference-backend", choices=("transformers", "vllm"), default="transformers",
                        help="text generation backend (vLLM supports standard checkpoints only)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="number of GPUs for vLLM tensor parallelism")


def cmd_generate(args) -> None:
    from finetune.eval.generation import generate, load_for_inference

    model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype,
                                          backend=args.inference_backend,
                                          tensor_parallel_size=args.tensor_parallel_size)
    outputs = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, chat=not args.raw,
                       template_kwargs=json.loads(args.template_kwargs), backend=args.inference_backend)
    for prompt, output in zip(args.prompt, outputs):
        print(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False))


def cmd_eval(args) -> None:
    from finetune.eval import run_eval

    print(json.dumps(run_eval(args), indent=2))


def cmd_score(args) -> None:
    from finetune.data.io import load_records
    from finetune.eval.metrics import compute_metrics

    records = load_records(args.predictions)
    result = compute_metrics([r[args.pred_field] for r in records], [r[args.ref_field] for r in records],
                             args.metrics.split(","), extract=args.extract, bootstrap=args.bootstrap)
    result.pop("per_example")
    print(json.dumps(result, indent=2))


def cmd_report(args) -> None:
    from pathlib import Path

    from finetune.eval.report import build_report
    from finetune.eval.results import load_results

    report = build_report(load_results(args.results), metrics=args.metrics.split(",") if args.metrics else None,
                          baseline=args.baseline, percent=args.percent, title=args.title)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report["markdown"], encoding="utf-8")
        (out / "results.csv").write_text(report["csv"], encoding="utf-8")
        (out / "table.tex").write_text(report["latex"], encoding="utf-8")
    print(report["latex"] if args.latex else report["markdown"])


def cmd_variants(args) -> None:
    from finetune.variants import available_variants

    if not args.variant:
        for name, module in sorted(available_variants().items()):
            summary = (module.__doc__ or "").strip().splitlines()[0]
            print(f"{name:16s} {','.join(module.FAMILIES):20s} {summary}")
        return
    import torch

    from finetune.config import ModelConfig
    from finetune.models import build_model

    reference = build_model(ModelConfig(path=args.model, dtype="float32"))
    model = build_model(ModelConfig(path=args.model, dtype="float32", variant=args.variant,
                                    variant_args=json.loads(args.args)))
    spec = model.config.finetune_variant
    ids = torch.randint(0, min(model.config.vocab_size, 256), (1, 8))
    with torch.no_grad():
        model.eval()(ids)
    print(json.dumps({"title": spec["title"], "args": spec["args"],
                      "reference_parameters": sum(p.numel() for p in reference.parameters()),
                      "variant_parameters": sum(p.numel() for p in model.parameters()),
                      "new_parameters": spec["new_parameters"], "removed_parameters": spec["removed_parameters"],
                      "changed_config": {k: [v, getattr(model.config, k)] for k, v in spec["reference_config"].items()}},
                     indent=2, default=str))


def cmd_study(args) -> None:
    from finetune.study import run_study

    run_study(args.study, dry_run=args.dry_run, report_only=args.report_only, rerun=args.rerun)


def cmd_analyze(args) -> None:
    from finetune.eval.behavior import run_analysis

    print(json.dumps(run_analysis(args), indent=2))


def build_parser() -> argparse.ArgumentParser:
    from finetune.eval.metrics import EXTRACTORS as EXTRACTOR_NAMES
    from finetune.eval.metrics import METRICS as METRIC_NAMES

    parser = argparse.ArgumentParser(prog="finetune", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="run a training stage from a YAML config")
    train.add_argument("config")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    train.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    train.set_defaults(func=cmd_train)

    prep = sub.add_parser("prepare", help="convert a dataset with its recipe (src/finetune/data/recipes/)")
    prep.add_argument("recipe", nargs="?", help="recipe name or path to a recipe .py file")
    prep.add_argument("--list", action="store_true", help="list available recipes")
    prep.add_argument("--peek", type=int, default=0, metavar="N", help="print N raw rows and their conversion, write nothing")
    prep.add_argument("--out-dir", help="default: data/<recipe>")
    prep.add_argument("--eval-ratio", type=float, help="held-out fraction when the recipe has no EVAL_SOURCE")
    prep.add_argument("--max-samples", type=int, help="cap records per split (quick trials)")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--tokenizer", help="SFT: drop examples longer than --max-length under this chat template")
    prep.add_argument("--max-length", type=int, default=2048)
    prep.add_argument("--template-kwargs", default="{}", help='JSON, e.g. \'{"enable_thinking": false}\'')
    prep.set_defaults(func=cmd_prepare)

    gen = sub.add_parser("generate", help="generate completions (greedy unless --temperature > 0)")
    _model_args(gen)
    gen.add_argument("--prompt", action="append", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--raw", action="store_true", help="do not apply the chat template")
    gen.add_argument("--template-kwargs", default="{}", help='JSON, e.g. \'{"enable_thinking": false}\'')
    gen.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="evaluate a model")
    ev.add_argument("task", choices=["task", "ppl", "reasoning", "retrieval", "benchmarks"],
                    help="task: generate on a dataset's eval split and score it with its metrics")
    _model_args(ev)
    ev.add_argument("--recipe", help="task: dataset recipe (prompts, references, METRICS); data defaults to "
                                     "data/<recipe>/eval.jsonl")
    ev.add_argument("--data", help="JSONL: text (ppl), prompt/answer (reasoning, task), messages (task), "
                                   "query/positive (retrieval)")
    ev.add_argument("--metrics", help="task: comma-separated metric names (default: the recipe's METRICS)")
    ev.add_argument("--extract", choices=sorted(EXTRACTOR_NAMES), help="task: prediction cleanup before scoring")
    ev.add_argument("--system", help="task: system prompt")
    ev.add_argument("--raw", action="store_true", help="task: do not apply the chat template")
    ev.add_argument("--bootstrap", type=int, default=1000, help="task: bootstrap samples for 95%% CIs (0 = off)")
    ev.add_argument("--text-field", default="text")
    ev.add_argument("--max-length", type=int, default=1024)
    ev.add_argument("--max-samples", type=int)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--samples", type=int, default=1, help="reasoning: completions per prompt (pass@k)")
    ev.add_argument("--temperature", type=float)
    ev.add_argument("--max-new-tokens", type=int, help="default: 256 (task, or the recipe's GENERATION), 1024 (reasoning)")
    ev.add_argument("--template-kwargs", default="{}")
    ev.add_argument("--pooling", default="mean")
    ev.add_argument("--tasks", default="gsm8k", help="benchmarks: comma-separated lm-eval tasks")
    ev.add_argument("--output", help="write per-example results to this JSONL")
    ev.add_argument("--results", metavar="DIR", help="store a result record here for `finetune report`")
    ev.add_argument("--label", help="model name in reports (default: variant title, run name, or model dir)")
    ev.add_argument("--dataset", help="dataset name in reports (default: recipe or data folder name)")
    ev.add_argument("--seed", type=int, help="training seed of this model (default: read from experiment.json)")
    ev.set_defaults(func=cmd_eval)

    score = sub.add_parser("score", help="compute metrics for a JSONL of predictions and references")
    score.add_argument("predictions", help="JSONL with prediction and reference fields")
    score.add_argument("--metrics", default="exact_match,f1", help=f"comma-separated: {', '.join(sorted(METRIC_NAMES))}")
    score.add_argument("--pred-field", default="prediction")
    score.add_argument("--ref-field", default="reference", help="string or list of acceptable strings")
    score.add_argument("--extract", default="strip_think", choices=sorted(EXTRACTOR_NAMES))
    score.add_argument("--bootstrap", type=int, default=1000)
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report", help="aggregate result records into tables (markdown, CSV, LaTeX)")
    report.add_argument("results", nargs="+", help="results directories or files")
    report.add_argument("--metrics", help="comma-separated metrics/columns to include, in order")
    report.add_argument("--baseline", help="model label to compare against (differences + significance)")
    report.add_argument("--percent", action="store_true", help="show [0, 1] scores as percentages")
    report.add_argument("--title", default="Results")
    report.add_argument("--out", help="write report.md, results.csv, table.tex to this directory")
    report.add_argument("--latex", action="store_true", help="print the LaTeX table instead of markdown")
    report.set_defaults(func=cmd_report)

    var = sub.add_parser("variants", help="list model variants, or build one and show what it changes")
    var.add_argument("variant", nargs="?", help="variant name or .py path")
    var.add_argument("--model", default="tiny:qwen2", help="reference to apply it to (default: tiny:qwen2)")
    var.add_argument("--args", default="{}", help='variant args as JSON, e.g. \'{"init": 0.1}\'')
    var.set_defaults(func=cmd_variants)

    study = sub.add_parser("study", help="run a study (models x datasets x seeds): train -> eval -> report")
    study.add_argument("study", help="study YAML (see configs/studies/)")
    study.add_argument("--dry-run", action="store_true", help="print the plan and commands only")
    study.add_argument("--report-only", action="store_true", help="only rebuild the report from existing results")
    study.add_argument("--rerun", action="store_true", help="re-run steps that already finished")
    study.set_defaults(func=cmd_study)

    an = sub.add_parser("analyze", help="compare a base and a trained model")
    an.add_argument("analysis", choices=["weights", "kl", "logit-lens", "cka", "router"])
    an.add_argument("--base", required=True)
    an.add_argument("--tuned", required=True, help="trained model dir (full model or PEFT adapter dir)")
    an.add_argument("--data", help="JSONL with a text field (kl, cka, router) or prompt/answer (logit-lens)")
    an.add_argument("--text-field", default="text")
    an.add_argument("--max-samples", type=int, default=64)
    an.add_argument("--max-length", type=int, default=512)
    an.add_argument("--dtype", default="float32")
    an.add_argument("--top", type=int, default=20)
    an.add_argument("--output", help="write the full result to this JSON file")
    an.set_defaults(func=cmd_analyze)
    return parser


def _single_process_gpu(preserve_visible_gpus: bool = False) -> None:
    """Without torchrun, Trainer would silently use legacy nn.DataParallel over all visible
    GPUs (which mis-scales token-normalised losses). Pin single-process runs to one GPU;
    use torchrun for multi-GPU training. vLLM inference manages visible GPUs itself."""
    import os

    if "LOCAL_RANK" in os.environ or preserve_visible_gpus:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = (visible or "0").split(",")[0]


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _single_process_gpu(preserve_visible_gpus=(
        getattr(args, "command", None) in {"generate", "eval"}
        and getattr(args, "inference_backend", "transformers") == "vllm"
    ))
    args.func(args)


if __name__ == "__main__":
    main()
