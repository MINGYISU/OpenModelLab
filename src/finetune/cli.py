"""Command-line entry point.

  finetune prepare alpaca --tokenizer Qwen/Qwen2.5-1.5B-Instruct   # dataset recipe -> data/alpaca/*.jsonl
  finetune train configs/experiments/sft_qwen2.5_0.5b_lora.yaml --set train.lr=1e-4
  torchrun --nproc_per_node 4 -m finetune.cli train <config>
  finetune generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
  finetune eval <ppl|reasoning|retrieval|benchmarks> ...
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


def cmd_generate(args) -> None:
    from finetune.eval.generation import generate, load_for_inference

    model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype)
    outputs = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, chat=not args.raw,
                       template_kwargs=json.loads(args.template_kwargs))
    for prompt, output in zip(args.prompt, outputs):
        print(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False))


def cmd_eval(args) -> None:
    from finetune.eval import run_eval

    print(json.dumps(run_eval(args), indent=2))


def cmd_analyze(args) -> None:
    from finetune.eval.behavior import run_analysis

    print(json.dumps(run_analysis(args), indent=2))


def build_parser() -> argparse.ArgumentParser:
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
    ev.add_argument("task", choices=["ppl", "reasoning", "retrieval", "benchmarks"])
    _model_args(ev)
    ev.add_argument("--data", help="JSONL: text (ppl), prompt/answer (reasoning), query/positive (retrieval)")
    ev.add_argument("--text-field", default="text")
    ev.add_argument("--max-length", type=int, default=1024)
    ev.add_argument("--max-samples", type=int)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--samples", type=int, default=1, help="reasoning: completions per prompt (pass@k)")
    ev.add_argument("--temperature", type=float, default=0.0)
    ev.add_argument("--max-new-tokens", type=int, default=1024)
    ev.add_argument("--template-kwargs", default="{}")
    ev.add_argument("--pooling", default="mean")
    ev.add_argument("--tasks", default="gsm8k", help="benchmarks: comma-separated lm-eval tasks")
    ev.add_argument("--output", help="write per-example results to this JSONL")
    ev.set_defaults(func=cmd_eval)

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


def _single_process_gpu() -> None:
    """Without torchrun, Trainer would silently use legacy nn.DataParallel over all visible
    GPUs (which mis-scales token-normalised losses). Pin single-process runs to one GPU;
    use torchrun for multi-GPU (DDP/FSDP/DeepSpeed)."""
    import os

    if "LOCAL_RANK" in os.environ:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = (visible or "0").split(",")[0]


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _single_process_gpu()
    args.func(args)


if __name__ == "__main__":
    main()
