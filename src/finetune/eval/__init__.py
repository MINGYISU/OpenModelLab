"""Evaluation entry points used by `finetune eval`."""

from __future__ import annotations

import json
from pathlib import Path

from finetune.data.io import load_records
from finetune.utils import write_jsonl


def _load(args, **kwargs):
    from finetune.eval.generation import load_for_inference

    model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype, **kwargs)
    args.num_parameters = sum(p.numel() for p in model.parameters())
    return model, tokenizer


def _task(args) -> tuple[dict, list[dict], dict, str]:
    from finetune.data.recipes import get_recipe, recipe_name
    from finetune.eval.tasks import run_task, task_settings

    recipe = get_recipe(args.recipe) if args.recipe else None
    data = args.data or (f"data/{recipe_name(recipe)}/eval.jsonl" if recipe else None)
    if not data:
        raise SystemExit("eval task needs --recipe and/or --data")
    records = load_records(data, args.max_samples)
    settings = task_settings(recipe, records, args)
    model, tokenizer = _load(args)
    result, rows = run_task(model, tokenizer, records, settings, recipe, args.batch_size, chat=not args.raw,
                            template_kwargs=json.loads(args.template_kwargs), bootstrap=args.bootstrap)
    settings.update(data=str(data), max_samples=args.max_samples)
    dataset = args.dataset or (recipe_name(recipe) if recipe else Path(data).parent.name or Path(data).stem)
    return result, rows, settings, dataset


def run_eval(args) -> dict:
    """Run one evaluation; with --results also store a result record for `finetune report`."""
    rows, settings = None, {}
    dataset = args.dataset or (Path(args.data).parent.name if args.data else args.task)
    if args.task == "task":
        result, rows, settings, dataset = _task(args)
        summary = {"n": result["n"], "scores": result["scores"], "ci95": result["ci95"]}
        if args.output:
            write_jsonl(args.output, rows)
    else:
        result = summary = _standard_eval(args)
    if args.results:
        from finetune.eval.results import default_label, model_info, write_result

        info = model_info(args.model, args.adapter)
        label = args.label or default_label(args.model, info)
        seed = args.seed if args.seed is not None else info["seed"]
        path = write_result(args.results, label=label, dataset=dataset, task=args.task, seed=seed, result=result,
                            model=args.adapter or args.model, info=info, settings=settings,
                            num_parameters=getattr(args, "num_parameters", None), rows=rows)
        summary = {**summary, "result_file": str(path)}
    return summary


def _standard_eval(args) -> dict:
    records = load_records(args.data, args.max_samples) if args.data else []
    max_new_tokens = args.max_new_tokens or 1024
    temperature = args.temperature or 0.0
    if args.task == "ppl":
        from finetune.eval.perplexity import perplexity

        model, tokenizer = _load(args)
        texts = [r[args.text_field] for r in records]
        return perplexity(model, tokenizer, texts, args.max_length, args.batch_size)

    if args.task == "reasoning":
        from finetune.data.rl import normalize_answer
        from finetune.eval.generation import generate
        from finetune.eval.reasoning import score

        model, tokenizer = _load(args)
        prompts = [r.get("prompt") or r["question"] for r in records]
        answers = [normalize_answer(r["answer"]) for r in records]
        outputs = generate(model, tokenizer, prompts, max_new_tokens, temperature,
                           num_samples=args.samples, batch_size=args.batch_size,
                           template_kwargs=json.loads(args.template_kwargs))
        completions = [o if isinstance(o, list) else [o] for o in outputs]
        summary, rows = score(prompts, answers, completions, tokenizer)
        if args.output:
            write_jsonl(args.output, rows)
        return summary

    if args.task == "retrieval":
        from finetune.eval.embeddings import embedding_settings, retrieval_metrics

        model, tokenizer = _load(args, causal_lm=False)
        settings = embedding_settings(args.adapter or args.model)
        return retrieval_metrics(records, model, tokenizer, settings.get("pooling", args.pooling), settings,
                                 args.batch_size)

    if args.task == "benchmarks":
        from finetune.eval.benchmarks import run_benchmarks

        model, tokenizer = _load(args, merge=True)
        return run_benchmarks(model, tokenizer, args.tasks.split(","), args.batch_size, args.max_samples)
    raise ValueError(f"Unknown eval task {args.task}")
