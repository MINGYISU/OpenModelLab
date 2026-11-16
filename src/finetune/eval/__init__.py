"""Evaluation entry points used by `finetune eval`."""

from __future__ import annotations

import json

from finetune.data.io import load_records
from finetune.utils import write_jsonl


def run_eval(args) -> dict:
    from finetune.eval.generation import load_for_inference

    records = load_records(args.data, args.max_samples) if args.data else []
    if args.task == "ppl":
        from finetune.eval.perplexity import perplexity

        model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype)
        texts = [r[args.text_field] for r in records]
        return perplexity(model, tokenizer, texts, args.max_length, args.batch_size)

    if args.task == "reasoning":
        from finetune.data.rl import normalize_answer
        from finetune.eval.generation import generate
        from finetune.eval.reasoning import score

        model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype)
        prompts = [r.get("prompt") or r["question"] for r in records]
        answers = [normalize_answer(r["answer"]) for r in records]
        outputs = generate(model, tokenizer, prompts, args.max_new_tokens, args.temperature,
                           num_samples=args.samples, batch_size=args.batch_size,
                           template_kwargs=json.loads(args.template_kwargs))
        completions = [o if isinstance(o, list) else [o] for o in outputs]
        summary, rows = score(prompts, answers, completions, tokenizer)
        if args.output:
            write_jsonl(args.output, rows)
        return summary

    if args.task == "retrieval":
        from finetune.eval.embeddings import embedding_settings, retrieval_metrics

        model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype, causal_lm=False)
        settings = embedding_settings(args.adapter or args.model)
        return retrieval_metrics(records, model, tokenizer, settings.get("pooling", args.pooling), settings,
                                 args.batch_size)

    if args.task == "benchmarks":
        from finetune.eval.benchmarks import run_benchmarks

        model, tokenizer = load_for_inference(args.model, args.revision, args.adapter, args.dtype, merge=True)
        return run_benchmarks(model, tokenizer, args.tasks.split(","), args.batch_size, args.max_samples)
    raise ValueError(f"Unknown eval task {args.task}")
