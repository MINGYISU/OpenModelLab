"""Behavior-change analysis between a base model and a trained model (`finetune analyze`)."""

from __future__ import annotations

import json
from pathlib import Path

from finetune.data.io import load_records


def run_analysis(args) -> dict:
    from finetune.eval.generation import load_for_inference

    base, tokenizer = load_for_inference(args.base, dtype=args.dtype)
    tuned, tuned_tokenizer = load_for_inference(args.tuned, dtype=args.dtype, merge=True)
    records = load_records(args.data, args.max_samples) if args.data else []
    texts = [r[args.text_field] for r in records if args.text_field in r]
    if args.analysis != "weights" and args.analysis != "logit-lens" and not texts:
        raise ValueError(f"--data with a '{args.text_field}' field is required for {args.analysis}")

    if args.analysis == "weights":
        from finetune.eval.behavior.weight_diff import weight_diff

        result = weight_diff(base, tuned)
    elif args.analysis == "kl":
        from finetune.eval.behavior.kl_drift import kl_drift

        result = kl_drift(base, tuned, tokenizer, texts, args.max_length, args.top)
    elif args.analysis == "logit-lens":
        from finetune.eval.behavior.logit_lens import logit_lens

        result = {"base": logit_lens(base, tokenizer, records), "tuned": logit_lens(tuned, tuned_tokenizer, records)}
    elif args.analysis == "cka":
        from finetune.eval.behavior.representation import cka_by_layer

        result = cka_by_layer(base, tuned, tokenizer, texts, args.max_length)
    elif args.analysis == "router":
        from finetune.eval.behavior.moe_router import router_shift

        result = router_shift(base, tuned, tokenizer, texts, args.max_length)
    else:
        raise ValueError(args.analysis)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    # Keep stdout readable: drop the bulky per-parameter / per-expert tables.
    summary = {k: v for k, v in result.items() if k not in {"per_parameter"}}
    if "layers" in summary:
        summary["layers"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("usage")}
                             for k, v in summary["layers"].items()}
    return summary
