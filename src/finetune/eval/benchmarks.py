"""Standard benchmarks through lm-evaluation-harness, using our vendored model instance.

Install with `pip install -e '.[eval]'`. Passing an instantiated model to HFLM keeps
the vendored classes in charge (lm-eval never calls AutoModel).
"""

from __future__ import annotations


def run_benchmarks(model, tokenizer, tasks: list[str], batch_size: int = 8, limit: int | None = None,
                   apply_chat_template: bool = False) -> dict:
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError("Install lm-eval: pip install -e '.[eval]'") from error
    tokenizer.padding_side = "left"
    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(model=wrapped, tasks=tasks, limit=limit, apply_chat_template=apply_chat_template)
    return {task: metrics for task, metrics in results["results"].items()}
