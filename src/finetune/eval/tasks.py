"""Task evaluation: generate answers for a dataset's eval split and score them.

  finetune eval task --recipe my_dataset --model outputs/my_run --results results/my_study

Prompts and references come from the records (SFT: the conversation minus the final assistant
turn is the prompt, that turn is the reference; RL: prompt/question and answer; any record may
carry an explicit "reference" (string or list of acceptable strings)). A recipe can customise
everything dataset-specific:

  METRICS      built-in metric names, e.g. ["exact_match", "f1"] (see finetune.eval.metrics)
  EXTRACT      how to clean predictions before scoring: "strip_think" (default), "answer_tag",
               "first_line", "last_number", "choice", or "none"
  GENERATION   defaults such as {"max_new_tokens": 64, "temperature": 0.0, "system": "..."}
  build_prompt(record) -> str | messages,  reference(record) -> str | list[str]
  metric(prediction, reference, record) -> float | dict    only when no built-in metric fits
"""

from __future__ import annotations

from finetune.eval.metrics import compute_metrics

DEFAULT_METRICS = {"sft": ["exact_match", "f1", "rouge_l"], "rl": ["numeric_match"]}


def default_prompt(record: dict):
    if "messages" in record:
        messages = record["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("SFT eval records must end with the assistant turn used as the reference")
        return messages[:-1]
    prompt = record.get("prompt") or record.get("question")
    if prompt is None:
        raise ValueError("Eval records need 'messages', or 'prompt'/'question' (or define build_prompt)")
    return prompt


def default_reference(record: dict):
    if "reference" in record:
        return record["reference"]
    if "messages" in record:
        return record["messages"][-1]["content"]
    answer = str(record["answer"])
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()


def build_examples(records: list[dict], recipe=None) -> tuple[list, list]:
    to_prompt = getattr(recipe, "build_prompt", default_prompt)
    to_reference = getattr(recipe, "reference", default_reference)
    return [to_prompt(r) for r in records], [to_reference(r) for r in records]


def task_settings(recipe, records: list[dict], args) -> dict:
    """Resolve metrics / extractor / generation settings: CLI flags > recipe > defaults."""
    generation = {"max_new_tokens": 256, "temperature": 0.0, "system": None,
                  **getattr(recipe, "GENERATION", {})}
    for key in ("max_new_tokens", "temperature", "system"):
        if getattr(args, key, None) is not None:
            generation[key] = getattr(args, key)
    fmt = getattr(recipe, "FORMAT", None) or ("sft" if records and "messages" in records[0] else "rl")
    if args.metrics:
        metrics = args.metrics.split(",")
    elif hasattr(recipe, "METRICS"):
        metrics = list(recipe.METRICS)
    else:
        metrics = [] if callable(getattr(recipe, "metric", None)) else DEFAULT_METRICS.get(fmt, ["exact_match"])
    return {"metrics": metrics, "extract": args.extract or getattr(recipe, "EXTRACT", "strip_think"),
            "generation": generation}


def run_task(model, tokenizer, records: list[dict], settings: dict, recipe=None, batch_size: int = 8,
             chat: bool = True, template_kwargs=None, bootstrap: int = 1000) -> tuple[dict, list[dict]]:
    from finetune.eval.generation import generate

    prompts, references = build_examples(records, recipe)
    gen = settings["generation"]
    predictions = generate(model, tokenizer, prompts, gen["max_new_tokens"], gen["temperature"],
                           chat=chat, template_kwargs=template_kwargs, batch_size=batch_size,
                           system=gen.get("system"))
    result = compute_metrics(predictions, references, settings["metrics"], extract=settings["extract"],
                             bootstrap=bootstrap, custom=getattr(recipe, "metric", None), records=records)
    rows = [{"prompt": p, "reference": r, "prediction": o,
             "scores": {name: values[i] for name, values in result["per_example"].items()}}
            for i, (p, r, o) in enumerate(zip(prompts, references, predictions))]
    return result, rows
