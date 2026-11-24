"""Dataset recipes: one small module per dataset that converts its raw rows into the
standard record format of a training stage. See docs/ADDING_A_DATASET.md.

A recipe module defines:
  NAME          short id used by `finetune prepare NAME` (default: the module name)
  FORMAT        "sft", "preference", "rl", "contrastive", or "pretrain"
  SOURCE        where the raw rows come from: {"hf": id, "config":, "split":} or a JSONL path
  EVAL_SOURCE   optional held-out source; if None, EVAL_RATIO of the rows are split off
  convert(row)  raw row -> standard record, a list of records, or None to skip the row
  load(split)   optional: replaces SOURCE/EVAL_SOURCE for custom files (CSV, zip, ...);
                `split` is "train" or "eval"; return None for "eval" to split off EVAL_RATIO

Recipes live in this package (files starting with "_" are ignored), or can be passed
to `finetune prepare` as a path to any .py file.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
import random
from pathlib import Path
from types import ModuleType
from typing import Iterable

from finetune.data.io import iter_records
from finetune.data.sft import validate_conversation

FORMATS = ("sft", "preference", "rl", "contrastive", "pretrain")


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_messages_or_text(record: dict, key: str) -> None:
    value = record.get(key)
    if _is_text(value):
        return
    if isinstance(value, list) and value and all(
        isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"} and _is_text(m.get("content"))
        for m in value
    ):
        return
    raise ValueError(f"'{key}' must be a non-empty string or a list of {{role, content}} messages")


def validate_record(record, fmt: str) -> None:
    """Raise ValueError if `record` is not a valid `fmt` record (what the trainer expects)."""
    if not isinstance(record, dict):
        raise ValueError(f"convert() must return a dict, got {type(record).__name__}")
    if fmt == "sft":
        validate_conversation(record, 0)
    elif fmt == "preference":
        for key in ("prompt", "chosen", "rejected"):
            _check_messages_or_text(record, key)
        if record["chosen"] == record["rejected"]:
            raise ValueError("'chosen' and 'rejected' are identical")
    elif fmt == "rl":
        if not (_is_text(record.get("prompt")) or _is_text(record.get("question"))):
            raise ValueError("needs a non-empty 'prompt' (or 'question')")
        if record.get("answer") is None or not str(record["answer"]).strip():
            raise ValueError("needs a non-empty 'answer'")
    elif fmt == "contrastive":
        if not (_is_text(record.get("query")) and _is_text(record.get("positive"))):
            raise ValueError("needs non-empty 'query' and 'positive'")
        negatives = record.get("negatives", [])
        if not isinstance(negatives, list) or not all(_is_text(n) for n in negatives):
            raise ValueError("'negatives' must be a list of non-empty strings")
    elif fmt == "pretrain":
        if not _is_text(record.get("text")):
            raise ValueError("needs a non-empty 'text'")
    else:
        raise ValueError(f"Unknown FORMAT {fmt!r}; choose from {FORMATS}")


def _check_module(module: ModuleType) -> ModuleType:
    if getattr(module, "FORMAT", None) not in FORMATS:
        raise ValueError(f"Recipe {module.__name__}: FORMAT must be one of {FORMATS}")
    if not callable(getattr(module, "convert", None)):
        raise ValueError(f"Recipe {module.__name__}: missing convert(row)")
    if getattr(module, "SOURCE", None) is None and not callable(getattr(module, "load", None)):
        raise ValueError(f"Recipe {module.__name__}: define SOURCE or load(split)")
    return module


def recipe_name(module: ModuleType) -> str:
    return getattr(module, "NAME", module.__name__.rsplit(".", 1)[-1])


def available_recipes() -> dict[str, ModuleType]:
    recipes = {}
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            module = _check_module(importlib.import_module(f"{__name__}.{info.name}"))
            recipes[recipe_name(module)] = module
    return recipes


def get_recipe(name_or_path: str) -> ModuleType:
    path = Path(name_or_path)
    if path.suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"Recipe file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"finetune_recipe_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _check_module(module)
    recipes = available_recipes()
    if name_or_path not in recipes:
        raise ValueError(f"Unknown recipe {name_or_path!r}; available: {sorted(recipes)}")
    return recipes[name_or_path]


def iter_rows(recipe: ModuleType, split: str = "train") -> Iterable[dict] | None:
    """Raw rows for "train" or "eval" (None when the recipe has no separate eval data)."""
    if callable(getattr(recipe, "load", None)):
        return recipe.load(split)
    source = recipe.SOURCE if split == "train" else getattr(recipe, "EVAL_SOURCE", None)
    return None if source is None else iter_records(source)


def _sft_length(tokenizer, record: dict, template_kwargs: dict) -> int:
    text = tokenizer.apply_chat_template(record["messages"], tokenize=False, **template_kwargs)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _convert_split(recipe, rows, seen, stats, max_samples, tokenizer, max_length, template_kwargs) -> list[dict]:
    records = []
    for index, row in enumerate(rows, 1):
        if max_samples is not None and len(records) >= max_samples:
            break
        outputs = recipe.convert(row)
        if outputs is None:
            stats["skipped_by_recipe"] += 1
            continue
        for record in outputs if isinstance(outputs, list) else [outputs]:
            try:
                validate_record(record, recipe.FORMAT)
            except ValueError as error:
                stats["invalid"] += 1
                stats.setdefault("first_invalid", f"row {index}: {error}")
                continue
            key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if key in seen:
                stats["duplicate"] += 1
                continue
            if (tokenizer is not None and recipe.FORMAT == "sft"
                    and _sft_length(tokenizer, record, template_kwargs) > max_length):
                stats["too_long"] += 1
                continue
            seen.add(key)
            records.append(record)
    return records


def prepare(recipe: ModuleType, out_dir, eval_ratio: float | None = None, max_samples: int | None = None,
            seed: int = 42, tokenizer=None, max_length: int = 2048, template_kwargs: dict | None = None) -> dict:
    """Convert, validate, de-duplicate (also across splits), length-filter (SFT, with a
    tokenizer), split, and write train.jsonl / eval.jsonl / prepare_stats.json to `out_dir`."""
    stats = {"skipped_by_recipe": 0, "invalid": 0, "duplicate": 0, "too_long": 0}
    shared = (set(), stats, max_samples, tokenizer, max_length, template_kwargs or {})
    eval_rows = iter_rows(recipe, "eval")
    # Eval is converted first so training rows that duplicate an eval row are dropped from train.
    evaluation = _convert_split(recipe, eval_rows, *shared) if eval_rows is not None else None
    train = _convert_split(recipe, iter_rows(recipe, "train"), *shared)
    if evaluation is None:
        ratio = getattr(recipe, "EVAL_RATIO", 0.05) if eval_ratio is None else eval_ratio
        random.Random(seed).shuffle(train)
        n_eval = min(max(1, round(len(train) * ratio)), len(train) - 1) if ratio > 0 and len(train) > 1 else 0
        evaluation, train = train[:n_eval], train[n_eval:]
    if not train:
        raise ValueError(f"Recipe {recipe_name(recipe)} produced no training records: {stats}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("train.jsonl", train), ("eval.jsonl", evaluation)):
        with (out_dir / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats.update(recipe=recipe_name(recipe), format=recipe.FORMAT, train=len(train), eval=len(evaluation),
                 out_dir=str(out_dir))
    (out_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
