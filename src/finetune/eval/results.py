"""Result records: one JSON file per (model, dataset, seed) evaluation, collected by `finetune report`.

Layout:  <results_dir>/<dataset>/<label>__seed<seed>.json  (+ .predictions.jsonl for task evals)
Record:  {"label", "dataset", "task", "seed", "scores", "ci95", "per_example", "n", "model",
          "variant", "reference", "num_parameters", "settings", "git_sha", "created"}
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from finetune.utils import git_sha, write_jsonl


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def model_info(model: str, adapter: str | None = None) -> dict:
    """What a model directory says about itself: training run, seed, and variant (if any)."""
    from finetune.variants import SPEC_KEY, read_spec_file

    info = {"variant": None, "seed": None, "run_name": None, "trained": False, "num_parameters": None}
    directories = [Path(d) for d in (adapter, model) if d and Path(d).is_dir()]
    directories += [d.parent for d in directories if d.name.startswith("checkpoint-")]  # intermediate checkpoints
    for directory in directories:
        experiment = _read_json(directory / "experiment.json")
        if experiment and not info["trained"]:
            info.update(trained=True, seed=experiment["config"]["run"]["seed"],
                        run_name=experiment["config"]["run"]["name"],
                        num_parameters=experiment.get("num_parameters"))
        spec = read_spec_file(directory) or (_read_json(directory / "config.json") or {}).get(SPEC_KEY)
        if spec and info["variant"] is None:
            info["variant"] = {k: spec.get(k) for k in ("name", "title", "args")}
    return info


def default_label(model: str, info: dict) -> str:
    """Variant title (+ any non-default args, so ablations stay apart), else run name, else dir name."""
    variant = info["variant"]
    if variant:
        from finetune.variants import get_variant

        try:
            defaults = getattr(get_variant(variant["name"]), "ARGS", {})
        except (ValueError, FileNotFoundError):
            defaults = {}
        changed = {k: v for k, v in (variant.get("args") or {}).items() if defaults.get(k) != v}
        return variant["title"] + (f" ({', '.join(f'{k}={v}' for k, v in changed.items())})" if changed else "")
    if info["trained"] and info["run_name"] and info["run_name"] != "experiment":
        return info["run_name"]
    return model.rstrip("/").split("/")[-1]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", text).strip("_") or "model"


def write_result(results_dir, *, label: str, dataset: str, task: str, seed, result: dict, model: str,
                 info: dict, settings: dict | None = None, num_parameters: int | None = None,
                 rows: list[dict] | None = None) -> Path:
    """`result` is {"scores", "ci95", "per_example", "n"} (task evals) or a flat summary dict."""
    scores = result.get("scores") or {k: v for k, v in result.items() if isinstance(v, (int, float))
                                      and not isinstance(v, bool)}
    record = {
        "label": label, "dataset": dataset, "task": task, "seed": seed,
        "scores": scores, "ci95": result.get("ci95", {}), "per_example": result.get("per_example", {}),
        "n": result.get("n", result.get("prompts")),
        "model": model, "variant": info["variant"], "reference": not info["trained"] and not info["variant"],
        "num_parameters": num_parameters or info["num_parameters"], "settings": settings or {},
        "git_sha": git_sha(), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(results_dir) / slug(dataset)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{slug(label)}__seed{seed if seed is not None else 'na'}.json"
    previous = _read_json(path)
    if previous and previous.get("model") != model:
        raise ValueError(f"{path} already holds results of a different model ({previous.get('model')}); "
                         "pass a distinct --label (or delete the old file)")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows is not None:
        write_jsonl(path.with_suffix(".predictions.jsonl"), rows)
    return path


def load_results(paths) -> list[dict]:
    records = []
    for root in [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]:
        files = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in files:
            record = _read_json(path)
            if isinstance(record, dict) and "scores" in record and "label" in record:
                records.append(record)
    return records
