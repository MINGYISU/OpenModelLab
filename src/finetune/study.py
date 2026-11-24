"""Studies: run a paper's experiment matrix (models x datasets x seeds) end to end.

  finetune study configs/studies/example_layerscale.yaml --dry-run   # print every command
  finetune study configs/studies/example_layerscale.yaml             # train -> eval -> report

A study YAML lists reference models and your variants, the datasets, the seeds, one training
config shared by all runs, and the evaluation/report settings (see configs/studies/). Each run is
skipped when its outputs already exist, so an interrupted study resumes where it stopped, and
adding a model or seed later only runs what is new. Finished runs land in:

  <output_root>/<dataset>/<model>/seed<k>/     trained weights or adapter, logs, experiment.json
  <results_dir>/<dataset>/<model>__seed<k>.json (+ .predictions.jsonl)   one record per evaluation
  <results_dir>/report.md, results.csv, table.tex
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from finetune.eval.results import slug


@dataclass
class Dataset:
    name: str
    recipe: str | None = None
    train: str | None = None
    eval: str | None = None
    metrics: list | None = None


@dataclass
class Model:
    label: str
    path: str
    variant: str | None = None
    variant_args: dict = field(default_factory=dict)
    train: bool = True
    set: list = field(default_factory=list)


@dataclass
class Study:
    name: str
    train_config: str
    datasets: list
    models: list
    seeds: list = field(default_factory=lambda: [42])
    launcher: str = "python"
    output_root: str | None = None
    results_dir: str | None = None
    set: list = field(default_factory=list)
    eval: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def load_study(path) -> Study:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = set(Study.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown study keys {sorted(unknown)}; allowed: {sorted(known)}")
    raw["train_config"] = str((path.parent / raw["train_config"]).resolve())
    datasets = []
    for item in raw.get("datasets") or []:
        item = {"name": item, "recipe": item} if isinstance(item, str) else dict(item)
        dataset = Dataset(**item)
        if dataset.recipe and not (dataset.train and dataset.eval):
            dataset.train = dataset.train or f"data/{dataset.recipe}/train.jsonl"
            dataset.eval = dataset.eval or f"data/{dataset.recipe}/eval.jsonl"
        if not dataset.eval:
            raise ValueError(f"Dataset {dataset.name}: needs a recipe or an eval file")
        datasets.append(dataset)
    models = [Model(**m) for m in raw.get("models") or []]
    labels = [m.label for m in models]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Model labels must be unique: {labels}")
    if not datasets or not models:
        raise ValueError("A study needs at least one dataset and one model")
    raw.update(datasets=datasets, models=models)
    study = Study(**raw)
    study.output_root = study.output_root or f"outputs/studies/{slug(study.name)}"
    study.results_dir = study.results_dir or f"results/{slug(study.name)}"
    return study


def _train_prefix(launcher: str) -> list[str]:
    if launcher in ("python", "inprocess"):
        return [sys.executable, "-m", "finetune.cli", "train"]
    return shlex.split(launcher)  # e.g. scripts/train.sh: accepts <config> [--set ...] like `finetune train`


def _restart(run_dir: Path) -> list[str]:
    """An unfinished run (no experiment.json yet): resume from its latest checkpoint, or start over."""
    if (run_dir / "experiment.json").exists() or not run_dir.is_dir() or not any(run_dir.iterdir()):
        return []
    checkpoints = [p for p in run_dir.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        return [f"train.resume_from={latest}"]
    return ["run.overwrite=true"]


def plan(study: Study) -> list[dict]:
    """Every step of the study, in order: {"kind": train|eval, "argv", "done", "label", ...}."""
    steps = []
    ev = study.eval
    for dataset in study.datasets:
        for model in study.models:
            for seed in (study.seeds if model.train else [None]):
                tag = f"seed{seed}" if seed is not None else "seedna"
                run_dir = Path(study.output_root) / slug(dataset.name) / slug(model.label) / tag
                if model.train:
                    if not dataset.train:
                        raise ValueError(f"Dataset {dataset.name} has no train split for model {model.label}")
                    overrides = [
                        f"run.name={slug(model.label)}-{slug(dataset.name)}-{tag}", f"run.output_dir={run_dir}",
                        f"run.seed={seed}", f"model.path={model.path}",
                        f"data.train={dataset.train}", f"data.eval={dataset.eval}",
                    ]
                    if model.variant:
                        overrides += [f"model.variant={model.variant}",
                                      f"model.variant_args={json.dumps(model.variant_args)}"]
                    argv = _train_prefix(study.launcher) + [study.train_config]
                    for item in overrides + list(study.set) + list(model.set) + _restart(run_dir):
                        argv += ["--set", item]
                    steps.append({"kind": "train", "label": model.label, "dataset": dataset.name, "seed": seed,
                                  "argv": argv, "done": (run_dir / "experiment.json").exists()})
                result = Path(study.results_dir) / slug(dataset.name) / f"{slug(model.label)}__{tag}.json"
                argv = [sys.executable, "-m", "finetune.cli", "eval", "task",
                        "--model", str(run_dir) if model.train else model.path,
                        "--dataset", dataset.name, "--label", model.label, "--results", study.results_dir]
                argv += ["--recipe", dataset.recipe] if dataset.recipe else []
                argv += ["--data", dataset.eval]
                if seed is not None:
                    argv += ["--seed", str(seed)]
                metrics = dataset.metrics or ev.get("metrics")
                if metrics:
                    argv += ["--metrics", ",".join(metrics)]
                if model.variant and not model.train:
                    raise ValueError(f"{model.label}: untrained variants are not evaluated; set train: true")
                for key in ("max_samples", "max_new_tokens", "batch_size", "temperature", "extract", "dtype",
                            "bootstrap", "system"):
                    if ev.get(key) is not None:
                        argv += [f"--{key.replace('_', '-')}", str(ev[key])]
                if ev.get("raw"):
                    argv.append("--raw")
                if ev.get("template_kwargs"):
                    argv += ["--template-kwargs", json.dumps(ev["template_kwargs"])]
                steps.append({"kind": "eval", "label": model.label, "dataset": dataset.name, "seed": seed,
                              "argv": argv, "done": result.exists()})
    return steps


def write_report(study: Study) -> str:
    from finetune.eval.report import build_report
    from finetune.eval.results import load_results

    options = study.report
    report = build_report(load_results(study.results_dir), metrics=options.get("metrics"),
                          baseline=options.get("baseline"), percent=options.get("percent", False),
                          title=options.get("title", study.name))
    out = Path(study.results_dir)
    (out / "report.md").write_text(report["markdown"], encoding="utf-8")
    (out / "results.csv").write_text(report["csv"], encoding="utf-8")
    (out / "table.tex").write_text(report["latex"], encoding="utf-8")
    return report["markdown"]


def run_study(path, dry_run: bool = False, report_only: bool = False, rerun: bool = False) -> str | None:
    study = load_study(path)
    steps = plan(study)
    if not report_only:
        for step in steps:
            status = "done" if step["done"] and not rerun else ("dry-run" if dry_run else "run")
            seed = "" if step["seed"] is None else f" seed={step['seed']}"
            print(f"[{status:7s}] {step['kind']:5s} {step['dataset']} / {step['label']}{seed}", flush=True)
            if dry_run:
                print("          " + shlex.join(step["argv"]), flush=True)
            if status != "run":
                continue
            argv = list(step["argv"])
            if step["kind"] == "train" and rerun:  # start fresh: drop any resume, overwrite the run dir
                first = argv.index("--set")
                sets = [s for s in argv[first + 1::2] if not s.startswith(("train.resume_from=", "run.overwrite="))]
                argv = argv[:first] + [x for s in sets + ["run.overwrite=true"] for x in ("--set", s)]
            if study.launcher == "inprocess":
                from finetune.cli import main

                main(argv[argv.index("finetune.cli") + 1:])
            else:
                subprocess.run(argv, check=True)
    if dry_run:
        return None
    markdown = write_report(study)
    print(markdown)
    return markdown
