"""Shared helpers: distributed info, seeding, run metadata, output-dir safety."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return rank() == 0


def log(*args) -> None:
    if is_main():
        print(*args, flush=True)


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_output_dir(path: str | Path, overwrite: bool = False, resume: bool = False) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()) and not (overwrite or resume):
        raise ValueError(
            f"Output directory {output} is not empty; set run.overwrite=true or train.resume_from"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def file_sha256(path: str | Path, limit_bytes: int | None = None) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit_bytes
        while chunk := handle.read(1 << 20):
            if remaining is not None:
                chunk = chunk[:remaining]
                remaining -= len(chunk)
            digest.update(chunk)
            if remaining == 0:
                break
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if sha.returncode == 0 else None
    except OSError:
        return None


def run_metadata(config, model=None) -> dict:
    import torch
    import transformers

    meta = {
        "config": config.to_dict(),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "world_size": world_size(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "dataset_sha256": {
            split: file_sha256(spec) for split, spec in
            (("train", config.data.train), ("eval", config.data.eval)) if isinstance(spec, str)
        },
    }
    if model is not None:
        base = getattr(model, "base_model", model)
        meta["resolved_model_revision"] = getattr(getattr(base, "config", None), "_commit_hash", None)
        meta["model_class"] = f"{type(base).__module__}.{type(base).__name__}"
    for name in ("peft", "accelerate", "datasets", "trl", "deepspeed"):
        try:
            meta[name] = __import__(name).__version__
        except ImportError:
            pass
    return meta


def write_metadata(output_dir: str | Path, config, model=None) -> None:
    if is_main():
        path = Path(output_dir) / "experiment.json"
        path.write_text(json.dumps(run_metadata(config, model), indent=2, default=str), encoding="utf-8")


def write_jsonl(path: str | Path, rows) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
