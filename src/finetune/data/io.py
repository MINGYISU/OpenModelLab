"""Load records from local JSONL files or Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            yield record


def iter_records(spec: Any) -> Iterator[dict]:
    """Yield records from a JSONL path or {"hf": name, "config": ..., "split": ..., "streaming": bool}."""
    if spec is None:
        return iter(())
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return _read_jsonl(path)
    if isinstance(spec, dict) and "hf" in spec:
        from datasets import load_dataset

        dataset = load_dataset(
            spec["hf"], spec.get("config"), split=spec.get("split", "train"),
            streaming=spec.get("streaming", False), revision=spec.get("revision"),
        )
        return iter(dataset)
    raise ValueError(f"Unsupported dataset spec: {spec!r}")


def load_records(spec: Any, max_samples: int | None = None) -> list[dict]:
    records = []
    for record in iter_records(spec):
        if max_samples is not None and len(records) >= max_samples:
            break
        records.append(record)
    return records


def mix_records(main: Iterator[dict], replay: Iterator[dict], ratio: float, seed: int = 0) -> Iterator[dict]:
    """Interleave documents so ~`ratio` of the output comes from `replay` (stops when `main` ends).

    The ratio is per document, not per token; check token counts if documents differ a lot in length.
    """
    import random

    if not 0 <= ratio < 1:
        raise ValueError("replay_ratio must be in [0, 1)")
    rng = random.Random(seed)
    replay = iter(replay)
    for record in main:
        while ratio and rng.random() < ratio:
            extra = next(replay, None)
            if extra is None:
                ratio = 0.0
                break
            yield extra
        yield record


def check_no_overlap(train: list[dict], evaluation: list[dict], key=None) -> None:
    """Reject exact duplicates between splits (semantic duplicates need manual curation)."""
    key = key or (lambda record: json.dumps(record, sort_keys=True))
    seen = {key(record) for record in train}
    overlap = sum(key(record) in seen for record in evaluation)
    if overlap:
        raise ValueError(f"{overlap} evaluation records also appear in the training data")
