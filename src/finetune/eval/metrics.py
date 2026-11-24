"""Evaluation metrics as a reusable tool: library, CLI (`finetune score`), and recipe hook.

  from finetune.eval.metrics import compute_metrics
  result = compute_metrics(predictions, references, ["exact_match", "f1", "rouge_l"])
  result["scores"]["f1"], result["ci95"]["f1"]

  finetune score preds.jsonl --metrics exact_match,f1        # fields: prediction, reference

Each metric maps one (prediction, references) pair to a small vector of sufficient statistics
and aggregates the column sums. Mean metrics use [score]; corpus metrics such as BLEU use n-gram
counts. This makes every metric bootstrappable: the 95% CI resamples examples and re-aggregates.

Built-in metrics (references may be a string or a list of acceptable strings):
  exact_match   normalised string equality (SQuAD normalisation: case, punctuation, articles)
  contains      normalised reference appears in the prediction
  f1            token-overlap F1 (SQuAD)
  rouge_l       ROUGE-L F1 (longest common subsequence over lowercase word tokens)
  bleu          corpus BLEU-4 with brevity penalty, 0-100 (use sacrebleu for official numbers)
  numeric_match the extracted answer (<answer>, \\boxed{}, or last number) equals the reference
  choice        the first standalone option letter (A-J) in the prediction equals the reference
  length        prediction length in words (descriptive, not a score)

Add your own with `@register_metric("name")` on a `fn(prediction, reference) -> float`, or define
`metric(prediction, reference, record)` in a dataset recipe for dataset-specific scoring.
Predictions are first passed through an extractor (default "strip_think": drop <think> traces).
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from finetune.training.rewards import ANSWER_TAG, answers_match, extract_answer


@dataclass(frozen=True)
class Metric:
    name: str
    stats: Callable[[str, list[str]], Sequence[float]]   # one example -> sufficient statistics
    aggregate: Callable[[np.ndarray, int], float]         # column sums, n -> score
    higher_is_better: bool = True


METRICS: dict[str, Metric] = {}


def _mean(totals: np.ndarray, n: int) -> float:
    return float(totals[0] / n) if n else float("nan")


def register_metric(name: str, higher_is_better: bool = True):
    """Register `fn(prediction, reference) -> float`; with several references the best is used."""
    def decorator(fn):
        METRICS[name] = Metric(name, lambda pred, refs: [max(fn(pred, ref) for ref in refs)], _mean,
                               higher_is_better)
        return fn
    return decorator


# ---- normalisation and extraction -------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_WORD = re.compile(r"\w+")


def normalize_text(text: str) -> str:
    """SQuAD normalisation: lowercase, drop punctuation and articles, collapse whitespace."""
    return " ".join(_ARTICLES.sub(" ", str(text).lower().translate(_PUNCT)).split())


def strip_think(text: str) -> str:
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


def answer_tag(text: str) -> str:
    matches = ANSWER_TAG.findall(text)
    return matches[-1].strip() if matches else strip_think(text)


_ANSWER_CHOICE = re.compile(r"answer(?:\s+is)?\s*[:：]?\s*\**\(?([A-J])\b", re.IGNORECASE)
_CHOICE = re.compile(r"(?:^|[\s(\[:*])\(?([A-J])(?=$|[).\]:,*])", re.MULTILINE)


def first_choice(text: str) -> str:
    """Option letter: "answer is (C)" / "Answer: C" first, else the first "(C)", "C.", "C)" or a bare
    leading letter (so the pronoun "I" or the article "A" inside a sentence is not taken)."""
    text = strip_think(text)
    match = _ANSWER_CHOICE.search(text) or _CHOICE.search(text)
    if not match:
        return text.strip().upper() if re.fullmatch(r"\s*[A-Ja-j]\s*", text) else ""
    return next(group for group in match.groups() if group).upper()


EXTRACTORS: dict[str, Callable[[str], str]] = {
    "none": lambda text: text,
    "strip_think": strip_think,
    "answer_tag": answer_tag,
    "first_line": lambda text: (strip_think(text).splitlines() or [""])[0].strip(),
    "last_number": lambda text: extract_answer(strip_think(text)) or "",
    "choice": first_choice,
}


# ---- per-example metrics ----------------------------------------------------------------------

@register_metric("exact_match")
def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


@register_metric("contains")
def contains(prediction: str, reference: str) -> float:
    return float(normalize_text(reference) in normalize_text(prediction))


@register_metric("f1")
def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize_text(prediction).split(), normalize_text(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _lcs(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            current.append(previous[j] + 1 if x == y else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


@register_metric("rouge_l")
def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = _WORD.findall(prediction.lower()), _WORD.findall(reference.lower())
    if not pred or not ref:
        return float(pred == ref)
    lcs = _lcs(pred, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


@register_metric("numeric_match")
def numeric_match(prediction: str, reference: str) -> float:
    return float(answers_match(extract_answer(prediction), extract_answer(reference) or reference.strip()))


@register_metric("choice")
def choice(prediction: str, reference: str) -> float:
    return float(first_choice(prediction) == reference.strip().strip("()").upper()[:1])


@register_metric("length", higher_is_better=False)
def length(prediction: str, reference: str) -> float:
    return float(len(prediction.split()))


# ---- corpus BLEU --------------------------------------------------------------------------------

_BLEU_TOKEN = re.compile(r"\w+|[^\w\s]")


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_stats(prediction: str, references: list[str], order: int = 4) -> list[float]:
    pred = _BLEU_TOKEN.findall(prediction.lower())
    refs = [_BLEU_TOKEN.findall(r.lower()) for r in references]
    closest = min((abs(len(r) - len(pred)), len(r)) for r in refs)[1]
    stats = [len(pred), closest]
    for n in range(1, order + 1):
        pred_counts = _ngrams(pred, n)
        max_ref = Counter()
        for ref in refs:
            max_ref |= _ngrams(ref, n)
        stats += [sum((pred_counts & max_ref).values()), max(len(pred) - n + 1, 0)]
    return stats


def _bleu_aggregate(totals: np.ndarray, n: int, order: int = 4) -> float:
    pred_len, ref_len = totals[0], totals[1]
    if pred_len == 0:
        return 0.0
    log_precision = 0.0
    for k in range(order):
        matches, possible = totals[2 + 2 * k], totals[3 + 2 * k]
        if matches == 0 or possible == 0:
            return 0.0
        log_precision += math.log(matches / possible) / order
    brevity = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return 100.0 * brevity * math.exp(log_precision)


METRICS["bleu"] = Metric("bleu", _bleu_stats, _bleu_aggregate)


# ---- computing ----------------------------------------------------------------------------------

def _as_refs(reference) -> list[str]:
    refs = [str(r) for r in reference] if isinstance(reference, (list, tuple)) else [str(reference)]
    if not refs:
        raise ValueError("Every example needs at least one reference")
    return refs


def bootstrap_ci(stats: np.ndarray, aggregate, samples: int = 1000, seed: int = 0,
                 level: float = 0.95) -> list[float] | None:
    """Percentile bootstrap over examples (Efron & Tibshirani, 1993)."""
    n = len(stats)
    if samples <= 0 or n < 2:
        return None
    rng = np.random.default_rng(seed)
    values = np.array([aggregate(stats[rng.integers(0, n, n)].sum(axis=0), n) for _ in range(samples)])
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def compute_metrics(predictions: Sequence[str], references: Sequence, metrics: Sequence[str],
                    extract: str | Callable[[str], str] = "strip_think", bootstrap: int = 1000,
                    seed: int = 0, custom: Callable | None = None, records: Sequence[dict] | None = None) -> dict:
    """Score predictions against references.

    `custom(prediction, reference, record)` (e.g. a recipe's `metric`) may return a float (reported
    as "metric") or a dict of named floats; its scores are averaged and bootstrapped like the rest.
    Returns {"n", "scores", "ci95", "per_example"} where per_example holds one score per example
    for mean metrics (corpus metrics such as BLEU have none).
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} references")
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; available: {sorted(METRICS)}")
    extractor = EXTRACTORS[extract] if isinstance(extract, str) else extract
    cleaned = [extractor(str(p)) for p in predictions]
    refs = [_as_refs(r) for r in references]
    n = len(cleaned)
    result = {"n": n, "scores": {}, "ci95": {}, "per_example": {}}

    def add(name: str, stats: np.ndarray, aggregate) -> None:
        result["scores"][name] = aggregate(stats.sum(axis=0), n) if n else float("nan")
        result["ci95"][name] = bootstrap_ci(stats, aggregate, bootstrap, seed)
        if aggregate is _mean:
            result["per_example"][name] = stats[:, 0].tolist()

    for name in metrics:
        metric = METRICS[name]
        add(name, np.array([metric.stats(p, r) for p, r in zip(cleaned, refs)], dtype=float).reshape(n, -1),
            metric.aggregate)
    if custom is not None:
        rows = []
        for i, (prediction, reference) in enumerate(zip(cleaned, references)):
            value = custom(prediction, reference, records[i] if records else {})
            rows.append(value if isinstance(value, dict) else {"metric": value})
        for name in rows[0] if rows else []:
            add(name, np.array([[float(row[name])] for row in rows]), _mean)
    return result
