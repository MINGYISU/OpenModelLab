"""Aggregate result records into paper-ready tables (`finetune report`).

  finetune report results/my_study --baseline "Qwen2.5-0.5B + SFT" --percent --out reports/my_study

Rows are models (references first, then fine-tuned references, then your variants), columns are
dataset/metric pairs. Cells are mean ± std over seeds; the best value per column is bold. With
--baseline, each cell also shows the difference to the baseline and a † when a paired bootstrap
test over the same eval examples gives p < 0.05. Writes report.md, results.csv (long format,
for plotting), and table.tex (booktabs).
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict

import numpy as np

from finetune.eval.metrics import METRICS

LOWER_IS_BETTER = re.compile(r"loss|ppl|perplexity|nll|error|length")
DESCRIPTIVE = {"n", "prompts", "samples_per_prompt", "length", "format_rate", "closed_think_rate"}


def higher_is_better(metric: str) -> bool:
    return METRICS[metric].higher_is_better if metric in METRICS else not LOWER_IS_BETTER.search(metric)


def _kind(record: dict) -> int:
    return 0 if record.get("reference") else (2 if record.get("variant") else 1)


KINDS = ("reference", "fine-tuned", "variant")


def paired_pvalue(a: np.ndarray, b: np.ndarray, samples: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap test of mean(a - b) = 0 over the same examples."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) < 2 or not diff.any():
        return 1.0
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (samples, len(diff)))].mean(axis=1) - diff.mean()
    return float(min(1.0, (np.abs(means) >= abs(diff.mean())).mean()))


def summarize(records: list[dict], metrics: list[str] | None = None) -> list[dict]:
    """One row per (model label, dataset, metric) with mean/std over seeds."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        for metric, value in record["scores"].items():
            wanted = metric in metrics if metrics else metric not in DESCRIPTIVE and not metric.startswith("mean_")
            if wanted and isinstance(value, (int, float)):
                groups[(record["label"], record["dataset"], metric)].append(record)
    rows = []
    for (label, dataset, metric), group in groups.items():
        values = np.array([r["scores"][metric] for r in group], dtype=float)
        per_example = [r.get("per_example", {}).get(metric) for r in group]
        aligned = all(p is not None for p in per_example) and len({len(p) for p in per_example}) == 1
        first = group[0]
        rows.append({
            "label": label, "dataset": dataset, "metric": metric, "kind": KINDS[_kind(first)],
            "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_seeds": len(values), "seeds": sorted((r["seed"] for r in group), key=str),
            "ci95": first.get("ci95", {}).get(metric) if len(values) == 1 else None,
            "n": first.get("n"), "num_parameters": first.get("num_parameters"),
            "variant": first.get("variant"),
            "_per_example": np.mean(per_example, axis=0) if aligned else None,
        })
    return rows


_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(text: str) -> str:
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def _params(n) -> str:
    if not n:
        return "-"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.3g}{unit}"
    return str(n)


def build_report(records: list[dict], metrics: list[str] | None = None, baseline: str | None = None,
                 percent: bool = False, title: str = "Results") -> dict:
    rows = summarize(records, metrics)
    if not rows:
        raise ValueError("No scores to report (check the results directory and --metrics)")
    order = {r["label"]: (KINDS.index(r["kind"]), r["label"]) for r in rows}
    labels = sorted(order, key=order.get)
    if baseline and baseline not in order:
        raise ValueError(f"Baseline {baseline!r} not found; models: {labels}")
    metric_order = {m: i for i, m in enumerate(metrics or [])}
    columns = sorted({(r["dataset"], r["metric"]) for r in rows},
                     key=lambda c: (c[0], metric_order.get(c[1], len(metric_order)), c[1]))
    cell = {(r["label"], r["dataset"], r["metric"]): r for r in rows}

    scale, best, values_of = {}, {}, {}
    for column in columns:
        values = [cell[(label, *column)]["mean"] for label in labels if (label, *column) in cell]
        values_of[column] = values
        scale[column] = 100.0 if percent and all(0.0 <= v <= 1.0 for v in values) else 1.0
        best[column] = (max if higher_is_better(column[1]) else min)(values)

    def fmt(label, column, latex=False):
        row = cell.get((label, *column))
        if row is None:
            return "-"
        s = scale[column]
        digits = 1 if s == 100.0 else 3
        text = f"{row['mean'] * s:.{digits}f}"
        if row["std"] is not None:
            text += (r" {\scriptsize$\pm$" + f"{row['std'] * s:.{digits}f}" + "}") if latex else f" ± {row['std'] * s:.{digits}f}"
        if math.isclose(row["mean"], best[column]) and len(set(values_of[column])) > 1:
            text = rf"\textbf{{{text}}}" if latex else f"**{text}**"
        base = cell.get((baseline, *column)) if baseline else None
        if base is not None and label != baseline:
            if not latex:
                text += f" ({(row['mean'] - base['mean']) * s:+.{digits}f})"
            if (row["_per_example"] is not None and base["_per_example"] is not None
                    and len(row["_per_example"]) == len(base["_per_example"])
                    and paired_pvalue(row["_per_example"], base["_per_example"]) < 0.05):
                text += r"$^\dagger$" if latex else "†"
        return text

    params = {label: _params(next(r["num_parameters"] for r in rows if r["label"] == label)) for label in labels}
    header = ["Model", "Params"] + [f"{d} / {m}" for d, m in columns]
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label in labels:
        lines.append("| " + " | ".join([label, params[label]] + [fmt(label, c) for c in columns]) + " |")
    n_seeds = sorted({r["n_seeds"] for r in rows})
    lines += ["", f"Mean{' ± std' if max(n_seeds) > 1 else ''} over {'/'.join(map(str, n_seeds))} seed(s); "
              "best per column in bold" + ("; scores in %" if percent else "") + "."]
    if baseline:
        lines.append(f"(±x) = difference to **{baseline}**; † = p < 0.05, paired bootstrap over eval examples.")
    lines += ["", "## Models", "", "| Model | Kind | Variant | Args | Params | Seeds |", "|---|---|---|---|---|---|"]
    for label in labels:
        row = next(r for r in rows if r["label"] == label)
        variant = row["variant"] or {}
        seeds = ", ".join(str(s) for s in row["seeds"] if s is not None) or "-"
        lines.append(f"| {label} | {row['kind']} | {variant.get('name', '-')} | {variant.get('args') or '-'} | "
                     f"{params[label]} | {seeds} |")
    lines += ["", "## Details", "", "| Dataset | Metric | Model | Mean | Std | Seeds | 95% CI (1 seed) | N |",
              "|---|---|---|---|---|---|---|---|"]
    for dataset, metric in columns:
        for label in labels:
            row = cell.get((label, dataset, metric))
            if row:
                ci = "-" if not row["ci95"] else f"[{row['ci95'][0]:.4g}, {row['ci95'][1]:.4g}]"
                std = "-" if row["std"] is None else f"{row['std']:.4g}"
                lines.append(f"| {dataset} | {metric} | {label} | {row['mean']:.4g} | {std} | {row['n_seeds']} | "
                             f"{ci} | {row['n'] or '-'} |")

    latex = [r"\begin{table}[t]", r"\centering", r"\begin{tabular}{l r " + "c" * len(columns) + "}", r"\toprule",
             " & ".join(["Model", "Params"] + [_tex(f"{d} {m}") for d, m in columns]) + r" \\",
             r"\midrule"]
    for i, label in enumerate(labels):
        if i and order[label][0] != order[labels[i - 1]][0]:
            latex.append(r"\midrule")
        latex.append(" & ".join([_tex(label), params[label]]
                                + [fmt(label, c, latex=True) for c in columns]) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{_tex(title)}}}", r"\end{table}"]

    buffer = io.StringIO()
    fields = ["label", "kind", "dataset", "metric", "mean", "std", "n_seeds", "seeds", "ci95", "n", "num_parameters"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(rows, key=lambda r: (r["dataset"], r["metric"], order[r["label"]])):
        writer.writerow({**row, "seeds": " ".join(str(s) for s in row["seeds"] if s is not None),
                         "ci95": row["ci95"] or ""})
    return {"markdown": "\n".join(lines) + "\n", "latex": "\n".join(latex) + "\n", "csv": buffer.getvalue()}
