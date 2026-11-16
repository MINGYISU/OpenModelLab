"""Reasoning evaluation: accuracy, pass@k, and anatomy of <think> traces."""

from __future__ import annotations

import re
from math import comb

from finetune.training.rewards import answers_match, extract_answer, format_reward

THINK = re.compile(r"<think>([\s\S]*?)(?:</think>|$)")
# Phrases associated with self-verification / backtracking in R1-style traces.
REFLECTION = re.compile(r"\b(wait|hmm|let me (?:check|verify|re-?check)|double-check|actually|alternatively)\b",
                        re.IGNORECASE)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def trace_stats(completion: str, tokenizer=None) -> dict:
    # Many chat templates put <think> in the prompt, so a completion may only contain "...</think>".
    if "</think>" in completion and "<think>" not in completion:
        thinking = completion.split("</think>", 1)[0]
    else:
        match = THINK.search(completion)
        thinking = match.group(1) if match else ""
    count = (lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])) if tokenizer else len
    return {
        "length": count(completion),
        "think_length": count(thinking),
        "closed_think": "</think>" in completion,
        "reflections": len(REFLECTION.findall(thinking or completion)),
        "format_ok": bool(format_reward(completion, "")),
    }


def score(prompts, answers, completions, tokenizer=None, ks=(1, 4, 8, 16)) -> tuple[dict, list[dict]]:
    """`completions[i]` is a list of samples for prompt i."""
    rows, summary = [], {}
    n = len(completions[0])
    for prompt, answer, samples in zip(prompts, answers, completions):
        correct = [answers_match(extract_answer(s), answer) for s in samples]
        stats = [trace_stats(s, tokenizer) for s in samples]
        rows.append({"prompt": prompt, "answer": answer, "completions": samples, "correct": correct,
                     "stats": stats})
    flat = [(c, s) for row in rows for c, s in zip(row["correct"], row["stats"])]
    summary["accuracy"] = sum(c for c, _ in flat) / len(flat)
    for k in ks:
        if k <= n:
            summary[f"pass@{k}"] = sum(pass_at_k(n, sum(r["correct"]), k) for r in rows) / len(rows)
    for key in ("length", "think_length", "reflections"):
        summary[f"mean_{key}"] = sum(s[key] for _, s in flat) / len(flat)
        right = [s[key] for c, s in flat if c]
        wrong = [s[key] for c, s in flat if not c]
        if right and wrong:
            summary[f"mean_{key}_correct"] = sum(right) / len(right)
            summary[f"mean_{key}_incorrect"] = sum(wrong) / len(wrong)
    summary["format_rate"] = sum(s["format_ok"] for _, s in flat) / len(flat)
    summary["closed_think_rate"] = sum(s["closed_think"] for _, s in flat) / len(flat)
    summary["samples_per_prompt"] = n
    summary["prompts"] = len(rows)
    return summary, rows
