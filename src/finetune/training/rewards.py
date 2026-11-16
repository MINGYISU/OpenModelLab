"""Verifiable reward functions for RL (and answer extraction shared with evaluation)."""

from __future__ import annotations

import re

ANSWER_TAG = re.compile(r"<answer>([\s\S]*?)</answer>")
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
THINK_THEN_ANSWER = re.compile(r"</think>\s*<answer>[\s\S]*?</answer>\s*$")


def extract_answer(text: str) -> str | None:
    """<answer>...</answer>, else the last \\boxed{...}, else the last number in the text."""
    for pattern in (ANSWER_TAG, BOXED):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    numbers = NUMBER.findall(text)
    return numbers[-1] if numbers else None


def answers_match(predicted: str | None, reference: str) -> bool:
    if predicted is None:
        return False
    predicted, reference = predicted.replace(",", "").strip(), reference.replace(",", "").strip()
    if predicted == reference:
        return True
    try:
        return abs(float(predicted) - float(reference)) < 1e-6
    except ValueError:
        numbers = NUMBER.findall(predicted)
        return len(numbers) == 1 and numbers[0] == reference


def correctness_reward(completion: str, answer: str) -> float:
    return 1.0 if answers_match(extract_answer(completion), answer) else 0.0


def format_reward(completion: str, answer: str) -> float:
    """1 if the completion closes its reasoning with </think> and then gives <answer>...</answer>."""
    return 1.0 if THINK_THEN_ANSWER.search(completion) else 0.0


def length_penalty(completion: str, answer: str, target_chars: int = 2000) -> float:
    return -min(1.0, max(0, len(completion) - target_chars) / target_chars)


def countdown_reward(completion: str, answer: str) -> float:
    """Countdown (TinyZero): the <answer> must be an arithmetic expression that uses each given
    number exactly once and evaluates to the target. `answer` is JSON {"target": t, "nums": [...]}."""
    import json

    spec = json.loads(answer)
    matches = ANSWER_TAG.findall(completion)
    if not matches:
        return 0.0
    expression = matches[-1].split("=")[0].strip()
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return 0.0
    if sorted(int(n) for n in re.findall(r"\d+", expression)) != sorted(spec["nums"]):
        return 0.0
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # input restricted to digits/operators above
    except Exception:
        return 0.0
    return 1.0 if abs(value - spec["target"]) < 1e-6 else 0.0


REWARDS = {"correctness": correctness_reward, "format": format_reward, "length": length_penalty,
           "countdown": countdown_reward}


def total_reward(completion: str, answer: str, names, weights=None) -> tuple[float, dict]:
    weights = weights or [1.0] * len(names)
    parts = {name: REWARDS[name](completion, answer) for name in names}
    return sum(w * parts[n] for n, w in zip(names, weights)), parts
