"""RL with verifiable rewards: prompts with reference answers.

Record: {"prompt": "question text", "answer": "42"}. GSM8K-style answers
("... #### 42") are normalised to the final number.
"""

from __future__ import annotations

# DeepSeek-R1-Zero style prompt for *base* models (no chat template).
R1_ZERO_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with "
    "the answer. The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>.\nUser: {prompt}\nAssistant: <think>"
)

DEFAULT_SYSTEM_PROMPT = (
    "Think step by step inside <think> </think>, then give only the final answer inside <answer> </answer>."
)


def normalize_answer(answer) -> str:
    text = str(answer)
    if "####" in text:
        text = text.split("####")[-1]
    return text.strip().replace(",", "")


def format_prompt(tokenizer, prompt: str, grpo_cfg, template_kwargs=None) -> str:
    if not grpo_cfg.use_chat_template:
        return R1_ZERO_TEMPLATE.format(prompt=prompt)
    messages = []
    system = grpo_cfg.system_prompt if grpo_cfg.system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **(template_kwargs or {}),
    )


def countdown_prompt(target: int, nums: list[int]) -> str:
    return (f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic "
            "operations (+, -, *, /) and each number must be used exactly once. Put the final equation inside "
            "<answer> </answer>, for example <answer> (1 + 2) / 3 </answer>.")


def build_rl_dataset(records, tokenizer, grpo_cfg, template_kwargs=None) -> list[dict]:
    """Accepts {"prompt"|"question", "answer"} records, or Countdown {"target", "nums"} records
    (the answer becomes JSON for the `countdown` reward)."""
    import json

    examples = []
    for index, record in enumerate(records, 1):
        if "target" in record and "nums" in record:
            prompt = countdown_prompt(record["target"], list(record["nums"]))
            answer = json.dumps({"target": record["target"], "nums": list(record["nums"])})
        else:
            prompt = record.get("prompt") or record.get("question")
            if not isinstance(prompt, str) or record.get("answer") is None:
                raise ValueError(f"Record {index}: needs string 'prompt' (or 'question') and an 'answer'")
            answer = normalize_answer(record["answer"])
        examples.append({"prompt_text": format_prompt(tokenizer, prompt, grpo_cfg, template_kwargs), "answer": answer})
    return examples
