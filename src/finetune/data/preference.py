"""Preference data for DPO-style training: (prompt, chosen, rejected).

Accepted record shapes:
  {"prompt": "question", "chosen": "good answer", "rejected": "bad answer"}
  {"prompt": [messages...], "chosen": [assistant msg], "rejected": [assistant msg]}  # UltraFeedback-style
Prompt tokens are masked with -100 so only response log-probs are compared.
"""

from __future__ import annotations

from finetune.data.sft import IGNORE_INDEX, render


def _as_messages(value, role):
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    if isinstance(value, list) and all(isinstance(m, dict) for m in value):
        return value
    raise ValueError(f"Expected a string or a list of messages, got {type(value).__name__}")


def _strip_prompt(prompt, response):
    # UltraFeedback stores chosen/rejected as full conversations that repeat the prompt.
    if response[: len(prompt)] == prompt:
        response = response[len(prompt):]
    if len(response) != 1 or response[0].get("role") != "assistant":
        raise ValueError("chosen/rejected must be a single assistant response")
    return response


def tokenize_preference(record, tokenizer, max_length, max_prompt_length, **template_kwargs) -> dict:
    prompt = _as_messages(record["prompt"], "user")
    prompt_text = render(tokenizer, prompt, add_generation_prompt=True, **template_kwargs)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(f"Prompt has {len(prompt_ids)} tokens > max_prompt_length={max_prompt_length}")
    example = {}
    for side in ("chosen", "rejected"):
        response = _strip_prompt(prompt, _as_messages(record[side], "assistant"))
        full_text = render(tokenizer, prompt + response, **template_kwargs)
        if not full_text.startswith(prompt_text):
            raise ValueError("Chat template output for prompt+response does not start with the prompt")
        ids = prompt_ids + tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            raise ValueError(f"{side} sequence has {len(ids)} tokens > max_length={max_length}")
        example[f"{side}_input_ids"] = ids
        example[f"{side}_labels"] = [IGNORE_INDEX] * len(prompt_ids) + ids[len(prompt_ids):]
    return example


def build_preference_dataset(records, tokenizer, data_cfg, dpo_cfg) -> list[dict]:
    examples = []
    for index, record in enumerate(records, 1):
        try:
            examples.append(tokenize_preference(
                record, tokenizer, data_cfg.max_length, dpo_cfg.max_prompt_length,
                **data_cfg.chat_template_kwargs,
            ))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Record {index}: {error}") from error
    return examples
