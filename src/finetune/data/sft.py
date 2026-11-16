"""SFT: render conversations with the tokenizer's chat template and build loss masks.

`loss_on="all"`: every non-padding token is supervised (a common simple baseline).
`loss_on="assistant"`: only assistant turns (their content plus the end-of-turn
token) are supervised; system/user turns and the assistant header get -100.

Assistant spans are found by rendering growing prefixes of the conversation and
mapping character spans onto token offsets. This works for any template whose
rendering of a shorter conversation is a prefix of the longer one; templates that
rewrite history (e.g. Qwen3 stripping earlier <think> blocks) are rejected with a
clear error rather than producing a wrong mask.
"""

from __future__ import annotations

IGNORE_INDEX = -100
ROLES = {"system", "user", "assistant"}


def validate_conversation(record: dict, index: int) -> list[dict]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {index}: 'messages' must be a non-empty list")
    for turn, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise ValueError(f"Record {index}, message {turn}: role must be one of {sorted(ROLES)}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Record {index}, message {turn}: content must be a non-empty string")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Record {index}: conversation must end with an assistant message")
    return messages


def render(tokenizer, messages, add_generation_prompt=False, **template_kwargs) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs,
    )


def assistant_char_spans(tokenizer, messages, **template_kwargs) -> tuple[str, list[tuple[int, int]]]:
    full = render(tokenizer, messages, **template_kwargs)
    spans = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start_text = render(tokenizer, messages[:i], add_generation_prompt=True, **template_kwargs)
        end_text = render(tokenizer, messages[: i + 1], **template_kwargs)
        if not (full.startswith(end_text) and end_text.startswith(start_text)):
            raise ValueError(
                "Chat template is not prefix-consistent (it rewrites earlier turns); "
                "assistant-only masking needs single-turn data or loss_on='all'"
            )
        spans.append((len(start_text), len(end_text)))
    return full, spans


def tokenize_conversation(tokenizer, messages, max_length, loss_on="all", **template_kwargs) -> dict:
    if not tokenizer.chat_template:
        raise ValueError("Use a tokenizer with a chat template (an instruct/chat checkpoint)")
    if loss_on not in {"all", "assistant"}:
        raise ValueError("data.loss_on must be 'all' or 'assistant'")
    if loss_on == "assistant":
        text, spans = assistant_char_spans(tokenizer, messages, **template_kwargs)
    else:
        text, spans = render(tokenizer, messages, **template_kwargs), []
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=loss_on == "assistant")
    input_ids = encoded["input_ids"]
    if len(input_ids) > max_length:
        raise ValueError(f"Conversation has {len(input_ids)} tokens > max_length={max_length}")
    if loss_on == "all":
        labels = list(input_ids)
    else:
        labels = [
            token if any(start <= offset[0] < end for start, end in spans) else IGNORE_INDEX
            for token, offset in zip(input_ids, encoded["offset_mapping"])
        ]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def build_sft_dataset(records, tokenizer, data_cfg) -> list[dict]:
    examples = []
    for index, record in enumerate(records, 1):
        messages = validate_conversation(record, index)
        try:
            examples.append(tokenize_conversation(
                tokenizer, messages, data_cfg.max_length, data_cfg.loss_on, **data_cfg.chat_template_kwargs,
            ))
        except ValueError as error:
            raise ValueError(f"Record {index}: {error}") from error
    return examples
