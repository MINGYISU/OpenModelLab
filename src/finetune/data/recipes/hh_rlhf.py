"""Example preference recipe: Anthropic HH-RLHF.

`chosen` and `rejected` are raw transcripts ("\\n\\nHuman: ...\\n\\nAssistant: ...") that share
every turn except the last. The recipe parses them into messages: the kind of
dataset-specific parsing a recipe is for.
"""

import re

NAME = "hh_rlhf"
FORMAT = "preference"
SOURCE = {"hf": "Anthropic/hh-rlhf", "split": "train"}
EVAL_SOURCE = {"hf": "Anthropic/hh-rlhf", "split": "test"}

_TURN = re.compile(r"\n\n(Human|Assistant): ")
_ROLES = {"Human": "user", "Assistant": "assistant"}


def parse_transcript(text: str) -> list[dict] | None:
    # With a capture group, re.split returns ["", role, content, role, content, ...].
    parts = _TURN.split("\n\n" + text.lstrip("\n"))
    if len(parts) < 3 or parts[0].strip():
        return None
    messages = [{"role": _ROLES[role], "content": content.strip()}
                for role, content in zip(parts[1::2], parts[2::2])]
    return messages if all(m["content"] for m in messages) else None


def convert(row: dict):
    chosen, rejected = parse_transcript(row["chosen"]), parse_transcript(row["rejected"])
    if not chosen or not rejected or chosen[:-1] != rejected[:-1]:
        return None
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        return None
    return {"prompt": chosen[:-1], "chosen": [chosen[-1]], "rejected": [rejected[-1]]}
