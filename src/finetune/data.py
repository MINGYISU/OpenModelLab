"""Dependency-free validation of local conversation datasets."""

import json
from pathlib import Path


def load_conversations(path):
    """Read JSONL conversations ending in an assistant answer."""
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{location}: expected a nonempty messages list")
            roles = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(f"{location}: each message must be an object")
                role, content = message.get("role"), message.get("content")
                if role not in ("system", "user", "assistant"):
                    raise ValueError(f"{location}: unsupported message role")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"{location}: message content must be nonempty text")
                roles.append(role)
            turns = roles[1:] if roles[0] == "system" else roles
            if (
                len(turns) < 2
                or len(turns) % 2
                or any(role != ("user" if i % 2 == 0 else "assistant")
                       for i, role in enumerate(turns))
            ):
                raise ValueError(
                    f"{location}: expected alternating user/assistant turns, "
                    "optionally preceded by one system message"
                )
            records.append({"messages": messages})
    if not records:
        raise ValueError(f"{path}: dataset contains no conversations")
    return records
