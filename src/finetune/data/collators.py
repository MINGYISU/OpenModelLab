"""Batch collators: right padding, -100 on padded labels."""

from __future__ import annotations

from finetune.data.sft import IGNORE_INDEX


def pad(sequences, value, side="right"):
    import torch

    length = max(len(s) for s in sequences)
    rows = [
        (list(s) + [value] * (length - len(s))) if side == "right" else ([value] * (length - len(s)) + list(s))
        for s in sequences
    ]
    return torch.tensor(rows, dtype=torch.long)


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        ids = [f["input_ids"] for f in features]
        return {
            "input_ids": pad(ids, self.pad_token_id),
            "attention_mask": pad([[1] * len(x) for x in ids], 0),
            "labels": pad([f["labels"] for f in features], IGNORE_INDEX),
        }


class PreferenceCollator:
    """Stacks chosen then rejected: rows [0:B] are chosen, rows [B:2B] are rejected."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        ids = [f["chosen_input_ids"] for f in features] + [f["rejected_input_ids"] for f in features]
        labels = [f["chosen_labels"] for f in features] + [f["rejected_labels"] for f in features]
        batch = {
            "input_ids": pad(ids, self.pad_token_id),
            "attention_mask": pad([[1] * len(x) for x in ids], 0),
            "labels": pad(labels, IGNORE_INDEX),
        }
        if "ref_chosen_logp" in features[0]:
            import torch

            batch["ref_logps"] = torch.tensor(
                [f["ref_chosen_logp"] for f in features] + [f["ref_rejected_logp"] for f in features],
                dtype=torch.float32,
            )
        return batch


class ContrastiveCollator:
    def __init__(self, tokenizer, max_query_length: int, max_passage_length: int):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_passage_length = max_passage_length

    def _encode(self, texts, max_length):
        # Embedding inputs are truncated (retrieval convention); EOS is appended for last-token pooling.
        eos = self.tokenizer.eos_token or ""
        return self.tokenizer(
            [t + eos for t in texts], padding=True, truncation=True, max_length=max_length,
            add_special_tokens=False, return_tensors="pt",
        )

    def __call__(self, features):
        queries = self._encode([f["query"] for f in features], self.max_query_length)
        passages = self._encode([p for f in features for p in f["passages"]], self.max_passage_length)
        return {
            "query_input_ids": queries["input_ids"], "query_attention_mask": queries["attention_mask"],
            "passage_input_ids": passages["input_ids"], "passage_attention_mask": passages["attention_mask"],
        }
