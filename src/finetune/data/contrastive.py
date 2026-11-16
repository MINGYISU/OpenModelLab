"""Contrastive data for embedding training: (query, positive, [hard negatives]).

Record: {"query": "...", "positive": "...", "negatives": ["...", ...]}. Aliases:
positive <- answer | pos | document, negatives <- negative | neg (a string or a list).
Every example in a batch must carry the same number of hard negatives so passages
form a rectangular [batch, 1 + num_negatives] layout.
"""

from __future__ import annotations


def _first(record, keys):
    return next((record[k] for k in keys if record.get(k) is not None), None)


def build_contrastive_dataset(records, embed_cfg, num_negatives: int | None = None) -> list[dict]:
    examples = []
    for index, record in enumerate(records, 1):
        query = record.get("query")
        positive = _first(record, ("positive", "answer", "pos", "document"))
        negatives = _first(record, ("negatives", "negative", "neg")) or []
        negatives = [negatives] if isinstance(negatives, str) else list(negatives)
        if not isinstance(query, str) or not isinstance(positive, str):
            raise ValueError(f"Record {index}: needs string 'query' and 'positive'")
        if num_negatives is None:
            num_negatives = len(negatives)
        if len(negatives) < num_negatives:
            raise ValueError(f"Record {index}: expected at least {num_negatives} negatives")
        examples.append({
            "query": embed_cfg.query_prefix + query,
            "passages": [embed_cfg.passage_prefix + p for p in [positive, *negatives[:num_negatives]]],
        })
    return examples
