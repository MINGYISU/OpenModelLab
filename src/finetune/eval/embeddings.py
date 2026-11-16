"""Embedding evaluation: retrieval metrics on (query, positive[, negatives]) JSONL, plus an MTEB adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from finetune.data.contrastive import _first
from finetune.training.embedding import encode

POSITIVE, NEGATIVES = ("positive", "answer", "pos", "document"), ("negatives", "negative", "neg")


def embedding_settings(path: str) -> dict:
    config = Path(path) / "embedding_config.json"
    return json.loads(config.read_text()) if config.exists() else {}


@torch.no_grad()
def embed_texts(model, tokenizer, texts, pooling="mean", max_length=256, batch_size=32, normalize=True):
    tokenizer.padding_side = "right"
    eos = tokenizer.eos_token or ""
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer([t + eos for t in texts[start:start + batch_size]], padding=True, truncation=True,
                          max_length=max_length, add_special_tokens=False, return_tensors="pt").to(model.device)
        outputs.append(encode(model, batch["input_ids"], batch["attention_mask"], pooling, normalize).cpu())
    return torch.cat(outputs)


def retrieval_metrics(records, model, tokenizer, pooling="mean", settings=None, batch_size=32) -> dict:
    """Corpus = all positives and negatives in the file; each query has exactly one relevant passage."""
    settings = settings or {}
    query_prefix, passage_prefix = settings.get("query_prefix", ""), settings.get("passage_prefix", "")
    corpus, index = [], {}
    positives = [_first(r, POSITIVE) for r in records]
    for record, positive in zip(records, positives):
        negatives = _first(record, NEGATIVES) or []
        for passage in [positive, *([negatives] if isinstance(negatives, str) else negatives)]:
            if passage not in index:
                index[passage] = len(corpus)
                corpus.append(passage)
    queries = embed_texts(model, tokenizer, [query_prefix + r["query"] for r in records], pooling,
                          settings.get("max_query_length", 128), batch_size)
    passages = embed_texts(model, tokenizer, [passage_prefix + p for p in corpus], pooling,
                           settings.get("max_passage_length", 256), batch_size)
    scores = queries @ passages.T
    ranks = []
    for row, positive in enumerate(positives):
        target = index[positive]
        ranks.append(int((scores[row] > scores[row, target]).sum()) + 1)
    metrics = {f"recall@{k}": sum(r <= k for r in ranks) / len(ranks) for k in (1, 5, 10)}
    metrics["mrr"] = sum(1 / r for r in ranks) / len(ranks)
    metrics["ndcg@10"] = sum(1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks) / len(ranks)
    metrics.update(queries=len(records), corpus=len(corpus))
    return metrics


class MTEBEncoder:
    """Minimal adapter so `mteb` can evaluate our vendored embedding models:
        import mteb; mteb.MTEB(tasks=mteb.get_tasks(tasks=["SciFact"])).run(MTEBEncoder(model, tok))"""

    def __init__(self, model, tokenizer, pooling="mean", max_length=512, query_prefix="", passage_prefix=""):
        self.model, self.tokenizer, self.pooling, self.max_length = model, tokenizer, pooling, max_length
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix

    def encode(self, sentences, batch_size=32, prompt_type=None, **kwargs):
        prefix = self.query_prefix if str(prompt_type).endswith("query") else self.passage_prefix
        texts = [prefix + s for s in sentences]
        return embed_texts(self.model, self.tokenizer, texts, self.pooling, self.max_length, batch_size).numpy()
