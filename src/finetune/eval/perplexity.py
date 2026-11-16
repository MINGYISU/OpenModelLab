"""Held-out perplexity: exp(mean next-token negative log-likelihood) over all tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, tokenizer, texts, max_length=1024, batch_size=8) -> dict:
    tokenizer.padding_side = "right"
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True,
                          max_length=max_length).to(model.device)
        logits = model(**batch).logits[:, :-1].float()
        targets = batch["input_ids"][:, 1:]
        mask = batch["attention_mask"][:, 1:].bool()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
        total_nll += nll[mask.reshape(-1)].sum().item()
        total_tokens += int(mask.sum())
    mean = total_nll / max(1, total_tokens)
    return {"perplexity": math.exp(mean), "nll_per_token": mean, "tokens": total_tokens}
