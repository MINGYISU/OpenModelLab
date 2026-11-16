"""Logit lens (nostalgebraist, 2020): decode each layer's residual stream with the final norm + LM head.

For prompts with a known answer, we track the probability and rank of the answer's
first token at every layer. Comparing base vs. trained shows *where* in depth the
answer becomes decodable and whether training moves it earlier.
"""

from __future__ import annotations

import torch


def _decoder_parts(model):
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "norm") or not hasattr(model, "lm_head"):
        raise ValueError("Expected a *ForCausalLM with .model.norm and .lm_head")
    return inner.norm, model.lm_head


@torch.no_grad()
def logit_lens(model, tokenizer, records, chat: bool = True) -> dict:
    """records: [{"prompt": ..., "answer": ...}]. Returns per-layer mean prob/rank of the answer token."""
    norm, head = _decoder_parts(model)
    probs, ranks, count = None, None, 0
    for record in records:
        if chat and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": record["prompt"]}],
                                                   tokenize=False, add_generation_prompt=True)
        else:
            prompt = record["prompt"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + str(record["answer"]), add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) == len(prompt_ids):
            continue
        target = full_ids[len(prompt_ids)]
        inputs = torch.tensor([prompt_ids], device=model.device)
        hidden_states = model(input_ids=inputs, output_hidden_states=True).hidden_states
        # hidden_states[0] = embeddings, [1..L-1] = residual stream after each layer, [L] = already normed.
        layer_probs, layer_ranks = [], []
        for index, hidden in enumerate(hidden_states):
            last = hidden[0, -1]
            logits = head(last if index == len(hidden_states) - 1 else norm(last)).float()
            distribution = logits.softmax(-1)
            layer_probs.append(distribution[target].item())
            layer_ranks.append(int((logits > logits[target]).sum()) + 1)
        probs = layer_probs if probs is None else [a + b for a, b in zip(probs, layer_probs)]
        ranks = layer_ranks if ranks is None else [a + b for a, b in zip(ranks, layer_ranks)]
        count += 1
    if not count:
        raise ValueError("No usable prompt/answer records (answer must tokenize as a continuation)")
    first_top1 = next((i for i, r in enumerate(ranks) if r / count < 1.5), None)
    return {"examples": count, "mean_answer_prob_by_layer": [p / count for p in probs],
            "mean_answer_rank_by_layer": [r / count for r in ranks], "first_layer_mean_rank_1": first_top1}
