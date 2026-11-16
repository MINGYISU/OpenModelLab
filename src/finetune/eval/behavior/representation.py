"""Representation similarity between base and trained models: linear CKA per layer.

Linear CKA (Kornblith et al., 2019) between token representations X (base) and Y
(trained) at the same layer:  ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), after
centering. 1.0 = same representation up to rotation/scale; lower = changed.
"""

from __future__ import annotations

import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    cross = (y.T @ x).norm() ** 2
    return (cross / ((x.T @ x).norm() * (y.T @ y).norm()).clamp(min=1e-12)).item()


@torch.no_grad()
def hidden_states(model, tokenizer, texts, max_length=512, max_tokens=4096) -> list[torch.Tensor]:
    tokenizer.padding_side = "right"
    per_layer = None
    seen = 0
    for text in texts:
        batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        states = model(**batch, output_hidden_states=True).hidden_states
        if per_layer is None:
            per_layer = [[] for _ in states]
        for layer, state in enumerate(states):
            per_layer[layer].append(state[0].float().cpu())
        seen += batch["input_ids"].shape[1]
        if seen >= max_tokens:
            break
    return [torch.cat(layer)[:max_tokens] for layer in per_layer]


def cka_by_layer(base, tuned, tokenizer, texts, max_length=512, max_tokens=4096) -> dict:
    base_states = hidden_states(base, tokenizer, texts, max_length, max_tokens)
    tuned_states = hidden_states(tuned, tokenizer, texts, max_length, max_tokens)
    values = [linear_cka(x, y) for x, y in zip(base_states, tuned_states)]
    return {"tokens": base_states[0].shape[0], "cka_by_layer": values,
            "least_similar_layer": min(range(len(values)), key=values.__getitem__)}
