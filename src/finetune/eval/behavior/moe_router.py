"""Mixture-of-experts routing analysis (DeepSeek-V2/V3): expert usage and its shift after training."""

from __future__ import annotations

import math
from collections import defaultdict

import torch


@torch.no_grad()
def expert_usage(model, tokenizer, texts, max_length=512) -> dict[str, list[float]]:
    """Fraction of routed (token, slot) assignments per expert, per MoE layer."""
    counts, hooks = defaultdict(lambda: None), []
    routers = [(name, module) for name, module in model.named_modules() if type(module).__name__.endswith("TopkRouter")]
    if not routers:
        raise ValueError("No MoE routers found (expected DeepSeek-V2/V3 *TopkRouter modules)")

    def hook(name, num_experts):
        def capture(module, args, output):
            indices = output[2].reshape(-1)
            histogram = torch.bincount(indices, minlength=num_experts).float().cpu()
            counts[name] = histogram if counts[name] is None else counts[name] + histogram
        return capture

    for name, module in routers:
        hooks.append(module.register_forward_hook(hook(name, module.num_experts)))
    try:
        tokenizer.padding_side = "right"
        for text in texts:
            batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            model(**batch)
    finally:
        for handle in hooks:
            handle.remove()
    return {name: (h / h.sum()).tolist() for name, h in counts.items()}


def entropy(p) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


def router_shift(base, tuned, tokenizer, texts, max_length=512) -> dict:
    before, after = expert_usage(base, tokenizer, texts, max_length), expert_usage(tuned, tokenizer, texts, max_length)
    layers = {}
    for name in before:
        p, q = before[name], after[name]
        layers[name] = {
            "entropy_base": entropy(p), "entropy_tuned": entropy(q), "max_entropy": math.log(len(p)),
            "total_variation": 0.5 * sum(abs(a - b) for a, b in zip(p, q)),
            "usage_base": p, "usage_tuned": q,
        }
    return {"layers": layers,
            "mean_total_variation": sum(v["total_variation"] for v in layers.values()) / max(1, len(layers))}
