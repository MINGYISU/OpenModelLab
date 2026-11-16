"""Optimizer parameter groups and learning-rate schedules.

Schedules (all start with linear warmup from 0 to peak LR):
  constant: hold peak LR.
  linear:   decay linearly to min_lr_ratio * peak.
  cosine:   half-cosine decay to min_lr_ratio * peak.
  wsd:      warmup-stable-decay (MiniCPM); hold peak, then decay linearly over the
            final `decay_ratio` of training. Lets you branch "decayed" checkpoints
            from any point of a long stable phase.
"""

from __future__ import annotations

import math

NO_DECAY_KEYWORDS = ("bias", "norm", "embed_tokens", "e_score_correction_bias")


def param_groups(model, weight_decay: float):
    """No weight decay on biases, norms, and embeddings (standard LLM practice)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or any(k in name.lower() for k in NO_DECAY_KEYWORDS) else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model, train_cfg):
    import torch

    return torch.optim.AdamW(
        param_groups(model, train_cfg.weight_decay), lr=train_cfg.lr,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2), eps=train_cfg.adam_eps,
        fused=torch.cuda.is_available(),
    )


def lr_lambda(step: int, *, name: str, warmup: int, total: int, min_ratio: float = 0.0,
              decay_ratio: float = 0.1) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if name == "constant":
        return 1.0
    if name == "wsd":
        decay_start = int(total * (1 - decay_ratio))
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total - decay_start)
    else:
        progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if name == "linear" or name == "wsd":
        return min_ratio + (1 - min_ratio) * (1 - progress)
    if name == "cosine":
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown scheduler {name!r}; use constant, linear, cosine, or wsd")


def warmup_steps(train_cfg, total_steps: int) -> int:
    return train_cfg.warmup_steps if train_cfg.warmup_steps > 0 else math.ceil(train_cfg.warmup_ratio * total_steps)


def build_scheduler(optimizer, train_cfg, total_steps: int):
    import functools

    import torch

    function = functools.partial(
        lr_lambda, name=train_cfg.scheduler, warmup=warmup_steps(train_cfg, total_steps),
        total=total_steps, min_ratio=train_cfg.min_lr_ratio, decay_ratio=train_cfg.decay_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, function)
