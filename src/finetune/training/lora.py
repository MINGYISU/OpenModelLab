"""LoRA: PEFT integration plus a from-scratch implementation for learning.

LoRA freezes W and learns a low-rank update:  y = W x + (alpha / r) * B(A(dropout(x)))
A is initialised randomly (Kaiming), B with zeros, so training starts exactly at
the base model. rsLoRA scales by alpha / sqrt(r) instead, which keeps update
magnitude stable as r grows.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, rslora: bool = False):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / math.sqrt(r) if rslora else alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, r, bias=False, dtype=base.weight.dtype, device=base.weight.device)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, dtype=base.weight.dtype, device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight += self.delta_weight().to(self.base.weight.dtype)
        return self.base


def _targets(model, target_modules):
    lm_head = getattr(model, "lm_head", None)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module is lm_head:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if target_modules == "all-linear" or leaf in target_modules:
            yield name, module


def apply_scratch_lora(model, lora_cfg):
    """Freeze the model and wrap target nn.Linear layers with LoRALinear in place."""
    for param in model.parameters():
        param.requires_grad_(False)
    replaced = list(_targets(model, lora_cfg.target_modules))
    if not replaced:
        raise ValueError(f"No nn.Linear modules matched target_modules={lora_cfg.target_modules!r}")
    for name, module in replaced:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, lora_cfg.r, lora_cfg.alpha, lora_cfg.dropout, lora_cfg.use_rslora))
    return model


def merge_scratch_lora(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, leaf, module.merge())
    return model


def merge_for_save(model, output_dir=None):
    """Scratch LoRA: store adapter weights separately, then merge so the saved model
    loads with the vendored class like any checkpoint. PEFT models save adapters natively.
    Model variants also write variant.json / variant_state.safetensors here."""
    from finetune.variants import save_variant_files

    save_variant_files(model, output_dir)
    adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n}
    if adapters and any(isinstance(m, LoRALinear) for m in model.modules()):
        if output_dir is not None:
            from pathlib import Path

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            torch.save(adapters, Path(output_dir) / "scratch_lora_adapters.pt")
        merge_scratch_lora(model)
    return model


def apply_lora(model, lora_cfg, task_type: str = "CAUSAL_LM"):
    if not lora_cfg.enabled:
        return model
    if lora_cfg.impl == "scratch":
        return apply_scratch_lora(model, lora_cfg)
    if lora_cfg.impl != "peft":
        raise ValueError("lora.impl must be 'peft' or 'scratch'")
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        task_type=task_type, r=lora_cfg.r, lora_alpha=lora_cfg.alpha, lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules, use_dora=lora_cfg.use_dora, use_rslora=lora_cfg.use_rslora,
    )
    return get_peft_model(model, config)


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


@contextlib.contextmanager
def lora_disabled(model):
    """Run the frozen base model (the reference policy) by switching adapters off. Parameters
    added by a model variant (trained alongside LoRA) are swapped back to their initial values."""
    from finetune.variants import initial_values

    with initial_values(model):
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield
            return
        modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
        if not modules:
            raise ValueError("Model has no LoRA adapters to disable")
        scalings = [m.scaling for m in modules]
        for module in modules:
            module.scaling = 0.0
        try:
            yield
        finally:
            for module, scaling in zip(modules, scalings):
                module.scaling = scaling
