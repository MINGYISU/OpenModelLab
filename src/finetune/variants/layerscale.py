"""LayerScale-LM: a learnable per-channel gain on each residual branch (adds a component).

Touvron et al., 2021, "Going deeper with Image Transformers" (CaiT): x + diag(gamma) * f(x).
Adds 2 * num_layers * hidden_size parameters. With init=1.0 the model computes exactly the
reference function, so it can start from pretrained weights; CaiT uses small values
(0.1 .. 1e-6) when training from scratch.

Pattern shown: add a submodule plus a forward hook, so the reference forward() stays untouched
and LoRA / merging keep working on the existing linear layers.
"""

import torch
from torch import nn

from finetune.variants import layers, like

NAME = "layerscale"
TITLE = "LayerScale-LM"
FAMILIES = ("qwen2", "qwen3", "llama")
ARGS = {"init": 1.0, "branches": ["attn", "mlp"]}


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float, **factory):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init), **factory))

    def forward(self, x):
        return x * self.gamma


def _scale_attention(module, inputs, output):
    # Attention returns (attn_output, attn_weights); only the residual-branch output is scaled.
    return (module.layer_scale(output[0]), *output[1:])


def _scale_mlp(module, inputs, output):
    return module.layer_scale(output)


def modify(model, args):
    unknown = set(args["branches"]) - {"attn", "mlp"}
    if unknown:
        raise ValueError(f"branches must be a subset of ['attn', 'mlp'], got {sorted(unknown)}")
    hidden = model.config.hidden_size
    for layer in layers(model):
        if "attn" in args["branches"]:
            layer.self_attn.layer_scale = LayerScale(hidden, args["init"], **like(layer.self_attn.o_proj.weight))
            layer.self_attn.register_forward_hook(_scale_attention)
        if "mlp" in args["branches"]:
            layer.mlp.layer_scale = LayerScale(hidden, args["init"], **like(layer.mlp.down_proj.weight))
            layer.mlp.register_forward_hook(_scale_mlp)
