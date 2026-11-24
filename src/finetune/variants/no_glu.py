"""NoGLU-LM: replace the gated SwiGLU MLP with a plain two-layer MLP (replaces a component).

Ablates the gating from Shazeer, 2020, "GLU Variants Improve Transformer":
  reference  down(act(gate(x)) * up(x))      variant  down(act(up(x)))
With `match_params: true` the hidden width grows 1.5x so the MLP keeps the reference parameter
count (the fair comparison in that paper); new weights are then freshly initialised. Otherwise
up_proj / down_proj keep their (pretrained) weights and gate_proj is removed.

Pattern shown: swap a submodule for your own class (keep the reference names when you can, so
LoRA target_modules and analysis tools still match).
"""

from torch import nn
from transformers.activations import ACT2FN

from finetune.variants import init_new, layers, like

NAME = "no_glu"
TITLE = "NoGLU-LM"
FAMILIES = ("qwen2", "qwen3", "llama")
ARGS = {"activation": "gelu", "match_params": False}


class PlainMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int, activation: str, bias: bool = False, **factory):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias, **factory)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias, **factory)
        self.act_fn = ACT2FN[activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


def modify(model, args):
    hidden = model.config.hidden_size
    for layer in layers(model):
        old = layer.mlp
        width = old.up_proj.out_features
        new_width = width * 3 // 2 if args["match_params"] else width
        mlp = PlainMLP(hidden, new_width, args["activation"], bias=old.up_proj.bias is not None,
                       **like(old.up_proj.weight))
        if new_width == width:
            mlp.up_proj.load_state_dict(old.up_proj.state_dict())
            mlp.down_proj.load_state_dict(old.down_proj.state_dict())
        else:
            init_new(model, mlp)
        layer.mlp = mlp
