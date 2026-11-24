"""Pruned-LM: remove decoder layers (removes components).

Gromov et al., 2024, "The Unreasonable Ineffectiveness of the Deeper Layers": many deep layers
can be dropped from a pretrained LLM with little loss after a short fine-tune ("healing").
`drop` lists layer indices to remove (negative indices count from the end).

Pattern shown: change the architecture *and* the config fields that describe it. The reference
values are recorded automatically, so checkpoints rebuild the full model and drop the same layers.
"""

from torch import nn

from finetune.variants import decoder

NAME = "drop_layers"
TITLE = "Pruned-LM"
FAMILIES = ("qwen2", "qwen3", "llama")
ARGS = {"drop": [-2]}


def modify(model, args):
    stack = decoder(model)
    total = len(stack.layers)
    drop = {index % total for index in args["drop"]}
    if not drop or len(drop) >= total:
        raise ValueError(f"drop must remove between 1 and {total - 1} of {total} layers, got {args['drop']}")
    keep = [i for i in range(total) if i not in drop]
    stack.layers = nn.ModuleList(stack.layers[i] for i in keep)
    for new_index, layer in enumerate(stack.layers):
        layer.self_attn.layer_idx = new_index  # KV-cache slot

    config = model.config
    config.num_hidden_layers = len(keep)
    if getattr(config, "layer_types", None):
        config.layer_types = [config.layer_types[i] for i in keep]
