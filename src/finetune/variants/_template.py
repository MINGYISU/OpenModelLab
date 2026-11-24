"""TEMPLATE for a new model variant (your own architecture). Copy it to `<your_variant>.py` in
this folder (no leading "_"), or anywhere and pass the path as `model.variant`. Then:

  finetune variants                                  # list variants and check yours loads
  finetune variants <your_variant> --model tiny:qwen2   # build it on a tiny model and print the diff
  finetune train configs/smoke/sft.yaml --set model.variant=<your_variant>   # 4-step CPU smoke run
  finetune train <config> --set model.variant=<your_variant> --set model.variant_args.alpha=0.5

How it works: build_model builds the reference model (pretrained weights, random init with
`model.init: scratch`, or `tiny:<family>`), then calls modify(model, args). The reference
source in src/finetune/models/ is never edited, so references stay valid baselines.

Rules of thumb
  * Keep reference parameter names where the component is the same, so pretrained weights,
    LoRA target_modules ("q_proj", "down_proj", ...) and analysis tools still line up.
  * New tensors must be created on the same device/dtype as the model: use `**like(tensor)`.
  * Initialise new modules deliberately: `init_new(model, module)` uses the family's own init;
    an identity-preserving init (zeros on a residual branch, ones on a gain) lets a variant
    start from pretrained weights without destroying them.
  * If you change shapes or depth, also update the config fields that describe them
    (e.g. num_hidden_layers, layer_types); the reference values are recorded automatically.
  * modify() must be deterministic given (reference config, args): checkpoints are reloaded by
    rebuilding the reference model, re-running modify(), then loading the trained weights.
  * New parameters are reported at build time, trained even when LoRA freezes the rest, and
    saved next to LoRA adapters (variant_state.safetensors).
"""

from torch import nn

from finetune.variants import init_new, layers, like  # noqa: F401  (helpers you will likely need)

NAME = "my_variant"                  # TODO: id used in configs: model.variant: my_variant
TITLE = "MyModel"                    # TODO: the name your paper uses for this model
FAMILIES = ("qwen2", "qwen3", "llama")  # reference families whose module layout you rely on
ARGS = {"alpha": 1.0}                # TODO: knobs for ablations; overridden by model.variant_args
CONFIG = {}                          # optional config overrides applied before building (e.g. sizes)


def modify(model, args):
    """Edit `model` in place. Common patterns:

    1. Replace a component: write a module (keep the reference attribute names) and assign it.
           layer.mlp = MyMLP(...)                                       # see no_glu.py
    2. Add a component without touching forward(): attach a module and a forward hook.
           layer.self_attn.extra = MyModule(...); layer.self_attn.register_forward_hook(fn)  # layerscale.py
    3. Remove components: rebuild the ModuleList and fix the config.     # see drop_layers.py
    4. Change how a layer computes (e.g. parallel attention + MLP): subclass the family's
       decoder layer with a new forward() and swap `layer.__class__`, or wrap the layer:
           from finetune.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
           class ParallelLayer(Qwen2DecoderLayer):
               def forward(self, hidden_states, **kwargs): ...
           for layer in layers(model): layer.__class__ = ParallelLayer
    """
    for layer in layers(model):
        # TODO: your change. Example: a zero-initialised extra residual branch (starts as identity).
        branch = nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=False,
                           **like(layer.mlp.down_proj.weight))
        nn.init.zeros_(branch.weight)
        layer.mlp.extra = branch
        layer.mlp.register_forward_hook(
            lambda module, inputs, output: output + args["alpha"] * module.extra(inputs[0]))
