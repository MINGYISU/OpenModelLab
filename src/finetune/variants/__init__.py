"""Model variants: your own architectures, written as edits on top of a vendored reference model.

The reference models in src/finetune/models/<family>/ are never changed; they stay the
baselines (their logits are tested against upstream). A *variant* is a small module that
receives a freshly built reference model and edits it in place: add, remove, or replace
components. Select one with `model.variant: <name>` (plus optional `model.variant_args`).
See docs/RESEARCH_WORKFLOW.md and _template.py.

A variant module defines:
  NAME            id used in configs (default: the module name)
  TITLE           display name for reports and papers, e.g. "LayerScale-LM" (default: NAME)
  FAMILIES        reference families it supports, e.g. ("qwen2", "qwen3", "llama")
  ARGS            default arguments; `model.variant_args` overrides them (unknown keys are errors)
  CONFIG          optional config overrides applied before the reference model is built
  modify(model, args)
                  edit `model` (the *ForCausalLM, or the base *Model for embeddings) in place

The variant spec (name, args, source hash, and the reference config values that modify()
changed) is saved in the checkpoint's config.json under `finetune_variant`, so a trained
variant reloads through build_model / `finetune eval` / `finetune generate` with no flags.
LoRA runs save the variant's new parameters next to the adapter (variant_state.safetensors).
Files starting with "_" are ignored; a path to any .py file also works as `model.variant`.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import pkgutil
from pathlib import Path
from types import ModuleType

from torch import nn

SPEC_KEY = "finetune_variant"
SPEC_FILE = "variant.json"
STATE_FILE = "variant_state.safetensors"


def _check_module(module: ModuleType) -> ModuleType:
    families = getattr(module, "FAMILIES", None)
    if not isinstance(families, (tuple, list)) or not families:
        raise ValueError(f"Variant {module.__name__}: FAMILIES must list the supported model families")
    if not callable(getattr(module, "modify", None)):
        raise ValueError(f"Variant {module.__name__}: missing modify(model, args)")
    if not isinstance(getattr(module, "ARGS", {}), dict) or not isinstance(getattr(module, "CONFIG", {}), dict):
        raise ValueError(f"Variant {module.__name__}: ARGS and CONFIG must be dicts")
    return module


def variant_name(module: ModuleType) -> str:
    return getattr(module, "NAME", module.__name__.rsplit(".", 1)[-1])


def variant_title(module: ModuleType) -> str:
    return getattr(module, "TITLE", variant_name(module))


def available_variants() -> dict[str, ModuleType]:
    variants = {}
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            module = _check_module(importlib.import_module(f"{__name__}.{info.name}"))
            variants[variant_name(module)] = module
    return variants


def get_variant(name_or_path: str) -> ModuleType:
    path = Path(name_or_path)
    if path.suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"Variant file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"finetune_variant_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return _check_module(module)
    variants = available_variants()
    if name_or_path not in variants:
        raise ValueError(f"Unknown variant {name_or_path!r}; available: {sorted(variants)}")
    return variants[name_or_path]


def source_file(module: ModuleType) -> Path:
    return Path(inspect.getsourcefile(module))


def source_sha256(module: ModuleType) -> str:
    return hashlib.sha256(source_file(module).read_bytes()).hexdigest()


def resolve_args(module: ModuleType, args: dict | None) -> dict:
    defaults = getattr(module, "ARGS", {})
    unknown = set(args or {}) - set(defaults)
    if unknown:
        raise ValueError(f"Variant {variant_name(module)}: unknown args {sorted(unknown)}; "
                         f"accepted: {sorted(defaults)}")
    return {**defaults, **(args or {})}


# ---- helpers for variant authors -------------------------------------------------------------

def decoder(model: nn.Module) -> nn.Module:
    """The transformer stack (`*Model`) whether `model` is a *ForCausalLM or the base model."""
    inner = getattr(model, "model", None)
    return inner if inner is not None and hasattr(inner, "layers") else model


def layers(model: nn.Module) -> nn.ModuleList:
    return decoder(model).layers


def init_new(model: nn.Module, module: nn.Module) -> nn.Module:
    """Initialise a newly created module the way the reference family initialises its own
    (e.g. normal(0, initializer_range) for Linear, ones for RMSNorm weights)."""
    module.apply(model._init_weights)
    return module


def like(tensor) -> dict:
    """`device`/`dtype` kwargs matching an existing tensor, for creating new modules/parameters."""
    return {"device": tensor.device, "dtype": tensor.dtype}


# ---- applying, saving, and reloading ----------------------------------------------------------

def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def apply_variant(model: nn.Module, family: str, name_or_path: str, args: dict | None = None,
                  expected_sha256: str | None = None) -> dict:
    """Run the variant's modify() on `model`, record the spec in model.config, and return it."""
    from finetune import utils

    module = get_variant(name_or_path)
    if family not in module.FAMILIES:
        raise ValueError(f"Variant {variant_name(module)} supports {list(module.FAMILIES)}, not {family!r}")
    args = resolve_args(module, args)
    sha = source_sha256(module)
    if expected_sha256 and expected_sha256 != sha:
        utils.log(f"WARNING: variant {variant_name(module)} source changed since this checkpoint was saved "
                  f"({expected_sha256[:12]} -> {sha[:12]}); weights may not match the code")

    before_config = model.config.to_dict()
    before = {name: tuple(p.shape) for name, p in model.named_parameters()}
    module.modify(model, args)
    after = dict(model.named_parameters())
    after_config = model.config.to_dict()

    new = sorted(n for n, p in after.items() if before.get(n) != tuple(p.shape))
    removed = sorted(set(before) - set(after))
    spec = {
        "name": variant_name(module),
        "title": variant_title(module),
        "args": _jsonable(args),
        "source": str(source_file(module)) if Path(name_or_path).suffix == ".py" else None,
        "sha256": sha,
        # Reference config values that modify() changed, so a checkpoint can rebuild the
        # reference architecture before re-applying the variant.
        "reference_config": _jsonable({k: v for k, v in before_config.items()
                                       if k != SPEC_KEY and after_config.get(k) != v}),
        "new_parameters": new,
        "removed_parameters": removed,
    }
    model.config.finetune_variant = spec
    model._variant_new_params = {name: after[name] for name in new}
    added = sum(after[n].numel() for n in new)
    utils.log(f"variant {spec['title']} ({spec['name']}) on {family}: +{len(new)} tensors ({added:,} params), "
              f"-{len(removed)} tensors; total {sum(p.numel() for p in after.values()):,} params")
    return spec


def variant_owner(model: nn.Module) -> nn.Module | None:
    """The module carrying the variant spec (works through PEFT / scratch-LoRA wrappers)."""
    return next((m for m in model.modules() if hasattr(m, "_variant_new_params")), None)


def new_parameters(model: nn.Module) -> dict:
    owner = variant_owner(model)
    return owner._variant_new_params if owner is not None else {}


def save_variant_files(model: nn.Module, output_dir) -> None:
    """Write variant.json (+ variant_state.safetensors with the parameters the variant added).

    Full checkpoints already contain everything; these files make LoRA adapter directories,
    which only store the adapter, reloadable on top of the reference weights."""
    owner = variant_owner(model)
    if owner is None or output_dir is None:
        return
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / SPEC_FILE).write_text(json.dumps(owner.config.finetune_variant, indent=2), encoding="utf-8")
    params = owner._variant_new_params
    if params:
        save_file({n: p.detach().cpu().contiguous().clone() for n, p in params.items()}, str(output / STATE_FILE))


def read_spec_file(directory) -> dict | None:
    path = Path(directory) / SPEC_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_variant_state(model: nn.Module, directory) -> bool:
    """Copy variant_state.safetensors (saved next to a LoRA adapter or checkpoint) into a built variant.
    Works on wrapped models too (PEFT, scratch LoRA): parameters are matched by their variant names."""
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        return False
    from safetensors.torch import load_file

    params = new_parameters(model) or dict(model.named_parameters())
    state = load_file(str(path))
    missing = sorted(set(state) - set(params))
    if missing:
        raise ValueError(f"{path} has tensors the model does not: {missing[:5]}")
    for name, tensor in state.items():
        params[name].data.copy_(tensor.to(params[name].dtype))
    return True


def snapshot_initial_values(model: nn.Module) -> None:
    """Remember the variant's new parameters at the start of training (the frozen reference policy
    for methods that disable LoRA to get the reference, e.g. GRPO)."""
    owner = variant_owner(model)
    if owner is not None:
        owner._variant_initial = {n: p.detach().clone() for n, p in owner._variant_new_params.items()}


@contextlib.contextmanager
def initial_values(model: nn.Module):
    """Temporarily restore the snapshot taken by snapshot_initial_values()."""
    owner = variant_owner(model)
    initial = getattr(owner, "_variant_initial", None) if owner is not None else None
    if not initial:
        yield
        return
    params = owner._variant_new_params
    current = {n: params[n].data.clone() for n in initial}
    for name, value in initial.items():
        params[name].data.copy_(value)
    try:
        yield
    finally:
        for name, value in current.items():
            params[name].data.copy_(value)
