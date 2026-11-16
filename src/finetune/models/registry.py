"""Map model families to vendored classes and build models without AutoModel.

`model_type` in a checkpoint's config.json selects the family. DeepSeek-R1-Distill-Qwen
checkpoints report `qwen2`, SmolLM2/Llama-3 report `llama`, DeepSeek-V2-Lite reports
`deepseek_v2`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

FAMILY_CLASSES = {
    "qwen2": ("Qwen2Config", "Qwen2Model", "Qwen2ForCausalLM"),
    "qwen3": ("Qwen3Config", "Qwen3Model", "Qwen3ForCausalLM"),
    "llama": ("LlamaConfig", "LlamaModel", "LlamaForCausalLM"),
    "deepseek_v2": ("DeepseekV2Config", "DeepseekV2Model", "DeepseekV2ForCausalLM"),
    "deepseek_v3": ("DeepseekV3Config", "DeepseekV3Model", "DeepseekV3ForCausalLM"),
}

_COMMON_TINY = dict(
    vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
    tie_word_embeddings=False,
)

# Tiny random-init configs for CPU tests and architecture experiments.
TINY_CONFIGS = {
    "qwen2": dict(_COMMON_TINY),
    "qwen3": dict(_COMMON_TINY, head_dim=16),
    "llama": dict(_COMMON_TINY),
    "deepseek_v2": dict(
        _COMMON_TINY, num_key_value_heads=4, moe_intermediate_size=32, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1, kv_lora_rank=16,
        q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
        n_group=1, topk_group=1, topk_method="greedy",
    ),
    "deepseek_v3": dict(
        _COMMON_TINY, num_key_value_heads=4, moe_intermediate_size=32, n_routed_experts=4,
        n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1, kv_lora_rank=16,
        q_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16,
        n_group=2, topk_group=1, num_mtp_layers=0,
    ),
}


@dataclass(frozen=True)
class Family:
    name: str
    config_class: type
    base_model_class: type
    causal_lm_class: type


def get_family(name: str) -> Family:
    if name not in FAMILY_CLASSES:
        raise ValueError(f"No vendored model for family {name!r}; available: {sorted(FAMILY_CLASSES)}")
    module = importlib.import_module(f"finetune.models.{name}")
    return Family(name, *(getattr(module, cls) for cls in FAMILY_CLASSES[name]))


def read_model_type(path: str, revision: str | None = None) -> str:
    local = Path(path) / "config.json"
    if local.exists():
        return json.loads(local.read_text())["model_type"]
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(path, "config.json", revision=revision)).read_text())["model_type"]


def resolve_family(path: str, family: str | None = None, revision: str | None = None) -> Family:
    if family in TINY_CONFIGS and path.startswith("tiny:"):
        return get_family(family)
    if path.startswith("tiny:"):
        return get_family(path.split(":", 1)[1])
    detected = read_model_type(path, revision)
    if family and family != detected:
        raise ValueError(f"Config says family={family!r} but {path} has model_type={detected!r}")
    return get_family(detected)


def tiny_config(family: str, **overrides):
    return get_family(family).config_class(**{**TINY_CONFIGS[family], **overrides})


def _dtype(name: str):
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def build_model(model_cfg, *, causal_lm: bool = True, dtype: str | None = None):
    """Instantiate a vendored model from a ModelConfig.

    `path: tiny:<family>` builds a tiny random model (no download). `init: scratch`
    builds a randomly initialised model from the checkpoint's config (pre-training).
    """
    family = resolve_family(model_cfg.path, model_cfg.family, model_cfg.revision)
    cls = family.causal_lm_class if causal_lm else family.base_model_class
    torch_dtype = _dtype(dtype or model_cfg.dtype)
    attn = model_cfg.attn_implementation
    if model_cfg.path.startswith("tiny:"):
        config = tiny_config(family.name, **model_cfg.config_overrides)
        config._attn_implementation = attn
        return cls(config).to(torch_dtype)
    if model_cfg.init == "scratch":
        base = family.config_class.from_pretrained(model_cfg.path, revision=model_cfg.revision)
        values = {**base.to_dict(), **model_cfg.config_overrides}
        if "num_hidden_layers" in model_cfg.config_overrides and "layer_types" not in model_cfg.config_overrides:
            values.pop("layer_types", None)  # per-layer list is re-derived for the new depth
        for key in ("model_type", "transformers_version", "architectures", "_name_or_path"):
            values.pop(key, None)
        config = family.config_class(**values)
        config._attn_implementation = attn
        return cls(config).to(torch_dtype)
    if model_cfg.init != "pretrained":
        raise ValueError("model.init must be 'pretrained' or 'scratch'")
    return cls.from_pretrained(
        model_cfg.path, revision=model_cfg.revision, dtype=torch_dtype,
        attn_implementation=attn, use_safetensors=True, **model_cfg.config_overrides,
    )
