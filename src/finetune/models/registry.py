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
    "glm4": ("Glm4Config", "Glm4Model", "Glm4ForCausalLM"),
    "glm4_moe": ("Glm4MoeConfig", "Glm4MoeModel", "Glm4MoeForCausalLM"),
    # Multimodal: the third class is the generation head (image+text -> text, or image tokens for Janus).
    "qwen2_5_vl": ("Qwen2_5_VLConfig", "Qwen2_5_VLModel", "Qwen2_5_VLForConditionalGeneration"),
    "janus": ("JanusConfig", "JanusModel", "JanusForConditionalGeneration"),
    # Dual-encoder (contrastive image/text embeddings); has no generation head.
    "siglip2": ("Siglip2Config", "Siglip2Model", None),
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
    "glm4": dict(_COMMON_TINY, head_dim=16, pad_token_id=0, eos_token_id=1),
    "glm4_moe": dict(
        _COMMON_TINY, head_dim=16, pad_token_id=0, eos_token_id=1, moe_intermediate_size=32, n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, first_k_dense_replace=1, n_group=1, topk_group=1,
    ),
    "qwen2_5_vl": dict(
        text_config=dict(_COMMON_TINY, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                        "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=4, temporal_patch_size=2, spatial_merge_size=2, window_size=16,
                           fullatt_block_indexes=[1]),
        image_token_id=300, video_token_id=301, vision_start_token_id=302, vision_end_token_id=303,
        tie_word_embeddings=False,
    ),
    "janus": dict(
        text_config=dict(_COMMON_TINY),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           image_size=16, patch_size=4, projection_dim=64, num_image_tokens=16),
        vq_config=dict(embed_dim=8, num_embeddings=32, base_channels=32, channel_multiplier=[1, 1],
                       num_res_blocks=1, latent_channels=8, projection_dim=64, image_token_embed_dim=64,
                       num_patches=4),
        image_token_id=300,
    ),
    "siglip2": dict(
        text_config=dict(vocab_size=320, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, max_position_embeddings=64),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                           patch_size=4, num_patches=16),
    ),
}


@dataclass(frozen=True)
class Family:
    name: str
    config_class: type
    base_model_class: type
    causal_lm_class: type | None  # None for encoder-only families (siglip2)


def get_family(name: str) -> Family:
    if name not in FAMILY_CLASSES:
        raise ValueError(f"No vendored model for family {name!r}; available: {sorted(FAMILY_CLASSES)}")
    module = importlib.import_module(f"finetune.models.{name}")
    return Family(name, *(getattr(module, cls) if cls else None for cls in FAMILY_CLASSES[name]))


def read_config_json(path: str, revision: str | None = None) -> dict:
    local = Path(path) / "config.json"
    if local.exists():
        return json.loads(local.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(path, "config.json", revision=revision)).read_text())


def read_model_type(path: str, revision: str | None = None) -> str:
    return read_config_json(path, revision)["model_type"]


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
    `variant` applies a model variant (src/finetune/variants/) to the built reference model;
    checkpoints saved from a variant are recognised and rebuilt automatically.
    """
    from finetune.variants import SPEC_KEY, apply_variant, get_variant

    family = resolve_family(model_cfg.path, model_cfg.family, model_cfg.revision)
    cls = family.causal_lm_class if causal_lm else family.base_model_class
    if cls is None:
        raise ValueError(f"Family {family.name!r} has no generation head; build it with causal_lm=False")
    torch_dtype = _dtype(dtype or model_cfg.dtype)
    attn = model_cfg.attn_implementation
    tiny = model_cfg.path.startswith("tiny:")
    saved = None if tiny else read_config_json(model_cfg.path, model_cfg.revision).get(SPEC_KEY)
    variant, variant_args = model_cfg.variant, model_cfg.variant_args
    if saved and model_cfg.init == "pretrained":
        if variant and variant not in (saved["name"], saved.get("source")):
            raise ValueError(f"{model_cfg.path} is a {saved['name']!r} checkpoint; stacking variant {variant!r} "
                             "on top of it is not supported")
        if variant_args and variant_args != saved["args"]:
            raise ValueError(f"{model_cfg.path} was saved with variant_args={saved['args']}; got {variant_args}")
        return _load_variant_checkpoint(model_cfg, family, cls, saved, torch_dtype)
    if saved:  # init: scratch from a variant checkpoint's config: start from its reference config
        variant = variant or saved.get("source") or saved["name"]
        variant_args = variant_args or saved["args"]
    overrides = {**(getattr(get_variant(variant), "CONFIG", {}) if variant else {}), **model_cfg.config_overrides}

    if tiny:
        config = tiny_config(family.name, **overrides)
        config._attn_implementation = attn
        model = cls(config)
    elif model_cfg.init == "scratch":
        base = family.config_class.from_pretrained(model_cfg.path, revision=model_cfg.revision)
        values = {**base.to_dict(), **(saved or {}).get("reference_config", {}), **overrides}
        if "num_hidden_layers" in overrides and "layer_types" not in overrides:
            values.pop("layer_types", None)  # per-layer list is re-derived for the new depth
        for key in ("model_type", "transformers_version", "architectures", "_name_or_path", SPEC_KEY):
            values.pop(key, None)
        config = family.config_class(**values)
        config._attn_implementation = attn
        model = cls(config)
    elif model_cfg.init == "pretrained":
        model = cls.from_pretrained(
            model_cfg.path, revision=model_cfg.revision, dtype=torch_dtype,
            attn_implementation=attn, use_safetensors=True, **overrides,
        )
    else:
        raise ValueError("model.init must be 'pretrained' or 'scratch'")
    if variant:
        apply_variant(model, family.name, variant, variant_args)
    # from_pretrained already loaded in `torch_dtype` (and keeps any fp32-pinned modules).
    return model if not tiny and model_cfg.init == "pretrained" else model.to(torch_dtype)


def _checkpoint_state_dict(path: str, revision: str | None) -> dict:
    from safetensors.torch import load_file

    root = Path(path)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(path, revision=revision, allow_patterns=["*.json", "*.safetensors"]))
    index = root / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    state = {}
    for name in files:
        state.update(load_file(str(root / name)))
    return state


def _load_variant_checkpoint(model_cfg, family: Family, cls, spec: dict, torch_dtype):
    """Rebuild the reference architecture, re-apply the variant, then load the trained weights."""
    from finetune.variants import SPEC_KEY, apply_variant

    config = family.config_class.from_pretrained(model_cfg.path, revision=model_cfg.revision)
    for key, value in {**spec.get("reference_config", {}), **model_cfg.config_overrides}.items():
        setattr(config, key, value)
    if hasattr(config, SPEC_KEY):
        delattr(config, SPEC_KEY)
    config._attn_implementation = model_cfg.attn_implementation
    model = cls(config)
    apply_variant(model, family.name, spec.get("source") or spec["name"], spec["args"], spec.get("sha256"))

    state = _checkpoint_state_dict(model_cfg.path, model_cfg.revision)
    prefix = getattr(cls, "base_model_prefix", "model") + "."
    is_base = not hasattr(model, "lm_head")
    if is_base and any(k.startswith(prefix) for k in state):  # base model from a *ForCausalLM checkpoint
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    elif not is_base and not any(k.startswith(prefix) for k in state):  # *ForCausalLM from a base checkpoint
        state = {prefix + k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if getattr(config, "tie_word_embeddings", False):
        missing = [k for k in missing if k != "lm_head.weight"]
        model.tie_weights()
    if missing or unexpected:
        raise ValueError(f"Variant checkpoint {model_cfg.path} does not match the rebuilt model: "
                         f"missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.to(torch_dtype)
