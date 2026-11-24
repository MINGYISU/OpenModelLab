"""Typed experiment configuration loaded from YAML.

An experiment YAML may set `base: other.yaml` or `base: [a.yaml, b.yaml]` (paths
relative to the including file) to inherit and override configs; later bases win.
Command-line overrides use dotted keys, for example
`--set train.lr=1e-4 --set lora.enabled=false`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    name: str = "experiment"
    output_dir: str = "outputs/experiment"
    seed: int = 42
    report_to: str = "tensorboard"  # "none", "tensorboard", "wandb"
    overwrite: bool = False


@dataclass
class ModelConfig:
    path: str = ""                     # hub id or local dir with config.json (+ weights)
    family: str | None = None          # vendored family; inferred from config.json model_type
    revision: str | None = None        # pin a hub commit for reproducibility
    tokenizer: str | None = None       # defaults to `path`
    init: str = "pretrained"           # "pretrained" or "scratch" (random init from config)
    config_overrides: dict = field(default_factory=dict)
    variant: str | None = None         # your architecture: name in src/finetune/variants/ or a .py path
    variant_args: dict = field(default_factory=dict)  # overrides the variant's ARGS (ablations)
    dtype: str = "bfloat16"            # "bfloat16", "float16", "float32"
    attn_implementation: str = "sdpa"  # "eager", "sdpa", or "flash_attention_2" (requires flash-attn)
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    train: Any = None                  # JSONL path or {"hf": name, "config":..., "split":..., "streaming":...}
    eval: Any = None
    format: str | None = None          # inferred from stage: text, chat, preference, rl, contrastive
    max_length: int = 2048
    loss_on: str = "all"               # SFT: "all" non-padding tokens or "assistant" tokens only
    packing: bool = False              # pretraining: pack documents into fixed-length blocks
    replay: Any = None                 # pretraining: general-domain data mixed in to reduce forgetting
    replay_ratio: float = 0.0          # fraction of documents drawn from `replay`
    text_field: str = "text"
    max_samples: int | None = None
    max_eval_samples: int | None = None
    chat_template_kwargs: dict = field(default_factory=dict)  # e.g. {"enable_thinking": false}


@dataclass
class LoraConfig:
    enabled: bool = False
    impl: str = "peft"                 # "peft" or "scratch" (our LoRALinear, for learning)
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Any = "all-linear"
    use_dora: bool = False
    use_rslora: bool = False


@dataclass
class TrainConfig:
    stage: str = "sft"                 # pretrain, sft, dpo, grpo, embed
    backend: str = "trainer"           # "trainer" (transformers.Trainer) or "loop" (raw PyTorch)
    epochs: float = 1.0
    max_steps: int = -1
    per_device_batch_size: int = 1
    eval_batch_size: int | None = None
    grad_accum: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    scheduler: str = "cosine"          # constant, linear, cosine, wsd
    warmup_ratio: float = 0.03
    warmup_steps: int = 0              # overrides warmup_ratio when > 0
    min_lr_ratio: float = 0.0
    decay_ratio: float = 0.1           # WSD: fraction of steps spent decaying
    max_grad_norm: float = 1.0
    precision: str = "bf16"            # bf16, fp16, fp32
    tf32: bool = True
    logging_steps: int = 10
    eval_steps: int | None = None      # None -> evaluate each epoch (if eval data)
    save_steps: int | None = None      # None -> save each epoch
    save_total_limit: int | None = 2
    dataloader_num_workers: int = 0
    fsdp: bool = False                 # FSDP (v2 by default) via Trainer; see configs/dist/fsdp2.yaml
    fsdp_config: dict = field(default_factory=dict)
    deepspeed: str | None = None       # path to a DeepSpeed JSON config
    resume_from: str | None = None


@dataclass
class DPOConfig:
    beta: float = 0.1
    loss_type: str = "sigmoid"         # sigmoid (DPO), ipo, simpo
    label_smoothing: float = 0.0       # conservative DPO
    simpo_gamma: float = 0.5
    max_prompt_length: int = 1024


@dataclass
class GRPOConfig:
    group_size: int = 8
    prompts_per_step: int = 8          # per device
    max_prompt_length: int = 512
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.04                 # KL(policy || reference) coefficient
    clip_eps: float = 0.2
    updates_per_batch: int = 1         # >1 makes the PPO ratio/clip meaningful
    rewards: list = field(default_factory=lambda: ["correctness", "format"])
    reward_weights: list | None = None
    system_prompt: str | None = None
    use_chat_template: bool = True
    log_completions_every: int = 10


@dataclass
class EmbedConfig:
    pooling: str = "mean"              # mean, last, cls
    normalize: bool = True
    temperature: float = 0.02
    query_prefix: str = ""
    passage_prefix: str = ""
    max_query_length: int = 128
    max_passage_length: int = 256
    gradcache: bool = False
    chunk_size: int = 8
    gather_across_ranks: bool = True


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _merge(base: dict, update: dict) -> dict:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = raw.pop("base", None) or []
    merged: dict = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = _merge(merged, _read_yaml((path.parent / base).resolve()))
    return _merge(merged, raw)


def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        node = raw
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_value(value)
    return raw


def _coerce(cls, values: dict) -> dict:
    """YAML 1.1 reads `2e-4` (no dot) as a string; coerce numeric strings for numeric fields."""
    types = {f.name: str(f.type) for f in dataclasses.fields(cls)}
    coerced = dict(values)
    for key, value in values.items():
        if isinstance(value, str) and key in types:
            try:
                if "float" in types[key]:
                    coerced[key] = float(value)
                elif types[key].startswith("int"):
                    coerced[key] = int(float(value))
            except ValueError:
                pass
    return coerced


def from_dict(raw: dict) -> ExperimentConfig:
    sections = {}
    for section in dataclasses.fields(ExperimentConfig):
        values = raw.get(section.name) or {}
        cls = section.default_factory().__class__
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown keys in [{section.name}]: {sorted(unknown)}")
        sections[section.name] = cls(**_coerce(cls, values))
    unknown_sections = set(raw) - set(sections)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")
    return ExperimentConfig(**sections)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = _read_yaml(Path(path).resolve())
    return from_dict(apply_overrides(raw, overrides or []))


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
