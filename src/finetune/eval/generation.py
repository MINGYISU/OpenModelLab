"""Load trained models for inference (no AutoModel) and generate text."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from finetune.config import ModelConfig
from finetune.data.tokenizer import load_tokenizer
from finetune.models import build_model
from finetune.variants import load_variant_state, read_spec_file


def device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _adapter_base(path: str) -> str | None:
    config = Path(path) / "adapter_config.json"
    return json.loads(config.read_text())["base_model_name_or_path"] if config.exists() else None


def load_for_inference(path: str, revision: str | None = None, adapter: str | None = None,
                       dtype: str = "bfloat16", causal_lm: bool = True, merge: bool = False,
                       backend: str = "transformers", tensor_parallel_size: int = 1):
    """Load a hub/local checkpoint through the vendored classes.

    `path` may be a PEFT adapter directory (its base model is read from adapter_config.json),
    or pass the base as `path` and the adapter as `adapter`. `merge=True` folds LoRA into W.
    The vLLM backend supports standard checkpoints, not PEFT adapters or project variants.
    """
    if backend == "vllm":
        if not causal_lm:
            raise ValueError("The vLLM inference backend supports text generation only")
        if adapter is not None or _adapter_base(path):
            raise ValueError("The vLLM inference backend does not currently support PEFT adapters")
        from finetune.eval.vllm_inference import load_model

        model = load_model(path, revision=revision, dtype=dtype,
                           tensor_parallel_size=tensor_parallel_size)
        tokenizer = load_tokenizer(ModelConfig(path=path, revision=revision), padding_side="left")
        return model, tokenizer
    if backend != "transformers":
        raise ValueError(f"Unknown inference backend {backend!r}; choose 'transformers' or 'vllm'")

    if adapter is None and _adapter_base(path):
        adapter, path = path, _adapter_base(path)
    if not torch.cuda.is_available():
        dtype = "float32"
    # A LoRA run on a variant saves the variant spec (and its new parameters) next to the adapter.
    spec = read_spec_file(adapter) if adapter else None
    model_cfg = ModelConfig(path=path, revision=revision, dtype=dtype)
    if spec:
        model_cfg.variant, model_cfg.variant_args = spec.get("source") or spec["name"], spec["args"]
    model = build_model(model_cfg, causal_lm=causal_lm)
    if spec:
        load_variant_state(model, adapter)
    tokenizer = load_tokenizer(ModelConfig(path=adapter or path, revision=None if adapter else revision),
                               padding_side="left")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    return model.to(device()).eval(), tokenizer


def _as_messages(prompt) -> list[dict]:
    return list(prompt) if isinstance(prompt, (list, tuple)) else [{"role": "user", "content": prompt}]


def format_prompts(tokenizer, prompts, chat: bool = True, template_kwargs=None, system: str | None = None):
    """Each prompt is a user string or a list of {role, content} messages (a conversation prefix)."""
    if not chat:
        return [p if isinstance(p, str) else "\n\n".join(m["content"] for m in p) for p in prompts]
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; pass chat=False (--raw) for base models")
    head = [{"role": "system", "content": system}] if system else []
    return [tokenizer.apply_chat_template(head + _as_messages(p), tokenize=False,
                                          add_generation_prompt=True, **(template_kwargs or {})) for p in prompts]


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=256, temperature=0.0, top_p=1.0, num_samples=1,
             chat=True, template_kwargs=None, batch_size=8, system=None, backend="transformers") -> list:
    """Returns one string per prompt (num_samples=1) or a list of strings per prompt."""
    texts = format_prompts(tokenizer, prompts, chat, template_kwargs, system)
    if backend == "vllm":
        from finetune.eval.vllm_inference import generate as vllm_generate

        results = []
        for start in range(0, len(texts), batch_size):
            results.extend(vllm_generate(model, texts[start:start + batch_size], max_new_tokens,
                                         temperature, top_p, num_samples))
        return results
    if backend != "transformers":
        raise ValueError(f"Unknown inference backend {backend!r}; choose 'transformers' or 'vllm'")

    tokenizer.padding_side = "left"
    results = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False)
        outputs = model.generate(**encoded, max_new_tokens=max_new_tokens, num_return_sequences=num_samples,
                                 pad_token_id=tokenizer.pad_token_id, **sampling)
        decoded = tokenizer.batch_decode(outputs[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for i in range(0, len(decoded), num_samples):
            group = decoded[i:i + num_samples]
            results.append(group[0] if num_samples == 1 else group)
    return results
