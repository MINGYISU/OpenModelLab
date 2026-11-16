"""Load trained models for inference (no AutoModel) and generate text."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from finetune.config import ModelConfig
from finetune.data.tokenizer import load_tokenizer
from finetune.models import build_model


def device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _adapter_base(path: str) -> str | None:
    config = Path(path) / "adapter_config.json"
    return json.loads(config.read_text())["base_model_name_or_path"] if config.exists() else None


def load_for_inference(path: str, revision: str | None = None, adapter: str | None = None,
                       dtype: str = "bfloat16", causal_lm: bool = True, merge: bool = False):
    """Load a hub/local checkpoint through the vendored classes.

    `path` may be a PEFT adapter directory (its base model is read from adapter_config.json),
    or pass the base as `path` and the adapter as `adapter`. `merge=True` folds LoRA into W.
    """
    if adapter is None and _adapter_base(path):
        adapter, path = path, _adapter_base(path)
    if not torch.cuda.is_available():
        dtype = "float32"
    model = build_model(ModelConfig(path=path, revision=revision, dtype=dtype), causal_lm=causal_lm)
    tokenizer = load_tokenizer(ModelConfig(path=adapter or path, revision=None if adapter else revision),
                               padding_side="left")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    return model.to(device()).eval(), tokenizer


def format_prompts(tokenizer, prompts, chat: bool = True, template_kwargs=None, system: str | None = None):
    if not chat:
        return list(prompts)
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; pass chat=False (--raw) for base models")
    head = [{"role": "system", "content": system}] if system else []
    return [tokenizer.apply_chat_template(head + [{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, **(template_kwargs or {})) for p in prompts]


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens=256, temperature=0.0, top_p=1.0, num_samples=1,
             chat=True, template_kwargs=None, batch_size=8, system=None) -> list:
    """Returns one string per prompt (num_samples=1) or a list of strings per prompt."""
    texts = format_prompts(tokenizer, prompts, chat, template_kwargs, system)
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
