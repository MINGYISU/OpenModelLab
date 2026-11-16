"""Tokenizer loading, plus a tiny offline byte-level tokenizer for tests and smoke runs."""

from __future__ import annotations

CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
TINY_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def tiny_tokenizer():
    """Byte-level tokenizer (256 byte tokens + ChatML specials, no merges); needs no download."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {char: index for index, char in enumerate(sorted(alphabet))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens({
        "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
        "additional_special_tokens": TINY_SPECIAL_TOKENS,
    })
    tokenizer.chat_template = CHATML_TEMPLATE
    return tokenizer


def load_tokenizer(model_cfg, padding_side: str = "right"):
    path = model_cfg.tokenizer or model_cfg.path
    if path.startswith("tiny:"):
        tokenizer = tiny_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path, revision=model_cfg.revision, trust_remote_code=False,
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer
