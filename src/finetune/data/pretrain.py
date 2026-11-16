"""Pre-training data: tokenize raw documents and pack them into fixed-length blocks.

Packing concatenates documents separated by EOS and slices the stream into
`max_length` blocks, so no compute is wasted on padding. Tokens attend across
document boundaries within a block (standard GPT-style packing).
"""

from __future__ import annotations

from typing import Iterable, Iterator


def pack_token_stream(token_lists: Iterable[list[int]], block_size: int, eos_id: int) -> Iterator[list[int]]:
    buffer: list[int] = []
    for tokens in token_lists:
        buffer.extend(tokens)
        buffer.append(eos_id)
        while len(buffer) >= block_size:
            yield buffer[:block_size]
            buffer = buffer[block_size:]
    # The trailing partial block is dropped so every block has exactly block_size tokens.


def _tokenize_texts(records, tokenizer, text_field) -> Iterator[list[int]]:
    for record in records:
        text = record.get(text_field)
        if isinstance(text, str) and text.strip():
            yield tokenizer(text, add_special_tokens=False)["input_ids"]


def _example(block: list[int]) -> dict:
    return {"input_ids": block, "attention_mask": [1] * len(block), "labels": list(block)}


def build_pretrain_dataset(records, tokenizer, data_cfg) -> list[dict]:
    """Map-style dataset for corpora that fit in memory."""
    eos = tokenizer.eos_token_id
    tokens = _tokenize_texts(records, tokenizer, data_cfg.text_field)
    if data_cfg.packing:
        return [_example(block) for block in pack_token_stream(tokens, data_cfg.max_length, eos)]
    return [_example((ids + [eos])[: data_cfg.max_length]) for ids in tokens]


def streaming_pretrain_dataset(spec, tokenizer, data_cfg, shard_rank=0, shard_world=1, seed=0):
    """IterableDataset over a (possibly huge, streamed) corpus, packed on the fly.

    Records are sharded round-robin by (rank, dataloader worker) when used with the
    raw loop. With transformers.Trainer keep shard_world=1: accelerate already
    dispatches iterable batches across ranks.
    """
    import torch

    from finetune.data.io import iter_records, mix_records

    class PackedStream(torch.utils.data.IterableDataset):
        def __iter__(self):
            info = torch.utils.data.get_worker_info()
            workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
            shards, shard = shard_world * workers, shard_rank * workers + worker_id
            stream = iter_records(spec)
            if data_cfg.replay and data_cfg.replay_ratio > 0:
                # Same seed on every shard: all shards see one mixed stream, then take disjoint slices.
                stream = mix_records(stream, iter_records(data_cfg.replay), data_cfg.replay_ratio, seed)
            records = (r for i, r in enumerate(stream) if i % shards == shard)
            tokens = _tokenize_texts(records, tokenizer, data_cfg.text_field)
            for block in pack_token_stream(tokens, data_cfg.max_length, tokenizer.eos_token_id):
                yield _example(block)

    return PackedStream()
