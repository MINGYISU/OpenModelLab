"""Contrastive embedding training (InfoNCE) with optional GradCache and cross-rank negatives.

Loss for query q_i with its positive p_i, hard negatives, and every other passage
in the (global) batch as negatives:
    L = -log( exp(s(q_i, p_i)/T) / sum_j exp(s(q_i, p_j)/T) ),   s = cosine similarity

Larger batches give more negatives and usually better embeddings, but activation
memory grows with batch size. GradCache (Gao et al., 2021) decouples the two:
  1. encode all inputs in chunks WITHOUT grad -> representations (cheap to store)
  2. compute the full-batch loss on detached reps and backprop to get d loss / d rep
  3. re-encode each chunk WITH grad and backprop  sum(rep * cached_grad)
The resulting parameter gradient equals the full-batch gradient (verified in tests),
with peak activation memory of one chunk. RNG state is replayed so dropout matches.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from finetune import utils
from finetune.data.collators import ContrastiveCollator
from finetune.data.contrastive import build_contrastive_dataset
from finetune.data.io import check_no_overlap, load_records
from finetune.data.tokenizer import load_tokenizer
from finetune.training.common import ResearchTrainer, finish, setup_model, training_arguments


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    if method == "last":
        # Right padding: the last real token (EOS appended by the collator) sits at length - 1.
        last = attention_mask.sum(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    if method == "cls":
        return hidden[:, 0]
    raise ValueError(f"Unknown pooling {method!r}; use mean, last, or cls")


def encode(model, input_ids, attention_mask, pooling: str, normalize: bool = True) -> torch.Tensor:
    hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    reps = pool(hidden, attention_mask, pooling).float()
    return F.normalize(reps, dim=-1) if normalize else reps


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather across ranks while keeping gradients for the local shard."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor), dim=0)


def contrastive_loss(query_reps, passage_reps, temperature: float, gather: bool = True) -> torch.Tensor:
    """InfoNCE; passages are laid out as [positive, negatives...] per query."""
    group = passage_reps.shape[0] // query_reps.shape[0]
    offset = 0
    if gather and dist.is_available() and dist.is_initialized():
        offset = dist.get_rank() * passage_reps.shape[0]
        passage_reps = gather_with_grad(passage_reps)
    scores = query_reps @ passage_reps.T / temperature
    targets = offset + torch.arange(query_reps.shape[0], device=scores.device) * group
    return F.cross_entropy(scores, targets)


class RandContext:
    """Capture CPU/CUDA RNG state so a re-forward reproduces dropout masks exactly."""

    def __init__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    @contextlib.contextmanager
    def replay(self):
        devices = list(range(torch.cuda.device_count())) if self.cuda is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state_all(self.cuda)
            yield


def gradcache_step(model, inputs, embed_cfg, *, no_sync=None, backward=None, encoder=None) -> torch.Tensor:
    """Run forward+backward for one contrastive batch with GradCache; returns the detached loss.

    `model` is the (possibly DDP-wrapped) module used for gradient re-forwards; `encoder` is
    an unwrapped module for the no-grad pass. `no_sync(model)` suppresses DDP all-reduce for
    all but the last chunk. `backward(t)` defaults to t.backward().
    """
    encoder = encoder or model
    backward = backward or (lambda tensor: tensor.backward())
    chunk = embed_cfg.chunk_size
    groups = [("query", inputs["query_input_ids"], inputs["query_attention_mask"]),
              ("passage", inputs["passage_input_ids"], inputs["passage_attention_mask"])]
    reps, states = {}, {}
    with torch.no_grad():
        for name, ids, mask in groups:
            states[name], parts = [], []
            for start in range(0, ids.shape[0], chunk):
                states[name].append(RandContext())
                parts.append(encode(encoder, ids[start:start + chunk], mask[start:start + chunk],
                                    embed_cfg.pooling, embed_cfg.normalize))
            reps[name] = torch.cat(parts).requires_grad_(True)
    loss = contrastive_loss(reps["query"], reps["passage"], embed_cfg.temperature, embed_cfg.gather_across_ranks)
    loss.backward()  # gradients w.r.t. the cached representations only
    jobs = [(name, ids, mask, start, state)
            for name, ids, mask in groups
            for start, state in zip(range(0, ids.shape[0], chunk), states[name])]
    for index, (name, ids, mask, start, state) in enumerate(jobs):
        last = index == len(jobs) - 1
        context = no_sync(model) if (no_sync is not None and not last) else contextlib.nullcontext()
        with context, state.replay():
            chunk_reps = encode(model, ids[start:start + chunk], mask[start:start + chunk],
                                embed_cfg.pooling, embed_cfg.normalize)
            backward((chunk_reps * reps[name].grad[start:start + chunk]).sum())
    return loss.detach()


class EmbeddingTrainer(ResearchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg = self.research_config.embed
        queries = encode(model, inputs["query_input_ids"], inputs["query_attention_mask"], cfg.pooling, cfg.normalize)
        passages = encode(model, inputs["passage_input_ids"], inputs["passage_attention_mask"], cfg.pooling, cfg.normalize)
        loss = contrastive_loss(queries, passages, cfg.temperature, cfg.gather_across_ranks)
        return (loss, {}) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.research_config.embed.gradcache:
            return super().training_step(model, inputs, num_items_in_batch)
        model.train()
        inputs = self._prepare_inputs(inputs)
        scale = 1.0 / self.current_gradient_accumulation_steps
        with self.compute_loss_context_manager():
            loss = gradcache_step(
                model, inputs, self.research_config.embed,
                no_sync=self.accelerator.no_sync if self.accelerator.num_processes > 1 else None,
                backward=lambda tensor: self.accelerator.backward(tensor * scale),
                encoder=self.accelerator.unwrap_model(model),
            )
        return loss * scale


def run(config) -> dict:
    utils.prepare_output_dir(config.run.output_dir, config.run.overwrite, bool(config.train.resume_from))
    utils.set_seed(config.run.seed)
    if config.train.deepspeed and config.embed.gradcache:
        raise ValueError("GradCache's chunked backward is supported with DDP/FSDP, not DeepSpeed")
    tokenizer = load_tokenizer(config.model)
    data = config.data
    train_records = load_records(data.train, data.max_samples)
    eval_records = load_records(data.eval, data.max_eval_samples) if data.eval else []
    check_no_overlap(train_records, eval_records)
    train_dataset = build_contrastive_dataset(train_records, config.embed)
    num_negatives = len(train_dataset[0]["passages"]) - 1
    eval_dataset = build_contrastive_dataset(eval_records, config.embed, num_negatives) if eval_records else None
    model = setup_model(config, causal_lm=False, task_type="FEATURE_EXTRACTION")
    trainer = EmbeddingTrainer(
        model=model,
        args=training_arguments(config, eval_dataset is not None, prediction_loss_only=True,
                                label_names=["query_input_ids"]),
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=ContrastiveCollator(tokenizer, config.embed.max_query_length, config.embed.max_passage_length),
        processing_class=tokenizer, research_config=config,
    )
    result = trainer.train(resume_from_checkpoint=config.train.resume_from)
    finish(trainer, config, model, tokenizer, result, eval_dataset)
    if utils.is_main():
        embed = config.embed
        Path(config.run.output_dir, "embedding_config.json").write_text(json.dumps({
            "pooling": embed.pooling, "normalize": embed.normalize, "query_prefix": embed.query_prefix,
            "passage_prefix": embed.passage_prefix, "max_query_length": embed.max_query_length,
            "max_passage_length": embed.max_passage_length,
        }, indent=2))
    return result.metrics
