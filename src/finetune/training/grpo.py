"""GRPO: Group Relative Policy Optimization (DeepSeekMath, DeepSeek-R1) with verifiable rewards.

One optimizer step:
  1. Sample G completions for each of P prompts from the current policy.
  2. Score each completion with programmatic rewards (correctness, format, ...).
  3. Advantage A_i = (r_i - mean(r_group)) / (std(r_group) + eps): no value network.
  4. Token-level PPO-clipped objective with a KL penalty to the reference policy:
        L = -min(rho * A, clip(rho, 1 - eps, 1 + eps) * A) + beta * KL_k3(pi || ref)
     rho = pi / pi_old. With updates_per_batch=1, pi_old == pi (rho == 1, gradient = A * grad log pi).
     KL_k3 = exp(ref - logp) - (ref - logp) - 1 (unbiased, always >= 0; Schulman's k3 estimator).
  5. Per-sequence mean over completion tokens, then mean over sequences.

Written as an explicit loop (DDP via torchrun) so every step is visible. The
reference policy is the base model with LoRA adapters disabled, so LoRA is required
when beta > 0 (it avoids holding a second full model in memory).
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from finetune import utils
from finetune.data.io import load_records
from finetune.data.rl import build_rl_dataset
from finetune.data.tokenizer import load_tokenizer
from finetune.training.common import setup_model
from finetune.training.lora import lora_disabled, merge_for_save
from finetune.training.loop import all_reduce_sum, setup_distributed
from finetune.training.optim import build_optimizer, build_scheduler
from finetune.training.rewards import total_reward


def completion_mask(completion_ids: torch.Tensor, eos_ids: set[int]) -> torch.Tensor:
    """1 for tokens up to and including the first EOS, 0 afterwards."""
    is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for eos in eos_ids:
        is_eos |= completion_ids == eos
    after_eos = (is_eos.cumsum(dim=1) - is_eos.long()) > 0
    return (~after_eos).long()


def group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-4) -> torch.Tensor:
    grouped = rewards.view(-1, group_size)
    return ((grouped - grouped.mean(1, keepdim=True)) / (grouped.std(1, keepdim=True) + eps)).view(-1)


def token_logps(model, input_ids, attention_mask, prompt_length: int) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, prompt_length - 1:-1].float()
    targets = input_ids[:, prompt_length:]
    return torch.gather(logits.log_softmax(-1), 2, targets.unsqueeze(-1)).squeeze(-1)


def grpo_token_loss(logps, old_logps, ref_logps, advantages, mask, clip_eps: float, beta: float):
    ratio = torch.exp(logps - old_logps)
    unclipped = ratio * advantages.unsqueeze(1)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.unsqueeze(1)
    per_token = -torch.min(unclipped, clipped)
    kl = torch.zeros_like(logps)
    if ref_logps is not None:
        delta = ref_logps - logps
        kl = torch.exp(delta) - delta - 1
        per_token = per_token + beta * kl
    lengths = mask.sum(1).clamp(min=1)
    loss = ((per_token * mask).sum(1) / lengths).mean()
    stats = {
        "kl": ((kl * mask).sum(1) / lengths).mean().detach(),
        "clip_frac": ((((ratio - 1).abs() > clip_eps).float() * mask).sum() / mask.sum().clamp(min=1)).detach(),
    }
    return loss, stats


def _micro_batches(total: int, size: int):
    for start in range(0, total, size):
        yield slice(start, min(total, start + size))


def run(config) -> dict:
    cfg, train_cfg = config.grpo, config.train
    if train_cfg.max_steps <= 0:
        raise ValueError("GRPO needs train.max_steps")
    if cfg.beta > 0 and not config.lora.enabled:
        raise ValueError("GRPO with beta > 0 uses the adapter-disabled base as reference; enable LoRA or set grpo.beta=0")
    output = utils.prepare_output_dir(config.run.output_dir, config.run.overwrite)
    device = setup_distributed()
    utils.set_seed(config.run.seed + utils.rank())
    tokenizer = load_tokenizer(config.model, padding_side="left")
    template_kwargs = config.data.chat_template_kwargs
    dataset = build_rl_dataset(load_records(config.data.train, config.data.max_samples), tokenizer, cfg, template_kwargs)
    model = setup_model(config).to(device)
    policy = model
    distributed = dist.is_initialized()
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        policy = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, train_cfg.max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    eos_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and train_cfg.precision == "bf16" else contextlib.nullcontext())
    shard = dataset[utils.rank()::utils.world_size()]
    rng = random.Random(config.run.seed + utils.rank())
    log_file = (output / "log.jsonl").open("a") if utils.is_main() else None
    samples_file = (output / "completions.jsonl").open("a") if utils.is_main() else None
    micro = train_cfg.per_device_batch_size
    started = time.time()
    history = []

    for step in range(1, train_cfg.max_steps + 1):
        batch = rng.sample(shard, min(cfg.prompts_per_step, len(shard)))
        # 1. Rollout with the current policy.
        model.eval()
        encoded = tokenizer([b["prompt_text"] for b in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False, truncation=True, max_length=cfg.max_prompt_length).to(device)
        with torch.no_grad(), autocast:
            sequences = model.generate(
                **encoded, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens, num_return_sequences=cfg.group_size,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        prompt_length = encoded["input_ids"].shape[1]
        completions = sequences[:, prompt_length:]
        mask = completion_mask(completions, eos_ids)
        attention = torch.cat([encoded["attention_mask"].repeat_interleave(cfg.group_size, 0), mask], dim=1)
        texts = tokenizer.batch_decode(completions * mask + tokenizer.pad_token_id * (1 - mask), skip_special_tokens=True)
        answers = [b["answer"] for b in batch for _ in range(cfg.group_size)]

        # 2-3. Rewards and group-relative advantages.
        scored = [total_reward(t, a, cfg.rewards, cfg.reward_weights) for t, a in zip(texts, answers)]
        rewards = torch.tensor([s[0] for s in scored], device=device)
        advantages = group_advantages(rewards, cfg.group_size)

        # 4. Old-policy and reference log-probs (no grad).
        n = sequences.shape[0]
        with torch.no_grad(), autocast:
            old_logps = torch.cat([token_logps(model, sequences[s], attention[s], prompt_length)
                                   for s in _micro_batches(n, micro)])
            ref_logps = None
            if cfg.beta > 0:
                with lora_disabled(model):
                    ref_logps = torch.cat([token_logps(model, sequences[s], attention[s], prompt_length)
                                           for s in _micro_batches(n, micro)])
        model.train()

        stats_sum = {"loss": 0.0, "kl": 0.0, "clip_frac": 0.0}
        for _ in range(cfg.updates_per_batch):
            slices = list(_micro_batches(n, micro))
            for index, s in enumerate(slices):
                sync = index == len(slices) - 1
                context = policy.no_sync() if distributed and not sync else contextlib.nullcontext()
                with context, autocast:
                    logps = token_logps(policy, sequences[s], attention[s], prompt_length)
                    loss, stats = grpo_token_loss(
                        logps, old_logps[s], None if ref_logps is None else ref_logps[s],
                        advantages[s], mask[s].float(), cfg.clip_eps, cfg.beta,
                    )
                    (loss * (s.stop - s.start) / n).backward()
                weight = (s.stop - s.start) / n / cfg.updates_per_batch
                stats_sum["loss"] += loss.item() * weight
                stats_sum["kl"] += stats["kl"].item() * weight
                stats_sum["clip_frac"] += stats["clip_frac"].item() * weight
            grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # Logging (averaged over ranks).
        parts = {name: sum(s[1][name] for s in scored) / len(scored) for name in cfg.rewards}
        local = torch.tensor([rewards.mean().item(), mask.sum(1).float().mean().item(),
                              stats_sum["loss"], stats_sum["kl"], stats_sum["clip_frac"], *parts.values()],
                             device=device)
        local = all_reduce_sum(local) / utils.world_size()
        record = {"step": step, "reward": local[0].item(), "completion_tokens": local[1].item(),
                  "loss": local[2].item(), "kl": local[3].item(), "clip_frac": local[4].item(),
                  **{f"reward/{k}": local[5 + i].item() for i, k in enumerate(parts)},
                  "grad_norm": grad_norm.item(), "lr": scheduler.get_last_lr()[0],
                  "elapsed_sec": time.time() - started}
        history.append(record)
        if step % train_cfg.logging_steps == 0 or step == 1:
            utils.log(json.dumps(record))
        if log_file:
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
        if samples_file and (step % cfg.log_completions_every == 0 or step == 1):
            samples_file.write(json.dumps({"step": step, "prompt": batch[0]["prompt_text"], "answer": batch[0]["answer"],
                                           "completion": texts[0], "reward": rewards[0].item()}) + "\n")
            samples_file.flush()

    if utils.is_main():
        merge_for_save(model, output).save_pretrained(output)
        tokenizer.padding_side = "right"
        tokenizer.save_pretrained(output)
        log_file.close()
        samples_file.close()
        utils.write_metadata(output, config, model)
    if distributed:
        dist.barrier()
    last = history[-min(10, len(history)):]
    return {"steps": len(history), "final_reward_mean_last10": sum(r["reward"] for r in last) / len(last),
            "runtime_sec": time.time() - started, "nan_loss": any(math.isnan(r["loss"]) for r in history)}
