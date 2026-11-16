"""A minimal, explicit PyTorch training loop for causal LMs (the `loop` backend).

This shows what transformers.Trainer does for you:
  * torch.distributed init + DistributedDataParallel (launch with torchrun)
  * DistributedSampler so each rank sees a disjoint shard, reseeded every epoch
  * gradient accumulation with DDP.no_sync() on all but the last micro-batch
  * token-weighted loss across the accumulation window (each token counts once)
  * bf16 autocast, gradient clipping, optimizer + LR scheduler steps
  * rank-0 logging/checkpointing, resumable state (model, optimizer, scheduler, step, RNG)

Global batch (sequences/step) = per_device_batch_size * grad_accum * world_size.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from finetune import utils
from finetune.data.sft import IGNORE_INDEX
from finetune.training.lora import merge_for_save
from finetune.training.optim import build_optimizer, build_scheduler


def setup_distributed() -> torch.device:
    if utils.world_size() > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(utils.local_rank())
        return torch.device("cuda", utils.local_rank())
    return torch.device("cpu")


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def token_loss_sum(model, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of next-token cross-entropy over supervised tokens, and the token count."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    labels = batch["labels"][:, 1:]
    loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)), labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    )
    return loss, (labels != IGNORE_INDEX).sum()


def _autocast(device, precision):
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, precision) -> float:
    model.eval()
    total = torch.zeros(2, device=device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with _autocast(device, precision):
            loss, count = token_loss_sum(model, batch)
        total += torch.stack([loss.float(), count.float()])
    model.train()
    total = all_reduce_sum(total)
    return (total[0] / total[1].clamp(min=1)).item()


def _save_checkpoint(path: Path, model, optimizer, scheduler, step: int, epoch: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({
        "model": trainable, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "epoch": epoch, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path / "training_state.pt")


def _load_checkpoint(path: Path, model, optimizer, scheduler) -> tuple[int, int]:
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    missing = set(state["model"]) - {n for n, _ in model.named_parameters()}
    if missing:
        raise ValueError(f"Checkpoint has parameters not in model: {sorted(missing)[:5]}")
    model.load_state_dict(state["model"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], state["epoch"]


def train_loop(config, model, train_dataset, eval_dataset, collator, tokenizer=None) -> dict:
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

    train_cfg, output = config.train, Path(config.run.output_dir)
    device = setup_distributed()
    if train_cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    utils.set_seed(config.run.seed)
    model.to(device).train()
    raw_model = model
    distributed = dist.is_initialized()
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    iterable = isinstance(train_dataset, IterableDataset)
    # A seeded, epoch-indexed sampler even on one process makes data order exactly reproducible on resume.
    sampler = None if iterable else DistributedSampler(
        train_dataset, num_replicas=utils.world_size(), rank=utils.rank(), shuffle=True, seed=config.run.seed)
    loader = DataLoader(
        train_dataset, batch_size=train_cfg.per_device_batch_size, sampler=sampler,
        shuffle=False, collate_fn=collator,
        num_workers=train_cfg.dataloader_num_workers, drop_last=True,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if distributed else None
        eval_loader = DataLoader(eval_dataset, batch_size=train_cfg.eval_batch_size or train_cfg.per_device_batch_size,
                                 sampler=eval_sampler, collate_fn=collator)

    if train_cfg.max_steps > 0:
        total_steps = train_cfg.max_steps
    elif iterable:
        raise ValueError("Streaming datasets need train.max_steps")
    else:
        total_steps = math.ceil(train_cfg.epochs * (len(loader) // train_cfg.grad_accum))
    if total_steps <= 0:
        raise ValueError("Not enough data for one optimizer step; lower batch size or grad_accum")
    optimizer = build_optimizer(raw_model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, total_steps)
    step, epoch = 0, 0
    if train_cfg.resume_from:
        step, epoch = _load_checkpoint(Path(train_cfg.resume_from), raw_model, optimizer, scheduler)
        utils.log(f"resumed from {train_cfg.resume_from} at step {step}")

    writer = None
    if utils.is_main() and config.run.report_to == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output / "runs"))
    log_file = (output / "log.jsonl").open("a", encoding="utf-8") if utils.is_main() else None
    save_every = train_cfg.save_steps or max(1, total_steps // max(1, math.ceil(train_cfg.epochs)))
    params = [p for p in raw_model.parameters() if p.requires_grad]
    started, tokens_seen = time.time(), 0

    def batches():
        nonlocal epoch
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from loader
            epoch += 1

    stream = batches()
    if step and not iterable:
        # Skip micro-batches already consumed in the resumed epoch.
        consumed = (step * train_cfg.grad_accum) % max(1, len(loader))
        stream = islice(stream, consumed, None)

    while step < total_steps:
        window = [{k: v.to(device) for k, v in b.items()} for b in islice(stream, train_cfg.grad_accum)]
        if len(window) < train_cfg.grad_accum:
            break
        window_tokens = all_reduce_sum(sum((b["labels"][:, 1:] != IGNORE_INDEX).sum() for b in window).float())
        # Scale so DDP's gradient *average* equals d(sum loss over all ranks' tokens)/d(theta) / total tokens.
        scale = utils.world_size() / window_tokens.clamp(min=1)
        loss_sum = torch.zeros((), device=device)
        for index, batch in enumerate(window):
            sync = index == len(window) - 1
            context = model.no_sync() if distributed and not sync else contextlib.nullcontext()
            with context, _autocast(device, train_cfg.precision):
                loss, _ = token_loss_sum(model, batch)
                (loss * scale).backward()
            loss_sum += loss.detach()
            tokens_seen += int(batch["attention_mask"].sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, train_cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % train_cfg.logging_steps == 0 or step == 1 or step == total_steps:
            mean_loss = (all_reduce_sum(loss_sum) / window_tokens).item()
            record = {
                "step": step, "epoch": epoch, "loss": mean_loss, "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": tokens_seen * utils.world_size() / (time.time() - started),
            }
            if torch.cuda.is_available():
                record["max_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if eval_loader is not None and train_cfg.eval_steps and step % train_cfg.eval_steps == 0:
                record["eval_loss"] = evaluate(model, eval_loader, device, train_cfg.precision)
            utils.log(json.dumps(record))
            if log_file:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            if writer:
                for key, value in record.items():
                    writer.add_scalar(f"train/{key}", value, step)
        if step % save_every == 0 and step < total_steps and utils.is_main():
            _save_checkpoint(output / f"checkpoint-{step}", raw_model, optimizer, scheduler, step, epoch)

    metrics = {"steps": step, "train_runtime": time.time() - started}
    if eval_loader is not None:
        metrics["eval_loss"] = evaluate(model, eval_loader, device, train_cfg.precision)
    if utils.is_main():
        _save_checkpoint(output / f"checkpoint-{step}", raw_model, optimizer, scheduler, step, epoch)
        merge_for_save(raw_model, output).save_pretrained(output)
        if tokenizer is not None:
            tokenizer.save_pretrained(output)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log_file.close()
        if writer:
            writer.close()
    if distributed:
        dist.barrier()
    return metrics
