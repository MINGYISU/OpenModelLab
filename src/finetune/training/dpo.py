"""Direct Preference Optimization (Rafailov et al., 2023) and variants.

For a prompt x with chosen y_w and rejected y_l, with log-ratios against a frozen
reference policy:
    h = [log pi(y_w|x) - log ref(y_w|x)] - [log pi(y_l|x) - log ref(y_l|x)]
  sigmoid (DPO): loss = -log sigmoid(beta * h)            (label_smoothing -> cDPO)
  ipo:           loss = (h - 1 / (2 * beta))^2
  simpo:         reference-free, length-normalised:  -log sigmoid(beta * (avg_w - avg_l) - gamma)

Reference log-probs are precomputed once with the *initial* policy (LoRA starts
as the identity; full fine-tuning starts at the reference), so no second model is
kept in memory and every distributed strategy works unchanged.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from finetune import utils
from finetune.data.collators import PreferenceCollator
from finetune.data.io import check_no_overlap, load_records
from finetune.data.preference import build_preference_dataset
from finetune.data.sft import IGNORE_INDEX
from finetune.data.tokenizer import load_tokenizer
from finetune.training.common import ResearchTrainer, finish, setup_model, training_arguments


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence sum of token log-probs over supervised (response) tokens, and token counts."""
    labels = labels[:, 1:]
    logits = logits[:, :-1].float()
    mask = labels != IGNORE_INDEX
    token_logps = torch.gather(logits.log_softmax(-1), 2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1), mask.sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, *, beta, loss_type="sigmoid",
             label_smoothing=0.0, simpo_gamma=0.5, chosen_lengths=None, rejected_lengths=None):
    if loss_type == "simpo":
        logits = beta * (policy_chosen / chosen_lengths - policy_rejected / rejected_lengths) - simpo_gamma
        return -F.logsigmoid(logits), logits
    h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    if loss_type == "sigmoid":
        logits = beta * h
        return -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing, logits
    if loss_type == "ipo":
        return (h - 1 / (2 * beta)) ** 2, h
    raise ValueError(f"Unknown dpo.loss_type {loss_type!r}; use sigmoid, ipo, or simpo")


@torch.no_grad()
def precompute_reference(model, dataset, collator, batch_size: int) -> None:
    device = torch.device("cuda", utils.local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    original_device = next(model.parameters()).device
    model.to(device).eval()
    for start in range(0, len(dataset), batch_size):
        chunk = dataset[start:start + batch_size]
        batch = {k: v.to(device) for k, v in collator(chunk).items()}
        logps, _ = sequence_logps(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
                                  batch["labels"])
        n = len(chunk)
        for example, chosen, rejected in zip(chunk, logps[:n].tolist(), logps[n:].tolist()):
            example["ref_chosen_logp"], example["ref_rejected_logp"] = chosen, rejected
    model.to(original_device).train()


class DPOTrainer(ResearchTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # our loss is already a per-pair mean
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg = self.research_config.dpo
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        logps, lengths = sequence_logps(logits, inputs["labels"])
        n = logps.shape[0] // 2
        ref = inputs["ref_logps"]
        losses, margins = dpo_loss(
            logps[:n], logps[n:], ref[:n], ref[n:], beta=cfg.beta, loss_type=cfg.loss_type,
            label_smoothing=cfg.label_smoothing, simpo_gamma=cfg.simpo_gamma,
            chosen_lengths=lengths[:n], rejected_lengths=lengths[n:],
        )
        split = "train" if model.training else "eval"
        chosen_reward = cfg.beta * (logps[:n] - ref[:n]).detach()
        rejected_reward = cfg.beta * (logps[n:] - ref[n:]).detach()
        for key, value in {
            "rewards/chosen": chosen_reward.mean(), "rewards/rejected": rejected_reward.mean(),
            "rewards/accuracy": (chosen_reward > rejected_reward).float().mean(),
            "rewards/margin": (chosen_reward - rejected_reward).mean(),
            "logps/chosen": logps[:n].detach().mean(), "logps/rejected": logps[n:].detach().mean(),
        }.items():
            self._metrics[split][key].append(value.item())
        loss = losses.mean()
        return (loss, {"margins": margins.detach()}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        split = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        prefix = "eval_" if split == "eval" else ""
        for key, values in self._metrics[split].items():
            if values:
                logs[prefix + key] = sum(values) / len(values)
        self._metrics[split].clear()
        super().log(logs, *args, **kwargs)


def run(config) -> dict:
    utils.prepare_output_dir(config.run.output_dir, config.run.overwrite, bool(config.train.resume_from))
    utils.set_seed(config.run.seed)
    tokenizer = load_tokenizer(config.model)
    data = config.data
    train_records = load_records(data.train, data.max_samples)
    eval_records = load_records(data.eval, data.max_eval_samples) if data.eval else []
    check_no_overlap(train_records, eval_records)
    train_dataset = build_preference_dataset(train_records, tokenizer, data, config.dpo)
    eval_dataset = build_preference_dataset(eval_records, tokenizer, data, config.dpo) if eval_records else None
    model = setup_model(config)
    collator = PreferenceCollator(tokenizer.pad_token_id)
    if config.dpo.loss_type != "simpo":
        batch = config.train.eval_batch_size or config.train.per_device_batch_size
        for dataset in filter(None, (train_dataset, eval_dataset)):
            precompute_reference(model, dataset, collator, batch)
    else:
        for dataset in filter(None, (train_dataset, eval_dataset)):
            for example in dataset:
                example["ref_chosen_logp"] = example["ref_rejected_logp"] = 0.0
    trainer = DPOTrainer(
        model=model, args=training_arguments(config, eval_dataset is not None, label_names=["labels"]),
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator,
        processing_class=tokenizer, research_config=config,
    )
    result = trainer.train(resume_from_checkpoint=config.train.resume_from)
    finish(trainer, config, model, tokenizer, result, eval_dataset)
    return result.metrics
