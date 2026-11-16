"""Shared setup for transformers.Trainer-based stages.

ResearchTrainer replaces Trainer's default optimizer/scheduler with ours
(training/optim.py) so the param groups and schedules are explicit and
identical between the Trainer and the raw-loop backends.
"""

from __future__ import annotations

import torch
import transformers

from finetune import utils
from finetune.models import build_model
from finetune.training.lora import apply_lora, count_parameters, merge_for_save
from finetune.training.optim import build_optimizer, build_scheduler


class ResearchTrainer(transformers.Trainer):
    def __init__(self, *args, research_config=None, **kwargs):
        self.research_config = research_config
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            self.optimizer = build_optimizer(self.model if model is None else model, self.research_config.train)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(optimizer or self.optimizer, self.research_config.train, num_training_steps)
            self._created_lr_scheduler = True
        return self.lr_scheduler


def precision_flags(train_cfg) -> dict:
    cuda = torch.cuda.is_available()
    if train_cfg.precision == "bf16" and cuda and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requested but this GPU does not support it; use precision: fp16 or fp32")
    return {
        "bf16": train_cfg.precision == "bf16" and cuda,
        "fp16": train_cfg.precision == "fp16" and cuda,
        "tf32": train_cfg.tf32 if cuda else None,
        "use_cpu": not cuda,
    }


def model_load_dtype(config) -> str:
    """Load weights in fp32 when training full parameters with mixed precision (fp32 master weights);
    LoRA keeps the frozen base in the low-precision dtype to save memory."""
    if not config.lora.enabled and config.train.precision in {"bf16", "fp16"}:
        return "float32"
    if not torch.cuda.is_available():
        return "float32"
    return config.model.dtype


def setup_model(config, causal_lm: bool = True, task_type: str = "CAUSAL_LM"):
    model = build_model(config.model, causal_lm=causal_lm, dtype=model_load_dtype(config))
    model.config.use_cache = False
    if config.model.gradient_checkpointing:
        model.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        if config.lora.enabled:
            model.enable_input_require_grads()
    model = apply_lora(model, config.lora, task_type=task_type)
    trainable, total = count_parameters(model)
    utils.log(f"model {type(model).__name__}: trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def training_arguments(config, has_eval: bool, **overrides) -> transformers.TrainingArguments:
    train = config.train
    eval_strategy = "no" if not has_eval else ("steps" if train.eval_steps else "epoch")
    save_strategy = "steps" if train.save_steps else "epoch"
    kwargs = dict(
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        data_seed=config.run.seed,
        num_train_epochs=train.epochs,
        max_steps=train.max_steps,
        per_device_train_batch_size=train.per_device_batch_size,
        per_device_eval_batch_size=train.eval_batch_size or train.per_device_batch_size,
        gradient_accumulation_steps=train.grad_accum,
        learning_rate=train.lr,
        weight_decay=train.weight_decay,
        max_grad_norm=train.max_grad_norm,
        logging_steps=train.logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=train.eval_steps,
        save_strategy=save_strategy,
        save_steps=train.save_steps,
        save_total_limit=train.save_total_limit,
        report_to=[] if config.run.report_to == "none" else [config.run.report_to],
        dataloader_num_workers=train.dataloader_num_workers,
        remove_unused_columns=False,
        include_num_input_tokens_seen="non_padding",
        ddp_find_unused_parameters=False,
        fsdp=train.fsdp or None,
        fsdp_config=train.fsdp_config or None,
        deepspeed=train.deepspeed,
        **precision_flags(train),
    )
    kwargs.update(overrides)
    return transformers.TrainingArguments(**kwargs)


def finish(trainer, config, model, tokenizer, train_result, eval_dataset=None) -> None:
    if eval_dataset is not None:
        trainer.save_metrics("eval", trainer.evaluate())
    merge_for_save(trainer.model, config.run.output_dir if trainer.is_world_process_zero() else None)
    trainer.save_model(config.run.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config.run.output_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    utils.write_metadata(config.run.output_dir, config, model)
