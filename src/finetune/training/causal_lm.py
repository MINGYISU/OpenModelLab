"""Pre-training / continued pre-training and supervised fine-tuning (causal LM loss)."""

from __future__ import annotations

from finetune import utils
from finetune.data.collators import CausalLMCollator
from finetune.data.io import check_no_overlap, iter_records, load_records, mix_records
from finetune.data.pretrain import build_pretrain_dataset, streaming_pretrain_dataset
from finetune.data.sft import build_sft_dataset
from finetune.data.tokenizer import load_tokenizer
from finetune.training.common import ResearchTrainer, finish, setup_model, training_arguments


def _is_streaming(spec) -> bool:
    return isinstance(spec, dict) and spec.get("streaming", False)


def build_datasets(config, tokenizer):
    data = config.data
    if config.train.stage == "pretrain":
        if _is_streaming(data.train):
            shard_world = utils.world_size() if config.train.backend == "loop" else 1
            shard_rank = utils.rank() if config.train.backend == "loop" else 0
            train = streaming_pretrain_dataset(data.train, tokenizer, data, shard_rank, shard_world, config.run.seed)
        else:
            records = load_records(data.train, data.max_samples)
            if data.replay and data.replay_ratio > 0:
                records = list(mix_records(iter(records), iter_records(data.replay), data.replay_ratio, config.run.seed))
            train = build_pretrain_dataset(records, tokenizer, data)
        evaluation = (build_pretrain_dataset(load_records(data.eval, data.max_eval_samples), tokenizer, data)
                      if data.eval else None)
        return train, evaluation
    train_records = load_records(data.train, data.max_samples)
    eval_records = load_records(data.eval, data.max_eval_samples) if data.eval else []
    check_no_overlap(train_records, eval_records, key=lambda r: str(r.get("messages")))
    train = build_sft_dataset(train_records, tokenizer, data)
    evaluation = build_sft_dataset(eval_records, tokenizer, data) if eval_records else None
    return train, evaluation


def run(config) -> dict:
    utils.prepare_output_dir(config.run.output_dir, config.run.overwrite, bool(config.train.resume_from))
    utils.set_seed(config.run.seed)
    tokenizer = load_tokenizer(config.model)
    train_dataset, eval_dataset = build_datasets(config, tokenizer)
    if hasattr(train_dataset, "__len__"):
        utils.log(f"{config.train.stage}: {len(train_dataset)} train examples"
                  + (f", {len(eval_dataset)} eval examples" if eval_dataset else ""))
    model = setup_model(config)
    collator = CausalLMCollator(tokenizer.pad_token_id)

    if config.train.backend == "loop":
        from finetune.training.loop import train_loop

        metrics = train_loop(config, model, train_dataset, eval_dataset, collator, tokenizer)
        utils.write_metadata(config.run.output_dir, config, model)
        return metrics
    if config.train.backend != "trainer":
        raise ValueError("train.backend must be 'trainer' or 'loop'")

    trainer = ResearchTrainer(
        model=model, args=training_arguments(config, eval_dataset is not None),
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collator,
        processing_class=tokenizer, research_config=config,
    )
    result = trainer.train(resume_from_checkpoint=config.train.resume_from)
    finish(trainer, config, model, tokenizer, result, eval_dataset)
    return result.metrics
