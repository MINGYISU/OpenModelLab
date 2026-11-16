"""LoRA SFT, generation, and embedding baselines using Hugging Face models."""

import json
from pathlib import Path

from finetune.data import load_conversations


def ml_stack():
    try:
        import torch
        import transformers
    except ImportError as error:
        raise ImportError("Install ML dependencies with pip install -e '.[ml]'") from error
    return torch, transformers


def load_tokenizer(transformers, model, revision):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_conversations(records, tokenizer, max_length):
    if not tokenizer.chat_template:
        raise ValueError("Use an instruction/chat model with a tokenizer chat template")
    examples = []
    for index, record in enumerate(records, 1):
        text = tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, add_special_tokens=False)
        if len(tokens["input_ids"]) > max_length:
            raise ValueError(
                f"Conversation {index} exceeds --max-length={max_length}; "
                "shorten the conversation or increase the limit"
            )
        tokens["labels"] = tokens["input_ids"].copy()
        examples.append(tokens)
    return examples


def train(args):
    records = load_conversations(args.data)
    evaluation = load_conversations(args.eval_data) if args.eval_data else None
    if evaluation:
        training_messages = {json.dumps(row["messages"], sort_keys=True) for row in records}
        if any(json.dumps(row["messages"], sort_keys=True) in training_messages
               for row in evaluation):
            raise ValueError("Training and evaluation data contain overlapping conversations")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting an experiment")
    torch, transformers = ml_stack()
    from peft import LoraConfig, TaskType, get_peft_model

    transformers.set_seed(args.seed)
    tokenizer = load_tokenizer(transformers, args.model, args.revision)
    training_data = tokenize_conversations(records, tokenizer, args.max_length)
    eval_data = tokenize_conversations(evaluation, tokenizer, args.max_length) if evaluation else None
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules="all-linear",
    ))
    model.print_trainable_parameters()
    trainer = transformers.Trainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(output), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=2e-4, seed=args.seed, data_seed=args.seed,
            bf16=bf16, use_cpu=not torch.cuda.is_available(),
            eval_strategy="epoch" if eval_data else "no",
            save_strategy="epoch", logging_steps=1, report_to="none",
        ),
        train_dataset=training_data, eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100,
        ),
    )
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    trainer.save_metrics("train", result.metrics)
    if eval_data:
        trainer.save_metrics("eval", trainer.evaluate())
    with (output / "experiment.json").open("w", encoding="utf-8") as destination:
        json.dump({
            **vars(args), "resolved_model_revision": getattr(model.config, "_commit_hash", None),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        }, destination, indent=2)


def generate(args):
    torch, transformers = ml_stack()
    tokenizer = load_tokenizer(transformers, args.adapter or args.model, args.revision)
    if not tokenizer.chat_template:
        raise ValueError("Use an instruction/chat model with a tokenizer chat template")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False, use_safetensors=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def embed(args):
    torch, transformers = ml_stack()
    tokenizer = load_tokenizer(transformers, args.model, args.revision)
    model = transformers.AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False, use_safetensors=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    inputs = tokenizer(
        args.text, padding=True, truncation=True, max_length=args.max_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1)
    print(json.dumps(embeddings.cpu().tolist()))
