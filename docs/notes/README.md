# Learning notes

Short notes that map concepts to the code that implements them. Read them in order:

1. [Model anatomy](01_model_anatomy.md): what runs inside the vendored Qwen/Llama/DeepSeek code.
2. [Training setup](02_training_setup.md): tokens, losses, batches, optimizer, schedules, precision, LoRA.
3. [Distributed training](03_distributed.md): torchrun, DDP, FSDP2, ZeRO, multi-node.
4. [Post-training and reasoning](04_post_training_and_reasoning.md): SFT, DPO, GRPO, R1-style reasoning.
5. [Embeddings and GradCache](05_embeddings_gradcache.md).
6. [Measuring behavior change](06_behavior_analysis.md).
