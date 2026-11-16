# Post-training and reasoning models

**SFT** imitates demonstrations: maximise log p(response | prompt). It is cheap and
effective, but limited by the data and prone to copying style.

**DPO** (`training/dpo.py`) learns from preference pairs without a reward model or
sampling. The loss is `-log σ(β[(log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))])`.
- β sets how far the policy may drift from the reference.
- Log `logps/chosen`: it often *falls* during training even though the margin grows
  ("likelihood displacement").
- IPO replaces the log-sigmoid with a squared loss. SimPO drops the reference and
  length-normalises.

**GRPO** (`training/grpo.py`) is online RL with verifiable rewards (DeepSeekMath, R1):
1. Sample G answers per prompt.
2. Score them with programs (`training/rewards.py`).
3. Advantage = reward standardised *within the group*, so no value network is needed.
4. Apply the PPO-clipped policy gradient plus β·KL to the reference (the k3 estimator).

Things to watch in `log.jsonl`:
- `reward`, and each `reward/<name>` part.
- `completion_tokens`: long reasoning often emerges as RL proceeds.
- `kl`, `clip_frac`.
- `completions.jsonl` for qualitative changes.

Groups where every answer gets the same reward have zero advantage and give no signal,
so prompt difficulty matters.

**How reasoning models are made.**
- *DeepSeek-R1-Zero*: GRPO straight from a base model with correctness and format
  rewards. Long chains of thought, self-verification ("wait, let me check") and longer
  answers emerge. Reproduce this in miniature with
  `e5_grpo_r1zero_qwen2.5_1.5b_countdown.yaml`.
- *DeepSeek-R1*: a cold-start SFT, then RL, then rejection-sampling SFT, then RL again.
- *R1-Distill-Qwen*: plain SFT of Qwen2.5 on about 800k R1 traces
  (`e5_distill_sft_qwen2.5_1.5b.yaml` is the small version). For small models,
  distillation beats running RL directly.
- *Qwen3*: one model with thinking and non-thinking modes, switched by the chat
  template (`enable_thinking`).

**Test-time scaling.** Accuracy grows with thinking tokens. Measure it with
`finetune eval reasoning --samples k` (pass@k) and by capping `--max-new-tokens`.
`eval/reasoning.py` reports think length, the share of traces that close `</think>`,
reflection-phrase counts, and length for correct vs. incorrect answers.
