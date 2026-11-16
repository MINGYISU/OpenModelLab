# Measuring behavior change

Compare a model *before* and *after* a training stage. Loss alone does not say what changed.

| Question | Tool | Reading the output |
| --- | --- | --- |
| Which weights moved, and how much? | `analyze weights` | `rel_change_by_layer` and `rel_change_by_kind`, the top-changed matrices, and the `effective_rank` of ΔW (LoRA updates are ≤ r by construction; are full fine-tuning updates also low-rank?) |
| Did the output distribution move, and where? | `analyze kl` | mean token KL(tuned‖base), KL by position, and the most boosted or suppressed tokens (style vs. content) |
| Did internal representations change? | `analyze cka` | CKA ≈ 1 means a layer's representation barely changed; find where the change concentrates |
| When does the answer become decodable? | `analyze logit-lens` | the answer token's per-layer probability and rank, base vs. tuned |
| Did MoE routing shift? | `analyze router` | expert-usage entropy and total variation per MoE layer |
| Is the model better, and what was forgotten? | `eval ppl/reasoning/benchmarks` | in-domain gain vs. general-benchmark loss; report the mean ± std over ≥3 seeds |
| Did behavior shift qualitatively? | `eval reasoning --output`, `generate` | length, format, refusals, think traces |

Good controls:
- The same data, seed and step budget across the arms you compare.
- A seed-only baseline (two runs that differ only in seed) to calibrate noise in every metric.
- Fixed decoding settings (greedy, or a fixed temperature with many samples).
- Held-out data that is separate from the training data by source.
