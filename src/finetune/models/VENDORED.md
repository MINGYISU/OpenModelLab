# Vendored model source

The model code in this folder is copied verbatim from the pinned
`transformers==5.23.0` release (Apache-2.0) by `scripts/vendor_models.py`. The
only automated change rewrites relative imports into absolute `transformers.*`
imports. We still rely on `transformers` for `PreTrainedModel` (weight loading
and saving), attention kernels, caching, masking utilities, generation and `Trainer`.

| Family | Upstream path | Covers checkpoints |
| --- | --- | --- |
| `qwen2` | `transformers/models/qwen2/` | Qwen2, Qwen2.5 (base/instruct/math), DeepSeek-R1-Distill-Qwen |
| `qwen3` | `transformers/models/qwen3/` | Qwen3 dense (hybrid thinking) |
| `llama` | `transformers/models/llama/` | Llama 3.x, SmolLM2, other Llama-style models |
| `deepseek_v2` | `transformers/models/deepseek_v2/` | DeepSeek-V2 / V2-Lite (MLA + MoE) |
| `deepseek_v3` | `transformers/models/deepseek_v3/` | DeepSeek-V3 / R1 architecture (MLA + MoE + MTP) |

## Rules

- `tests/test_models.py` checks that each vendored class produces the same logits as the
  upstream class (tiny random configs). It must keep passing until you deliberately
  change the math.
- Record every intentional local edit below (file, reason, date) so `git diff`
  against a fresh vendoring stays explainable.
- To upgrade, bump `transformers` in `pyproject.toml`, reinstall, commit, then
  run `python scripts/vendor_models.py` and review the diff. Re-vendoring
  overwrites local edits.

## Where to read first

- `qwen2/modeling_qwen2.py`: `Qwen2RMSNorm`, `Qwen2RotaryEmbedding` + `apply_rotary_pos_emb`
  (RoPE), `Qwen2Attention` (GQA via `repeat_kv`), `Qwen2MLP` (SwiGLU),
  `Qwen2DecoderLayer` (pre-norm residual blocks), `Qwen2ForCausalLM` (LM head + loss).
- `qwen3/modeling_qwen3.py`: the same skeleton plus `q_norm` / `k_norm` (QK-norm).
- `deepseek_v3/modeling_deepseek_v3.py`: `DeepseekV3Attention` (multi-head latent
  attention: low-rank `q_a/q_b`, `kv_a_proj_with_mqa` -> `kv_b_proj`, decoupled RoPE dims),
  `DeepseekV3TopkRouter` (sigmoid scores, group-limited top-k, bias-based load
  balancing via `e_score_correction_bias`), `DeepseekV3MoE` (routed + shared experts).

## Local edits

None yet.
