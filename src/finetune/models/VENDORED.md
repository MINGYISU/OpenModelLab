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
| `deepseek_v3` | `transformers/models/deepseek_v3/` | DeepSeek-V3 / R1 architecture (MLA + MoE + MTP); also Kimi-K2 / K2-Thinking and Moonlight-16B-A3B (`model_type: deepseek_v3`) |
| `glm4` | `transformers/models/glm4/` | GLM-4-0414, GLM-Z1 (dense) |
| `glm4_moe` | `transformers/models/glm4_moe/` | GLM-4.5 / 4.5-Air / 4.6 (MoE) |
| `qwen2_5_vl` | `transformers/models/qwen2_5_vl/` | Qwen2.5-VL (image/video + text -> text); base for GME / VLM2Vec / jina-v4 embedders |
| `siglip2` | `transformers/models/siglip2/` | SigLIP2 dual encoder (contrastive image/text embeddings; no generation head, use `causal_lm=False`) |
| `janus` | `transformers/models/janus/` | Janus / Janus-Pro (image understanding + image-token generation) |

Only `configuration_*.py` and `modeling_*.py` are vendored. Multimodal processors
(image preprocessing, tokenizers) still come from `transformers`.

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

- `janus/modeling_janus.py` (2026-11-24): `JanusModel.language_model` is built with the
  vendored `finetune.models.llama.LlamaModel` instead of `transformers.AutoModel`, so no
  upstream model class is used. Weight names are unchanged; parity is checked by
  `tests/test_models.py`. Re-vendoring reverts this; re-apply it afterwards.
