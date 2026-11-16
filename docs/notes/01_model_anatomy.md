# Model anatomy

Read `src/finetune/models/qwen2/modeling_qwen2.py` top to bottom; every other dense family is a variation.

| Piece | Where | What to notice |
| --- | --- | --- |
| Embedding | `Qwen2Model.embed_tokens` | token id -> vector; tied to `lm_head` in small models (`tie_word_embeddings`) |
| RMSNorm | `Qwen2RMSNorm` | `x / rms(x) * weight`, computed in fp32; no mean subtraction, no bias |
| RoPE | `Qwen2RotaryEmbedding`, `apply_rotary_pos_emb` | rotates pairs of q/k dims by position-dependent angles, so attention scores depend on *relative* position |
| GQA | `Qwen2Attention`, `repeat_kv` | `num_key_value_heads < num_attention_heads`: K/V heads are shared, shrinking the KV cache |
| Attention kernel | `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` | `eager` (readable), `sdpa`, `flash_attention_2`; same math |
| SwiGLU MLP | `Qwen2MLP` | `down(silu(gate(x)) * up(x))` |
| Block | `Qwen2DecoderLayer` | pre-norm residual: `x + attn(norm(x))`, then `x + mlp(norm(x))` |
| LM head + loss | `Qwen2ForCausalLM.forward` | logits = `lm_head(h)`; labels are shifted inside the loss so position t predicts t+1 |

Family differences:

- **Qwen3** (`qwen3/`): RMSNorm on queries and keys (`q_norm`, `k_norm`) before RoPE,
  which stabilises attention logits. The chat template has `enable_thinking`.
- **Llama** (`llama/`): the same skeleton as Qwen2, without q/k/v biases.
- **DeepSeek-V2/V3** (`deepseek_v3/`):
  - *MLA*: keys and values are compressed into a `kv_lora_rank` latent
    (`kv_a_proj_with_mqa`), then expanded (`kv_b_proj`). Only the latent and a small
    RoPE part need caching, so the KV cache is far smaller than with MHA or GQA. RoPE is
    applied to separate `qk_rope_head_dim` dims ("decoupled RoPE"), because rotation
    does not commute with the up-projection.
  - *MoE* (`DeepseekV3MoE`): every token goes to `n_shared_experts` plus top-k of
    `n_routed_experts` fine-grained experts. The router (`DeepseekV3TopkRouter`) uses
    sigmoid scores and group-limited routing (pick `topk_group` of `n_group` groups,
    then top-k inside them). Load balancing works without an auxiliary loss: a bias
    (`e_score_correction_bias`) shifts *selection*, not the mixing weights.
  - The first `first_k_dense_replace` layers use a dense MLP.
- **DeepSeek-R1-Distill-Qwen** checkpoints are plain Qwen2 architecture. Their
  "reasoning" lives entirely in the weights (post-training), not in the architecture.

Exercises:
1. Load a tiny model (`finetune.models.tiny_config("qwen2")`). Set
   `config._attn_implementation = "eager"` and step through `Qwen2Attention.forward`.
2. Count parameters per component for Qwen2.5-0.5B: how much is embedding vs. attention vs. MLP?
3. Compute KV-cache bytes per token for Qwen2.5-7B (GQA) vs. DeepSeek-V3 (MLA).
