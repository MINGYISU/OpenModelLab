# Embeddings and GradCache

**From LM to encoder.** Take the base model (`*Model`, no LM head) and pool its hidden states:
- `mean`: mask-aware average, the classic choice for bidirectional encoders.
- `last`: the hidden state of the appended EOS token. For causal LMs this is the only
  position that has seen the whole text, so it is the usual choice for LLM embedders
  (E5-Mistral, GTE-Qwen).
- `cls`: the first token. It suits BERT-style models, but is a poor fit for causal LMs.

Normalise the vectors and score pairs with cosine similarity.

**InfoNCE** (`training/embedding.py:contrastive_loss`): softmax over
`similarity / temperature`, where each query's own positive is the correct "class".
Every other passage in the batch, including other queries' positives and all hard
negatives, acts as a negative.
- Low temperatures (0.01–0.05) sharpen the distribution.
- Instruction prefixes on queries ("Instruct: ...\nQuery: ") make the embedding
  task-aware.

**Why batch size matters.** More in-batch negatives make a harder and more informative
loss. With `gather_across_ranks`, passages are all-gathered across GPUs, so the
negatives come from the *global* batch. `torch.distributed.nn.functional.all_gather`
keeps gradients flowing back to every rank.

**GradCache** (`gradcache_step`): memory grows with batch size only through the
activations, so the computation is split in two:
1. Encode every chunk without gradients and keep only the small representations.
2. Compute the full-batch loss on those representations and backprop to get
   ∂L/∂rep, the "cache".
3. Re-encode each chunk with gradients and backprop `sum(rep · cached_grad)`. By the
   chain rule this is exactly ∂L/∂θ. Under DDP, all but the last chunk run inside
   `no_sync`, and RNG state is replayed so dropout masks match.

`tests/test_training.py::GradCacheTest` checks that the gradients equal a normal
full-batch backward pass.
