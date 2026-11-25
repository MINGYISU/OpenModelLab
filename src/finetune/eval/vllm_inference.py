"""Optional vLLM-backed text generation for standard Hugging Face checkpoints."""

from __future__ import annotations

from pathlib import Path


def load_model(path: str, revision: str | None = None, dtype: str = "bfloat16",
               tensor_parallel_size: int = 1):
    """Load a standard checkpoint with vLLM, rejecting project-specific variants."""
    if path.startswith("tiny:"):
        raise ValueError("The vLLM inference backend does not support tiny test models")

    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError("Install the optional vLLM dependency with `uv sync --extra vllm`") from exc

    from finetune.models.registry import read_config_json
    from finetune.variants import SPEC_KEY

    config = read_config_json(path, revision)
    if SPEC_KEY in config or (Path(path) / "variant.json").exists():
        raise ValueError("The vLLM inference backend does not currently support project model variants")

    kwargs = {
        "model": path,
        "tokenizer": path,
        "dtype": dtype,
        "trust_remote_code": False,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if revision is not None:
        kwargs["revision"] = revision
    return LLM(**kwargs)


def generate(model, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float,
             num_samples: int) -> list[str | list[str]]:
    """Generate text from already formatted prompts with vLLM."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError("Install the optional vLLM dependency with `uv sync --extra vllm`") from exc

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p if temperature > 0 else 1.0,
        n=num_samples,
    )
    outputs = model.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    results = []
    for request in outputs:
        completions = [completion.text for completion in request.outputs]
        if len(completions) != num_samples:
            raise RuntimeError(f"vLLM returned {len(completions)} samples; expected {num_samples}")
        results.append(completions[0] if num_samples == 1 else completions)
    return results
