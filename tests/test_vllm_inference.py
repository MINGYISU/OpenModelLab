import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from finetune.eval.generation import generate, load_for_inference
from finetune.eval import run_eval
from finetune.cli import _single_process_gpu
from finetune.eval.vllm_inference import load_model as load_vllm_model


class VllmInferenceTest(unittest.TestCase):
    def test_generation_preserves_sampling_groups_and_greedy_settings(self):
        requests = []

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModel:
            def generate(self, prompts, sampling_params, use_tqdm):
                requests.append((prompts, sampling_params.kwargs, use_tqdm))
                label = "first" if prompts[0].startswith("one") else "second"
                return [types.SimpleNamespace(outputs=[
                    types.SimpleNamespace(text=f"{label}-a"),
                    types.SimpleNamespace(text=f"{label}-b"),
                ])]

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            output = generate(
                FakeModel(),
                tokenizer=types.SimpleNamespace(
                    chat_template="template",
                    apply_chat_template=lambda messages, **kwargs: messages[-1]["content"] + " formatted",
                ),
                prompts=["one", "two"],
                max_new_tokens=12,
                temperature=0.0,
                num_samples=2,
                batch_size=1,
                backend="vllm",
            )

        self.assertEqual(output, [["first-a", "first-b"], ["second-a", "second-b"]])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0][0], ["one formatted"])
        self.assertEqual(requests[0][1], {
            "max_tokens": 12, "temperature": 0.0, "top_p": 1.0, "n": 2,
        })
        self.assertFalse(requests[0][2])

    def test_vllm_rejects_adapters_before_loading(self):
        with self.assertRaisesRegex(ValueError, "does not currently support PEFT adapters"):
            load_for_inference("base-model", adapter="adapter-dir", backend="vllm")
        with self.assertRaisesRegex(ValueError, "supports text generation only"):
            load_for_inference("base-model", causal_lm=False, backend="vllm")

    def test_vllm_rejects_project_variants(self):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = lambda **kwargs: self.fail("variant checkpoint must not load")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({
                "model_type": "qwen2", "finetune_variant": {},
            }))
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                with self.assertRaisesRegex(ValueError, "does not currently support project model variants"):
                    load_vllm_model(tmp)

    def test_vllm_load_forwards_revision_and_tensor_parallel_size(self):
        class FakeLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                model = load_vllm_model(tmp, revision="rev", dtype="float16", tensor_parallel_size=2)

        self.assertEqual(model.kwargs["revision"], "rev")
        self.assertEqual(model.kwargs["dtype"], "float16")
        self.assertEqual(model.kwargs["tensor_parallel_size"], 2)
        self.assertFalse(model.kwargs["trust_remote_code"])

    def test_vllm_preserves_visible_gpus(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True):
            _single_process_gpu(preserve_visible_gpus=True)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_vllm_is_restricted_to_generation_evaluations(self):
        with self.assertRaisesRegex(ValueError, "only for eval task and eval reasoning"):
            run_eval(types.SimpleNamespace(task="ppl", inference_backend="vllm"))


if __name__ == "__main__":
    unittest.main()
