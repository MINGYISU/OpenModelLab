import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from finetune.cli import main
from finetune.data.tokenizer import tiny_tokenizer
from finetune.eval.behavior.kl_drift import kl_drift
from finetune.eval.behavior.moe_router import router_shift
from finetune.eval.behavior.representation import linear_cka
from finetune.eval.behavior.weight_diff import effective_rank, weight_diff
from finetune.models import get_family, tiny_config

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTest(unittest.TestCase):
    def test_identical_models_show_no_change(self):
        torch.manual_seed(0)
        model = get_family("qwen2").causal_lm_class(tiny_config("qwen2")).eval()
        result = weight_diff(model, model)
        self.assertEqual(result["parameters_changed"], 0)
        drift = kl_drift(model, model, tiny_tokenizer(), ["hello world, hello again"])
        self.assertAlmostEqual(drift["mean_kl_per_token"], 0.0, places=6)

    def test_effective_rank_and_cka(self):
        rank_one = torch.outer(torch.arange(1.0, 5.0), torch.arange(1.0, 4.0))
        rank, top1 = effective_rank(rank_one)
        self.assertAlmostEqual(rank, 1.0, places=4)
        self.assertAlmostEqual(top1, 1.0, places=5)
        x = torch.randn(50, 8)
        rotation, _ = torch.linalg.qr(torch.randn(8, 8))
        self.assertAlmostEqual(linear_cka(x, 3 * x @ rotation), 1.0, places=6)

    def test_router_usage_is_a_distribution(self):
        torch.manual_seed(0)
        model = get_family("deepseek_v3").causal_lm_class(tiny_config("deepseek_v3")).eval()
        result = router_shift(model, model, tiny_tokenizer(), ["mixture of experts routing test"])
        layer = next(iter(result["layers"].values()))
        self.assertAlmostEqual(sum(layer["usage_base"]), 1.0, places=5)
        self.assertEqual(result["mean_total_variation"], 0.0)


class EndToEndTest(unittest.TestCase):
    def test_sft_both_backends_then_generate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for backend in ("trainer", "loop"):
                output = Path(tmp) / backend
                main(["train", str(ROOT / "configs/smoke/sft.yaml"), "--set", f"run.output_dir={output}",
                      "--set", f"train.backend={backend}", "--set", "train.max_steps=2",
                      "--set", "lora.impl=scratch", "--set", "train.precision=fp32",
                      "--set", f"data.train={ROOT / 'examples/sft_train.jsonl'}",
                      "--set", f"data.eval={ROOT / 'examples/sft_eval.jsonl'}"])
                meta = json.loads((output / "experiment.json").read_text())
                self.assertEqual(meta["config"]["train"]["backend"], backend)
                self.assertTrue((output / "model.safetensors").exists())
                self.assertTrue((output / "scratch_lora_adapters.pt").exists())
            main(["generate", "--model", str(Path(tmp) / "loop"), "--prompt", "hi", "--max-new-tokens", "4"])


if __name__ == "__main__":
    unittest.main()
