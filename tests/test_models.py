import os
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers

from finetune.config import ModelConfig
from finetune.models import FAMILY_CLASSES, build_model, get_family, tiny_config


class VendoredModelTest(unittest.TestCase):
    def test_logits_match_upstream_transformers(self):
        """The vendored source must be numerically identical to the pinned upstream classes."""
        for family, (_, _, causal_lm) in FAMILY_CLASSES.items():
            with self.subTest(family=family):
                torch.manual_seed(0)
                ours = get_family(family).causal_lm_class(tiny_config(family)).eval()
                self.assertTrue(type(ours).__module__.startswith("finetune.models."))
                with tempfile.TemporaryDirectory() as tmp:
                    ours.save_pretrained(tmp)
                    upstream = getattr(transformers, causal_lm).from_pretrained(tmp).eval()
                    reloaded = build_model(ModelConfig(path=tmp, dtype="float32"))
                ids = torch.randint(0, 256, (2, 10))
                with torch.no_grad():
                    expected = upstream(ids).logits
                    torch.testing.assert_close(ours(ids).logits, expected, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(reloaded(ids).logits, expected, atol=1e-5, rtol=1e-5)
                self.assertIs(type(reloaded), type(ours))

    def test_family_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            get_family("llama").causal_lm_class(tiny_config("llama")).save_pretrained(tmp)
            with self.assertRaisesRegex(ValueError, "model_type='llama'"):
                build_model(ModelConfig(path=tmp, family="qwen2"))


if __name__ == "__main__":
    unittest.main()
