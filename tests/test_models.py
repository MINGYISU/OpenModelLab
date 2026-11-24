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
            if causal_lm is None:
                continue
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

    def _upstream_pair(self, family, upstream_cls):
        torch.manual_seed(0)
        fam = get_family(family)
        ours = (fam.causal_lm_class or fam.base_model_class)(tiny_config(family)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            ours.save_pretrained(tmp)
            upstream = getattr(transformers, upstream_cls).from_pretrained(tmp).eval()
        return ours, upstream

    def test_multimodal_inputs_match_upstream(self):
        """Image paths (vision towers, projectors, M-RoPE, contrastive heads) must also match upstream."""
        torch.manual_seed(1)
        text = torch.randint(0, 256, (1, 6))

        ours, upstream = self._upstream_pair("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration")
        grid = torch.tensor([[1, 4, 4]])  # 16 patches -> 4 merged image tokens
        ids = torch.cat([text[:, :2], torch.tensor([[302] + [300] * 4 + [303]]), text[:, 2:]], dim=1)
        inputs = dict(input_ids=ids, pixel_values=torch.randn(16, 3 * 2 * 4 * 4), image_grid_thw=grid)
        with torch.no_grad():
            torch.testing.assert_close(ours(**inputs).logits, upstream(**inputs).logits, atol=1e-5, rtol=1e-5)

        ours, upstream = self._upstream_pair("janus", "JanusForConditionalGeneration")
        ids = torch.cat([text[:, :2], torch.full((1, 16), 300), text[:, 2:]], dim=1)
        inputs = dict(input_ids=ids, pixel_values=torch.randn(1, 3, 16, 16))
        with torch.no_grad():
            torch.testing.assert_close(ours(**inputs).logits, upstream(**inputs).logits, atol=1e-5, rtol=1e-5)

        ours, upstream = self._upstream_pair("siglip2", "Siglip2Model")
        inputs = dict(input_ids=text.repeat(2, 1), pixel_values=torch.randn(2, 16, 3 * 4 * 4),
                      pixel_attention_mask=torch.ones(2, 16, dtype=torch.long),
                      spatial_shapes=torch.tensor([[4, 4], [4, 4]]))
        with torch.no_grad():
            got, expected = ours(**inputs), upstream(**inputs)
        torch.testing.assert_close(got.image_embeds, expected.image_embeds, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(got.text_embeds, expected.text_embeds, atol=1e-5, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "no generation head"):
            build_model(ModelConfig(path="tiny:siglip2"))

    def test_family_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            get_family("llama").causal_lm_class(tiny_config("llama")).save_pretrained(tmp)
            with self.assertRaisesRegex(ValueError, "model_type='llama'"):
                build_model(ModelConfig(path=tmp, family="qwen2"))


if __name__ == "__main__":
    unittest.main()
