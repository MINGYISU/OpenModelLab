import os
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from finetune.config import ExperimentConfig, LoraConfig, ModelConfig
from finetune.data.tokenizer import tiny_tokenizer
from finetune.eval.generation import load_for_inference
from finetune.models import build_model, get_family, tiny_config
from finetune.training.common import setup_model
from finetune.training.lora import lora_disabled, merge_for_save
from finetune.variants import new_parameters


def tiny(variant=None, family="qwen2", **args):
    torch.manual_seed(0)
    return build_model(ModelConfig(path=f"tiny:{family}", dtype="float32", variant=variant,
                                   variant_args=args)).eval()


class VariantTest(unittest.TestCase):
    def test_identity_init_preserves_the_reference_function(self):
        ids = torch.randint(0, 256, (2, 9))
        with torch.no_grad():
            torch.testing.assert_close(tiny("layerscale", init=1.0)(ids).logits, tiny()(ids).logits)
            self.assertFalse(torch.allclose(tiny("layerscale", init=0.5)(ids).logits, tiny()(ids).logits))

    def test_checkpoints_rebuild_the_variant(self):
        ids = torch.randint(0, 256, (2, 6))
        cases = [("qwen2", "layerscale", {"init": 0.3}), ("qwen2", "no_glu", {"match_params": True}),
                 ("llama", "drop_layers", {"drop": [0]})]
        for family, variant, args in cases:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                model = tiny(variant, family, **args)
                model.save_pretrained(tmp)
                reloaded = build_model(ModelConfig(path=tmp, dtype="float32")).eval()
                self.assertEqual(reloaded.config.finetune_variant["args"], model.config.finetune_variant["args"])
                with torch.no_grad():
                    torch.testing.assert_close(reloaded(ids).logits, model(ids).logits)
                    if variant == "drop_layers":  # renumbered layers must still index the KV cache correctly
                        self.assertTrue(torch.equal(model.generate(ids, max_new_tokens=3, do_sample=False),
                                                    reloaded.generate(ids, max_new_tokens=3, do_sample=False)))

    def test_drop_layers_records_the_reference_depth(self):
        model = tiny("drop_layers", drop=[-1])
        self.assertEqual(len(model.model.layers), 1)
        self.assertEqual(model.config.finetune_variant["reference_config"]["num_hidden_layers"], 2)

    def test_lora_adapter_of_a_variant_reloads_with_its_new_parameters(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as out:
            torch.manual_seed(0)
            get_family("qwen2").causal_lm_class(tiny_config("qwen2")).save_pretrained(base)
            config = ExperimentConfig(model=ModelConfig(path=base, dtype="float32", variant="layerscale"),
                                      lora=LoraConfig(enabled=True, r=4, alpha=8, dropout=0.0))
            model = setup_model(config)
            gammas = new_parameters(model)
            self.assertTrue(gammas and all(p.requires_grad for p in gammas.values()))
            with torch.no_grad():  # stand-in for training: move the new params and the adapter
                for param in gammas.values():
                    param.uniform_(0.5, 1.5)
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.normal_()
            trained = {n: p.detach().clone() for n, p in gammas.items()}
            with lora_disabled(model):  # the reference policy (GRPO) uses the variant's initial values
                self.assertTrue(all(torch.all(p == 1.0) for p in gammas.values()))
            self.assertTrue(all(torch.equal(gammas[n], trained[n]) for n in trained))
            merge_for_save(model, out)
            model.save_pretrained(out)
            tiny_tokenizer().save_pretrained(out)
            reloaded, _ = load_for_inference(out, dtype="float32")
            ids = torch.randint(0, 256, (1, 7))
            with torch.no_grad():
                torch.testing.assert_close(reloaded(ids).logits, model.eval()(ids).logits)

    def test_invalid_use_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown args"):
            tiny("layerscale", scale=2)
        with self.assertRaisesRegex(ValueError, "supports"):
            tiny("layerscale", family="deepseek_v3")


if __name__ == "__main__":
    unittest.main()
