import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from finetune.config import apply_overrides, from_dict, load_config


class ConfigTest(unittest.TestCase):
    def test_multiple_bases_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.yaml").write_text("train: {lr: 1.0e-4, epochs: 2}\nrun: {seed: 1}\n")
            (root / "b.yaml").write_text("train: {lr: 5.0e-5}\n")
            (root / "exp.yaml").write_text("base: [a.yaml, b.yaml]\nrun: {name: x}\n")
            config = load_config(root / "exp.yaml", ["train.grad_accum=4", "lora.target_modules=[q_proj,v_proj]",
                                                     "train.lr=3e-4", "train.max_steps=1e3"])
        self.assertEqual(config.train.lr, 3e-4)
        self.assertEqual(config.train.max_steps, 1000)
        self.assertEqual(config.train.epochs, 2)
        self.assertEqual((config.run.seed, config.run.name), (1, "x"))
        self.assertEqual(config.train.grad_accum, 4)
        self.assertEqual(config.lora.target_modules, ["q_proj", "v_proj"])

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown keys in \\[train\\]"):
            from_dict({"train": {"learning_rate": 1}})
        with self.assertRaisesRegex(ValueError, "Unknown config sections"):
            from_dict({"optimizer": {}})
        with self.assertRaises(ValueError):
            apply_overrides({}, ["no_equals_sign"])

    def test_example_configs_parse(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        for path in list(root.glob("smoke/*.yaml")) + list(root.glob("experiments/*.yaml")):
            with self.subTest(config=path.name):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
