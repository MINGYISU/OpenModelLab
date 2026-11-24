import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from finetune.eval.metrics import compute_metrics
from finetune.eval.report import build_report, paired_pvalue
from finetune.eval.results import default_label
from finetune.study import _restart, load_study, plan, run_study

ROOT = Path(__file__).resolve().parents[1]


class MetricsTest(unittest.TestCase):
    def score(self, metric, prediction, reference, **kwargs):
        return compute_metrics([prediction], [reference], [metric], bootstrap=0, **kwargs)["scores"][metric]

    def test_builtin_metrics(self):
        self.assertEqual(self.score("exact_match", "The Eiffel Tower!", "eiffel tower"), 1.0)
        self.assertAlmostEqual(self.score("f1", "paris france", "paris"), 2 / 3)
        self.assertAlmostEqual(self.score("rouge_l", "a b c d", "a c d e"), 0.75)
        self.assertEqual(self.score("bleu", "the cat sat on the mat", "the cat sat on the mat"), 100.0)
        self.assertEqual(self.score("numeric_match", "so 1,234 apples", "#### 1234"), 1.0)
        self.assertEqual(self.score("choice", "<think>maybe A</think>The answer is (C).", "C"), 1.0)
        self.assertEqual(self.score("exact_match", "Lyon", ["Paris", "lyon"]), 1.0)  # any reference counts

    def test_extractor_and_custom_metric(self):
        result = compute_metrics(["<think>x</think><answer>42</answer>"], ["42"], ["exact_match"],
                                 extract="answer_tag", bootstrap=0,
                                 custom=lambda p, r, rec: {"tagged": float(p == r), "short": float(len(p) < 5)})
        self.assertEqual(result["scores"], {"exact_match": 1.0, "tagged": 1.0, "short": 1.0})

    def test_bootstrap_interval_brackets_the_score(self):
        result = compute_metrics(["a", "b", "c", "x"] * 10, ["a", "b", "c", "d"] * 10, ["exact_match", "bleu"])
        for name in ("exact_match", "bleu"):
            low, high = result["ci95"][name]
            self.assertLessEqual(low, result["scores"][name])
            self.assertGreaterEqual(high, result["scores"][name])


def record(label, seed, scores, per_example=None, **extra):
    return {"label": label, "dataset": "qa", "task": "task", "seed": seed, "scores": scores, "ci95": {},
            "per_example": per_example or {}, "n": 40, "reference": False, "variant": None,
            "num_parameters": 494_000_000, **extra}


class ReportTest(unittest.TestCase):
    def test_table_aggregates_seeds_and_marks_significance(self):
        good, bad = [1.0] * 36 + [0.0] * 4, [1.0] * 16 + [0.0] * 24
        records = [record("Base", s, {"exact_match": 0.4}, {"exact_match": bad}) for s in (1, 2)]
        records += [record("Mine", 1, {"exact_match": 0.85}, {"exact_match": good}, variant={"name": "v"}),
                    record("Mine", 2, {"exact_match": 0.95}, {"exact_match": good}, variant={"name": "v"})]
        report = build_report(records, baseline="Base", percent=True)
        mine = next(line for line in report["markdown"].splitlines() if line.startswith("| Mine"))
        self.assertIn("**90.0 ± 7.1** (+50.0)†", mine)
        self.assertIn(r"\textbf{90.0", report["latex"])
        self.assertIn("Mine,variant,qa,exact_match,0.9", report["csv"].replace("0.8999999999999999", "0.9"))
        self.assertGreater(paired_pvalue(good, good), 0.05)


class StudyTest(unittest.TestCase):
    def test_smoke_study_runs_end_to_end_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = {
                "name": "t", "train_config": str(ROOT / "configs/smoke/sft.yaml"), "launcher": "inprocess",
                "output_root": f"{tmp}/runs", "results_dir": f"{tmp}/results", "seeds": [1],
                "set": ["lora.enabled=false", "train.max_steps=1", "train.precision=fp32", "run.report_to=none"],
                "datasets": [{"name": "smoke", "train": str(ROOT / "examples/sft_train.jsonl"),
                              "eval": str(ROOT / "examples/sft_eval.jsonl")}],
                "models": [{"label": "Ref", "path": "tiny:qwen2", "train": False},
                           {"label": "Pruned", "path": "tiny:qwen2", "variant": "drop_layers"}],
                "eval": {"metrics": ["f1"], "max_new_tokens": 4, "bootstrap": 0},
            }
            path = Path(tmp) / "study.yaml"
            path.write_text(json.dumps(spec))
            run_study(path)
            results = json.loads((Path(tmp) / "results/smoke/Pruned__seed1.json").read_text())
            self.assertEqual(results["variant"]["name"], "drop_layers")
            self.assertIn("| Pruned |", (Path(tmp) / "results/report.md").read_text())
            self.assertTrue(all(step["done"] for step in plan(load_study(path))))

    def test_interrupted_training_resumes_from_its_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.assertEqual(_restart(run), [])
            (run / "trainer_state.json").write_text("{}")
            self.assertEqual(_restart(run), ["run.overwrite=true"])
            for step in (2, 10):
                (run / f"checkpoint-{step}").mkdir()
            self.assertEqual(_restart(run), [f"train.resume_from={run / 'checkpoint-10'}"])

    def test_default_labels_keep_ablations_apart(self):
        info = {"variant": {"name": "layerscale", "title": "LayerScale-LM", "args": {"init": 1.0, "branches": ["attn"]}},
                "trained": True, "run_name": "experiment", "seed": 1, "num_parameters": None}
        self.assertEqual(default_label("outputs/x", info), "LayerScale-LM (branches=['attn'])")

    def test_shipped_studies_are_valid(self):
        for path in sorted((ROOT / "configs/studies").glob("*.yaml")):
            if "train_config" in path.read_text():
                with self.subTest(study=path.name):
                    self.assertTrue(plan(load_study(path)))


if __name__ == "__main__":
    unittest.main()
