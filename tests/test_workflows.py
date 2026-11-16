import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finetune.cli import main
from finetune.data import load_conversations
from finetune.workflows import tokenize_conversations


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(content, encoding="utf-8")
            return load_conversations(path)

    def test_examples(self):
        self.assertEqual(len(load_conversations(ROOT / "examples/train.jsonl")), 3)
        self.assertEqual(len(load_conversations(ROOT / "examples/eval.jsonl")), 1)

    def test_valid_multiturn_unicode_and_blank_lines(self):
        messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "Explain NLP."},
            {"role": "assistant", "content": "Natural language processing."},
        ]
        self.assertEqual(self.load("\n" + json.dumps({"messages": messages}) + "\n"),
                         [{"messages": messages}])

    def test_invalid_records(self):
        cases = [
            "", "\n", "{bad json}", "null", "[]", "{}", '{"messages":[]}',
            '{"messages":[null]}',
            '{"messages":[{"role":"tool","content":"result"}]}',
            '{"messages":[{"role":"user","content":"question"}]}',
            '{"messages":[{"role":"assistant","content":"answer"}]}',
        ]
        for content in cases:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(content)

    def test_invalid_content_and_turn_order(self):
        for roles in (["user", "user"], ["assistant", "user"],
                      ["system", "system", "user", "assistant"]):
            with self.subTest(roles=roles), self.assertRaises(ValueError):
                self.load(json.dumps({"messages": [
                    {"role": role, "content": "text"} for role in roles
                ]}))
        for content in ("", " ", None, 4):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(json.dumps({"messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": content},
                ]}))

    def test_line_number_in_error(self):
        with self.assertRaisesRegex(ValueError, r":2: invalid JSON"):
            self.load("\ninvalid")

    def test_validate_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["validate", "--data", str(ROOT / "examples/train.jsonl")])
        self.assertEqual(json.loads(output.getvalue()), {"conversations": 3})

    def test_missing_dataset_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["validate", "--data", str(Path(directory) / "missing.jsonl")])
        self.assertEqual(error.exception.code, 2)

    def test_positive_cli_limits(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["embed", "--model", "unused", "--text", "hello", "--max-length", "0"])

    def test_training_rejects_overlap_before_loading_models(self):
        data = str(ROOT / "examples/train.jsonl")
        with patch("finetune.workflows.ml_stack") as stack:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["train", "--model", "unused", "--data", data,
                      "--eval-data", data, "--output-dir", "unused"])
            self.assertEqual(error.exception.code, 2)
            stack.assert_not_called()

    def test_training_does_not_overwrite_existing_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "adapter_config.json"
            artifact.write_text("existing", encoding="utf-8")
            with patch("finetune.workflows.ml_stack") as stack:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["train", "--model", "unused",
                          "--data", str(ROOT / "examples/train.jsonl"),
                          "--output-dir", directory])
                stack.assert_not_called()
            self.assertEqual(artifact.read_text(encoding="utf-8"), "existing")


class TokenizerStub:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        return "rendered conversation"

    def __call__(self, text, **kwargs):
        assert text == "rendered conversation"
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


class TokenizationTests(unittest.TestCase):
    def test_labels_preserve_eos_and_are_copied(self):
        data = tokenize_conversations([{"messages": []}], TokenizerStub(), 3)
        self.assertEqual(data[0]["labels"], [1, 2, 3])
        self.assertIsNot(data[0]["labels"], data[0]["input_ids"])

    def test_overlong_conversations_rejected_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            tokenize_conversations([{"messages": []}], TokenizerStub(), 2)

    def test_chat_template_required(self):
        tokenizer = TokenizerStub()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "chat template"):
            tokenize_conversations([], tokenizer, 3)


if __name__ == "__main__":
    unittest.main()
