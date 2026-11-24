import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from finetune.config import DataConfig, EmbedConfig, GRPOConfig
from finetune.data.collators import CausalLMCollator, PreferenceCollator
from finetune.data.contrastive import build_contrastive_dataset
from finetune.data.io import check_no_overlap
from finetune.data.preference import tokenize_preference
from finetune.data.pretrain import build_pretrain_dataset, pack_token_stream
from finetune.data.recipes import available_recipes, get_recipe, prepare, validate_record
from finetune.data.rl import build_rl_dataset, normalize_answer
from finetune.data.sft import IGNORE_INDEX, tokenize_conversation
from finetune.data.tokenizer import tiny_tokenizer

CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
    {"role": "user", "content": "again"},
    {"role": "assistant", "content": "ok"},
]


def supervised_text(tokenizer, example):
    return tokenizer.decode([t for t in example["labels"] if t != IGNORE_INDEX])


class SFTMaskTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()

    def test_assistant_only_mask_covers_answers_and_end_of_turn(self):
        example = tokenize_conversation(self.tokenizer, CONVERSATION, 512, loss_on="assistant")
        self.assertEqual(supervised_text(self.tokenizer, example), "yo<|im_end|>\nok<|im_end|>\n")

    def test_full_loss_supervises_every_token(self):
        example = tokenize_conversation(self.tokenizer, CONVERSATION, 512, loss_on="all")
        self.assertEqual(example["labels"], example["input_ids"])

    def test_overlength_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "max_length"):
            tokenize_conversation(self.tokenizer, CONVERSATION, 8)

    def test_history_rewriting_template_is_rejected(self):
        self.tokenizer.chat_template = (
            "{% for m in messages %}{% if loop.last %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            "{% else %}[{{ m.role }}]{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        with self.assertRaisesRegex(ValueError, "prefix-consistent"):
            tokenize_conversation(self.tokenizer, CONVERSATION, 512, loss_on="assistant")
        example = tokenize_conversation(self.tokenizer, CONVERSATION, 512, loss_on="all")
        self.assertEqual(example["labels"], example["input_ids"])

    def test_collator_masks_only_padding(self):
        short = tokenize_conversation(self.tokenizer, CONVERSATION[:3], 512)
        long = tokenize_conversation(self.tokenizer, CONVERSATION, 512)
        batch = CausalLMCollator(self.tokenizer.pad_token_id)([short, long])
        length = len(short["input_ids"])
        self.assertTrue((batch["labels"][0, length:] == IGNORE_INDEX).all())
        self.assertEqual(batch["labels"][0, :length].tolist(), short["labels"])
        self.assertEqual(int(batch["attention_mask"][0].sum()), length)


class PretrainTest(unittest.TestCase):
    def test_packing_concatenates_with_eos_and_drops_remainder(self):
        blocks = list(pack_token_stream([[1, 2, 3], [4, 5], [6, 7, 8, 9]], block_size=4, eos_id=0))
        self.assertEqual(blocks, [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]])

    def test_dataset_blocks_have_fixed_length(self):
        tokenizer = tiny_tokenizer()
        data = build_pretrain_dataset([{"text": "abc " * 50}, {"text": "xyz " * 30}], tokenizer,
                                      DataConfig(max_length=32, packing=True))
        self.assertTrue(data and all(len(e["input_ids"]) == 32 == len(e["labels"]) for e in data))


class OtherFormatsTest(unittest.TestCase):
    def test_preference_masks_prompt_tokens(self):
        tokenizer = tiny_tokenizer()
        example = tokenize_preference({"prompt": "q?", "chosen": "good", "rejected": "bad"}, tokenizer, 512, 256)
        self.assertEqual(supervised_text(tokenizer, {"labels": example["chosen_labels"]}), "good<|im_end|>\n")
        self.assertEqual(supervised_text(tokenizer, {"labels": example["rejected_labels"]}), "bad<|im_end|>\n")
        batch = PreferenceCollator(tokenizer.pad_token_id)([dict(example, ref_chosen_logp=-1.0, ref_rejected_logp=-2.0)])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["ref_logps"].tolist(), [-1.0, -2.0])

    def test_preference_accepts_conversation_format(self):
        tokenizer = tiny_tokenizer()
        prompt = [{"role": "user", "content": "q?"}]
        record = {"prompt": prompt, "chosen": prompt + [{"role": "assistant", "content": "good"}],
                  "rejected": prompt + [{"role": "assistant", "content": "bad"}]}
        example = tokenize_preference(record, tokenizer, 512, 256)
        self.assertEqual(supervised_text(tokenizer, {"labels": example["chosen_labels"]}), "good<|im_end|>\n")

    def test_rl_prompts_and_answers(self):
        self.assertEqual(normalize_answer("so 1,234 #### 1,234"), "1234")
        rows = build_rl_dataset([{"question": "2+2?", "answer": "4"}], tiny_tokenizer(), GRPOConfig())
        self.assertTrue(rows[0]["prompt_text"].endswith("<|im_start|>assistant\n"))
        raw = build_rl_dataset([{"prompt": "2+2?", "answer": "4"}], None, GRPOConfig(use_chat_template=False))
        self.assertIn("User: 2+2?", raw[0]["prompt_text"])

    def test_contrastive_layout_and_prefixes(self):
        rows = build_contrastive_dataset([{"query": "q", "positive": "p", "negatives": ["n1", "n2"]}],
                                         EmbedConfig(query_prefix="query: ", passage_prefix="passage: "))
        self.assertEqual(rows[0], {"query": "query: q", "passages": ["passage: p", "passage: n1", "passage: n2"]})

    def test_overlap_check(self):
        with self.assertRaisesRegex(ValueError, "1 evaluation records"):
            check_no_overlap([{"a": 1}, {"a": 2}], [{"a": 2}])


class RecipeTest(unittest.TestCase):
    def write_recipe(self, directory, body):
        path = Path(directory) / "toy.py"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return get_recipe(str(path))

    def test_prepare_validates_dedupes_across_splits_and_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"q": "a", "r": "1"}, {"q": "a", "r": "1"}, {"q": "b", "r": ""}, {"q": "c", "r": "3"},
                    {"q": "", "r": "x"}]
            Path(tmp, "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            Path(tmp, "raw_eval.jsonl").write_text(json.dumps({"q": "c", "r": "3"}), encoding="utf-8")
            recipe = self.write_recipe(tmp, f"""
                FORMAT = "sft"
                SOURCE = {str(Path(tmp, "raw.jsonl"))!r}
                EVAL_SOURCE = {str(Path(tmp, "raw_eval.jsonl"))!r}
                def convert(row):
                    if not row["q"]:
                        return None
                    return {{"messages": [{{"role": "user", "content": row["q"]}},
                                          {{"role": "assistant", "content": row["r"]}}]}}
            """)
            stats = prepare(recipe, Path(tmp, "out"))
            self.assertEqual((stats["train"], stats["eval"]), (1, 1))
            # duplicate "a" row + train "c" that leaks into eval; empty answer is invalid; empty q skipped
            self.assertEqual((stats["duplicate"], stats["invalid"], stats["skipped_by_recipe"]), (2, 1, 1))
            train = [json.loads(line) for line in Path(tmp, "out", "train.jsonl").read_text().splitlines()]
            self.assertEqual(train[0]["messages"][0]["content"], "a")

    def test_prepare_drops_overlength_sft_and_splits_eval(self):
        tokenizer = tiny_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            recipe = self.write_recipe(tmp, """
                FORMAT = "sft"
                def load(split):
                    return None if split == "eval" else [{"n": n} for n in (1, 2, 3, 400)]
                def convert(row):
                    return {"messages": [{"role": "user", "content": "x" * row["n"]},
                                         {"role": "assistant", "content": "ok"}]}
            """)
            stats = prepare(recipe, tmp, eval_ratio=0.34, tokenizer=tokenizer, max_length=200)
            self.assertEqual((stats["too_long"], stats["train"], stats["eval"]), (1, 2, 1))

    def test_bundled_recipes_load_and_hh_rlhf_parses_transcripts(self):
        self.assertTrue({"alpaca", "hh_rlhf"} <= set(available_recipes()))
        record = get_recipe("hh_rlhf").convert({
            "chosen": "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: joke?\n\nAssistant: good one",
            "rejected": "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: joke?\n\nAssistant: no",
        })
        validate_record(record, "preference")
        self.assertEqual(len(record["prompt"]), 3)
        self.assertEqual(record["chosen"], [{"role": "assistant", "content": "good one"}])


if __name__ == "__main__":
    unittest.main()
