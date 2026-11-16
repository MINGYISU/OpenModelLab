import copy
import math
import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from finetune.config import EmbedConfig, LoraConfig, TrainConfig
from finetune.models import get_family, tiny_config
from finetune.training.dpo import dpo_loss, sequence_logps
from finetune.training.embedding import contrastive_loss, encode, gradcache_step
from finetune.training.grpo import completion_mask, group_advantages, grpo_token_loss
from finetune.training.lora import LoRALinear, apply_lora, lora_disabled, merge_scratch_lora
from finetune.training.optim import lr_lambda, param_groups
from finetune.training.rewards import correctness_reward, countdown_reward, extract_answer, format_reward
from finetune.eval.reasoning import pass_at_k, trace_stats


class ScheduleTest(unittest.TestCase):
    def test_shapes(self):
        f = lambda name, step, **kw: lr_lambda(step, name=name, warmup=10, total=100, **kw)
        self.assertAlmostEqual(f("cosine", 0), 0.1)
        self.assertAlmostEqual(f("cosine", 9), 1.0)
        self.assertAlmostEqual(f("cosine", 55), 0.5)
        self.assertAlmostEqual(f("cosine", 100, min_ratio=0.1), 0.1)
        self.assertAlmostEqual(f("linear", 55), 0.5)
        self.assertEqual(f("constant", 99), 1.0)
        self.assertEqual(f("wsd", 89, decay_ratio=0.1), 1.0)
        self.assertAlmostEqual(f("wsd", 95, decay_ratio=0.1), 0.5)

    def test_no_weight_decay_on_norms_biases_embeddings(self):
        model = get_family("qwen2").causal_lm_class(tiny_config("qwen2"))
        decay, no_decay = param_groups(model, 0.1)
        names = {id(p): n for n, p in model.named_parameters()}
        self.assertTrue(all("norm" not in names[id(p)] and "bias" not in names[id(p)] for p in decay["params"]))
        self.assertTrue(any("embed_tokens" in names[id(p)] for p in no_decay["params"]))
        self.assertEqual(no_decay["weight_decay"], 0.0)


class LoRATest(unittest.TestCase):
    def test_scratch_lora_matches_peft(self):
        torch.manual_seed(0)
        base = get_family("qwen2").causal_lm_class(tiny_config("qwen2")).eval()
        cfg = LoraConfig(enabled=True, r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
        ours = apply_lora(copy.deepcopy(base), LoraConfig(**{**cfg.__dict__, "impl": "scratch"}))
        theirs = apply_lora(copy.deepcopy(base), cfg)
        ids = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            torch.testing.assert_close(ours(ids).logits, base(ids).logits)  # B = 0 at init
            for name, module in ours.named_modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.weight.normal_()
                    peft = theirs.get_submodule("base_model.model." + name)
                    peft.lora_A["default"].weight.copy_(module.lora_A.weight)
                    peft.lora_B["default"].weight.copy_(module.lora_B.weight)
            torch.testing.assert_close(ours(ids).logits, theirs(ids).logits, atol=1e-5, rtol=1e-5)
            with lora_disabled(ours):
                torch.testing.assert_close(ours(ids).logits, base(ids).logits)
            expected = ours(ids).logits
            torch.testing.assert_close(merge_scratch_lora(ours)(ids).logits, expected, atol=1e-5, rtol=1e-5)
        self.assertFalse(any(isinstance(m, LoRALinear) for m in ours.modules()))


class GradCacheTest(unittest.TestCase):
    def test_gradients_match_full_batch(self):
        torch.manual_seed(0)
        model = get_family("qwen2").base_model_class(tiny_config("qwen2"))
        cfg = EmbedConfig(pooling="last", chunk_size=2, temperature=0.05)
        inputs = {
            "query_input_ids": torch.randint(0, 256, (4, 6)), "query_attention_mask": torch.ones(4, 6, dtype=torch.long),
            "passage_input_ids": torch.randint(0, 256, (8, 7)), "passage_attention_mask": torch.ones(8, 7, dtype=torch.long),
        }
        inputs["passage_attention_mask"][1, 5:] = 0
        reference = copy.deepcopy(model)
        q = encode(reference, inputs["query_input_ids"], inputs["query_attention_mask"], "last")
        p = encode(reference, inputs["passage_input_ids"], inputs["passage_attention_mask"], "last")
        expected_loss = contrastive_loss(q, p, cfg.temperature)
        expected_loss.backward()
        loss = gradcache_step(model, inputs, cfg)
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)
        for (name, a), b in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4, msg=name)

    def test_contrastive_targets_follow_group_layout(self):
        q = torch.eye(2)
        p = torch.tensor([[1.0, 0], [0, 0], [0, 1.0], [0, 0]])  # [pos0, neg0, pos1, neg1]
        self.assertLess(contrastive_loss(q, p, 0.01).item(), 1e-6)


class DPOTest(unittest.TestCase):
    def test_losses(self):
        zero = torch.zeros(3)
        loss, _ = dpo_loss(zero, zero, zero, zero, beta=0.1)
        torch.testing.assert_close(loss, torch.full((3,), math.log(2)))
        winning = torch.tensor([5.0])
        loss, margin = dpo_loss(winning, -winning, torch.zeros(1), torch.zeros(1), beta=1.0)
        torch.testing.assert_close(margin, torch.tensor([10.0]))
        self.assertLess(loss.item(), 1e-4)
        ipo, _ = dpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), beta=0.5, loss_type="ipo")
        torch.testing.assert_close(ipo, torch.tensor([1.0]))
        simpo, _ = dpo_loss(torch.tensor([-2.0]), torch.tensor([-8.0]), None, None, beta=1.0, loss_type="simpo",
                            simpo_gamma=0.0, chosen_lengths=torch.tensor([2]), rejected_lengths=torch.tensor([4]))
        torch.testing.assert_close(simpo, -torch.nn.functional.logsigmoid(torch.tensor([1.0])))

    def test_sequence_logps_ignore_prompt(self):
        logits = torch.zeros(1, 4, 5)  # uniform -> log(1/5) per token
        labels = torch.tensor([[-100, -100, 2, 3]])
        logps, count = sequence_logps(logits, labels)
        torch.testing.assert_close(logps, torch.tensor([2 * math.log(0.2)]))
        self.assertEqual(count.item(), 2)


class GRPOTest(unittest.TestCase):
    def test_advantages_and_mask(self):
        advantages = group_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0, 2.0, 2.0]), group_size=2)
        torch.testing.assert_close(advantages[:2], advantages[2:4])
        self.assertGreater(advantages[0].item(), 0)
        self.assertLess(advantages[1].item(), 0)
        self.assertTrue(torch.all(advantages[4:] == 0))
        mask = completion_mask(torch.tensor([[5, 9, 7, 9], [5, 6, 7, 8]]), eos_ids={9})
        self.assertEqual(mask.tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])

    def test_on_policy_gradient_is_advantage_weighted_log_prob(self):
        logps = torch.tensor([[-1.0, -2.0], [-0.5, -0.1]], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        loss, stats = grpo_token_loss(logps, logps.detach(), logps.detach(), advantages, mask, 0.2, beta=0.1)
        loss.backward()
        # d/dlogp of -A * ratio, averaged per sequence then over the batch; KL(=0) has zero grad at ref == pi.
        torch.testing.assert_close(logps.grad, torch.tensor([[-0.25, -0.25], [0.5, 0.0]]))
        self.assertEqual(stats["kl"].item(), 0.0)

    def test_clipping_stops_gradient(self):
        logps = torch.tensor([[0.5]], requires_grad=True)
        loss, stats = grpo_token_loss(logps, torch.zeros(1, 1), None, torch.tensor([1.0]), torch.ones(1, 1), 0.2, 0.0)
        loss.backward()
        self.assertEqual(logps.grad.item(), 0.0)
        self.assertEqual(stats["clip_frac"].item(), 1.0)


class RewardTest(unittest.TestCase):
    def test_extraction_and_rewards(self):
        self.assertEqual(countdown_reward("<answer> (3 + 5) * 2 </answer>", '{"target": 16, "nums": [2, 3, 5]}'), 1.0)
        self.assertEqual(countdown_reward("<answer> 3 * 5 + 1 </answer>", '{"target": 16, "nums": [2, 3, 5]}'), 0.0)
        self.assertEqual(countdown_reward("<answer> __import__('os') </answer>", '{"target": 1, "nums": [1]}'), 0.0)
        self.assertEqual(extract_answer("so <answer> 42 </answer>"), "42")
        self.assertEqual(extract_answer(r"thus \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(extract_answer("1 then 2, finally 3,000"), "3,000")
        self.assertEqual(correctness_reward("<answer>3000.0</answer>", "3,000"), 1.0)
        self.assertEqual(correctness_reward("<answer>x = 7</answer>", "7"), 1.0)
        self.assertEqual(correctness_reward("no idea", "7"), 0.0)
        self.assertEqual(format_reward("reasoning</think> <answer>7</answer>", "7"), 1.0)
        self.assertEqual(format_reward("<answer>7</answer>", "7"), 0.0)

    def test_pass_at_k_and_trace_stats(self):
        self.assertEqual(pass_at_k(4, 0, 2), 0.0)
        self.assertEqual(pass_at_k(4, 4, 1), 1.0)
        self.assertAlmostEqual(pass_at_k(4, 1, 2), 0.5)
        stats = trace_stats("Hmm, 2+2. Wait, let me check.</think><answer>4</answer>")
        self.assertTrue(stats["closed_think"] and stats["format_ok"])
        self.assertEqual(stats["reflections"], 3)


if __name__ == "__main__":
    unittest.main()
