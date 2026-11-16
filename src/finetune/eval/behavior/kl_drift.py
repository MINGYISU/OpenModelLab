"""Output-distribution drift: token-level KL(tuned || base) on held-out text.

Which tokens move most? The "superficial alignment" hypothesis (LIMA; Lin et al.,
2023 "Unlocking Spurious...") predicts that post-training mostly shifts stylistic /
discourse tokens while content tokens stay put; this measures it directly.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


@torch.no_grad()
def kl_drift(base, tuned, tokenizer, texts, max_length=512, top=20) -> dict:
    tokenizer.padding_side = "right"
    per_token = defaultdict(lambda: [0.0, 0])        # token id -> [sum KL, count] (KL at positions predicting it)
    prob_shift = defaultdict(lambda: [0.0, 0])       # token id -> [sum (p_tuned - p_base), count]
    by_position = defaultdict(lambda: [0.0, 0])
    total_kl, total_tokens = 0.0, 0
    for text in texts:
        batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=max_length).to(base.device)
        log_p = F.log_softmax(tuned(**batch).logits[0, :-1].float(), -1)
        log_q = F.log_softmax(base(**batch).logits[0, :-1].float(), -1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        targets = batch["input_ids"][0, 1:]
        shift = log_p.exp().gather(1, targets[:, None]).squeeze(1) - log_q.exp().gather(1, targets[:, None]).squeeze(1)
        for position, (token, value, delta) in enumerate(zip(targets.tolist(), kl.tolist(), shift.tolist())):
            per_token[token][0] += value
            per_token[token][1] += 1
            prob_shift[token][0] += delta
            prob_shift[token][1] += 1
            bucket = min(position // 64, 7) * 64
            by_position[bucket][0] += value
            by_position[bucket][1] += 1
        total_kl += kl.sum().item()
        total_tokens += kl.numel()
    decode = lambda token: tokenizer.decode([token])
    mean = lambda table, min_count=2: sorted(
        ((decode(t), v[0] / v[1], v[1]) for t, v in table.items() if v[1] >= min_count), key=lambda x: -x[1])
    return {
        "mean_kl_per_token": total_kl / max(1, total_tokens),
        "tokens": total_tokens,
        "kl_by_position": {f"{k}-{k + 63}": v[0] / v[1] for k, v in sorted(by_position.items())},
        "highest_kl_tokens": [dict(token=t, mean_kl=v, count=c) for t, v, c in mean(per_token)[:top]],
        "most_boosted_tokens": [dict(token=t, mean_prob_shift=v, count=c) for t, v, c in mean(prob_shift)[:top]],
        "most_suppressed_tokens": [dict(token=t, mean_prob_shift=v, count=c) for t, v, c in mean(prob_shift)[::-1][:top]],
    }
