"""Weight-space comparison of a base and a trained model.

For every weight matrix W (base) and W' (trained), with dW = W' - W:
  rel_change   = ||dW||_F / ||W||_F
  effective_rank(dW) = exp(entropy of normalised singular values) (Roy & Vetterli, 2007)
  top1_energy  = sigma_1^2 / sum sigma_i^2  (how close dW is to rank-1)
LoRA updates are rank <= r by construction; full fine-tuning updates often turn
out to be surprisingly low-rank as well, which this lets you measure.
"""

from __future__ import annotations

import re
from collections import defaultdict

import torch

LAYER = re.compile(r"\.layers\.(\d+)\.")


def module_kind(name: str) -> str:
    for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                 "gate_proj", "up_proj", "down_proj", "gate_up_proj", "shared_experts", "experts", "mlp.gate",
                 "embed_tokens", "lm_head", "norm"):
        if kind in name:
            return kind
    return "other"


@torch.no_grad()
def effective_rank(matrix: torch.Tensor) -> tuple[float, float]:
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.pow(2)
    if energy.sum() == 0:
        return 0.0, 0.0
    p = singular / singular.sum()
    entropy = -(p * torch.log(p.clamp(min=1e-12))).sum()
    return torch.exp(entropy).item(), (energy[0] / energy.sum()).item()


@torch.no_grad()
def weight_diff(base, tuned, svd_max_dim: int = 8192) -> dict:
    base_params = dict(base.named_parameters())
    rows = []
    for name, tuned_param in tuned.named_parameters():
        if name not in base_params:
            continue
        w, w2 = base_params[name].float().cpu(), tuned_param.float().cpu()
        if w.shape != w2.shape:
            continue
        delta = w2 - w
        row = {"name": name, "kind": module_kind(name), "layer": int(m.group(1)) if (m := LAYER.search(name)) else None,
               "numel": w.numel(), "delta_norm": delta.norm().item(), "weight_norm": w.norm().item()}
        row["rel_change"] = row["delta_norm"] / max(row["weight_norm"], 1e-12)
        if delta.ndim == 2 and max(delta.shape) <= svd_max_dim and row["delta_norm"] > 0:
            row["effective_rank"], row["top1_energy"] = effective_rank(delta)
        rows.append(row)
    by_layer, by_kind = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        for table, key in ((by_layer, row["layer"]), (by_kind, row["kind"])):
            table[key][0] += row["delta_norm"] ** 2
            table[key][1] += row["weight_norm"] ** 2
    summarize = lambda table: {str(k): (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12) for k, v in sorted(table.items(), key=lambda kv: str(kv[0]))}
    changed = [r for r in rows if r["delta_norm"] > 0]
    return {
        "parameters_changed": sum(r["numel"] for r in changed),
        "parameters_total": sum(r["numel"] for r in rows),
        "rel_change_by_layer": summarize(by_layer),
        "rel_change_by_kind": summarize(by_kind),
        "top_changed": sorted(changed, key=lambda r: -r["rel_change"])[:20],
        "per_parameter": rows,
    }
