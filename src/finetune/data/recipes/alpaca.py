"""Example SFT recipe: Stanford Alpaca (instruction + optional input -> output)."""

NAME = "alpaca"
FORMAT = "sft"
SOURCE = {"hf": "tatsu-lab/alpaca", "split": "train"}
EVAL_SOURCE = None
EVAL_RATIO = 0.02


def convert(row: dict):
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("input") or "").strip()
    output = (row.get("output") or "").strip()
    if not instruction or not output:
        return None
    prompt = f"{instruction}\n\n{context}" if context else instruction
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]}
