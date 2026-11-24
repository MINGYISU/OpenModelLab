"""TEMPLATE for a new dataset. Copy it to `<your_dataset>.py` in this folder (no leading "_"),
fill in the parts marked TODO, then run:

  finetune prepare <your_dataset> --peek 3          # convert 3 rows and print them
  finetune prepare <your_dataset> --tokenizer <model> --max-length 2048
  # -> data/<your_dataset>/train.jsonl, eval.jsonl, prepare_stats.json

Target record per FORMAT (what convert() must return):
  sft          {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]}  (ends with assistant)
  preference   {"prompt": str|messages, "chosen": str|messages, "rejected": str|messages}
  rl           {"prompt": str, "answer": str}             (answer is checked by the reward function)
  contrastive  {"query": str, "positive": str, "negatives": [str, ...]}  (negatives optional)
  pretrain     {"text": str}
"""

NAME = "my_dataset"   # TODO: the name used on the command line
FORMAT = "sft"        # TODO: one of sft, preference, rl, contrastive, pretrain

# TODO: where the raw data lives. Either a Hugging Face dataset...
SOURCE = {"hf": "org/dataset-name", "config": None, "split": "train"}
# ...or a local JSONL file:  SOURCE = "raw/my_dataset.jsonl"

# Optional held-out split. If None, EVAL_RATIO of the converted training rows are held out.
EVAL_SOURCE = None    # e.g. {"hf": "org/dataset-name", "split": "test"}
EVAL_RATIO = 0.05

SYSTEM_PROMPT = None  # e.g. "You are a helpful assistant." (added to every example if set)

# ---- Evaluation (finetune eval task --recipe <your_dataset>, and studies) -----------------------
# Prefer built-in metrics; see `finetune score --help` and src/finetune/eval/metrics.py:
#   exact_match, contains, f1, rouge_l, bleu, numeric_match, choice, length
METRICS = ["exact_match", "f1"]   # TODO: e.g. ["choice"] for multiple choice, ["numeric_match"] for math
EXTRACT = "strip_think"           # prediction cleanup: strip_think, answer_tag, first_line, last_number, choice, none
GENERATION = {"max_new_tokens": 256, "temperature": 0.0}


def convert(row: dict):
    """Turn one raw row into one standard record (or a list of records, or None to skip it).

    Put every dataset-specific cleanup step here: renaming fields, joining columns,
    stripping markup, parsing labels, filtering bad rows. Return None for rows you do not
    want. Anything returned is checked against FORMAT; invalid records are counted and
    skipped (see prepare_stats.json).
    """
    prompt = row.get("question")   # TODO: your column names
    answer = row.get("answer")
    if not prompt or not answer:
        return None

    messages = [{"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": answer.strip()}]
    if SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"messages": messages}


# Optional: for data that is not JSONL or on the Hugging Face Hub (CSV, Excel, a folder of
# files, an API ...), define load() instead of SOURCE. It returns an iterable of dict rows.
# def load(split: str):
#     if split == "eval":
#         return None                      # None -> hold out EVAL_RATIO of the train rows
#     import csv
#     with open("raw/my_dataset.csv", newline="", encoding="utf-8") as handle:
#         return list(csv.DictReader(handle))


# Optional evaluation hooks. Defaults: SFT prompt = conversation minus the last assistant turn,
# reference = that turn; RL prompt = "prompt"/"question", reference = "answer".
# def build_prompt(record):          # -> str or list of messages
#     return record["messages"][:-1]
# def reference(record):             # -> str or list of acceptable strings
#     return record["messages"][-1]["content"]

# Only when no built-in metric fits: a dataset-specific metric (averaged + bootstrapped like the
# built-ins). Return a float, or a dict to report several named scores.
# def metric(prediction: str, reference: str, record: dict):
#     return float(prediction.strip().lower().startswith(reference.strip().lower()))
