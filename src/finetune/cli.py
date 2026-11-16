"""Command-line entry points; heavy dependencies are loaded only when needed."""

import argparse
import json

from finetune.data import load_conversations


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a conversation JSONL file")
    validate.add_argument("--data", required=True)

    train = commands.add_parser("train", help="Supervised fine-tuning with LoRA")
    train.add_argument("--data", required=True)
    train.add_argument("--eval-data", help="Separate held-out conversation JSONL")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=positive_int, default=1)
    train.add_argument("--batch-size", type=positive_int, default=1)
    train.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    train.add_argument("--max-length", type=positive_int, default=1024)
    train.add_argument("--seed", type=int, default=42)

    generate = commands.add_parser("generate", help="Run a base model or saved adapter")
    generate.add_argument("--adapter", help="Local LoRA adapter directory")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=positive_int, default=128)

    embed = commands.add_parser("embed", help="Extract normalized mean-pooled embeddings")
    embed.add_argument("--text", required=True, action="append")
    embed.add_argument("--max-length", type=positive_int, default=512)

    for command in (train, generate, embed):
        command.add_argument("--model", required=True, help="Hugging Face model ID or local path")
        command.add_argument("--revision", default="main", help="Model revision; pin a commit for reproducibility")

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            records = load_conversations(args.data)
            print(json.dumps({"conversations": len(records)}))
        else:
            from finetune import workflows

            getattr(workflows, args.command)(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
