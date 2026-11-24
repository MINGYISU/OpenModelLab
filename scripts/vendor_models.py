"""Copy model source files from the installed (pinned) transformers into our repo.

Usage:
    python scripts/vendor_models.py            # vendor all families
    python scripts/vendor_models.py qwen2 llama

Only relative imports are rewritten (`from ...x` -> `from transformers.x`), so a
diff against upstream shows exactly our own edits. Re-running overwrites local
edits; commit before re-vendoring.
"""

import re
import sys
from pathlib import Path

import transformers

FAMILIES = ["qwen2", "qwen3", "llama", "deepseek_v2", "deepseek_v3", "glm4", "glm4_moe", "qwen2_5_vl", "siglip2", "janus"]
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "finetune" / "models"
UPSTREAM = Path(transformers.__file__).parent / "models"

HEADER = (
    "# Vendored from transformers=={version}: transformers/models/{family}/{name}\n"
    "# Licensed under the Apache License, Version 2.0 (see the original header below).\n"
    "# Local edits are allowed; record them in src/finetune/models/VENDORED.md.\n\n"
)


def rewrite_imports(source: str) -> str:
    source = re.sub(r"^(\s*)from \.\.\. import ", r"\1from transformers import ", source, flags=re.M)
    source = re.sub(r"^(\s*)from \.\.\.(\w)", r"\1from transformers.\2", source, flags=re.M)
    source = re.sub(r"^(\s*)from \.\.(\w)", r"\1from transformers.models.\2", source, flags=re.M)
    return source


def vendor(family: str) -> None:
    destination = TARGET / family
    destination.mkdir(parents=True, exist_ok=True)
    exports = []
    for kind in ("configuration", "modeling"):
        name = f"{kind}_{family}.py"
        source = (UPSTREAM / family / name).read_text(encoding="utf-8")
        text = HEADER.format(version=transformers.__version__, family=family, name=name)
        (destination / name).write_text(text + rewrite_imports(source), encoding="utf-8")
        exports.append(f"from .{kind}_{family} import *  # noqa: F401,F403")
    (destination / "__init__.py").write_text("\n".join(exports) + "\n", encoding="utf-8")
    print(f"vendored {family} -> {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    for name in sys.argv[1:] or FAMILIES:
        if name not in FAMILIES:
            raise SystemExit(f"Unknown family {name!r}; choose from {FAMILIES}")
        vendor(name)
