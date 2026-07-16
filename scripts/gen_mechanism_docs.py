"""Batch-generate per-model mechanism docs (Mermaid diagrams + text) into
``docs/generated/<model>_mechanism.md``.

Model-general: pass any list of maths model names (built for the ~40-model zoo).
Each doc is map-derived (no model forward passes) so this is cheap to batch.

Run:  PYTHONPATH=. python scripts/gen_mechanism_docs.py [model_name ...]
      (defaults to the accurate model set if none given)
"""
from __future__ import annotations

import os
import sys

from quanta_maths import build_mechanism_markdown, ACCURATE_MODELS

OUT_DIR = "docs/generated"


def main(models):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in models:
        try:
            doc = build_mechanism_markdown(name)
        except Exception as e:  # a model with no published map, etc.
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            continue
        path = os.path.join(OUT_DIR, f"{name}_mechanism.md")
        with open(path, "w") as f:
            f.write(doc)
        print(f"  wrote {path}")


if __name__ == "__main__":
    models = sys.argv[1:] or ACCURATE_MODELS
    print(f"generating mechanism docs for {len(models)} model(s) -> {OUT_DIR}/")
    main(models)
