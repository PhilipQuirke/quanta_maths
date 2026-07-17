"""Stage 2 of the visualization pipeline: generate per-model mechanism docs
(Mermaid diagrams + text) from the maximal map JSONs into
``results/maps/<model>_mechanism.md``.

The doc is generated *from the map JSON* (``results/maps/<model>.json``, stage 1
= ``scripts/gen_model_maps.py``), which is the single source of truth. If a
model's JSON is missing this script captures it from HuggingFace and saves it
first, so the two stages can also be run as one. Doc generation itself is
offline once the JSON exists.

Model-general: pass any list of maths model names (built for the ~40-model zoo).

Run:  PYTHONPATH=. python scripts/gen_mechanism_docs.py [model_name ...]
      (defaults to the accurate model set if none given)
"""
from __future__ import annotations

import json
import os
import sys

from quanta_maths import build_mechanism_markdown, capture_model_map
from quanta_maths.maths_hf_update import ordered_analysis_models

MAP_DIR = "results/maps"


def load_or_capture(name):
    """Load ``results/maps/<name>.json`` if present, else capture it from HF and
    save it (keeping the JSON as the source of truth for the doc)."""
    path = os.path.join(MAP_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    model_map = capture_model_map(name)
    os.makedirs(MAP_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(model_map, f, indent=2)
    print(f"  captured {path} ({model_map['n_map_nodes']} map nodes)")
    return model_map


def main(models):
    os.makedirs(MAP_DIR, exist_ok=True)
    for name in models:
        try:
            model_map = load_or_capture(name)
            doc = build_mechanism_markdown(model_map)
        except Exception as e:  # a model with no published map, network, etc.
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            continue
        path = os.path.join(MAP_DIR, f"{name}_mechanism.md")
        with open(path, "w") as f:
            f.write(doc)
        print(f"  wrote {path}")


if __name__ == "__main__":
    models = sys.argv[1:] or ordered_analysis_models()
    print(f"generating mechanism docs for {len(models)} model(s) -> {MAP_DIR}/")
    main(models)
