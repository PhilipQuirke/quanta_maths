"""Stage 1 of the visualization pipeline: batch-write the *maximal* per-model map
JSON to ``results/maps/<model>.json``.

Each map is the single source of truth the mechanism diagrams are generated from
(stage 2 is ``scripts/gen_mechanism_docs.py``). Map capture is map-only (no model
forward pass) so it is cheap to batch across the ~40-model zoo, but it does need
HuggingFace access to download each model's verified ``*_maths.json`` /
``*_behavior.json``.

Not yet captured by this map-only pass (see the map-completeness item in
docs/maths-next-steps.md): per-class positive-control accuracy (needs a model
load; the study scripts such as ``scripts/mixed_map.py`` add it), model
provenance (``init_from``), and the map-absent ``SV/MV/NV`` /
``STC/MTC/NTC`` role tags.

Run:  PYTHONPATH=. python scripts/gen_model_maps.py [model_name ...]
      (defaults to the accurate model set if none given)
"""
from __future__ import annotations

import json
import os
import sys

from quanta_maths import capture_model_map
from quanta_maths.maths_hf_update import ordered_analysis_models

OUT_DIR = "results/maps"


def main(models):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in models:
        try:
            model_map = capture_model_map(name)
        except Exception as e:  # a model with no published map, network, etc.
            print(f"  SKIP {name}: {type(e).__name__}: {e}")
            continue
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(model_map, f, indent=2)
        print(f"  wrote {path} ({model_map['n_map_nodes']} map nodes)")


if __name__ == "__main__":
    models = sys.argv[1:] or ordered_analysis_models()
    print(f"writing maximal map JSONs for {len(models)} model(s) -> {OUT_DIR}/")
    main(models)
