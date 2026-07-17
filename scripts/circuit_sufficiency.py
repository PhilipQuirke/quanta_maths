"""Circuit sufficiency: keep ONLY the map-useful nodes and destroy the
position-specific complement (resample- / mean-ablation from same-class inputs);
does the model retain accuracy?

The reusable core is in `quanta_maths.maths_sufficiency` (model-general, runnable
across the zoo). This script just drives it and writes results.

Run: PYTHONPATH=. python scripts/circuit_sufficiency.py [model_name ...]
     (default: the best-understood addition model, then the 6-digit mixed model)
"""
from __future__ import annotations

import json
import os
import sys

from quanta_maths import load_maths_model_from_hf, circuit_sufficiency

OUT_DIR = "results/study-circuit-sufficiency"
DEFAULTS = ["add_d5_l2_h3_t15K_s372001", "ins1_mix_d6_l3_h4_t40K_s372001"]


def main():
    models = sys.argv[1:] or DEFAULTS
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in models:
        model, cfg = load_maths_model_from_hf(name, device="cpu")
        out = circuit_sufficiency(model, cfg, name)
        print(f"=== {name}: keep {out['keep_heads']}/{out['total_heads']} heads, "
              f"{out['keep_mlps']}/{out['total_mlps']} MLPs ===")
        for cls, r in out["classes"].items():
            print(f"  {cls}: baseline={r['baseline']:.3f} | keep-useful "
                  f"mean={r['keep_useful_mean']:.3f} resample={r['keep_useful_resample']:.3f} "
                  f"| keep-random={r['keep_random_mean']:.3f}")
        with open(os.path.join(OUT_DIR, f"results_{name}.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"  wrote {OUT_DIR}/results_{name}.json")


if __name__ == "__main__":
    main()
