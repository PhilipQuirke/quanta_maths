"""Entry 2(i): Mixed-model >=2-depth carry/borrow delivery sweep (ADD/SUB/NEG).

Extends CE20's single-step (k=1) delivery result to cascade depths k=2,3,4: which
delivery arm carries the resolved cascade into the L2 combiner, per class,
carry/borrow-specifically (deciding-matched null)?

The reusable, model-general core now lives in ``quanta_maths.maths_cascade``
(``make_cascade_operands`` + ``combiner_delivery_sweep``) so the sweep can be run
across the model zoo (delivery may differ by model). This script just drives it on
the mixed model + an untrained control and writes results.

Run: PYTHONPATH=. python scripts/mixed_delivery_depth.py
"""
from __future__ import annotations

import json
import os

from quanta_maths import (load_maths_model_from_hf, make_untrained_control,
                          combiner_delivery_sweep, combiner_delivery_flip)

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
OUT_DIR = "results/study-mixed-delivery-depth"
CLASSES = ["ADD", "SUB", "NEG"]
DEPTHS = [2, 3, 4]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")

    print("=== trained: combiner-delivery sweep (classes x depths x arms) ===")
    trained = combiner_delivery_sweep(model, cfg, classes=CLASSES, depths=DEPTHS)
    for cls in CLASSES:
        for k in DEPTHS:
            v = trained[cls][k]
            print(f"  {cls} depth {k} valid={v['full_resid']['valid']}: "
                  f"full={v['full_resid']['flip']:.2f}/n{v['full_resid']['null']:.2f} "
                  f"resid_pre={v['resid_pre']['flip']:.2f}/n{v['resid_pre']['null']:.2f} "
                  f"attn={v['lastlayer_attn']['flip']:.2f}/n{v['lastlayer_attn']['null']:.2f}")

    print("=== untrained control: lastlayer_attn (discriminating arm) must be ~0 ===")
    ctrl = make_untrained_control(cfg)
    control = {}
    for cls in CLASSES:
        r = combiner_delivery_flip(ctrl, cfg, cls, 2, "lastlayer_attn")
        control[cls] = {"lastlayer_attn_flip_d2": r["flip"]["rate"],
                        "null": r["null"]["rate"], "valid": r["stimulus_valid"]}
        print(f"  {cls} d2 lastlayer_attn flip={r['flip']['rate']:.2f} valid={r['stimulus_valid']}")

    result = {"model": MODEL, "depths": DEPTHS, "trained": trained,
              "untrained_control": control}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
