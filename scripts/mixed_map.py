"""M0 (mixed-model SV replication): calibration + verified-map capture.

Loads the accurate mixed add/sub model, verifies per-class ADD/SUB/NEG accuracy
(the standing positive control for every downstream mixed study), then reuses the
shared library builder ``capture_model_map`` (pulls the published
behavior.json + maths.json verified maps and groups nodes by algorithmic role at
the layer-general positions) and merges the positive-control accuracy in.

Evidence-integrity: all numbers here are computed and written to
results/study-mixed-map/results.json; the study note cites the JSON. The map
schema is now the shared ``results/maps/<model>.json`` schema (see
``quanta_maths.maths_diagram.build_model_map``).

Run: python scripts/mixed_map.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from quanta_maths import capture_model_map, load_maths_model_from_hf
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_edge_patch import answer_positions
from quanta_maths.maths_probe import last_layer

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
HF_REPO = "PhilipQuirke/VerifiedArithmetic"
OUT_DIR = "results/study-mixed-map"


def make_class_question(cfg, rng, op, cls):
    """Return (a, b) forcing a given question class. ADD: op=+. SUB: op=-, a>=b.
    NEG: op=-, a<b."""
    lim = 10 ** cfg.n_digits
    if op == MathsToken.PLUS:  # ADD: keep answer within range
        return int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
    a, b = int(rng.integers(0, lim)), int(rng.integers(0, lim))
    if cls == "SUB" and a < b:
        a, b = b, a
    if cls == "NEG":
        if a == b:
            b = (b + 1) % lim
        if a > b:
            a, b = b, a
    return a, b


def class_accuracy(model, cfg, op, cls, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ap = answer_positions(cfg)
    hit = 0
    for _ in range(n):
        a, b = make_class_question(cfg, rng, op, cls)
        q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(cfg, q, 0, a, b, op)
        q = q[0]
        with torch.no_grad():
            lg = model(q.unsqueeze(0))
        pred = lg[0, [p - 1 for p in ap]].argmax(-1)
        hit += int((pred == q[ap]).all())
    return hit / n


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")

    print("=== M0 [1/2] per-class accuracy (positive control) ===")
    acc = {
        "ADD": class_accuracy(model, cfg, MathsToken.PLUS, "ADD"),
        "SUB": class_accuracy(model, cfg, MathsToken.MINUS, "SUB"),
        "NEG": class_accuracy(model, cfg, MathsToken.MINUS, "NEG"),
    }
    for k, v in acc.items():
        print(f"  {k}: {v:.4f}")

    print("=== M0 [2/2] verified-map registry (shared capture_model_map) ===")
    m = capture_model_map(MODEL, hf_repo=HF_REPO, cfg=cfg)
    for task in sorted(m["roles"], key=lambda t: -len(m["roles"][t])):
        locs = [e["loc"] for e in m["roles"][task]]
        print(f"  {task:6} x{len(locs):2}: {locs}")
    print(f"  combiners (L{last_layer(cfg)} answer MLPs) x{len(m['combiners'])}: "
          f"{[c['loc'] for c in m['combiners']]}")
    print(f"  positions: {m['positions']}")

    # Merge the positive control into the shared map schema (accuracy needs a
    # model load, so it is not part of the map-only library builder).
    result = {
        "model": m["model"], "config": m["config"],
        "positive_control_accuracy": acc,
        "n_map_nodes": m["n_map_nodes"], "roles": m["roles"],
        "combiners": m["combiners"], "positions": m["positions"],
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
