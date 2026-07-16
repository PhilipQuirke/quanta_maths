"""M0 (mixed-model SV replication): calibration + verified-map capture.

Loads the accurate mixed add/sub model, verifies per-class ADD/SUB/NEG accuracy
(the standing positive control for every downstream mixed study), pulls the
published behavior.json + maths.json verified maps, and emits a node registry
grouped by algorithmic role (writers / consumers / combiners) at the correct
layer-general positions for the 3-layer architecture.

Evidence-integrity: all numbers here are computed and written to
results/study-mixed-map/results.json; the study note cites the JSON.

Run: python scripts/mixed_map.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_edge_patch import answer_positions
from quanta_maths.maths_probe import first_layer, last_layer

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
HF_REPO = "PhilipQuirke/VerifiedArithmetic"
OUT_DIR = "results/study-mixed-map"


def node_loc(node) -> str:
    kind = "H" if node["is_head"] else "M"
    return f"P{node['position']}L{node['layer']}{kind}{node['num']}"


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


def build_registry(cfg):
    from QuantaMechInterp.model_train_json import download_huggingface_json
    maths = download_huggingface_json(HF_REPO, f"{MODEL}_maths.json")
    behav = download_huggingface_json(HF_REPO, f"{MODEL}_behavior.json")

    # index behavior tags by (position, layer, is_head, num)
    btags = {}
    for node in behav:
        key = (node["position"], node["layer"], node["is_head"], node["num"])
        btags[key] = node.get("tags", [])

    def behav_of(node, prefix):
        key = (node["position"], node["layer"], node["is_head"], node["num"])
        return [t for t in btags.get(key, []) if t.startswith(prefix)]

    roles = collections.defaultdict(list)
    for node in maths:
        key = (node["position"], node["layer"], node["is_head"], node["num"])
        for t in node.get("tags", []):
            if not t.startswith("Algo:"):
                continue
            body = t.split(":", 1)[1]
            task = body.split(".")[-1] if "." in body else body
            entry = {
                "loc": node_loc(node),
                "position": node["position"], "layer": node["layer"],
                "is_head": node["is_head"], "num": node["num"],
                "algo": body,
                "impact": behav_of(node, "Impact"),
                "fail": behav_of(node, "Fail%"),
                "attn": behav_of(node, "Attn"),
            }
            roles[task].append(entry)

    # Combiner candidates: last-layer answer-position MLP nodes (high Fail%).
    ll = last_layer(cfg)
    answer_produce_pos = {int(cfg.an_to_position_name(k)[1:]) - 1: k
                          for k in range(cfg.n_digits + 2)}
    combiners = []
    for node in behav:
        if node["is_head"] or node["layer"] != ll:
            continue
        if node["position"] in answer_produce_pos:
            fail = [t for t in node.get("tags", []) if t.startswith("Fail%")]
            combiners.append({
                "loc": node_loc(node),
                "position": node["position"], "layer": node["layer"],
                "produces_A": answer_produce_pos[node["position"]],
                "fail": fail,
                "impact": [t for t in node.get("tags", []) if t.startswith("Impact")],
            })
    return roles, combiners, len(maths)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")

    print("=== M0 [1/3] per-class accuracy (positive control) ===")
    acc = {
        "ADD": class_accuracy(model, cfg, MathsToken.PLUS, "ADD"),
        "SUB": class_accuracy(model, cfg, MathsToken.MINUS, "SUB"),
        "NEG": class_accuracy(model, cfg, MathsToken.MINUS, "NEG"),
    }
    for k, v in acc.items():
        print(f"  {k}: {v:.4f}")

    print("=== M0 [2/3] verified-map registry ===")
    roles, combiners, n_nodes = build_registry(cfg)
    for task in sorted(roles, key=lambda t: -len(roles[t])):
        locs = [e["loc"] for e in roles[task]]
        print(f"  {task:6} x{len(locs):2}: {locs}")
    print(f"  combiners (L{last_layer(cfg)} answer MLPs) x{len(combiners)}: "
          f"{[c['loc'] for c in combiners]}")

    print("=== M0 [3/3] key positions ===")
    pos = {
        "OPR": cfg.op_position_name(),
        "eq": f"P{2*cfg.n_digits+1}",
        "SGN": cfg.an_to_position_name(cfg.n_digits + 1),
        "answer_digits": [cfg.an_to_position_name(k) for k in range(cfg.n_digits, -1, -1)],
        "first_layer": first_layer(cfg), "last_layer": last_layer(cfg),
    }
    print(f"  {pos}")

    result = {
        "model": MODEL,
        "config": {"n_digits": cfg.n_digits, "n_layers": cfg.n_layers,
                   "n_heads": cfg.n_heads, "n_ctx": cfg.n_ctx,
                   "perc_add": cfg.perc_add, "perc_sub": cfg.perc_sub},
        "positive_control_accuracy": acc,
        "n_map_nodes": n_nodes,
        "roles": {t: roles[t] for t in roles},
        "combiners": combiners,
        "positions": pos,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
