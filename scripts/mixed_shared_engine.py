"""Entry 2(b): Mixed-model SLT-sited A7-vs-C2 shared-engine test.

CE21 showed the operator is broadcast-decodable but NOT additively steerable at
the L2 combiner input (consumed upstream). The map says the SLT head (L1 H1)
selects the S/M/N outputs, so the operator control must act at/before L1. Here we
steer at the L1 selector stage and ask whether a compact (rank-1) operator
direction flips the SA<->MD readout for the SAME operands (A7: shared engine under
low-rank control) or whether that fails/garbles while the full selector-stage
residual flips it (C2 / not-low-rank).

Arms (matched a>=b; ADD a+b vs SUB a-b; middle digit k=2; selector layer L1):
  1 rank1_steer   -- add (mean_add-mean_sub) of resid_post(L1)[pp] to a SUB run.
  2 slt_head      -- replace SLT head L1H1 z[pp] with its ADD value on a SUB run.
  3 full_l1       -- patch the whole resid_post(L1)[pp] ADD->SUB (positive control).
Success counts a flip only if A_k becomes the correct ADD digit (valid, not garbage).

Run: PYTHONPATH=. python scripts/mixed_shared_engine.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_probe import last_layer
from quanta_maths.maths_edge_patch import answer_positions, consuming_pos
from mixed_sv import to_q

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
OUT_DIR = "results/study-mixed-shared-engine"
SLT_LAYER, SLT_HEAD = 1, 1  # map: SLT at L1 H1


def digit_of(v, k, nd):
    return int(str(abs(v)).zfill(nd + 1)[::-1][k])


def matched_pair(cfg, rng):
    lim = 10 ** cfg.n_digits
    a, b = int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
    if a < b:
        a, b = b, a
    return a, b


def resid_post_l1(model, cfg, q, pp):
    h = f"blocks.{SLT_LAYER}.hook_resid_post"
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == h)
    return c[h][0, pp, :].clone()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")
    k = 2
    pp = consuming_pos(cfg, k)
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - k
    nd = cfg.n_digits
    hL1 = f"blocks.{SLT_LAYER}.hook_resid_post"
    zL1 = f"blocks.{SLT_LAYER}.attn.hook_z"

    # op_dir: rank-1 operator direction at the L1 selector, matched pairs
    rng = np.random.default_rng(0)
    adds, subs = [], []
    for _ in range(200):
        a, b = matched_pair(cfg, rng)
        adds.append(resid_post_l1(model, cfg, to_q(cfg, a, b, "ADD"), pp).numpy())
        subs.append(resid_post_l1(model, cfg, to_q(cfg, a, b, "SUB"), pp).numpy())
    op_dir = torch.tensor(np.mean(adds, 0) - np.mean(subs, 0), dtype=torch.float32)  # sub->add

    def read_Ak(q, fwd_hooks):
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=fwd_hooks)
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    def clean_Ak(q):
        with torch.no_grad():
            lg = model(q.unsqueeze(0))
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    arms = {"rank1_steer": [], "slt_head": [], "full_l1": []}
    valid = {"rank1_steer": [], "slt_head": [], "full_l1": []}
    rng = np.random.default_rng(7)
    n = 60
    for _ in range(n):
        a, b = matched_pair(cfg, rng)
        add_dig = digit_of(a + b, k, nd)
        sub_dig = digit_of(a - b, k, nd)
        if add_dig == sub_dig:
            continue
        sub_q = to_q(cfg, a, b, "SUB")
        add_q = to_q(cfg, a, b, "ADD")
        base = clean_Ak(sub_q)  # == sub_dig
        # arm 1: rank-1 steer sub -> add
        def h_steer(act, hook):
            act[:, pp, :] = act[:, pp, :] + op_dir.to(act.dtype); return act
        r1 = read_Ak(sub_q, [(hL1, h_steer)])
        # arm 2: SLT head z patch add -> sub
        with torch.no_grad():
            _, ca = model.run_with_cache(add_q.unsqueeze(0), names_filter=lambda nm: nm == zL1)
        z_add = ca[zL1][0, pp, SLT_HEAD, :].clone()
        def h_slt(act, hook):
            act[:, pp, SLT_HEAD, :] = z_add.to(act.dtype); return act
        r2 = read_Ak(sub_q, [(zL1, h_slt)])
        # arm 3: full L1 resid patch add -> sub (positive control)
        r_add = resid_post_l1(model, cfg, add_q, pp)
        r_sub = resid_post_l1(model, cfg, sub_q, pp)
        delta = r_add - r_sub
        def h_full(act, hook):
            act[:, pp, :] = act[:, pp, :] + delta.to(act.dtype); return act
        r3 = read_Ak(sub_q, [(hL1, h_full)])
        for name, r in [("rank1_steer", r1), ("slt_head", r2), ("full_l1", r3)]:
            arms[name].append(float(r == add_dig))   # flipped to the correct ADD digit
            valid[name].append(float(r != base))     # changed at all (validity/diagnostic)

    res = {
        "model": MODEL, "digit_k": k, "consuming_pos": pp,
        "selector": f"L{SLT_LAYER}H{SLT_HEAD} (SLT)", "n_discriminating": len(arms["full_l1"]),
        "flip_to_add_digit_rate": {a: float(np.mean(v)) for a, v in arms.items()},
        "changed_rate": {a: float(np.mean(v)) for a, v in valid.items()},
        "op_dir_norm": float(op_dir.norm()),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2))
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
