"""M4/M5/M6: the mixed-only mechanism experiments the model's algorithm forces.

M4  OPR  -- how is the operator (+/-) used? A7 (low-rank control selecting the
            readout of shared machinery) vs C2 (routing between separate circuits).
M5  SGN  -- how is the answer sign computed? Hypothesis: the D>=D' comparison
            resolves at '=' into the sign, delivered to the sign-position combiner
            (the CE15 leading-digit analog, now producing SGN).
M6  A7 vs C2 -- on the polysemantic SA/MD/ND heads the map shares, measure the
            per-operation readout subspace overlap (principal angles).

Evidence integrity: all numbers written to results/study-mixed-opr-sgn/results.json.
Run: PYTHONPATH=. python scripts/mixed_opr_sgn.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf, make_untrained_control
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_probe import (
    first_layer, last_layer, probe_accuracy_with_null,
    class_mean_subspace, principal_angles_deg,
)
from quanta_maths.maths_edge_patch import answer_positions, consuming_pos

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
OUT_DIR = "results/study-mixed-opr-sgn"
OP = {"ADD": MathsToken.PLUS, "SUB": MathsToken.MINUS, "NEG": MathsToken.MINUS}


def to_q(cfg, a, b, op):
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, op)
    return q[0]


def read_answer(model, cfg, q):
    ap = answer_positions(cfg)
    with torch.no_grad():
        lg = model(q.unsqueeze(0))
    return lg[0, [p - 1 for p in ap]].argmax(-1)


# ===========================================================================
# M6: shared-engine readout overlap on the polysemantic SA/MD/ND heads
# ===========================================================================

def m6_shared_engine(model, cfg, rng, n_q=500, k=2):
    """At the answer-producing position of digit k (first layer, where the map's
    SA/MD/ND heads write), collect activations labelled by the emitted answer digit
    for ADD / SUB / NEG, and measure principal angles between the per-operation
    digit-readout subspaces. C2 -> near-orthogonal (~90); A7 -> overlapping (small)."""
    layer = first_layer(cfg)
    pp = consuming_pos(cfg, k)
    hook = f"blocks.{layer}.hook_resid_post"
    lim = 10 ** cfg.n_digits
    acts = {"ADD": [], "SUB": [], "NEG": []}
    labs = {"ADD": [], "SUB": [], "NEG": []}
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - k
    for _ in range(n_q):
        for cls in ("ADD", "SUB", "NEG"):
            if cls == "ADD":
                a, b = int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
            else:
                a, b = int(rng.integers(0, lim)), int(rng.integers(0, lim))
                if cls == "SUB" and a < b: a, b = b, a
                if cls == "NEG":
                    if a == b: b = (b + 1) % lim
                    if a > b: a, b = b, a
            q = to_q(cfg, a, b, OP[cls])
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
            acts[cls].append(c[hook][0, pp, :].numpy())
            labs[cls].append(int(read_answer(model, cfg, q)[idx]))
    subs = {cls: class_mean_subspace(np.asarray(acts[cls]), np.asarray(labs[cls]))
            for cls in acts}
    out = {}
    for c1, c2 in [("ADD", "SUB"), ("ADD", "NEG"), ("SUB", "NEG")]:
        ang = principal_angles_deg(subs[c1], subs[c2])
        out[f"{c1}_{c2}"] = {"mean_angle_deg": float(np.mean(ang)) if ang.size else float("nan"),
                             "min_angle_deg": float(np.min(ang)) if ang.size else float("nan"),
                             "n_angles": int(ang.size)}
    return out


# ===========================================================================
# M4: OPR as a low-rank control direction
# ===========================================================================

def m4_opr(model, cfg, rng, n_dir=300, n_test=60, k=2):
    """(1) operator decodable at the combiner input? (2) is a RANK-1 operator
    direction (add-mean - sub-mean at the combiner input) causally sufficient to
    flip the answer-digit readout add<->sub (A7) without a full re-route?"""
    ll = last_layer(cfg)
    pp = consuming_pos(cfg, k)
    hook = f"blocks.{ll}.hook_resid_mid"
    ap = answer_positions(cfg); idx = len(ap) - 1 - k
    lim = 10 ** cfg.n_digits

    # matched (a,b) with a>=b so both a+b and a-b are valid; collect combiner-input
    # activations for ADD vs SUB and the emitted digit each would produce.
    add_acts, sub_acts = [], []
    for _ in range(n_dir):
        a, b = int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
        if a < b: a, b = b, a
        qa, qs = to_q(cfg, a, b, MathsToken.PLUS), to_q(cfg, a, b, MathsToken.MINUS)
        with torch.no_grad():
            _, ca = model.run_with_cache(qa.unsqueeze(0), names_filter=lambda nm: nm == hook)
            _, cs = model.run_with_cache(qs.unsqueeze(0), names_filter=lambda nm: nm == hook)
        add_acts.append(ca[hook][0, pp, :].numpy())
        sub_acts.append(cs[hook][0, pp, :].numpy())
    add_acts = np.asarray(add_acts); sub_acts = np.asarray(sub_acts)
    # aligned X/y: operator label matches each row's source
    X = np.concatenate([add_acts, sub_acts], 0)
    y = np.concatenate([np.ones(len(add_acts), int), np.zeros(len(sub_acts), int)])
    dec = probe_accuracy_with_null(X, y, rng, n_perm=30)

    add_mean = add_acts.mean(0)
    sub_mean = sub_acts.mean(0)
    op_dir = torch.tensor(add_mean - sub_mean, dtype=torch.float32)  # rank-1, sub->add
    op_dir_norm = float(np.linalg.norm(add_mean - sub_mean))
    act_norm = float(np.linalg.norm(sub_acts.mean(0)))

    # causal rank-1 steer: on fresh SUB questions, add op_dir at the combiner input;
    # does A_k move toward the ADD digit (a+b) and away from the SUB digit (a-b)?
    def steer_pred(q):
        def hk(act, hook):
            act[:, pp, :] = act[:, pp, :] + op_dir.to(act.dtype)
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[(hook, hk)])
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    to_add, digits_kept = [], []
    for _ in range(n_test):
        a, b = int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
        if a < b: a, b = b, a
        add_dig = int(str((a + b)).zfill(cfg.n_digits + 1)[::-1][k])
        sub_dig = int(str(abs(a - b)).zfill(cfg.n_digits)[::-1][k])
        qs = to_q(cfg, a, b, MathsToken.MINUS)
        base = int(read_answer(model, cfg, qs)[idx])
        steered = steer_pred(qs)
        if add_dig != sub_dig:
            to_add.append(float(steered == add_dig))
            digits_kept.append(float(steered != base))
    return {
        "operator_decode_acc": dec["observed_acc"], "operator_decode_null_p": dec["null_p"],
        "op_dir_norm": op_dir_norm, "site_act_norm": act_norm,
        "rank1_steer_to_add_rate": float(np.mean(to_add)) if to_add else float("nan"),
        "rank1_steer_changed_rate": float(np.mean(digits_kept)) if digits_kept else float("nan"),
        "n_discriminating": len(to_add),
    }


# ===========================================================================
# M5: SGN = magnitude comparison -> sign, delivered to the sign combiner
# ===========================================================================

def m5_sgn(model, cfg, rng, n_pairs=40):
    """(1) positive control: crossing the D>=D' <-> D<D' boundary at the deciding
    digit flips SGN. (2) sign decodable + binary at '='. (3) full-resid patch at the
    sign-producing position ('=') flips SGN comparison-specifically (deciding null)."""
    nd = cfg.n_digits
    sgn_k = nd + 1                      # SGN == A_{n+1}
    sign_pos = consuming_pos(cfg, sgn_k)  # produces SGN == '=' position
    ap = answer_positions(cfg)
    sgn_idx = 0                         # SGN is the top answer token
    ll = last_layer(cfg)

    def sgn_token(q):
        return int(read_answer(model, cfg, q)[sgn_idx])

    # (1) boundary positive control: base equal-high-digit, toggle the top digit
    pc = []
    for _ in range(n_pairs):
        mid = int(rng.integers(0, 10 ** (nd - 1)))
        top = int(rng.integers(1, 9))
        a = top * 10 ** (nd - 1) + mid          # D
        b_lt = (top - 1) * 10 ** (nd - 1) + mid  # D' < D  -> SUB (SGN +)
        b_gt = (top + 1) * 10 ** (nd - 1) + mid  # D' > D  -> NEG (SGN -)
        s_pos = sgn_token(to_q(cfg, a, b_lt, MathsToken.MINUS))
        s_neg = sgn_token(to_q(cfg, a, b_gt, MathsToken.MINUS))
        pc.append(float(s_pos != s_neg))

    # (2) sign decodable + binary at '=' (last layer)
    hook = f"blocks.{ll}.hook_resid_post"
    acts, labs = [], []
    for _ in range(300):
        a, b = int(rng.integers(0, 10 ** nd)), int(rng.integers(0, 10 ** nd))
        if a == b: b = (b + 1) % (10 ** nd)
        q = to_q(cfg, a, b, MathsToken.MINUS)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
        acts.append(c[hook][0, sign_pos, :].numpy())
        labs.append(int(a < b))  # 1 == negative answer
    dec = probe_accuracy_with_null(np.asarray(acts), np.asarray(labs), rng, n_perm=30)

    # (3) full-resid patch at the sign-producing position: does swapping the '='
    # combiner input flip SGN comparison-specifically?
    def build(cross):
        mid = int(rng.integers(0, 10 ** (nd - 1)))
        top = int(rng.integers(1, 9))
        a = top * 10 ** (nd - 1) + mid
        b = ((top + 1) if cross else (top - 1)) * 10 ** (nd - 1) + mid
        return a, b
    a_neg, b_neg = build(cross=True)    # source: D<D' (SGN -)
    a_pos, b_pos = build(cross=False)   # target: D>=D' (SGN +)
    src_s = sgn_token(to_q(cfg, a_neg, b_neg, MathsToken.MINUS))
    tgt_s = sgn_token(to_q(cfg, a_pos, b_pos, MathsToken.MINUS))

    def patch_resid(sq, tq):
        h = f"blocks.{ll}.hook_resid_mid"
        with torch.no_grad():
            _, sc = model.run_with_cache(sq.unsqueeze(0), names_filter=lambda nm: nm == h)
            _, tc = model.run_with_cache(tq.unsqueeze(0), names_filter=lambda nm: nm == h)
        delta = sc[h][0, sign_pos, :] - tc[h][0, sign_pos, :]

        def hk(act, hook):
            act[:, sign_pos, :] = act[:, sign_pos, :] + delta
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(tq.unsqueeze(0), fwd_hooks=[(h, hk)])
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[sgn_idx])

    flips, nulls = [], []
    for _ in range(n_pairs):
        an, bn = build(cross=True); ap_, bp_ = build(cross=False)
        sq, tq = to_q(cfg, an, bn, MathsToken.MINUS), to_q(cfg, ap_, bp_, MathsToken.MINUS)
        clean = sgn_token(tq)
        flips.append(float(patch_resid(sq, tq) != clean))
        # deciding-matched null: both D>=D' (same sign), different filler
        an2, bn2 = build(cross=False); ap2, bp2 = build(cross=False)
        sq2, tq2 = to_q(cfg, an2, bn2, MathsToken.MINUS), to_q(cfg, ap2, bp2, MathsToken.MINUS)
        nclean = sgn_token(tq2)
        nulls.append(float(patch_resid(sq2, tq2) != nclean))

    return {
        "boundary_flips_sgn_rate": float(np.mean(pc)),
        "sign_decode_acc": dec["observed_acc"], "sign_decode_null_p": dec["null_p"],
        "sign_edge_flip_rate": float(np.mean(flips)), "sign_edge_null_rate": float(np.mean(nulls)),
        "stimulus_valid": bool(src_s != tgt_s),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")

    print("=== M6 shared-engine readout overlap (A7 vs C2) ===")
    m6 = m6_shared_engine(model, cfg, np.random.default_rng(0))
    for pair, v in m6.items():
        print(f"  {pair}: mean angle {v['mean_angle_deg']:.1f} deg (min {v['min_angle_deg']:.1f}, n={v['n_angles']})")
    # untrained control for overlap reference
    ctrl = make_untrained_control(cfg)
    m6c = m6_shared_engine(ctrl, cfg, np.random.default_rng(0), n_q=200)

    print("=== M4 OPR low-rank control ===")
    m4 = m4_opr(model, cfg, np.random.default_rng(1))
    print(f"  operator decode acc={m4['operator_decode_acc']:.3f} (null p={m4['operator_decode_null_p']:.3f}) "
          f"| op_dir_norm={m4['op_dir_norm']:.2f} vs site_norm={m4['site_act_norm']:.2f}")
    print(f"  rank-1 steer sub->add: to_add={m4['rank1_steer_to_add_rate']:.2f} "
          f"changed={m4['rank1_steer_changed_rate']:.2f} (n={m4['n_discriminating']})")

    print("=== M5 SGN = comparison -> sign ===")
    m5 = m5_sgn(model, cfg, np.random.default_rng(2))
    print(f"  boundary flips SGN={m5['boundary_flips_sgn_rate']:.2f}")
    print(f"  sign decode acc={m5['sign_decode_acc']:.3f} (null p={m5['sign_decode_null_p']:.3f})")
    print(f"  sign edge flip={m5['sign_edge_flip_rate']:.2f} null={m5['sign_edge_null_rate']:.2f} "
          f"valid={m5['stimulus_valid']}")

    result = {"model": MODEL, "M6_shared_engine": m6, "M6_untrained_control": m6c,
              "M4_opr": m4, "M5_sgn": m5}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
