"""Combiner vs conduit at the U-flip transmitters (study-combiner-vs-conduit.md).

The U-resolution study located MLP-heavy nodes that TRANSMIT the tri-state
U-resolution flip but couldn't tell a COMBINER (outputs carry_out(n), reads
digit-n's sum-class AND the lower carry) from a CONDUIT (relays carry_in).

Primary discriminator (amendment A-1), NODE-LEVEL, endpoint-independent:
  A combiner's OUTPUT is carry_out(n). For a DEFINITE digit n, carry_out is
  independent of carry_in, so a combiner's activation is INVARIANT to a carry_in
  toggle; but in the U regime (sum==9) it MUST vary (U depends on carry_in).
  A conduit relays carry_in, so its activation VARIES in BOTH regimes.
  Measured at the node -> a downstream recompute cannot mask it.

Signatures:
  combiner : definite-regime activation-diff LOW, U-regime activation-diff HIGH
  conduit  : activation-diff HIGH in both regimes
  inert    : activation-diff LOW in both regimes

Positive controls (A-2): planted CONDUIT (resid at the lower-carry position,
must vary in both regimes) and INERT (SA operand-fetch head, must be ~flat).

CPU-only. Run:
    PYTHONPATH=. python3 scripts/combiner_vs_conduit.py control
    PYTHONPATH=. python3 scripts/combiner_vs_conduit.py models
    PYTHONPATH=. python3 scripts/combiner_vs_conduit.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    _digits_to_int, operand_attention,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-combiner-vs-conduit")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260715)

# candidate U-flip transmitters (CE4) + CE3 make-carry heads (as datapoints)
CANDIDATES = {
    "add_d5_l2_h3_t15K_s372001": {
        "u_transmitters": [("mlp", 10, 0, None), ("mlp", 14, 1, None)],
        "make_carry": [("z", 13, 0, 0), ("z", 14, 0, 0), ("z", 15, 0, 0)],
        "sa_head": ("z", 14, 0, 1),          # co-located base-add head (flat: 0.0)
        "inert_ref": ("z", 0, 0, 0),          # early head, guaranteed carry-inert
    },
    "add_d6_l2_h3_t20K_s173289": {
        "u_transmitters": [("z", 11, 0, 2), ("mlp", 11, 0, None), ("mlp", 16, 1, None)],
        "make_carry": [("z", 15, 0, 2), ("z", 16, 0, 2)],
        "sa_head": ("z", 20, 0, 1),
        "inert_ref": ("z", 0, 0, 0),          # early head, guaranteed carry-inert
    },
}


def hook_name(kind, L):
    return f"blocks.{L}.attn.hook_z" if kind == "z" else f"blocks.{L}.hook_mlp_out"


def get_activation(model, cfg, q, kind, pos, L, head):
    with torch.no_grad():
        _, cache = model.run_with_cache(q.unsqueeze(0))
    if kind == "z":
        return cache[f"blocks.{L}.attn.hook_z"][0, pos, head, :].numpy()
    return cache[f"blocks.{L}.hook_mlp_out"][0, pos, :].numpy()


def build_pair_fixed_digit_n(cfg, n, digit_n_class):
    """Return two questions IDENTICAL at digit n (same operands) but with
    opposite lower carry (digit n-1 sum>=10 vs <10). Amendment A-1/A-3: digit n
    operands are matched within the pair, so only the lower carry (+ digit n-1
    operands) differ. Returns (q_carry, q_nocarry, a, b)."""
    nd = cfg.n_digits
    idx = nd - 1 - n
    if digit_n_class == "U":
        a = int(RNG.integers(0, 10)); b = 9 - a
    elif digit_n_class == "lo":
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b <= 8: break
    else:  # hi
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b >= 10: break

    def build(lower_carry):
        d1 = [0] * nd; d2 = [0] * nd
        d1[idx] = a; d2[idx] = b
        ikm = nd - 1 - (n - 1)
        if lower_carry:
            while True:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                if x + y >= 10: break
        else:
            x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
        d1[ikm] = x; d2[ikm] = y
        for k in range(n - 1):
            ik = nd - 1 - k
            u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
            d1[ik] = u; d2[ik] = v
        return _digits_to_int(d1), _digits_to_int(d2)

    ca, cb = build(True)
    na_, nb_ = build(False)
    return make_q(cfg, ca, cb), make_q(cfg, na_, nb_), a, b


def activation_diff(model, cfg, n, kind, pos, L, head, digit_n_class, n_pairs=60):
    """Mean cosine distance between the node's activation under carry_in=1 vs 0,
    with digit n held at digit_n_class. Also normalized-L2."""
    cos_ds, l2_ds = [], []
    for _ in range(n_pairs):
        qc, qn, a, b = build_pair_fixed_digit_n(cfg, n, digit_n_class)
        ac = get_activation(model, cfg, qc, kind, pos, L, head)
        an = get_activation(model, cfg, qn, kind, pos, L, head)
        denom = (np.linalg.norm(ac) * np.linalg.norm(an)) + 1e-9
        cos_ds.append(1.0 - float(np.dot(ac, an) / denom))
        scale = (np.linalg.norm(ac) + np.linalg.norm(an)) / 2 + 1e-9
        l2_ds.append(float(np.linalg.norm(ac - an) / scale))
    return {"cos_dist": float(np.mean(cos_ds)), "l2_norm": float(np.mean(l2_ds))}


def classify(defn_diff, u_diff, inert_ref_diff, conduit_ref_diff):
    """A-1/A-5 classification on cos_dist.
    combiner: definite diff <= 0.5 * U diff AND U diff >= inert_ref + margin
    conduit : definite diff >= 0.5 * U diff (carry-dependent regardless of class)
    inert   : both low (<= inert_ref-ish)
    """
    margin = 0.5 * (conduit_ref_diff - inert_ref_diff)  # midpoint-based margin
    u_signal = u_diff >= (inert_ref_diff + max(0.02, margin))
    if not u_signal and defn_diff < (inert_ref_diff + 0.02):
        return "inert"
    ratio = defn_diff / (u_diff + 1e-9)
    if u_signal and ratio <= 0.5:
        return "combiner"
    if ratio > 0.5:
        return "conduit"
    return "ambiguous"


def run_control(model_name):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg)
    C = CANDIDATES[model_name]
    n = max(2, cfg.n_digits // 2)
    # conduit reference: LAYER-0 resid at the A_{n+1} token position. Before the
    # L1 MLP resolves, this carries the raw lower-digit context and varies in
    # BOTH regimes (definite and U) -> the conduit signature. (The *predicting*
    # position / layer-1 resid is combiner-like because it holds the RESOLVED
    # carry; verified empirically.)
    an1_tok_pos = cfg.n_ctx - 1 - (n + 1)
    def conduit_ref_diff(cls):
        return activation_diff_resid(model, cfg, n, an1_tok_pos, 0, cls, n_pairs=40)
    # inert reference: early head (reads early digit tokens only; carry-inert)
    (_, sp, sL, sh) = C["inert_ref"]
    inert_def = activation_diff(model, cfg, n, "z", sp, sL, sh, "lo", n_pairs=40)["cos_dist"]
    inert_u = activation_diff(model, cfg, n, "z", sp, sL, sh, "U", n_pairs=40)["cos_dist"]
    cond_def = conduit_ref_diff("lo"); cond_u = conduit_ref_diff("U")
    out = {"model": model_name, "accuracy": acc, "n": n,
           "conduit_ref": {"pos": an1_tok_pos, "def_diff": cond_def, "u_diff": cond_u},
           "inert_ref": {"node": f"P{sp}L{sL}H{sh}", "def_diff": inert_def, "u_diff": inert_u}}
    with open(os.path.join(RESULT_DIR, f"control_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    del model
    return out


def activation_diff_resid(model, cfg, n, pos, L, digit_n_class, n_pairs=40):
    cos_ds = []
    for _ in range(n_pairs):
        qc, qn, a, b = build_pair_fixed_digit_n(cfg, n, digit_n_class)
        with torch.no_grad():
            _, cc = model.run_with_cache(qc.unsqueeze(0))
            _, cn = model.run_with_cache(qn.unsqueeze(0))
        ac = cc[f"blocks.{L}.hook_resid_post"][0, pos, :].numpy()
        an = cn[f"blocks.{L}.hook_resid_post"][0, pos, :].numpy()
        denom = (np.linalg.norm(ac) * np.linalg.norm(an)) + 1e-9
        cos_ds.append(1.0 - float(np.dot(ac, an) / denom))
    return float(np.mean(cos_ds))


def run_models():
    results = {}
    for mn in CANDIDATES:
        ctrl = run_control(mn)
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg)
        assert acc > 0.9
        inert_ref = ctrl["inert_ref"]["def_diff"]   # inert baseline (cos)
        conduit_ref = ctrl["conduit_ref"]["def_diff"]
        C = CANDIDATES[mn]
        print(f"=== {mn} acc={acc:.3f} ===")
        print(f"  conduit-ref def_diff={ctrl['conduit_ref']['def_diff']:.3f} "
              f"u_diff={ctrl['conduit_ref']['u_diff']:.3f} | "
              f"inert-ref def_diff={ctrl['inert_ref']['def_diff']:.3f} "
              f"u_diff={ctrl['inert_ref']['u_diff']:.3f}")
        refs_separate = ctrl["conduit_ref"]["def_diff"] > ctrl["inert_ref"]["def_diff"] + 0.05
        nodes = {}
        allc = [("u_transmitter",) + c for c in C["u_transmitters"]] + \
               [("make_carry",) + c for c in C["make_carry"]]
        for (role, kind, pos, L, head) in allc:
            n = max(2, cfg.n_digits // 2)
            # ensure digit n valid for this node's position (node must be at/after digit n operands)
            dd = activation_diff(model, cfg, n, kind, pos, L, head, "lo", n_pairs=60)
            dh = activation_diff(model, cfg, n, kind, pos, L, head, "hi", n_pairs=60)
            du = activation_diff(model, cfg, n, kind, pos, L, head, "U", n_pairs=60)
            defn = 0.5 * (dd["cos_dist"] + dh["cos_dist"])
            label = classify(defn, du["cos_dist"], inert_ref, conduit_ref)
            name = (f"P{pos}L{L}H{head}" if kind == "z" else f"P{pos}L{L}MLP")
            nodes[name] = {"prior_role": role, "kind": kind, "target_digit": n,
                           "def_diff_cos": defn, "u_diff_cos": du["cos_dist"],
                           "def_diff_lo": dd["cos_dist"], "def_diff_hi": dh["cos_dist"],
                           "ratio_def_over_u": defn / (du["cos_dist"] + 1e-9),
                           "label": label}
            print(f"  [{role:13s}] {name} D{n} def_diff={defn:.3f} u_diff={du['cos_dist']:.3f} "
                  f"ratio={defn/(du['cos_dist']+1e-9):.2f} -> {label}")
        results[mn] = {"accuracy": acc, "refs_separate": refs_separate,
                       "inert_ref_diff": inert_ref, "conduit_ref_diff": conduit_ref,
                       "control": ctrl, "nodes": nodes}
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "control":
        for mn in CANDIDATES:
            c = run_control(mn)
            print(mn, "conduit-ref def/u", round(c["conduit_ref"]["def_diff"], 3),
                  round(c["conduit_ref"]["u_diff"], 3),
                  "| inert-ref def/u", round(c["inert_ref"]["def_diff"], 3),
                  round(c["inert_ref"]["u_diff"], 3))
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
