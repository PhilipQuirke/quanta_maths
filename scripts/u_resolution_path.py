"""Locate the tri-state U-resolution path (study-u-resolution-path.md).

The confirm-ST-node study found binary make-carry heads but showed the tri-state
U-resolution (sum==9, carry-out depends on the lower carry) is a SEPARATE path.
This script sweeps all heads / MLP-out / resid at all positions/layers with the
U counterfactual to find WHERE U is resolved, and distinguishes a true
U-COMBINER from a mere carry-bit CONDUIT.

Implements amendments A-1..A-6:
  * combiner gate: (a) carry-matched-operands consistency + null,
    (b) U-regime interaction (carry-tracking flip specific to digit-n=U) [A-1]
  * node-level positive control on a CE3 make-carry head [A-2]
  * joint (head+MLP) patching before any distributed verdict [A-3]
  * exact lower-carry construction + assertions [A-6]

CPU-only. Run:
    PYTHONPATH=. python3 scripts/u_resolution_path.py control
    PYTHONPATH=. python3 scripts/u_resolution_path.py models
    PYTHONPATH=. python3 scripts/u_resolution_path.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

# reuse validated helpers from the confirm-ST-node study
from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    patched_prediction, _digits_to_int, operand_attention, CONFIRMED_SA_HEAD,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-u-resolution-path")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260715)

# CE3 make-carry heads (answer-position, per model) for the node-level control
MAKE_CARRY = {
    "add_d5_l2_h3_t15K_s372001": [(13, 0, 0, 3), (14, 0, 0, 2), (15, 0, 0, 1)],
    "add_d6_l2_h3_t20K_s173289": [(15, 0, 2, 4), (16, 0, 2, 3), (17, 0, 2, 2)],
}


# ---------------------------------------------------------------------------
# stimulus builders with assertions (A-6)
# ---------------------------------------------------------------------------

def _carry_out(x, y, cin):
    return 1 if (x + y + cin) >= 10 else 0


def build_U_question(cfg, n, lower_carry, digit_n_class="U"):
    """Digit n: sum==9 (U) by default, or a definite class. Digit n-1: sum>=10
    (lower_carry True) or <10 (False). Digits <n-1: no carry. Digits >n: 0.
    Returns (a, b) ints and a dict of invariants for assertion."""
    nd = cfg.n_digits
    d1 = [0] * nd; d2 = [0] * nd
    idx = nd - 1 - n
    if digit_n_class == "U":
        a = int(RNG.integers(0, 10)); b = 9 - a
    elif digit_n_class == "lo":   # definite no-carry, sum<=8
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b <= 8: break
    else:                          # 'hi' definite carry, sum>=10
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b >= 10: break
    d1[idx] = a; d2[idx] = b
    # digit n-1
    ikm = nd - 1 - (n - 1)
    if lower_carry:
        while True:
            x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
            if x + y >= 10: break
    else:
        x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
    d1[ikm] = x; d2[ikm] = y
    # digits < n-1 : no carry
    for k in range(n - 1):
        ik = nd - 1 - k
        u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
        d1[ik] = u; d2[ik] = v
    inv = {"digit_n_sum": a + b, "lower_carry_out": _carry_out(x, y, 0)}
    return _digits_to_int(d1), _digits_to_int(d2), inv


def u_counterfactual_flip(model, cfg, n, hooks_builder, n_pairs=60,
                          digit_n_class="U", src_lower=1, tgt_lower=0):
    """Patch node source(lower_carry=src_lower)->target(tgt_lower); measure
    A_{n+1} flip. digit_n_class controls whether digit n is U or definite."""
    ap = answer_positions(cfg); na = len(ap)
    flips = []
    for _ in range(n_pairs):
        sa, sb, si = build_U_question(cfg, n, bool(src_lower), digit_n_class)
        ta, tb, ti = build_U_question(cfg, n, bool(tgt_lower), digit_n_class)
        # assertions (A-6)
        if digit_n_class == "U":
            assert si["digit_n_sum"] == 9 and ti["digit_n_sum"] == 9
        assert si["lower_carry_out"] == src_lower and ti["lower_carry_out"] == tgt_lower
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        idx = na - 1 - (n + 1)
        flips.append(float(diff[idx]) if 0 <= idx < na else 0.0)
    return float(np.mean(flips))


def same_carry_null(model, cfg, n, hooks_builder, n_pairs=60, digit_n_class="U"):
    """Patch between two sources with the SAME lower carry (both=1) but
    different lower operands -> false-positive floor (A-1a null)."""
    ap = answer_positions(cfg); na = len(ap)
    flips = []
    for _ in range(n_pairs):
        sa, sb, _ = build_U_question(cfg, n, True, digit_n_class)
        ta, tb, _ = build_U_question(cfg, n, True, digit_n_class)
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        idx = na - 1 - (n + 1)
        flips.append(float(diff[idx]) if 0 <= idx < na else 0.0)
    return float(np.mean(flips))


# ---------------------------------------------------------------------------
# node-level positive control (A-2): make-carry head, binary carry counterfactual
# ---------------------------------------------------------------------------

def binary_carry_flip(model, cfg, n, hooks_builder, n_pairs=50):
    """Patch node between digit-n definite carry (hi) vs no-carry (lo); measure
    A_{n+1} flip. This is the CE3 make-carry signature -> node-level control."""
    ap = answer_positions(cfg); na = len(ap)
    flips = []
    for _ in range(n_pairs):
        sa, sb, _ = build_U_question(cfg, n, False, "hi")
        ta, tb, _ = build_U_question(cfg, n, False, "lo")
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        idx = na - 1 - (n + 1)
        flips.append(float(diff[idx]) if 0 <= idx < na else 0.0)
    return float(np.mean(flips))


def run_control(model_name):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg)
    pos, layer, head, digit = MAKE_CARRY[model_name][0]
    def hb():
        return [{"name": f"blocks.{layer}.attn.hook_z", "pos": pos, "head": head}]
    node_ctrl = binary_carry_flip(model, cfg, digit, hb, n_pairs=50)
    # coarse readout sanity: patch resid at the position whose logit PREDICTS
    # A_{n+1} (i.e. the A_{n+1} token position minus 1, autoregressive).
    an1_pred_pos = (cfg.n_ctx - 1 - (digit + 1)) - 1
    def hb_readout():
        return [{"name": "blocks.1.hook_resid_post", "pos": an1_pred_pos}]
    readout = u_counterfactual_flip(model, cfg, digit, hb_readout, n_pairs=40)
    out = {"model": model_name, "accuracy": acc,
           "node_level_control": {"node": f"P{pos}L{layer}H{head}", "digit": digit,
                                  "binary_carry_A_n+1_flip": node_ctrl},
           "coarse_readout_control": {"A_n+1_flip_on_U": readout}}
    with open(os.path.join(RESULT_DIR, f"control_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    del model
    return out, node_ctrl


# ---------------------------------------------------------------------------
# combiner gate (A-1)
# ---------------------------------------------------------------------------

def classify_node(model, cfg, n, hb, bar):
    """Full A-1 gate. Returns dict with the two-part gate outcome + label."""
    u_flip = u_counterfactual_flip(model, cfg, n, hb, n_pairs=60, digit_n_class="U",
                                   src_lower=1, tgt_lower=0)
    u_flip_rev = u_counterfactual_flip(model, cfg, n, hb, n_pairs=60, digit_n_class="U",
                                       src_lower=0, tgt_lower=1)
    null = same_carry_null(model, cfg, n, hb, n_pairs=60, digit_n_class="U")
    # (b) U-regime interaction: same carry-toggle but digit n definite
    def_flip = 0.5 * (
        u_counterfactual_flip(model, cfg, n, hb, n_pairs=40, digit_n_class="lo",
                              src_lower=1, tgt_lower=0)
        + u_counterfactual_flip(model, cfg, n, hb, n_pairs=40, digit_n_class="hi",
                                src_lower=1, tgt_lower=0))
    both_dir = min(u_flip, u_flip_rev)
    passes_a = (both_dir >= bar) and (null <= 0.10)
    regime_specific = (u_flip - def_flip) >= 0.4
    if passes_a and regime_specific:
        label = "combiner"
    elif passes_a and not regime_specific:
        label = "carry-conduit"
    elif both_dir < bar and null <= 0.10:
        label = "none"
    else:
        label = "ambiguous"
    return {"u_flip": u_flip, "u_flip_rev": u_flip_rev, "both_dir": both_dir,
            "same_carry_null": null, "definite_regime_flip": def_flip,
            "regime_specific_gap": u_flip - def_flip, "label": label}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def run_models():
    results = {}
    control_bars = {}
    for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
        c, ncrate = run_control(mn)
        control_bars[mn] = ncrate
        results.setdefault(mn, {})["control"] = c
        print(f"=== CONTROL {mn} ===")
        print(f"  acc={c['accuracy']:.3f} node-ctrl {c['node_level_control']['node']} "
              f"binary_carry_flip={ncrate:.2f} readout_U_flip={c['coarse_readout_control']['A_n+1_flip_on_U']:.2f}")

    for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg)
        assert acc > 0.9
        bar = max(0.50, control_bars[mn] - 0.10)
        print(f"=== SWEEP {mn} (bar={bar:.2f}) ===", flush=True)
        # Stage 1: cheap single-node U-flip screen over all nodes/positions/layers
        # for a representative digit n (mid), to find candidates, then full gate.
        screen_digit = cfg.n_digits // 2
        cand = []
        for L in range(cfg.n_layers):
            for pos in range(cfg.n_ctx):
                for h in range(cfg.n_heads):
                    def hb(pos=pos, L=L, h=h):
                        return [{"name": f"blocks.{L}.attn.hook_z", "pos": pos, "head": h}]
                    f = u_counterfactual_flip(model, cfg, screen_digit, hb, n_pairs=20)
                    if f >= 0.30:
                        cand.append(("z", pos, L, h))
                # MLP-out at this position/layer
                def hbm(pos=pos, L=L):
                    return [{"name": f"blocks.{L}.hook_mlp_out", "pos": pos}]
                fm = u_counterfactual_flip(model, cfg, screen_digit, hbm, n_pairs=20)
                if fm >= 0.30:
                    cand.append(("mlp", pos, L, None))
        # Stage 2: full A-1 gate on candidates across all valid digits
        nodes = {}
        for (kind, pos, L, h) in cand:
            name = (f"P{pos}L{L}H{h}" if kind == "z" else f"P{pos}L{L}MLP")
            def hb(pos=pos, L=L, h=h, kind=kind):
                if kind == "z":
                    return [{"name": f"blocks.{L}.attn.hook_z", "pos": pos, "head": h}]
                return [{"name": f"blocks.{L}.hook_mlp_out", "pos": pos}]
            # pick the digit whose U-flip is strongest for this node
            best = None
            for n in range(1, cfg.n_digits):
                f = u_counterfactual_flip(model, cfg, n, hb, n_pairs=20)
                if best is None or f > best[1]:
                    best = (n, f)
            n = best[0]
            gate = classify_node(model, cfg, n, hb, bar)
            attn = operand_attention(model, cfg, pos, L, h, n) if kind == "z" else None
            nodes[name] = {"kind": kind, "target_digit": n, "bar": bar,
                           "operand_attn": attn, **gate}
        # Stage 3: joint head+MLP at the answer positions (A-3), for the top nodes
        joint = {}
        for (kind, pos, L, h) in cand:
            if kind != "z":
                continue
            def hbj(pos=pos, L=L, h=h):
                return [{"name": f"blocks.{L}.attn.hook_z", "pos": pos, "head": h},
                        {"name": f"blocks.{L}.hook_mlp_out", "pos": pos}]
            best = None
            for n in range(1, cfg.n_digits):
                f = u_counterfactual_flip(model, cfg, n, hbj, n_pairs=20)
                if best is None or f > best[1]:
                    best = (n, f)
            n = best[0]
            gate = classify_node(model, cfg, n, hbj, bar)
            joint[f"P{pos}L{L}H{h}+MLP"] = {"target_digit": n, **gate}
        results[mn]["sweep"] = {"accuracy": acc, "bar": bar, "screen_digit": screen_digit,
                                "n_candidates": len(cand), "nodes": nodes, "joint": joint}
        # console
        for name, nr in nodes.items():
            print(f"  {name} D{nr['target_digit']} u_flip={nr['u_flip']:.2f} "
                  f"both_dir={nr['both_dir']:.2f} null={nr['same_carry_null']:.2f} "
                  f"def_flip={nr['definite_regime_flip']:.2f} gap={nr['regime_specific_gap']:.2f} "
                  f"-> {nr['label']}")
        for name, nr in joint.items():
            print(f"  [joint] {name} D{nr['target_digit']} u_flip={nr['u_flip']:.2f} "
                  f"gap={nr['regime_specific_gap']:.2f} -> {nr['label']}")
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("control",):
        for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
            c, r = run_control(mn)
            print(mn, "node-ctrl binary_carry_flip", round(r, 3),
                  "readout_U_flip", round(c["coarse_readout_control"]["A_n+1_flip_on_U"], 3))
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
