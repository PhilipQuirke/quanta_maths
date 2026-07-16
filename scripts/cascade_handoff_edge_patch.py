"""Deep-cascade hand-off: L1-head->combiner edge path-patch
(study-cascade-handoff-edge-patch.md).

Tests whether a single answer-position L1 attention head's output causally drives
the answer-position L1-MLP combiner (CE5) along the head->MLP edge, and whether
that edge carries the DECIDING digit's carry in a `...999` chain (A9's
selection->combine hand-off).

Implements the pre-run design + Gate-1 amendments E-1..E-6:
  * E-1: per-digit combiner edge patch (each affected digit at its OWN consuming
    position); "joint" = same head carries the edge at >=2 digits.
  * E-2: LN-fair arm (freeze ln2 scale) + scaled-direct-path power control; no
    underpowered R-direct-path/R-none verdict.
  * E-3: MLP-only routing arm (flip must be carried by the MLP-out delta, not the
    direct skip) for an A9 verdict.
  * E-4: deciding vs wrong vs irrelevant selectivity (D-9 discipline) + edge norms.
  * E-5: same-cell logic (CE9 selective/tracking heads).
  * E-6: physical consuming_pos(j), not CE5 digit labels.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/cascade_handoff_edge_patch.py control
    PYTHONPATH=. python3 scripts/cascade_handoff_edge_patch.py models
    PYTHONPATH=. python3 scripts/cascade_handoff_edge_patch.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
)
from scripts.deep_cascade_mechanism import (
    build_chain, affected_digits, consuming_pos, dn_pos, dpn_pos, behavioral_gate,
    RNG,
)
# Edge/OV-path primitives now live in the library (Stage 3 migration).
from quanta_maths.maths_edge_patch import (
    head_edge_delta as _lib_head_edge_delta,
    direct_path_delta as _lib_direct_path_delta,
    run_edge_patch as _lib_run_edge_patch,
    ln_scale as _ln_scale,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-cascade-handoff-edge-patch")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716

MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]

# CE9 flagged heads (6-digit): causally-selective consumer + tracking head.
CE9_HEADS = {
    "add_d6_l2_h3_t20K_s173289": {"selective": (1, 0, 14), "tracking": (1, 2, 15)},
    "add_d5_l2_h3_t15K_s372001": {},  # 5-digit had no clean causal L1 head
}


def battery_plan(cfg):
    n_top = cfg.n_digits - 2
    depths = list(range(1, n_top + 1))
    return n_top, depths


# ===========================================================================
# edge patching primitives -- thin shims over quanta_maths.maths_edge_patch.
# This study always receives at L1 (the answer-position combiner), so recv_layer=1.
# ===========================================================================

def edge_delta(model, src_cache, tgt_cache, cpos, h):
    """The head-h -> resid_mid edge delta at position cpos (source - target)."""
    return _lib_head_edge_delta(model, src_cache, tgt_cache, cpos, layer=1, head=h)


def run_edge_patch(model, cfg, target_q, cpos, delta, arm="raw", tgt_cache=None):
    return _lib_run_edge_patch(model, cfg, target_q, cpos, delta,
                               recv_layer=1, arm=arm, tgt_cache=tgt_cache)


def direct_path_delta(src_cache, tgt_cache, cpos, scale=1.0):
    return _lib_direct_path_delta(src_cache, tgt_cache, cpos, layer=0, scale=scale)


# ===========================================================================
# per-digit edge-flip battery (E-1)
# ===========================================================================

def digit_flip(model, cfg, n, k, source_cls, n_pairs, patch_fn):
    """Over matched pairs (source class vs opposite), apply patch_fn(sch, tcl, cpos)
    per affected digit, read that digit's flip. Returns per-digit flip dict."""
    ap = answer_positions(cfg); na = len(ap)
    aff = affected_digits(n, k)
    per = {j: [] for j in aff}
    tgt_cls = "lo" if source_cls == "hi" else "hi"
    for _ in range(n_pairs):
        sh = {}
        sa, sb, si, sh = build_chain(cfg, n, k, source_cls, shared=sh)
        ta, tb, ti, _ = build_chain(cfg, n, k, tgt_cls, shared=sh)
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        with torch.no_grad():
            _, sch = model.run_with_cache(sq.unsqueeze(0))
            _, tcl = model.run_with_cache(tq.unsqueeze(0))
        for j in aff:
            cpos = consuming_pos(cfg, j)
            patched = patch_fn(sch, tcl, cpos, tq)
            idx = na - 1 - j
            per[j].append(float(patched[idx] != clean[idx]))
    return {j: float(np.mean(v)) for j, v in per.items()}


def per_head_edge_battery(model, cfg, n, k, n_pairs=40):
    """E-1/E-2/E-3: per-head edge flips (raw + lnfair + mlp_only arms), direct-path
    edge (full), and the PER-CELL power control (E-7 fix): a direct-path edge
    scaled DOWN to that cell's own mean single-head edge norm. If the scaled-down
    direct edge is sub-bar at a cell, single-head edges are NOT detectable there
    and the cell is `underpowered` (no direct-path/none verdict allowed)."""
    out = {}
    for h in range(cfg.n_heads):
        for arm in ("raw", "lnfair", "mlp_only"):
            def pf(sch, tcl, cpos, tq, h=h, arm=arm):
                d = edge_delta(model, sch, tcl, cpos, h)
                return run_edge_patch(model, cfg, tq, cpos, d, arm=arm, tgt_cache=tcl)
            out[f"H{h}_{arm}"] = digit_flip(model, cfg, n, k, "hi", n_pairs, pf)

    def dp(sch, tcl, cpos, tq, scale=1.0):
        d = direct_path_delta(sch, tcl, cpos, scale=scale)
        return run_edge_patch(model, cfg, tq, cpos, d, arm="raw", tgt_cache=tcl)

    out["direct_full"] = digit_flip(model, cfg, n, k, "hi", n_pairs,
                                    lambda s, t, c, q: dp(s, t, c, q, 1.0))

    # E-7 fix: per-cell power control. At each (pair, digit) scale the direct-path
    # delta DOWN to min(1, mean_head_edge_norm / direct_norm) so its magnitude
    # matches a single head's edge. A flip here => a single-head-magnitude edge IS
    # detectable at this cell.
    def dp_percell(sch, tcl, cpos, tq):
        hd = np.mean([float(edge_delta(model, sch, tcl, cpos, h).norm())
                      for h in range(cfg.n_heads)])
        dd = float(direct_path_delta(sch, tcl, cpos, 1.0).norm())
        sc = min(1.0, hd / dd) if dd > 1e-6 else 0.0
        return dp(sch, tcl, cpos, tq, sc)
    out["direct_scaled_percell"] = digit_flip(model, cfg, n, k, "hi", n_pairs, dp_percell)
    return out


# ===========================================================================
# selectivity (E-4): deciding vs wrong vs irrelevant edge
# ===========================================================================

def selectivity_battery(model, cfg, n, k, layer, head, n_pairs=40):
    """For the flagged head: does the edge flip specifically when the DECIDING
    digit differs, vs a wrong (non-chain) digit, vs an irrelevant-token edge?"""
    ap = answer_positions(cfg); na = len(ap)
    aff = affected_digits(n, k)
    # deciding: matched hi/lo (standard). wrong: toggle a below-chain digit.
    # We score the top flipped digit A_{n+1} (the chain top) as the summary.
    top = n + 1
    cpos = consuming_pos(cfg, top)
    idx = na - 1 - top

    def edge_flip_rate(source_builder):
        flips = []
        for _ in range(n_pairs):
            sq, tq = source_builder()
            clean = predict_answer(model, cfg, tq)
            with torch.no_grad():
                _, sch = model.run_with_cache(sq.unsqueeze(0))
                _, tcl = model.run_with_cache(tq.unsqueeze(0))
            d = edge_delta(model, sch, tcl, cpos, head)
            p = run_edge_patch(model, cfg, tq, cpos, d, arm="lnfair", tgt_cache=tcl)
            flips.append(float(p[idx] != clean[idx]))
        return float(np.mean(flips))

    def deciding():
        sh = {}
        sa, sb, _, sh = build_chain(cfg, n, k, "hi", shared=sh)
        ta, tb, _, _ = build_chain(cfg, n, k, "lo", shared=sh)
        return make_q(cfg, sa, sb), make_q(cfg, ta, tb)

    def same_class_diff_operand():
        # both HI (same carry-out) but the deciding operand VALUE differs (chain
        # fillers held shared) -> only the deciding operand differs. Low flip =>
        # the edge carries the COMPUTED carry, not the operand content.
        sh = {}
        sa, sb, _, sh = build_chain(cfg, n, k, "hi", shared=sh)
        sh2 = dict(sh)  # keep chain/low/above fillers; only the deciding operand is re-drawn
        ta, tb, _, _ = build_chain(cfg, n, k, "hi", shared=sh2)
        return make_q(cfg, sa, sb), make_q(cfg, ta, tb)

    def wrong_digit():
        # E-4: toggle the class of a BELOW-deciding (non-chain) digit instead of the
        # deciding digit; the deciding class is held. A head carrying the *selected
        # deciding* carry should NOT flip for a wrong-digit source.
        sh = {}
        sa, sb, _, sh = build_chain(cfg, n, k, "hi", shared=sh)
        sh2 = dict(sh)
        aj = int(RNG.integers(0, 5)); sh2[("low", 0)] = (aj + 5, 5 - aj + 4)  # force carry at units
        ta, tb, _, _ = build_chain(cfg, n, k, "hi", shared=sh2)
        return make_q(cfg, sa, sb), make_q(cfg, ta, tb)

    dec = edge_flip_rate(deciding)
    null = edge_flip_rate(same_class_diff_operand)
    wrong = edge_flip_rate(wrong_digit)
    return {"deciding_flip": dec, "same_class_null": null, "wrong_digit_flip": wrong,
            "carries_computed_carry": bool(dec >= 0.50 and null <= 0.20),
            "deciding_selective": bool(dec - wrong >= 0.20 and dec - null >= 0.20)}


# ===========================================================================
# controls
# ===========================================================================

def run_controls(model_name):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg, n=64)
    n_top, depths = battery_plan(cfg)
    out = {"model": model_name, "accuracy": acc, "seed": SEED, "n_top": n_top}

    # control 4: behavioral gate (reused)
    out["control_behavioral_gate"] = {k: behavioral_gate(model, cfg, n_top, k, n_q=40)
                                      for k in depths}

    # control 2: full resid_mid patch per affected digit reproduces the flip
    k = min(2, n_top)
    def full_pf(sch, tcl, cpos, tq):
        src = sch["blocks.1.hook_resid_mid"][0, cpos, :]
        dt = tcl["blocks.1.hook_resid_mid"][0, cpos, :]
        return run_edge_patch(model, cfg, tq, cpos, src - dt, arm="raw", tgt_cache=tcl)
    out["control2_full_resid_mid"] = digit_flip(model, cfg, n_top, k, "hi", 40, full_pf)

    # control 5: additivity — sum of per-head edges + direct-path == full resid_mid
    out["control5_additivity"] = additivity_check(model, cfg, n_top, k)

    # control 1: edge-patch validity — the direct-path edge (a known-causal, large
    # contribution) must move the answer (proves an edge patch CAN move it).
    def dp_pf(sch, tcl, cpos, tq):
        d = direct_path_delta(sch, tcl, cpos, 1.0)
        return run_edge_patch(model, cfg, tq, cpos, d, arm="raw", tgt_cache=tcl)
    out["control1_edge_validity_directpath"] = digit_flip(model, cfg, n_top, k, "hi", 40, dp_pf)

    with open(os.path.join(RESULT_DIR, f"control_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    del model
    return out


def additivity_check(model, cfg, n, k, n_samp=10):
    """Sum of per-head edge deltas + direct-path delta should equal the full
    resid_mid delta (source-target) at the combiner position."""
    errs = []
    WO = model.blocks[1].attn.W_O
    for _ in range(n_samp):
        sh = {}
        sa, sb, _, sh = build_chain(cfg, n, k, "hi", shared=sh)
        ta, tb, _, _ = build_chain(cfg, n, k, "lo", shared=sh)
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        with torch.no_grad():
            _, sch = model.run_with_cache(sq.unsqueeze(0))
            _, tcl = model.run_with_cache(tq.unsqueeze(0))
        for j in affected_digits(n, k):
            cpos = consuming_pos(cfg, j)
            full = (sch["blocks.1.hook_resid_mid"][0, cpos, :]
                    - tcl["blocks.1.hook_resid_mid"][0, cpos, :])
            heads = sum(edge_delta(model, sch, tcl, cpos, h) for h in range(cfg.n_heads))
            direct = direct_path_delta(sch, tcl, cpos, 1.0)
            errs.append(float((full - (heads + direct)).abs().max()))
    return {"max_abs_reconstruction_error": float(np.max(errs))}


# ===========================================================================
# driver
# ===========================================================================

def run_models():
    results = {}
    for mn in MODELS:
        print(f"=== BATTERIES {mn} ===", flush=True)
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        n_top, depths = battery_plan(cfg)
        gate = {k: behavioral_gate(model, cfg, n_top, k, n_q=40) for k in depths}
        m = {"model": mn, "accuracy": acc, "n_top": n_top, "gate": gate, "seed": SEED}
        m["edge_battery"] = {}
        for k in depths:
            if k < 1 or gate[k] < 0.90:
                m["edge_battery"][k] = {"skipped": f"gate {gate.get(k)}"}
                continue
            eb = per_head_edge_battery(model, cfg, n_top, k, n_pairs=40)
            m["edge_battery"][k] = eb
            # console: max single-head lnfair flip vs direct-path
            def maxflip(prefix):
                vals = [max(v.values()) for kk, v in eb.items()
                        if kk.startswith(prefix) and isinstance(v, dict)]
                return max(vals) if vals else float("nan")
            print(f"  k={k}: max single-head lnfair flip="
                  f"{max((max(eb[f'H{h}_lnfair'].values()) for h in range(cfg.n_heads))):.2f} "
                  f"direct_full={max(eb['direct_full'].values()):.2f} "
                  f"direct_scaled_percell={max(eb['direct_scaled_percell'].values()):.2f}")
        # selectivity (E-4/E-5): for each depth, run on EVERY L1 head at the top
        # digit so we can report it for whichever head actually carries the edge,
        # and flag whether that head is a CE9-flagged cell (E-5).
        m["selectivity"] = {}
        ce9 = {v[:2] for v in CE9_HEADS.get(mn, {}).values()}  # {(L,h)}
        for k in depths:
            if gate[k] < 0.90 or k < 2:
                continue
            for h in range(cfg.n_heads):
                s = selectivity_battery(model, cfg, n_top, k, 1, h, n_pairs=40)
                s["is_ce9_head"] = (1, h) in ce9
                m["selectivity"][f"L1H{h}_k{k}"] = s
        m["verdict"] = derive_verdict(m, cfg)
        print(f"  VERDICT {mn}: {m['verdict']['read']} -- {m['verdict']['reason']}")
        results[mn] = m
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nResults ->", os.path.join(RESULT_DIR, "results.json"))
    return results


def derive_verdict(m, cfg):
    """Honest per-digit reading (Gate-2-round-2 lesson: no blind max over cells).
    Classify each affected (depth, digit) cell, then summarize."""
    bar = 0.50
    n_head_carried = 0     # cells where the carry is delivered by attention (a single head >= bar)
    n_direct_carried = 0   # cells where the direct residual path carries it
    n_none = 0             # cells where neither single-head nor direct >= bar
    n_underpowered = 0     # cells where single-head undetectable-in-principle
    total = 0
    for k, eb in m["edge_battery"].items():
        if "skipped" in eb:
            continue
        # per affected digit, look across heads (lnfair primary)
        digits = list(eb["direct_full"].keys())
        for j in digits:
            total += 1
            head_flip = max(eb[f"H{h}_lnfair"].get(j, 0.0) for h in range(cfg.n_heads))
            direct_flip = eb["direct_full"].get(j, 0.0)
            # PER-CELL power gate (E-7): is a single-head-magnitude edge detectable here?
            scaled_det = eb["direct_scaled_percell"].get(j, 0.0) >= bar
            if head_flip >= bar:
                n_head_carried += 1
            elif direct_flip >= bar:
                n_direct_carried += 1
            elif not scaled_det:
                n_underpowered += 1
            else:
                n_none += 1
    frac_head = n_head_carried / total if total else 0.0
    frac_direct = n_direct_carried / total if total else 0.0
    reason = (f"{n_head_carried}/{total} cells carry via a single L1 attention head edge; "
              f"{n_direct_carried}/{total} via the direct residual path; "
              f"{n_none}/{total} neither (powered); {n_underpowered}/{total} underpowered")
    # dominant reading
    if frac_head >= 0.5 and frac_direct == 0.0:
        read = "R-attention-delivered (edge; A9-consistent)"
    elif frac_direct >= 0.5:
        read = "R-direct-path"
    elif frac_head > 0 and frac_direct == 0.0:
        read = "R-attention-partial (top digits head-carried; lower digits not localized)"
    else:
        read = "R-mixed/distributed"
    return {"read": read, "reason": reason,
            "n_head_carried": n_head_carried, "n_direct_carried": n_direct_carried,
            "n_none": n_none, "n_underpowered": n_underpowered, "total": total}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("control", "controls", "all"):
        for mn in MODELS:
            print(f"=== CONTROLS {mn} ===", flush=True)
            c = run_controls(mn)
            print(f"  acc={c['accuracy']:.3f} gate={c['control_behavioral_gate']}")
            print(f"  [C1] direct-path edge validity flips={c['control1_edge_validity_directpath']}")
            print(f"  [C2] full resid_mid flips={c['control2_full_resid_mid']}")
            print(f"  [C5] additivity max err={c['control5_additivity']['max_abs_reconstruction_error']:.2e}")
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
