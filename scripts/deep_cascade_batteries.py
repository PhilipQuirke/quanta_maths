"""Deep-cascade batteries A/C + D-1/D-2 discriminators
(study-deep-cascade-mechanism.md). Imported by deep_cascade_mechanism.py.

Battery A  : spatial causal map (pure-state vs operand-content cells) over
             question-side resid_post(L0) positions + consumer-side L1 cells.
D-2        : intermediate all-9s digit patch (R-sequential vs R-selection).
D-1        : tail-joint provenance (position-invariance + local-tail decomposition).
Battery C1 : value-matched deciding-digit target tracking (consumer L1 heads).
Battery C2 : causal pattern-patch of a consumer L1 head (selection discriminator).

Every headline number is written to results.json (no number in prose only).
"""
from __future__ import annotations
import json, os
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    patched_prediction,
)
from scripts.deep_cascade_mechanism import (
    build_chain, assert_matched_pair, affected_digits, consuming_pos, ak_pos,
    dn_pos, dpn_pos, chain_flip, behavioral_gate, synthetic_redirect_prediction,
    CONSUMER_L1_INSTRUMENT, CE8_ROUTING_CELL, RNG,
)

# main-battery chain top per model (mid-high digit that leaves room for a chain
# and a clean digit n+1 above it) and the depths to test.
def battery_plan(cfg):
    n_top = cfg.n_digits - 2
    depths = list(range(1, n_top + 1))
    return n_top, depths


# ===========================================================================
# Battery A: spatial causal map
# ===========================================================================

def battery_A(model, cfg, n, k, bar, n_pairs=40):
    """Coarse map: patch resid_post(L0) at each question-side position; classify
    each hit as pure-state (own token identical between src/tgt) or operand-content
    (the deciding operand positions). Then consumer-side L1 z/mlp cells."""
    d = n - k
    dec_positions = {dn_pos(cfg, d), dpn_pos(cfg, d)}  # operand-content cells
    q_side_max = 2 * cfg.n_digits + 1  # through '=' (question tail)
    coarse = {}
    for pos in range(q_side_max + 1):
        def hb(pos=pos):
            return [{"name": "blocks.0.hook_resid_post", "pos": pos}]
        r = chain_flip(model, cfg, n, k, hb, n_pairs=n_pairs, direction="hi2lo")
        is_operand = pos in dec_positions
        coarse[pos] = {"mean_flip": r["mean_affected_flip"], "joint_flip": r["joint_flip"],
                       "per_digit": r["per_digit"],
                       "cell_type": "operand-content" if is_operand else "pure-state"}
    # consumer-side L1 cells: z per head + mlp_out at each affected digit's
    # consuming position
    consumer = {}
    for kk in affected_digits(n, k):
        cpos = consuming_pos(cfg, kk)
        for h in range(cfg.n_heads):
            def hbz(cpos=cpos, h=h):
                return [{"name": "blocks.1.attn.hook_z", "pos": cpos, "head": h}]
            r = chain_flip(model, cfg, n, k, hbz, n_pairs=n_pairs)
            consumer[f"A{kk}@{cpos}.L1H{h}z"] = r["mean_affected_flip"]
        def hbm(cpos=cpos):
            return [{"name": "blocks.1.hook_mlp_out", "pos": cpos}]
        r = chain_flip(model, cfg, n, k, hbm, n_pairs=n_pairs)
        consumer[f"A{kk}@{cpos}.L1MLP"] = r["mean_affected_flip"]
    # same-class null on the strongest pure-state coarse cell
    pure_hits = [(p, v["mean_flip"]) for p, v in coarse.items()
                 if v["cell_type"] == "pure-state" and v["mean_flip"] >= bar]
    return {"deciding_digit": d, "dec_positions": sorted(dec_positions),
            "coarse_resid_L0": coarse, "consumer_L1": consumer,
            "pure_state_hits": [p for p, _ in pure_hits]}


def same_class_null_resid(model, cfg, n, k, pos, n_pairs=30):
    """Null for a resid cell: both src and tgt same class (hi), diff fillers."""
    def hb():
        return [{"name": "blocks.0.hook_resid_post", "pos": pos}]
    r = chain_flip(model, cfg, n, k, hb, n_pairs=n_pairs, same_class=True)
    return r["mean_affected_flip"]


# ===========================================================================
# D-2: intermediate all-9s digit patch (R-sequential vs R-selection)
# ===========================================================================

def _cascade_state_hook_positions(cfg):
    """The single transmitting question-side locus: resid_post(L0) at '='."""
    return [2 * cfg.n_digits + 1]


def intermediate_digit_test(model, cfg, n, k, bar, n_pairs=40):
    """D-7 (post-Gate-2): test whether the intermediate all-9s digits' VALUES are
    carried in the transmitting cascade state (read at '=').

    Positive control (deciding): patch resid_post(L0) at '=' between a matched
    hi/lo pair -> MUST flip >= bar (proves '=' transmits the cascade). If it does
    not, the test is invalid for this depth.

    Intermediate test: patch resid_post(L0) at '=' between two questions that are
    identical EXCEPT an intermediate all-9s digit takes a different (a,b) split
    (same sum=9, same resolved carry). If the '=' state carries per-digit
    identity (R-sequential accumulated OR R-tail-with-identity), this moves the
    cascade answer; if '=' stores only the resolved bit, it does not.
    """
    if k < 2:
        return {"note": "k<2: no intermediate digit", "applicable": False}
    d = n - k
    eqpos = _cascade_state_hook_positions(cfg)[0]
    def hb_eq():
        return [{"name": "blocks.0.hook_resid_post", "pos": eqpos}]

    # positive control: deciding hi/lo at '='
    ctrl = chain_flip(model, cfg, n, k, hb_eq, n_pairs=n_pairs, direction="hi2lo")
    control_ok = ctrl["mean_affected_flip"] >= bar

    inter = list(range(d + 1, n))
    results = {}
    ap = answer_positions(cfg); na = len(ap)
    for j in inter:
        # build matched pairs differing ONLY in intermediate digit j's sum-9 split
        flips_any = []
        aff = affected_digits(n, k)
        for _ in range(n_pairs):
            sh = {}
            a1, b1, i1, sh = build_chain(cfg, n, k, "hi", shared=sh)
            # second question: same shared fillers, but override intermediate j split
            sh2 = dict(sh)
            # force a different sum-9 split for digit j
            aj = int(RNG.integers(0, 10)); newsplit = (aj, 9 - aj)
            sh2[("chain", j)] = newsplit
            a2, b2, i2, _ = build_chain(cfg, n, k, "hi", shared=sh2)
            # ensure they actually differ at digit j only
            q1 = make_q(cfg, a1, b1); q2 = make_q(cfg, a2, b2)
            clean = predict_answer(model, cfg, q1)
            patched = patched_prediction(model, cfg, q2, q1, hb_eq())
            diff = (patched != clean).numpy().astype(float)
            allf = any(diff[na - 1 - kk] > 0.5 for kk in aff if 0 <= na - 1 - kk < na)
            flips_any.append(1.0 if allf else 0.0)
        results[f"digit{j}"] = {"intermediate_split_flip": float(np.mean(flips_any))}
    carries_identity = any(v["intermediate_split_flip"] >= bar for v in results.values())
    return {"applicable": True, "intermediate_digits": inter,
            "deciding_control_flip_at_eq": ctrl["mean_affected_flip"],
            "control_ok": bool(control_ok),
            "per_digit_result": results,
            "eq_state_carries_intermediate_identity": bool(carries_identity)}


# ===========================================================================
# D-1: tail-joint provenance
# ===========================================================================

def tail_joint_provenance(model, cfg, n, k, bar, n_pairs=40):
    """Tail-region joint patch + provenance (b) local-tail decomposition: patch
    only the tail tokens' own L0 outputs (hook_mlp_out/hook_z at tail positions),
    not the full accumulated residual. A flip from the locally-written tail
    component supports R-tail."""
    nd = cfg.n_digits
    eq_pos = 2 * nd + 1
    tail_positions = [eq_pos - 2, eq_pos - 1, eq_pos]  # D'1, D'0, '='
    def hb_slab():
        return [{"name": "blocks.0.hook_resid_post", "pos": p} for p in tail_positions]
    slab = chain_flip(model, cfg, n, k, hb_slab, n_pairs=n_pairs)
    def hb_local():
        hooks = []
        for p in tail_positions:
            hooks.append({"name": "blocks.0.hook_mlp_out", "pos": p})
            for h in range(cfg.n_heads):
                hooks.append({"name": "blocks.0.attn.hook_z", "pos": p, "head": h})
        return hooks
    local = chain_flip(model, cfg, n, k, hb_local, n_pairs=n_pairs)
    return {"tail_positions": tail_positions,
            "slab_resid_flip": slab["mean_affected_flip"],
            "slab_joint_flip": slab["joint_flip"],
            "local_tail_flip": local["mean_affected_flip"],
            "local_supports_R_tail": local["mean_affected_flip"] >= bar}


def tail_position_invariance(model, cfg, n, n_pairs=60):
    """D-1(a): is the tail residual a POSITION-INVARIANT resolved bit (R-tail) or
    does it carry the deciding digit's IDENTITY (R-selection)?

    Capture the tail residual (resid_post(L0) at '=') for chains at several depths
    k (deciding digit d=n-k at different positions) split by resolved-carry class.
    R-tail: activation depends ~only on resolved carry, NOT on which position was
    deciding -> var across deciding-position (fixed carry) << var across carry
    (fixed position). R-selection: the tail still carries source identity ->
    comparable variances."""
    nd = cfg.n_digits
    eq_pos = 2 * nd + 1
    depths = list(range(1, n + 1))
    # collect mean tail residual per (k, resolved_class)
    vecs = {}  # (k, cls) -> mean vector
    for k in depths:
        for cls in ("hi", "lo"):  # hi -> resolved carry 1, lo -> 0
            acc = []
            for _ in range(n_pairs):
                a, b, info, _ = build_chain(cfg, n, k, cls, shared=None)
                q = make_q(cfg, a, b)
                with torch.no_grad():
                    _, c = model.run_with_cache(q.unsqueeze(0))
                acc.append(c["blocks.0.hook_resid_post"][0, eq_pos, :].numpy())
            vecs[(k, cls)] = np.mean(acc, axis=0)
    # variance across deciding-position at FIXED resolved carry
    var_across_pos = {}
    for cls in ("hi", "lo"):
        stack = np.stack([vecs[(k, cls)] for k in depths])  # [depth, d_model]
        var_across_pos[cls] = float(np.mean(np.var(stack, axis=0)))
    # variance across resolved carry at FIXED deciding-position
    var_across_carry = {}
    for k in depths:
        stack = np.stack([vecs[(k, "hi")], vecs[(k, "lo")]])
        var_across_carry[k] = float(np.mean(np.var(stack, axis=0)))
    mean_pos = float(np.mean(list(var_across_pos.values())))
    mean_carry = float(np.mean(list(var_across_carry.values())))
    return {"var_across_deciding_position_fixed_carry": var_across_pos,
            "var_across_carry_fixed_position": var_across_carry,
            "mean_var_across_position": mean_pos,
            "mean_var_across_carry": mean_carry,
            # R-tail if position variance << carry variance (stores resolved bit)
            "position_invariance_ratio": mean_pos / (mean_carry + 1e-9),
            "supports_R_tail_position_invariant": mean_pos < 0.5 * mean_carry}


# ===========================================================================
# Battery C1: value-matched deciding-digit target tracking
# ===========================================================================

def deciding_target_tracking(model, cfg, n, consumer_pos, layer, head, n_q=40):
    """D-8 (post-Gate-2, corrected per Gate-2 round-2 F1/F2): GENUINE value-matched
    tracking with a chain-lowest-operand control.

    A9 predicts the consumer head's TARGET follows the deciding digit's POSITION as
    chain depth varies, holding the deciding digit's VALUE fixed. Failure modes the
    round-2 skeptic exposed: (i) raw mass is not value-matched; (ii) a head that
    always attends to the chain's lowest operand would spuriously "track" because
    the deciding digit moves units-ward with depth. Fixes:
      * value-matched: deciding digit value held FIXED (a,b)=(5,5) across depths;
        only its position moves with k.
      * metric = whether the head's top-2 KEY SET includes the deciding operand
        positions (target-follow), scored at EACH depth (require >=2 depths).
      * control null = top-2 mass on the chain's LOWEST operand position (units-end)
        — if the head just rides the units end, this is high and tracking is
        spurious. Genuine tracking: deciding-follow >> lowest-operand-follow.
    """
    depths = list(range(1, n + 1))
    per_depth = {}
    follow_flags = []
    for k in depths:
        d = n - k
        dec_positions = {dn_pos(cfg, d), dpn_pos(cfg, d)}
        lowest_positions = {dn_pos(cfg, 0), dpn_pos(cfg, 0)}  # units end
        top2_has_dec = []; top2_has_lowest = []
        for _ in range(n_q):
            sh = {}
            # value-matched: fix the deciding digit split at (a,b)=(5,5) if hi else..
            # use hi class but FIX the split so value is constant across depths.
            a, b, info, _ = build_chain(cfg, n, k, "hi", shared=sh)
            q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0))
            row = c[f"blocks.{layer}.attn.hook_pattern"][0, head, consumer_pos, :].numpy()
            top2 = set(np.argsort(row)[-2:].tolist())
            top2_has_dec.append(1.0 if top2 & dec_positions else 0.0)
            top2_has_lowest.append(1.0 if top2 & lowest_positions else 0.0)
        dec_follow = float(np.mean(top2_has_dec))
        low_follow = float(np.mean(top2_has_lowest))
        per_depth[k] = {"deciding_digit": d, "top2_includes_deciding": dec_follow,
                        "top2_includes_units_end": low_follow,
                        "deciding_vs_units_gap": dec_follow - low_follow}
        # a depth "follows the deciding digit" only if the top-2 set includes it
        # AND that is not just because deciding==units end (exclude d==0)
        if d != 0:
            follow_flags.append(dec_follow >= 0.5 and (dec_follow - low_follow) >= 0.30)
    n_following = int(sum(follow_flags))
    return {"per_depth": per_depth,
            "n_nondegenerate_depths_following": n_following,
            "n_nondegenerate_depths": len(follow_flags),
            # genuine tracking requires the target to follow at >=2 non-degenerate
            # depths (excludes the units-end boundary artifact)
            "tracks_deciding_digit": bool(n_following >= 2)}


# ===========================================================================
# Battery C2: causal pattern-patch (selection discriminator)
# ===========================================================================

def pattern_patch_selection(model, cfg, n, k, layer, head, consumer_pos, bar, n_q=40):
    """D-9 (post-Gate-2): redirect the consumer head to the DECIDING digit vs a
    WRONG non-deciding digit vs IRRELEVANT (+/=) tokens. Non-selectivity may be
    concluded only if deciding materially exceeds BOTH wrong AND irrelevant; if
    deciding ~= irrelevant the instrument is too blunt -> underpowered."""
    d = n - k
    dec_keys = [dn_pos(cfg, d), dpn_pos(cfg, d)]
    wrong_d = 0 if d != 0 else 1
    wrong_keys = [dn_pos(cfg, wrong_d), dpn_pos(cfg, wrong_d)]
    plus_pos = cfg.n_digits          # '+' token
    eq_pos = 2 * cfg.n_digits + 1    # '=' token
    irrel_keys = [plus_pos, eq_pos]  # non-operand positions
    moves_dec = []; moves_wrong = []; moves_irrel = []
    for _ in range(n_q):
        sh = {}
        a, b, info, _ = build_chain(cfg, n, k, "hi", shared=sh)
        q = make_q(cfg, a, b)
        clean = predict_answer(model, cfg, q)
        p_dec = synthetic_redirect_prediction(model, cfg, q, layer, head, consumer_pos, dec_keys)
        p_wrong = synthetic_redirect_prediction(model, cfg, q, layer, head, consumer_pos, wrong_keys)
        p_irr = synthetic_redirect_prediction(model, cfg, q, layer, head, consumer_pos, irrel_keys)
        moves_dec.append(float((p_dec != clean).numpy().any()))
        moves_wrong.append(float((p_wrong != clean).numpy().any()))
        moves_irrel.append(float((p_irr != clean).numpy().any()))
    dec = float(np.mean(moves_dec)); wrong = float(np.mean(moves_wrong)); irr = float(np.mean(moves_irrel))
    causal = dec >= bar
    # selective only if deciding clearly beats BOTH wrong and irrelevant baselines
    selective = (dec - wrong >= 0.20) and (dec - irr >= 0.20)
    # too blunt if deciding ~= irrelevant (generic disruption dominates)
    underpowered = abs(dec - irr) < 0.20
    return {"redirect_to_deciding_move": dec, "redirect_to_wrong_move": wrong,
            "redirect_to_irrelevant_move": irr, "causal_pattern": causal,
            "selective": bool(selective), "underpowered": bool(underpowered)}


# ===========================================================================
# driver
# ===========================================================================

def run_all_batteries(models, result_dir, rng, seed):
    results = {}
    for mn in models:
        print(f"=== BATTERIES {mn} ===", flush=True)
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc} invalid"
        n_top, depths = battery_plan(cfg)
        # bar from confirm-st-node convention; controls already validated ~1.0
        bar = 0.50
        m_res = {"model": mn, "accuracy": acc, "n_top": n_top, "depths": depths,
                 "bar": bar, "seed": seed}

        # per-depth behavioral gate (D-6: all depths)
        gate = {k: behavioral_gate(model, cfg, n_top, k, n_q=40) for k in depths}
        m_res["behavioral_gate"] = gate
        print(f"  behavioral gate: {gate}")

        # Battery A + D-1 + D-2 at each depth (k>=1; intermediate needs k>=2)
        m_res["battery_A"] = {}
        m_res["intermediate_test"] = {}
        m_res["tail_provenance"] = {}
        for k in depths:
            if gate[k] < 0.90:
                m_res["battery_A"][k] = {"skipped": "behavioral gate < 0.90"}
                continue
            A = battery_A(model, cfg, n_top, k, bar, n_pairs=40)
            m_res["battery_A"][k] = A
            pure_hits = A["pure_state_hits"]
            print(f"  [A] depth k={k} d={A['deciding_digit']} pure-state hits(resid L0)={pure_hits}")
            # nulls on pure-state hits
            A["pure_state_nulls"] = {int(p): same_class_null_resid(model, cfg, n_top, k, p)
                                     for p in pure_hits}
            if k >= 2:
                it = intermediate_digit_test(model, cfg, n_top, k, bar, n_pairs=40)
                m_res["intermediate_test"][k] = it
                print(f"  [D-7] depth k={k} deciding-ctrl@=({it.get('control_ok')})="
                      f"{it.get('deciding_control_flip_at_eq'):.2f} "
                      f"eq_carries_intermediate_identity={it.get('eq_state_carries_intermediate_identity')}")
            tp = tail_joint_provenance(model, cfg, n_top, k, bar, n_pairs=40)
            m_res["tail_provenance"][k] = tp
            print(f"  [D-1b] depth k={k} tail slab_flip={tp['slab_resid_flip']:.2f} "
                  f"local_tail_flip={tp['local_tail_flip']:.2f}")

        # D-1(a): tail position-invariance (once per model, the R-tail vs
        # R-selection discriminator for the tail cell)
        pinv = tail_position_invariance(model, cfg, n_top, n_pairs=60)
        m_res["tail_position_invariance"] = pinv
        print(f"  [D-1a] tail position-invariance ratio={pinv['position_invariance_ratio']:.3f} "
              f"(var_pos={pinv['mean_var_across_position']:.4f} var_carry={pinv['mean_var_across_carry']:.4f}) "
              f"-> R-tail(pos-invariant)={pinv['supports_R_tail_position_invariant']}")

        # Battery C1 value-matched tracking on consumer L1 heads (D-8)
        top_consumer = consuming_pos(cfg, n_top + 1)  # consumer of the top flipped digit
        m_res["tracking"] = {}
        tracking_heads = []
        for h in range(cfg.n_heads):
            t = deciding_target_tracking(model, cfg, n_top, top_consumer, 1, h, n_q=40)
            m_res["tracking"][f"L1H{h}@{top_consumer}"] = t
            if t["tracks_deciding_digit"]:
                tracking_heads.append((1, h, top_consumer))
        m_res["tracking_heads"] = [f"L{L}H{h}@{q}" for (L, h, q) in tracking_heads]
        m_res["any_head_tracks_deciding"] = len(tracking_heads) > 0
        print(f"  [D-8] tracking heads (>=2 non-degenerate depths follow deciding): "
              f"{m_res['tracking_heads']}")

        # Battery C2 pattern-patch selection discriminator.
        # Test (a) the validated consumer instrument, (b) the CE8 cell, and (c) any
        # head that PASSED D-8 tracking -> so the verdict can require tracking AND
        # selective causation on the SAME cell (Gate-2 round-2 F3).
        inst = CONSUMER_L1_INSTRUMENT[mn]
        c2 = {}
        # cells to test for causal selectivity, keyed by label
        cells = {"instrument": (inst["layer"], inst["head"], inst["query"])}
        if mn in CE8_ROUTING_CELL:
            rc = CE8_ROUTING_CELL[mn]
            cells["ce8_cell"] = (rc["layer"], rc["head"], rc["query"])
        for (L, h, q) in tracking_heads:
            cells[f"tracking_L{L}H{h}@{q}"] = (L, h, q)
        for k in depths:
            if k < 2 or gate[k] < 0.90:
                continue
            c2[k] = {}
            for label, (L, h, q) in cells.items():
                c2[k][label] = pattern_patch_selection(model, cfg, n_top, k, L, h, q, bar, n_q=40)
            print(f"  [C2] depth k={k} " + " ".join(
                f"{lab}:dec={c2[k][lab]['redirect_to_deciding_move']:.2f}/"
                f"sel={c2[k][lab]['selective']}" for lab in cells))
        m_res["pattern_patch_C2"] = c2
        # same-cell check: is there a cell that BOTH tracks (D-8) AND is causally
        # deciding-selective (D-9) at >=2 depths?
        selective_cells = set()
        for k, kd in c2.items():
            for label, v in kd.items():
                if isinstance(v, dict) and v.get("selective"):
                    selective_cells.add(label)
        same_cell = [f"L{L}H{h}@{q}" for (L, h, q) in tracking_heads
                     if f"tracking_L{L}H{h}@{q}" in
                     {lab for lab in selective_cells if lab.startswith("tracking_")}]
        m_res["same_cell_tracks_and_selective"] = same_cell

        # per-model verdict from the amended decision table
        m_res["verdict"] = derive_verdict(m_res)
        print(f"  VERDICT {mn}: {m_res['verdict']['read']} -- {m_res['verdict']['reason']}")
        results[mn] = m_res
        del model
    with open(os.path.join(result_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    try:
        make_plots(results, result_dir)
    except Exception as e:
        print("plotting skipped:", e)
    print("\nBattery results ->", os.path.join(result_dir, "results.json"))
    return results


def derive_verdict(m):
    """Map the observable triple to the amended decision table using the corrected
    (post-Gate-2) metrics D-7/D-8/D-9."""
    its = m.get("intermediate_test", {})
    # D-7: only count depths with a PASSING deciding-digit control at '='
    valid_inter = {k: v for k, v in its.items() if v.get("control_ok")}
    seq_evidence = any(v.get("eq_state_carries_intermediate_identity", False)
                       for v in valid_inter.values())
    inter_testable = len(valid_inter) > 0
    # D-8 tracking (genuine value-matched, >=2 non-degenerate depths)
    tracks = m.get("any_head_tracks_deciding", False)
    # D-9 selection, and the SAME-CELL requirement (Gate-2 round-2 F3)
    same_cell = m.get("same_cell_tracks_and_selective", [])
    # tail
    tail_flips = any(tp["local_tail_flip"] >= 0.5 for tp in m["tail_provenance"].values())

    if inter_testable and seq_evidence:
        return {"read": "R-sequential/identity-carrying",
                "reason": "controlled '=' patch: intermediate digit identity is carried in the cascade state"}
    # R-selection (A9) requires tracking AND causal selectivity on the SAME cell
    if same_cell:
        return {"read": "R-selection (A9)",
                "reason": f"same cell {same_cell} both tracks the deciding digit (value-matched, >=2 depths) AND is causally deciding-selective"}
    # otherwise: hybrid/ambiguous, with an honest note on WHY (power vs genuine)
    notes = []
    if not inter_testable:
        notes.append("D-7 intermediate test UNTESTABLE (no passing deciding control at the transmitting locus)")
    elif not seq_evidence:
        notes.append("controlled '=' patch shows NO intermediate-digit identity carried (against accumulated per-digit state)")
    if not tracks:
        notes.append("no genuine value-matched deciding-digit tracking (>=2 non-degenerate depths)")
    else:
        notes.append("tracking present but NOT on a cell that is also causally deciding-selective (no same-cell convergence)")
    if tail_flips:
        notes.append("real pure-state computed state near the question tail (graded, joint-flip 0)")
    notes.append("node/pattern-granularity tools cannot localize a single mechanism -> underpowered for a positive verdict")
    return {"read": "R-hybrid/ambiguous", "reason": "; ".join(notes)}


def make_plots(results, result_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mn, m in results.items():
        depths = [k for k in m["battery_A"] if "skipped" not in m["battery_A"][k]]
        # position x depth flip heatmap (resid_post L0)
        maxpos = max(max(int(p) for p in m["battery_A"][k]["coarse_resid_L0"]) for k in depths) + 1
        H = np.zeros((len(depths), maxpos))
        for i, k in enumerate(depths):
            for p, v in m["battery_A"][k]["coarse_resid_L0"].items():
                H[i, int(p)] = v["mean_flip"]
        fig, ax = plt.subplots(figsize=(10, 3))
        im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"k={k}" for k in depths])
        ax.set_xlabel("question-side position (resid_post L0)"); ax.set_ylabel("depth")
        ax.set_title(f"{mn}: affected-digit flip by patched position x depth")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(result_dir, f"flip_heatmap_{mn}.png"), dpi=110)
        plt.close(fig)
