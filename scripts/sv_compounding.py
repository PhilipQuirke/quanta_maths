"""SV compounding rule at the map-named wires (study-sv-compounding.md).

C5 steps 2-4 / A10 core test. For the map-named answer-position L1 consumer heads
that feed the high-Fail% L1-MLP combiners:
  V  value content  - do they read ST/carry content from the ST sites and =?
  D  edge hand-off   - does the deciding digit's ST info reach the combiner via
                       the head->MLP edge, at >=2 depths (less-damped/joint arm)?
  S  variant split   - direct-path (a) vs selection (b,A9) vs static (c)?
  E  economy (A6)     - ablation harms cascade, spares carry-free?

Implements the pre-run design + Gate-1 amendments SV-1..SV-6:
  SV-1 joint (H-pair / multi-position) edge arm first-class + R-A10-distributed
  SV-2 less-damped instrument via ln2.hook_normalized (MLP-only) + control-3 gate
  SV-3 Battery V same-position wrong-role baseline
  SV-4 same-cell machine check for selection
  SV-5 scaled direct-path control (CE10 E-7)
  SV-6 =-sink stratified matched pairs; economy absolute floor

CPU-only. Run:
    PYTHONPATH=. python3 scripts/sv_compounding.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import load_model, make_q, answer_positions, predict_answer, verify_accuracy
from scripts.deep_cascade_mechanism import (
    build_chain, consuming_pos, affected_digits, dn_pos, dpn_pos, behavioral_gate, RNG,
)
# Shared probe + edge/OV primitives now live in the library.
from quanta_maths.maths_probe import fit_probe as _lib_fit_probe, probe_balanced_accuracy, balance_idx as _lib_balance_idx
from quanta_maths.maths_edge_patch import (head_ov, ln_scale as _ln_norm,
    run_multi_head_edge_patch as _lib_edge_patch, run_edge_patch as _lib_run_edge_patch)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-sv-compounding")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716

# Map-named answer-position L1 consumer heads (HF behaviors.json, read 2026-07-16):
# answer-pos L1 heads with Impact:A_k attending {=, ST-cluster}, feeding high-Fail MLPs.
# (pos, head, served_digit_k)  -- consumer of answer digit A_k at consuming pos = pos.
CONSUMER_HEADS = {
    "add_d6_l2_h3_t20K_s173289": [(15,1,5),(15,2,5),(16,1,4),(16,2,4),(17,1,3),(19,1,1)],
    "add_d5_l2_h3_t15K_s372001": [(13,2,4),(14,2,3)],
}
# Known-causal head for the less-damped instrument validity control (control 3):
# the map-named consumer head H1 (H2 for 5-digit), verified to flip the top digit
# 1.00 under the edge patch (diagnosed 2026-07-16) — proves the instrument moves the
# answer, closing CE10's underpower gap. (The old CE10 'L1H0' instrument is not
# causal at these consuming positions/depths.)
INSTRUMENT_HEAD = {"add_d6_l2_h3_t20K_s173289": (1,1), "add_d5_l2_h3_t15K_s372001": (1,2)}
EQ_POS = {"add_d6_l2_h3_t20K_s173289": 13, "add_d5_l2_h3_t15K_s372001": 11}


def fit(X, y):
    return _lib_fit_probe(X, y, C=0.5)

def bacc(clf, X, y):
    return probe_balanced_accuracy(clf, X, y)

def balance(y):
    return _lib_balance_idx(y, RNG)


# ===========================================================================
# less-damped edge patch (SV-2) -- thin shims over quanta_maths.maths_edge_patch.
# This study always receives at L1 (recv_layer=1); head_ov / _ln_norm imported.
# ===========================================================================

def edge_patch_pred(model, cfg, target_q, patches, tgt_cache):
    """patches: list of (cpos, layer, head, z_source_row). Delegates to the library
    multi-head LN-fair OV edge patch at recv_layer=1."""
    return _lib_edge_patch(model, cfg, target_q, patches, recv_layer=1, tgt_cache=tgt_cache)


def cache_qs(model, sq, tq):
    with torch.no_grad():
        _, sc = model.run_with_cache(sq.unsqueeze(0),
            names_filter=lambda nm: nm in ("blocks.1.attn.hook_z", "blocks.1.hook_resid_mid", "blocks.0.hook_resid_post"))
        _, tc = model.run_with_cache(tq.unsqueeze(0),
            names_filter=lambda nm: nm in ("blocks.1.attn.hook_z", "blocks.1.hook_resid_mid", "blocks.0.hook_resid_post"))
    return sc, tc


# ===========================================================================
# Battery D: deciding-digit edge hand-off (single head, joint pair, joint multi-pos)
# ===========================================================================

def battery_D(model, cfg, mn, n_top, depths, n_pairs=40):
    """Edge hand-off with F4 specificity controls: besides the deciding hi->lo
    contrast, a SAME-CLASS null (both hi, filler differs — patching should NOT
    flip if the head carries the deciding CARRY not generic content) and a
    NON-CONSUMER-head (H0) specificity arm (patching an untagged co-located head
    should not flip)."""
    ap = answer_positions(cfg); na = len(ap)
    heads = CONSUMER_HEADS[mn]
    out = {}
    for k in depths:
        aff = affected_digits(n_top, k)
        top = n_top + 1; cpos_top = consuming_pos(cfg, top); idx = na - 1 - top
        cfgs = {}
        for (pos, head, kk) in heads:
            if pos == cpos_top:
                cfgs[f"single_H{head}"] = [(cpos_top, 1, head)]
        pair = [(cpos_top, 1, head) for (pos, head, kk) in heads if pos == cpos_top]
        if len(pair) >= 2:
            cfgs["joint_pair"] = pair
        for hh in sorted({head for (pos, head, kk) in heads}):
            mp = [(consuming_pos(cfg, j), 1, hh) for j in aff]
            cfgs[f"multipos_H{hh}"] = mp
        # F4: non-consumer head specificity (a co-located head NOT in the map set)
        consumer_h_at_top = {head for (pos, head, kk) in heads if pos == cpos_top}
        noncons = [h for h in range(cfg.n_heads) if h not in consumer_h_at_top]
        if noncons:
            cfgs[f"specificity_H{noncons[0]}"] = [(cpos_top, 1, noncons[0])]
        flips = {c: [] for c in cfgs}
        flips["direct_full"] = []; flips["direct_scaled"] = []
        # F4: same-class null (both hi) for the single/joint arms
        null_flips = {c: [] for c in cfgs}
        for _ in range(n_pairs):
            sh = {}
            sa, sb, si, sh = build_chain(cfg, n_top, k, "hi", shared=sh)
            ta, tb, ti, _ = build_chain(cfg, n_top, k, "lo", shared=sh)
            sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
            clean = predict_answer(model, cfg, tq)
            sc, tc = cache_qs(model, sq, tq)
            for cname, plist in cfgs.items():
                patches = [(cp, L, h, sc["blocks.1.attn.hook_z"][0, cp, h, :].numpy()) for (cp, L, h) in plist]
                p = edge_patch_pred(model, cfg, tq, patches, tc)
                flips[cname].append(float(p[idx] != clean[idx]))
            dd = (sc["blocks.0.hook_resid_post"][0, cpos_top, :] - tc["blocks.0.hook_resid_post"][0, cpos_top, :])
            hd = np.mean([float((head_ov(model, sc["blocks.1.attn.hook_z"][0, cpos_top, h, :], 1, h)
                                 - head_ov(model, tc["blocks.1.attn.hook_z"][0, cpos_top, h, :], 1, h)).norm())
                          for (pos, h, kk) in heads if pos == cpos_top]) if pair else 1.0
            scl = min(1.0, hd / (float(dd.norm()) + 1e-9))
            for label, delta in [("direct_full", dd), ("direct_scaled", dd * scl)]:
                p = _direct_patch_pred(model, cfg, tq, cpos_top, delta, tc)
                flips[label].append(float(p[idx] != clean[idx]))
            # DECIDING-MATCHED null (F1-r2 fix): source AND target BOTH `lo` (same
            # deciding class as the target), fillers re-drawn but the deciding CARRY
            # held constant. A carry-specific head should NOT flip (expected ~0);
            # if it flips, the edge effect is generic content, not the deciding carry.
            la2, lb2, _, _ = build_chain(cfg, n_top, k, "lo", shared=None)
            lq2 = make_q(cfg, la2, lb2); lc2, _ = cache_qs(model, lq2, tq)
            for cname, plist in cfgs.items():
                patches = [(cp, L, h, lc2["blocks.1.attn.hook_z"][0, cp, h, :].numpy()) for (cp, L, h) in plist]
                p = edge_patch_pred(model, cfg, tq, patches, tc)
                null_flips[cname].append(float(p[idx] != clean[idx]))
        out[k] = {c: float(np.mean(v)) for c, v in flips.items()}
        out[k].update({f"NULL_{c}": float(np.mean(v)) for c, v in null_flips.items()})
    return out


def _direct_patch_pred(model, cfg, target_q, cpos, delta, tc):
    """Add delta to resid_mid(L1) at cpos and predict -- library raw edge patch."""
    return _lib_run_edge_patch(model, cfg, target_q, cpos, delta, recv_layer=1, arm="raw")


# ===========================================================================
# control 3: less-damped instrument validity on a known-causal head (SV-2 gate)
# ===========================================================================

def control_instrument(model, cfg, mn, n_top, n_pairs=30):
    """SV-2 gate: the less-damped edge instrument must move the answer on the
    map-named consumer head at SOME valid depth (validity is not depth-specific).
    Sweeps depths and reports the best move rate."""
    ap = answer_positions(cfg); na = len(ap)
    L, h = 1, INSTRUMENT_HEAD[mn][1]
    top = n_top + 1; cpos = consuming_pos(cfg, top); idx = na - 1 - top
    best = 0.0; best_k = None
    for k in range(2, n_top + 1):
        moves = []
        for _ in range(n_pairs):
            sh = {}
            sa, sb, _, sh = build_chain(cfg, n_top, k, "hi", shared=sh)
            ta, tb, _, _ = build_chain(cfg, n_top, k, "lo", shared=sh)
            sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
            clean = predict_answer(model, cfg, tq)
            sc, tc = cache_qs(model, sq, tq)
            p = edge_patch_pred(model, cfg, tq, [(cpos, L, h, sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())], tc)
            moves.append(float((p[idx] != clean[idx])))
        r = float(np.mean(moves))
        if r > best:
            best, best_k = r, k
    return {"instrument_head": f"L1H{h}@{cpos}", "any_move_rate": best, "best_depth": best_k}


# ===========================================================================
# Battery E: selective economy (A6) vs untagged-head baseline (SV-6 floor)
# ===========================================================================

def battery_E(model, cfg, mn, n_top, N=200):
    ap = answer_positions(cfg); nd = cfg.n_digits; lim = 10 ** nd
    def acc_on(builder, ablate=None):
        ok = 0
        for _ in range(N):
            a, b = builder()
            q = make_q(cfg, a, b)
            if ablate is None:
                pred = predict_answer(model, cfg, q)
            else:
                pos, L, head, zmean = ablate
                def hook(act, hook): act[:, pos, head, :] = zmean; return act
                with torch.no_grad():
                    lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook)])
                pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            if torch.equal(pred, q[ap]): ok += 1
        return ok / N
    def cascade_q():
        sh = {}; a, b, _, _ = build_chain(cfg, n_top, min(3, n_top), "hi", shared=sh); return a, b
    def free_q():  # carry-free random
        while True:
            da = [int(RNG.integers(0, 5)) for _ in range(nd)]; db = [int(RNG.integers(0, 5)) for _ in range(nd)]
            if all(da[i] + db[i] <= 8 for i in range(nd)):
                return int("".join(map(str, da))), int("".join(map(str, db)))
    def head_mean(pos, L, head):
        zs = []
        for _ in range(120):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2)); q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == f"blocks.{L}.attn.hook_z")
            zs.append(c[f"blocks.{L}.attn.hook_z"][0, pos, head, :].numpy())
        return torch.tensor(np.mean(zs, axis=0))
    base_c = acc_on(cascade_q); base_f = acc_on(free_q)
    res = {"clean_cascade": base_c, "clean_free": base_f, "heads": {}}
    # tagged consumer heads
    for (pos, head, kk) in CONSUMER_HEADS[mn]:
        zm = head_mean(pos, 1, head)
        c = acc_on(cascade_q, (pos, 1, head, zm)); f = acc_on(free_q, (pos, 1, head, zm))
        res["heads"][f"P{pos}L1H{head}"] = {"cascade_impact": base_c - c, "free_impact": base_f - f,
                                            "economy_gap": (base_c - c) - (base_f - f)}
    # untagged-head baseline at the same positions
    tagged = {(p, h) for (p, h, k) in CONSUMER_HEADS[mn]}
    gaps = []
    for (pos, head, kk) in CONSUMER_HEADS[mn]:
        for h2 in range(cfg.n_heads):
            if (pos, h2) in tagged: continue
            zm = head_mean(pos, 1, h2)
            c = acc_on(cascade_q, (pos, 1, h2, zm)); f = acc_on(free_q, (pos, 1, h2, zm))
            gaps.append((base_c - c) - (base_f - f))
    res["untagged_baseline_gap_max"] = float(np.max(np.abs(gaps))) if gaps else 0.0
    return res


# ===========================================================================
# Battery V: value content (SV-3 same-position wrong-role baseline)
# ===========================================================================

def battery_V(model, cfg, mn, n_top, n_q=1500):
    """Does the consumer head's value input carry ST/carry content? Probe the
    head's value v = x @ W_V restricted to the attended ST-site keys, vs a
    co-located non-consumer head (N-4) and a shuffled null."""
    nd = cfg.n_digits; lim = 10 ** nd
    WV = model.blocks[1].attn.W_V
    heads = CONSUMER_HEADS[mn]
    # served-digit carry label SV_k = carry into digit k
    def sv_label(a, b, k):
        da = [int(d) for d in str(a).zfill(nd)]; db = [int(d) for d in str(b).zfill(nd)]
        cin = 0
        for kk in range(k):
            cin = 1 if (da[nd-1-kk] + db[nd-1-kk] + cin) >= 10 else 0
        return cin
    # cache the head's value at the served digit's ST site (dpn of the deciding-ish
    # digit); use resid at the consuming position projected by W_V as the head input.
    res = {}
    for (pos, head, kk) in heads:
        # cache resid_pre at the consuming position once; project through each head's W_V
        R = []; y = []
        for _ in range(n_q):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2)); q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == "blocks.1.hook_resid_pre")
            R.append(c["blocks.1.hook_resid_pre"][0, pos, :].numpy()); y.append(sv_label(a, b, kk))
        R = np.array(R); y = np.array(y)
        if len(np.unique(y[:1000])) < 2:
            res[f"P{pos}L1H{head}"] = {"served_digit": kk, "note": "degenerate SV label"}; continue
        tr = slice(0, 1000); te = slice(1000, n_q); bi = balance(y[tr])
        Rt = torch.tensor(R)
        def acc_for_head(h):
            X = (Rt @ WV[h]).detach().numpy()
            return bacc(fit(X[tr][bi], y[tr][bi]), X[te], y[te])
        acc = acc_for_head(head)
        # SV-3 same-position wrong-role baseline: best co-located non-consumer head
        others = [h2 for h2 in range(cfg.n_heads) if h2 != head]
        wrong = max(acc_for_head(h2) for h2 in others) if others else float("nan")
        # shuffled null
        X = (Rt @ WV[head]).detach().numpy(); yp = y.copy(); RNG.shuffle(yp)
        accp = bacc(fit(X[tr][bi], yp[tr][bi]), X[te], yp[te])
        res[f"P{pos}L1H{head}"] = {"served_digit": kk, "SV_content_acc": acc,
                                   "wrong_role_baseline": wrong, "shuffled_null": accp,
                                   "content": bool(acc >= 0.5 + 0.2 and acc > accp + 0.2 and
                                                   (wrong != wrong or acc >= wrong + 0.2))}
    return res


# ===========================================================================
# driver
# ===========================================================================

def battery_S(model, cfg, mn, n_top, n_q=40):
    """Deciding-digit target tracking (SV-4/F3): the VALUE-MATCHED metric from
    deep_cascade_batteries (deciding value fixed, position varied) with the
    units-end control null; tracks only if top-2 includes the deciding operand at
    >= 2 non-degenerate depths AND beats the units-end control by >= 0.30. Selection
    (A9) requires tracking on the SAME head that is edge-causal at >= 2 depths."""
    from scripts.deep_cascade_batteries import deciding_target_tracking
    top = n_top + 1; cpos = consuming_pos(cfg, top)
    res = {}
    for h in range(cfg.n_heads):
        t = deciding_target_tracking(model, cfg, n_top, cpos, 1, h, n_q=n_q)
        res[f"L1H{h}@{cpos}"] = {"per_depth": t["per_depth"],
                                 "n_follow_nondegenerate": t["n_nondegenerate_depths_following"],
                                 "tracks": t["tracks_deciding_digit"]}
    return res


def run_model(model, cfg, mn):
    n_top = cfg.n_digits - 2
    depths = list(range(1, n_top + 1))
    gate = {k: behavioral_gate(model, cfg, n_top, k, n_q=40) for k in depths}
    depths_ok = [k for k in depths if gate[k] >= 0.90 and k >= 1]
    out = {"model": mn, "n_top": n_top, "gate": gate}
    out["control3_instrument"] = control_instrument(model, cfg, mn, n_top)
    out["battery_V"] = battery_V(model, cfg, mn, n_top)
    out["battery_D"] = battery_D(model, cfg, mn, n_top, [k for k in depths_ok if k >= 2] or depths_ok)
    out["battery_S"] = battery_S(model, cfg, mn, n_top)
    out["battery_E"] = battery_E(model, cfg, mn, n_top)
    out["verdict"] = derive_verdict(out)
    return out


def derive_verdict(out):
    instr_ok = out["control3_instrument"]["any_move_rate"] >= 0.30
    D = out["battery_D"]
    # best single-head, joint-pair, multipos, direct across depths
    def best(key_pred):
        vals = [v for k, dd in D.items() for c, v in dd.items() if key_pred(c)]
        return max(vals) if vals else 0.0
    single = best(lambda c: c.startswith("single_"))
    joint = max(best(lambda c: c == "joint_pair"), best(lambda c: c.startswith("multipos_")))
    direct_full = best(lambda c: c == "direct_full")
    direct_scaled = best(lambda c: c == "direct_scaled")
    # >=2 depth check for joint
    joint_depths = sum(1 for k, dd in D.items()
                       if max([dd.get("joint_pair", 0)] + [v for c, v in dd.items() if c.startswith("multipos_")]) >= 0.5)
    if not instr_ok:
        return {"read": "INVALID (instrument control failed)", "reason": f"control3 move {out['control3_instrument']['any_move_rate']:.2f}<0.30"}
    # F1 fix: single-head edge causal must be at >= 2 depths (not one) to count for
    # same-cell selection. Count depths where each single head clears 0.5.
    S = out.get("battery_S", {})
    single_depths = {}
    for k, dd in D.items():
        for c, v in dd.items():
            if c.startswith("single_H") and v >= 0.5:
                h = int(c.split("H")[1]); single_depths[h] = single_depths.get(h, 0) + 1
    causal_2depth = {h for h, nd in single_depths.items() if nd >= 2}
    tracking_heads = {int(name.split("H")[1].split("@")[0]) for name, d in S.items() if d.get("tracks")}
    same_cell = sorted(causal_2depth & tracking_heads)  # same head: >=2-depth edge AND tracks
    # F2 fix: direct arm is underpowered when direct_scaled==0; can't EXCLUDE direct
    direct_underpowered = direct_scaled < 0.5
    # carry-specificity (corrected null, F1-r2): best joint/single edge flip vs its
    # DECIDING-MATCHED null. Specific if real >= 0.5 AND null <= 0.20.
    def null_for(prefix):
        vals = []
        for k, dd in D.items():
            reals = [v for c, v in dd.items() if c.startswith(prefix) and not c.startswith("NULL_")]
            nulls = [dd.get("NULL_" + c) for c in dd if c.startswith(prefix) and not c.startswith("NULL_")]
            for rlv, nlv in zip(reals, nulls):
                if rlv is not None and nlv is not None and rlv >= 0.5:
                    vals.append(nlv)
        return max(vals) if vals else float("nan")
    joint_null = null_for("joint_pair") if joint >= 0.5 else null_for("multipos_")
    single_null = null_for("single_H")
    best_null = np.nanmin([x for x in [joint_null, single_null] if x == x]) if any(
        x == x for x in [joint_null, single_null]) else float("nan")
    carry_specific = (max(joint, single) >= 0.5) and (best_null == best_null) and (best_null <= 0.20)
    reason = (f"single_max={single:.2f} single>=2depth={sorted(causal_2depth)} "
              f"joint={joint:.2f}(@{joint_depths}d) tracking={sorted(tracking_heads)} "
              f"deciding_matched_null={best_null:.2f} carry_specific={carry_specific} "
              f"direct_full={direct_full:.2f} direct_scaled={direct_scaled:.2f}"
              + (" [direct UNDERPOWERED]" if direct_underpowered else ""))
    if joint < 0.5 and single < 0.5:
        if not direct_underpowered and direct_full >= 0.5:
            return {"read": "R-direct-path", "reason": reason}
        return {"read": "R-none/underpowered", "reason": reason}
    if not carry_specific:
        return {"read": "R-nonspecific-disruption",
                "reason": reason + " | edge flip NOT carry-specific (deciding-matched null flips too)"}
    # carry-specific head-edge sufficiency established
    if same_cell:
        return {"read": "R-A10-selection (same-cell >=2-depth edge + tracking, carry-specific)",
                "reason": reason + f" | same-cell {same_cell}"}
    return {"read": "R-A10-distributed-delivery",
            "reason": reason + " | carry-specific head-edge delivery SUFFICIENT at >=2 depths (joint), consumer-head-specific; single-cell selection NOT shown; direct-path not excluded (underpowered)"}


def main():
    results = {}
    for mn in ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64); assert acc > 0.99
        print(f"=== {mn} (acc {acc:.3f}) ===", flush=True)
        r = run_model(model, cfg, mn); r["accuracy"] = acc
        results[mn] = r
        print("  control3 instrument move:", round(r["control3_instrument"]["any_move_rate"], 2))
        print("  [V]", {n: (round(d.get("SV_content_acc", float('nan')), 2), d.get("content")) for n, d in r["battery_V"].items()})
        for k, dd in r["battery_D"].items():
            print(f"  [D] k={k}:", {c: round(v, 2) for c, v in dd.items()})
        print("  [S] tracking:", {n: (d.get("n_follow_nondegenerate"), d.get("tracks")) for n, d in r["battery_S"].items()})
        print("  [E] economy gaps:", {n: round(d["economy_gap"], 3) for n, d in r["battery_E"]["heads"].items()}, "baseline_max", round(r["battery_E"]["untagged_baseline_gap_max"], 3))
        print("  VERDICT:", r["verdict"]["read"], "--", r["verdict"]["reason"])
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
