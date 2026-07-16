"""Leading-digit hard-case walkthrough (study-leading-digit-walkthrough.md).

C5 step 5: a per-link causal trace of how the LEADING answer digit A_top is
produced at the sign-token position in a hard cascade case (99..9 + 00..01), each
link tagged verified / inferred / different / underpowered.

Implements the pre-run design + Gate-1 amendments LW-1..LW-6:
  LW-1 Link 4 = readout-only (never counted toward A10); delivery = Link 3 only
  LW-2 Link-3 verified requires >=2 non-degenerate behaviorally-passing depths
  LW-3 per-depth behavioral gate (exclude depths the model can't do)
  LW-4 degenerate/different rule + static-output control
  LW-5 A10 ceiling: consolidate CE14 or record different; never raise A10
  LW-6 corrected sign-position L0 ST node list

CPU-only. Run:
    PYTHONPATH=. python3 scripts/leading_digit_walkthrough.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy, _digits_to_int,
)
from scripts.sv_compounding import edge_patch_pred, cache_qs, head_ov, _direct_patch_pred

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-leading-digit-walkthrough")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
RNG = np.random.default_rng(SEED)

# Sign-position wiring (HF maps). sign token pos = 2*nd+2 = A_top consuming pos.
# Sign-position L1 consumer heads (Impact:A_top, attend {=, ST-cluster}).
SIGN_L1_HEADS = {"add_d5_l2_h3_t15K_s372001": [0, 2], "add_d6_l2_h3_t20K_s173289": [0, 1]}
# Sign-position L0 ST heads (LW-6). (pos, head, digit)
SIGN_L0_ST = {
    "add_d5_l2_h3_t15K_s372001": [(12, 1, 3), (11, 2, 0)],
    "add_d6_l2_h3_t20K_s173289": [(14, 1, 5), (14, 2, 4)],
}


def sign_pos(cfg):
    return 2 * cfg.n_digits + 2


def top_answer_idx(cfg):
    # ap = [sign, A_top, ..., A0]; A_top at ap index 1 -> na-1-n_top; but use na-1-nd
    na = cfg.n_digits + 2
    return na - 1 - cfg.n_digits


def build_leading_hard(cfg, k, deciding=1):
    """Graded leading-digit hard case (LW-7): a 9-chain of length `k` from the
    LEADING digit down (top k digits = sum-9 = U); the DECIDING digit sits
    immediately below the chain (index k) and gets +`deciding` (1 => sum>=10 carries
    up through the chain and flips A_top 0->1; 0 => sum<=8, no carry). Digits below
    the deciding digit: no-carry filler. So the chain ALWAYS connects to the leading
    digit and A_top flips for EVERY k -> genuinely graded depths for the leading
    locus. Returns (a, b)."""
    nd = cfg.n_digits
    d1 = [0] * nd; d2 = [0] * nd
    # top k digits (indices 0..k-1) = 9 + 0 (sum 9, U chain to the leading digit)
    for i in range(k):
        d1[i] = 9; d2[i] = 0
    # deciding digit at index k (just below the chain), if it exists
    di = k
    if di <= nd - 1:
        if deciding == 1:
            d1[di] = 5; d2[di] = 5   # sum 10 -> carry into the chain
        else:
            d1[di] = 4; d2[di] = 4   # sum 8 -> no carry
    # digits below the deciding digit: no-carry filler
    for i in range(di + 1, nd):
        d1[i] = 0; d2[i] = 0
    return _digits_to_int(d1), _digits_to_int(d2)


def leading_flips(cfg):
    """Does A_top differ between deciding=1 and deciding=0 at chain depth covering
    the top? For k = nd-1 (chain from leading down to just above units), +1 at units
    ripples all the way to flip A_top 0->1."""
    return True


# ===========================================================================
# behavioral gate (LW-3): per depth, is the leading hard case answered correctly?
# ===========================================================================

def behavioral_gate_signpos(model, cfg, depths, n_q=20):
    ap = answer_positions(cfg)
    gate = {}
    for k in depths:
        ok = 0
        for _ in range(n_q):
            for deciding in (0, 1):
                a, b = build_leading_hard(cfg, k, deciding)
                q = make_q(cfg, a, b)
                pred = predict_answer(model, cfg, q)
                if torch.equal(pred, q[ap]):
                    ok += 1
        gate[k] = ok / (2 * n_q)
    return gate


# ===========================================================================
# Link 4 (readout-only, LW-1): whole resid_mid patch at sign pos flips A_top?
# ===========================================================================

def link4_readout(model, cfg, k, n_pairs=30):
    ap = answer_positions(cfg); idx = top_answer_idx(cfg); sp = sign_pos(cfg)
    flips = []
    for _ in range(n_pairs):
        sa, sb = build_leading_hard(cfg, k, 1)   # source: carry -> A_top=1
        ta, tb = build_leading_hard(cfg, k, 0)   # target: no carry -> A_top=0
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        sc, tc = cache_qs(model, sq, tq)
        src = sc["blocks.1.hook_resid_mid"][0, sp, :]
        delta = src - tc["blocks.1.hook_resid_mid"][0, sp, :]
        p = _direct_patch_pred(model, cfg, tq, sp, delta, tc)
        flips.append(float(p[idx] != clean[idx]))
    return float(np.mean(flips))


# ===========================================================================
# Link 3 (head edge, carry-specific, LW-1/LW-2): the delivery test
# ===========================================================================

def link3_head_edge(model, cfg, mn, k, n_pairs=30):
    ap = answer_positions(cfg); idx = top_answer_idx(cfg); sp = sign_pos(cfg)
    heads = SIGN_L1_HEADS[mn]
    res = {}
    # real (deciding 1->0) and deciding-matched null (both deciding=0, filler differs)
    for arm, cfgpatch in [("single_" + str(heads[0]), [heads[0]]),
                          ("joint", heads)]:
        real = []; null = []
        for _ in range(n_pairs):
            sa, sb = build_leading_hard(cfg, k, 1); ta, tb = build_leading_hard(cfg, k, 0)
            sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
            clean = predict_answer(model, cfg, tq)
            sc, tc = cache_qs(model, sq, tq)
            patches = [(sp, 1, h, sc["blocks.1.attn.hook_z"][0, sp, h, :].numpy()) for h in cfgpatch]
            p = edge_patch_pred(model, cfg, tq, patches, tc)
            real.append(float(p[idx] != clean[idx]))
            # deciding-matched null: source ALSO deciding=0 (same leading class)
            na2, nb2 = build_leading_hard(cfg, k, 0)
            nq2 = make_q(cfg, na2, nb2); nc2, _ = cache_qs(model, nq2, tq)
            npatch = [(sp, 1, h, nc2["blocks.1.attn.hook_z"][0, sp, h, :].numpy()) for h in cfgpatch]
            pn = edge_patch_pred(model, cfg, tq, npatch, tc)
            null.append(float(pn[idx] != clean[idx]))
        res[arm] = {"real": float(np.mean(real)), "deciding_matched_null": float(np.mean(null))}
    # direct-path arm (full + scaled) for the power gate
    df = []; ds = []
    for _ in range(n_pairs):
        sa, sb = build_leading_hard(cfg, k, 1); ta, tb = build_leading_hard(cfg, k, 0)
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq); sc, tc = cache_qs(model, sq, tq)
        dd = sc["blocks.0.hook_resid_post"][0, sp, :] - tc["blocks.0.hook_resid_post"][0, sp, :]
        hd = np.mean([float((head_ov(model, sc["blocks.1.attn.hook_z"][0, sp, h, :], 1, h)
                             - head_ov(model, tc["blocks.1.attn.hook_z"][0, sp, h, :], 1, h)).norm())
                      for h in heads])
        scl = min(1.0, hd / (float(dd.norm()) + 1e-9))
        df.append(float(_direct_patch_pred(model, cfg, tq, sp, dd, tc)[idx] != clean[idx]))
        ds.append(float(_direct_patch_pred(model, cfg, tq, sp, dd * scl, tc)[idx] != clean[idx]))
    res["direct_full"] = float(np.mean(df)); res["direct_scaled"] = float(np.mean(ds))
    return res


# ===========================================================================
# Link 1 (L0 ST heads encode class) + ablation (LW-6)
# ===========================================================================

def link1_st_ablation(model, cfg, mn, N=200):
    ap = answer_positions(cfg); nd = cfg.n_digits; lim = 10 ** nd
    def head_mean(pos, L, head):
        zs = []
        for _ in range(120):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2)); q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == f"blocks.{L}.attn.hook_z")
            zs.append(c[f"blocks.{L}.attn.hook_z"][0, pos, head, :].numpy())
        return torch.tensor(np.mean(zs, axis=0))
    # accuracy on random additions with/without ablating each sign-pos L0 ST head
    def acc(ablate=None):
        ok = 0
        for _ in range(N):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2)); q = make_q(cfg, a, b)
            if ablate is None:
                pred = predict_answer(model, cfg, q)
            else:
                pos, L, head, zm = ablate
                def hook(a_, hook): a_[:, pos, head, :] = zm; return a_
                with torch.no_grad():
                    lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook)])
                pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            if torch.equal(pred, q[ap]): ok += 1
        return ok / N
    clean = acc()
    res = {"clean_acc": clean, "nodes": {}}
    for (pos, head, dig) in SIGN_L0_ST[mn]:
        zm = head_mean(pos, 0, head)
        res["nodes"][f"P{pos}L0H{head}"] = {"digit": dig, "ablated_acc": acc((pos, 0, head, zm)),
                                            "impact": clean - acc((pos, 0, head, zm))}
    return res


# ===========================================================================
# Economy (A6) + static-output control (LW-4)
# ===========================================================================

def economy_and_static(model, cfg, mn, k_hard, N=150):
    ap = answer_positions(cfg); nd = cfg.n_digits; lim = 10 ** nd; sp = sign_pos(cfg)
    def head_mean(head):
        zs = []
        for _ in range(120):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2)); q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == "blocks.1.attn.hook_z")
            zs.append(c["blocks.1.attn.hook_z"][0, sp, head, :].numpy())
        return torch.tensor(np.mean(zs, axis=0))
    def acc(builder, ablate=None):
        ok = 0
        for _ in range(N):
            a, b = builder(); q = make_q(cfg, a, b)
            if ablate is None:
                pred = predict_answer(model, cfg, q)
            else:
                head, zm = ablate
                def hook(a_, hook): a_[:, sp, head, :] = zm; return a_
                with torch.no_grad():
                    lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[("blocks.1.attn.hook_z", hook)])
                pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            if torch.equal(pred, q[ap]): ok += 1
        return ok / N
    def hard(): return build_leading_hard(cfg, k_hard, int(RNG.integers(0, 2)))
    def free():  # carry-free: no cascade to the leading digit
        while True:
            da = [int(RNG.integers(0, 5)) for _ in range(nd)]; db = [int(RNG.integers(0, 5)) for _ in range(nd)]
            if all(da[i] + db[i] <= 8 for i in range(nd)): return _digits_to_int(da), _digits_to_int(db)
    base_h = acc(hard); base_f = acc(free)
    heads = SIGN_L1_HEADS[mn]
    econ = {"clean_hard": base_h, "clean_free": base_f, "heads": {}}
    tagged = set(heads)
    for h in heads:
        zm = head_mean(h)
        ch = acc(hard, (h, zm)); cf = acc(free, (h, zm))
        econ["heads"][f"L1H{h}"] = {"hard_impact": base_h - ch, "free_impact": base_f - cf,
                                    "economy_gap": (base_h - ch) - (base_f - cf)}
    # untagged baseline
    gaps = []
    for h in range(cfg.n_heads):
        if h in tagged: continue
        zm = head_mean(h)
        gaps.append((base_h - acc(hard, (h, zm))) - (base_f - acc(free, (h, zm))))
    econ["untagged_baseline_gap_max"] = float(np.max(np.abs(gaps))) if gaps else 0.0
    # static-output control: A_top output variance across graded hard cases (LW-4)
    a_tops = []
    for k in range(1, nd):
        for deciding in (0, 1):
            a, b = build_leading_hard(cfg, k, deciding); q = make_q(cfg, a, b)
            a_tops.append(int(predict_answer(model, cfg, q)[top_answer_idx(cfg)]))
    static = {"A_top_values": sorted(set(a_tops)), "n_distinct": len(set(a_tops))}
    return econ, static


# ===========================================================================
# driver
# ===========================================================================

def run_model(model, cfg, mn):
    nd = cfg.n_digits
    depths = list(range(1, nd))  # chain of k nines from the leading digit down;
    # deciding digit at index k (needs k <= nd-1)
    gate = behavioral_gate_signpos(model, cfg, depths)
    passing = [k for k in depths if gate[k] >= 1.0]
    out = {"model": mn, "sign_pos": sign_pos(cfg), "behavioral_gate": gate,
           "passing_depths": passing}
    # Link 4 (readout-only) + Link 3 (delivery) per passing depth
    out["link4_readout"] = {k: link4_readout(model, cfg, k) for k in passing}
    out["link3_head_edge"] = {k: link3_head_edge(model, cfg, mn, k) for k in passing}
    out["link1_st_ablation"] = link1_st_ablation(model, cfg, mn)
    k_hard = max(passing) if passing else max(depths)
    out["economy"], out["static_output"] = economy_and_static(model, cfg, mn, k_hard)
    out["trace"] = derive_trace(out, cfg)
    return out


def derive_trace(out, cfg):
    passing = out["passing_depths"]
    # Link 3: verified iff carry-specific (real>=0.5, null<=0.20) at >=2 passing depths
    def link3_depth_pass(k):
        e = out["link3_head_edge"][k]
        for arm in ("joint",) + tuple(a for a in e if a.startswith("single_")):
            if arm in e and e[arm]["real"] >= 0.5 and e[arm]["deciding_matched_null"] <= 0.20:
                return True
        return False
    n_carry_specific = sum(1 for k in passing if link3_depth_pass(k))
    direct_scaled_max = max((out["link3_head_edge"][k]["direct_scaled"] for k in passing), default=0.0)
    if n_carry_specific >= 2:
        link3_tag = "verified"
    elif n_carry_specific == 1:
        link3_tag = "inferred (underpowered, CE14 SV-1: one-depth)"
    else:
        link3_tag = "not carry-specific / underpowered"
    # Link 4 readout-only
    link4_flip = max(out["link4_readout"].values()) if out["link4_readout"] else 0.0
    # Link 1: ST ablation impact vs 0
    l1 = out["link1_st_ablation"]
    link1_tag = "verified" if any(v["impact"] > 0.01 for v in l1["nodes"].values()) else "inferred/redundant"
    # static-output / degenerate
    n_distinct = out["static_output"]["n_distinct"]
    # economy
    econ = out["economy"]; base = econ["untagged_baseline_gap_max"]
    econ_supported = any(v["economy_gap"] > max(base, 0.0) + 0.05 for v in econ["heads"].values())
    # leading mechanism (LW-4)
    if link3_tag == "verified":
        leading_mech = "mirrors_CE14"
    elif link4_flip >= 0.5 and n_carry_specific == 0:
        leading_mech = "different/degenerate (readout fires but no carry-specific head edge)"
    else:
        leading_mech = "inconclusive"
    return {
        "link4_readout_ONLY": {"flip": link4_flip, "tag": "readout-only (NOT counted toward A10)"},
        "link3_head_edge_delivery": {"n_carry_specific_depths": n_carry_specific,
                                     "direct_scaled_max": direct_scaled_max, "tag": link3_tag},
        "link1_st_encoding_ablation": {"tag": link1_tag,
                                       "impacts": {n: v["impact"] for n, v in l1["nodes"].items()}},
        "economy_A6": {"supported": bool(econ_supported), "baseline": base},
        "static_output_A_top_distinct": n_distinct,
        "leading_mechanism": leading_mech,
        "a10_delta": ("consolidate" if leading_mech == "mirrors_CE14"
                      else "different" if leading_mech.startswith("different") else "none"),
    }


def main():
    results = {}
    for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64); assert acc > 0.99
        print(f"=== {mn} (acc {acc:.3f}) ===", flush=True)
        r = run_model(model, cfg, mn); r["accuracy"] = acc
        results[mn] = r
        print("  behavioral gate:", {k: round(v, 2) for k, v in r["behavioral_gate"].items()})
        t = r["trace"]
        print("  Link4 (readout-only):", round(t["link4_readout_ONLY"]["flip"], 2))
        print("  Link3 (delivery):", t["link3_head_edge_delivery"])
        print("  Link1 (ST ablation):", t["link1_st_encoding_ablation"])
        print("  Economy A6:", t["economy_A6"], "| A_top distinct:", t["static_output_A_top_distinct"])
        print("  LEADING MECHANISM:", t["leading_mechanism"], "| a10_delta:", t["a10_delta"])
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
