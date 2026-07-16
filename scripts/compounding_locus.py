"""Compounding locus — decisive A11 (L0 relay) vs L1-read vs self-computation
(study-compounding-locus.md).

CE17 left the compounding locus R-mixed because (F2) single-site interchange nulls
are causally undetermined, and (Battery L) the consumer edge is local-class-
sufficient since local class and resolved carry CORRELATE on chain stimuli. This
study clears both via a DECORRELATION lever and (post-skeptic, LOC-1..LOC-6) a
relay-vs-self-computation discriminator.

Decorrelation lever (verified): a chain-ST site tagging digit j that is INSIDE the
999-run (deciding digit d < j) has local class fixed at U (sum=9) while the resolved
carry reaching it varies 0/1. Two kinds of decorrelated cell:
  - VISIBLE-decorrelated (m <= d < j): the site can SEE the deciding operands, so a
    carry decode there is neutral between A11-relay and "site sums what it sees".
  - INVISIBLE-decorrelated (d < m < j): the site CANNOT see the deciding digit, so a
    carry decode there can ONLY be a RELAYED carry (A11). THIS IS THE DISCRIMINATOR.

Batteries:
  DH  decorrelated horizon decode: decode carry_out(top) from the site L0 write on
      visible vs INVISIBLE decorrelated cells; baselines wrole + shuffled + a
      value-shuffled null (LOC-1); local-class control (must be ~0.5 = decorrelated);
      per-cell bootstrap CI (LOC-6).
  KO  cumulative knock-out (class-level causal): all-sufficient-relays ablate
      (necessity anchor), leave-deepest-in (at visible AND invisible depth, LOC-2),
      leave-shallowest-in; each with a deciding-matched specificity null (LOC-2).
  RC  reconstruction of the consumer edge on visible vs INVISIBLE decorrelated
      subsets separately (LOC-5).

CPU-only. Run:
    PYTHONPATH=. python3 scripts/compounding_locus.py all
    PYTHONPATH=. python3 scripts/compounding_locus.py all --fast
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
)
from scripts.deep_cascade_mechanism import (
    build_chain, consuming_pos, ak_pos, affected_digits, dn_pos, dpn_pos,
    behavioral_gate, RNG,
)
from scripts.compounding_arithmetic import (
    chain_st_sites, horizon_of_site, chain_carry_out, _site_ov_write,
    cache_full_L0,
)
from scripts.sv_implementation import (
    pair_at_top, twin_pair, same_class_twin, edge_contribution, lnfair_project,
    mean_ci, cache_full,
)
from scripts.sv_compounding import head_ov, edge_patch_pred
from scripts.st_tristate_geometry import build_class_question

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-compounding-locus")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fit(X, y):
    return LogisticRegression(max_iter=2000, C=0.5).fit(X, y)


def _bootstrap_bacc(X, y, n_boot=200, folds=3):
    """held-out bacc with a bootstrap CI over resampled train/test splits."""
    scores = []
    n = len(y)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        te = np.array([i for i in range(n) if i not in set(idx)])
        if len(te) < 5 or len(np.unique(y[idx])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = fit(X[idx], y[idx])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    if not scores:
        return {"bacc": float("nan"), "ci": [float("nan"), float("nan")], "n": n}
    return {"bacc": float(np.mean(scores)),
            "ci": [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))],
            "n": n}


# ---------------------------------------------------------------------------
# cell enumeration: visible vs invisible decorrelated
# ---------------------------------------------------------------------------

def decorrelated_cells(mn, cfg):
    """Per site, the visible-decorrelated (m<=d<j) and INVISIBLE-decorrelated
    (d<m<j) depths. Only sites with >=1 invisible cell can discriminate A11-relay."""
    n_top = cfg.n_digits - 2
    depths = [k for k in (2, 3, 4) if k <= n_top]
    out = []
    for s in chain_st_sites(mn):
        j = s["digit"]; m = s["m"]
        if j > n_top:
            continue
        vis, inv = [], []
        for k in depths:
            d = n_top - k
            if d >= j:      # deciding at/above j -> site NOT a 9-run cell (local class != U)
                continue
            (vis if d >= m else inv).append(k)
        if vis or inv:
            out.append({**s, "visible_depths": vis, "invisible_depths": inv,
                        "discriminating": len(inv) > 0})
    return out


# ---------------------------------------------------------------------------
# Battery DH: decorrelated horizon decode
# ---------------------------------------------------------------------------

def _decode_at_correlated(model, cfg, mn, s, WO0, n_q=200):
    """Positive control (4): at the CORRELATED depth (deciding digit == site digit,
    k=n_top-j), the site's OWN local class varies lo/hi and equals the carry. Can
    the tagged ST write decode it? If YES the instrument can read this write, so a
    decorrelated null is a real 'no relayed carry' finding; if NO the write is
    unreadable -> DH underpowered at this site."""
    n_top = cfg.n_digits - 2; j = s["digit"]
    k = n_top - j
    if k < 1 or k > n_top:
        return {"testable": False}
    X, y = [], []
    for _ in range(n_q):
        dc = "hi" if RNG.random() < 0.5 else "lo"
        a, b, info, _ = build_chain(cfg, n_top, k, dc, shared={})
        c = cache_full_L0(model, make_q(cfg, a, b))
        X.append(_site_ov_write(model, c, s["pos"], s["head"], WO0))
        y.append(chain_carry_out(cfg, a, b, n_top))
    X = np.array(X); y = np.array(y)
    if len(np.unique(y)) < 2:
        return {"testable": False}
    bb = _bootstrap_bacc(X, y)
    return {"testable": True, "k": k, "carry_bacc": bb["bacc"], "ci": bb["ci"],
            "instrument_can_read": bool(bb["bacc"] >= 0.7)}


def battery_DH(model, cfg, mn, cells, n_q=250):
    WO0 = model.blocks[0].attn.W_O
    ap = answer_positions(cfg); na = len(ap)
    n_top = cfg.n_digits - 2
    out = {}
    for s in cells:
        pos, head, j, m = s["pos"], s["head"], s["digit"], s["m"]
        key = f"P{pos}H{head}_d{j}_m{m}"
        wrole = [h for h in range(cfg.n_heads) if h != head][0]
        out[key] = {"m": m, "digit": j, "per_depth": {},
                    "instrument_control": _decode_at_correlated(model, cfg, mn, s, WO0,
                                                                 n_q=min(200, n_q))}
        for kind, ks in [("visible", s["visible_depths"]), ("invisible", s["invisible_depths"])]:
            for k in ks:
                d = n_top - k
                X, Xw, y, yloc, yopa = [], [], [], [], []
                for _ in range(n_q):
                    dc = "hi" if RNG.random() < 0.5 else "lo"
                    a, b, info, _ = build_chain(cfg, n_top, k, dc, shared={})
                    c = cache_full_L0(model, make_q(cfg, a, b))
                    X.append(_site_ov_write(model, c, pos, head, WO0))
                    Xw.append(_site_ov_write(model, c, pos, wrole, WO0))
                    y.append(chain_carry_out(cfg, a, b, n_top))
                    # local class at the site's own digit (should be U==constant here)
                    da = int(str(a).zfill(cfg.n_digits)[cfg.n_digits - 1 - j])
                    db = int(str(b).zfill(cfg.n_digits)[cfg.n_digits - 1 - j])
                    sm = da + db
                    yloc.append(0 if sm <= 8 else (1 if sm >= 10 else 2))
                    # F2 readability signal: the site's own first-operand value at this
                    # depth (a in 0..9) -- known-present, locally visible, varies even
                    # when local class is fixed at U. Binarize (a<5 vs a>=5).
                    yopa.append(int(da >= 5))
                X = np.array(X); Xw = np.array(Xw); y = np.array(y)
                yloc = np.array(yloc); yopa = np.array(yopa)
                r = {"depth": k, "d": d, "kind": kind}
                if len(np.unique(y)) < 2:
                    r["status"] = "degenerate"; out[key]["per_depth"][k] = r; continue
                # carry decode + baselines
                r["carry"] = _bootstrap_bacc(X, y)
                r["wrole_baseline"] = _bootstrap_bacc(Xw, y)
                ysh = y.copy(); RNG.shuffle(ysh)
                r["shuffled_null"] = _bootstrap_bacc(X, ysh)
                # LOC-6: value-shuffled null — shuffle the write rows, keep labels
                Xvs = X.copy(); RNG.shuffle(Xvs)
                r["value_shuffled_null"] = _bootstrap_bacc(Xvs, y)
                # local-class control (LOC-1 / control 2): must be ~0.5 (decorrelated)
                r["local_class_control"] = (_bootstrap_bacc(X, yloc)["bacc"]
                                            if len(np.unique(yloc)) >= 2 else 0.5)
                # F2 (post-result): per-depth READABILITY control -- decode a
                # known-present locally-visible signal (site's own operand a<5 vs >=5)
                # from the SAME write AT THIS DEPTH. If readable (>=0.7), the write is
                # non-degenerate here, so the chance carry-decode is a real 'no relay'
                # not a washed-out write; if chance too, the cell is UNDERPOWERED.
                rd = (_bootstrap_bacc(X, yopa) if len(np.unique(yopa)) >= 2
                      else {"bacc": float("nan")})
                r["depth_readability_bacc"] = rd["bacc"]
                r["depth_readable"] = bool(rd["bacc"] >= 0.7)
                base = max(r["wrole_baseline"]["bacc"], r["shuffled_null"]["bacc"],
                          r["value_shuffled_null"]["bacc"])
                # decodes iff carry bacc >= base+0.2 AND its CI lower > base (LOC-6)
                r["decodes"] = bool(r["carry"]["bacc"] >= base + 0.2 and r["carry"]["ci"][0] > base)
                out[key]["per_depth"][k] = r
    return out


# ---------------------------------------------------------------------------
# Battery KO: cumulative knock-out (class-level causal)
# ---------------------------------------------------------------------------

def _ablate_set_pred(model, cfg, q, ablate_sites, zmeans, ap):
    def hook(act, hook):
        for s in ablate_sites:
            act[:, s["pos"], s["head"], :] = zmeans[(s["pos"], s["head"])]
        return act
    with torch.no_grad():
        lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[("blocks.0.attn.hook_z", hook)])
    return lg[0, [p - 1 for p in ap]].argmax(-1)


def battery_KO(model, cfg, mn, cells, n_pairs=40):
    ap = answer_positions(cfg); na = len(ap)
    n_top = cfg.n_digits - 2
    top = n_top + 1; idx = na - 1 - top
    all_sites = chain_st_sites(mn)
    # mean z per ST head
    zmeans = {}
    for a, b in [(int(RNG.integers(0, 10 ** cfg.n_digits)), int(RNG.integers(0, 10 ** cfg.n_digits))) for _ in range(80)]:
        c = cache_full_L0(model, make_q(cfg, a, b))
        for s in all_sites:
            zmeans.setdefault((s["pos"], s["head"]), []).append(c["blocks.0.attn.hook_z"][0, s["pos"], s["head"], :].numpy())
    zmeans = {k: torch.tensor(np.mean(v, 0)) for k, v in zmeans.items()}
    # untagged baseline heads (co-located, not ST)
    st_ph = {(s["pos"], s["head"]) for s in all_sites}
    untag = []
    for s in all_sites:
        wr = [h for h in range(cfg.n_heads) if (s["pos"], h) not in st_ph]
        if wr:
            untag.append({"pos": s["pos"], "head": wr[0]})
            zmeans.setdefault((s["pos"], wr[0]), torch.tensor(
                np.mean([cache_full_L0(model, make_q(cfg, int(RNG.integers(0, 10 ** cfg.n_digits)),
                        int(RNG.integers(0, 10 ** cfg.n_digits))))["blocks.0.attn.hook_z"][0, s["pos"], wr[0], :].numpy()
                        for _ in range(20)], 0)))

    out = {"per_depth": {}}
    depths = sorted({k for s in cells for k in (s["visible_depths"] + s["invisible_depths"])})
    for k in depths:
        d = n_top - k
        # sufficient relays at this depth: chain-ST sites with m<=d (can resolve)
        suff = [s for s in all_sites if horizon_of_site(mn, s["pos"]) <= d and s["digit"] <= n_top]
        if not suff:
            continue
        by_m = sorted(suff, key=lambda s: horizon_of_site(mn, s["pos"]))
        deepest_m = max(horizon_of_site(mn, s["pos"]) for s in suff)
        shallow_m = min(horizon_of_site(mn, s["pos"]) for s in suff)
        deepest_set = [s for s in suff if horizon_of_site(mn, s["pos"]) == deepest_m]
        shallow_set = [s for s in suff if horizon_of_site(mn, s["pos"]) == shallow_m]
        arms = {
            "all_ablate": suff,
            "leave_deepest_in": [s for s in suff if s not in deepest_set],
            "leave_shallowest_in": [s for s in suff if s not in shallow_set],
        }
        # is this an invisible-decorrelated depth for the deepest survivor? (LOC-2)
        invis_depth = any(k in s["invisible_depths"] for s in cells)
        flips = {a: [] for a in arms}
        null_flips = {a: [] for a in arms}
        untag_flips = []
        for _ in range(n_pairs):
            (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
            for (a, b) in [(ha, hb), (la, lb)]:
                q = make_q(cfg, a, b); gold = predict_answer(model, cfg, q)
                for arm, sset in arms.items():
                    if not sset:
                        flips[arm].append(0.0); continue
                    pred = _ablate_set_pred(model, cfg, q, sset, zmeans, ap)
                    flips[arm].append(float(pred[idx] != gold[idx]))
                # untagged baseline: all-ablate on untagged co-located heads
                pu = _ablate_set_pred(model, cfg, q, untag, zmeans, ap)
                untag_flips.append(float(pu[idx] != gold[idx]))
            # LOC-2 specificity null: carry-free stimuli (no 9-run) — ablate same sets,
            # a carry-attributable break must be DIFFERENTIAL vs this.
            da = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
            db = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
            qa = int("".join(map(str, da))); qb = int("".join(map(str, db)))
            qn = make_q(cfg, qa, qb); goldn = predict_answer(model, cfg, qn)
            for arm, sset in arms.items():
                if not sset:
                    null_flips[arm].append(0.0); continue
                pn = _ablate_set_pred(model, cfg, qn, sset, zmeans, ap)
                null_flips[arm].append(float(pn[idx] != goldn[idx]))
        res = {"d": d, "invisible_decorrelated_depth": bool(invis_depth),
               "deepest_m": deepest_m, "n_sufficient": len(suff),
               "untagged_all_ablate_flip": mean_ci(untag_flips)}
        for arm in arms:
            f = mean_ci(flips[arm]); nf = mean_ci(null_flips[arm])
            res[arm] = {"flip": f, "specificity_null": nf,
                        "differential": f["rate"] - nf["rate"]}
        out["per_depth"][k] = res
    return out


# ---------------------------------------------------------------------------
# Battery RC: reconstruction on visible vs invisible decorrelated subsets
# ---------------------------------------------------------------------------

def battery_RC(model, cfg, mn, cells, n_q=250):
    cpos, heads, top = pair_at_top(cfg, mn)
    n_top = cfg.n_digits - 2
    from scripts.sv_implementation import carry_axis
    try:
        ax = carry_axis(model, cfg, mn, n_q=120)
    except Exception:
        ax = None
    out = {}
    for kind in ("visible", "invisible"):
        Xloc, Xhor, y = [], [], []
        # gather cells of this kind
        cell_depths = [(s, k) for s in cells for k in (s["visible_depths"] if kind == "visible" else s["invisible_depths"])]
        if not cell_depths:
            out[kind] = {"status": "no cells"}; continue
        for _ in range(n_q):
            s, k = cell_depths[int(RNG.integers(0, len(cell_depths)))]
            d = n_top - k
            dc = "hi" if RNG.random() < 0.5 else "lo"
            a, b, info, _ = build_chain(cfg, n_top, k, dc, shared={})
            q = make_q(cfg, a, b); c = cache_full(model, q)
            delta = edge_contribution(model, cfg, c, cpos, heads)
            rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
            if ax is not None:
                proj, _ = lnfair_project(model, cfg, rm, delta, ax["axis"])
            else:
                proj = float(delta.norm())
            y.append(proj)
            # phi_local: local class at the site (constant U on decorrelated -> degenerate)
            da = int(str(a).zfill(cfg.n_digits)[cfg.n_digits - 1 - s["digit"]])
            db = int(str(b).zfill(cfg.n_digits)[cfg.n_digits - 1 - s["digit"]])
            sm = da + db; lc = [0, 0, 0]; lc[0 if sm <= 8 else (1 if sm >= 10 else 2)] = 1
            Xloc.append(lc)
            # phi_horizon: resolved carry (only informative if relayed on invisible)
            Xhor.append(lc + [chain_carry_out(cfg, a, b, n_top)])
        Xloc = np.array(Xloc, float); Xhor = np.array(Xhor, float); y = np.array(y)
        def cv_r2(X):
            n = len(y); idx = np.arange(n); RNG.shuffle(idx)
            tr, te = idx[:3 * n // 4], idx[3 * n // 4:]
            if np.allclose(X[tr], X[tr][0]):   # degenerate constant features
                return 0.0
            return float(r2_score(y[te], LinearRegression().fit(X[tr], y[tr]).predict(X[te])))
        out[kind] = {"r2_local": cv_r2(Xloc), "r2_horizon": cv_r2(Xhor),
                     "delta_r2": cv_r2(Xhor) - cv_r2(Xloc), "n": len(y)}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_model(model, cfg, mn, fast=False):
    n_top = cfg.n_digits - 2
    cells = decorrelated_cells(mn, cfg)
    disc = [c for c in cells if c["discriminating"]]
    out = {"model": mn, "n_digits": cfg.n_digits, "n_top": n_top,
           "decorrelated_cells": [{"pos": c["pos"], "head": c["head"], "digit": c["digit"],
                                    "m": c["m"], "visible": c["visible_depths"],
                                    "invisible": c["invisible_depths"]} for c in cells],
           "n_discriminating_sites": len(disc)}
    nq = 120 if fast else 250
    npairs = 20 if fast else 40
    out["battery_DH"] = battery_DH(model, cfg, mn, cells, n_q=nq)
    out["battery_KO"] = battery_KO(model, cfg, mn, cells, n_pairs=npairs)
    out["battery_RC"] = battery_RC(model, cfg, mn, cells, n_q=nq)
    out["verdict"] = derive_verdict(out)
    return out


def derive_verdict(out):
    v = {}
    DH = out["battery_DH"]
    # invisible-decorrelated decode (the discriminator) + local-class control
    invis_decode = []; vis_decode = []; lc_controls = []
    corr_readable = {}   # site key -> reader works at correlated depth (control 4)
    for key, sd in DH.items():
        ic = sd.get("instrument_control", {})
        corr_readable[key] = bool(ic.get("instrument_can_read", False))
        for k, r in sd["per_depth"].items():
            if r.get("status") == "degenerate":
                continue
            lc_controls.append(r.get("local_class_control", float("nan")))
            # F2: per-depth readability gate — the write must be non-degenerate AT
            # THIS depth (decodes a known-present local signal) for a chance
            # carry-decode to mean 'no relay'.
            rec = (key, k, r["decodes"], r["carry"]["bacc"], bool(r.get("depth_readable")))
            if r["kind"] == "invisible":
                invis_decode.append(rec)
            else:
                vis_decode.append(rec)
    v["sites_correlated_readable"] = corr_readable
    # an invisible NON-decode only counts as 'no relay' if the write is readable AT
    # THAT invisible depth (F2); else that cell is underpowered.
    invis_readable = [(key, k, dec, b) for (key, k, dec, b, dr) in invis_decode if dr]
    v["n_invisible_readable_cells"] = len(invis_readable)
    v["n_invisible_underpowered_cells"] = sum(1 for (_, _, _, _, dr) in invis_decode if not dr)
    v["local_class_control_max"] = float(np.nanmax(lc_controls)) if lc_controls else float("nan")
    v["decorrelation_ok"] = bool(v["local_class_control_max"] < 0.65)   # control 2
    v["invisible_decode_cells"] = [(k, kk, dec, round(b, 2), dr) for (k, kk, dec, b, dr) in invis_decode]
    v["visible_decode_cells"] = [(k, kk, dec, round(b, 2), dr) for (k, kk, dec, b, dr) in vis_decode]
    any_vis_decode = any(dec for _, _, dec, _, _ in vis_decode)
    # only cells readable AT THEIR OWN depth can support a "no relay" conclusion
    any_invis_decode_readable = any(dec for (_, _, dec, _) in invis_readable)
    n_invis_testable = len(invis_readable)

    KO = out["battery_KO"]
    # invisible-depth leave-deepest-in survives differentially? all-ablate breaks?
    ko_invis = [d for d in KO["per_depth"].values() if d["invisible_decorrelated_depth"]]
    ko_all_break = any(d["all_ablate"]["differential"] > 0.3 for d in KO["per_depth"].values())
    ko_deepest_survives_invis = None
    if ko_invis:
        # F4: "survives" = leave-deepest flip is BOTH low (<0.3) AND well below the
        # all-ablate flip (<50% of it) -- so 6d k=3 (0.225 vs all 0.237) does NOT count.
        ko_deepest_survives_invis = all(
            d["leave_deepest_in"]["flip"]["rate"] < 0.3
            and d["leave_deepest_in"]["flip"]["rate"] < 0.5 * max(d["all_ablate"]["flip"]["rate"], 1e-9)
            for d in ko_invis)
    v["KO_all_ablate_breaks_differentially"] = ko_all_break
    v["KO_deepest_survives_at_invisible"] = ko_deepest_survives_invis

    RC = out["battery_RC"]
    rc_invis = RC.get("invisible", {})
    v["RC_invisible_delta_r2"] = rc_invis.get("delta_r2")

    # decision table (LOC-4), gated on the instrument control (4)
    if not v["decorrelation_ok"]:
        v["read"] = "INVALID: decorrelation control failed (local class decodes carry)"
    elif n_invis_testable == 0:
        v["read"] = ("inconclusive/UNDERPOWERED: no invisible-decorrelated cell whose "
                     "write is readable AT ITS OWN depth (F2 per-depth control) — DH "
                     "cannot separate relay from L1-read; A11 stays low")
    elif any_invis_decode_readable and (ko_deepest_survives_invis in (None, True)):
        v["read"] = ("A11-RELAY: carry decodes at INVISIBLE-decorrelated readable cell(s) "
                     "(can only be relayed) " + ("+ KO deepest survives" if ko_deepest_survives_invis else "(KO corroboration weak)"))
    elif (not any_invis_decode_readable) and any_vis_decode:
        v["read"] = ("SELF-COMPUTATION at L0: carry decodes where the site can SEE the "
                     "deciding digit (visible) but NOT at readable invisible cells — each "
                     "site sums what it sees; A11-relay REFUTED (not L1-read)")
    elif (not any_invis_decode_readable) and (not any_vis_decode):
        v["read"] = ("R-L1-read: no L0 site (readable) carries the resolved carry on "
                     "decorrelated stimuli — resolution is not at the ST writes")
    else:
        v["read"] = "final-mixed (report split)"
    # power cap
    if v["read"].startswith("A11-RELAY") and n_invis_testable <= 1:
        v["read"] += " [SINGLE-CELL, not replicated -> A11 relay-consistent trace, not CONFIRMED]"
    return v


def main():
    fast = "--fast" in sys.argv
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (n={cfg.n_digits}, acc {acc:.3f}) fast={fast} ===", flush=True)
        r = run_model(model, cfg, mn, fast=fast); r["accuracy"] = acc
        results[mn] = r
        v = r["verdict"]
        print(f"  decorrelated cells: {[(c['pos'],c['head'],'vis',c['visible'],'INV',c['invisible']) for c in r['decorrelated_cells']]}", flush=True)
        print(f"  decorrelation_ok={v['decorrelation_ok']} (lc_control_max={v['local_class_control_max']:.2f})", flush=True)
        print(f"  invisible decode cells: {v['invisible_decode_cells']}", flush=True)
        print(f"  visible decode cells:   {v['visible_decode_cells']}", flush=True)
        print(f"  KO all-ablate-breaks-diff={v['KO_all_ablate_breaks_differentially']} deepest-survives-invis={v['KO_deepest_survives_at_invisible']}", flush=True)
        print(f"  RC invisible dR2={v['RC_invisible_delta_r2']}", flush=True)
        print(f"  VERDICT: {v['read']}", flush=True)
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
