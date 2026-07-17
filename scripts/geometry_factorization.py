"""Rail-and-address factorization of ST storage (study-geometry-factorization.md).

Scores agent conjectures G1 (rail-and-address factorization) and G4 (one shared
unit-adjust rail across ADD/SUB/NEG on the mixed model). Touches A3 (F1 write-site
manifold shape), A4/A8 (CE11 reinterpretation), C1/C2.

The central object is the RAIL PRE-IMAGE: the single per-model direction in
write-site residual space (resid_post(L0) at the ST-writer key position, = CE11's
Dpn_L0 read site) that the consumer head pair reads as the CE16 carry axis. Its
linearization (verified faithful to CE16 lnfair_project by the pre-launch gate):

    r = w1 (.) ( (W_V[1,h] @ W_O[1,h]) @ (w2 (.) chat) )   summed over pair heads h

with w1=blocks.1.ln1.w, w2=blocks.1.ln2.w, chat = CE16 carry axis at
blocks.1.ln2.hook_normalized (post-gamma). r is mean-centered (LN nuisance) and
unit-normed. It is a SINGLE shared direction applied identically at every digit
site -- refitting per site would defeat the point.

Batteries (addition models):
  F1  write-site manifold shape (collinear vs simplex; U off-rail height; cin-split)
  F2  OV-aligned transfer -- the G1 killer test (binary PRIMARY + tri-state parity;
      centered [train-only offset] + uncentered; raw = CE11 floor)
  F3  entanglement re-read (|cos|(r,SV), angle(r,ST-plane), pre/residual ST-SV angle)
  PC1 harness parity with CE11 (within-site diag >= 0.9; raw cross-site <= 0.2)
  PC3 wrong-axis nulls (variance-matched random; OV-pre-image-of-random [headline];
      random-in-ST-subspace)

Mixed model:
  F4  rail identity across ADD/SUB/NEG at the shared combiner input (pairwise |cos|;
      variance explained by one shared direction; |cos| to SGN axis) + resid_pre arm
  PC2 binary carry/borrow/neg-borrow decodability at combiner input vs untrained

CPU-only. Run:
    PYTHONPATH=. python3 scripts/geometry_factorization.py smoke   # d6 only, small n
    PYTHONPATH=. python3 scripts/geometry_factorization.py add     # both addition models
    PYTHONPATH=. python3 scripts/geometry_factorization.py mixed   # mixed model F4/PC2
    PYTHONPATH=. python3 scripts/geometry_factorization.py all
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from scripts.confirm_st_node import load_model, make_q, verify_accuracy
from scripts.st_tristate_geometry import CFG as GEO_CFG, build_class_question
from scripts.sv_compounding import CONSUMER_HEADS
from quanta_maths.maths_probe import (
    collect_site_activations, fit_probe as _lib_fit_probe, probe_balanced_accuracy,
    class_mean_subspace, principal_angles_deg, balance_idx, train_test_split_idx,
    last_layer,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-geometry-factorization")
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 20260716
ADD_MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]  # CE11 parity
MIXED_MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
C_REG = 0.5
COMBINER_HOOK = "blocks.1.ln2.hook_normalized"  # addition combiner input (L1)


def fit(X, y):
    return _lib_fit_probe(X, y, C=C_REG)


def bacc(clf, X, y):
    return probe_balanced_accuracy(clf, X, y)


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


# ===========================================================================
# CE16 carry axis + rail pre-image
# ===========================================================================

def carry_axis(model, cfg, mn, rng, n_q=250):
    """Committed-carry axis c1 - c0 at the L1 combiner input (CE6/CE16)."""
    gc = GEO_CFG[mn]
    n = gc["digit"]; pos = gc["combiner_pos"]
    means = {}
    for cls in ("c0", "c1"):
        acts = []
        for _ in range(n_q):
            a, b, _sa = build_class_question(cfg, n, cls)
            q = make_q(cfg, a, b)
            with torch.no_grad():
                _, c = model.run_with_cache(
                    q.unsqueeze(0), names_filter=lambda nm: nm == COMBINER_HOOK)
            acts.append(c[COMBINER_HOOK][0, pos, :].numpy())
        means[cls] = np.mean(acts, 0)
    axis = unit(means["c1"] - means["c0"])
    sep = float((means["c1"] - means["c0"]) @ axis)
    return {"axis": axis, "sep": sep, "digit": n, "combiner_pos": pos}


def head_preimage(model, chat, head, w1, w2, layer=1):
    """OV pre-image of chat for one L1 head, in write-site (resid_post L0) space.
    r_h = w1 (.) ( (W_V @ W_O) @ (w2 (.) chat) )  -- see module docstring."""
    WV = model.blocks[layer].attn.W_V[head].detach().numpy()  # [d_model, d_head]
    WO = model.blocks[layer].attn.W_O[head].detach().numpy()  # [d_head, d_model]
    WVO = WV @ WO                                              # [d_model, d_model]
    return w1 * (WVO @ (w2 * chat))


def build_rail(model, cfg, mn, chat, layer=1):
    """Shared rail pre-image r (centered, unit) + per-head diagnostics (cond 1a/1d)."""
    w1 = model.blocks[layer].ln1.w.detach().numpy()
    w2 = model.blocks[layer].ln2.w.detach().numpy()
    heads = sorted({h for (p, h, k) in CONSUMER_HEADS[mn]})
    per_head = {h: head_preimage(model, chat, h, w1, w2, layer) for h in heads}
    # pairwise cos among per-head pre-images (cond 1d)
    pair_cos = {}
    hs = list(per_head)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            a, b = per_head[hs[i]], per_head[hs[j]]
            pair_cos[f"H{hs[i]}-H{hs[j]}"] = float(unit(a) @ unit(b))
    r = np.sum([per_head[h] for h in heads], axis=0)
    r = r - r.mean()          # cond 1a: remove LN mean-subtraction nuisance
    r = unit(r)
    return {"r": r, "heads": heads,
            "per_head": {h: unit(v).tolist() for h, v in per_head.items()},
            "per_head_pair_cos": pair_cos}


# ===========================================================================
# transfer (1-D coordinate) with train-only per-site offset centering
# ===========================================================================

def _split(n, rng, frac=0.7):
    return train_test_split_idx(n, rng, frac=frac)


def transfer_1d(coord_by_site, y_by_site, digits, chance, rng, center=True,
                classes=None):
    """Retention on a 1-D coordinate. Trains a probe on coord at site i, tests at
    site j; per-site offset estimated on TRAIN split only (cond 2a). Returns
    diag/offdiag mean balanced accuracy + chance-relative retention gain."""
    probes = {}; te_idx = {}; mu = {}
    for n in digits:
        x = coord_by_site[n].reshape(-1, 1); y = y_by_site[n]
        if classes is not None:
            m = np.isin(y, classes)
            x, y = x[m], y[m]
        tr, te = _split(len(y), rng)
        mu[n] = float(x[tr].mean()) if center else 0.0
        te_idx[n] = (x, y, te)
        bi = balance_idx(y[tr], rng)
        probes[n] = fit((x[tr] - mu[n])[bi], y[tr][bi])
    diag, off = [], []
    for i in digits:
        for j in digits:
            xj, yj, te = te_idx[j]
            acc = bacc(probes[i], xj[te] - mu[j], yj[te])
            (diag if i == j else off).append(acc)
    diagm = float(np.mean(diag)); offm = float(np.mean(off))
    ret = (offm - chance) / (diagm - chance) if diagm > chance + 1e-6 else float("nan")
    return {"diag": diagm, "offdiag": offm, "retention_gain": ret,
            "diag_gain": diagm - chance, "offdiag_gain": offm - chance}


def transfer_full(acts_by_site, y_by_site, digits, chance, rng, classes=None,
                  center=True):
    """FULL-activation probe transfer (CE11 style). ``classes`` restricts labels
    (e.g. [0,1] for the binary-carry-bit transfer, [0,1,2] for CE11 tri-state).
    ``center`` subtracts the per-site TRAIN-split mean (CE11 ``matrix_centered``);
    ``center=False`` is the CE11 ``matrix_raw`` floor. Offset from TRAIN only."""
    probes = {}; te_idx = {}; mu = {}
    for n in digits:
        X = acts_by_site[n]; y = y_by_site[n]
        if classes is not None:
            m = np.isin(y, classes); X = X[m]; y = y[m]
        tr, te = _split(len(y), rng)
        mu[n] = X[tr].mean(0) if center else np.zeros(X.shape[1])
        te_idx[n] = (X, y, te)
        bi = balance_idx(y[tr], rng)
        probes[n] = fit((X - mu[n])[tr][bi], y[tr][bi])
    diag, off = [], []
    for i in digits:
        for j in digits:
            X, y, te = te_idx[j]
            acc = bacc(probes[i], (X - mu[j])[te], y[te])
            (diag if i == j else off).append(acc)
    diagm = float(np.mean(diag)); offm = float(np.mean(off))
    ret = (offm - chance) / (diagm - chance) if diagm > chance + 1e-6 else float("nan")
    return {"diag": diagm, "offdiag": offm, "retention_gain": ret,
            "diag_gain": diagm - chance, "offdiag_gain": offm - chance}


def per_digit_axis_cos(Xs, y_by_site, digits, classes=(0, 1)):
    """Pairwise |cos| between the per-digit binary class-mean (carry) axes.
    High = the carry direction is shared across positions; low = position-specific."""
    import itertools
    axes = {}
    for n in digits:
        y = y_by_site[n]
        a = Xs[n][y == classes[1]].mean(0) - Xs[n][y == classes[0]].mean(0)
        axes[n] = unit(a)
    cos = {f"D{i}-D{j}": float(abs(axes[i] @ axes[j]))
           for i, j in itertools.combinations(digits, 2)}
    return {"pairwise_abs_cos": cos, "mean_abs_cos": float(np.mean(list(cos.values())))}


# ===========================================================================
# F1 write-site manifold shape (A3 descriptive)
# ===========================================================================

def battery_F1(model, cfg, mn, rng, r, n_q=300):
    """Per strong-writer digit: tri-state centroid geometry on the write site.
    U off-rail height / (0-1 base) with permutation null; cin-split of U."""
    gc = GEO_CFG[mn]
    nd = cfg.n_digits
    digits = list(range(1, nd - 1))
    hook = "blocks.0.hook_resid_post"
    out = {}
    for n in digits:
        means = {}; pooled = {}
        for cls in ("c0", "c1", "u0", "u1"):
            A = []
            for _ in range(n_q):
                a, b, _sa = build_class_question(cfg, n, cls)
                q = make_q(cfg, a, b)
                pos = 2 * nd - n  # ddn position (D'n), matches Dpn site
                with torch.no_grad():
                    _, c = model.run_with_cache(
                        q.unsqueeze(0), names_filter=lambda nm: nm == hook)
                A.append(c[hook][0, pos, :].numpy())
            A = np.array(A); pooled[cls] = A; means[cls] = A.mean(0)
        c0, c1 = means["c0"], means["c1"]
        umean = 0.5 * (means["u0"] + means["u1"])
        axis = c1 - c0; axlen = np.linalg.norm(axis) + 1e-9; axhat = axis / axlen
        v = umean - c0
        along = float(v @ axhat)
        perp = v - along * axhat
        off_frac = float(np.linalg.norm(perp) / axlen)
        # permutation null for off-axis fraction
        labs = np.array(sum([[c] * len(pooled[c]) for c in pooled], []))
        X = np.vstack([pooled[c] for c in ("c0", "c1", "u0", "u1")])
        ge = 0; NP = 500
        for _ in range(NP):
            perm = rng.permutation(labs)
            mm = {c: X[perm == c].mean(0) for c in ("c0", "c1", "u0", "u1")}
            ax2 = mm["c1"] - mm["c0"]; al2 = np.linalg.norm(ax2) + 1e-9; ah2 = ax2 / al2
            um2 = 0.5 * (mm["u0"] + mm["u1"]); vv = um2 - mm["c0"]
            pp = vv - (vv @ ah2) * ah2
            if np.linalg.norm(pp) / al2 >= off_frac:
                ge += 1
        # cin-split of U on the rail (u0 vs u1 rail coordinate, class means)
        rail_u0 = float((pooled["u0"] @ r).mean()); rail_u1 = float((pooled["u1"] @ r).mean())
        rail_c0 = float((pooled["c0"] @ r).mean()); rail_c1 = float((pooled["c1"] @ r).mean())
        out[n] = {
            "u_offaxis_frac": off_frac, "offaxis_perm_p": (ge + 1) / (NP + 1),
            "u_along_frac": float(along / axlen),
            "d_c0_c1": float(np.linalg.norm(c1 - c0)),
            "rail_c0": rail_c0, "rail_c1": rail_c1,
            "rail_u0": rail_u0, "rail_u1": rail_u1,
            "u_cin_split_on_rail": rail_u1 - rail_u0,
            "shape": "simplex(U off-rail)" if (ge + 1) / (NP + 1) < 0.05 and off_frac > 0.25
                     else "collinear(ordered)",
        }
    return out


# ===========================================================================
# F2 + F3 + PC1 + PC3
# ===========================================================================

def collect_transfer_data(model, cfg, rng, n_q):
    nd = cfg.n_digits
    digits = list(range(1, nd - 1))
    acts, labs = collect_site_activations(model, cfg, n_q, digits, ["Dpn_L0"], rng)
    Xs = {n: acts[("Dpn_L0", n)] for n in digits}
    ST = {n: labs["ST"][n] for n in digits}
    SV = {n: labs["SV"][n] for n in digits}
    return digits, Xs, ST, SV


def battery_F2(model, cfg, mn, rng, rail, digits, Xs, ST):
    r = rail["r"]
    coord = {n: Xs[n] @ r for n in digits}
    res = {}
    # PRIMARY: binary ST0-vs-ST1 retention on the rail (cond 3c)
    res["rail_binary_centered"] = transfer_1d(coord, ST, digits, 0.5, rng,
                                              center=True, classes=[0, 1])
    res["rail_binary_uncentered"] = transfer_1d(coord, ST, digits, 0.5, rng,
                                                center=False, classes=[0, 1])
    # tri-state parity on the rail
    res["rail_tristate_centered"] = transfer_1d(coord, ST, digits, 1/3, rng, center=True)
    res["rail_tristate_uncentered"] = transfer_1d(coord, ST, digits, 1/3, rng, center=False)
    # RAW full-activation tri-state (CE11 reproduction = the floor). CE11 reports the
    # UNCENTERED (matrix_raw) retention (~0.10-0.12); centered is the offset-removed arm.
    res["raw_tristate"] = transfer_full(Xs, ST, digits, 1/3, rng, classes=[0, 1, 2],
                                        center=False)
    res["raw_tristate_centered"] = transfer_full(Xs, ST, digits, 1/3, rng,
                                                  classes=[0, 1, 2], center=True)
    # HONEST G1-spirit test: does ANY shared linear direction transfer? full-activation
    # BINARY carry-bit transfer (CE11 only tested tri-state), uncentered + centered.
    res["full_binary_transfer"] = transfer_full(Xs, ST, digits, 0.5, rng, classes=[0, 1],
                                                center=False)
    res["full_binary_transfer_centered"] = transfer_full(Xs, ST, digits, 0.5, rng,
                                                         classes=[0, 1], center=True)
    # is the carry DIRECTION shared across positions? per-digit binary-axis alignment
    res["per_digit_binary_axis_alignment"] = per_digit_axis_cos(Xs, ST, digits)
    res["diag_weak_flag"] = bool(res["rail_binary_centered"]["diag_gain"] < 0.10)
    return res


def battery_F3(model, cfg, rng, rail, digits, Xs, ST, SV):
    """Entanglement re-read (redesigned per gate cond 4a-4c)."""
    r = rail["r"]
    mid = digits[len(digits) // 2]
    X = Xs[mid]; yst = ST[mid]; ysv = SV[mid]
    Bst = class_mean_subspace(X, yst)      # ST subspace (<=2-D)
    Bsv = class_mean_subspace(X, ysv)      # SV subspace (1-D, binary)
    st_rank = int(Bst.shape[0]); sv_rank = int(Bsv.shape[0])
    ang = principal_angles_deg(Bst, Bsv)
    pre_angle = float(np.max(ang)) if ang.size else float("nan")
    # label-correlation null (as CE11/probe_transfer)
    yb = ysv.copy()
    for cl in np.unique(yst):
        m = yst == cl
        yb[m] = rng.permutation(yb[m])
    Bnull = class_mean_subspace(X, yb)
    angn = principal_angles_deg(Bst, Bnull)
    null_angle = float(np.max(angn)) if angn.size else float("nan")
    # (4a) |cos| of rail to the SV carry axis, and angle of rail into ST plane
    sv_axis = unit(X[ysv == 1].mean(0) - X[ysv == 0].mean(0))
    cos_r_sv = float(abs(unit(r) @ sv_axis))
    rmat = unit(r).reshape(1, -1)
    a_rst = principal_angles_deg(rmat, Bst)
    angle_r_to_st = float(np.min(a_rst)) if a_rst.size else float("nan")
    # (4b) residual angle: remove r from ST only, vs full SV, recomputed null
    rhat = unit(r)
    Xperp = X - np.outer(X @ rhat, rhat)
    Bst_res = class_mean_subspace(Xperp, yst)
    res_rank = int(Bst_res.shape[0])
    a_res = principal_angles_deg(Bst_res, Bsv)
    residual_angle = float(np.max(a_res)) if a_res.size else float("nan")
    yb2 = ysv.copy()
    for cl in np.unique(yst):
        m = yst == cl
        yb2[m] = rng.permutation(yb2[m])
    Bnull_res = class_mean_subspace(Xperp, yb2)
    a_resn = principal_angles_deg(Bst_res, Bnull_res)
    residual_null = float(np.max(a_resn)) if a_resn.size else float("nan")
    return {"digit": mid, "st_rank": st_rank, "sv_rank": sv_rank,
            "pre_removal_ST_SV_angle": pre_angle, "label_null_angle": null_angle,
            "cos_rail_to_SV_axis": cos_r_sv, "angle_rail_into_ST_plane": angle_r_to_st,
            "residual_ST_rank": res_rank,
            "residual_ST_SV_angle": residual_angle,
            "residual_label_null": residual_null}


def _matched_variance_random(rng, d, target_var, X):
    v = unit(rng.standard_normal(d))
    cur = float((X @ v).var())
    return v * np.sqrt(target_var / (cur + 1e-12))


def battery_PC3(model, cfg, mn, rng, rail, digits, Xs, ST):
    """Wrong-axis nulls: must NOT rescue binary transfer (cond 5a/5b/5c)."""
    r = rail["r"]; mid = digits[len(digits) // 2]
    d = Xs[mid].shape[1]
    target_var = float((Xs[mid] @ r).var())
    out = {}
    # 5a variance-matched random unit
    vr = _matched_variance_random(rng, d, target_var, Xs[mid])
    coord = {n: Xs[n] @ vr for n in digits}
    out["random_var_matched"] = transfer_1d(coord, ST, digits, 0.5, rng,
                                            center=True, classes=[0, 1])
    # 5b OV-pre-image of a RANDOM axis (headline: same OV/LN geometry as r)
    w1 = model.blocks[1].ln1.w.detach().numpy()
    w2 = model.blocks[1].ln2.w.detach().numpy()
    rand_chat = unit(rng.standard_normal(d))
    heads = rail["heads"]
    rr = np.sum([head_preimage(model, rand_chat, h, w1, w2) for h in heads], axis=0)
    rr = unit(rr - rr.mean())
    coord = {n: Xs[n] @ rr for n in digits}
    out["ov_preimage_random_axis"] = transfer_1d(coord, ST, digits, 0.5, rng,
                                                 center=True, classes=[0, 1])
    # 5c random direction within the ST class-mean subspace
    Bst = class_mean_subspace(Xs[mid], ST[mid])
    if Bst.shape[0] >= 1:
        w = rng.standard_normal(Bst.shape[0])
        vin = unit(w @ Bst)
        coord = {n: Xs[n] @ vin for n in digits}
        out["random_in_ST_subspace"] = transfer_1d(coord, ST, digits, 0.5, rng,
                                                    center=True, classes=[0, 1])
    return out


# ===========================================================================
# addition-model driver
# ===========================================================================

def run_addition_model(mn, rng, n_q, f1_nq):
    print(f"=== ADDITION {mn} ===", flush=True)
    model, cfg = load_model(mn)
    acc = verify_accuracy(model, cfg, n=64)
    assert acc > 0.99, f"{mn} acc {acc}"
    print(f"  [1/6] acc={acc:.3f}; fitting carry axis", flush=True)
    ax = carry_axis(model, cfg, mn, rng)
    print(f"        carry-axis sep={ax['sep']:.2f}", flush=True)
    print("  [2/6] building rail pre-image", flush=True)
    rail = build_rail(model, cfg, mn, ax["axis"])
    print(f"        heads={rail['heads']} per-head pair-cos={rail['per_head_pair_cos']}", flush=True)
    print("  [3/6] collecting transfer data (Dpn_L0)", flush=True)
    digits, Xs, ST, SV = collect_transfer_data(model, cfg, rng, n_q)
    print("  [4/6] F2 OV-aligned transfer (the G1 killer test)", flush=True)
    f2 = battery_F2(model, cfg, mn, rng, rail, digits, Xs, ST)
    print(f"        raw tri retention={f2['raw_tristate']['retention_gain']:.3f} "
          f"(CE11 floor ~0.12); rail(OV-preimage)-binary offdiag_gain="
          f"{f2['rail_binary_centered']['offdiag_gain']:.3f} diag_gain="
          f"{f2['rail_binary_centered']['diag_gain']:.3f}", flush=True)
    fb = f2["full_binary_transfer"]
    print(f"        full-act BINARY carry transfer: diag={fb['diag']:.3f} offdiag={fb['offdiag']:.3f} "
          f"retention={fb['retention_gain']:.3f} | per-digit axis mean|cos|="
          f"{f2['per_digit_binary_axis_alignment']['mean_abs_cos']:.2f}", flush=True)
    print("  [5/6] F3 entanglement re-read + PC3 nulls", flush=True)
    f3 = battery_F3(model, cfg, rng, rail, digits, Xs, ST, SV)
    print(f"        pre ST-SV angle={f3['pre_removal_ST_SV_angle']:.1f} "
          f"|cos(r,SV)|={f3['cos_rail_to_SV_axis']:.2f} "
          f"angle(r,ST-plane)={f3['angle_rail_into_ST_plane']:.1f} "
          f"residual ST-SV={f3['residual_ST_SV_angle']:.1f}/null={f3['residual_label_null']:.1f}", flush=True)
    pc3 = battery_PC3(model, cfg, mn, rng, rail, digits, Xs, ST)
    for k, v in pc3.items():
        print(f"        PC3 {k}: offdiag_gain={v['offdiag_gain']:.3f} diag_gain={v['diag_gain']:.3f} "
              f"(rail offdiag_gain={f2['rail_binary_centered']['offdiag_gain']:.3f})", flush=True)
    print("  [6/6] F1 write-site manifold shape", flush=True)
    f1 = battery_F1(model, cfg, mn, rng, rail["r"], n_q=f1_nq)
    for n, v in f1.items():
        print(f"        D{n}: off-frac={v['u_offaxis_frac']:.2f} p={v['offaxis_perm_p']:.3f} "
              f"cin-split={v['u_cin_split_on_rail']:.2f} -> {v['shape']}", flush=True)
    del model
    return {"model": mn, "accuracy": acc, "carry_axis_sep": ax["sep"],
            "rail": {"heads": rail["heads"], "per_head_pair_cos": rail["per_head_pair_cos"]},
            "F1_manifold_shape": f1, "F2_transfer": f2, "F3_entanglement": f3,
            "PC3_wrong_axis_nulls": pc3}


# ===========================================================================
# F4 mixed rail identity + PC2
# ===========================================================================

def _mixed_axis_at(model, cfg, cls, read_digit, hook, rng, n_q, want_bit):
    """Carry-bit axis from NATURAL random questions (cond 6a): read combiner input
    at the consuming position of read_digit, split by resolved SV/MV/NV bit."""
    from scripts.mixed_sv import class_question, class_labels, to_q
    from quanta_maths.maths_edge_patch import consuming_pos
    cpos = consuming_pos(cfg, read_digit)
    acts = {0: [], 1: []}
    tries = 0
    while (len(acts[0]) < n_q or len(acts[1]) < n_q) and tries < n_q * 20:
        tries += 1
        a, b = class_question(cfg, rng, cls)
        SA, ST, SV = class_labels(cfg, a, b, cls)
        bit = int(SV[read_digit])
        if len(acts[bit]) >= n_q:
            continue
        q = to_q(cfg, a, b, cls)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
        acts[bit].append(c[hook][0, cpos, :].numpy())
    if len(acts[0]) < 20 or len(acts[1]) < 20:
        return None, None
    A0 = np.array(acts[0]); A1 = np.array(acts[1])
    axis = A1.mean(0) - A0.mean(0)
    return unit(axis), (A0, A1)


def _decode_bit(A0, A1, rng):
    X = np.vstack([A0, A1]); y = np.r_[np.zeros(len(A0)), np.ones(len(A1))].astype(int)
    tr, te = _split(len(y), rng)
    bi = balance_idx(y[tr], rng)
    clf = fit(X[tr][bi], y[tr][bi])
    return bacc(clf, X[te], y[te])


def run_mixed_model(rng, n_q):
    print(f"=== MIXED {MIXED_MODEL} ===", flush=True)
    from quanta_maths import load_maths_model_from_hf, make_untrained_control
    model, cfg = load_maths_model_from_hf(MIXED_MODEL, device="cpu")
    ll = last_layer(cfg)
    hook = f"blocks.{ll}.ln2.hook_normalized"
    hook_pre = f"blocks.{ll}.hook_resid_pre"
    read_digit = 2
    print(f"  n_digits={cfg.n_digits} last_layer={ll}", flush=True)
    axes = {}; axes_pre = {}; decode = {}
    ctrl = make_untrained_control(cfg)
    ctrl_decode = {}
    for cls in ("ADD", "SUB", "NEG"):
        print(f"  [{cls}] fitting carry-bit axis (combiner input + resid_pre)", flush=True)
        ax, pools = _mixed_axis_at(model, cfg, cls, read_digit, hook, rng, n_q, 1)
        axp, _ = _mixed_axis_at(model, cfg, cls, read_digit, hook_pre, rng, n_q, 1)
        if ax is None:
            print(f"        {cls}: insufficient bit variation, skipped", flush=True)
            continue
        axes[cls] = ax; axes_pre[cls] = axp
        decode[cls] = _decode_bit(pools[0], pools[1], rng)
        # untrained PC2
        _, cpools = _mixed_axis_at(ctrl, cfg, cls, read_digit, hook, rng, max(60, n_q // 3), 1)
        ctrl_decode[cls] = _decode_bit(cpools[0], cpools[1], rng) if cpools else float("nan")
        print(f"        {cls}: decode={decode[cls]:.3f} (untrained {ctrl_decode[cls]:.3f})", flush=True)
    # pairwise |cos| + shared-direction variance explained
    clss = [c for c in ("ADD", "SUB", "NEG") if c in axes]
    pair_cos = {}
    for i in range(len(clss)):
        for j in range(i + 1, len(clss)):
            pair_cos[f"{clss[i]}-{clss[j]}"] = float(abs(axes[clss[i]] @ axes[clss[j]]))
    M = np.stack([axes[c] for c in clss])  # rows = unit axes
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    var_explained_top = float((S[0] ** 2) / (S ** 2).sum()) if S.size else float("nan")
    pre_pair_cos = {}
    for i in range(len(clss)):
        for j in range(i + 1, len(clss)):
            pre_pair_cos[f"{clss[i]}-{clss[j]}"] = float(abs(axes_pre[clss[i]] @ axes_pre[clss[j]]))
    # SGN axis (descriptive): mean over SUB - mean over NEG at sign consuming position
    sgn = descriptive_sgn_axis(model, cfg, hook, rng, n_q=max(80, n_q // 2))
    sgn_cos = {c: float(abs(axes[c] @ sgn)) for c in clss} if sgn is not None else {}
    shared_bar = (all(v >= 0.7 for v in pair_cos.values()) and var_explained_top >= 0.70)
    specific_bar = all(v <= 0.3 for v in pair_cos.values())
    verdict = ("G4 confirmed (one shared rail)" if shared_bar else
               "G4 refuted (class-specific rails)" if specific_bar else
               "G4 partial (intermediate cosines)")
    del model, ctrl
    return {"model": MIXED_MODEL, "read_digit": read_digit,
            "pairwise_abs_cos_combiner": pair_cos,
            "pairwise_abs_cos_resid_pre": pre_pair_cos,
            "shared_direction_var_explained": var_explained_top,
            "abs_cos_to_SGN_axis": sgn_cos,
            "PC2_bit_decode_trained": decode, "PC2_bit_decode_untrained": ctrl_decode,
            "verdict": verdict}


def descriptive_sgn_axis(model, cfg, hook, rng, n_q=100):
    from scripts.mixed_sv import class_question, class_labels, to_q
    from quanta_maths.maths_edge_patch import consuming_pos
    sign_cpos = cfg.n_ctx - (cfg.n_digits + 2)  # sign token consuming position (first answer)
    pools = {"SUB": [], "NEG": []}
    for cls in ("SUB", "NEG"):
        for _ in range(n_q):
            a, b = class_question(cfg, rng, cls)
            q = to_q(cfg, a, b, cls)
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
            pools[cls].append(c[hook][0, sign_cpos, :].numpy())
    return unit(np.mean(pools["SUB"], 0) - np.mean(pools["NEG"], 0))


# ===========================================================================
# scoring
# ===========================================================================

def score_G1(model_results):
    """G1 scoring.

    The retention *ratio* on a fixed 1-D coordinate is near-tautological (a shared
    axis transfers its own threshold trivially), so the discriminator is the
    ABSOLUTE cross-site (off-diagonal) accuracy gain of the carry rail vs the
    wrong-axis nulls (PC3), plus the CE11 floor (PC1) and the rail's own
    within-site diagonal. Rescue is real only if the carry rail decodes ST
    cross-site (off-diag gain high) AND that is SPECIFIC to the carry axis
    (rail off-diag gain >> OV-pre-image-of-random off-diag gain)."""
    out = {}
    for mn, r in model_results.items():
        f2 = r["F2_transfer"]; f3 = r["F3_entanglement"]; pc3 = r["PC3_wrong_axis_nulls"]
        rb = f2["rail_binary_centered"]
        rail_off = rb["offdiag_gain"]          # OV-preimage rail: absolute cross-site gain
        rail_diag = rb["diag_gain"]
        floor_raw = f2["raw_tristate"]["retention_gain"]
        fb = f2["full_binary_transfer"]        # honest G1-spirit: any shared direction?
        bin_ret = fb["retention_gain"]; bin_off_gain = fb["offdiag"] - 0.5
        axis_share = f2["per_digit_binary_axis_alignment"]["mean_abs_cos"]
        worst_null = max(v["offdiag_gain"] for v in pc3.values()) if pc3 else 0.0
        pc1_floor_ok = floor_raw <= 0.30       # CE11 floor reproduced (full-act tri no transfer)
        # (a) G1 SPECIFIC mechanism: is the OV-preimage of the consumer carry axis the rail?
        #     It must transfer (offdiag_gain>=0.10), carry the class (diag_gain>=0.10),
        #     AND beat the wrong-axis PC3 null by a clear margin.
        rail_specific_ok = (rail_off >= 0.10) and (rail_diag >= 0.10) and \
                           (rail_off >= worst_null + 0.05)
        # (b) G1 SPIRIT: does ANY shared linear carry direction rescue transfer to >=0.6?
        if not pc1_floor_ok:
            v = "INVALID (PC1 floor not reproduced)"
        elif rail_specific_ok and bin_ret >= 0.6:
            v = "G1 CONFIRMED (OV-preimage carry rail rescues transfer)"
        elif not rail_specific_ok and bin_ret >= 0.6:
            v = "G1 SPIRIT-ONLY (a shared carry rail rescues transfer, but it is NOT the OV pre-image)"
        elif bin_ret < 0.6 and bin_ret > floor_raw + 0.05:
            v = ("G1 REFUTED for OV-preimage (rail at wrong-axis null level); "
                 "PARTIAL shared carry subspace (binary transfers better than tri-state "
                 "but below bar) -> read-time rotation with partial storage sharing")
        elif rail_off <= 0.03 and bin_ret <= floor_raw + 0.05:
            v = "G1 REFUTED (no shared rail; canonicalization is a read-time rotation)"
        else:
            v = "G1 PARTIAL"
        out[mn] = {
            "verdict": v,
            "ov_preimage_rail": {"offdiag_gain": rail_off, "diag_gain": rail_diag,
                                 "worst_wrong_axis_null_gain": worst_null,
                                 "specific_ok": bool(rail_specific_ok)},
            "full_binary_carry_transfer": {"diag": fb["diag"], "offdiag": fb["offdiag"],
                                           "retention": bin_ret, "offdiag_gain": bin_off_gain},
            "full_tristate_transfer_retention": floor_raw,
            "per_digit_binary_axis_mean_cos": axis_share,
            "pc1_floor_ok": bool(pc1_floor_ok),
            "F3_pre_ST_SV_angle": f3["pre_removal_ST_SV_angle"],
            "F3_cos_rail_to_SV": f3["cos_rail_to_SV_axis"],
            "F3_residual_ST_SV_angle": f3["residual_ST_SV_angle"],
        }
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    rng = np.random.default_rng(SEED)
    results = {}
    if mode == "smoke":
        results["addition"] = {ADD_MODELS[0]:
                               run_addition_model(ADD_MODELS[0], rng, n_q=250, f1_nq=120)}
        results["G1_score"] = score_G1(results["addition"])
    if mode in ("add", "all"):
        add = {mn: run_addition_model(mn, np.random.default_rng(SEED), n_q=2000, f1_nq=300)
               for mn in ADD_MODELS}
        results["addition"] = add
        results["G1_score"] = score_G1(add)
    if mode in ("mixed", "all"):
        results["mixed_F4"] = run_mixed_model(np.random.default_rng(SEED), n_q=400)
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\n=== SCORES ===")
    if "G1_score" in results:
        for mn, s in results["G1_score"].items():
            print(f"  G1 {mn}: {s['verdict']}")
            print(f"     OV-preimage rail offdiag_gain={s['ov_preimage_rail']['offdiag_gain']:.3f} "
                  f"| full-binary transfer retention={s['full_binary_carry_transfer']['retention']:.3f} "
                  f"(offdiag={s['full_binary_carry_transfer']['offdiag']:.3f}) "
                  f"| full-tri floor={s['full_tristate_transfer_retention']:.3f} "
                  f"| per-digit axis mean|cos|={s['per_digit_binary_axis_mean_cos']:.2f}")
    if "mixed_F4" in results:
        print(f"  G4: {results['mixed_F4']['verdict']} | "
              f"pairwise|cos|={results['mixed_F4']['pairwise_abs_cos_combiner']}")
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
