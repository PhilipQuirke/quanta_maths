"""Battery F4b: ADD-nuisance-controlled G4 re-fit (study-geometry-factorization addendum).

Human-green-lit 2026-07-17 (~T-7h). Tests whether the CE27-F4 two-rail verdict
(SUB-NEG share a borrow rail; ADD-carry ~orthogonal) survives removal of the two
flagged nuisances:
  (i)  the ADD axis's untrained-decodable operand/format component (0.68), and
  (ii) the SUB/NEG shared-subtraction-format confound on their 0.897 alignment.

Arm 1  untrained-nuisance projection: project the untrained twin's per-class
       axes (nuisance subspace N, fit on an untrained train-split) out of the
       trained activations, re-fit axes, recompute pairwise |cos| + shared-dir
       variance. Controls: trained decode >=0.9 after projection; untrained
       held-out decode -> ~0.5 after projection.
Arm 2  cross-digit canonical axes: per-class axes fit independently at read
       digits 2 and 3; digit-bound nuisance cannot align axes fit at different
       digits. Within-class cross-digit |cos| (canonicality; untrained twin =
       nuisance-replication control) + cross-class x cross-digit |cos| matrix.

Usage:
    PYTHONPATH=. python3 scripts/geometry_g4_refit.py smoke   # tiny n
    PYTHONPATH=. python3 scripts/geometry_g4_refit.py full
"""
import json
import os
import sys

import numpy as np

from scripts.geometry_factorization import (
    MIXED_MODEL, SEED, _decode_bit, _mixed_axis_at, _split, bacc, fit,
    descriptive_sgn_axis, last_layer, unit,
)

RESULT_DIR = "results/study-geometry-factorization"
CLASSES = ("ADD", "SUB", "NEG")
READ_DIGITS = (2, 3)


def collect_pools(model, cfg, hook, rng, n_q, tag):
    """(cls, digit) -> (A0, A1) activation pools at the consuming position."""
    pools = {}
    total = len(CLASSES) * len(READ_DIGITS)
    i = 0
    for cls in CLASSES:
        for rd in READ_DIGITS:
            i += 1
            print(f"=== [{i}/{total}] collect {tag} {cls} read_digit={rd} "
                  f"(n_q={n_q}/bit) ===", flush=True)
            ax, p = _mixed_axis_at(model, cfg, cls, rd, hook, rng, n_q, 1)
            if ax is None:
                print(f"    insufficient bit variation, skipped", flush=True)
                continue
            pools[(cls, rd)] = p
            print(f"    N0={len(p[0])} N1={len(p[1])}", flush=True)
    return pools


def axis_of(p):
    return unit(np.asarray(p[1]).mean(0) - np.asarray(p[0]).mean(0))


def split_pool(p, rng, frac=0.6):
    """Split (A0, A1) into train/test halves (for the untrained N fit)."""
    out = []
    for A in p:
        A = np.asarray(A)
        idx = rng.permutation(len(A))
        k = int(frac * len(A))
        out.append((A[idx[:k]], A[idx[k:]]))
    (tr0, te0), (tr1, te1) = out
    return (tr0, tr1), (te0, te1)


def project_out(N_basis, X):
    """Remove span(N_basis) (rows) from activations X."""
    if N_basis is None or len(N_basis) == 0:
        return X
    Q, _ = np.linalg.qr(np.asarray(N_basis).T)  # d x k orthonormal
    return X - (X @ Q) @ Q.T


def pairwise_cos(axes):
    keys = [c for c in CLASSES if c in axes]
    out = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            out[f"{keys[i]}-{keys[j]}"] = float(abs(axes[keys[i]] @ axes[keys[j]]))
    return out


def shared_dir_var(axes):
    keys = [c for c in CLASSES if c in axes]
    if len(keys) < 2:
        return float("nan")
    M = np.stack([axes[c] for c in keys])
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    return float((S[0] ** 2) / (S ** 2).sum())


def null_band(d, rng, n=200):
    cs = [abs(float(unit(rng.standard_normal(d)) @ unit(rng.standard_normal(d))))
          for _ in range(n)]
    return {"mean": float(np.mean(cs)), "p95": float(np.quantile(cs, 0.95))}


def decode_pools(p, rng):
    return _decode_bit(np.asarray(p[0]), np.asarray(p[1]), rng)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    n_tr, n_un = (40, 30) if mode == "smoke" else (300, 120)
    rng = np.random.default_rng(SEED + 41)
    os.makedirs(RESULT_DIR, exist_ok=True)

    from quanta_maths import load_maths_model_from_hf, make_untrained_control
    import torch  # noqa: F401  (torch session config side-effects as in F4)

    print(f"=== F4b ADD-nuisance-controlled G4 re-fit ({mode}) ===", flush=True)
    model, cfg = load_maths_model_from_hf(MIXED_MODEL, device="cpu")
    ll = last_layer(cfg)
    hook = f"blocks.{ll}.ln2.hook_normalized"
    ctrl = make_untrained_control(cfg)

    tr_pools = collect_pools(model, cfg, hook, rng, n_tr, "trained")
    un_pools = collect_pools(ctrl, cfg, hook, rng, n_un, "untrained")

    d_model = np.asarray(next(iter(tr_pools.values()))[0]).shape[1]
    res = {"model": MIXED_MODEL, "mode": mode, "n_q_trained": n_tr,
           "n_q_untrained": n_un, "read_digits": list(READ_DIGITS),
           "null_band_abs_cos": null_band(d_model, rng)}

    # ------------------------------------------------------------------
    # Baseline (reproduce F4 at each read digit) + PC2 decodes
    # ------------------------------------------------------------------
    print("=== baseline axes + decodes ===", flush=True)
    base = {}
    for rd in READ_DIGITS:
        axes = {c: axis_of(tr_pools[(c, rd)]) for c in CLASSES if (c, rd) in tr_pools}
        base[rd] = {
            "pairwise_abs_cos": pairwise_cos(axes),
            "shared_dir_var": shared_dir_var(axes),
            "decode_trained": {c: decode_pools(tr_pools[(c, rd)], rng)
                               for c in CLASSES if (c, rd) in tr_pools},
            "decode_untrained": {c: decode_pools(un_pools[(c, rd)], rng)
                                 for c in CLASSES if (c, rd) in un_pools},
        }
        print(f"  rd={rd} cos={base[rd]['pairwise_abs_cos']} "
              f"un_decode={base[rd]['decode_untrained']}", flush=True)
    res["baseline"] = base

    # ------------------------------------------------------------------
    # Arm 1: untrained-nuisance projection (per read digit)
    # ------------------------------------------------------------------
    print("=== Arm 1: untrained-nuisance projection ===", flush=True)
    arm1 = {}
    for rd in READ_DIGITS:
        tr_splits = {c: split_pool(un_pools[(c, rd)], rng)
                     for c in CLASSES if (c, rd) in un_pools}
        N = [axis_of(tr_splits[c][0]) for c in CLASSES if c in tr_splits]
        # trained pools projected
        proj_tr = {c: tuple(project_out(N, np.asarray(A)) for A in tr_pools[(c, rd)])
                   for c in CLASSES if (c, rd) in tr_pools}
        axes_clean = {c: axis_of(p) for c, p in proj_tr.items()}
        # controls
        dec_tr_clean = {c: decode_pools(p, rng) for c, p in proj_tr.items()}
        dec_un_heldout = {}
        for c in CLASSES:
            if c not in tr_splits:
                continue
            te = tr_splits[c][1]
            te_proj = tuple(project_out(N, np.asarray(A)) for A in te)
            dec_un_heldout[c] = decode_pools(te_proj, rng)
        arm1[rd] = {
            "pairwise_abs_cos_clean": pairwise_cos(axes_clean),
            "shared_dir_var_clean": shared_dir_var(axes_clean),
            "decode_trained_after_projection": dec_tr_clean,
            "decode_untrained_heldout_after_projection": dec_un_heldout,
            "axis_shift_abs_cos_raw_vs_clean": {
                c: float(abs(axes_clean[c] @ axis_of(tr_pools[(c, rd)])))
                for c in axes_clean},
        }
        print(f"  rd={rd} clean_cos={arm1[rd]['pairwise_abs_cos_clean']} "
              f"tr_dec={dec_tr_clean} un_heldout_dec={dec_un_heldout}", flush=True)
    res["arm1_projection"] = arm1

    # ------------------------------------------------------------------
    # Arm 2: cross-digit canonical axes
    # ------------------------------------------------------------------
    print("=== Arm 2: cross-digit axes ===", flush=True)
    ax_tr = {(c, rd): axis_of(tr_pools[(c, rd)]) for (c, rd) in tr_pools}
    ax_un = {(c, rd): axis_of(un_pools[(c, rd)]) for (c, rd) in un_pools}
    within = {c: float(abs(ax_tr[(c, 2)] @ ax_tr[(c, 3)]))
              for c in CLASSES if (c, 2) in ax_tr and (c, 3) in ax_tr}
    within_un = {c: float(abs(ax_un[(c, 2)] @ ax_un[(c, 3)]))
                 for c in CLASSES if (c, 2) in ax_un and (c, 3) in ax_un}
    crossx = {}
    for i, c1 in enumerate(CLASSES):
        for c2 in CLASSES[i + 1:]:
            vals = []
            for rd1, rd2 in ((2, 3), (3, 2)):
                if (c1, rd1) in ax_tr and (c2, rd2) in ax_tr:
                    vals.append(float(abs(ax_tr[(c1, rd1)] @ ax_tr[(c2, rd2)])))
            if vals:
                crossx[f"{c1}-{c2}"] = {"mean": float(np.mean(vals)), "vals": vals}
    res["arm2_cross_digit"] = {
        "within_class_cross_digit_abs_cos_trained": within,
        "within_class_cross_digit_abs_cos_untrained": within_un,
        "cross_class_cross_digit_abs_cos": crossx,
    }
    print(f"  within(trained)={within} within(untrained)={within_un}", flush=True)
    print(f"  cross-class-cross-digit={ {k: round(v['mean'], 3) for k, v in crossx.items()} }",
          flush=True)

    # SGN descriptive vs cleaned axes (recomputed at read digit 2)
    sgn = descriptive_sgn_axis(model, cfg, hook, rng, n_q=60 if mode == "smoke" else 120)
    tr_splits2 = {c: split_pool(un_pools[(c, 2)], np.random.default_rng(SEED + 7))
                  for c in CLASSES if (c, 2) in un_pools}
    N2 = [axis_of(tr_splits2[c][0]) for c in CLASSES if c in tr_splits2]
    sgn_cos = {}
    for c in CLASSES:
        if (c, 2) not in tr_pools:
            continue
        pp = tuple(project_out(N2, np.asarray(A)) for A in tr_pools[(c, 2)])
        sgn_cos[c] = float(abs(axis_of(pp) @ sgn))
    res["abs_cos_SGN_vs_cleaned_axes_rd2"] = sgn_cos

    # ------------------------------------------------------------------
    # Verdict per pre-stated F4b bars (study note addendum)
    # ------------------------------------------------------------------
    nb = res["null_band_abs_cos"]["p95"]
    a1 = arm1.get(2, {})
    cc = a1.get("pairwise_abs_cos_clean", {})
    ctrl_ok = (all(v >= 0.9 for v in a1.get("decode_trained_after_projection", {}).values())
               and all(v <= 0.6 for v in
                       a1.get("decode_untrained_heldout_after_projection", {}).values()))
    subneg_genuine = (cc.get("SUB-NEG", 0) >= 0.7
                      and crossx.get("SUB-NEG", {}).get("mean", 0) >= 0.6
                      and within_un.get("SUB", 1.0) <= max(0.2, 2 * nb)
                      and within_un.get("NEG", 1.0) <= max(0.2, 2 * nb))
    add_orth = (cc.get("ADD-SUB", 1) <= 0.3 and cc.get("ADD-NEG", 1) <= 0.3
                and a1.get("decode_trained_after_projection", {}).get("ADD", 0) >= 0.9
                and crossx.get("ADD-SUB", {}).get("mean", 1) <= 0.3
                and crossx.get("ADD-NEG", {}).get("mean", 1) <= 0.3)
    shared_bar = (all(v >= 0.7 for v in cc.values()) and a1.get("shared_dir_var_clean", 0) >= 0.7)
    res["verdict"] = {
        "arm1_controls_pass": ctrl_ok,
        "SUB_NEG_shared_borrow_rail_genuine": subneg_genuine,
        "ADD_orthogonal_established": add_orth,
        "single_shared_rail_after_cleaning": shared_bar,
        "read": ("two-rail FIRMED" if (ctrl_ok and subneg_genuine and add_orth) else
                 "single-rail (nuisance-masked) — G4 revives" if (ctrl_ok and shared_bar) else
                 "see components / scoped"),
    }
    print(f"=== VERDICT: {res['verdict']} ===", flush=True)

    out = os.path.join(RESULT_DIR, "results_f4b.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"Artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
