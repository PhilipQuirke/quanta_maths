"""Cross-position and cross-subtask probe transfer (study-probe-transfer.md).

Tests C2 (same-subtask template sharing + cross-subtask orthogonality), A4
(position-addressed templates + positional binding), A8 (low interference).

Per-digit addition sub-tasks at question positions:
  SA_n = (Dn+D'n) % 10  (10-way)
  ST_n = 0 if sum<=8, 1 if sum>=10, U(=2) if sum==9  (tri-state carry class)
  SV_n = resolved carry INTO digit n after the full cascade  (binary)

Implements the pre-run design + Gate-1 amendments T-1..T-6:
  * T-1 read-site pre-flight (each sub-task read at its own validated site)
  * T-2 C2-vs-A4 split via positional-offset + position-only null (not angle)
  * T-3 cross-position cross-subtask floor + angle label-correlation null
  * T-4 principal angles on class-mean activation subspaces (not LR weights)
  * T-5 ST-only degenerate accepted; SV read late
  * T-6 balanced train+test; low-variance flag

CPU-only. Run:
    PYTHONPATH=. python3 scripts/probe_transfer.py preflight
    PYTHONPATH=. python3 scripts/probe_transfer.py models
    PYTHONPATH=. python3 scripts/probe_transfer.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import load_model, make_q, answer_positions, verify_accuracy
# Shared probe/geometry primitives now live in the library (Stage 2 migration).
from quanta_maths.maths_probe import (
    sub_labels, site_hook_and_pos, collect_site_activations, TASK_CHANCE,
    train_test_split_idx, balance_idx as _lib_balance_idx, fit_probe as _lib_fit_probe,
    probe_balanced_accuracy, class_mean_subspace, principal_angles_deg)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-probe-transfer")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
RNG = np.random.default_rng(SEED)
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]
C_REG = 0.5  # L2 logistic-regression strength


# ===========================================================================
# labels
# ===========================================================================

# Sub-task labels + site algebra + activation collection now come from the library
# (quanta_maths.maths_probe). Local shims keep this script's call sites unchanged.
ALL_SITES = ["Dpn_L0", "Dpn_L1", "Dn_L0", "ans_L0", "eq_L1"]


def collect(model, cfg, n_q, digits, sites):
    return collect_site_activations(model, cfg, n_q, digits, sites, RNG)


def balance_idx(y, rng, cap=None):
    return _lib_balance_idx(y, rng, cap=cap)


def chance(task):
    return TASK_CHANCE[task]


def decodable_threshold(task):
    """A site 'holds' a sub-task if its balanced accuracy clears chance by a
    margin. A multiplicative 2x bar is impossible for binary SV (2*0.5=1.0), so
    use chance + 0.15 (well above the ~0.02 resolvable noise at this n)."""
    return chance(task) + 0.15


def fit_probe(X, y):
    return _lib_fit_probe(X, y, C=C_REG)


def probe_acc(clf, X, y):
    return probe_balanced_accuracy(clf, X, y)


def split(n, frac=0.7):
    return train_test_split_idx(n, RNG, frac=frac)


# ===========================================================================
# read-site pre-flight (T-1)
# ===========================================================================

def preflight(model, cfg, digits, n_q=1500):
    mid = digits[len(digits) // 2]
    acts, labs = collect(model, cfg, n_q, [mid], ALL_SITES)
    table = {}
    best_site = {}
    for task in ("SA", "ST", "SV"):
        table[task] = {}
        y = labs[task][mid]
        for s in ALL_SITES:
            X = acts[(s, mid)]
            tr, te = split(len(y))
            bi = balance_idx(y[tr], RNG)
            clf = fit_probe(X[tr][bi], y[tr][bi])
            acc = probe_acc(clf, X[te], y[te])
            table[task][s] = acc
        # earliest site clearing chance by the margin (site order = ALL_SITES).
        # Exclude ans_L0 for SA/ST as a QUESTION-position study? No: A4/C2 are about
        # where the subtask is computed; we read each subtask at its own best site
        # but record all. Prefer a question-side site if it clears the bar, else the
        # answer site (SA genuinely lives at the answer position per CE2/CE3).
        thr = decodable_threshold(task)
        q_sites = ["Dpn_L0", "Dpn_L1", "Dn_L0"]
        chosen = next((s for s in q_sites if table[task][s] >= thr), None)
        if chosen is None:  # fall back to answer/eq site
            chosen = next((s for s in ("ans_L0", "eq_L1") if table[task][s] >= thr), None)
        best_site[task] = chosen
    return {"digit": mid, "table": table, "chosen_site": best_site}


# ===========================================================================
# transfer matrix (T-2/T-3) + class-mean subspace angles (T-4)
# ===========================================================================

# class_mean_subspace imported from the library.


def principal_angle_deg(B1, B2):
    ang = principal_angles_deg(B1, B2)  # library returns all angles in degrees
    return float(np.max(ang)) if ang.size else float("nan")


PRIMARY_SITE = "Dpn_L0"  # T-7: all cross-subtask comparisons at one shared question site


def run_model(model, cfg, digits, chosen_site, n_q=4000):
    # collect at the primary shared site (transfer + cross-subtask) AND each
    # subtask's own best site (descriptive diagonal only).
    sites = sorted(set([PRIMARY_SITE] + [v for v in chosen_site.values() if v]))
    acts, labs = collect(model, cfg, n_q, digits, sites)
    out = {"digits": digits, "chosen_site": chosen_site, "primary_site": PRIMARY_SITE}
    s = PRIMARY_SITE
    tasks = ("SA", "ST", "SV")

    # --- transfer matrices at the primary site (raw + centered) ---
    transfer = {}
    for task in tasks:
        probes_raw = {}; probes_cen = {}; means = {}; te_idx = {}
        for n in digits:
            X = acts[(s, n)]; y = labs[task][n]
            tr, te = split(len(y)); te_idx[n] = te; means[n] = X[tr].mean(0)
            bi = balance_idx(y[tr], RNG)
            probes_raw[n] = fit_probe(X[tr][bi], y[tr][bi])
            probes_cen[n] = fit_probe((X - means[n])[tr][bi], y[tr][bi])
        mat_raw = {}; mat_cen = {}
        for i in digits:
            for j in digits:
                Xj = acts[(s, j)]; yj = labs[task][j]
                mat_raw[f"{i}->{j}"] = probe_acc(probes_raw[i], Xj[te_idx[j]], yj[te_idx[j]])
                mat_cen[f"{i}->{j}"] = probe_acc(probes_cen[i], (Xj - means[j])[te_idx[j]], yj[te_idx[j]])
        ch = chance(task)
        diagm = np.mean([mat_raw[f"{n}->{n}"] for n in digits])
        offr = np.mean([mat_raw[f"{i}->{j}"] for i in digits for j in digits if i != j])
        offc = np.mean([mat_cen[f"{i}->{j}"] for i in digits for j in digits if i != j])
        # low-variance frac (corrected): var captured by class-mean subspace / total
        mid = digits[len(digits) // 2]
        B = class_mean_subspace(acts[(s, mid)], labs[task][mid])
        Xm = acts[(s, mid)]
        proj = Xm @ B.T if B.size else np.zeros((len(Xm), 1))
        lowvar = float(proj.var(0).sum() / (Xm.var(0).sum() + 1e-9))
        transfer[task] = {"site": s, "matrix_raw": mat_raw, "matrix_centered": mat_cen,
                          "chance": ch, "diag_mean": float(diagm),
                          "offdiag_raw_mean": float(offr), "offdiag_centered_mean": float(offc),
                          # chance-relative retention (cardinality-robust)
                          "diag_gain": float(diagm - ch), "offdiag_gain": float(offr - ch),
                          "retention_gain": float((offr - ch) / (diagm - ch)) if diagm > ch else float("nan"),
                          "positional_offset_gain": float(offc - offr),
                          "lowvar_frac": lowvar}

    # --- cross-subtask floor at the primary site (chance-relative gain) ---
    # train task ta at position i, measure how much a tb-probe-shaped readout of
    # ta's directions predicts tb; cardinality-robust via gain over tb-chance.
    # Implemented as: cross-decodability = accuracy of a tb-probe restricted to the
    # ta class-mean subspace, minus tb chance.
    floor = {}
    subspaces = {t: class_mean_subspace(acts[(s, digits[len(digits)//2])], labs[t][digits[len(digits)//2]])
                 for t in tasks}
    for tb in tasks:
        y = labs[tb][digits[len(digits)//2]]; X = acts[(s, digits[len(digits)//2])]
        tr, te = split(len(y)); bi = balance_idx(y[tr], RNG)
        for ta in tasks:
            if ta == tb:
                continue
            Bta = subspaces[ta]
            if Bta.size == 0:
                continue
            Xp = X @ Bta.T  # project onto ta's class-mean subspace
            clf = fit_probe(Xp[tr][bi], y[tr][bi])
            acc = probe_acc(clf, Xp[te], y[te])
            floor[f"{tb}_from_{ta}subspace_gain"] = float(acc - chance(tb))

    # --- principal angles on class-mean subspaces at the primary site (T-4) ---
    angles = {}
    mid = digits[len(digits) // 2]
    for ta in tasks:
        for tb in tasks:
            if ta >= tb:
                continue
            Ba = subspaces[ta]; Bb = subspaces[tb]
            if Ba.size == 0 or Bb.size == 0:
                continue
            ang = principal_angle_deg(Ba, Bb)
            ya = labs[ta][mid]; yb = labs[tb][mid].copy()
            for cl in np.unique(ya):
                m = ya == cl
                yb[m] = RNG.permutation(yb[m])
            Bnull = class_mean_subspace(acts[(s, mid)], yb)
            angles[f"{ta}-{tb}"] = {"max_principal_angle_deg": ang,
                                    "label_corr_null_deg": principal_angle_deg(Ba, Bnull) if Bnull.size else float("nan")}

    # --- position-only null: decode digit position from the activation ---
    Xpos = np.concatenate([acts[(s, n)] for n in digits])
    ypos = np.concatenate([np.full(len(acts[(s, n)]), n) for n in digits])
    trp, tep = split(len(ypos)); bip = balance_idx(ypos[trp], RNG)
    posclf = fit_probe(Xpos[trp][bip], ypos[trp][bip])
    pos_acc = probe_acc(posclf, Xpos[tep], ypos[tep])

    # --- diagonal at each subtask's own best site (descriptive) ---
    best_diag = {}
    for task in tasks:
        bs = chosen_site[task]
        if bs is None:
            best_diag[task] = None; continue
        y = labs[task][mid]; X = acts[(bs, mid)]
        tr, te = split(len(y)); bi = balance_idx(y[tr], RNG)
        best_diag[task] = {"site": bs, "acc": probe_acc(fit_probe(X[tr][bi], y[tr][bi]), X[te], y[te])}

    out["transfer"] = transfer
    out["cross_subtask_floor"] = floor
    out["subspace_angles"] = angles
    out["position_only_null_acc"] = pos_acc
    out["best_site_diagonal"] = best_diag
    out["verdict"] = derive_verdict(out, tasks)
    return out


def derive_verdict(out, tasks):
    # template sharing per task: chance-relative retention >= 0.6 AND off-diagonal
    # gain beats the cross-subtask-subspace floor (label-correlation control).
    template = {}
    for t in tasks:
        tr = out["transfer"].get(t, {})
        if "retention_gain" not in tr:
            continue
        # floor: best cross-subtask readout of THIS task's labels from another
        # subtask's subspace (chance-relative gain)
        floor_vals = [v for k, v in out["cross_subtask_floor"].items()
                      if k.startswith(f"{t}_from_")]
        floor_gain = max(floor_vals) if floor_vals else 0.0
        diag_weak = tr["diag_gain"] < 0.10  # barely present at the primary site
        shared = (tr["retention_gain"] >= 0.6) and (tr["offdiag_gain"] >= 2 * max(floor_gain, 0.0)) \
            and not diag_weak
        template[t] = {"shared": bool(shared), "retention_gain": tr["retention_gain"],
                       "offdiag_gain": tr["offdiag_gain"], "floor_gain": float(floor_gain),
                       "diag_gain": tr["diag_gain"], "diag_weak": bool(diag_weak),
                       "positional_offset_gain": tr["positional_offset_gain"],
                       "lowvar_frac": tr["lowvar_frac"]}
    # orthogonality vs the label-correlation null
    orth = {}
    for pair, a in out["subspace_angles"].items():
        obs = a["max_principal_angle_deg"]; null = a["label_corr_null_deg"]
        orth[pair] = {"angle": obs, "null": null,
                      "orthogonal": bool(obs >= 60),
                      "entangled_vs_null": bool(obs < null - 10)}  # more aligned than label-corr null
    strong = [t for t, v in template.items() if v["shared"] and not v["diag_weak"]]
    weak = [t for t, v in template.items() if v["shared"] and v["diag_weak"]]
    a4_offset = any(v["positional_offset_gain"] >= 0.15 for v in template.values())
    all_orth = all(v["orthogonal"] for v in orth.values()) if orth else None
    read = []
    if strong:
        read.append(f"template-shared for {strong}" + ("(A4 positional-offset)" if a4_offset else "(position-invariant)"))
    else:
        read.append("template NOT shared at question site")
    if all_orth is True:
        read.append("subtasks orthogonal (C2/A8 supported)")
    elif all_orth is False:
        read.append("subtasks NOT all orthogonal (C2 orthogonality half challenged)")
    read.append(f"position-decodable acc={out['position_only_null_acc']:.2f}")
    return {"template": template, "orthogonality": orth, "read": "; ".join(read)}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        nd = cfg.n_digits
        digits = list(range(1, nd - 1))  # middle digits (exclude units + top)
        print(f"=== {mn} (acc {acc:.3f}, digits {digits}) ===", flush=True)
        pf = preflight(model, cfg, digits)
        print("  preflight chosen sites:", pf["chosen_site"])
        for t in ("SA", "ST", "SV"):
            print(f"    {t}:", {s: round(v, 2) for s, v in pf["table"][t].items()})
        mres = {"model": mn, "accuracy": acc, "preflight": pf}
        if mode in ("models", "all"):
            r = run_model(model, cfg, digits, pf["chosen_site"])
            mres.update(r)
            print("  VERDICT:", r.get("verdict", {}).get("read"))
        results[mn] = mres
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
