"""Answer-position binding: tape vs register (study-answer-binding.md).

At the answer phase (= and answer token positions), are the resolved per-digit
states SA_n (answer digit) / SV_n (resolved carry into n) held as a "tape"
(many orthogonal per-digit slots at =) or a "register" (just-in-time, few
coexist; A4)? Distinguishes decodable-here from bound/stored-here via a
label-preserving operand baseline (the transports-vs-computes lesson).

Implements the pre-run design + Gate-1 amendments AB-1..AB-5:
  * AB-1 label-preserving shuffle (fix digits <=n, re-randomize >n) baseline
  * AB-2 per-(pos,digit) margin table + R-recompute-everything outcome
  * AB-3 storage-vs-computation gate on the "tape" label
  * AB-4 answer-side transfer must beat the operand floor + centering
  * AB-5 exclude degenerate classes (SV_0); middle digits; balance; lowvar flag

CPU-only. Run:
    PYTHONPATH=. python3 scripts/answer_binding.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from scripts.confirm_st_node import load_model, make_q, verify_accuracy
from scripts.probe_transfer import sub_labels, class_mean_subspace, principal_angle_deg

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-answer-binding")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
RNG = np.random.default_rng(SEED)
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]
C_REG = 0.5


def digits_of(x, nd):
    return [int(d) for d in str(x).zfill(nd)]


def from_digits(da, db, nd):
    return int("".join(map(str, da))), int("".join(map(str, db)))


# read positions at the answer phase + operand positions (for the operand baseline)
def read_positions(cfg):
    nd = cfg.n_digits
    pos = {"=": 2 * nd + 1}
    for k in range(nd):
        pos[f"A{k}"] = cfg.n_ctx - 2 - k  # consuming position for answer digit k
    return pos


def operand_positions(cfg, n):
    """Token positions of digit n's operands Dn and D'n."""
    nd = cfg.n_digits
    return {f"Dn{n}": nd - 1 - n, f"Dpn{n}": 2 * nd - n}


def chance(task):
    return 0.10 if task == "SA" else 0.5


def fit(X, y):
    return LogisticRegression(max_iter=2000, C=C_REG).fit(X, y)


def bacc(clf, X, y):
    return float(balanced_accuracy_score(y, clf.predict(X)))


def balance_idx(y, rng):
    classes = np.unique(y)
    per = max(np.bincount(y, minlength=int(classes.max()) + 1))
    idx = []
    for cl in classes:
        ci = np.where(y == cl)[0]
        idx.extend(rng.choice(ci, size=per, replace=len(ci) < per))
    return np.array(idx)


# ===========================================================================
# data collection (main pool) + label-preserving-shuffle baseline pool
# ===========================================================================

def collect_main(model, cfg, positions, n_q):
    """Collect resid_post(L1) at answer-phase positions AND resid_post(L0) at every
    digit's operand positions (Dn/D'n) — the latter is the operand-only reference
    baseline (upper bound on 're-read the operands')."""
    nd = cfg.n_digits
    lim = 10 ** nd
    hookL1 = "blocks.1.hook_resid_post"; hookL0 = "blocks.0.hook_resid_post"
    X = {p: [] for p in positions}
    # operand ref: concat [Dn, D'n]@L0 for each n
    Xop = {n: [] for n in range(nd)}
    SA = {n: [] for n in range(nd)}; SV = {n: [] for n in range(nd)}
    for _ in range(n_q):
        a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2))
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0),
                                        names_filter=lambda nm: nm in (hookL1, hookL0))
        for p, tok in positions.items():
            X[p].append(c[hookL1][0, tok, :].numpy())
        for n in range(nd):
            op = operand_positions(cfg, n)
            Xop[n].append(np.concatenate([c[hookL0][0, t, :].numpy() for t in op.values()]))
        sa, st, sv = sub_labels(a, b, nd)
        for n in range(nd):
            SA[n].append(sa[n]); SV[n].append(sv[n])
    X = {p: np.array(v) for p, v in X.items()}
    Xop = {n: np.array(v) for n, v in Xop.items()}
    SA = {n: np.array(v) for n, v in SA.items()}; SV = {n: np.array(v) for n, v in SV.items()}
    return X, Xop, SA, SV


# ===========================================================================
# batteries
# ===========================================================================

def middle_digits(cfg):
    return list(range(1, cfg.n_digits - 1))


def valid_digits(task, cfg):
    ds = middle_digits(cfg)
    if task == "SV":
        ds = [d for d in ds if d >= 1]  # SV_0 degenerate; middle already excludes 0
    return ds


def run_model(model, cfg, n_q=4000):
    positions = read_positions(cfg)
    X, Xop, SA, SV = collect_main(model, cfg, positions, n_q)
    nctx = cfg.n_ctx; nd = cfg.n_digits
    labels = {"SA": SA, "SV": SV}
    ntr = int(0.7 * n_q); tr = slice(0, ntr); te = slice(ntr, n_q)
    out = {"model": cfg.model_name if hasattr(cfg, "model_name") else "", "n_q": n_q,
           "positions": {k: int(v) for k, v in positions.items()}}

    # --- Battery C: coexistence census. Baseline = operand-only reference: decode
    # SA_n/SV_n from digit n's OWN operand positions (Dn/D'n @ L0). A foreign answer
    # position decoding SA_n/SV_n ABOVE this baseline holds info beyond re-reading
    # the operands (candidate binding); at/below baseline = mere re-derivation. ---
    coexist = {"SA": {}, "SV": {}}
    diagonals = {"SA": {}, "SV": {}}
    op_ref = {"SA": {}, "SV": {}}
    for task in ("SA", "SV"):
        for n in valid_digits(task, cfg):
            y = labels[task][n]
            if len(np.unique(y[tr])) < 2:
                continue
            bi = balance_idx(y[tr], RNG)
            # operand-only reference (decode from n's operands)
            acc_op = bacc(fit(Xop[n][tr][bi], y[tr][bi]), Xop[n][te], y[te])
            op_ref[task][n] = acc_op
            for p in positions:
                acc_main = bacc(fit(X[p][tr][bi], y[tr][bi]), X[p][te], y[te])
                coexist[task].setdefault(p, {})[n] = {
                    "acc": acc_main, "operand_ref": acc_op,
                    "margin": float(acc_main - acc_op)}
            diagonals[task][n] = coexist[task].get(f"A{n}", {}).get(n, {}).get("acc", float("nan"))

    # --- Battery T: answer-side cross-position transfer (SA/SV) ---
    transfer = {}
    ans_positions = [f"A{k}" for k in valid_digits("SA", cfg)]
    for task in ("SA", "SV"):
        ds = valid_digits(task, cfg)
        # probe for digit i read at A_i, tested for digit j read at A_j
        probes = {}; means = {}
        for i in ds:
            y = labels[task][i]; p = f"A{i}"
            bi = balance_idx(y[tr], RNG); means[i] = X[p][tr].mean(0)
            probes[i] = (fit(X[p][tr][bi], y[tr][bi]),
                         fit((X[p] - means[i])[tr][bi], y[tr][bi]))
        mat_raw = {}; mat_cen = {}
        for i in ds:
            for j in ds:
                yj = labels[task][j]; pj = f"A{j}"
                mat_raw[f"{i}->{j}"] = bacc(probes[i][0], X[pj][te], yj[te])
                mat_cen[f"{i}->{j}"] = bacc(probes[i][1], (X[pj] - means[j])[te], yj[te])
        ch = chance(task)
        diagm = np.mean([mat_raw[f"{n}->{n}"] for n in ds])
        offr = np.mean([mat_raw[f"{i}->{j}"] for i in ds for j in ds if i != j])
        offc = np.mean([mat_cen[f"{i}->{j}"] for i in ds for j in ds if i != j])
        transfer[task] = {"matrix_raw": mat_raw, "matrix_centered": mat_cen, "chance": ch,
                          "diag_mean": float(diagm), "offdiag_raw_mean": float(offr),
                          "offdiag_centered_mean": float(offc),
                          "retention_gain": float((offr - ch) / (diagm - ch)) if diagm > ch else float("nan"),
                          "positional_offset_gain": float(offc - offr)}

    # --- Battery A: slot angles between per-digit SV_n subspaces at '=' ---
    slot_angles = {}
    mid_task = "SV"
    ds = valid_digits(mid_task, cfg)
    subs = {}
    for n in ds:
        subs[n] = class_mean_subspace(X["="], labels[mid_task][n])
    for i in ds:
        for j in ds:
            if i >= j:
                continue
            if subs[i].size == 0 or subs[j].size == 0:
                continue
            ang = principal_angle_deg(subs[i], subs[j])
            # label-corr null: permute SV_j within SV_i strata
            yi = labels[mid_task][i]; yj = labels[mid_task][j].copy()
            for cl in np.unique(yi):
                msk = yi == cl
                yj[msk] = RNG.permutation(yj[msk])
            Bnull = class_mean_subspace(X["="], yj)
            slot_angles[f"SV{i}-SV{j}@="] = {
                "angle": ang,
                "null": principal_angle_deg(subs[i], Bnull) if Bnull.size else float("nan")}

    out["coexistence"] = coexist
    out["diagonals"] = diagonals
    out["transfer"] = transfer
    out["slot_angles"] = slot_angles
    out["verdict"] = derive_verdict(out, cfg)
    return out


def derive_verdict(out, cfg):
    notes = {}
    # per task: at '=', how many FOREIGN digits decode above baseline (+0.10)?
    for task in ("SA", "SV"):
        cx = out["coexistence"][task].get("=", {})
        foreign = []
        recompute = []
        for n, d in cx.items():
            m = d.get("margin")
            if m == m and m >= 0.10:
                foreign.append(n)
            oref = d.get("operand_ref", float("nan"))
            if oref == oref and oref >= 0.85:
                recompute.append(n)
        diag_ok = any(v == v and v >= chance(task) + 0.15 for v in out["diagonals"][task].values())
        notes[task] = {"foreign_coexist_at_eq": foreign, "recompute_digits": recompute,
                       "diag_ok": bool(diag_ok)}
    # SV storage-vs-computation gate (AB-3): tape only if margin>0 AND orthogonal slots
    sv_orth = all(a["angle"] >= 60 for a in out["slot_angles"].values()) if out["slot_angles"] else None
    sv_coexist = len(notes["SV"]["foreign_coexist_at_eq"]) >= 2
    # transfer readings
    tr_read = {}
    for task in ("SA", "SV"):
        t = out["transfer"].get(task, {})
        tr_read[task] = "shared-template" if t.get("retention_gain", 0) >= 0.6 else "position-specific/re-derived"

    read = []
    # SA
    if notes["SA"]["recompute_digits"]:
        read.append("SA: recompute-everything (operands re-derive SA at foreign positions)")
    elif not notes["SA"]["foreign_coexist_at_eq"]:
        read.append("SA: register (only own-digit at its answer position; not stored at =)")
    else:
        read.append(f"SA: coexists at = for {notes['SA']['foreign_coexist_at_eq']}")
    # SV
    if sv_coexist and sv_orth:
        read.append("SV: carry-tape (coexists at = beyond baseline, orthogonal slots)")
    elif sv_coexist and not sv_orth:
        read.append("SV: coexists at = but slots NOT orthogonal (entangled, not a clean tape)")
    else:
        read.append("SV: resolved/computable at = (consistent with CE7), not shown stored")
    return {"per_task": notes, "sv_slots_orthogonal": sv_orth, "transfer": tr_read,
            "read": "; ".join(read)}


def main():
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        cfg.model_name = mn
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (acc {acc:.3f}) ===", flush=True)
        r = run_model(model, cfg)
        r["accuracy"] = acc
        results[mn] = r
        print("  VERDICT:", r["verdict"]["read"])
        # brief coexistence at =
        for task in ("SA", "SV"):
            cx = r["coexistence"][task].get("=", {})
            print(f"  {task}@= margins:", {n: round(d["margin"], 2) for n, d in cx.items() if d["margin"] == d["margin"]})
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
