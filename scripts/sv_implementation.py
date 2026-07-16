"""SV implementation sprint (study-sv-implementation.md).

Parameter-estimation study built ON the confirmed SV wiring (CE13/CE14/CE15).
Working axioms: the SV mechanism EXISTS; every battery ESTIMATES a parameter
(incl. zero); redundancy is the norm (necessity tested at CLASS level via paired
ablation). Four batteries:

  M  message identity   canonical resolved carry vs source/format-tagged, tested by
                        FORMAT INVARIANCE across families/deciding positions (on
                        chains the compound carry == deciding make-carry bit, so M
                        cannot be settled by value).  [SI-2 lnfair proj; SI-3 primary
                        = within-chain cross-deciding-position transfer + domain-null]
  R  source attribution which keys the consumer pair reads (= vs deciding-ST vs rest),
                        causal per-key-group value patch.  [SI-4 leave-one-out
                        brackets + arm-sum residual; SI-6 same_class_diff_operand null]
  P  path shares + necessity  powered direct/skip arm (SI-1: inject the MEASURED
                        real-skip carry direction at 1x/2x its own norm, in resid_post
                        (L0) space; invalid if real-skip carry ~ 0) + class necessity
                        (joint H1+H2 mean-ablate vs carry-free, untagged baseline).
  F  combiner transfer (stretch)  alpha-sweep of the delivered carry -> transfer class.

Positive controls (fail -> `invalid` for that battery, never a negative):
  PC1 regression: reproduce CE14 joint-pair flip 1.00 / null 0.00 at 6d k=3.
  PC2 arm-sum: patching ALL keys reproduces the full edge effect (R composition).
  PC3 skip power (P-i) SI-1.
  PC4 axis anchor: in-script CE6 carry axis separates committed c0/c1.
  PC5 per-family behavioral gates.

SI-7: R and P(i) retained in scope but headline numbers gated on their own controls.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/sv_implementation.py all
    PYTHONPATH=. python3 scripts/sv_implementation.py all --fast   # smaller n, dev
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
)
from scripts.deep_cascade_mechanism import (
    build_chain, consuming_pos, ak_pos, affected_digits, dn_pos, dpn_pos,
    behavioral_gate, RNG,
)
from scripts.sv_compounding import (
    CONSUMER_HEADS, INSTRUMENT_HEAD, EQ_POS, head_ov, _ln_norm,
    edge_patch_pred, cache_qs, _direct_patch_pred,
)
from scripts.st_tristate_geometry import CFG as GEO_CFG, build_class_question

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-sv-implementation")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]

Z_NAMES = ("blocks.1.attn.hook_z", "blocks.1.attn.hook_pattern",
           "blocks.1.attn.hook_v", "blocks.1.hook_resid_mid",
           "blocks.0.hook_resid_post")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mean_ci(vals):
    a = np.asarray(vals, float)
    n = len(a)
    m = float(a.mean()) if n else float("nan")
    lo, hi = wilson_ci(int(round(a.sum())), n) if n else (float("nan"), float("nan"))
    return {"rate": m, "ci": [lo, hi], "n": n}


def fit(X, y):
    return LogisticRegression(max_iter=2000, C=0.5).fit(X, y)


def bacc(clf, X, y):
    return float(balanced_accuracy_score(y, clf.predict(X)))


def cache_full(model, q):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0),
                                    names_filter=lambda nm: nm in Z_NAMES)
    return c


# ---------------------------------------------------------------------------
# CE6 carry axis (PC4): c1 - c0 at combiner input (blocks.1.ln2.hook_normalized)
# ---------------------------------------------------------------------------

def carry_axis(model, cfg, mn, n_q=250):
    """Committed-carry axis c1-c0 at the combiner INPUT, per CE6/st_tristate."""
    gc = GEO_CFG[mn]
    n = gc["digit"]; pos = gc["combiner_pos"]; hook = "blocks.1.ln2.hook_normalized"
    means = {}
    for cls in ("c0", "c1"):
        acts = []
        for _ in range(n_q):
            a, b, _sa = build_class_question(cfg, n, cls)
            c = _cache_named(model, make_q(cfg, a, b), (hook,))
            acts.append(c[hook][0, pos, :].numpy())
        means[cls] = np.mean(acts, 0)
    axis = means["c1"] - means["c0"]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    # PC4 anchor: projection separation of the two class means
    sep = float((means["c1"] - means["c0"]) @ axis)
    return {"axis": axis, "c0": means["c0"], "c1": means["c1"],
            "pos": pos, "hook": hook, "digit": n, "sep": sep}


def _cache_named(model, q, names):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm in names)
    return c


# ---------------------------------------------------------------------------
# matched twin chain pair (deciding hi vs lo, everything else shared)
# ---------------------------------------------------------------------------

def twin_pair(cfg, n_top, k):
    sh = {}
    ha, hb, hi_info, sh = build_chain(cfg, n_top, k, "hi", shared=sh)
    la, lb, lo_info, _ = build_chain(cfg, n_top, k, "lo", shared=sh)
    return (ha, hb), (la, lb)


def same_class_twin(cfg, n_top, k, dec_class):
    """SI-6 null partner: SAME deciding class, deciding operand RE-DRAWN, chain
    fillers shared with the target (only the deciding operand values differ)."""
    sh = {}
    ta, tb, _, sh = build_chain(cfg, n_top, k, dec_class, shared=sh)
    # re-draw deciding operand only by NOT sharing it (build_chain never shares it)
    na, nb, _, _ = build_chain(cfg, n_top, k, dec_class, shared=sh)
    return (na, nb)


# ===========================================================================
# consumer-pair edge contribution at the combiner input
# ===========================================================================

def pair_at_top(cfg, mn):
    """The consumer head pair at the leading-digit consuming position."""
    n_top = cfg.n_digits - 2
    top = n_top + 1
    cpos = consuming_pos(cfg, top)
    heads = sorted({h for (pos, h, kk) in CONSUMER_HEADS[mn] if pos == cpos})
    return cpos, heads, top


def edge_contribution(model, cfg, cache, cpos, heads):
    """Sum over the consumer pair of z@W_O at cpos -> a resid_mid delta vector."""
    v = torch.zeros(model.cfg.d_model)
    for h in heads:
        z = cache["blocks.1.attn.hook_z"][0, cpos, h, :]
        v = v + head_ov(model, z, 1, h)
    return v


def lnfair_project(model, cfg, rm_clean, delta, axis):
    """SI-2: push the edge delta through the clean-frozen-std LN (lnfair) then
    project onto `axis`. rm_clean is the clean resid_mid at cpos."""
    xm, std = _ln_norm(rm_clean)
    nc = xm / std
    xm2 = (rm_clean + delta) - (rm_clean + delta).mean()
    nf = xm2 / std
    gamma = model.blocks[1].ln2.w
    normed_edge = (nf - nc) * gamma          # edge in the space the MLP reads
    ax = torch.tensor(axis, dtype=normed_edge.dtype)
    proj = float(normed_edge @ ax)
    return proj, float(std)


# ===========================================================================
# Battery M: message identity (SI-2 lnfair, SI-3 within-chain primary + domain-null)
# ===========================================================================

def battery_M(model, cfg, mn, ax, n_pairs=120, committed=True):
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    axis = ax["axis"]

    # families: chain k=2, chain k=3 (differ in deciding position), committed
    fams = {}
    if n_top >= 2:
        fams["chain_k2"] = ("chain", 2)
    if n_top >= 3:
        fams["chain_k3"] = ("chain", 3)
    # feature rows: (family, carry_bit) -> list[proj], plus full edge vectors for probe
    proj = {}
    edge_vecs = {}   # (family) -> list[(edge_vector_np, carry_bit, decpos)]
    lnstds = {}
    for fam, (kind, k) in fams.items():
        proj[fam] = {0: [], 1: []}
        edge_vecs[fam] = []
        lnstds[fam] = []
        for _ in range(n_pairs):
            (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
            for (a, b), bit in [((ha, hb), 1), ((la, lb), 0)]:
                q = make_q(cfg, a, b)
                c = cache_full(model, q)
                rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
                delta = edge_contribution(model, cfg, c, cpos, heads)
                p, std = lnfair_project(model, cfg, rm, delta, axis)
                proj[fam][bit].append(p)
                lnstds[fam].append(std)
                # full normed edge vector for the transfer probe
                xm, s = _ln_norm(rm)
                nf = ((rm + delta) - (rm + delta).mean()) / s
                nc = xm / s
                vec = ((nf - nc) * model.blocks[1].ln2.w).detach().numpy()
                edge_vecs[fam].append((vec, bit, k))

    if committed and cfg.n_digits >= 6:
        # committed-carry family at the combiner digit (domain differs from chains)
        gc = GEO_CFG[mn]; nd_dig = gc["digit"]
        proj["committed"] = {0: [], 1: []}
        edge_vecs["committed"] = []
        lnstds["committed"] = []
        for _ in range(n_pairs):
            for cls, bit in [("c1", 1), ("c0", 0)]:
                a, b, _sa = build_class_question(cfg, nd_dig, cls)
                q = make_q(cfg, a, b)
                c = cache_full(model, q)
                rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
                delta = edge_contribution(model, cfg, c, cpos, heads)
                p, std = lnfair_project(model, cfg, rm, delta, axis)
                proj["committed"][bit].append(p)
                lnstds["committed"].append(std)
                xm, s = _ln_norm(rm)
                nf = ((rm + delta) - (rm + delta).mean()) / s
                nc = xm / s
                vec = ((nf - nc) * model.blocks[1].ln2.w).detach().numpy()
                edge_vecs["committed"].append((vec, bit, -1))

    # stat (i): per-family carry-1 vs carry-0 projection separation & c1-side coord
    proj_stats = {}
    for fam, d in proj.items():
        m1 = float(np.mean(d[1])); m0 = float(np.mean(d[0]))
        proj_stats[fam] = {
            "carry1_mean_proj": m1, "carry0_mean_proj": m0,
            "separation": m1 - m0,
            "carry1_ci": list(_mean_num_ci(d[1])), "carry0_ci": list(_mean_num_ci(d[0])),
            "ln_std_mean": float(np.mean(lnstds[fam])),
        }

    # stat (ii) SI-3: primary within-chain cross-deciding-position transfer (k2<->k3)
    def probe_xfer(train_fam, test_fam):
        Xtr = np.array([v for (v, b, kk) in edge_vecs[train_fam]])
        ytr = np.array([b for (v, b, kk) in edge_vecs[train_fam]])
        Xte = np.array([v for (v, b, kk) in edge_vecs[test_fam]])
        yte = np.array([b for (v, b, kk) in edge_vecs[test_fam]])
        clf = fit(Xtr, ytr)
        within = bacc(clf, Xtr, ytr)
        transfer = bacc(clf, Xte, yte)
        return within, transfer

    transfer = {}
    if "chain_k2" in edge_vecs and "chain_k3" in edge_vecs:
        w, t = probe_xfer("chain_k2", "chain_k3")
        transfer["within_chain_k2_to_k3"] = {"within": w, "transfer": t}
        w2, t2 = probe_xfer("chain_k3", "chain_k2")
        transfer["within_chain_k3_to_k2"] = {"within": w2, "transfer": t2}
    if "committed" in edge_vecs and "chain_k3" in edge_vecs:
        w, t = probe_xfer("committed", "chain_k3")
        transfer["committed_to_chain(secondary)"] = {"within": w, "transfer": t}
        # SI-3 domain-shift null: probe a carry-IRRELEVANT attribute (deciding
        # operand parity of first operand) across the same families.
        transfer["domain_shift_null"] = _domain_null_transfer(
            model, cfg, mn, cpos, heads, n_pairs=max(60, n_pairs // 2))

    # stat (iii): residual-variance test -- after removing carry-axis component,
    # is family still decodable from the edge? (tag => yes)
    resid_family = _residual_family_decode(edge_vecs, axis)

    return {"cpos": cpos, "heads": heads,
            "proj_stats": proj_stats, "transfer": transfer,
            "residual_family_decode": resid_family}


def _mean_num_ci(vals, z=1.96):
    a = np.asarray(vals, float); n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    m = a.mean(); se = a.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return (float(m - z * se), float(m + z * se))


def _domain_null_transfer(model, cfg, mn, cpos, heads, n_pairs=60):
    """SI-3 domain-shift null: a carry-IRRELEVANT attribute (first-operand
    deciding-digit parity) trained committed / tested chain. If THIS also fails
    to transfer, a carry-probe transfer failure is domain shift, not a tag."""
    n_top = cfg.n_digits - 2
    def collect(kind, k):
        X, y = [], []
        for _ in range(n_pairs):
            if kind == "chain":
                (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
                cand = [(ha, hb), (la, lb)]
            else:
                gc = GEO_CFG[mn]; dg = gc["digit"]
                cand = []
                for cls in ("c0", "c1"):
                    a, b, _ = build_class_question(cfg, dg, cls); cand.append((a, b))
            for (a, b) in cand:
                q = make_q(cfg, a, b)
                c = cache_full(model, q)
                rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
                delta = edge_contribution(model, cfg, c, cpos, heads)
                xm, s = _ln_norm(rm)
                nf = ((rm + delta) - (rm + delta).mean()) / s
                vec = ((nf - xm / s) * model.blocks[1].ln2.w).detach().numpy()
                # irrelevant label: parity of first operand
                X.append(vec); y.append(int(a % 2))
        return np.array(X), np.array(y)
    gc = GEO_CFG[mn]
    Xtr, ytr = collect("committed", -1)
    Xte, yte = collect("chain", 3 if n_top >= 3 else 2)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"within": float("nan"), "transfer": float("nan"), "note": "degenerate"}
    clf = fit(Xtr, ytr)
    return {"within": bacc(clf, Xtr, ytr), "transfer": bacc(clf, Xte, yte)}


def _residual_family_decode(edge_vecs, axis):
    """SI-9: scale-normalized residual-family test. After removing the carry-axis
    component AND z-scoring/norm-matching (so family SCALE can't fake a tag), is
    chain FAMILY (=deciding position) still decodable from the edge? (tag => yes).
    Restricted to chain families (holds chain-ness). Raw (unnormalized) also
    reported descriptively to show why it is uninformative."""
    fams = [f for f in edge_vecs if f.startswith("chain")]
    if len(fams) < 2:
        return {"note": "need >=2 chain families", "bacc_norm": float("nan")}
    ax = np.asarray(axis, float); ax = ax / (np.linalg.norm(ax) + 1e-9)
    Xr, Xn, y = [], [], []
    for fi, f in enumerate(fams):
        for (v, b, kk) in edge_vecs[f]:
            vr = v - (v @ ax) * ax               # remove carry-axis component
            Xr.append(vr)
            n = np.linalg.norm(vr) + 1e-9        # norm-match (kill scale/domain)
            Xn.append(vr / n)
            y.append(fi)
    Xr = np.array(Xr); Xn = np.array(Xn); y = np.array(y)
    return {"families": fams,
            "bacc_family_after_axis_removal_RAW": bacc(fit(Xr, y), Xr, y),
            "bacc_family_after_axis_removal_norm": bacc(fit(Xn, y), Xn, y)}


# ===========================================================================
# Battery R: source attribution (causal per-key-group value patch)  SI-4, SI-6
# ===========================================================================

def _key_groups(cfg, mn, n_top, k):
    """Key-position groups for the consuming cell: '=' , deciding-ST, rest."""
    eq = EQ_POS[mn]
    d = n_top - k                       # deciding digit index
    dec_keys = sorted({dn_pos(cfg, d), dpn_pos(cfg, d)})
    top = n_top + 1
    cpos = consuming_pos(cfg, top)
    all_keys = list(range(cpos + 1))    # causal (<= query pos)
    eq_keys = [eq] if eq <= cpos else []
    dec_keys = [p for p in dec_keys if p <= cpos]
    named = set(eq_keys) | set(dec_keys)
    rest = [p for p in all_keys if p not in named]
    return {"=": eq_keys, "deciding_ST": dec_keys, "rest": rest,
            "all": all_keys}, cpos


def battery_R(model, cfg, mn, ax, n_pairs=40, depths=None):
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    if depths is None:
        depths = [k for k in range(2, n_top + 1)]
    axis = torch.tensor(ax["axis"], dtype=torch.float32)
    out = {"cpos": cpos, "heads": heads, "depths": {}}

    for k in depths:
        groups, _ = _key_groups(cfg, mn, n_top, k)
        arm_names = ["=", "deciding_ST", "rest"]
        flips = {a: [] for a in arm_names}
        flips_all = []
        comp_flips = {a: [] for a in arm_names}      # SI-4 leave-one-group-out
        nulls = {a: [] for a in arm_names}
        ov_proj = {a: [] for a in arm_names}          # descriptive per-key OV proj

        for _ in range(n_pairs):
            (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
            sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
            clean = predict_answer(model, cfg, tq)
            sc = cache_full(model, sq); tc = cache_full(model, tq)

            def patch_keys(keyset, src_cache=sc):
                # SI-8: per-key CONTRIBUTION decomposition. z = Σ_key pattern·v.
                # For keys in `keyset` use the SOURCE pattern AND source v (the full
                # contribution, incl. attention-pattern-borne carry); elsewhere target.
                patches = []
                ks = set(keyset)
                for h in heads:
                    pat_t = tc["blocks.1.attn.hook_pattern"][0, h, cpos, :].clone()
                    v_t = tc["blocks.1.attn.hook_v"][0, :, h, :].clone()
                    pat_s = src_cache["blocks.1.attn.hook_pattern"][0, h, cpos, :]
                    v_s = src_cache["blocks.1.attn.hook_v"][0, :, h, :]
                    pat = pat_t.clone(); v = v_t.clone()
                    for kp in ks:
                        pat[kp] = pat_s[kp]; v[kp] = v_s[kp]
                    z_new = torch.einsum("k,kd->d", pat, v)
                    patches.append((cpos, 1, h, z_new.detach().numpy()))
                p = edge_patch_pred(model, cfg, tq, patches, tc)
                return float(p[idx] != clean[idx])

            # single-group arms
            for a in arm_names:
                flips[a].append(patch_keys(groups[a]) if groups[a] else 0.0)
            # all-keys (PC2 arm-sum / full effect)
            flips_all.append(patch_keys(groups["all"]))
            # leave-one-group-out complements (SI-4)
            for a in arm_names:
                comp = [p for p in groups["all"] if p not in set(groups[a])]
                comp_flips[a].append(patch_keys(comp) if comp else 0.0)

            # SI-6 deciding-matched null per arm: source is a SAME-CLASS (lo) twin,
            # deciding operand re-drawn -> a flip means operand content, not carry.
            na, nb = same_class_twin(cfg, n_top, k, "lo")
            nc = cache_full(model, make_q(cfg, na, nb))
            for a in arm_names:
                # SI-6 null: source = same-class (lo) twin, deciding operand re-drawn.
                # Same per-key contribution decomposition (SI-8) as the real arm.
                nulls[a].append(patch_keys(groups[a], src_cache=nc) if groups[a] else 0.0)

            # descriptive per-key OV contribution projected on carry axis
            for a in arm_names:
                s = 0.0
                for h in heads:
                    for kp in groups[a]:
                        pat = float(tc["blocks.1.attn.hook_pattern"][0, h, cpos, kp])
                        vk = tc["blocks.1.attn.hook_v"][0, kp, h, :]
                        contrib = head_ov(model, pat * vk, 1, h)
                        s += float((contrib @ axis).detach())
                ov_proj[a].append(s)

        full = float(np.mean(flips_all))
        arm_means = {a: float(np.mean(flips[a])) for a in arm_names}
        comp_means = {a: float(np.mean(comp_flips[a])) for a in arm_names}
        arm_sum = sum(arm_means.values())
        residual = full - arm_sum
        # SI-4 brackets: [single-group, full - complement]
        brackets = {a: sorted([arm_means[a], full - comp_means[a]]) for a in arm_names}
        leaky = abs(residual) > 0.15
        out["depths"][k] = {
            "flip_full_all_keys": {**mean_ci(flips_all)},
            "arm_flip": {a: mean_ci(flips[a]) for a in arm_names},
            "complement_flip": {a: mean_ci(comp_flips[a]) for a in arm_names},
            "arm_sum_residual": residual,
            "leaky_gt_0.15": bool(leaky),
            "share_bracket": brackets,
            "null_flip": {a: mean_ci(nulls[a]) for a in arm_names},
            "ov_proj_mean": {a: float(np.mean(ov_proj[a])) for a in arm_names},
            "report_mode": "ordinal_dominance" if leaky else "normalized_share",
            "key_groups": {a: groups[a] for a in arm_names},
        }
    return out


# ===========================================================================
# Battery P: path shares (SI-1 powered arms) + class necessity
# ===========================================================================

def _measure_skip_carry_dir(model, cfg, mn, cpos, n_pairs=60, k=None):
    """SI-1: the MEASURED real-skip carry direction = mean twin resid_post(L0)
    carry-difference at the consuming position, with non-carry variance regressed
    out (project out the top principal component orthogonal to the mean diff is
    overkill; we use the mean deciding-toggle difference as the carry direction and
    report its norm)."""
    n_top = cfg.n_digits - 2
    if k is None:
        k = min(3, n_top)
    diffs = []
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        sc = _cache_named(model, make_q(cfg, ha, hb), ("blocks.0.hook_resid_post",))
        tc = _cache_named(model, make_q(cfg, la, lb), ("blocks.0.hook_resid_post",))
        d = (sc["blocks.0.hook_resid_post"][0, cpos, :]
             - tc["blocks.0.hook_resid_post"][0, cpos, :]).numpy()
        diffs.append(d)
    diffs = np.array(diffs)
    mean_dir = diffs.mean(0)
    norm = float(np.linalg.norm(mean_dir))
    unit = mean_dir / (norm + 1e-9)
    # non-carry variance: residual after projecting each diff onto unit
    proj_mags = diffs @ unit
    carry_component_norm = float(np.abs(proj_mags).mean())
    return {"unit": unit, "carry_norm": norm, "per_pair_carry_mag": carry_component_norm,
            "k": k, "n_pairs": n_pairs}


def battery_P(model, cfg, mn, ax, n_pairs=40, n_measure=60):
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    k = min(3, n_top)

    # SI-1: measure the real-skip carry direction and its magnitude
    skip = _measure_skip_carry_dir(model, cfg, mn, cpos, n_pairs=n_measure, k=k)
    unit = torch.tensor(skip["unit"], dtype=torch.float32)
    base_mag = skip["carry_norm"]

    out = {"cpos": cpos, "heads": heads, "k": k, "skip_measure": {
        "carry_norm": skip["carry_norm"], "per_pair_carry_mag": skip["per_pair_carry_mag"]}}

    # PC3 / power control: inject along the MEASURED skip direction at 1x and 2x
    # into resid_post(L0) at cpos; must flip if the skip arm is causal.
    def inject_pred(tq, tc, mag):
        def hook(act, hook):
            act[:, cpos, :] = act[:, cpos, :] + mag * unit
            return act
        ap = answer_positions(cfg)
        with torch.no_grad():
            lg = model.run_with_hooks(tq.unsqueeze(0),
                                      fwd_hooks=[("blocks.0.hook_resid_post", hook)])
        return lg[0, [p - 1 for p in ap]].argmax(-1)

    power = {"1x": [], "2x": []}
    skip_real = []; head_real = []
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
        clean = predict_answer(model, cfg, tq)
        sc = cache_full(model, sq); tc = cache_full(model, tq)
        # power injections (into lo target, direction toward hi carry)
        for tag, mag in [("1x", base_mag), ("2x", 2 * base_mag)]:
            p = inject_pred(tq, tc, mag)
            power[tag].append(float(p[idx] != clean[idx]))
        # real skip arm: patch resid_post(L0) at cpos from twin (heads frozen via
        # recompute -- here we patch the residual directly, L1 heads recompute)
        dd = (sc["blocks.0.hook_resid_post"][0, cpos, :]
              - tc["blocks.0.hook_resid_post"][0, cpos, :])
        def skip_hook(act, hook):
            act[:, cpos, :] = act[:, cpos, :] + dd; return act
        ap = answer_positions(cfg)
        with torch.no_grad():
            lg = model.run_with_hooks(tq.unsqueeze(0),
                                      fwd_hooks=[("blocks.0.hook_resid_post", skip_hook)])
        p = lg[0, [pp - 1 for pp in ap]].argmax(-1)
        skip_real.append(float(p[idx] != clean[idx]))
        # real head-pair arm: lnfair edge patch of the consumer pair (skip frozen)
        patches = [(cpos, 1, h, sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()) for h in heads]
        p2 = edge_patch_pred(model, cfg, tq, patches, tc)
        head_real.append(float(p2[idx] != clean[idx]))

    power_1x = float(np.mean(power["1x"])); power_2x = float(np.mean(power["2x"]))
    # SI-1: skip arm invalid if measured real-skip carry component ~ 0 OR power fails
    skip_powered = (skip["per_pair_carry_mag"] > 1e-3) and (power_1x >= 0.5)
    out["direct_arm"] = {
        "power_1x": mean_ci(power["1x"]), "power_2x": mean_ci(power["2x"]),
        "skip_real_flip": mean_ci(skip_real),
        "head_pair_real_flip": mean_ci(head_real),
        "powered": bool(skip_powered),
        "skip_share": (float(np.mean(skip_real)) if skip_powered else None),
        "note": ("skip arm valid" if skip_powered else
                 "skip arm INVALID (SI-1): measured real-skip carry ~0 or power<0.5; no skip-share claimed"),
    }

    # class necessity: joint H1+H2 mean-ablate on cascade vs carry-free family
    out["class_necessity"] = _class_necessity(model, cfg, mn, cpos, heads, n_pairs=n_pairs)
    return out


def _mean_ablate_pred(model, cfg, cpos, heads, q, mean_z):
    """Mean-ablate the consumer pair's z at cpos, read the answer digits."""
    ap = answer_positions(cfg)
    def hook(act, hook):
        for h in heads:
            act[:, cpos, h, :] = torch.tensor(mean_z[h])
        return act
    with torch.no_grad():
        lg = model.run_with_hooks(q.unsqueeze(0),
                                  fwd_hooks=[("blocks.1.attn.hook_z", hook)])
    return lg[0, [p - 1 for p in ap]].argmax(-1)


def _class_necessity(model, cfg, mn, cpos, heads, n_pairs=40):
    """Joint H1+H2 mean-ablate: accuracy drop on cascade (chain) vs carry-free.
    Untagged-pair baseline (a non-consumer head pair). CE14 SV-6 absolute floor."""
    n_top = cfg.n_digits - 2
    top = n_top + 1
    idx = len(answer_positions(cfg)) - 1 - top
    k = min(3, n_top)
    # estimate mean z over a mix of questions
    zs = {h: [] for h in heads}
    for _ in range(60):
        a = int(RNG.integers(0, 10 ** cfg.n_digits)); b = int(RNG.integers(0, 10 ** cfg.n_digits))
        c = cache_full(model, make_q(cfg, a, b))
        for h in heads:
            zs[h].append(c["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())
    mean_z = {h: np.mean(zs[h], 0) for h in heads}
    # untagged baseline pair: heads NOT in the consumer set (co-located)
    untagged = [h for h in range(cfg.n_heads) if h not in heads][:len(heads)] or heads
    mean_zu = {h: np.mean(zs.get(h, [cache_full(model, make_q(cfg, 0, 0))["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()]), 0)
               if h in zs else cache_full(model, make_q(cfg, 0, 0))["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()
               for h in untagged}

    def family_acc(kind, ablate_heads, mean_map):
        correct = 0; nn = 0
        for _ in range(n_pairs):
            if kind == "cascade":
                (ha, hb), _ = twin_pair(cfg, n_top, k); a, b = ha, hb
            else:  # carry-free: sum with no carries at all
                da = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
                db = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
                a = int("".join(map(str, da))); b = int("".join(map(str, db)))
            q = make_q(cfg, a, b)
            gold = predict_answer(model, cfg, q)  # model's own clean answer as ref
            pred = _mean_ablate_pred(model, cfg, cpos, ablate_heads, q, mean_map)
            correct += int(pred[idx] == gold[idx]); nn += 1
        return correct / nn

    casc_abl = family_acc("cascade", heads, mean_z)
    free_abl = family_acc("carry_free", heads, mean_z)
    casc_base = family_acc("cascade", untagged, mean_zu)
    free_base = family_acc("carry_free", untagged, mean_zu)
    return {
        "consumer_pair": heads, "untagged_pair": untagged,
        "cascade_acc_after_ablate": casc_abl,
        "carryfree_acc_after_ablate": free_abl,
        "selective_gap": free_abl - casc_abl,
        "untagged_cascade_acc_after_ablate": casc_base,
        "untagged_carryfree_acc_after_ablate": free_base,
        "untagged_selective_gap": free_base - casc_base,
        "necessity_over_baseline": (free_abl - casc_abl) - (free_base - casc_base),
    }


# ===========================================================================
# Battery F (stretch): combiner transfer function
# ===========================================================================

def battery_F(model, cfg, mn, ax, n_q=60, alphas=(-2, -1, 0, 1, 2)):
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    axis = torch.tensor(ax["axis"], dtype=torch.float32)
    gc = GEO_CFG[mn]; dg = gc["digit"]; comb = gc["combiner_pos"]
    # local class fixed committed-lo; sweep delivered carry along axis at combiner input
    curve = {}
    for al in alphas:
        flips_to_1 = []
        for _ in range(n_q):
            a, b, _ = build_class_question(cfg, dg, "c0")
            q = make_q(cfg, a, b)
            clean = predict_answer(model, cfg, q)
            def hook(act, hook):
                act[:, comb, :] = act[:, comb, :] + al * axis * ax["sep"]
                return act
            ap = answer_positions(cfg)
            with torch.no_grad():
                lg = model.run_with_hooks(q.unsqueeze(0),
                                          fwd_hooks=[("blocks.1.ln2.hook_normalized", hook)])
            pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            di = len(ap) - 1 - dg
            flips_to_1.append(float(pred[di] != clean[di]))
        curve[str(al)] = float(np.mean(flips_to_1))
    # step-vs-linear: monotone jump vs proportional
    vals = [curve[str(a)] for a in alphas]
    jump = max(np.diff(vals)) if len(vals) > 1 else 0.0
    spread = max(vals) - min(vals)
    return {"cpos_combiner": comb, "digit": dg, "curve": curve,
            "max_step": float(jump), "spread": float(spread),
            "class": "step" if jump > 0.5 * (spread + 1e-9) else "graded"}


# ===========================================================================
# positive control PC1: regression to CE14 joint-pair flip / null
# ===========================================================================

def pc1_regression(model, cfg, mn, n_pairs=40):
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    k = min(3, n_top)
    flips = []; nulls = []
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
        clean = predict_answer(model, cfg, tq)
        sc, tc = cache_qs(model, sq, tq)
        patches = [(cpos, 1, h, sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()) for h in heads]
        p = edge_patch_pred(model, cfg, tq, patches, tc)
        flips.append(float(p[idx] != clean[idx]))
        na, nb = same_class_twin(cfg, n_top, k, "lo")
        ncc, _ = cache_qs(model, make_q(cfg, na, nb), tq)
        pn = edge_patch_pred(model, cfg, tq,
                             [(cpos, 1, h, ncc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()) for h in heads], tc)
        nulls.append(float(pn[idx] != clean[idx]))
    return {"k": k, "joint_pair_flip": mean_ci(flips), "deciding_matched_null": mean_ci(nulls)}


# ===========================================================================
# driver
# ===========================================================================

def run_model(model, cfg, mn, fast=False):
    np_ = 20 if fast else 40
    mp = 60 if fast else 120
    n_top = cfg.n_digits - 2
    out = {"model": mn, "n_top": n_top}
    # PC5 per-family gates
    depths = list(range(2, n_top + 1)) or [1]
    out["gates"] = {k: behavioral_gate(model, cfg, n_top, k, n_q=40) for k in depths}
    ax = carry_axis(model, cfg, mn, n_q=(120 if fast else 250))
    out["PC4_axis_anchor"] = {"c0_c1_sep_on_axis": ax["sep"], "digit": ax["digit"], "pos": ax["pos"]}
    out["PC1_regression"] = pc1_regression(model, cfg, mn, n_pairs=np_)
    out["battery_R"] = battery_R(model, cfg, mn, ax, n_pairs=np_, depths=depths)
    out["battery_P"] = battery_P(model, cfg, mn, ax, n_pairs=np_, n_measure=(40 if fast else 60))
    out["battery_M"] = battery_M(model, cfg, mn, ax, n_pairs=mp, committed=(cfg.n_digits >= 6))
    if not fast:
        out["battery_F"] = battery_F(model, cfg, mn, ax)
    out["verdict"] = derive_verdict(out)
    return out


def derive_verdict(out):
    v = {}
    # PC gates
    pc1 = out["PC1_regression"]
    pc1_ok = pc1["joint_pair_flip"]["rate"] >= 0.8 and pc1["deciding_matched_null"]["rate"] <= 0.2
    axis_ok = out["PC4_axis_anchor"]["c0_c1_sep_on_axis"] > 0
    v["controls"] = {"PC1_regression_ok": bool(pc1_ok), "PC4_axis_ok": bool(axis_ok)}

    # M message identity
    M = out["battery_M"]
    xf = M["transfer"]
    prim = [d["transfer"] for kk, d in xf.items() if kk.startswith("within_chain")]
    prim_within = [d["within"] for kk, d in xf.items() if kk.startswith("within_chain")]
    m_read = "insufficient"
    if prim:
        tr = float(np.mean(prim)); wi = float(np.mean(prim_within))
        # SI-9: use the SCALE-NORMALIZED residual only; raw is domain-shift confounded.
        resid = M["residual_family_decode"].get("bacc_family_after_axis_removal_norm", float("nan"))
        # canonical: within-chain transfer ~ within-acc AND family NOT decodable after
        # axis removal on norm-matched vectors (chance for 2 families ~ 0.5).
        # SI-10: two SEPARABLE facts, not a binary. (1) carry canonical iff transfer
        # ~ within-acc; (2) position co-rider iff family decodable after axis removal.
        carry_canonical = tr >= 0.9 * wi
        position_corider = (resid == resid) and (resid >= 0.65)
        if carry_canonical and position_corider:
            m_read = "M-canonical-carry + position co-rider"
        elif carry_canonical:
            m_read = "M-canonical (format-invariant resolved carry, no co-rider)"
        elif not carry_canonical and position_corider:
            m_read = "M-tagged (carry not format-invariant; position dominates)"
        else:
            m_read = "M-mixed/insufficient"
        dsn = xf.get("domain_shift_null", {})
        v["M"] = {"read": m_read, "carry_canonical": bool(carry_canonical),
                  "position_corider": bool(position_corider),
                  "within_chain_transfer": tr, "within_acc": wi,
                  "family_decode_after_axis_removal_norm": resid,
                  "family_decode_after_axis_removal_RAW_descriptive":
                      M["residual_family_decode"].get("bacc_family_after_axis_removal_RAW"),
                  "committed_transfer_descriptive_domain_confounded":
                      xf.get("committed_to_chain(secondary)"),
                  "domain_shift_null_transfer": dsn.get("transfer")}

    # R source attribution
    R = out["battery_R"]
    r_summary = {}
    revive_depths = 0
    for k, d in R["depths"].items():
        arms = {a: d["arm_flip"][a]["rate"] for a in ("=", "deciding_ST", "rest")}
        nulls = {a: d["null_flip"][a]["rate"] for a in ("=", "deciding_ST", "rest")}
        dominant = max(arms, key=lambda a: arms[a])
        r_summary[k] = {"arms": arms, "dominant": dominant, "leaky": d["leaky_gt_0.15"],
                        "nulls": nulls}
        # A9 (SI-5): deciding-ST dominant AND carry-specific (null low) counts as a depth
        if dominant == "deciding_ST" and arms["deciding_ST"] >= 0.5 and nulls["deciding_ST"] <= 0.2:
            revive_depths += 1
    # SI-5: >=2 GENUINELY INDEPENDENT depths -> revive A9 SOURCE PREMISE only
    a9 = ("A9 source premise SUPPORTED (deciding-ST dominant at >=2 depths); "
          "selection MECHANISM still needs CE14 same-cell tracking+edge bar (unmet)"
          if revive_depths >= 2 else
          "A9 not revived at source level (=-dominant or <2 deciding-ST depths) -> depot reading")
    v["R"] = {"per_depth": r_summary, "revive_depths": revive_depths, "A9": a9}

    # P path + necessity
    P = out["battery_P"]
    da = P["direct_arm"]
    cn = P["class_necessity"]
    v["P"] = {
        "skip_powered": da["powered"], "skip_share": da["skip_share"],
        "head_pair_real_flip": da["head_pair_real_flip"]["rate"],
        "power_1x": da["power_1x"]["rate"], "power_2x": da["power_2x"]["rate"],
        "class_necessity_over_baseline": cn["necessity_over_baseline"],
        "selective_gap": cn["selective_gap"],
    }
    if "battery_F" in out:
        v["F"] = {"transfer_class": out["battery_F"]["class"], "spread": out["battery_F"]["spread"]}
    return v


def main():
    fast = "--fast" in sys.argv
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (acc {acc:.3f}) fast={fast} ===", flush=True)
        r = run_model(model, cfg, mn, fast=fast)
        r["accuracy"] = acc
        results[mn] = r
        v = r["verdict"]
        print("  PC1", r["PC1_regression"]["joint_pair_flip"]["rate"],
              "null", r["PC1_regression"]["deciding_matched_null"]["rate"],
              "| PC4 axis sep", round(r["PC4_axis_anchor"]["c0_c1_sep_on_axis"], 3), flush=True)
        print("  [M]", v.get("M"), flush=True)
        print("  [R] A9:", v["R"]["A9"], "revive_depths", v["R"]["revive_depths"], flush=True)
        for k, d in v["R"]["per_depth"].items():
            print(f"      k={k} arms={ {a: round(x,2) for a,x in d['arms'].items()} } dom={d['dominant']} leaky={d['leaky']}", flush=True)
        print("  [P]", v["P"], flush=True)
        if "F" in v:
            print("  [F]", v["F"], flush=True)
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
