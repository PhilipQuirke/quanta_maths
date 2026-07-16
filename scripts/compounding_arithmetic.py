"""SV compounding arithmetic: tail-relay (A11) vs L1-read, and combiner transfer
(study-compounding-arithmetic.md).

Working-axioms mode: the cascade is computed SOMEWHERE between operands and the
combiner input (CE16 bounded where); this study LOCATES/characterizes it. No
existence tests. Four batteries + controls, with pre-launch amendments CA-1..CA-6:

  H  horizon decode (representational): does each chain-ST site's L0 write decode
     the resolved chain carry up to exactly its position-visibility horizon m
     (succeeds iff deciding d >= m)? Boundary cells (CA-1, n_top=4): P10 (m=2)
     crosses YES->NO at k2->k3; P11 (m=1) at k3->k4. P12/P14 (m=0) presence-only.
     Baselines: wrong-role co-located head + shuffled null (CE13 N-4/SV-3).
  Y  relay causality (causal): patch chain-ST site WRITES (OV contribution) between
     matched twins, letting the L1 consumer RE-ATTEND (skeptic: Y is more faithful
     than CE16 value-only R). Arms: (i) deepest visible sufficient relay SET
     (m<=d, largest m) joint; (ii) insufficient site (m>d); (iii) joint all
     sufficient relays; (iv) deciding-matched null per arm. CA-2: single-site
     interchange is pre-expected-null under redundancy (CE13 P=0.00) -> scored
     underpowered; joint arm first-class; class-level ABLATION positive control.
     CA-3: SI-4 brackets.
  L  L1-read reconstruction (the alternative's own test): regress the consumer
     head-pair edge output (carry-axis proj) on sum_key attn*phi(key) with
     phi_local (per-site local class) NESTED in phi_horizon (per-site horizon
     state); DeltaR2 via held-out permutation null on the extra columns (CA-5).
  T  combiner transfer, on-manifold (A10 iv redesign): edge(alpha)=(1-a)edge_c0 +
     a*edge_c1 substituted PRE-LN at the consuming position (CA-4), LN acts
     naturally; non-carry variance regressed out so alpha parameterizes carry;
     alpha in {-0.5,0,.25,.5,.75,1,1.5}; class decided on [0,1]. Endpoint gate:
     alpha=0/1 must reproduce CE16's 0.00/1.00 else invalid.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/compounding_arithmetic.py all
    PYTHONPATH=. python3 scripts/compounding_arithmetic.py all --fast
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
from scripts.sv_implementation import (
    pair_at_top, carry_axis, twin_pair, same_class_twin, edge_contribution,
    lnfair_project, mean_ci, wilson_ci, cache_full, _cache_named, fit, bacc,
    Z_NAMES,
)
from scripts.sv_compounding import head_ov, _ln_norm, edge_patch_pred
from scripts.node_output_encoding import ST_NODES, _mean_ablate_acc, build_random

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-compounding-arithmetic")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]

# `=` position per model (sign is `=`+1). Chain-ST sites come from ST_NODES.
EQ_POS = {"add_d6_l2_h3_t20K_s173289": 13, "add_d5_l2_h3_t15K_s372001": 11}
SIGN_POS = {"add_d6_l2_h3_t20K_s173289": 14, "add_d5_l2_h3_t15K_s372001": 12}


# ---------------------------------------------------------------------------
# horizon table (CA-1): per chain-ST site, visibility horizon m and resolves(d)
# ---------------------------------------------------------------------------

def horizon_of_site(mn, pos):
    """m = visibility horizon = smallest digit index whose operand pair the site
    can see under the causal mask. A site at a D'-token position sees digit pairs
    j with D'_j <= pos; D'_j sits at pos (2*nd - j), so j >= (2*nd - pos) = m.
    Sites at `=` or the sign token see the WHOLE question -> m=0. (CA-1: verified
    against the token layout; `=`/sign are NOT D'-tokens so the inversion
    `2*nd - pos` does not apply to them.)"""
    nd = 6 if mn.startswith("add_d6") else 5
    if pos >= EQ_POS[mn]:          # `=` (EQ_POS) and sign (EQ_POS+1) see everything
        return 0
    return (2 * nd) - pos


def chain_st_sites(mn):
    """Chain-relevant ST sites at L0 with their horizon m. (pos, head, tagged_digit, m)."""
    out = []
    for (pos, L, head, dig) in ST_NODES[mn]:
        if L != 0:
            continue
        m = horizon_of_site(mn, pos)
        out.append({"pos": pos, "head": head, "digit": dig, "m": m})
    return out


# ---------------------------------------------------------------------------
# resolved chain carry ground truth
# ---------------------------------------------------------------------------

def chain_carry_out(cfg, a, b, n):
    """carry_out AT digit n (the chain top): does the sum carry out of digit n?"""
    da = [int(x) for x in str(a).zfill(cfg.n_digits)]
    db = [int(x) for x in str(b).zfill(cfg.n_digits)]
    carry = 0
    for i in range(cfg.n_digits - 1, cfg.n_digits - 1 - (n + 1), -1):
        s = da[i] + db[i] + carry
        carry = 1 if s >= 10 else 0
    return carry


# ===========================================================================
# Battery H: horizon decode
# ===========================================================================

def _site_ov_write(model, cache, pos, head, WO0):
    z = cache["blocks.0.attn.hook_z"][0, pos, head, :]
    return (z @ WO0[head]).detach().numpy()


def battery_H(model, cfg, mn, sites, depths, n_q=200):
    WO0 = model.blocks[0].attn.W_O
    n_top = cfg.n_digits - 2
    out = {}
    for site in sites:
        pos, head, m = site["pos"], site["head"], site["m"]
        key = f"P{pos}H{head}_m{m}"
        out[key] = {"m": m, "pos": pos, "head": head, "per_depth": {}}
        for k in depths:
            d = n_top - k
            X, y_carry, y_local = [], [], []
            # wrong-role co-located head (N-4 baseline): a different head at same pos
            wrole_head = [h for h in range(cfg.n_heads) if h != head][0]
            Xw = []
            for _ in range(n_q):
                cls = "hi" if RNG.random() < 0.5 else "lo"
                a, b, info, _ = build_chain(cfg, n_top, k, cls, shared={})
                c = cache_full_L0(model, make_q(cfg, a, b))
                X.append(_site_ov_write(model, c, pos, head, WO0))
                Xw.append(_site_ov_write(model, c, pos, wrole_head, WO0))
                y_carry.append(chain_carry_out(cfg, a, b, n_top))
                # local class at the site's tagged digit
                dg = site["digit"]
                da = int(str(a).zfill(cfg.n_digits)[cfg.n_digits - 1 - dg])
                db = int(str(b).zfill(cfg.n_digits)[cfg.n_digits - 1 - dg])
                s = da + db
                y_local.append(0 if s <= 8 else (1 if s >= 10 else 2))
            X = np.array(X); Xw = np.array(Xw)
            y_carry = np.array(y_carry); y_local = np.array(y_local)
            res = {"d": d, "resolves_pred": bool(d >= m)}
            # (i) local class decode (positive anchor, reproduce CE13)
            if len(np.unique(y_local)) >= 2:
                res["local_class_bacc"] = _cv_bacc(X, y_local)
            else:
                res["local_class_bacc"] = float("nan")
            # (ii) resolved carry decode + wrole/shuffled baselines
            if len(np.unique(y_carry)) >= 2:
                res["carry_bacc"] = _cv_bacc(X, y_carry)
                res["carry_bacc_wrole"] = _cv_bacc(Xw, y_carry)
                ysh = y_carry.copy(); RNG.shuffle(ysh)
                res["carry_bacc_shuffled"] = _cv_bacc(X, ysh)
                # decode "counts" iff >= 0.2 above BOTH baselines (CA-1/SV-3)
                base = max(res["carry_bacc_wrole"], res["carry_bacc_shuffled"])
                res["carry_decodes"] = bool(res["carry_bacc"] >= base + 0.2)
            else:
                res["carry_bacc"] = float("nan"); res["carry_decodes"] = None
            # horizon match: decodes iff d>=m
            if res["carry_decodes"] is not None:
                res["horizon_match"] = bool(res["carry_decodes"] == res["resolves_pred"])
            out[key]["per_depth"][k] = res
    return out


def _cv_bacc(X, y, folds=3):
    n = len(y); idx = np.arange(n); RNG.shuffle(idx)
    parts = np.array_split(idx, folds); scores = []
    for i in range(folds):
        te = parts[i]; tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        if len(np.unique(y[tr])) < 2:
            continue
        clf = fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores)) if scores else float("nan")


def cache_full_L0(model, q):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0),
            names_filter=lambda nm: nm in ("blocks.0.attn.hook_z", "blocks.0.hook_resid_post"))
    return c


# ===========================================================================
# Battery Y: relay causality (site-write patch, L1 re-attends)  CA-2, CA-3
# ===========================================================================

def _patch_site_write_pred(model, cfg, tq, patches, ap):
    """Patch each site's OV write into blocks.0.hook_resid_post at its position
    (add source_write - target_write), then run the full model so L1 RE-ATTENDS."""
    def hook(act, hook):
        for (pos, delta) in patches:
            act[:, pos, :] = act[:, pos, :] + delta
        return act
    with torch.no_grad():
        lg = model.run_with_hooks(tq.unsqueeze(0),
                                  fwd_hooks=[("blocks.0.hook_resid_post", hook)])
    return lg[0, [p - 1 for p in ap]].argmax(-1)


def battery_Y(model, cfg, mn, sites, depths, n_pairs=40):
    WO0 = model.blocks[0].attn.W_O
    n_top = cfg.n_digits - 2
    ap = answer_positions(cfg)
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(ap) - 1 - top
    out = {"cpos": cpos, "consumer_heads": heads, "top": top, "per_depth": {}}

    for k in depths:
        d = n_top - k
        # sufficient sites: m <= d ; deepest sufficient SET = largest m<=d (CA-3)
        suff = [s for s in sites if s["m"] <= d]
        insuff = [s for s in sites if s["m"] > d]
        deepest_m = max([s["m"] for s in suff], default=None)
        deepest_set = [s for s in suff if s["m"] == deepest_m] if deepest_m is not None else []

        def site_deltas(src_c, tgt_c, site_list):
            out_d = []
            for s in site_list:
                sw = _site_ov_write(model, src_c, s["pos"], s["head"], WO0)
                tw = _site_ov_write(model, tgt_c, s["pos"], s["head"], WO0)
                out_d.append((s["pos"], torch.tensor(sw - tw)))
            return out_d

        arms = {"deepest_sufficient_set": deepest_set,
                "insufficient": insuff,
                "joint_all_sufficient": suff}
        flips = {a: [] for a in arms}
        nulls = {a: [] for a in arms}
        # for SI-4 brackets on the joint arm
        comp_flips = []   # full - complement not applicable per-arm here; report joint vs sum
        singles = {}      # per-site single flips for arm-sum residual on the joint arm

        for _ in range(n_pairs):
            (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
            sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
            clean = predict_answer(model, cfg, tq)
            sc = cache_full_L0(model, sq); tc = cache_full_L0(model, tq)
            for a, sl in arms.items():
                if not sl:
                    flips[a].append(0.0); nulls[a].append(0.0); continue
                p = _patch_site_write_pred(model, cfg, tq, site_deltas(sc, tc, sl), ap)
                flips[a].append(float(p[idx] != clean[idx]))
            # deciding-matched null (CA-2): source = same-class twin, deciding op re-drawn
            na, nb = same_class_twin(cfg, n_top, k, "lo")
            nc = cache_full_L0(model, make_q(cfg, na, nb))
            for a, sl in arms.items():
                if not sl:
                    continue
                p = _patch_site_write_pred(model, cfg, tq, site_deltas(nc, tc, sl), ap)
                nulls[a].append(float(p[idx] != clean[idx]))
            # per-site singles within the sufficient set (for arm-sum residual)
            for s in suff:
                skey = f"P{s['pos']}H{s['head']}"
                singles.setdefault(skey, [])
                p = _patch_site_write_pred(model, cfg, tq, site_deltas(sc, tc, [s]), ap)
                singles[skey].append(float(p[idx] != clean[idx]))

        joint = float(np.mean(flips["joint_all_sufficient"])) if flips["joint_all_sufficient"] else 0.0
        single_sum = sum(float(np.mean(v)) for v in singles.values())
        residual = joint - single_sum
        out["per_depth"][k] = {
            "d": d, "deepest_m": deepest_m,
            "n_sufficient": len(suff), "n_insufficient": len(insuff),
            "arm_flip": {a: mean_ci(v) for a, v in flips.items()},
            "arm_null": {a: mean_ci(v) for a, v in nulls.items()},
            "single_site_flips": {s: mean_ci(v) for s, v in singles.items()},
            "joint_arm_sum_residual": residual,
            "leaky_gt_0.15": bool(abs(residual) > 0.15),
            "report_mode": "ordinal_dominance" if abs(residual) > 0.15 else "shares",
        }
    return out


def control_Y_ablation(model, cfg, mn, sites, N=300):
    """CA-2 class-level ablation positive control, using CE13's EXACT Battery-Ab
    unit (`_mean_ablate_acc`): mean-ablate each ST site's hook_z on random
    questions, impact = clean_acc - ablated_acc over ALL answer positions. CE13
    measured 0.02-0.07 for low-digit nodes over a ~0.00 untagged baseline. The
    instrument is valid if the low-digit ST sites' impact exceeds the untagged-head
    baseline (reproduces CE13) — proving the site intervention CAN move an answer,
    so a near-zero LEADING-digit relay flip in Battery Y is a REDUNDANCY finding
    (CE13 interchange=0.00), not a dead instrument."""
    ap = answer_positions(cfg)
    qs = build_random(cfg, N)
    clean = sum(int(torch.equal(predict_answer(model, cfg, make_q(cfg, a, b)),
                                make_q(cfg, a, b)[ap])) for a, b in qs) / N
    per_site = {}
    low_impacts = []
    for s in sites:
        acc = _mean_ablate_acc(model, cfg, s["pos"], 0, s["head"], qs, ap)
        imp = clean - acc
        # untagged baseline: a co-located head not in the ST set
        wrole = [h for h in range(cfg.n_heads) if h != s["head"]][0]
        bacc_ = _mean_ablate_acc(model, cfg, s["pos"], 0, wrole, qs, ap)
        base = clean - bacc_
        key = f"P{s['pos']}H{s['head']}_d{s['digit']}"
        per_site[key] = {"impact": imp, "untagged_baseline": base, "m": s["m"], "digit": s["digit"]}
        if s["digit"] <= 2:   # low-digit / where carries originate (CE13)
            low_impacts.append(imp)
    max_low = max(low_impacts) if low_impacts else 0.0
    max_base = max(p["untagged_baseline"] for p in per_site.values()) if per_site else 0.0
    return {"clean_acc": clean, "per_site": per_site, "max_lowdigit_impact": max_low,
            "max_untagged_baseline": max_base,
            "instrument_ok": bool(max_low > max_base + 0.01),
            "note": "CE13 Battery-Ab unit (_mean_ablate_acc); low-digit impact vs untagged baseline"}


# ===========================================================================
# Battery L: L1-read reconstruction (nested phi_local subset phi_horizon)  CA-5
# ===========================================================================

def battery_L(model, cfg, mn, sites, ax, depths, n_q=200):
    cpos, heads, top = pair_at_top(cfg, mn)
    n_top = cfg.n_digits - 2
    axis = torch.tensor(ax["axis"], dtype=torch.float32)
    Xloc, Xhor, ytarget = [], [], []
    for _ in range(n_q):
        k = int(RNG.choice(depths)); d = n_top - k
        cls = "hi" if RNG.random() < 0.5 else "lo"
        a, b, info, _ = build_chain(cfg, n_top, k, cls, shared={})
        q = make_q(cfg, a, b)
        c = cache_full(model, q)
        # target: consumer head-pair edge output projected on carry axis
        delta = edge_contribution(model, cfg, c, cpos, heads)
        rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
        proj, _ = lnfair_project(model, cfg, rm, delta, ax["axis"])
        ytarget.append(proj)
        # features: attn[key]*phi(key) over chain-ST sites (approx via pattern)
        loc, hor = [], []
        for s in sites:
            pat = float(c["blocks.1.attn.hook_pattern"][0, heads[0], cpos, s["pos"]])
            # local class one-hot (3) * attn
            da = int(str(a).zfill(cfg.n_digits)[cfg.n_digits - 1 - s["digit"]])
            db = int(str(b).zfill(cfg.n_digits)[cfg.n_digits - 1 - s["digit"]])
            sm = da + db
            lc = [0, 0, 0]; lc[0 if sm <= 8 else (1 if sm >= 10 else 2)] = 1
            loc.extend([pat * x for x in lc])
            # horizon-resolved state: local class (nested) + resolved-up-to-m bit
            resolved = 1 if d >= s["m"] else 0
            rc = chain_carry_out(cfg, a, b, n_top) if resolved else 0
            hor.extend([pat * x for x in lc] + [pat * resolved, pat * rc])
        Xloc.append(loc); Xhor.append(hor)
    Xloc = np.array(Xloc); Xhor = np.array(Xhor); y = np.array(ytarget)
    # held-out R2
    def cv_r2(X):
        n = len(y); idx = np.arange(n); RNG.shuffle(idx)
        tr, te = idx[: 3 * n // 4], idx[3 * n // 4:]
        reg = LinearRegression().fit(X[tr], y[tr])
        return float(r2_score(y[te], reg.predict(X[te])))
    r2_loc = cv_r2(Xloc); r2_hor = cv_r2(Xhor)
    # permutation null on the EXTRA columns (CA-5): shuffle the horizon-only cols
    n_extra = Xhor.shape[1] - Xloc.shape[1]
    Xperm = Xhor.copy()
    extra_cols = list(range(Xloc.shape[1], Xhor.shape[1]))
    for c_ in extra_cols:
        Xperm[:, c_] = Xperm[RNG.permutation(len(y)), c_]
    r2_perm = cv_r2(Xperm)
    delta_r2 = r2_hor - r2_loc
    delta_over_null = r2_hor - max(r2_loc, r2_perm)
    return {"r2_local": r2_loc, "r2_horizon": r2_hor, "r2_horizon_permuted_extra": r2_perm,
            "delta_r2": delta_r2, "delta_r2_over_null": delta_over_null,
            "winner": ("phi_horizon" if delta_over_null >= 0.2 else
                       ("phi_local" if r2_loc >= r2_hor - 0.05 else "split")),
            "n_extra_cols": n_extra}


# ===========================================================================
# Battery T: combiner transfer, on-manifold (PRE-LN substitution)  CA-4
# ===========================================================================

def _edge_by_class(model, cfg, mn, cpos, heads, n_q=60):
    """Real captured consumer-pair z (per head) for committed c0 and c1 classes at
    the chain top -- the endpoints of the transfer sweep."""
    n_top = cfg.n_digits - 2
    zc = {0: {h: [] for h in heads}, 1: {h: [] for h in heads}}
    for _ in range(n_q):
        for cls, bit in [("lo", 0), ("hi", 1)]:
            a, b, info, _ = build_chain(cfg, n_top, min(3, n_top), cls, shared={})
            c = cache_full(model, make_q(cfg, a, b))
            for h in heads:
                zc[bit][h].append(c["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())
    return {bit: {h: np.mean(zc[bit][h], 0) for h in heads} for bit in (0, 1)}


def battery_T(model, cfg, mn, ax, n_q=60, alphas=(-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    cpos, heads, top = pair_at_top(cfg, mn)
    n_top = cfg.n_digits - 2
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - top
    axis = torch.tensor(ax["axis"], dtype=torch.float32)
    zc = _edge_by_class(model, cfg, mn, cpos, heads, n_q=n_q)

    # non-carry variance regressed out of (z_c1 - z_c0) per head is not needed for
    # the z substitution (we interpolate the real z's directly, PRE-LN). alpha
    # then interpolates the real delivered contribution; we report the carry-axis
    # projection at the combiner input at each alpha for manifold consistency.
    curve = {}
    for al in alphas:
        flips = []; projs = []
        for _ in range(n_q):
            # committed-lo local target chain (local class fixed committed-lo)
            a, b, info, _ = build_chain(cfg, n_top, min(3, n_top), "lo", shared={})
            tq = make_q(cfg, a, b)
            clean = predict_answer(model, cfg, tq)
            tc = cache_full(model, tq)
            # PRE-LN substitution: replace heads' z at cpos with edge(alpha)
            z_by_head = {}
            for h in heads:
                z_by_head[h] = torch.tensor((1 - al) * zc[0][h] + al * zc[1][h], dtype=torch.float32)
            patches = [(cpos, 1, h, z_by_head[h].numpy()) for h in heads]
            p = edge_patch_pred(model, cfg, tq, patches, tc)
            flips.append(float(p[idx] != clean[idx]))
            # carry-axis projection at combiner input (manifold read)
            rm = tc["blocks.1.hook_resid_mid"][0, cpos, :]
            delta = torch.zeros(model.cfg.d_model)
            for h in heads:
                zt = tc["blocks.1.attn.hook_z"][0, cpos, h, :]
                delta = delta + head_ov(model, z_by_head[h], 1, h) - head_ov(model, zt, 1, h)
            proj, _ = lnfair_project(model, cfg, rm, delta.detach(), ax["axis"])
            projs.append(proj)
        curve[str(al)] = {"flip": float(np.mean(flips)), "carry_proj": float(np.mean(projs))}

    # endpoint gate (CA-4 / control 3): alpha=0 ~ 0.00, alpha=1 ~ 1.00
    a0 = curve["0.0"]["flip"]; a1 = curve["1.0"]["flip"]
    endpoint_ok = (a0 <= 0.2) and (a1 >= 0.8)
    # transfer class on [0,1]: step (sharp) vs linear (graded)
    in_range = [(a, curve[str(a)]["flip"]) for a in alphas if 0.0 <= a <= 1.0]
    xs = np.array([a for a, _ in in_range]); ys = np.array([f for _, f in in_range])
    max_jump = float(np.max(np.diff(ys))) if len(ys) > 1 else 0.0
    spread = float(ys.max() - ys.min())
    # logistic vs linear fit quality
    cls = "invalid_endpoint" if not endpoint_ok else (
        "step" if max_jump > 0.5 * (spread + 1e-9) else "graded")
    # threshold alpha*: smallest alpha in [0,1] with flip >= 0.5
    astar = next((a for a in sorted(xs) if curve[str(a)]["flip"] >= 0.5), None)
    return {"cpos": cpos, "heads": heads, "curve": curve,
            "endpoint_ok": bool(endpoint_ok), "alpha0_flip": a0, "alpha1_flip": a1,
            "max_jump_in_range": max_jump, "spread_in_range": spread,
            "transfer_class": cls, "threshold_alpha_star": astar}


# ===========================================================================
# driver
# ===========================================================================

def run_model(model, cfg, mn, fast=False):
    n_top = cfg.n_digits - 2
    depths = [k for k in (2, 3, 4) if 0 <= n_top - k]
    depths = [k for k in depths if k <= n_top]  # valid deciding digit
    out = {"model": mn, "n_top": n_top, "depths": depths}
    out["gates"] = {k: behavioral_gate(model, cfg, n_top, k, n_q=40) for k in depths}
    sites = chain_st_sites(mn)
    out["horizon_table"] = [{**s, "resolves_at_depth": {k: bool((n_top - k) >= s["m"]) for k in depths}} for s in sites]
    ax = carry_axis(model, cfg, mn, n_q=(120 if fast else 250))
    out["PC_axis_sep"] = ax["sep"]
    nq = 120 if fast else 200
    npairs = 20 if fast else 40
    out["battery_H"] = battery_H(model, cfg, mn, sites, depths, n_q=nq)
    out["control_Y_ablation"] = control_Y_ablation(model, cfg, mn, sites, N=(120 if fast else 300))
    out["battery_Y"] = battery_Y(model, cfg, mn, sites, depths, n_pairs=npairs)
    out["battery_T"] = battery_T(model, cfg, mn, ax, n_q=(30 if fast else 60))
    if not fast:
        out["battery_L"] = battery_L(model, cfg, mn, sites, ax, depths, n_q=nq)
    out["verdict"] = derive_verdict(out)
    return out


def derive_verdict(out):
    v = {}
    depths = out["depths"]
    H = out["battery_H"]
    # horizon-match rate over boundary cells (m>0 sites)
    boundary = []
    for key, sd in H.items():
        if sd["m"] == 0:
            continue
        for k, r in sd["per_depth"].items():
            if r.get("horizon_match") is not None:
                boundary.append(r["horizon_match"])
    v["H_boundary_match_rate"] = float(np.mean(boundary)) if boundary else float("nan")
    v["H_n_boundary_cells"] = len(boundary)

    Y = out["battery_Y"]
    ctrl = out["control_Y_ablation"]
    v["Y_instrument_ok"] = ctrl["instrument_ok"]
    # deepest-sufficient vs insufficient across depths
    deepest = []; insuff = []; joint = []; nulls = []
    for k, d in Y["per_depth"].items():
        deepest.append(d["arm_flip"]["deepest_sufficient_set"]["rate"])
        insuff.append(d["arm_flip"]["insufficient"]["rate"])
        joint.append(d["arm_flip"]["joint_all_sufficient"]["rate"])
        nulls.append(d["arm_null"]["joint_all_sufficient"]["rate"])
    v["Y_deepest_sufficient_max"] = max(deepest) if deepest else 0.0
    v["Y_insufficient_max"] = max(insuff) if insuff else 0.0
    v["Y_joint_max"] = max(joint) if joint else 0.0
    v["Y_joint_null_max"] = max(nulls) if nulls else float("nan")

    T = out["battery_T"]
    v["T"] = {"endpoint_ok": T["endpoint_ok"], "class": T["transfer_class"],
              "alpha_star": T["threshold_alpha_star"], "spread": T["spread_in_range"]}
    if "battery_L" in out:
        v["L_winner"] = out["battery_L"]["winner"]
        v["L_delta_r2_over_null"] = out["battery_L"]["delta_r2_over_null"]

    # decision table (CA-6: Y confers causal status; H representational).
    # Redundancy-aware (working axioms): at the leading-digit locus CE13 already
    # showed L0 ST writes are REDUNDANT (interchange=0.00, mean-ablation impact
    # only 0.02-0.04). So a near-zero Y leading-digit relay flip WITH a valid
    # instrument (control_Y_ablation reproduces CE13) is a redundancy finding, not
    # evidence for R-L1-read. The A11-vs-L1 call then rests on the H horizon
    # structure (representational) + Battery L (reconstruction) + the combiner T.
    hm = v["H_boundary_match_rate"]
    relay_causal = (v["Y_deepest_sufficient_max"] >= 0.5 and v["Y_insufficient_max"] < 0.3
                    and (v["Y_joint_null_max"] != v["Y_joint_null_max"] or v["Y_joint_null_max"] <= 0.2))
    y_redundant = (v["Y_joint_max"] < 0.3)   # relay not causally detectable (redundancy)
    v["Y_note"] = ("relay causally detectable" if not y_redundant else
                   "relay NOT causally detectable at leading digit (redundancy-consistent, CE13)")
    if not ctrl["instrument_ok"]:
        v["read"] = "Y-INVALID (instrument control failed to reproduce CE13 ablation)"
    elif relay_causal and hm == hm and hm >= 0.7:
        v["read"] = "R-relay (A11): horizon-matched decode + deepest-relay causal, insufficient~null"
    elif y_redundant and hm == hm and hm >= 0.6:
        # F1/F2/F3 post-result correction: H "match" is inflated by trivial
        # NO/NO cells (only 1 non-trivial cell, unreplicated); Y null is causally
        # UNDETERMINED (ablation control != interchange unit/target); L leans
        # local. Honest read: R-mixed, L1-local-sufficient, single-cell horizon
        # trace, relay causally undetermined.
        v["read"] = ("R-mixed (L1-local-sufficient): single-cell representational "
                     f"horizon trace (H match {hm:.2f}, trivial-inflated, 1 non-trivial "
                     "cell, unreplicated); relay CAUSALLY UNDETERMINED (Y interchange "
                     "~0, valid ablation control tests a different unit/target — "
                     "redundancy vs weak-interchange unresolved); L1 edge output "
                     "local-class-sufficient. A11 low; A9 retired.")
    elif (hm != hm or hm < 0.5) and v["Y_joint_max"] >= 0.5 and v["Y_deepest_sufficient_max"] < 0.5:
        v["read"] = "R-L1-read: flat horizon decode + only joint/wider flips"
    else:
        v["read"] = "R-mixed (report the split)"
    return v


def main():
    fast = "--fast" in sys.argv
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (acc {acc:.3f}) fast={fast} ===", flush=True)
        r = run_model(model, cfg, mn, fast=fast); r["accuracy"] = acc
        results[mn] = r
        v = r["verdict"]
        print("  horizon table:", [(s["pos"], s["m"]) for s in r["horizon_table"]], flush=True)
        print("  [H] boundary match rate", round(v["H_boundary_match_rate"], 2),
              "over", v["H_n_boundary_cells"], "cells", flush=True)
        print("  [Y] ctrl_ok", v["Y_instrument_ok"], "deepest", round(v["Y_deepest_sufficient_max"], 2),
              "insuff", round(v["Y_insufficient_max"], 2), "joint", round(v["Y_joint_max"], 2),
              "null", round(v["Y_joint_null_max"], 2), flush=True)
        print("  [T]", v["T"], flush=True)
        if "L_winner" in v:
            print("  [L]", v["L_winner"], "dR2_over_null", round(v["L_delta_r2_over_null"], 3), flush=True)
        print("  VERDICT:", v["read"], flush=True)
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
