"""Geometry certificate: place-value dominance code on the carry rail
(study-geometry-certificate.md).

Scores G2 (dominance code + STEP-makes-TriAdd) and G3 (dynamic-range law).
Working-axioms mode: the SV mechanism is settled (ST writers -> L1 consumer read
where the canonical carry emerges, CE24 -> STEP combiner, CE17). This study asks
the GEOMETRIC question: what arrangement of per-site write manifolds makes a
mostly-static attention average yield a depth-invariant resolved carry?

Core measurement: per-prompt LN-fair OV decomposition of the combiner input at a
consumer's consuming position, along the CE16/CE17 carry rail, into per-source-
site contributions. Under a frozen (this-prompt) LN std the map delta->rail-proj
is LINEAR, so per-site contributions are exactly additive (faithfulness checked).

Batteries:
  T1 ordering & transparency : per-site class-conditional rail values
     p_i (local 0) < u_i (U) < q_i (local 1), with the CE13 cin-split
     u_i(cin0) < u_i(cin1), both inside (p_i, q_i). Midpointness index.
  T2 dominance profile       : g_i = folded class gap; super-increasing test
     g_i vs sum_{j<i} g_j in digit significance. T2b: above-digit sites ~0.
  T3 additivity / reconstruct: predict alpha on held-out real prompts from the
     STATIC per-site class values + measured per-prompt attention; R2 + the
     threshold-agreement of 1[alpha_hat > alpha*] with the model's carry-out.
  T4 certificate             : from measured (p_i, u_i+-d, q_i) + attention,
     enumerate reachable class configs, compute the feasible threshold interval
     for exact TriAdd at depth <= k; is it non-empty and does alpha* sit inside?
  T5 (stretch, G3)           : gap profiles g_i ~ rho^i at d10/d13; smallest gap
     vs noise floor; relate to CE18's carry-axis sep collapse (30 -> 6).

Positive controls:
  PC1 reproduce CE16 joint head-pair edge flip 1.00 / deciding-matched null 0.00.
  PC2 rail separates committed c0/c1 at the CE16/CE17 magnitude.
  PC3 wrong-axis null: a random unit rail must destroy T1 ordering / T2 structure.
  PC-faithful: sum of per-site rail contributions + bias reconstructs measured
     alpha (linearity of the frozen-std decomposition).

CPU-only. Run:
    PYTHONPATH=. python3 scripts/geometry_certificate.py all
    PYTHONPATH=. python3 scripts/geometry_certificate.py all --fast
    PYTHONPATH=. python3 scripts/geometry_certificate.py t5      # G3 large-n only
"""
from __future__ import annotations
import json, os, sys, math, itertools
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    _digits_to_int,
)
from scripts.deep_cascade_mechanism import (
    build_chain, consuming_pos, ak_pos, affected_digits, dn_pos, dpn_pos,
    behavioral_gate, RNG,
)
from scripts.sv_implementation import (
    pair_at_top, carry_axis, twin_pair, same_class_twin, edge_contribution,
    cache_full, _cache_named, mean_ci,
)
from scripts.sv_compounding import CONSUMER_HEADS, EQ_POS, head_ov, edge_patch_pred, cache_qs
from scripts.st_tristate_geometry import CFG as GEO_CFG
from scripts.node_output_encoding import ST_NODES
from quanta_maths.maths_edge_patch import ln_scale as _ln_norm

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-geometry-certificate")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716

MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]
LARGE_MODELS = ["add_d10_l2_h3_t40K_s572091", "add_d13_l2_h3_t50K_s572091"]

# cache names needed for the per-prompt decomposition
DEC_NAMES = ("blocks.1.hook_resid_mid", "blocks.1.hook_resid_pre",
             "blocks.1.attn.hook_v", "blocks.1.attn.hook_pattern",
             "blocks.1.attn.hook_z", "blocks.1.ln2.hook_normalized")


# ===========================================================================
# consumers per model: (name, cpos, heads, served_digit)
# ===========================================================================

def consumers_for(cfg, mn):
    """The sign-position (leading-digit) consumer plus one middle-digit consumer,
    both taken from the map-named CONSUMER_HEADS registry (do not re-locate)."""
    ch = CONSUMER_HEADS[mn]
    # leading-digit consumer (full-depth cascade, CE15)
    cpos_top, heads_top, top = pair_at_top(cfg, mn)
    cons = [{"name": "leading", "cpos": cpos_top, "heads": heads_top, "digit": top}]
    # middle-digit consumer: the served digit closest to floor(n_top/2), pref a pair
    n_top = cfg.n_digits - 2
    by_digit = {}
    for (pos, h, kk) in ch:
        by_digit.setdefault(kk, {"pos": pos, "heads": set()})
        by_digit[kk]["heads"].add(h)
    mids = [d for d in by_digit if d < top]
    if mids:
        target = max(mids, key=lambda d: (len(by_digit[d]["heads"]), -abs(d - n_top // 2)))
        info = by_digit[target]
        cons.append({"name": "middle", "cpos": info["pos"],
                     "heads": sorted(info["heads"]), "digit": target})
    return cons


def st_source_sites(mn):
    """L0 ST writer sites available as sources (pos, head, digit)."""
    return [{"pos": p, "head": h, "digit": d}
            for (p, L, h, d) in ST_NODES[mn] if L == 0]


def consumer_local_carry_axis(model, cfg, consumer, n_q=200):
    """Skeptic-gate control: refit the carry rail at THIS consumer's own consuming
    position, using the DELIVERED carry (carry INTO the consumer's digit) toggled by
    twin chains — carry-in=1 (deciding hi) vs carry-in=0 (deciding lo). This is the
    natural rail for that consumer; comparing per-site gaps on it vs the shared geo
    rail separates genuine gap compression from rail-misalignment (G1 rotation).
    Returns None if the consumer's digit is too low to build a delivering chain."""
    dg = consumer["digit"]; cpos = consumer["cpos"]
    n = dg - 1                              # chain top; carry OUT of n = carry INTO dg
    if n < 1:
        return None
    k = min(3, n)
    hook = "blocks.1.ln2.hook_normalized"
    means = {}
    for cls, bit in [("lo", 0), ("hi", 1)]:
        acts = []
        for _ in range(n_q):
            a, b, _, _ = build_chain(cfg, n, k, cls, shared={})
            c = _cache_named(model, make_q(cfg, a, b), (hook,))
            acts.append(c[hook][0, cpos, :].numpy())
        means[cls] = np.mean(acts, 0)
    axis = means["hi"] - means["lo"]
    sep = float(np.linalg.norm(axis))
    axis = axis / (sep + 1e-9)
    return {"axis": axis, "c0": means["lo"], "c1": means["hi"], "sep": sep,
            "pos": cpos, "digit": dg, "chain_top": n, "k": k}


def _rand_le8():
    while True:
        a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
        if a + b <= 8:
            return a, b


def build_site_class_question(cfg, d, cls, cin=None):
    """Isolate digit `d` in local class `cls` in {'c0','u0','u1','c1'} while keeping
    every OTHER digit a clean committed non-carry (sum<=8, never 9). Handles d==0
    (no lower digit -> cin forced 0). u0/u1 set the incoming single-step carry via
    digit d-1 (CE13). Returns (a, b, sa_d)."""
    nd = cfg.n_digits
    idx = nd - 1 - d
    d1 = [0] * nd; d2 = [0] * nd
    # digit d itself
    if cls in ("u0", "u1"):
        a = int(RNG.integers(0, 10)); b = 9 - a
        want_cin = (cls == "u1")
    elif cls == "c0":
        a, b = _rand_le8(); want_cin = bool(RNG.integers(0, 2))
    elif cls == "c1":
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b >= 10:
                break
        want_cin = bool(RNG.integers(0, 2))
    else:
        raise ValueError(cls)
    d1[idx] = a; d2[idx] = b
    # incoming carry via digit d-1
    if d >= 1:
        ikm = nd - 1 - (d - 1)
        if want_cin:
            while True:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                if x + y >= 10:
                    break
        else:
            x, y = _rand_le8()
        d1[ikm] = x; d2[ikm] = y
        low_start = d - 1
    else:
        low_start = 0  # d==0: no lower carry possible
    # remaining lower digits: clean committed non-carry
    for k in range(0, low_start):
        ik = nd - 1 - k
        x, y = _rand_le8(); d1[ik] = x; d2[ik] = y
    # digits above d: 0 (already)
    return _digits_to_int(d1), _digits_to_int(d2), (a + b) % 10


# ===========================================================================
# per-prompt LN-fair OV decomposition of the combiner input along the rail
# ===========================================================================

def _cache_dec(model, q):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm in DEC_NAMES)
    return c


@torch.no_grad()
def decompose(model, cfg, cpos, heads, ax, cache):
    """Per-prompt additive decomposition of the combiner input at `cpos` along the
    carry rail `ax`. Returns dict:
      alpha            : measured rail coordinate of the real combiner input
      per_key          : {key_pos: rail contribution}
      w_key            : {key_pos: summed attention over the head pair}
      skip             : direct/skip (resid_pre) rail contribution
      bias             : -(c0.chat)/sep  (constant)
      recon_err        : |alpha - (sum per_key + skip + bias)|  (faithfulness)
    All contributions are exactly additive under this prompt's frozen LN std.
    """
    chat = torch.tensor(ax["axis"], dtype=torch.float32)
    sep = ax["sep"]
    c0 = torch.tensor(ax["c0"], dtype=torch.float32)
    gamma = model.blocks[1].ln2.w
    rm = cache["blocks.1.hook_resid_mid"][0, cpos, :]
    _, std = _ln_norm(rm)

    def rail(part):
        # f_vec(part) = gamma * (part - mean(part)) / std ; project on chat / sep
        fv = gamma * (part - part.mean()) / std
        return float((fv @ chat) / sep)

    # real combiner input (actual ln2.hook_normalized) -> measured alpha
    x = cache["blocks.1.ln2.hook_normalized"][0, cpos, :]
    alpha = float(((x - c0) @ chat) / sep)
    bias = float(-(c0 @ chat) / sep)

    # per-key OV contribution summed over the consumer head pair
    pattern = cache["blocks.1.attn.hook_pattern"][0]      # [head, q, k]
    vv = cache["blocks.1.attn.hook_v"][0]                 # [k, head, d_head]
    seqk = pattern.shape[-1]
    per_key = {}
    w_key = {}
    for key in range(seqk):
        part = torch.zeros(model.cfg.d_model)
        wsum = 0.0
        for h in heads:
            p = float(pattern[h, cpos, key])
            if p == 0.0:
                continue
            part = part + head_ov(model, p * vv[key, h, :], 1, h)
            wsum += p
        if wsum == 0.0 and float(part.abs().sum()) == 0.0:
            continue
        per_key[key] = rail(part)
        w_key[key] = wsum
    # non-site baseline (everything NOT the consumer head-pair's per-site delivery):
    #   resid_pre (direct/skip path into L1) + attn output bias b_O + LN2 bias +
    #   the OTHER (non-consumer) heads' attention output at this position.
    skip = rail(cache["blocks.1.hook_resid_pre"][0, cpos, :])
    skip = skip + rail(model.blocks[1].attn.b_O)
    ln_b = model.blocks[1].ln2.b
    if ln_b is not None:
        skip = skip + float((ln_b @ chat) / sep)
    z_all = cache["blocks.1.attn.hook_z"][0, cpos]           # [head, d_head]
    for h in range(model.cfg.n_heads):
        if h in heads:
            continue
        skip = skip + rail(head_ov(model, z_all[h, :], 1, h))

    recon = sum(per_key.values()) + skip + bias
    return {"alpha": alpha, "per_key": per_key, "w_key": w_key, "skip": skip,
            "bias": bias, "recon_err": abs(alpha - recon)}


# ===========================================================================
# T1: ordering & transparency (per-site class-conditional rail values)
# ===========================================================================

def battery_T1(model, cfg, mn, consumer, sites, ax, n_q=200):
    """For each source site, vary ITS served digit's local class in {c0,u0,u1,c1}
    (u0/u1 = U with/without incoming single-step carry, CE13) and record the
    folded rail contribution at the consumer. Returns per-site p/u0/u1/q with CIs."""
    cpos, heads = consumer["cpos"], consumer["heads"]
    # digits that at least one source site serves and that are BELOW the consumer
    site_digits = sorted({s["digit"] for s in sites if s["digit"] < consumer["digit"]})
    # collect contributions: for each varied digit d and class, cache & decompose,
    # then read contrib at every site (only sites serving d vary meaningfully).
    # store per (site_pos, site_head) -> class -> [contribs]
    store = {(s["pos"], s["head"]): {"c0": [], "u0": [], "u1": [], "c1": [],
                                     "digit": s["digit"]} for s in sites}
    w_store = {(s["pos"], s["head"]): [] for s in sites}
    for d in site_digits:
        for cls in ("c0", "u0", "u1", "c1"):
            for _ in range(n_q):
                a, b, _sa = build_site_class_question(cfg, d, cls)
                cache = _cache_dec(model, make_q(cfg, a, b))
                dec = decompose(model, cfg, cpos, heads, ax, cache)
                for s in sites:
                    if s["digit"] != d:
                        continue
                    key = (s["pos"], s["head"])
                    store[key][cls].append(dec["per_key"].get(s["pos"], 0.0))
                    w_store[key].append(dec["w_key"].get(s["pos"], 0.0))
    out = {}
    for s in sites:
        key = (s["pos"], s["head"])
        st = store[key]
        if not st["c0"]:
            continue
        p = float(np.mean(st["c0"])); q = float(np.mean(st["c1"]))
        u0 = float(np.mean(st["u0"])); u1 = float(np.mean(st["u1"]))
        u = 0.5 * (u0 + u1)
        def ci(v):
            return list(_num_ci(v))
        lo, hi = (p, q) if p <= q else (q, p)
        span = hi - lo + 1e-9
        midpoint = abs(u - 0.5 * (p + q)) / span
        out[f"P{s['pos']}H{s['head']}_d{s['digit']}"] = {
            "digit": s["digit"], "w_mean": float(np.mean(w_store[key])),
            "p": p, "u0": u0, "u1": u1, "u": u, "q": q,
            "p_ci": ci(st["c0"]), "u0_ci": ci(st["u0"]), "u1_ci": ci(st["u1"]),
            "q_ci": ci(st["c1"]),
            # ordering: p < u < q (transparency: U strictly between committed values)
            "ordered": bool((p < u < q) or (q < u < p)),
            "u_inside": bool(min(p, q) < u < max(p, q)),
            "cin_split": bool((u1 - u0) * (q - p) > 0),  # u leans toward carry side
            "cin_split_mag": float((u1 - u0)),
            "midpointness": float(midpoint),
            "class_gap": float(q - p),
        }
    return out


def _num_ci(vals, z=1.96):
    a = np.asarray(vals, float); n = len(a)
    if n == 0:
        return (float("nan"), float("nan"))
    m = a.mean(); se = a.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    return (float(m - z * se), float(m + z * se))


# ===========================================================================
# T2: dominance profile (super-increasing weighted gaps in significance)
# ===========================================================================

def battery_T2(t1, consumer, sites):
    """Order sites by digit significance; test each site's folded class gap g_i
    against the running sum of all lower-significance gaps (place-value dominance).
    T2b: sites ABOVE the consumer digit must carry ~0 class-dependent gap."""
    # aggregate per digit (sum over redundant same-digit sites, since the consumer
    # sums their contributions)
    per_digit = {}
    for kkey, v in t1.items():
        d = v["digit"]
        per_digit.setdefault(d, 0.0)
        per_digit[d] += abs(v["class_gap"])
    digits = sorted(per_digit)          # ascending significance
    running = 0.0
    profile = []
    for d in digits:
        g = per_digit[d]
        dominates = bool(g >= running)   # super-increasing: g_i >= sum_{j<i} g_j
        profile.append({"digit": d, "g": g, "sum_below": running,
                        "dominates": dominates, "margin": g - running})
        running += g
    # top >=3 sites dominance (the study success key for the leading consumer)
    top3 = profile[-3:] if len(profile) >= 3 else profile
    return {"per_digit_gap": per_digit, "profile": profile,
            "all_dominate": all(p["dominates"] for p in profile),
            "top3_dominate": all(p["dominates"] for p in top3),
            "n_digits": len(digits)}


# ===========================================================================
# T3: additivity / reconstruction on held-out real prompts
# ===========================================================================

def _local_class(a, b, nd, d):
    da = int(str(a).zfill(nd)[nd - 1 - d]); db = int(str(b).zfill(nd)[nd - 1 - d])
    s = da + db
    return 0 if s <= 8 else (1 if s >= 10 else 2)


def _cin_at(a, b, nd, d):
    """incoming single-step carry into digit d (from digit d-1 chain)."""
    da = [int(x) for x in str(a).zfill(nd)]; db = [int(x) for x in str(b).zfill(nd)]
    c = 0
    for k in range(d):
        c = 1 if (da[nd - 1 - k] + db[nd - 1 - k] + c) >= 10 else 0
    return c


def _true_carry_into(a, b, nd, digit):
    """resolved carry INTO `digit` (== carry_out of digit-1)."""
    return _cin_at(a, b, nd, digit)


def gather_holdout(model, cfg, mn, consumer, sites, ax, t1, n_q=250):
    """Collect held-out REAL prompts: measured alpha (rail proj of the combiner
    input), the STATIC-class reconstruction (T1 means + static attention),
    the ATTENTION-rescaled reconstruction, and the true carry-out into the
    consumer's digit. Returns arrays for the threshold/reconstruction arbiter."""
    cpos, heads = consumer["cpos"], consumer["heads"]
    nd = cfg.n_digits; n_top = cfg.n_digits - 2
    table = {}
    for kkey, v in t1.items():
        table[(int(kkey.split("P")[1].split("H")[0]), int(kkey.split("H")[1].split("_")[0]))] = v
    site_by_ph = {(s["pos"], s["head"]): s for s in sites}
    lim = 10 ** nd

    def draw():
        if RNG.random() < 0.5:
            return int(RNG.integers(0, lim // 2)), int(RNG.integers(0, lim // 2))
        k = int(RNG.integers(1, n_top + 1))
        cls = "hi" if RNG.random() < 0.5 else "lo"
        a, b, _, _ = build_chain(cfg, n_top, k, cls, shared={})
        return a, b

    alphas, ahat_static, ahat_attn, true_c, recon_errs = [], [], [], [], []
    for _ in range(n_q):
        a, b = draw()
        dec = decompose(model, cfg, cpos, heads, ax, _cache_dec(model, make_q(cfg, a, b)))
        alphas.append(dec["alpha"]); recon_errs.append(dec["recon_err"])
        base = dec["bias"] + dec["skip"]
        s_stat = base; s_attn = base
        for (pos, hd), s in site_by_ph.items():
            if s["digit"] >= consumer["digit"]:
                continue
            v = table.get((pos, hd))
            if v is None:
                continue
            cls = _local_class(a, b, nd, s["digit"])
            if cls == 0:
                val = v["p"]
            elif cls == 1:
                val = v["q"]
            else:
                val = v["u1"] if _cin_at(a, b, nd, s["digit"]) else v["u0"]
            s_stat += val                              # static attention (folded in T1)
            w_meas = dec["w_key"].get(pos, 0.0); w_t1 = v["w_mean"] + 1e-9
            s_attn += val * (w_meas / w_t1)            # rescale to per-prompt attention
        ahat_static.append(s_stat); ahat_attn.append(s_attn)
        true_c.append(_true_carry_into(a, b, nd, consumer["digit"]))
    return {"alpha": np.array(alphas), "ahat_static": np.array(ahat_static),
            "ahat_attn": np.array(ahat_attn), "true_carry": np.array(true_c),
            "mean_recon_err": float(np.mean(recon_errs)), "max_recon_err": float(np.max(recon_errs))}


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
    return 1.0 - ss_res / ss_tot


def _best_threshold(alpha, y):
    """Threshold on alpha maximizing balanced accuracy of 1[alpha>t]==y."""
    if len(np.unique(y)) < 2:
        return None, float("nan")
    cands = np.unique(alpha)
    mids = (cands[:-1] + cands[1:]) / 2.0 if len(cands) > 1 else cands
    best_t, best_b = None, -1.0
    for t in mids:
        pred = (alpha > t).astype(int)
        tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
        tpr = tp / (tp + fn + 1e-9); tnr = tn / (tn + fp + 1e-9)
        b = 0.5 * (tpr + tnr)
        if b > best_b:
            best_b, best_t = b, float(t)
    return best_t, float(best_b)


def threshold_and_reconstruction(hold):
    """Empirical alpha* (rail STEP threshold the combiner operates at) + the T3
    reconstruction quality. alpha* is derived from the model's own measured alpha
    vs the true carry-out (a rail coordinate, coordinate-consistent with T4)."""
    alpha = hold["alpha"]; y = hold["true_carry"]
    astar, sep_bacc = _best_threshold(alpha, y)
    # frame-consistent threshold in the static-reconstruction coordinate (for T4)
    astar_static, sep_bacc_static = _best_threshold(hold["ahat_static"], y)
    r2_static = _r2(alpha, hold["ahat_static"])
    r2_attn = _r2(alpha, hold["ahat_attn"])
    # threshold agreement: 1[reconstruction > alpha*] matches true carry
    if astar is not None:
        agree_static = float(np.mean((hold["ahat_static"] > astar).astype(int) == y))
        agree_attn = float(np.mean((hold["ahat_attn"] > astar).astype(int) == y))
        model_agree = float(np.mean((alpha > astar).astype(int) == y))
    else:
        agree_static = agree_attn = model_agree = float("nan")
    # step sharpness: separation of alpha | carry=0 vs carry=1 (Cohen's d style)
    a0 = alpha[y == 0]; a1 = alpha[y == 1]
    if len(a0) > 1 and len(a1) > 1:
        pooled = math.sqrt(0.5 * (a0.var() + a1.var()) + 1e-12)
        cohen_d = float((a1.mean() - a0.mean()) / (pooled + 1e-9))
    else:
        cohen_d = float("nan")
    return {"alpha_star_rail": astar, "separation_bacc": sep_bacc,
            "alpha_star_static_frame": astar_static, "separation_bacc_static": sep_bacc_static,
            "r2_static": r2_static, "r2_attn": r2_attn, "r2": max(r2_static, r2_attn),
            "threshold_agreement_static": agree_static,
            "threshold_agreement_attn": agree_attn,
            "threshold_agreement": max(agree_static, agree_attn),
            "model_alpha_threshold_agreement": model_agree,
            "alpha_carry0_mean": float(a0.mean()) if len(a0) else float("nan"),
            "alpha_carry1_mean": float(a1.mean()) if len(a1) else float("nan"),
            "cohen_d": cohen_d, "carry_base_rate": float(y.mean()), "n": len(alpha),
            "mean_recon_err": hold.get("mean_recon_err"), "max_recon_err": hold.get("max_recon_err")}


# ===========================================================================
# T4: feasibility certificate (does alpha* make TriAdd provably correct?)
# ===========================================================================

def _resolved_carry(config):
    """TriAdd on a class config ordered LSB..MSB. class in {0,1,2(=U)}. carry
    starts 0; 0->c=0, 1->c=1, U-> transparent. Returns final carry_out."""
    c = 0
    for cls in config:
        if cls == 0:
            c = 0
        elif cls == 1:
            c = 1
        # U: pass-through
    return c


def battery_T4(t1, consumer, sites, astar_rail, bias, skip_mean, max_depth=8):
    """Enumerate reachable class configs over the consumer's source sites (ordered
    by significance, per-digit values aggregated). alpha(config) = bias + skip +
    sum_i v_i(class_i). true carry = TriAdd(config). Feasible interval =
    (max alpha over carry=0, min alpha over carry=1). Non-empty iff separable;
    alpha* feasible iff inside. Worst-case margins use the U +- CI half-width."""
    # aggregate per-digit static values (sum over redundant same-digit sites)
    per_digit = {}
    for kkey, v in t1.items():
        d = v["digit"]
        if d >= consumer["digit"]:
            continue
        agg = per_digit.setdefault(d, {"p": 0.0, "u0": 0.0, "u1": 0.0, "q": 0.0,
                                       "du": 0.0})
        agg["p"] += v["p"]; agg["q"] += v["q"]
        agg["u0"] += v["u0"]; agg["u1"] += v["u1"]
        # U uncertainty half-width (max of the two U CIs)
        du = max(abs(v["u0_ci"][1] - v["u0_ci"][0]),
                 abs(v["u1_ci"][1] - v["u1_ci"][0])) / 2.0
        agg["du"] = max(agg["du"], du)
    digits = sorted(per_digit)             # ascending significance (LSB..MSB)
    if len(digits) > max_depth:
        digits = digits[-max_depth:]       # keep the MOST significant (incl. top)
    base = bias + skip_mean
    # Correct TriAdd: propagate carry LSB->MSB; a U digit is TRANSPARENT and takes
    # its rail value from the INCOMING carry (u0 if cin=0, u1 if cin=1) -- this is
    # exactly the place-value transparency the code must implement. alpha(config)
    # is then a single value (no interval), and the certificate asks whether one
    # threshold separates carry=0 from carry=1 configs. Worst-case adds the U/commit
    # CI half-width to stress separability.
    rows = []
    for config in itertools.product([0, 1, 2], repeat=len(digits)):
        carry = 0
        alpha = base; alpha_lo = base; alpha_hi = base
        for d, cls in zip(digits, config):
            v = per_digit[d]
            if cls == 0:
                val = v["p"]; carry = 0
            elif cls == 1:
                val = v["q"]; carry = 1
            else:  # U: transparent, value set by the propagated (incoming) carry
                val = v["u1"] if carry == 1 else v["u0"]
                # carry unchanged (pass-through)
            alpha += val; alpha_lo += val - v["du"]; alpha_hi += val + v["du"]
        rows.append({"carry": carry, "alpha": alpha, "alpha_lo": alpha_lo, "alpha_hi": alpha_hi})
    carry0 = [r for r in rows if r["carry"] == 0]
    carry1 = [r for r in rows if r["carry"] == 1]
    # pointwise interval: (max alpha over carry=0, min alpha over carry=1)
    max0_pt = max((r["alpha"] for r in carry0), default=float("-inf"))
    min1_pt = min((r["alpha"] for r in carry1), default=float("inf"))
    # worst-case interval: carry=0 pushed up (alpha_hi), carry=1 pushed down (alpha_lo)
    max0 = max((r["alpha_hi"] for r in carry0), default=float("-inf"))
    min1 = min((r["alpha_lo"] for r in carry1), default=float("inf"))
    non_empty = bool(max0 < min1)
    non_empty_pt = bool(max0_pt < min1_pt)
    inside = bool(astar_rail is not None and max0 < astar_rail < min1)
    inside_pt = bool(astar_rail is not None and max0_pt < astar_rail < min1_pt)
    # graded certificate: fraction of ENUMERATED configs correctly classified by
    # the operating threshold (a "how close to certified" score, redundancy-safe).
    if astar_rail is not None:
        cfg_acc = float(np.mean([int((r["alpha"] > astar_rail)) == r["carry"] for r in rows]))
    else:
        cfg_acc = float("nan")
    return {"n_digits_enumerated": len(digits), "digits": digits,
            "worstcase_interval": [max0, min1], "worstcase_nonempty": non_empty,
            "pointwise_interval": [max0_pt, min1_pt], "pointwise_nonempty": non_empty_pt,
            "alpha_star_rail": astar_rail,
            "alpha_star_inside_worstcase": inside,
            "alpha_star_inside_pointwise": inside_pt,
            "margin_pointwise": float(min1_pt - max0_pt),
            "config_accuracy_at_astar": cfg_acc,
            "n_configs": len(rows), "n_carry0": len(carry0), "n_carry1": len(carry1)}


# ===========================================================================
# positive controls
# ===========================================================================

def pc1_regression(model, cfg, mn, n_pairs=40):
    """Reproduce CE16 joint head-pair edge flip 1.00 / deciding-matched null 0.00
    at the leading consumer."""
    n_top = cfg.n_digits - 2
    cpos, heads, top = pair_at_top(cfg, mn)
    idx = len(answer_positions(cfg)) - 1 - top
    k = min(3, n_top)
    flips, nulls = [], []
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
        clean = predict_answer(model, cfg, tq)
        sc, tc = cache_qs(model, sq, tq)
        patches = [(cpos, 1, h, sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()) for h in heads]
        flips.append(float(edge_patch_pred(model, cfg, tq, patches, tc)[idx] != clean[idx]))
        na, nb = same_class_twin(cfg, n_top, k, "lo")
        ncc, _ = cache_qs(model, make_q(cfg, na, nb), tq)
        pn = edge_patch_pred(model, cfg, tq,
                             [(cpos, 1, h, ncc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy()) for h in heads], tc)
        nulls.append(float(pn[idx] != clean[idx]))
    return {"k": k, "joint_pair_flip": mean_ci(flips), "deciding_matched_null": mean_ci(nulls)}


def pc3_wrong_axis(model, cfg, mn, consumer, ax, sites, n_q=80):
    """Wrong-axis null: replace the rail with a random unit vector; T1 ordering and
    class gaps must collapse (guards the low-variance-projection trap)."""
    rng = np.random.default_rng(SEED + 7)
    rand = rng.standard_normal(model.cfg.d_model)
    rand = rand / (np.linalg.norm(rand) + 1e-9)
    ax_rand = {"axis": rand, "c0": ax["c0"], "sep": ax["sep"]}
    t1r = battery_T1(model, cfg, mn, consumer, sites, ax_rand, n_q=n_q)
    ordered = [v["ordered"] for v in t1r.values()]
    gaps = [abs(v["class_gap"]) for v in t1r.values()]
    real_gaps = None
    return {"frac_ordered_random_axis": float(np.mean(ordered)) if ordered else float("nan"),
            "mean_abs_gap_random_axis": float(np.mean(gaps)) if gaps else float("nan"),
            "n_sites": len(t1r)}


# ===========================================================================
# T5 (stretch, G3): gap profiles at d10/d13
# ===========================================================================

def battery_T5(model, cfg, mn, n_q=150):
    """G3 dynamic-range law: per-digit class gaps g_i on the rail at the leading
    consumer; fit g_i ~ rho^i; smallest gap vs noise floor; record the carry-axis
    separation (CE18 collapse re-explanation)."""
    # build the rail from the leading-consumer served digit (large-n map-derived)
    from scripts.cross_size_sv import map_carry_axis, load_maps, combiner_mlps_from_map, \
        st_sites_from_map, identify_consumer_heads
    try:
        ma, b = load_maps(mn)
    except Exception as e:
        return {"note": f"map load failed: {e}"}
    comb = combiner_mlps_from_map(b)
    ax = map_carry_axis(model, cfg, mn, comb, n_q=n_q)
    st_sites = st_sites_from_map(ma)
    consumers, diag = identify_consumer_heads(model, cfg, mn, st_sites, ax, n_pairs=20)
    if not consumers:
        return {"note": "no consumer head identified", "carry_axis_sep": ax["sep"]}
    n_top = cfg.n_digits - 2
    cpos = diag["cpos"]; heads = consumers
    consumer = {"name": "leading", "cpos": cpos, "heads": heads, "digit": n_top + 1}
    sites = [{"pos": s["pos"], "head": s["head"], "digit": s["digit"]} for s in st_sites]
    t1 = battery_T1(model, cfg, mn, consumer, sites, ax, n_q=n_q)
    # per-digit gaps
    per_digit = {}
    for v in t1.values():
        per_digit.setdefault(v["digit"], 0.0)
        per_digit[v["digit"]] += abs(v["class_gap"])
    digits = sorted(per_digit)
    gaps = np.array([per_digit[d] for d in digits])
    # fit g ~ rho^i (log-linear) over significance order
    rho = None; r2 = None
    if len(gaps) >= 3 and np.all(gaps > 0):
        y = np.log(gaps); x = np.arange(len(gaps))
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rho = float(np.exp(coef[0]))
        pred = A @ coef
        r2 = float(1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-9))
    return {"carry_axis_sep": ax["sep"], "digits": digits,
            "per_digit_gap": {int(d): float(per_digit[d]) for d in digits},
            "smallest_gap": float(gaps.min()), "largest_gap": float(gaps.max()),
            "gap_ratio_large_small": float(gaps.max() / (gaps.min() + 1e-9)),
            "rho_fit": rho, "rho_fit_r2": r2, "consumer_heads": heads}


# ===========================================================================
# driver
# ===========================================================================

def run_model(model, cfg, mn, fast=False):
    n_q1 = 80 if fast else 200
    n_q3 = 120 if fast else 250
    n_qT = 30 if fast else 60
    out = {"model": mn}
    n_top = cfg.n_digits - 2
    depths = list(range(1, n_top + 1))
    out["gates"] = {k: behavioral_gate(model, cfg, n_top, k, n_q=30) for k in depths}
    ax = carry_axis(model, cfg, mn, n_q=(120 if fast else 250))
    out["PC2_axis_sep"] = ax["sep"]
    out["PC1_regression"] = pc1_regression(model, cfg, mn, n_pairs=(20 if fast else 40))
    out["untrained_control"] = untrained_control(model, cfg, mn, fast=fast)
    sites = st_source_sites(mn)
    out["sites"] = sites
    out["consumers"] = {}
    for consumer in consumers_for(cfg, mn):
        cname = consumer["name"]
        # PRIMARY: the shared CE16/CE17 geo rail (contract-specified single rail)
        res = analyze_consumer(model, cfg, mn, consumer, sites, ax, n_q1, n_q3, fast)
        res["meta"] = {k: consumer[k] for k in ("cpos", "heads", "digit")}
        # CONTROL (skeptic gate): per-consumer LOCAL delivered-carry rail — separates
        # genuine top-gap compression from shared-rail misalignment (G1 rotation).
        lax = consumer_local_carry_axis(model, cfg, consumer, n_q=(80 if fast else 150))
        if lax is not None:
            lres = analyze_consumer(model, cfg, mn, consumer, sites, lax, n_q1, n_q3, fast,
                                    do_pc3=False)
            res["local_rail"] = {"PC2_local_sep": lax["sep"],
                                 "T2_dominance": lres["T2_dominance"],
                                 "T1_frac_ordered": float(np.mean([x["ordered"] for x in lres["T1_ordering"].values()])) if lres["T1_ordering"] else float("nan"),
                                 "T3": lres["threshold"], "T4_certificate": lres["T4_certificate"],
                                 "per_digit_gap": lres["T2_dominance"]["per_digit_gap"]}
        out["consumers"][cname] = res
    out["verdict"] = derive_verdict(out)
    return out


def analyze_consumer(model, cfg, mn, consumer, sites, ax, n_q1, n_q3, fast, do_pc3=True):
    t1 = battery_T1(model, cfg, mn, consumer, sites, ax, n_q=n_q1)
    t2 = battery_T2(t1, consumer, sites)
    skip_vals = []; bias = 0.0
    for _ in range(30):
        a = int(RNG.integers(0, 10 ** cfg.n_digits // 2))
        bb = int(RNG.integers(0, 10 ** cfg.n_digits // 2))
        dec = decompose(model, cfg, consumer["cpos"], consumer["heads"], ax,
                        _cache_dec(model, make_q(cfg, a, bb)))
        skip_vals.append(dec["skip"]); bias = dec["bias"]
    skip_mean = float(np.mean(skip_vals))
    hold = gather_holdout(model, cfg, mn, consumer, sites, ax, t1, n_q=n_q3)
    thr = threshold_and_reconstruction(hold)
    t4 = battery_T4(t1, consumer, sites, thr["alpha_star_static_frame"], bias, skip_mean)
    combiner = combiner_sharpness(hold, thr)
    res = {"threshold": thr, "combiner_sharpness": combiner,
           "T1_ordering": t1, "T2_dominance": t2, "T4_certificate": t4,
           "skip_mean": skip_mean, "bias": bias, "PC2_sep": ax["sep"]}
    if do_pc3:
        res["PC3_wrong_axis"] = pc3_wrong_axis(model, cfg, mn, consumer, ax, sites,
                                               n_q=(40 if fast else 80))
    return res


def untrained_control(model, cfg, mn, fast=False):
    """Negative control: on an untrained twin, the geo 'carry axis' is meaningless,
    so PC2 sep should be small and T1 ordering / class gaps should collapse to the
    random-axis floor. Guards against the structure being an artifact of 10 points
    in high-D / the pipeline itself."""
    from quanta_maths import make_untrained_control
    ctrl = make_untrained_control(cfg)
    try:
        ax = carry_axis(ctrl, cfg, mn, n_q=(80 if fast else 150))
    except Exception as e:
        return {"error": str(e)}
    consumer = consumers_for(cfg, mn)[0]        # leading
    sites = st_source_sites(mn)
    t1 = battery_T1(ctrl, cfg, mn, consumer, sites, ax, n_q=(60 if fast else 120))
    gaps = [abs(v["class_gap"]) for v in t1.values()]
    ordered = [v["ordered"] for v in t1.values()]
    del ctrl
    return {"PC2_sep_untrained": ax["sep"],
            "T1_frac_ordered_untrained": float(np.mean(ordered)) if ordered else float("nan"),
            "T1_mean_abs_gap_untrained": float(np.mean(gaps)) if gaps else float("nan"),
            "max_abs_gap_untrained": float(np.max(gaps)) if gaps else float("nan")}


def combiner_sharpness(hold, thr):
    """Is the rail->carry map a clean STEP (well-separated classes) at alpha*?
    Reports the class-conditional means, the separation bacc, and Cohen's d."""
    return {"class": ("step" if thr["separation_bacc"] >= 0.9 else
                      ("graded" if thr["separation_bacc"] >= 0.7 else "weak")),
            "separation_bacc": thr["separation_bacc"], "cohen_d": thr["cohen_d"],
            "alpha_carry0_mean": thr["alpha_carry0_mean"],
            "alpha_carry1_mean": thr["alpha_carry1_mean"]}


def derive_verdict(out):
    v = {"controls": {}}
    pc1 = out["PC1_regression"]
    pc1_ok = pc1["joint_pair_flip"]["rate"] >= 0.8 and pc1["deciding_matched_null"]["rate"] <= 0.2
    v["controls"]["PC1_ok"] = bool(pc1_ok)
    v["controls"]["PC2_axis_sep"] = out["PC2_axis_sep"]
    v["consumers"] = {}
    for cname, cd in out["consumers"].items():
        t1 = cd["T1_ordering"]; t2 = cd["T2_dominance"]
        thr = cd["threshold"]; t4 = cd["T4_certificate"]; comb = cd["combiner_sharpness"]
        pc3 = cd["PC3_wrong_axis"]
        n_sites = len(t1)
        # sign-consistent gap: class 1 rail > class 0 (needed for the certificate)
        gaps = [x["class_gap"] for x in t1.values() if x["digit"] < cd["meta"]["digit"]]
        frac_pos_gap = float(np.mean([g > 0 for g in gaps])) if gaps else float("nan")
        frac_ordered = float(np.mean([x["ordered"] for x in t1.values()])) if t1 else float("nan")
        frac_inside = float(np.mean([x["u_inside"] for x in t1.values()])) if t1 else float("nan")
        frac_cinsplit = float(np.mean([x["cin_split"] for x in t1.values()])) if t1 else float("nan")
        pc3_ok = (pc3["frac_ordered_random_axis"] < frac_ordered - 0.1) if frac_ordered == frac_ordered else False
        v["consumers"][cname] = {
            "digit": cd["meta"]["digit"], "n_sites": n_sites,
            "T1_frac_ordered": frac_ordered, "T1_frac_U_inside": frac_inside,
            "T1_frac_cin_split": frac_cinsplit, "T1_frac_pos_gap": frac_pos_gap,
            "T2_top3_dominate": t2["top3_dominate"], "T2_all_dominate": t2["all_dominate"],
            "T3_r2_static": thr["r2_static"], "T3_r2_attn": thr["r2_attn"],
            "T3_r2": thr["r2"], "T3_threshold_agreement": thr["threshold_agreement"],
            "T3_model_threshold_agreement": thr["model_alpha_threshold_agreement"],
            "separation_bacc": thr["separation_bacc"],
            "T4_worstcase_nonempty": t4["worstcase_nonempty"],
            "T4_pointwise_nonempty": t4["pointwise_nonempty"],
            "T4_alpha_star_inside_pointwise": t4["alpha_star_inside_pointwise"],
            "T4_alpha_star_inside_worstcase": t4["alpha_star_inside_worstcase"],
            "T4_config_accuracy_at_astar": t4["config_accuracy_at_astar"],
            "alpha_star_rail": thr["alpha_star_rail"],
            "combiner_class": comb["class"],
            "PC3_ok": bool(pc3_ok),
        }
    # G2 success (per study success condition), evaluated at the leading consumer
    lead = v["consumers"].get("leading", {})
    g2 = (v["controls"]["PC1_ok"]
          and lead.get("T1_frac_ordered", 0) >= 0.8
          and lead.get("T3_r2", 0) >= 0.7
          and lead.get("T3_threshold_agreement", 0) >= 0.9
          and lead.get("T4_pointwise_nonempty", False)
          and lead.get("T4_alpha_star_inside_pointwise", False))
    v["G2_confirmed"] = bool(g2)
    v["G2_read"] = _g2_read(v)
    return v


def _g2_read(v):
    lead = v["consumers"].get("leading", {})
    if not v["controls"]["PC1_ok"]:
        return "INVALID (PC1 control failed)"
    if v["G2_confirmed"]:
        return "G2 CONFIRMED: place-value dominance code; STEP makes TriAdd provable"
    parts = []
    if lead.get("T1_frac_ordered", 0) >= 0.8:
        parts.append("ordering holds")
    else:
        parts.append(f"ordering weak ({lead.get('T1_frac_ordered')})")
    if lead.get("T2_top3_dominate"):
        parts.append("dominance holds")
    else:
        parts.append("raw-gap dominance fails (competing read 3: ordered-but-uncertified)")
    if lead.get("T3_r2", 0) >= 0.7:
        parts.append(f"additive (R2={lead.get('T3_r2'):.2f})")
    else:
        parts.append(f"non-additive read (R2={lead.get('T3_r2'):.2f}; competing read 2)")
    if lead.get("T4_pointwise_nonempty"):
        ins = "a* inside" if lead.get("T4_alpha_star_inside_pointwise") else "a* OUTSIDE"
        parts.append(f"certificate non-empty ({ins})")
    else:
        parts.append("certificate empty")
    return "G2 PARTIAL: " + "; ".join(parts)


def main():
    fast = "--fast" in sys.argv
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "all"
    results = {}
    if mode in ("all", "primary"):
        for mn in MODELS:
            model, cfg = load_model(mn)
            acc = verify_accuracy(model, cfg, n=64)
            assert acc > 0.99, f"{mn} acc {acc}"
            print(f"=== {mn} (acc {acc:.3f}) fast={fast} ===", flush=True)
            r = run_model(model, cfg, mn, fast=fast); r["accuracy"] = acc
            results[mn] = r
            v = r["verdict"]
            uc = r["untrained_control"]
            print("  PC1", r["PC1_regression"]["joint_pair_flip"]["rate"],
                  "null", r["PC1_regression"]["deciding_matched_null"]["rate"],
                  "| PC2 sep", round(r["PC2_axis_sep"], 2),
                  "| UNTRAINED sep", round(uc.get("PC2_sep_untrained", float("nan")), 2),
                  "ord", round(uc.get("T1_frac_ordered_untrained", float("nan")), 2),
                  "gap", round(uc.get("max_abs_gap_untrained", float("nan")), 4), flush=True)
            for cname, cd in r["consumers"].items():
                if "local_rail" in cd:
                    lr = cd["local_rail"]
                    print(f"    [{cname} LOCAL-rail] PC2_local={lr['PC2_local_sep']:.2f} "
                          f"ordered={lr['T1_frac_ordered']:.2f} per_digit_gap="
                          f"{ {int(k): round(v2,3) for k,v2 in lr['per_digit_gap'].items()} } "
                          f"T4_nonempty={lr['T4_certificate']['pointwise_nonempty']} "
                          f"cfgacc={lr['T4_certificate']['config_accuracy_at_astar']:.2f} "
                          f"R2s={lr['T3']['r2_static']:.2f}", flush=True)
            for cname, cv in v["consumers"].items():
                print(f"  [{cname} d{cv['digit']}] ordered={cv['T1_frac_ordered']:.2f} "
                      f"U_inside={cv['T1_frac_U_inside']:.2f} cin_split={cv['T1_frac_cin_split']:.2f} "
                      f"pos_gap={cv['T1_frac_pos_gap']:.2f} | T2 top3_dom={cv['T2_top3_dominate']} "
                      f"| T3 R2={cv['T3_r2']:.2f}(s{cv['T3_r2_static']:.2f}/a{cv['T3_r2_attn']:.2f}) "
                      f"agree={cv['T3_threshold_agreement']:.2f} sepbacc={cv['separation_bacc']:.2f} "
                      f"| T4 nonempty(pt)={cv['T4_pointwise_nonempty']} a*_in(pt)={cv['T4_alpha_star_inside_pointwise']} "
                      f"cfgacc={cv['T4_config_accuracy_at_astar']:.2f} "
                      f"a*={round(cv['alpha_star_rail'],3) if cv['alpha_star_rail'] is not None else None} "
                      f"comb={cv['combiner_class']} PC3ok={cv['PC3_ok']}", flush=True)
            print("  VERDICT:", v["G2_read"], flush=True)
            del model
    if mode in ("all", "t5"):
        results.setdefault("_T5_G3", {})
        for mn in LARGE_MODELS:
            try:
                model, cfg = load_model(mn)
                acc = verify_accuracy(model, cfg, n=48)
                print(f"=== T5/G3 {mn} (acc {acc:.3f}) ===", flush=True)
                t5 = battery_T5(model, cfg, mn, n_q=(80 if fast else 150))
                t5["accuracy"] = acc
                results["_T5_G3"][mn] = t5
                print("  [T5]", {k: (round(x, 3) if isinstance(x, float) else x)
                                  for k, x in t5.items() if k != "per_digit_gap"}, flush=True)
                del model
            except Exception as e:
                print(f"  T5 {mn} FAILED: {e}", flush=True)
                results["_T5_G3"][mn] = {"error": str(e)}
    tag = "_fast" if fast else ""
    with open(os.path.join(RESULT_DIR, f"results{tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
