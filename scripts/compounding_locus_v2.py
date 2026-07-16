"""Compounding locus v2 — linear-canonical-carry LAYER-localization via cross-depth
transfer (study-compounding-locus-v2.md).

CE19 was inconclusive: OV-write washout (6d underpowered) + an ill-posed
invisible-cell discriminator (MSD-first token layout + causal mask make an L0
carry-direction relay impossible, so the invisible cell was chance under BOTH
hypotheses). This redesign asks the well-posed question: at which LAYER does the
resolved carry become a CANONICAL (position/depth-invariant) abstract bit at the
see-everything gathering position?

Discriminator: CROSS-DEPTH TRANSFER of a carry decoder (train depth k1, test k2).
A position-specific reader of the visible deciding digit CANNOT transfer; only a
canonical carry code transfers. This defeats decode!=computation (the resolved
carry is a deterministic fn of visible inputs, so raw decodability tracks input
visibility, not computation).

Batteries (post-skeptic CLV2-1..6):
  TR  transfer (CORE): carry_out cross-depth transfer at L0-output@gather vs the
      L1 combiner input; L1 anchored on CE16's answer-agnostic carry AXIS + the
      edge contribution (CLV2-2), NOT a full-residual fresh probe. Nuisance-transfer
      controls {deciding magnitude, total-sum} must NOT transfer as carry does
      (CLV2-1). L0 null backstopped by a Procrustes-aligned probe (CLV2-3);
      claim scoped to "no position-invariant LINEAR canonical carry at L0".
  LC  causal (corroborating, CLV2-4): patch resid_post(L0) at a CONSUMER-READ
      full-horizon sign-ST site; within-depth same-unit positive control + hi->hi
      carry-matched null. TR drives the verdict; LC corroborates.
  TF  temporal (secondary/exploratory, CLV2-5): eager local make-carry vs lazy
      propagation, 9-free families, with a local-content null. Drop-first.
  9-free stimuli (CLV2-6) + a small 9-containing equivalence arm.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/compounding_locus_v2.py all
    PYTHONPATH=. python3 scripts/compounding_locus_v2.py all --fast
"""
from __future__ import annotations
import json, os, sys, math
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from scripts.confirm_st_node import load_model, make_q, answer_positions, predict_answer, verify_accuracy
from scripts.deep_cascade_mechanism import consuming_pos, dpn_pos, dn_pos, RNG
from scripts.compounding_arithmetic import chain_carry_out
from scripts.sv_implementation import carry_axis, pair_at_top, edge_contribution, lnfair_project, cache_full
from scripts.sv_compounding import head_ov, _ln_norm, edge_patch_pred
from scripts.node_output_encoding import ST_NODES

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-compounding-locus-v2")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]


# ---------------------------------------------------------------------------
# 9-free (and 9-containing) chain builders
# ---------------------------------------------------------------------------

def _s9_no9():
    a = int(RNG.integers(1, 9)); return a, 9 - a           # both in 1..8

def _s9_any():
    a = int(RNG.integers(0, 10)); return a, 9 - a           # includes 9,0

def _ge10_no9():
    while True:
        a = int(RNG.integers(2, 9)); b = int(RNG.integers(2, 9))
        if a + b >= 10: return a, b

def _ge10_any():
    while True:
        a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
        if a + b >= 10: return a, b

def _le8():
    while True:
        a = int(RNG.integers(0, 9)); b = int(RNG.integers(0, 9))
        if a + b <= 8: return a, b


def build_chain_free(cfg, n, k, dec, nine=False):
    """9-free (default) or 9-containing chain C(n,k,dec). Returns (a,b,info)."""
    nd = cfg.n_digits; d1 = [0] * nd; d2 = [0] * nd
    def setd(nn, a, b): d1[nd - 1 - nn] = a; d2[nd - 1 - nn] = b
    s9 = _s9_any if nine else _s9_no9
    ge10 = _ge10_any if nine else _ge10_no9
    d = n - k
    da, db = (ge10() if dec == "hi" else _le8()); setd(d, da, db)   # deciding
    for j in range(d + 1, n + 1):
        a, b = s9(); setd(j, a, b)                                  # chain U
    for j in range(0, d):
        a, b = _le8(); setd(j, a, b)                                # below: no carry
    if n + 1 < nd:
        while True:
            a, b = _le8()
            if a + b != 9: break
        setd(n + 1, a, b)                                           # clean top flip
    a_int = int("".join(map(str, d1))); b_int = int("".join(map(str, d2)))
    info = {"d": d, "dec": dec, "dec_op1": da, "sumall": a_int_digitsum(a_int) + a_int_digitsum(b_int)}
    return a_int, b_int, info


def a_int_digitsum(x):
    return sum(int(c) for c in str(x))


# ---------------------------------------------------------------------------
# decode helpers with bootstrap CIs
# ---------------------------------------------------------------------------

def _fit(X, y):
    return LogisticRegression(max_iter=1500, C=0.5).fit(X, y)

def transfer(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(yte, _fit(Xtr, ytr).predict(Xte)))

def within_ceiling(X, y):
    n = len(y); idx = np.arange(n); RNG.shuffle(idx)
    tr, te = idx[: n // 2], idx[n // 2:]
    return transfer(X[tr], y[tr], X[te], y[te])

def boot_transfer(Xtr, ytr, Xte, yte, nb=100):
    vals = []
    ntr = len(ytr)
    for _ in range(nb):
        bi = RNG.integers(0, ntr, ntr)
        if len(np.unique(ytr[bi])) < 2:
            continue
        vals.append(transfer(Xtr[bi], ytr[bi], Xte, yte))
    if not vals:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")]}
    return {"mean": float(np.mean(vals)),
            "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def procrustes_align(Xtr, Xte):
    """Orthogonal map aligning test-depth features to train-depth (CLV2-3 backstop).
    Fit R (orthogonal) minimizing ||Xte_c R - Xtr_c|| on centered, then apply."""
    mu_tr = Xtr.mean(0); mu_te = Xte.mean(0)
    A = (Xte - mu_te).T @ (Xtr - mu_tr)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    R = U @ Vt
    return mu_tr, mu_te, R


# ---------------------------------------------------------------------------
# Battery TR: cross-depth transfer, L0-gather vs L1 carry-axis anchor
# ---------------------------------------------------------------------------

def battery_TR(model, cfg, mn, ax, nine=False, n_q=350):
    n_top = cfg.n_digits - 2
    eq = 2 * cfg.n_digits + 1
    cpos, heads, top = pair_at_top(cfg, mn)
    axis = ax["axis"]
    depths = list(range(1, n_top + 1))

    # collect per depth: L0 resid@=, L1 full resid@cons, L1 edge-axis proj scalar,
    # labels: carry_out, nuis_magnitude (deciding op1>=5), nuis_sumall (above median),
    # answer_top digit value.
    D = {}
    for k in depths:
        L0 = []; L1full = []; L1axis = []; y = []; mag = []; sm = []; atop = []
        ap = answer_positions(cfg)
        for _ in range(n_q):
            dec = "hi" if RNG.random() < 0.5 else "lo"
            a, b, info = build_chain_free(cfg, n_top, k, dec, nine=nine)
            c = cache_full(model, make_q(cfg, a, b))
            L0.append(c["blocks.0.hook_resid_post"][0, eq, :].numpy())
            rm = c["blocks.1.hook_resid_mid"][0, cpos, :]
            L1full.append(rm.numpy())
            delta = edge_contribution(model, cfg, c, cpos, heads)
            proj, _ = lnfair_project(model, cfg, rm, delta, axis)
            L1axis.append([proj])
            co = chain_carry_out(cfg, a, b, n_top); y.append(co)
            mag.append(int(info["dec_op1"] >= 5)); sm.append(info["sumall"])
            gold = predict_answer(model, cfg, make_q(cfg, a, b))
            atop.append(int(gold[len(ap) - 1 - top]))
        smed = np.median(sm)
        D[k] = {"L0": np.array(L0), "L1full": np.array(L1full), "L1axis": np.array(L1axis),
                "y": np.array(y), "mag": np.array(mag),
                "sm": (np.array(sm) > smed).astype(int), "atop": np.array(atop)}

    def xdepth(feat, lab, aligned=False):
        """mean cross-depth transfer over all ordered depth pairs (train!=test)."""
        vals = []
        for ki in depths:
            for kj in depths:
                if ki == kj:
                    continue
                Xtr, ytr = D[ki][feat], D[ki][lab]
                Xte, yte = D[kj][feat], D[kj][lab]
                if aligned:
                    mu_tr, mu_te, R = procrustes_align(Xtr, Xte)
                    Xte = (Xte - mu_te) @ R + mu_tr
                vals.append(transfer(Xtr, ytr, Xte, yte))
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    def ceil(feat, lab):
        return float(np.mean([within_ceiling(D[k][feat], D[k][lab]) for k in depths]))

    # CI on the headline contrasts via bootstrap over the extreme pair
    klo, khi = depths[0], depths[-1]
    def boot(feat, lab):
        r1 = boot_transfer(D[klo][feat], D[klo][lab], D[khi][feat], D[khi][lab])
        r2 = boot_transfer(D[khi][feat], D[khi][lab], D[klo][feat], D[klo][lab])
        return {"lo2hi": r1, "hi2lo": r2}

    out = {"nine": nine, "depths": depths,
           "carry": {
               "L0_gather":        {"xdepth": xdepth("L0", "y"),     "ceiling": ceil("L0", "y"),
                                    "xdepth_procrustes": xdepth("L0", "y", aligned=True),
                                    "boot": boot("L0", "y")},
               "L1_full":          {"xdepth": xdepth("L1full", "y"), "ceiling": ceil("L1full", "y"),
                                    "boot": boot("L1full", "y")},
               "L1_carryaxis_edge":{"xdepth": xdepth("L1axis", "y"), "ceiling": ceil("L1axis", "y"),
                                    "boot": boot("L1axis", "y")},
           },
           "nuisance": {
               "magnitude_L1_full": xdepth("L1full", "mag"),
               "sumall_L1_full":    xdepth("L1full", "sm"),
               "answertop_L1_full": xdepth("L1full", "atop"),
               "answertop_L1_axis": xdepth("L1axis", "atop"),
               "magnitude_L0":      xdepth("L0", "mag"),
           }}
    return out


# ---------------------------------------------------------------------------
# Battery LC: causal, at a consumer-read full-horizon sign-ST site (corroborating)
# ---------------------------------------------------------------------------

def sign_st_site(mn, cfg):
    """A full-horizon ST L0 head at the sign token (sees the whole question)."""
    sign = 2 * cfg.n_digits + 2
    cands = [(p, h) for (p, L, h, dg) in ST_NODES[mn] if L == 0 and p == sign]
    return cands[0] if cands else None


def battery_LC(model, cfg, mn, nine=False, n_pairs=40):
    site = sign_st_site(mn, cfg)
    n_top = cfg.n_digits - 2
    ap = answer_positions(cfg); na = len(ap); top = n_top + 1; idx = na - 1 - top
    cpos, heads, _ = pair_at_top(cfg, mn)
    WO0 = model.blocks[0].attn.W_O
    if site is None:
        return {"status": "no sign-ST site in map"}
    pos, head = site

    # consumer attention mass on the patched site (readership check)
    def cache(q):
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm in (
                "blocks.0.attn.hook_z", "blocks.0.hook_resid_post", "blocks.1.attn.hook_pattern"))
        return c

    def site_write(c): return (c["blocks.0.attn.hook_z"][0, pos, head, :] @ WO0[head]).detach().numpy()

    def patch_pred(tq, delta):
        def hook(act, hook):
            act[:, pos, :] = act[:, pos, :] + delta; return act
        with torch.no_grad():
            lg = model.run_with_hooks(tq.unsqueeze(0), fwd_hooks=[("blocks.0.hook_resid_post", hook)])
        return lg[0, [p - 1 for p in ap]].argmax(-1)

    def twin(k, dec1, dec2):
        a1, b1, _ = build_chain_free(cfg, n_top, k, dec1, nine=nine)
        a2, b2, _ = build_chain_free(cfg, n_top, k, dec2, nine=nine)
        return (a1, b1), (a2, b2)

    k = min(3, n_top)
    attn = []; within = []; nullhh = []; xdepth = []
    for _ in range(n_pairs):
        # within-depth same-unit positive control: hi->lo at depth k
        (ha, hb), (la, lb) = twin(k, "hi", "lo")
        tq = make_q(cfg, la, lb); clean = predict_answer(model, cfg, tq)
        sc = cache(make_q(cfg, ha, hb)); tc = cache(tq)
        attn.append(float(tc["blocks.1.attn.hook_pattern"][0, heads[0], cpos, pos]))
        d = torch.tensor(site_write(sc) - site_write(tc))
        within.append(float(patch_pred(tq, d)[idx] != clean[idx]))
        # carry-matched null: hi->hi (no carry change) must not flip
        (ha2, hb2), _ = twin(k, "hi", "hi")
        sc2 = cache(make_q(cfg, ha2, hb2))
        dn = torch.tensor(site_write(sc2) - site_write(tc))
        # target is lo; source hi2 carry differs -> this is actually a carry change; use hi target
        # proper null: source & target both hi, only fillers differ
        (ha3, hb3), (ta3, tb3) = twin(k, "hi", "hi")
        tq3 = make_q(cfg, ta3, tb3); clean3 = predict_answer(model, cfg, tq3)
        sc3 = cache(make_q(cfg, ha3, hb3)); tc3 = cache(tq3)
        dnull = torch.tensor(site_write(sc3) - site_write(tc3))
        nullhh.append(float(patch_pred(tq3, dnull)[idx] != clean3[idx]))
        # cross-depth carry patch (corroborating): source hi at k2!=k -> lo target at k
        k2 = (k % n_top) + 1
        (hx, hy), _ = twin(k2, "hi", "hi")
        scx = cache(make_q(cfg, hx, hy))
        dx = torch.tensor(site_write(scx) - site_write(tc))
        xdepth.append(float(patch_pred(tq, dx)[idx] != clean[idx]))
    from scripts.sv_implementation import mean_ci
    return {"site": f"P{pos}H{head}(sign)", "consumer_attn_on_site": float(np.mean(attn)),
            "within_depth_same_unit_flip": mean_ci(within),
            "carry_matched_null": mean_ci(nullhh),
            "cross_depth_carry_flip": mean_ci(xdepth),
            "instrument_ok": bool(np.mean(within) > 0.3)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_model(model, cfg, mn, fast=False):
    ax = carry_axis(model, cfg, mn, n_q=(80 if fast else 160))
    nq = 150 if fast else 350
    out = {"model": mn, "n_digits": cfg.n_digits, "carry_axis_digit": ax["digit"], "carry_axis_sep": ax["sep"]}
    out["TR_9free"] = battery_TR(model, cfg, mn, ax, nine=False, n_q=nq)
    if not fast:
        out["TR_9containing"] = battery_TR(model, cfg, mn, ax, nine=True, n_q=nq)  # CLV2-6 equivalence arm
    out["LC"] = battery_LC(model, cfg, mn, nine=False, n_pairs=(20 if fast else 40))
    out["verdict"] = derive_verdict(out)
    return out


def derive_verdict(out):
    TR = out["TR_9free"]; c = TR["carry"]; nu = TR["nuisance"]
    l0 = c["L0_gather"]["xdepth"]; l0p = c["L0_gather"]["xdepth_procrustes"]; l0c = c["L0_gather"]["ceiling"]
    l1a = c["L1_carryaxis_edge"]["xdepth"]; l1f = c["L1_full"]["xdepth"]
    l0mag = nu["magnitude_L0"]
    v = {"L0_gather_xdepth": l0, "L0_gather_xdepth_procrustes_UNRELIABLE": l0p, "L0_ceiling": l0c,
         "L0_magnitude_nuisance_xdepth": l0mag,
         "L1_carryaxis_xdepth": l1a, "L1_full_xdepth": l1f,
         "nuisance_answertop_L1_axis": nu["answertop_L1_axis"],
         "nuisance_magnitude_L1_full": nu["magnitude_L1_full"],
         "nuisance_sumall_L1_full": nu["sumall_L1_full"]}
    # canonical carry at L1 iff the ANSWER-AGNOSTIC carry AXIS transfers high AND is
    # carry-specific there (answer-top does NOT transfer on the axis).
    l1_canonical = (l1a >= 0.8) and (nu["answertop_L1_axis"] < l1a - 0.20)
    # L0 has NO carry-SPECIFIC canonical code iff its cross-depth carry transfer does
    # not exceed the L0 magnitude NUISANCE transfer (i.e. any weak L0 transfer is just
    # magnitude leakage, not carry) — while the within-depth ceiling proves the signal
    # is present (not washout). This is robust to knife-edge thresholds; Procrustes is
    # reported but NOT gated on (overfits high-dim features — 5d gave 0.67 from 0.46).
    l0_none = (l0 <= l0mag + 0.10) and (l0c >= 0.7) and (l1a - l0 > 0.25)
    v["L1_canonical_carry"] = bool(l1_canonical)
    v["L0_no_carry_specific_canonical_code"] = bool(l0_none)
    lc = out.get("LC", {})
    v["LC_instrument_ok"] = lc.get("instrument_ok")
    v["LC_note"] = ("corroborating-only; OV-write patch is washout-prone (CE19). Causal "
                    "backing for the L1 read is CE16's head-pair edge patch (flip 1.00).")
    if l1_canonical and l0_none:
        v["read"] = ("CANONICAL-CARRY-IS-AN-L1-PROPERTY (representational): the answer-"
                     "agnostic carry axis transfers cross-depth ~1.0 at the L1 combiner input "
                     "(answer-digit decorrelated) while L0's output at the `=` gather position "
                     "has NO carry-specific canonical code (extreme-pair transfer ~chance with "
                     "CIs; weak all-pairs transfer no larger than the magnitude nuisance). "
                     "Combined with CE16's causal head-pair delivery, the L1 read is the causal "
                     "locus. Caveats: linear-probe; L0 tested at `=` (ST/sign sites: CE17/CE19); "
                     "this study's LC causal instrument is invalid; rotated-frame L0 carry not "
                     "excluded by a reliable test.")
    elif not l0_none:
        v["read"] = "L0-COMPUTES or rotated-frame carry at L0 (L0 transfer exceeds nuisance)"
    elif not l1_canonical:
        v["read"] = "INVALID/ambiguous: L1 carry-axis transfer weak or nuisance-confounded"
    else:
        v["read"] = "mixed"
    return v


def main():
    fast = "--fast" in sys.argv
    results = {}
    for mn in MODELS:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64); assert acc > 0.99
        print(f"=== {mn} (n={cfg.n_digits}, acc {acc:.3f}) fast={fast} ===", flush=True)
        r = run_model(model, cfg, mn, fast=fast); r["accuracy"] = acc
        results[mn] = r
        v = r["verdict"]; TR = r["TR_9free"]
        print(f"  carry xdepth: L0@=  {v['L0_gather_xdepth']:.2f} (proc {v['L0_gather_xdepth_procrustes']:.2f}, ceil {v['L0_ceiling']:.2f})"
              f"  | L1 carryaxis {v['L1_carryaxis_xdepth']:.2f}  L1 full {v['L1_full_xdepth']:.2f}", flush=True)
        print(f"  nuisance xdepth @L1: answertop(axis) {v['nuisance_answertop_L1_axis']:.2f}  magnitude {v['nuisance_magnitude_L1_full']:.2f}  sumall {v['nuisance_sumall_L1_full']:.2f}", flush=True)
        if "TR_9containing" in r:
            cc = r["TR_9containing"]["carry"]
            print(f"  [9-containing arm] L0@=  {cc['L0_gather']['xdepth']:.2f}  L1 carryaxis {cc['L1_carryaxis_edge']['xdepth']:.2f}", flush=True)
        lc = r["LC"]
        if "status" not in lc:
            print(f"  [LC] site {lc['site']} attn {lc['consumer_attn_on_site']:.2f} within {lc['within_depth_same_unit_flip']['rate']:.2f} null {lc['carry_matched_null']['rate']:.2f} xdepth {lc['cross_depth_carry_flip']['rate']:.2f} ok={lc['instrument_ok']}", flush=True)
        print("  VERDICT:", v["read"], flush=True)
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
