"""Full-space ST tri-state geometry at the L1-MLP combiner
(study-st-tristate-geometry.md).

CE5 confirmed the answer-position L1 MLP is the tri-state U-combiner (its OUTPUT
is the resolved binary carry_out). This measures the {0,1,U} geometry at the
combiner's INPUT (blocks.1.ln2.hook_normalized -- what the MLP reads).

KEY (amendment A-1): U is structurally identical to SA_n=9, so "U off-axis"
alone is un-attributable. The discriminator is the RESOLUTION variable: build
U with a real lower carry so the same SA_n=9 digit resolves to carry_out 0 or 1.
Four classes at the combiner input:
  committed-0 (definite sum<=8), committed-1 (definite sum>=10),
  U->0 (sum==9, no lower carry), U->1 (sum==9, lower carry).
Any structure distinguishing U->0 from U->1, or placing U off the
committed-0<->committed-1 line, CANNOT come from the SA_n=9 code (a single
point) -> genuine carry structure.

Shapes: R-scalar (U on 0-1 line), R-third-symbol/simplex (U off-axis),
R-square (two orthogonal binary axes).

CPU-only. Run:
    PYTHONPATH=. python3 scripts/st_tristate_geometry.py control
    PYTHONPATH=. python3 scripts/st_tristate_geometry.py models
    PYTHONPATH=. python3 scripts/st_tristate_geometry.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    _digits_to_int,
)
import torch

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-st-tristate-geometry")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260715)

# combiner (CE5) + make-carry site (CE3) per model: (combiner_pos, layer, digit_n)
CFG = {
    "add_d5_l2_h3_t15K_s372001": {"combiner_pos": 14, "digit": 2, "make_carry": (13, 0, 0)},
    "add_d6_l2_h3_t20K_s173289": {"combiner_pos": 16, "digit": 3, "make_carry": (15, 0, 2)},
}


# ---------------------------------------------------------------------------
# stimulus: four classes, digit n, controlling lower carry (A-1)
# ---------------------------------------------------------------------------

def build_class_question(cfg, n, cls):
    """cls in {'c0','c1','u0','u1'}:
      c0 = committed 0 (sum<=8, no lower carry), c1 = committed 1 (sum>=10),
      u0 = U (sum==9) with NO lower carry (resolves to 0),
      u1 = U (sum==9) with lower carry (resolves to 1).
    Digits above n = 0; digits below n-1 = no carry."""
    nd = cfg.n_digits
    idx = nd - 1 - n
    if cls in ("u0", "u1"):
        a = int(RNG.integers(0, 10)); b = 9 - a
        lower_carry = (cls == "u1")
    elif cls == "c0":
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b <= 8: break
        lower_carry = bool(RNG.integers(0, 2))  # committed: carry_out indep of carry_in
    else:  # c1
        while True:
            a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
            if a + b >= 10: break
        lower_carry = bool(RNG.integers(0, 2))
    d1 = [0] * nd; d2 = [0] * nd
    d1[idx] = a; d2[idx] = b
    ikm = nd - 1 - (n - 1)
    if lower_carry:
        while True:
            x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
            if x + y >= 10: break
    else:
        x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
    d1[ikm] = x; d2[ikm] = y
    for k in range(n - 1):
        ik = nd - 1 - k
        u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
        d1[ik] = u; d2[ik] = v
    return _digits_to_int(d1), _digits_to_int(d2), (a + b) % 10


def collect_acts(model, cfg, n, pos, hook, cls, n_q=300):
    A = []
    for _ in range(n_q):
        a, b, sa = build_class_question(cfg, n, cls)
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, cache = model.run_with_cache(q.unsqueeze(0))
        A.append(cache[hook][0, pos, :].numpy())
    return np.array(A)


# ---------------------------------------------------------------------------
# geometry metrics
# ---------------------------------------------------------------------------

def centroid_geometry(means):
    """means: dict class->vector. Returns pairwise dists, U off-axis fraction
    (perp distance of U-mean from the c0-c1 line / |c1-c0|), and the
    angle(c0->c1, c0->Umean)."""
    c0, c1 = means["c0"], means["c1"]
    umean = 0.5 * (means["u0"] + means["u1"])
    axis = c1 - c0
    axlen = np.linalg.norm(axis) + 1e-9
    axhat = axis / axlen
    v = umean - c0
    along = np.dot(v, axhat)
    perp = v - along * axhat
    off_axis_frac = float(np.linalg.norm(perp) / axlen)
    # angle c0->c1 vs c0->umean
    v2 = umean - c0
    cosang = np.dot(axis, v2) / (np.linalg.norm(axis) * np.linalg.norm(v2) + 1e-9)
    angle = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return {
        "d_c0_c1": float(np.linalg.norm(c1 - c0)),
        "d_c0_u": float(np.linalg.norm(umean - c0)),
        "d_c1_u": float(np.linalg.norm(umean - c1)),
        "u_offaxis_frac": off_axis_frac,
        "angle_c0c1_c0u_deg": angle,
        "d_u0_u1": float(np.linalg.norm(means["u1"] - means["u0"])),
        # per-class distances (Gate-2 F1: these are load-bearing, must be stored)
        "d_c0_u0": float(np.linalg.norm(means["u0"] - c0)),
        "d_c1_u0": float(np.linalg.norm(means["u0"] - c1)),
        "d_c0_u1": float(np.linalg.norm(means["u1"] - c0)),
        "d_c1_u1": float(np.linalg.norm(means["u1"] - c1)),
    }


def resolution_probe(acts, folds=5):
    """CV linear probe U->0 vs U->1 (the resolution). High accuracy = the input
    already encodes the eventual resolution (informative). acts: dict with
    'u0','u1' arrays."""
    from quanta_maths.maths_probe import cross_val_probe_accuracy
    X = np.vstack([acts["u0"], acts["u1"]])
    y = np.r_[np.zeros(len(acts["u0"])), np.ones(len(acts["u1"]))]
    return cross_val_probe_accuracy(X, y, folds=folds, C=1.0)


def perm_null_offaxis(all_acts, n_perm=1000):
    """Permutation null for U off-axis fraction: shuffle the 4-class labels
    across the pooled activations, recompute the off-axis fraction."""
    labels = []
    X = []
    for c in ("c0", "c1", "u0", "u1"):
        X.append(all_acts[c]); labels += [c] * len(all_acts[c])
    X = np.vstack(X); labels = np.array(labels)
    obs = centroid_geometry({c: all_acts[c].mean(0) for c in all_acts})["u_offaxis_frac"]
    ge = 0
    for _ in range(n_perm):
        perm = RNG.permutation(labels)
        means = {c: X[perm == c].mean(0) for c in ("c0", "c1", "u0", "u1")}
        if centroid_geometry(means)["u_offaxis_frac"] >= obs:
            ge += 1
    return {"observed": obs, "p_value": (ge + 1) / (n_perm + 1)}


def between_var_share(all_acts):
    """Fraction of total variance that is between-class (4 classes)."""
    X = np.vstack([all_acts[c] for c in all_acts])
    gm = X.mean(0)
    tot = ((X - gm) ** 2).sum()
    bet = sum(len(all_acts[c]) * ((all_acts[c].mean(0) - gm) ** 2).sum() for c in all_acts)
    return float(bet / (tot + 1e-9))


def classify_shape(geo, res_probe, perm):
    """A-1 verdict. off-axis significance from the permutation null (A-4);
    square vs simplex from d(u0,u1) relative to d(c0,c1) (u0==u1 => simplex;
    u0 far from u1 => square), since the angle alone does not separate them."""
    off = geo["u_offaxis_frac"]; ang = geo["angle_c0c1_c0u_deg"]
    sig = perm["p_value"] < 0.01
    d_u = geo["d_u0_u1"] / (geo["d_c0_c1"] + 1e-9)   # normalized U-split
    if not sig and off < 0.2:
        return "R-scalar (U on the 0-1 line; not a third symbol)"
    if sig and off >= 0.4:
        if d_u >= 0.5:
            return "R-square (U->0 and U->1 distinct; two binary axes)"
        return "R-third-symbol/simplex (U off-axis, U->0~U->1)"
    if sig and 0.2 <= off < 0.4:
        return "R-partial (U somewhat off-axis)"
    return "R-other/ambiguous"


# ---------------------------------------------------------------------------
# planted-shape controls (A-3)
# ---------------------------------------------------------------------------

def planted_control(shape, d=64, n=300, noise=0.3):
    """Synthetic 4-class data with a known shape + isotropic noise; verify the
    pipeline recovers the shape from CENTROIDS (n large so centroid noise is
    small) and via the permutation null. Classes: c0,c1,u0,u1."""
    RNGl = np.random.default_rng(1)
    R = RNGl.standard_normal((d, 2)); R, _ = np.linalg.qr(R)
    coords = {
        "scalar": {"c0": [0., 0.], "c1": [1., 0.], "u0": [0.05, 0.], "u1": [0.95, 0.]},
        "simplex": {"c0": [0., 0.], "c1": [1., 0.], "u0": [0.5, 0.87], "u1": [0.5, 0.87]},
        "square": {"c0": [0., 0.], "c1": [1., 0.], "u0": [0., 1.], "u1": [1., 1.]},
    }[shape]
    out = {}
    for c, v in coords.items():
        mu = np.array(v) @ R.T
        out[c] = mu[None, :] + noise * RNGl.standard_normal((n, d))
    geo = centroid_geometry({c: out[c].mean(0) for c in out})
    perm = perm_null_offaxis(out, n_perm=500)
    dnorm = geo["d_u0_u1"] / (geo["d_c0_c1"] + 1e-9)
    return {"shape": shape, "off_axis": round(geo["u_offaxis_frac"], 3),
            "angle": round(geo["angle_c0c1_c0u_deg"], 0),
            "d_u0u1_norm": round(dnorm, 2), "perm_p": round(perm["p_value"], 3)}


# ---------------------------------------------------------------------------
# per-model
# ---------------------------------------------------------------------------

def committed_lowercarry_control(model, cfg, n, pos, hook, n_q=150):
    """F1 control: for a COMMITTED-0 digit, does toggling the lower carry move
    the probe activation? (If small vs d(u0,u1), the U-split is the resolved
    carry, not lower-operand identity.) Returns the centroid distance."""
    def means(lower_carry):
        A = []
        for _ in range(n_q):
            nd = cfg.n_digits; idx = nd - 1 - n
            while True:
                a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
                if a + b <= 8: break
            d1 = [0]*nd; d2 = [0]*nd; d1[idx] = a; d2[idx] = b
            ikm = nd - 1 - (n - 1)
            if lower_carry:
                while True:
                    x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                    if x + y >= 10: break
            else:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
            d1[ikm] = x; d2[ikm] = y
            for k in range(n - 1):
                ik = nd - 1 - k
                u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
                d1[ik] = u; d2[ik] = v
            q = make_q(cfg, _digits_to_int(d1), _digits_to_int(d2))
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0))
            A.append(c[hook][0, pos, :].numpy())
        return np.array(A).mean(0)
    return float(np.linalg.norm(means(True) - means(False)))


def precursor_vs_decision(model, cfg, n, pos, hook, n_q=200):
    """F3 discriminator: are the DECISION-INGREDIENTS separately linearly
    present at this site? Probe (a) carry_in (lower carry present) among U cases,
    (b) the sum==9 (is-U) flag: U vs committed. If BOTH carry_in and is-U are
    separately decodable here, the site holds the *ingredients*, not necessarily
    a completed U->{0,1} decision made upstream."""
    from quanta_maths.maths_probe import cross_val_probe_accuracy
    # carry_in probe: among U cases, u1 (carry) vs u0 (no carry) == resolution,
    # but also test carry_in on COMMITTED digits (where it does NOT change carry_out)
    # -> if carry_in is decodable on committed digits, the raw carry_in bit is present.
    cc, cn = [], []
    for _ in range(n_q):
        for lc, bucket in [(True, cc), (False, cn)]:
            nd = cfg.n_digits; idx = nd - 1 - n
            while True:
                a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
                if a + b <= 8: break  # committed-0
            d1 = [0]*nd; d2 = [0]*nd; d1[idx] = a; d2[idx] = b
            ikm = nd - 1 - (n - 1)
            if lc:
                while True:
                    x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                    if x + y >= 10: break
            else:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
            d1[ikm] = x; d2[ikm] = y
            q = make_q(cfg, _digits_to_int(d1), _digits_to_int(d2))
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0))
            bucket.append(c[hook][0, pos, :].numpy())
    Xc = np.vstack([cc, cn]); yc = np.r_[np.ones(len(cc)), np.zeros(len(cn))]
    carryin_on_committed = cross_val_probe_accuracy(Xc, yc, folds=5)
    return {"carry_in_decodable_on_committed": carryin_on_committed}


def analyze_site(model, cfg, n, pos, hook, n_q=300, extra=False):
    acts = {c: collect_acts(model, cfg, n, pos, hook, c, n_q) for c in ("c0", "c1", "u0", "u1")}
    means = {c: acts[c].mean(0) for c in acts}
    geo = centroid_geometry(means)
    res = resolution_probe(acts)
    perm = perm_null_offaxis(acts, n_perm=1000)
    bvs = between_var_share(acts)
    label = classify_shape(geo, res, perm)
    out = {"geometry": geo, "resolution_probe_acc": res,
           "offaxis_perm": perm, "between_var_share": bvs, "shape": label}
    if extra:
        out["committed_lowercarry_move"] = committed_lowercarry_control(model, cfg, n, pos, hook)
        out["precursor"] = precursor_vs_decision(model, cfg, n, pos, hook)
    return out


def run_models():
    results = {}
    # planted controls once
    controls = {s: planted_control(s) for s in ("scalar", "simplex", "square")}
    for mn, mc in CFG.items():
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg)
        assert acc > 0.9, f"{mn} acc {acc}"
        n = mc["digit"]; cpos = mc["combiner_pos"]
        print(f"=== {mn} acc={acc:.3f} n={n} combiner pos {cpos} ===")
        # primary: L1 MLP input (post-LN) at combiner position
        primary = analyze_site(model, cfg, n, cpos, "blocks.1.ln2.hook_normalized", extra=True)
        # secondary: pre-LN resid_mid
        secondary = analyze_site(model, cfg, n, cpos, "blocks.1.hook_resid_mid")
        # discriminator control: make-carry site residual (must show U collapsed)
        mp, ml, mh = mc["make_carry"]
        mc_site = analyze_site(model, cfg, n, mp, "blocks.0.hook_resid_post", n_q=200)
        results[mn] = {"accuracy": acc, "digit": n, "combiner_pos": cpos,
                       "primary_ln2": primary, "secondary_resid_mid": secondary,
                       "make_carry_site": mc_site}
        p = primary; g = p["geometry"]
        print(f"  [combiner input ln2] shape={p['shape']}")
        print(f"    off_axis={g['u_offaxis_frac']:.2f} perm_p={p['offaxis_perm']['p_value']:.3f} "
              f"U0vsU1_probe={p['resolution_probe_acc']:.2f} d(u0,u1)={g['d_u0_u1']:.2f}")
        print(f"    d(c0,u0)={g['d_c0_u0']:.1f} d(c1,u0)={g['d_c1_u0']:.1f} "
              f"d(c0,u1)={g['d_c0_u1']:.1f} d(c1,u1)={g['d_c1_u1']:.1f} d(c0,c1)={g['d_c0_c1']:.1f}")
        print(f"    committed-0 lower-carry move={p['committed_lowercarry_move']:.2f} "
              f"| carry_in decodable on committed={p['precursor']['carry_in_decodable_on_committed']:.2f}")
        print(f"  [make-carry site] off_axis={mc_site['geometry']['u_offaxis_frac']:.2f} "
              f"U0vsU1_probe={mc_site['resolution_probe_acc']:.2f} "
              f"(U should collapse: low off_axis + chance probe)")
        del model
    results["_planted_controls"] = controls
    print("planted controls:", {s: (round(c["off_axis"], 2), round(c["angle"], 0)) for s, c in controls.items()})
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "control":
        for s in ("scalar", "simplex", "square"):
            print(s, planted_control(s))
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
