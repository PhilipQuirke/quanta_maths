"""Earliest tri-state site: where does {0,1,U} exist before it collapses?
(study-earliest-tristate-site.md)

CE6 found the carry is RESOLVED BINARY at the L1-MLP combiner input. This sweeps
EARLIER residual sites to find whether a DEDICATED unresolved U symbol exists.

Discriminator (amendment A-1): AXIS DECOMPOSITION, not centroid distance
(u0/u1 differ in lower operands by construction, so distance is confounded).
Three axes from class means at each site:
  is_u  = mean(u0,u1) - mean(c0,c1)     "digit n is U/sum==9"
  res   = u1 - u0                        "resolution (lower carry)"
  komm  = c1 - c0                        "committed binary carry"
Verdicts:
  RESOLVED           : is-U absorbed by committed axis; u0->c0 end, u1->c1 end
  UNRESOLVED-TRISTATE: is-U axis significant AND ~orthogonal to res AND komm
  INGREDIENT         : is-U vanishes after partialling carry_in
  INFO-ABSENT        : classes not separable at the site

CPU-only. Run:
    PYTHONPATH=. python3 scripts/earliest_tristate_site.py control
    PYTHONPATH=. python3 scripts/earliest_tristate_site.py models
    PYTHONPATH=. python3 scripts/earliest_tristate_site.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.st_tristate_geometry import build_class_question
from scripts.confirm_st_node import load_model, make_q, verify_accuracy, _digits_to_int

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-earliest-tristate-site")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260716)

CFG = {
    "add_d5_l2_h3_t15K_s372001": {"combiner_pos": 14, "digit": 2},
    "add_d6_l2_h3_t20K_s173289": {"combiner_pos": 16, "digit": 3},
}

# sweep sites at the answer position, increasing depth
SITES = [
    ("L0.resid_pre", "blocks.0.hook_resid_pre"),
    ("L0.mlp_in(ln2)", "blocks.0.ln2.hook_normalized"),
    ("L0.resid_post", "blocks.0.hook_resid_post"),
    ("L1.resid_pre", "blocks.1.hook_resid_pre"),
    ("L1.attn_in(ln1)", "blocks.1.ln1.hook_normalized"),
    ("L1.resid_mid", "blocks.1.hook_resid_mid"),
    ("L1.mlp_in(ln2)=combiner", "blocks.1.ln2.hook_normalized"),
]


def collect(model, cfg, n, pos, hook, cls, n_q=300):
    A = []
    for _ in range(n_q):
        a, b, _ = build_class_question(cfg, n, cls)
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, cache = model.run_with_cache(q.unsqueeze(0))
        A.append(cache[hook][0, pos, :].numpy())
    return np.array(A)


def unit(v):
    nrm = np.linalg.norm(v)
    return v / (nrm + 1e-9)


def var_share_along(X, direction):
    """Fraction of centered total variance captured by projection onto a unit
    direction."""
    Xc = X - X.mean(0)
    d = unit(direction)
    proj = Xc @ d
    return float((proj ** 2).sum() / ((Xc ** 2).sum() + 1e-9))


def axis_decomp(acts):
    """acts: dict class->array. Returns axis angles, variance shares, and the
    RESOLVED/UNRESOLVED diagnostics."""
    m = {c: acts[c].mean(0) for c in acts}
    is_u = 0.5 * (m["u0"] + m["u1"]) - 0.5 * (m["c0"] + m["c1"])
    res = m["u1"] - m["u0"]
    komm = m["c1"] - m["c0"]
    def cos(a, b): return float(abs(np.dot(unit(a), unit(b))))
    X = np.vstack([acts[c] for c in acts])
    # is-U variance share = share along is_u AFTER removing komm & res components
    # (so "is there a SEPARATE is-U direction")
    B = np.stack([unit(komm), unit(res)], 1)  # (d,2)
    # residual of is_u orthogonal to komm,res
    isu_perp = is_u - B @ (B.T @ is_u)
    return {
        "cos_isU_komm": cos(is_u, komm),
        "cos_isU_res": cos(is_u, res),
        "cos_res_komm": cos(res, komm),
        "isU_varshare": var_share_along(X, is_u),
        "isU_perp_varshare": var_share_along(X, isu_perp),
        "res_varshare": var_share_along(X, res),
        "komm_varshare": var_share_along(X, komm),
        "isU_perp_norm_frac": float(np.linalg.norm(isu_perp) / (np.linalg.norm(is_u) + 1e-9)),
    }


def separability(acts):
    """between-class variance share (4 classes) + d(c0,c1) vs within-class spread."""
    X = np.vstack([acts[c] for c in acts]); gm = X.mean(0)
    tot = ((X - gm) ** 2).sum()
    bet = sum(len(acts[c]) * ((acts[c].mean(0) - gm) ** 2).sum() for c in acts)
    within = np.mean([np.mean(np.linalg.norm(acts[c] - acts[c].mean(0), axis=1)) for c in acts])
    d_c0c1 = float(np.linalg.norm(acts["c1"].mean(0) - acts["c0"].mean(0)))
    bvs = float(bet / (tot + 1e-9))
    if tot < 1e-6:  # 0/0: activations identical across classes (info-absent)
        bvs = 0.0
    return {"between_var_share": bvs,
            "d_c0_c1": d_c0c1, "within_spread": float(within),
            "c0c1_over_within": float(d_c0c1 / (within + 1e-9))}


def perm_null_isU(acts, n_perm=500):
    """Permutation null for the is-U perpendicular variance share."""
    X = []; labels = []
    for c in ("c0", "c1", "u0", "u1"):
        X.append(acts[c]); labels += [c] * len(acts[c])
    X = np.vstack(X); labels = np.array(labels)
    obs = axis_decomp(acts)["isU_perp_varshare"]
    ge = 0
    for _ in range(n_perm):
        perm = RNG.permutation(labels)
        pa = {c: X[perm == c] for c in ("c0", "c1", "u0", "u1")}
        if axis_decomp(pa)["isU_perp_varshare"] >= obs:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def carry_in_partial(model, cfg, n, pos, hook, acts, n_q=200):
    """Contrastive ingredient probe (A-3): (1) carry_in decodable on COMMITTED
    digits; (2) does U-vs-committed separability SURVIVE partialling carry_in?"""
    from quanta_maths.maths_probe import cross_val_probe_accuracy
    # committed-0 with vs without lower carry -> carry_in direction
    cc, cn = [], []
    for _ in range(n_q):
        for lc, bucket in [(True, cc), (False, cn)]:
            nd = cfg.n_digits; idx = nd - 1 - n
            while True:
                a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
                if a + b <= 8: break
            d1 = [0]*nd; d2 = [0]*nd; d1[idx] = a; d2[idx] = b
            ikm = nd - 1 - (n - 1)
            if lc:
                while True:
                    x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                    if x + y >= 10: break
            else:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
            d1[ikm] = x; d2[ikm] = y
            for k in range(n - 1):
                ik = nd - 1 - k; u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
                d1[ik] = u; d2[ik] = v
            q = make_q(cfg, _digits_to_int(d1), _digits_to_int(d2))
            with torch.no_grad():
                _, c = model.run_with_cache(q.unsqueeze(0))
            bucket.append(c[hook][0, pos, :].numpy())
    Xc = np.vstack([cc, cn]); yc = np.r_[np.ones(len(cc)), np.zeros(len(cn))]
    carryin_acc = cross_val_probe_accuracy(Xc, yc, folds=5)
    carry_dir = unit(np.array(cc).mean(0) - np.array(cn).mean(0))
    # U vs committed separability, before and after partialling carry_dir
    Xu = np.vstack([acts["u0"], acts["u1"]]); Xk = np.vstack([acts["c0"], acts["c1"]])
    def partial(M): return M - np.outer(M @ carry_dir, carry_dir)
    def sep(A, B):
        X = np.vstack([A, B]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
        return cross_val_probe_accuracy(X, y, folds=5)
    return {"carry_in_decodable_committed": carryin_acc,
            "U_vs_committed_sep": sep(Xu, Xk),
            "U_vs_committed_sep_partialled": sep(partial(Xu), partial(Xk))}


def classify(dec, sep, perm_p, ing):
    if sep["between_var_share"] < 0.02 or sep["c0c1_over_within"] < 0.2:
        return "INFO-ABSENT"
    # significance from the permutation null (not an absolute varshare bar --
    # digit-embedding lesson: absolute shares are noise-scale dependent)
    isu_sig = (perm_p < 0.01)
    orthogonal = dec["cos_isU_komm"] < 0.5 and dec["cos_isU_res"] < 0.5
    # F5 fix: a clean resolution-split NULLS is_u (is-U-perp n.s.), it does not
    # make cos(is_u, komm) high -- so RESOLVED = "is-U-perp not significant" +
    # U separable from committed (routed toward the committed axis by resolution).
    survives = ing["U_vs_committed_sep_partialled"] > 0.5 + 0.5 * (ing["U_vs_committed_sep"] - 0.5)
    if isu_sig and orthogonal and survives:
        return "UNRESOLVED-TRISTATE"
    if not survives and ing["U_vs_committed_sep"] > 0.7:
        return "INGREDIENT"
    if (not isu_sig) and ing["U_vs_committed_sep"] > 0.9:
        return "RESOLVED (U split toward committed axis)"
    return "PARTIAL/AMBIGUOUS"


def per_class_distances(acts):
    """F3 (Gate-2): substantiate the resolution-split IN-ARTIFACT."""
    m = {c: acts[c].mean(0) for c in acts}
    def d(a, b): return float(np.linalg.norm(m[a] - m[b]))
    return {"d_c0_u0": d("c0", "u0"), "d_c1_u0": d("c1", "u0"),
            "d_c0_u1": d("c0", "u1"), "d_c1_u1": d("c1", "u1"),
            "d_c0_c1": d("c0", "c1"), "d_u0_u1": d("u0", "u1"),
            "u0_nearest": "c0" if d("c0", "u0") < d("c1", "u0") else "c1",
            "u1_nearest": "c0" if d("c0", "u1") < d("c1", "u1") else "c1"}


def analyze_site(model, cfg, n, pos, hook, ingredient=True, n_q=300):
    acts = {c: collect(model, cfg, n, pos, hook, c, n_q) for c in ("c0", "c1", "u0", "u1")}
    dec = axis_decomp(acts)
    sep = separability(acts)
    dists = per_class_distances(acts)
    perm_p = perm_null_isU(acts)
    ing = carry_in_partial(model, cfg, n, pos, hook, acts) if ingredient else {}
    label = classify(dec, sep, perm_p, ing) if ingredient else "n/a"
    return {"axes": dec, "separability": sep, "isU_perp_perm_p": perm_p,
            "per_class_distances": dists, "ingredient": ing, "label": label}


# ---------------------------------------------------------------------------
# planted controls (A-4)
# ---------------------------------------------------------------------------

def planted(kind, d=64, nq=300, noise=0.25):
    R = np.random.default_rng(1).standard_normal((d, 3)); R, _ = np.linalg.qr(R)
    # coords in a 3D synthetic space: [committed axis, is-U axis, carry_in axis]
    coords = {
        # RESOLVED: is-U collinear w/ committed; u0 at c0 end, u1 at c1 end
        "resolved":  {"c0": [0, 0, 0], "c1": [1, 0, 0], "u0": [0.1, 0, 0], "u1": [0.9, 0, 0]},
        # UNRESOLVED: u0,u1 share an is-U offset orthogonal to committed; small resolution
        "unresolved": {"c0": [0, 0, 0], "c1": [1, 0, 0], "u0": [0.5, 1, 0], "u1": [0.5, 1, 0.05]},
        # INGREDIENT: u0/u1 separated only along the carry_in axis (3rd dim)
        "ingredient": {"c0": [0, 0, 0], "c1": [1, 0, 0], "u0": [0.5, 0, 0], "u1": [0.5, 0, 1]},
        # INFO-ABSENT: all classes at the origin
        "absent":    {"c0": [0, 0, 0], "c1": [0, 0, 0], "u0": [0, 0, 0], "u1": [0, 0, 0]},
    }[kind]
    acts = {}
    for c, v in coords.items():
        mu = np.array(v, float) @ R.T
        acts[c] = mu[None, :] + noise * np.random.default_rng(hash(kind + c) % 2**32).standard_normal((nq, d))
    dec = axis_decomp(acts); sep = separability(acts); perm = perm_null_isU(acts, 300)
    return {"kind": kind, "isU_perp_vs": round(dec["isU_perp_varshare"], 3),
            "cos_isU_komm": round(dec["cos_isU_komm"], 2),
            "cos_isU_res": round(dec["cos_isU_res"], 2),
            "between_vs": round(sep["between_var_share"], 3),
            "c0c1_over_within": round(sep["c0c1_over_within"], 2), "perm_p": round(perm, 3)}


def run_models():
    results = {"_planted": {k: planted(k) for k in ("resolved", "unresolved", "ingredient", "absent")}}
    print("planted controls:")
    for k, v in results["_planted"].items():
        print(f"  {k:11s} isU_perp_vs={v['isU_perp_vs']} cos(isU,komm)={v['cos_isU_komm']} "
              f"cos(isU,res)={v['cos_isU_res']} between_vs={v['between_vs']} "
              f"c0c1/within={v['c0c1_over_within']} perm_p={v['perm_p']}")
    for mn, mc in CFG.items():
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg)
        assert acc > 0.9
        n = mc["digit"]; cpos = mc["combiner_pos"]
        print(f"\n=== {mn} acc={acc:.3f} n={n} pos={cpos} ===")
        sites = {}
        for tag, hook in SITES:
            r = analyze_site(model, cfg, n, cpos, hook)
            sites[tag] = r
            a = r["axes"]; s = r["separability"]
            print(f"  {tag:26s} {r['label']:20s} isU_perp_vs={a['isU_perp_varshare']:.3f} "
                  f"cos(isU,komm)={a['cos_isU_komm']:.2f} cos(isU,res)={a['cos_isU_res']:.2f} "
                  f"perm_p={r['isU_perp_perm_p']:.3f} bvs={s['between_var_share']:.2f} "
                  f"Usep={r['ingredient'].get('U_vs_committed_sep',0):.2f}"
                  f"/{r['ingredient'].get('U_vs_committed_sep_partialled',0):.2f}")
        results[mn] = {"accuracy": acc, "digit": n, "combiner_pos": cpos, "sites": sites}
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "control":
        for k in ("resolved", "unresolved", "ingredient", "absent"):
            print(planted(k))
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
