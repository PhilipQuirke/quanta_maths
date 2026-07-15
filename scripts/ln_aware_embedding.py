"""LN-aware digit-embedding close-out (A-9) (study-ln-aware-embedding.md).

Closes the pre-registered but un-run LN-aware secondary of the digit-embedding
study (CE1). CE1 (raw W_E) found the digit embedding near-isotropic with a WEAK,
provisional circular ORDERING signal. This re-measures the geometry on the
LN-EFFECTIVE embedding (what layer-0 attention actually reads) and reports the
raw-vs-LN comparison per the extended A-9/A-11 rule.

Key design (amendments A-11..A-15):
  * PRIMARY comparison = the angular-ordering permutation p (raw vs LN), since
    that is the provisional signal (the freq1-based classified read is pinned).
  * LN variants: normalize-only (center per-vector + unit-std) and full-LN
    (+ gamma/beta), isolating gamma's contribution [A-12].
  * (A) position-free LN of W_E rows (primary); (B) model-true
    blocks.0.ln1.hook_normalized at operand digit positions (faithfulness check).
  * planted-circle-through-LN + untrained-LN controls validate the LN-composed
    pipeline [A-12/A-14].
  * BOTH raw and LN numbers computed here -> results.json [A-13].

CPU-only. Run:
    PYTHONPATH=. python3 scripts/ln_aware_embedding.py control
    PYTHONPATH=. python3 scripts/ln_aware_embedding.py models
    PYTHONPATH=. python3 scripts/ln_aware_embedding.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.digit_embedding_geometry import (
    analyze_matrix, permutation_pvalues, freq1_plane_share, angular_order_stat,
    _plane_coords, centered, real_dft_basis,
)
from scripts.confirm_st_node import load_model

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-ln-aware-embedding")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260716)
N_PERM = 10000
EPS = 1e-5

ACCURATE = ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t15K_s372001",
            "add_d6_l2_h3_t20K_s173289", "add_d6_l2_h3_t20K_s572091"]


# ---------------------------------------------------------------------------
# LN transforms (option A: position-free LN of W_E rows)
# ---------------------------------------------------------------------------

def ln_apply(M, gamma=None, beta=None, eps=EPS):
    """LayerNorm each ROW over d_model: center per-vector, unit-std, then gamma/beta.
    M: (n, d_model). Returns (n, d_model)."""
    mu = M.mean(axis=1, keepdims=True)
    x = M - mu
    var = (x ** 2).mean(axis=1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    if gamma is not None:
        xn = xn * gamma[None, :]
    if beta is not None:
        xn = xn + beta[None, :]
    return xn


def load_WE_and_ln(model_name):
    """Return raw W_E digit rows (10,d) and layer-0 ln1 gamma,beta."""
    model, cfg = load_model(model_name)
    W_E = model.W_E.detach().numpy()[:10]
    gamma = model.blocks[0].ln1.w.detach().numpy()
    beta = model.blocks[0].ln1.b.detach().numpy()
    return model, cfg, W_E, gamma, beta


# ---------------------------------------------------------------------------
# ordering-p (the PRIMARY comparison statistic) -- reuse harness pieces
# ---------------------------------------------------------------------------

def ordering_pvalue(M, n_perm=N_PERM):
    """Angular-ordering permutation p on the centered 10-vector set (matches the
    digit-embedding harness's angular_order_p)."""
    obs = angular_order_stat(_plane_coords(M, "pc"))
    le = 0
    for _ in range(n_perm):
        perm = RNG.permutation(10)
        if angular_order_stat(_plane_coords(M[perm], "pc")) <= obs:
            le += 1
    return (le + 1) / (n_perm + 1)


def geom_summary(M, n_perm=N_PERM):
    """Compact geometry summary for one 10xd matrix."""
    a = analyze_matrix(M, n_perm=n_perm)  # freq1, ordering, classified read, PR
    return {
        "freq1_share": a["freq1_share"], "freq1_p": a["freq1_p"],
        "angular_order_p": a["angular_order_p"],
        "participation_ratio": a["participation_ratio"],
        "classified_read": a["classified_read"],
        "wraparound_ratio": a["wraparound_ratio"],
    }


# ---------------------------------------------------------------------------
# controls (A-12 / A-14)
# ---------------------------------------------------------------------------

def planted_circle(d, share=0.5, n_noise_scale=1.0):
    digits = np.arange(10)
    sig = np.stack([np.cos(2*np.pi*digits/10), np.sin(2*np.pi*digits/10)], 1)
    R = RNG.standard_normal((d, 2)); R, _ = np.linalg.qr(R)
    S = centered(sig @ R.T)
    N = centered(RNG.standard_normal((10, d)))
    sv, nv = (S**2).sum(), (N**2).sum()
    alpha = np.sqrt(share * nv / ((1 - share) * sv + 1e-12))
    return alpha * S + n_noise_scale * N


def run_controls(gamma, beta, d):
    """planted-circle-through-LN must still detect; untrained handled in models."""
    out = {}
    # planted circle, raw vs LN(full)
    pc = planted_circle(d, share=0.5)
    out["planted_circle_raw"] = {"freq1_share": freq1_plane_share(pc),
                                 "ordering_p": ordering_pvalue(pc, 2000)}
    pc_ln = ln_apply(pc, gamma, beta)
    out["planted_circle_ln_full"] = {"freq1_share": freq1_plane_share(pc_ln),
                                     "ordering_p": ordering_pvalue(pc_ln, 2000)}
    return out


# ---------------------------------------------------------------------------
# main per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model_name, n_perm=N_PERM):
    model, cfg, W_E, gamma, beta = load_WE_and_ln(model_name)
    d = W_E.shape[1]
    raw = geom_summary(W_E, n_perm)
    ln_norm = geom_summary(ln_apply(W_E, None, None), n_perm)          # normalize-only
    ln_full = geom_summary(ln_apply(W_E, gamma, beta), n_perm)         # full LN
    # Option B: model-true ln1.hook_normalized at operand digit positions
    b_positions = {}
    for n in [0, cfg.n_digits // 2]:  # units and a mid digit
        dn_pos = cfg.n_digits - 1 - n
        # feed each digit token at that input position, capture ln1 normalized
        acts = []
        for dig in range(10):
            q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
            q[0, dn_pos] = dig
            with torch.no_grad():
                _, c = model.run_with_cache(q)
            acts.append(c["blocks.0.ln1.hook_normalized"][0, dn_pos, :].numpy())
        b_positions[f"D{n}_pos{dn_pos}"] = geom_summary(np.array(acts), n_perm)
    gamma_spread = {"mean": float(gamma.mean()), "std": float(gamma.std()),
                    "min": float(gamma.min()), "max": float(gamma.max())}
    del model
    return {"raw": raw, "ln_normalize_only": ln_norm, "ln_full": ln_full,
            "option_B_model_true": b_positions, "gamma_spread": gamma_spread}


def untrained_ln_control(n_perm=N_PERM):
    from quanta_maths.maths_config import MathsConfig
    from transformer_lens import HookedTransformer
    cfg = MathsConfig(); cfg.set_model_names(ACCURATE[0])
    htc = cfg.get_HookedTransformerConfig(); htc.device = "cpu"; htc.init_weights = True; htc.seed = 999
    model = HookedTransformer(htc)
    W_E = model.W_E.detach().numpy()[:10]
    gamma = model.blocks[0].ln1.w.detach().numpy(); beta = model.blocks[0].ln1.b.detach().numpy()
    raw = geom_summary(W_E, n_perm)
    ln_full = geom_summary(ln_apply(W_E, gamma, beta), n_perm)
    # permutation-null FPR check on the LN untrained matrix
    lnM = ln_apply(W_E, gamma, beta)
    fp = 0; NC = 200
    for _ in range(NC):
        stats = permutation_pvalues(RNG.standard_normal((10, W_E.shape[1])), n_perm=1000)
        if stats["angular_order_p"] < 0.01:
            fp += 1
    del model
    return {"raw": raw, "ln_full": ln_full,
            "perm_null_fpr_alpha01": fp / NC}


def agreement(raw, ln):
    """A-11 rule: disagreement if classified read changes, OR ordering-p crosses
    p<0.01 boundary, OR freq1 crosses the 25%/40% band."""
    def band(x): return "hi" if x >= 0.40 else ("mid" if x >= 0.25 else "lo")
    read_change = raw["classified_read"] != ln["classified_read"]
    order_cross = (raw["angular_order_p"] < 0.01) != (ln["angular_order_p"] < 0.01)
    band_cross = band(raw["freq1_share"]) != band(ln["freq1_share"])
    dis = read_change or order_cross or band_cross
    return {"disagree": bool(dis), "read_change": bool(read_change),
            "ordering_sig_cross": bool(order_cross), "freq1_band_cross": bool(band_cross)}


def run_models():
    results = {"models": {}}
    print("=== controls ===", flush=True)
    _, _, _, g0, b0 = load_WE_and_ln(ACCURATE[0])
    ctrl = run_controls(g0, b0, 510)
    print(f"  planted circle: raw freq1={ctrl['planted_circle_raw']['freq1_share']:.2f} "
          f"ord_p={ctrl['planted_circle_raw']['ordering_p']:.4f} | "
          f"LN freq1={ctrl['planted_circle_ln_full']['freq1_share']:.2f} "
          f"ord_p={ctrl['planted_circle_ln_full']['ordering_p']:.4f}")
    unt = untrained_ln_control()
    print(f"  untrained: raw ord_p={unt['raw']['angular_order_p']:.3f} "
          f"LN ord_p={unt['ln_full']['angular_order_p']:.3f} "
          f"perm_FPR={unt['perm_null_fpr_alpha01']:.3f}")
    results["controls"] = ctrl
    results["untrained_control"] = unt

    sig_raw = 0; sig_ln = 0
    for mn in ACCURATE:
        print(f"=== {mn} ===", flush=True)
        r = analyze_model(mn)
        ag = agreement(r["raw"], r["ln_full"])
        r["raw_vs_lnfull_agreement"] = ag
        results["models"][mn] = r
        if r["raw"]["angular_order_p"] < 0.01: sig_raw += 1
        if r["ln_full"]["angular_order_p"] < 0.01: sig_ln += 1
        print(f"  raw:  ord_p={r['raw']['angular_order_p']:.4f} freq1={r['raw']['freq1_share']:.2f} read={r['raw']['classified_read']}")
        print(f"  LNn:  ord_p={r['ln_normalize_only']['angular_order_p']:.4f} freq1={r['ln_normalize_only']['freq1_share']:.2f}")
        print(f"  LNf:  ord_p={r['ln_full']['angular_order_p']:.4f} freq1={r['ln_full']['freq1_share']:.2f} read={r['ln_full']['classified_read']} | disagree={ag['disagree']} (ord_cross={ag['ordering_sig_cross']})")
        print(f"  gamma spread: mean={r['gamma_spread']['mean']:.3f} std={r['gamma_spread']['std']:.3f}")
    results["summary"] = {"n_accurate": len(ACCURATE),
                          "raw_ordering_significant": sig_raw,
                          "ln_ordering_significant": sig_ln,
                          "seed": 20260716, "n_perm": N_PERM}
    print(f"\nSUMMARY: raw ordering significant in {sig_raw}/{len(ACCURATE)}; "
          f"LN ordering significant in {sig_ln}/{len(ACCURATE)}")
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "control":
        _, _, _, g0, b0 = load_WE_and_ln(ACCURATE[0])
        print("planted:", run_controls(g0, b0, 510))
        print("untrained:", untrained_ln_control())
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
