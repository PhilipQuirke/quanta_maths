"""Digit-embedding geometry audit (study-digit-embedding-geometry.md).

Weights-only, CPU-only analysis of whether digit-token embeddings / unembeddings
of accurate trained addition models carry functional numeric geometry
(circular / helical / linear) versus an unstructured 10-way categorical code.

Implements the pre-registered metrics and the post-skeptic amendments A-1..A-10:
  * per-component variance-share statistics (freq-1 plane; unique linear share)
  * complete marginal real-DFT spectrum over 10 digits (k = 0..5)
  * label-permutation null (empirical p-values) for freq-1 and unique-linear
  * circular-order / wrap-around statistics with permutation p-values
  * embed vs unembed principal-angle comparison
  * positive-control sweep: planted circle / helix / noise, graded shares,
    permutation-null calibration (isotropic + anisotropic arms)

Run:
    python3 scripts/digit_embedding_geometry.py control   # positive control
    python3 scripts/digit_embedding_geometry.py models    # real weights
    python3 scripts/digit_embedding_geometry.py all
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "study-digit-embedding-geometry",
)
os.makedirs(RESULT_DIR, exist_ok=True)

DIGITS = np.arange(10)
N_PERM = 10000
N_CALIB = 200
ALPHA = 0.01
RNG = np.random.default_rng(20260714)


# ---------------------------------------------------------------------------
# Fourier / linear / ordering statistics -- now imported from the library
# (quanta_maths.maths_probe). Only the freq1-plane coords + adjacency descriptor
# (not exposed by the library) remain local.
# ---------------------------------------------------------------------------

from quanta_maths.maths_probe import (
    real_dft_basis, marginal_dft_spectrum, freq1_plane_share, unique_linear_share,
    angular_order_stat, wraparound_ratio, pc_plane_coords)

DFT_Q, DFT_NAMES = real_dft_basis()


def centered(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=0, keepdims=True)


def _plane_coords(M: np.ndarray, plane: str) -> np.ndarray:
    if plane == "pc":
        return pc_plane_coords(M)
    elif plane == "freq1":
        Mc = centered(M)
        idx = [DFT_NAMES.index("cos1"), DFT_NAMES.index("sin1")]
        P = DFT_Q[:, idx]
        return (P.T @ Mc).T
    raise ValueError(plane)


def adjacency_violations(coords2d: np.ndarray) -> int:
    """Human-readable descriptor: # of times circular neighbor by angle is not
    a value-neighbor (A-4 keeps this only as descriptor)."""
    ang = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    circ_order = np.argsort(ang)
    pos = {int(dig): i for i, dig in enumerate(circ_order)}
    viol = 0
    for dig in range(10):
        nxt = (dig + 1) % 10
        if abs((pos[dig] - pos[nxt]) % 10) not in (1, 9):
            viol += 1
    return viol


# ---------------------------------------------------------------------------
# Permutation null (A-1, A-3, A-4)
# ---------------------------------------------------------------------------

def permutation_pvalues(M: np.ndarray, n_perm: int = N_PERM) -> dict:
    """Empirical p-values for freq1-plane share, unique-linear share, and
    angular-order statistic, under random relabeling of the 10 digits.

    Relabeling permutes which row is digit 0..9 -> tests LABEL ALIGNMENT;
    the point cloud's intrinsic low-rankness is invariant to relabeling.
    """
    obs_f1 = freq1_plane_share(M)
    obs_lin = unique_linear_share(M)
    obs_ord = angular_order_stat(_plane_coords(M, "pc"))

    ge_f1 = ge_lin = le_ord = 0
    for _ in range(n_perm):
        perm = RNG.permutation(10)
        Mp = M[perm]
        if freq1_plane_share(Mp) >= obs_f1:
            ge_f1 += 1
        if unique_linear_share(Mp) >= obs_lin:
            ge_lin += 1
        # ordering is a "low is structured" statistic -> p = P(perm <= obs)
        if angular_order_stat(_plane_coords(Mp, "pc")) <= obs_ord:
            le_ord += 1
    return {
        "freq1_share": obs_f1,
        "freq1_p": (ge_f1 + 1) / (n_perm + 1),
        "unique_linear_share": obs_lin,
        "unique_linear_p": (ge_lin + 1) / (n_perm + 1),
        "angular_order_stat": obs_ord,
        "angular_order_p": (le_ord + 1) / (n_perm + 1),
    }


# ---------------------------------------------------------------------------
# Classification into R1 / R2 / R3 / R4 (A-1 decision rule)
# ---------------------------------------------------------------------------

def classify(stats: dict) -> str:
    f1, f1p = stats["freq1_share"], stats["freq1_p"]
    lin, linp = stats["unique_linear_share"], stats["unique_linear_p"]
    ordp = stats["angular_order_p"]
    wrap = stats.get("wraparound_ratio", None)
    wrap_ok = (ordp < ALPHA) and (wrap is not None and 0.5 <= wrap <= 2.0)

    f1_sig = f1p < ALPHA
    lin_sig = linp < ALPHA
    f1_partial = f1 >= 0.25 and f1_sig
    lin_partial = lin >= 0.25 and lin_sig
    f1_full = f1 >= 0.40 and f1_sig
    lin_full = lin >= 0.40 and lin_sig

    if not (f1_partial or lin_partial):
        return "R4"
    # helix: both individually partial, neither dominates by >2x
    if f1_partial and lin_partial and (0.5 <= (f1 / max(lin, 1e-9)) <= 2.0):
        return "R2"
    if (f1_full or (f1_partial and f1 >= lin)) and wrap_ok:
        return "R1"
    if (lin_full or (lin_partial and lin > f1)) and not wrap_ok:
        return "R3"
    # partial-but-unclear
    return "partial/ambiguous"


def analyze_matrix(M: np.ndarray, n_perm: int = N_PERM) -> dict:
    Mc = centered(M)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    var = S ** 2
    var_share = (var / var.sum()).tolist() if var.sum() > 0 else [0] * len(S)
    part_ratio = float((var.sum() ** 2) / (var ** 2).sum()) if var.sum() > 0 else 0.0

    stats = permutation_pvalues(M, n_perm)
    stats["singular_values"] = S.tolist()
    stats["variance_shares"] = var_share
    stats["participation_ratio"] = part_ratio
    stats["marginal_dft"] = marginal_dft_spectrum(M)
    pc = _plane_coords(M, "pc")
    stats["wraparound_ratio"] = wraparound_ratio(pc)
    stats["adjacency_violations"] = adjacency_violations(pc)
    stats["pc_plane_coords"] = pc.tolist()
    stats["classified_read"] = classify(stats)
    return stats


def principal_angles(A: np.ndarray, B: np.ndarray, k: int = 2) -> list:
    """Principal angles (deg) between top-k left-singular subspaces of two
    centered matrices (embed vs unembed geometry comparison, Metric 5)."""
    def topk(M):
        Mc = centered(M)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        return U[:, :k]
    Ua, Ub = topk(A), topk(B)
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    s = np.clip(s, -1, 1)
    return np.degrees(np.arccos(s)).tolist()


# ---------------------------------------------------------------------------
# Positive control (A-3, A-8)
# ---------------------------------------------------------------------------

def _planted(kind: str, share: float, d_model: int = 510,
             noise_cov_sqrt: np.ndarray | None = None) -> np.ndarray:
    """Build a 10 x d_model matrix with planted structure at target variance
    share, embedded in random directions, plus noise."""
    d = DIGITS
    if kind == "circle":
        sig = np.stack([np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10)], 1)
    elif kind == "helix":
        sig = np.stack([np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10),
                        (d - d.mean()) / 3.0], 1)
    elif kind == "noise":
        sig = np.zeros((10, 1))
    else:
        raise ValueError(kind)
    # random orthonormal embedding directions into d_model
    R = RNG.standard_normal((d_model, sig.shape[1]))
    R, _ = np.linalg.qr(R)
    S = sig @ R.T
    S = centered(S)
    if kind == "noise":
        share = 0.0
    # noise
    if noise_cov_sqrt is not None:
        N = RNG.standard_normal((10, d_model)) @ noise_cov_sqrt.T
    else:
        N = RNG.standard_normal((10, d_model))
    N = centered(N)
    sv, nv = (S ** 2).sum(), (N ** 2).sum()
    if kind == "noise":
        return N
    # scale so signal variance share == target
    alpha = np.sqrt(share * nv / ((1 - share) * sv + 1e-12))
    return alpha * S + N


def run_positive_control(n_perm: int = 2000) -> dict:
    out = {"detection": [], "calibration": {}}
    shares = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    for kind in ("circle", "helix"):
        for sh in shares:
            M = _planted(kind, sh)
            st = analyze_matrix(M, n_perm=n_perm)
            stat = st["freq1_share"] if kind == "circle" else \
                max(st["freq1_share"], st["unique_linear_share"])
            p = st["freq1_p"] if kind == "circle" else \
                min(st["freq1_p"], st["unique_linear_p"])
            out["detection"].append({
                "kind": kind, "planted_share": sh,
                "recovered_freq1": st["freq1_share"],
                "recovered_unique_linear": st["unique_linear_share"],
                "detect_stat": stat, "detect_p": p,
                "detected_at_40_p01": bool(stat >= 0.40 and p < ALPHA),
                "classified": st["classified_read"],
            })
    # calibration: isotropic noise FPR at alpha
    fp_f1 = fp_lin = 0
    for _ in range(N_CALIB):
        M = _planted("noise", 0.0)
        st = permutation_pvalues(M, n_perm=n_perm)
        if st["freq1_p"] < ALPHA:
            fp_f1 += 1
        if st["unique_linear_p"] < ALPHA:
            fp_lin += 1
    out["calibration"]["isotropic"] = {
        "n": N_CALIB, "alpha": ALPHA,
        "fpr_freq1": fp_f1 / N_CALIB, "fpr_unique_linear": fp_lin / N_CALIB,
    }
    # anisotropic arm: noise shaped by a decaying spectrum, planted circle at 40%
    d_model = 510
    spec = 1.0 / np.sqrt(np.arange(1, d_model + 1))
    cov_sqrt = np.diag(spec)
    M = _planted("circle", 0.40, noise_cov_sqrt=cov_sqrt)
    st = analyze_matrix(M, n_perm=n_perm)
    fp_a = 0
    for _ in range(N_CALIB):
        Mn = RNG.standard_normal((10, d_model)) @ cov_sqrt.T
        stn = permutation_pvalues(Mn, n_perm=n_perm)
        if stn["freq1_p"] < ALPHA:
            fp_a += 1
    out["calibration"]["anisotropic"] = {
        "planted_circle_40_recovered_freq1": st["freq1_share"],
        "planted_circle_40_detect_p": st["freq1_p"],
        "detected": bool(st["freq1_share"] >= 0.40 and st["freq1_p"] < ALPHA),
        "fpr_freq1": fp_a / N_CALIB, "n": N_CALIB, "alpha": ALPHA,
    }
    return out


# ---------------------------------------------------------------------------
# Real model weights
# ---------------------------------------------------------------------------

MODELS = {
    "primary": "add_d5_l2_h3_t15K_s372001",
    "repA_size": "add_d6_l2_h3_t15K_s372001",
    "repB_seed": "add_d6_l2_h3_t20K_s173289",
    "repC_seed": "add_d6_l2_h3_t20K_s572091",
    "descriptive_inaccurate": "add_d5_l1_h3_t30K_s372001",
}


def load_WE_WU(model_name: str):
    import torch
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="PhilipQuirke/VerifiedArithmetic",
                        filename=f"{model_name}.pth")
    sd = torch.load(p, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and "embed.W_E" not in sd:
        sd = sd["model"]
    W_E = sd["embed.W_E"].float().numpy()      # (15, 510)
    W_U = sd["unembed.W_U"].float().numpy().T  # (15, 510): rows = digit outputs
    return W_E[:10], W_U[:10]                  # digit tokens 0-9


def untrained_control():
    """Freshly initialized model of the primary's config, digit rows only."""
    import torch
    from quanta_maths.maths_config import MathsConfig
    from transformer_lens import HookedTransformer
    cfg = MathsConfig()
    cfg.set_model_names(MODELS["primary"])
    htc = cfg.get_HookedTransformerConfig()
    htc.device = "cpu"
    htc.init_weights = True
    htc.seed = 999
    model = HookedTransformer(htc)
    W_E = model.W_E.detach().cpu().numpy()[:10]
    W_U = model.W_U.detach().cpu().numpy().T[:10]
    return W_E, W_U


def run_models(n_perm: int = N_PERM) -> dict:
    results = {}
    for tag, name in MODELS.items():
        W_E, W_U = load_WE_WU(name)
        results[tag] = {
            "model_name": name,
            "embed": analyze_matrix(W_E, n_perm),
            "unembed": analyze_matrix(W_U, n_perm),
            "embed_unembed_principal_angles_deg": principal_angles(W_E, W_U),
        }
    W_E, W_U = untrained_control()
    results["untrained_control"] = {
        "model_name": "untrained_" + MODELS["primary"],
        "embed": analyze_matrix(W_E, n_perm),
        "unembed": analyze_matrix(W_U, n_perm),
        "embed_unembed_principal_angles_deg": principal_angles(W_E, W_U),
    }
    return results


def _slim(d):
    """Drop bulky coord arrays for the console summary."""
    if isinstance(d, dict):
        return {k: _slim(v) for k, v in d.items() if k != "pc_plane_coords"}
    return d


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("control", "all"):
        print("=== POSITIVE CONTROL [1/2] ===", flush=True)
        ctrl = run_positive_control()
        with open(os.path.join(RESULT_DIR, "positive_control.json"), "w") as f:
            json.dump(ctrl, f, indent=2)
        print(json.dumps(ctrl["calibration"], indent=2))
        for row in ctrl["detection"]:
            print(f"  {row['kind']:6s} planted={row['planted_share']:.2f} "
                  f"stat={row['detect_stat']:.3f} p={row['detect_p']:.4f} "
                  f"detect40={row['detected_at_40_p01']} -> {row['classified']}")
    if mode in ("models", "all"):
        print("=== REAL MODELS [2/2] ===", flush=True)
        res = run_models()
        with open(os.path.join(RESULT_DIR, "model_geometry.json"), "w") as f:
            json.dump(res, f, indent=2)
        for tag, r in res.items():
            e, u = r["embed"], r["unembed"]
            print(f"\n[{tag}] {r['model_name']}")
            print(f"  EMBED   read={e['classified_read']:16s} "
                  f"freq1={e['freq1_share']:.3f}(p={e['freq1_p']:.4f}) "
                  f"ulin={e['unique_linear_share']:.3f}(p={e['unique_linear_p']:.4f}) "
                  f"wrap={e['wraparound_ratio']:.2f} PR={e['participation_ratio']:.2f}")
            print(f"  UNEMBED read={u['classified_read']:16s} "
                  f"freq1={u['freq1_share']:.3f}(p={u['freq1_p']:.4f}) "
                  f"ulin={u['unique_linear_share']:.3f}(p={u['unique_linear_p']:.4f}) "
                  f"wrap={u['wraparound_ratio']:.2f} PR={u['participation_ratio']:.2f}")
            print(f"  embed/unembed principal angles (deg): "
                  f"{[round(a,1) for a in r['embed_unembed_principal_angles_deg']]}")
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
