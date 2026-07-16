"""Reusable linear-probe and geometry toolkit for maths models.

Consolidates the probe / null / subspace / DFT-geometry primitives that the
one-off study scripts (probe_transfer.py, node_output_encoding.py,
digit_embedding_geometry.py, st_tristate_geometry.py) each re-implemented. These
are the backbone of CE1 (embedding geometry), CE2 (linear operand transport), and
CE6/CE7 (no dedicated tri-state symbol; carry binary, resolved ~L1).

Scope note: this module owns the STUDY-AGNOSTIC primitives only. Each study keeps
its own decision/verdict logic (thresholds, classification tables) in its script;
those call into these primitives so the numerics are computed once, tested once,
and reused across all ~40 HF models.

Position algebra is delegated to MathsConfig (dn/ddn/an position helpers) rather
than re-derived inline.

CPU-friendly; numpy + scikit-learn only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

DIGITS = np.arange(10)


# ===========================================================================
# Per-digit sub-task labels (SA / ST / SV cascade)
# ===========================================================================

def sub_labels(a: int, b: int, n_digits: int) -> Tuple[dict, dict, dict]:
    """Per-digit SA, ST, SV labels for ``a + b``.

    Digits indexed 0 (units) .. n_digits-1 (top). Returns three dicts keyed by
    digit index:
      * SA[n] = (Dn + D'n) mod 10            -- base-sum digit
      * ST[n] = 0 (sum<=8) / 1 (sum>=10) / 2 (sum==9, the ambiguous 'U' class)
      * SV[n] = carry INTO digit n           -- 0 or 1
    """
    da = [int(d) for d in str(a).zfill(n_digits)]
    db = [int(d) for d in str(b).zfill(n_digits)]
    SA, ST, SV = {}, {}, {}
    carry = 0
    for n in range(n_digits):
        x = da[n_digits - 1 - n]
        y = db[n_digits - 1 - n]
        s = x + y
        SA[n] = s % 10
        ST[n] = 0 if s <= 8 else (1 if s >= 10 else 2)  # 2 == U (sum==9)
        SV[n] = carry
        carry = 1 if (s + carry) >= 10 else 0
    return SA, ST, SV


TASK_CHANCE = {"SA": 0.10, "ST": 1.0 / 3.0, "SV": 0.5}


# ===========================================================================
# Activation read sites (site -> (hook_name, token_pos)) via MathsConfig algebra
# ===========================================================================

ALL_SITES = ["Dpn_L0", "Dpn_L1", "Dn_L0", "ans_L0", "eq_L1"]


def _pos_index(position_name: str) -> int:
    """'P17' -> 17."""
    return int(position_name[1:])


def site_hook_and_pos(cfg, site: str, n: int) -> Tuple[str, int]:
    """Return ``(hook_name, token_pos)`` for a sub-task read site at digit ``n``.

    Positions come from the MathsConfig helpers so they stay correct across
    digit counts / operators:
      * Dpn_* -> operand-2 digit position (ddn_to_position_name)
      * Dn_L0 -> operand-1 digit position (dn_to_position_name)
      * ans_L0 -> the position that PRODUCES A_n (= pos(A_n) - 1)
      * eq_L1 -> the '=' token position, layer-1 residual
    """
    if site == "Dpn_L0":
        return "blocks.0.hook_resid_post", _pos_index(cfg.ddn_to_position_name(n))
    if site == "Dpn_L1":
        return "blocks.1.hook_resid_post", _pos_index(cfg.ddn_to_position_name(n))
    if site == "Dn_L0":
        return "blocks.0.hook_resid_post", _pos_index(cfg.dn_to_position_name(n))
    if site == "ans_L0":
        return "blocks.0.hook_resid_post", _pos_index(cfg.an_to_position_name(n)) - 1
    if site == "eq_L1":
        # '=' is the token right after operand-2 digit 0: pos = 2*n_digits + 1
        return "blocks.1.hook_resid_post", 2 * cfg.n_digits + 1
    raise ValueError(f"unknown site {site!r}")


def collect_site_activations(
    model, cfg, n_q: int, digits: Sequence[int], sites: Sequence[str],
    rng: np.random.Generator,
) -> Tuple[dict, dict]:
    """Gather residual activations at ``(site, digit)`` + SA/ST/SV labels.

    Returns ``(acts, labs)`` where ``acts[(site, n)]`` is ``[n_q, d_model]`` and
    ``labs[task][n]`` is ``[n_q]``.
    """
    import torch
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken

    nd = cfg.n_digits
    acts = {(s, n): [] for s in sites for n in digits}
    labs = {t: {n: [] for n in digits} for t in ("SA", "ST", "SV")}
    hooks_needed = sorted({site_hook_and_pos(cfg, s, digits[0])[0] for s in sites})
    lim = 10 ** nd
    for _ in range(n_q):
        a = int(rng.integers(0, lim // 2))
        b = int(rng.integers(0, lim // 2))
        q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(cfg, q, 0, a, b, MathsToken.PLUS)
        with torch.no_grad():
            _, c = model.run_with_cache(
                q, names_filter=lambda nm: nm in hooks_needed)
        SA, ST, SV = sub_labels(a, b, nd)
        for s in sites:
            for n in digits:
                hook, pos = site_hook_and_pos(cfg, s, n)
                acts[(s, n)].append(c[hook][0, pos, :].numpy())
        for n in digits:
            labs["SA"][n].append(SA[n])
            labs["ST"][n].append(ST[n])
            labs["SV"][n].append(SV[n])
    acts = {k: np.asarray(v) for k, v in acts.items()}
    for t in labs:
        labs[t] = {n: np.asarray(v) for n, v in labs[t].items()}
    return acts, labs


# ===========================================================================
# Probe fit / evaluate / balance / split
# ===========================================================================

def train_test_split_idx(n: int, rng: np.random.Generator, frac: float = 0.7):
    perm = rng.permutation(n)
    k = int(frac * n)
    return perm[:k], perm[k:]


def balance_idx(y: np.ndarray, rng: np.random.Generator, cap: int = None) -> np.ndarray:
    """Oversample minority classes to equal counts. Returns an index array."""
    y = np.asarray(y)
    classes = np.unique(y)
    per = int(max(np.bincount(y - y.min(), minlength=1)))
    if cap:
        per = min(per, cap)
    idx = []
    for cl in classes:
        ci = np.where(y == cl)[0]
        idx.extend(rng.choice(ci, size=per, replace=len(ci) < per))
    return np.asarray(idx)


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    """Fit a logistic-regression linear probe."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=C).fit(X, y)


def probe_balanced_accuracy(clf, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, clf.predict(X)))


def probe_accuracy_with_null(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    n_perm: int = 200, C: float = 1.0, frac: float = 0.7,
) -> dict:
    """Fit a balanced probe and compute a label-permutation null.

    Returns observed balanced accuracy, the null mean, and an empirical p-value
    ``P(null >= observed)``. A probe "decodes" a sub-task when acc clears both
    chance and the null.
    """
    tr, te = train_test_split_idx(len(y), rng, frac)
    bi = balance_idx(y[tr], rng)
    clf = fit_probe(X[tr][bi], y[tr][bi], C=C)
    obs = probe_balanced_accuracy(clf, X[te], y[te])

    ge = 0
    null_accs = []
    for _ in range(n_perm):
        yp = rng.permutation(y[tr])
        bi = balance_idx(yp, rng)
        clf = fit_probe(X[tr][bi], yp[bi], C=C)
        acc = probe_balanced_accuracy(clf, X[te], y[te])
        null_accs.append(acc)
        if acc >= obs:
            ge += 1
    return {
        "observed_acc": obs,
        "null_mean_acc": float(np.mean(null_accs)) if null_accs else float("nan"),
        "null_p": (ge + 1) / (n_perm + 1),
    }


# ===========================================================================
# Subspace geometry (class-mean subspace + principal angles)
# ===========================================================================

def class_mean_subspace(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Orthonormal basis (rows) of the span of class-conditional mean deviations."""
    classes = np.unique(y)
    mu = X.mean(0)
    M = np.stack([X[y == cl].mean(0) - mu for cl in classes])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = int((S > 1e-6).sum())
    return Vt[:k]


def principal_angles_deg(B1: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between two row-orthonormal subspaces."""
    from scipy.linalg import subspace_angles
    if B1.size == 0 or B2.size == 0:
        return np.array([])
    ang = subspace_angles(B1.T, B2.T)
    return np.degrees(ang)


# ===========================================================================
# DFT / circular-geometry toolkit (CE1 embedding geometry)
# ===========================================================================

def real_dft_basis() -> Tuple[np.ndarray, List[str]]:
    """Orthonormal real DFT basis over 10 points, excluding the constant."""
    cols, names = [], []
    d = DIGITS
    for k in range(1, 5):
        cols.append(np.cos(2 * np.pi * k * d / 10)); names.append(f"cos{k}")
        cols.append(np.sin(2 * np.pi * k * d / 10)); names.append(f"sin{k}")
    cols.append(np.cos(2 * np.pi * 5 * d / 10)); names.append("cos5")
    B = np.stack(cols, axis=1).astype(float)
    B = B - B.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(B)
    return Q, names


_DFT_Q, _DFT_NAMES = real_dft_basis()


def _centered(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=0, keepdims=True)


def marginal_dft_spectrum(M: np.ndarray) -> dict:
    """Variance share captured by each DFT frequency component."""
    Mc = _centered(M)
    tv = (Mc ** 2).sum()
    freq_of: Dict[int, list] = {}
    for i, nm in enumerate(_DFT_NAMES):
        freq_of.setdefault(int(nm[-1]), []).append(i)
    shares = {}
    for k, idxs in freq_of.items():
        proj = _DFT_Q[:, idxs] @ (_DFT_Q[:, idxs].T @ Mc)
        shares[f"k{k}"] = float((proj ** 2).sum() / tv) if tv > 0 else 0.0
    return shares


def freq1_plane_share(M: np.ndarray) -> float:
    """Variance share in the frequency-1 (cos1, sin1) plane."""
    Mc = _centered(M)
    tv = (Mc ** 2).sum()
    if tv == 0:
        return 0.0
    idx = [_DFT_NAMES.index("cos1"), _DFT_NAMES.index("sin1")]
    P = _DFT_Q[:, idx]
    proj = P @ (P.T @ Mc)
    return float((proj ** 2).sum() / tv)


def unique_linear_share(M: np.ndarray) -> float:
    """Variance share of the linear ramp orthogonalized against the freq-1 plane."""
    Mc = _centered(M)
    tv = (Mc ** 2).sum()
    if tv == 0:
        return 0.0
    lin = (DIGITS - DIGITS.mean()).astype(float)
    idx = [_DFT_NAMES.index("cos1"), _DFT_NAMES.index("sin1")]
    P = _DFT_Q[:, idx]
    lin_res = lin - P @ (P.T @ lin)
    nrm = np.linalg.norm(lin_res)
    if nrm < 1e-12:
        return 0.0
    u = (lin_res / nrm).reshape(-1, 1)
    proj = u @ (u.T @ Mc)
    return float((proj ** 2).sum() / tv)


def pc_plane_coords(M: np.ndarray) -> np.ndarray:
    """Top-2 principal-component coordinates of the 10 rows."""
    Mc = _centered(M)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    return U[:, :2] * S[:2]


def angular_order_stat(coords2d: np.ndarray) -> float:
    """Sum of squared angular gaps between value-consecutive digits (low=circular)."""
    ang = np.arctan2(coords2d[:, 1], coords2d[:, 0])
    order = np.argsort(DIGITS)
    a = ang[order]
    gaps = np.diff(np.concatenate([a, a[:1]]))
    gaps = (gaps + np.pi) % (2 * np.pi) - np.pi
    return float((gaps ** 2).sum())


def wraparound_ratio(coords2d: np.ndarray) -> float:
    """distance(9,0) / mean adjacent-by-value distance. ~1 if 9 sits next to 0."""
    d = coords2d
    adj = [np.linalg.norm(d[i + 1] - d[i]) for i in range(9)]
    d90 = np.linalg.norm(d[0] - d[9])
    return float(d90 / (np.mean(adj) + 1e-12))


def participation_ratio(M: np.ndarray) -> float:
    """Effective dimensionality of the centered point cloud."""
    Mc = _centered(M)
    _, S, _ = np.linalg.svd(Mc, full_matrices=False)
    var = S ** 2
    return float((var.sum() ** 2) / (var ** 2).sum()) if var.sum() > 0 else 0.0


def dft_permutation_pvalues(M: np.ndarray, rng: np.random.Generator,
                            n_perm: int = 2000) -> dict:
    """Label-permutation p-values for freq1-share, unique-linear-share, and
    angular-order (which digit is which is permuted; the cloud shape is invariant).
    """
    obs_f1 = freq1_plane_share(M)
    obs_lin = unique_linear_share(M)
    obs_ord = angular_order_stat(pc_plane_coords(M))
    ge_f1 = ge_lin = le_ord = 0
    for _ in range(n_perm):
        Mp = M[rng.permutation(10)]
        if freq1_plane_share(Mp) >= obs_f1:
            ge_f1 += 1
        if unique_linear_share(Mp) >= obs_lin:
            ge_lin += 1
        if angular_order_stat(pc_plane_coords(Mp)) <= obs_ord:
            le_ord += 1
    return {
        "freq1_share": obs_f1,
        "freq1_p": (ge_f1 + 1) / (n_perm + 1),
        "unique_linear_share": obs_lin,
        "unique_linear_p": (ge_lin + 1) / (n_perm + 1),
        "angular_order_stat": obs_ord,
        "angular_order_p": (le_ord + 1) / (n_perm + 1),
    }
