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

import contextlib
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

DIGITS = np.arange(10)


# ===========================================================================
# Per-digit sub-task labels (SA / ST / SV cascade)
# ===========================================================================

def sub_labels(a: int, b: int, n_digits: int, operation=None) -> Tuple[dict, dict, dict]:
    """Per-digit SA, ST, SV labels for ``a + b`` (addition) or ``a - b`` (subtraction).

    Digits indexed 0 (units) .. n_digits-1 (top). Returns three dicts keyed by
    digit index. The addition and subtraction cases are structural parallels:

    Addition (carry cascade):
      * SA[n] = (Dn + D'n) mod 10            -- base-sum digit
      * ST[n] = 0 (sum<=8) / 1 (sum>=10) / 2 (sum==9, the ambiguous 'U' class)
      * SV[n] = carry INTO digit n           -- 0 or 1

    Subtraction (borrow cascade; parallel roles):
      * SA[n] = (Dn - D'n) mod 10            -- base-difference digit
      * ST[n] = 1 (Dn<D'n, will borrow) / 0 (Dn>D'n, no borrow) / 2 (Dn==D'n, the
                ambiguous 'U' class where the borrow depends on the lower digit)
      * SV[n] = borrow INTO digit n          -- 0 or 1
    """
    from quanta_maths.maths_constants import MathsToken
    if operation is None:
        operation = MathsToken.PLUS
    da = [int(d) for d in str(a).zfill(n_digits)]
    db = [int(d) for d in str(b).zfill(n_digits)]
    SA, ST, SV = {}, {}, {}

    if operation == MathsToken.MINUS:
        borrow = 0
        for n in range(n_digits):
            x = da[n_digits - 1 - n]
            y = db[n_digits - 1 - n]
            diff = x - y
            SA[n] = (x - y) % 10
            ST[n] = 1 if diff < 0 else (0 if diff > 0 else 2)  # 2 == U (x==y)
            SV[n] = borrow
            borrow = 1 if (x - y - borrow) < 0 else 0
        return SA, ST, SV

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


def neg_labels(a: int, b: int, n_digits: int) -> Tuple[dict, dict, dict]:
    """Per-digit NEGATIVE-answer subtraction labels for ``a - b`` with ``a < b``.

    NEG questions (``D < D'``) have answer ``-(D' - D)``; the model emits the
    digits of the magnitude ``D' - D``. This is the third task family (ND/NB/NV,
    the parallel of the addition SV cascade and the positive-answer MV cascade):

      * SA[n] (== ND) = nth digit of ``(D' - D)``          -- emitted answer digit
      * ST[n] (== NT) = 1 (D'n<Dn, will borrow) / 0 (D'n>Dn) / 2 (D'n==Dn, U)
      * SV[n] (== NV) = neg-borrow INTO digit n            -- 0 or 1

    Implemented as the positive-answer borrow cascade on the SWAPPED operands
    (``D' - D``), which is exactly what makes the emitted digits come out right;
    the (SA, ST, SV) dict signature is kept so the probe collectors stay
    class-agnostic. Verified against the model's emitted digits in
    ``tests/test_scaling_and_sub.py``.

    Raises if ``a >= b`` (not a negative-answer question).
    """
    if a >= b:
        raise ValueError(f"neg_labels requires a < b (D < D'); got {a} >= {b}")
    from quanta_maths.maths_constants import MathsToken
    return sub_labels(b, a, n_digits, operation=MathsToken.MINUS)


TASK_CHANCE = {"SA": 0.10, "ST": 1.0 / 3.0, "SV": 0.5}


# ===========================================================================
# Activation read sites (site -> (hook_name, token_pos)) via MathsConfig algebra
# ===========================================================================

ALL_SITES = ["Dpn_L0", "Dpn_L1", "Dn_L0", "ans_L0", "eq_L1"]


def _pos_index(position_name: str) -> int:
    """'P17' -> 17."""
    return int(position_name[1:])


# ---------------------------------------------------------------------------
# Semantic layer helpers.
#
# IMPORTANT (multi-layer models): the ``_L0`` / ``_L1`` suffix in the legacy site
# names is a LITERAL layer index, tuned for the 2-layer models where L0 == "early
# / operand-fetch" and L1 == "late / combiner". On 3- and 4-layer models
# (mix_*_l3, ins2_*_l4, ...) those literal indices are NOT the combiner: the
# combiner is the LAST layer. Callers that mean "read where the answer is
# combined" must pass the last layer explicitly (site suffix does not track depth).
# ---------------------------------------------------------------------------

def first_layer(cfg) -> int:
    """The earliest transformer layer (operand-fetch / low-level features)."""
    return 0


def last_layer(cfg) -> int:
    """The final transformer layer (where the answer is combined)."""
    return cfg.n_layers - 1


def _site_role_pos(cfg, role: str, n: int) -> int:
    """Token position for a spatial role (independent of layer)."""
    if role == "Dpn":
        return _pos_index(cfg.ddn_to_position_name(n))
    if role == "Dn":
        return _pos_index(cfg.dn_to_position_name(n))
    if role == "ans":
        return _pos_index(cfg.an_to_position_name(n)) - 1  # position that PRODUCES A_n
    if role == "eq":
        return 2 * cfg.n_digits + 1  # '=' token
    raise ValueError(f"unknown site role {role!r}")


def site_hook_and_pos(cfg, site: str, n: int, layer: int = None) -> Tuple[str, int]:
    """Return ``(hook_name, token_pos)`` for a sub-task read site at digit ``n``.

    Token positions come from the MathsConfig helpers so they are correct across
    digit counts / operators. Site spatial roles:
      * ``Dpn`` -> operand-2 digit position (ddn_to_position_name)
      * ``Dn``  -> operand-1 digit position (dn_to_position_name)
      * ``ans`` -> the position that PRODUCES A_n (= pos(A_n) - 1)
      * ``eq``  -> the '=' token position

    Layer resolution:
      * If ``layer`` is given, it overrides everything and is bounds-checked
        against ``cfg.n_layers`` (raises on out-of-range). PREFER this on
        multi-layer models.
      * Otherwise the legacy ``_L{k}`` suffix is used as a literal layer index and
        is likewise bounds-checked (so ``Dpn_L1`` on a 1-layer model raises rather
        than silently reading a wrong layer).
    """
    # Split "Dpn_L1" -> role "Dpn", suffix layer 1.
    if "_L" in site:
        role, suffix = site.split("_L")
        suffix_layer = int(suffix)
    else:
        role, suffix_layer = site, None

    resolved_layer = layer if layer is not None else suffix_layer
    if resolved_layer is None:
        raise ValueError(f"site {site!r} has no layer suffix; pass layer=")
    if not (0 <= resolved_layer < cfg.n_layers):
        raise ValueError(
            f"layer {resolved_layer} out of range for a {cfg.n_layers}-layer model "
            f"(site {site!r}). On deeper models pass an explicit valid layer= "
            f"(e.g. last_layer(cfg)={cfg.n_layers - 1}).")

    pos = _site_role_pos(cfg, role, n)
    return f"blocks.{resolved_layer}.hook_resid_post", pos


def collect_site_activations(
    model, cfg, n_q: int, digits: Sequence[int], sites: Sequence[str],
    rng: np.random.Generator, operation=None, layer: int = None, cls: str = None,
) -> Tuple[dict, dict]:
    """Gather residual activations at ``(site, digit)`` + SA/ST/SV labels.

    Returns ``(acts, labs)`` where ``acts[(site, n)]`` is ``[n_q, d_model]`` and
    ``labs[task][n]`` is ``[n_q]``.

    ``operation`` selects the token (defaults to PLUS). ``cls`` in
    {"ADD","SUB","NEG"} selects the question CLASS and the correct labels — needed
    on MIXED models: "SUB" forces ``D>=D'`` with positive-answer borrow labels,
    "NEG" forces ``D<D'`` with negative-answer labels (``neg_labels``; the base
    difference is on ``D'-D``). If ``cls`` is None the legacy behaviour is kept
    (random operands, ``sub_labels``) — which mislabels the NEG subset of a MINUS
    run, so pass ``cls`` explicitly for mixed/subtraction models. ``layer``
    overrides the site's ``_L`` suffix (use ``last_layer(cfg)`` on deep models).
    """
    import torch
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken

    if cls is not None:
        operation = MathsToken.PLUS if cls == "ADD" else MathsToken.MINUS
    if operation is None:
        operation = MathsToken.PLUS
    nd = cfg.n_digits
    lim = 10 ** nd
    acts = {(s, n): [] for s in sites for n in digits}
    labs = {t: {n: [] for n in digits} for t in ("SA", "ST", "SV")}
    hooks_needed = sorted({site_hook_and_pos(cfg, s, digits[0], layer=layer)[0] for s in sites})

    def draw():
        if cls == "ADD" or cls is None:
            hi = lim // 2 if (cls == "ADD" or operation == MathsToken.PLUS) else lim
            return int(rng.integers(0, hi)), int(rng.integers(0, hi))
        a, b = int(rng.integers(0, lim)), int(rng.integers(0, lim))
        if cls == "SUB":
            if a < b:
                a, b = b, a
            if a == b:
                a = (a + 1) % lim
                if a < b:
                    a, b = b, a
        else:  # NEG
            if a == b:
                b = (b + 1) % lim
            if a > b:
                a, b = b, a
        return a, b

    for _ in range(n_q):
        a, b = draw()
        q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(cfg, q, 0, a, b, operation)
        with torch.no_grad():
            _, c = model.run_with_cache(
                q, names_filter=lambda nm: nm in hooks_needed)
        if cls == "NEG":
            SA, ST, SV = neg_labels(a, b, nd)
        else:
            SA, ST, SV = sub_labels(a, b, nd, operation=operation)
        for s in sites:
            for n in digits:
                hook, pos = site_hook_and_pos(cfg, s, n, layer=layer)
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


@contextlib.contextmanager
def _quiet_convergence():
    """Silence only sklearn's ConvergenceWarning.

    Probes are consumed as an accuracy metric with a permutation null; lbfgs not
    fully converging on high-dim unscaled activations (common on the permutation-
    shuffled null fits) only makes the accuracy conservative. Suppressing this one
    warning keeps batch logs clean WITHOUT changing any numerics (max_iter, solver
    and objective are unchanged, so fitted coefficients are identical). Scoped so
    other warnings still surface.
    """
    try:
        from sklearn.exceptions import ConvergenceWarning
    except Exception:  # pragma: no cover - sklearn always present here
        ConvergenceWarning = Warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        yield


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    """Fit a logistic-regression linear probe."""
    from sklearn.linear_model import LogisticRegression
    with _quiet_convergence():
        return LogisticRegression(max_iter=2000, C=C).fit(X, y)


def probe_balanced_accuracy(clf, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, clf.predict(X)))


def cross_val_probe_accuracy(X: np.ndarray, y: np.ndarray, folds: int = 5,
                             C: float = 1.0, balanced: bool = False) -> float:
    """K-fold cross-validated linear-probe accuracy.

    Consolidates the ``cross_val_score(LogisticRegression(...))`` pattern used by
    several studies. ``balanced=True`` uses balanced accuracy (recommended for
    class-imbalanced labels); default matches sklearn's plain accuracy scorer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=2000, C=C)
    scorer = "balanced_accuracy" if balanced else None
    with _quiet_convergence():
        return float(cross_val_score(clf, X, y, cv=folds, scoring=scorer).mean())


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
