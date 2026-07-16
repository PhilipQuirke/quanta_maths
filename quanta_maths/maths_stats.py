"""Small statistical helpers for reporting causal-intervention results.

Promoted from the CE16 SV-implementation study (scripts/sv_implementation.py),
where every headline flip rate was reported with a Wilson confidence interval.
Every causal battery reports rates over a finite number of paired trials, so a
single shared CI implementation keeps reporting consistent and testable.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion ``k / n``.

    More accurate than the normal approximation for small ``n`` and rates near
    0 or 1 (exactly the regime of flip-rate measurements). Returns ``(lo, hi)``;
    ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def mean_ci(vals: Sequence[float], z: float = 1.96) -> dict:
    """Mean rate + Wilson CI + n, for a sequence of 0/1 (or [0,1]) trial outcomes.

    Returns ``{"rate": mean, "ci": [lo, hi], "n": n}``. The Wilson interval treats
    the rounded sum as the binomial success count (the values are per-trial flip
    indicators in the study's usage).
    """
    a = np.asarray(vals, dtype=float)
    n = len(a)
    if n == 0:
        return {"rate": float("nan"), "ci": [float("nan"), float("nan")], "n": 0}
    rate = float(a.mean())
    lo, hi = wilson_ci(int(round(a.sum())), n, z=z)
    return {"rate": rate, "ci": [lo, hi], "n": n}
