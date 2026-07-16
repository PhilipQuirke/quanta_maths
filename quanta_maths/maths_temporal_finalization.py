"""Temporal finalization of the carry (Battery TF) — when, in token-time, does each
carry SVn become a canonical resolved bit?

Companion to the *layer*-localization result (the canonical resolved carry emerges
in the final-layer "read", not at the early-layer output). This module adds the
TOKEN-TIME dimension: for a graded carry chain it maps, per (token position, layer),
when the CANONICAL propagated carry becomes present, and contrasts that with the
EAGER, in-place local single-step make-carry.

Two measurements, both mindful of *decode != computation* (a resolved carry is a
deterministic function of the visible input digits, so raw decodability at a
position merely tracks input-visibility, not computation):

  * PROPAGATED carry (the eager-vs-lazy test) — guaranteed canonical by
    CROSS-DECIDING-POSITION transfer (train a decoder with the make-carry at digit
    d_i, test with it at d_j): a reader of a fixed local digit cannot transfer;
    only a position-invariant canonical carry does. Decorrelated from the local
    make-carry by a RUN-BREAK (the 9/U-run above the deciding digit is sometimes
    broken, so ``carry_out(top) = make_carry AND run_intact``).
  * LOCAL make-carry (eager reference) — a within-position decode of the deciding
    digit's own make-carry class. Not transfer-controlled; it is a reference to the
    established "single-step resolution is eager, in-place at the early layer".

Stimuli are "9-free" (the U-state comes from e.g. 6+3, make-carries from e.g. 6+4)
so no literal ``9`` token can leak into the carry decode.

Interpretation is left to the caller via the returned (position x layer) transfer
maps and onsets; a ``verdict`` summary is provided. **Answers may differ by model**
— run across the model zoo and compare.

Layer-general (scans ``model.cfg.n_layers``) so it works on 2-, 3-, 4-layer models.
CPU-friendly. Linear-probe; coarse for small models (short token tail).

Example::

    from quanta_maths.maths_temporal_finalization import run_temporal_finalization
    res = run_temporal_finalization("add_d6_l2_h3_t20K_s173289")
    print(res["verdict"]["read"])
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from quanta_maths.maths_model_loader import load_maths_model_from_hf

_RNG = np.random.default_rng(20260716)


# ===========================================================================
# position algebra (cfg-derived; matches the VerifiedArithmetic token layout)
# ===========================================================================

def dprime_pos(cfg, n: int) -> int:
    """Token position of the second-operand digit D'_n."""
    return int(cfg.ddn_to_position_name(n)[1:])


def consuming_pos(cfg, k: int) -> int:
    """Position whose logits produce answer digit A_k (= pos(A_k) - 1)."""
    return int(cfg.an_to_position_name(k)[1:]) - 1


def equals_pos(cfg) -> int:
    """The ``=`` token position."""
    return 2 * cfg.n_digits + 1


def full_input_pos(cfg) -> int:
    """Position at which ALL input digits are visible = D'_0 (units of operand B).
    carry_out(top) is not determinable before this (lower fillers arrive here)."""
    return dprime_pos(cfg, 0)


# ===========================================================================
# 9-free graded-carry stimulus with an optional run-break
# ===========================================================================

def _pair_sum9(nine_free: bool) -> Tuple[int, int]:
    a = int(_RNG.integers(1, 9)) if nine_free else int(_RNG.integers(0, 10))
    return a, 9 - a


def _pair_ge10(nine_free: bool) -> Tuple[int, int]:
    lo = 2 if nine_free else 0
    while True:
        a = int(_RNG.integers(lo, 9 if nine_free else 10))
        b = int(_RNG.integers(lo, 9 if nine_free else 10))
        if a + b >= 10:
            return a, b


def _pair_le8() -> Tuple[int, int]:
    while True:
        a = int(_RNG.integers(0, 9))
        b = int(_RNG.integers(0, 9))
        if a + b <= 8:
            return a, b


def build_carry_chain(
    cfg,
    deciding_digit: int,
    make_carry: bool,
    run_intact: bool,
    nine_free: bool = True,
) -> Tuple[int, int, int, int]:
    """Build one graded carry chain.

    - deciding digit d: a make-carry pair (sum>=10) if ``make_carry`` else no-carry.
    - digits d+1..n_top: a U-run (pair-sum 9), except one digit BROKEN to a
      no-carry pair when ``run_intact`` is False (blocks propagation).
    - digits < d and digit n_top+1: forced no-carry.

    Returns ``(a, b, carry_out_top, local_make_carry)`` where
    ``carry_out_top = make_carry AND run_intact`` and ``local_make_carry = make_carry``
    (the two are decorrelated by the run-break).
    """
    nd = cfg.n_digits
    n_top = nd - 2
    d1 = [0] * nd
    d2 = [0] * nd

    def setd(n: int, a: int, b: int) -> None:
        d1[nd - 1 - n] = a
        d2[nd - 1 - n] = b

    da, db = _pair_ge10(nine_free) if make_carry else _pair_le8()
    setd(deciding_digit, da, db)

    run = list(range(deciding_digit + 1, n_top + 1))
    brk = int(run[int(_RNG.integers(0, len(run)))]) if (run and not run_intact) else None
    for j in run:
        a, b = _pair_le8() if j == brk else _pair_sum9(nine_free)
        setd(j, a, b)

    if n_top + 1 < nd:
        while True:
            a, b = _pair_le8()
            if a + b != 9:
                break
        setd(n_top + 1, a, b)

    for j in range(0, deciding_digit):
        a, b = _pair_le8()
        setd(j, a, b)

    a_int = int("".join(map(str, d1)))
    b_int = int("".join(map(str, d2)))
    carry_top = _carry_out_top(d1, d2, n_top)
    return a_int, b_int, carry_top, (1 if make_carry else 0)


def _carry_out_top(d1: List[int], d2: List[int], n_top: int) -> int:
    """carry_out of the chain-top digit n_top (does a carry ripple out of digit n_top)."""
    nd = len(d1)
    carry = 0
    for k in range(nd - 1, nd - 1 - (n_top + 1), -1):
        s = d1[k] + d2[k] + carry
        carry = 1 if s >= 10 else 0
    return carry


# ===========================================================================
# question encoding (delegate to cfg so we match the model's tokenization)
# ===========================================================================

def _make_question(cfg, a: int, b: int) -> torch.Tensor:
    from quanta_maths.maths_run import make_question
    return make_question(cfg, a, b)


# ===========================================================================
# decoding helpers
# ===========================================================================

def _fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1500, C=0.5).fit(X, y)


def _transfer(Xtr, ytr, Xte, yte) -> float:
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(yte, _fit(Xtr, ytr).predict(Xte)))


def _within(X, y) -> float:
    idx = np.arange(len(y))
    _RNG.shuffle(idx)
    h = len(y) // 2
    return _transfer(X[idx[:h]], y[idx[:h]], X[idx[h:]], y[idx[h:]])


# ===========================================================================
# core map
# ===========================================================================

def temporal_finalization_map(model, cfg, n_q: int = 600) -> Dict:
    """Map, per (token position, layer), the canonical propagated carry (via
    cross-deciding-position transfer) and the local make-carry (within-position).

    Returns a dict with ``propagated_transfer[layer][pos]``, ``local_transfer``,
    onsets, and the reference positions. Layer-general.
    """
    nd = cfg.n_digits
    n_top = nd - 2
    n_layers = int(model.cfg.n_layers)
    # two earliest deciding positions that still carry a U-run above them
    decis = [n_top - 2, n_top - 1]
    mc_tokens = {d: dprime_pos(cfg, d) for d in decis}
    cons = consuming_pos(cfg, n_top + 1)
    scan = list(range(min(mc_tokens.values()), cons + 1))
    resid_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]

    def collect(d: int) -> Dict:
        acts = {L: {p: [] for p in scan} for L in range(n_layers)}
        yc: List[int] = []
        yl: List[int] = []
        for _ in range(n_q):
            mk = _RNG.random() < 0.5
            ri = _RNG.random() < 0.5
            a, b, ct, lmc = build_carry_chain(cfg, d, mk, ri)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    _make_question(cfg, a, b).unsqueeze(0),
                    names_filter=lambda nm: nm in resid_names,
                )
            for L in range(n_layers):
                rp = cache[f"blocks.{L}.hook_resid_post"][0]
                for p in scan:
                    acts[L][p].append(rp[p, :].numpy())
            yc.append(ct)
            yl.append(lmc)
        return {"acts": {L: {p: np.array(v) for p, v in acts[L].items()} for L in range(n_layers)},
                "yc": np.array(yc), "yl": np.array(yl)}

    data = {d: collect(d) for d in decis}
    d_lo, d_hi = decis

    prop: Dict[int, Dict[int, float]] = {}
    for L in range(n_layers):
        prop[L] = {}
        for p in scan:
            v1 = _transfer(data[d_lo]["acts"][L][p], data[d_lo]["yc"],
                           data[d_hi]["acts"][L][p], data[d_hi]["yc"])
            v2 = _transfer(data[d_hi]["acts"][L][p], data[d_hi]["yc"],
                           data[d_lo]["acts"][L][p], data[d_lo]["yc"])
            prop[L][p] = float(np.nanmean([v1, v2]))

    local: Dict[int, Dict[int, Dict[int, float]]] = {}
    for d in decis:
        local[d] = {}
        for L in range(n_layers):
            local[d][L] = {p: _within(data[d]["acts"][L][p], data[d]["yl"]) for p in scan}

    def onset(row: Dict[int, float], thr: float = 0.9) -> Optional[int]:
        for i, p in enumerate(scan):
            if row[p] == row[p] and row[p] >= thr:
                nxt = scan[i + 1] if i + 1 < len(scan) else None
                if nxt is None or (row[nxt] == row[nxt] and row[nxt] >= thr - 0.05):
                    return p
        return None

    prop_onset = {L: onset(prop[L]) for L in range(n_layers)}
    return {
        "n_layers": n_layers,
        "deciding_positions": decis,
        "make_carry_tokens": mc_tokens,
        "full_input_pos": full_input_pos(cfg),
        "equals_pos": equals_pos(cfg),
        "consuming_pos_top": cons,
        "scan_positions": scan,
        "propagated_transfer": {L: {int(p): prop[L][p] for p in scan} for L in range(n_layers)},
        "propagated_onset": prop_onset,
        "local_transfer": {int(d): {L: {int(p): local[d][L][p] for p in scan}
                                    for L in range(n_layers)} for d in decis},
        "n_q": n_q,
    }


def summarize(tf: Dict) -> Dict:
    """Derive the eager-vs-lazy verdict from a temporal_finalization_map result."""
    fip = tf["full_input_pos"]
    eq = tf["equals_pos"]
    decis = tf["deciding_positions"]
    mc = {int(k): v for k, v in tf["make_carry_tokens"].items()}
    # earliest layer with a sustained canonical onset
    onsets = {int(L): p for L, p in tf["propagated_onset"].items()}
    onset_layer = None
    onset_pos = None
    for L in sorted(onsets):
        if onsets[L] is not None:
            onset_layer, onset_pos = L, onsets[L]
            break
    # local make-carry decode at each deciding digit's own token, early layer 0
    lt = tf["local_transfer"]
    local_at_token = {}
    for d in decis:
        ld = lt.get(str(d), lt.get(d))
        l0 = ld.get("0", ld.get(0))
        tok = mc[d]
        local_at_token[d] = l0.get(tok, l0.get(str(tok), float("nan")))
    v = {
        "propagated_onset_pos": onset_pos,
        "propagated_onset_layer": onset_layer,
        "full_input_pos": fip,
        "equals_pos": eq,
        "genuine_deferral_tokens": (onset_pos - fip) if onset_pos is not None else None,
        "local_makecarry_decode_at_token_layer0": {int(d): local_at_token[d] for d in decis},
        "local_eager": bool(np.nanmax(list(local_at_token.values())) >= 0.7),
    }
    if onset_pos is None:
        v["read"] = "propagated canonical carry not found in scan (check power / thresholds)"
    elif onset_pos <= fip:
        v["read"] = (f"EAGER: propagated carry representationally present at layer {onset_layer} "
                     f"as soon as inputs are available (onset {onset_pos} <= full-input {fip}).")
    else:
        defer = onset_pos - fip
        v["read"] = (
            f"LAZY (representational): the canonical propagated carry is deferred to layer "
            f"{onset_layer} at the answer region (onset pos {onset_pos}) — {defer} token(s) past "
            f"full input availability (D'_0 pos {fip}){' and past the `=` gather' if onset_pos > eq else ''}; "
            f"the local single-step make-carry is present in-place at layer 0 at its own token "
            f"(max {np.nanmax(list(local_at_token.values())):.2f}). Linear-probe; interpret per model."
        )
    return v


def run_temporal_finalization(
    model_name: str,
    n_q: int = 600,
    device: str = "cpu",
) -> Dict:
    """Load ``model_name`` and run the temporal-finalization map + verdict.

    Returns a JSON-serializable dict. Run across models and compare — the
    eager-vs-lazy answer may differ by model/size.
    """
    model, cfg = load_maths_model_from_hf(model_name, device=device)
    tf = temporal_finalization_map(model, cfg, n_q=n_q)
    tf["model"] = model_name
    tf["n_digits"] = cfg.n_digits
    tf["verdict"] = summarize(tf)
    return tf
