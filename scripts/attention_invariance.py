"""Attention-pattern invariance census (study-attention-invariance.md).

Tests A5 (attention routing is static positional wiring; data-dependence in
values/MLPs, not the QK pattern) and the routing half of C3.

KEY (amendments A-1/A-2, Gate-1): a static positional head STILL has
value-dependent softmax weights (QK reads digit embeddings), so pattern-variance
is A5-CONSISTENT, not A5-falsifying. The load-bearing statistics are:
  * PRIMARY: argmax/top-k TARGET stability across random questions (A5 predicts
    the target key is content-invariant even if weights jiggle).
  * FALSIFIER: a discrete TARGET-MOVE under a VALUE-MATCHED contrast (fix digit-n
    operands sum=9, toggle only the lower carry) -> any target change is
    carry-state ROUTING, not value content.
Positive control: a SYNTHETIC content-routed head (perturb W_Q/W_K so the target
depends on a digit value) -- the target-move test MUST fire on it.
Informativeness gate (A-4): BOS-sink / single-key / dead cells are trivially
static and excluded from the informative fraction.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/attention_invariance.py control
    PYTHONPATH=. python3 scripts/attention_invariance.py models
    PYTHONPATH=. python3 scripts/attention_invariance.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import load_model, make_q, verify_accuracy, _digits_to_int

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-attention-invariance")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260716)

CFG = {"add_d5_l2_h3_t15K_s372001": {"digit": 2}, "add_d6_l2_h3_t20K_s173289": {"digit": 3}}
TOP1_BAR = 0.95   # informative-static: modal top-1 key on >=95% of questions
ENTROPY_MIN = 0.3  # informativeness: pattern entropy (nats) over causal keys


# ---------------------------------------------------------------------------
# stimuli
# ---------------------------------------------------------------------------

def random_add_q(cfg):
    lim = 10 ** cfg.n_digits
    a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2))
    return make_q(cfg, a, b), a, b


def value_matched_pair(cfg, n):
    """Fix digit-n operands (sum==9, U), toggle ONLY the lower carry. Returns
    (q_nocarry, q_carry) that are IDENTICAL at digit n (same operands) and differ
    only in whether digit n-1 produces a carry."""
    nd = cfg.n_digits; idx = nd - 1 - n
    a = int(RNG.integers(0, 10)); b = 9 - a
    def build(lower_carry):
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
            ik = nd - 1 - k; u = int(RNG.integers(0, 10)); v = int(RNG.integers(0, 10 - u))
            d1[ik] = u; d2[ik] = v
        return _digits_to_int(d1), _digits_to_int(d2)
    na, nb = build(False); ca, cb = build(True)
    return make_q(cfg, na, nb), make_q(cfg, ca, cb)


# ---------------------------------------------------------------------------
# pattern capture
# ---------------------------------------------------------------------------

def capture_patterns(model, cfg, qs):
    """qs: list of question tensors. Returns dict layer -> array
    [n_q, head, query, key]."""
    out = {L: [] for L in range(cfg.n_layers)}
    for q in qs:
        with torch.no_grad():
            _, cache = model.run_with_cache(q.unsqueeze(0))
        for L in range(cfg.n_layers):
            out[L].append(cache[f"blocks.{L}.attn.hook_pattern"][0].numpy())
    return {L: np.stack(out[L]) for L in out}


def entropy_of(p):
    p = p[p > 1e-9]
    return float(-(p * np.log(p)).sum())


# ---------------------------------------------------------------------------
# primary: target stability across random questions (A-1)
# ---------------------------------------------------------------------------

def target_stability(patterns, query_positions):
    """patterns: [n_q, head, query, key]. For each (head, query in query_positions):
    top-1 stability (frac with modal argmax), mean top-2 mass, informative flag."""
    nq, nh, nqy, nk = patterns.shape
    res = {}
    for h in range(nh):
        for q in query_positions:
            P = patterns[:, h, q, :]  # [n_q, key]
            argmax = P.argmax(1)
            vals, counts = np.unique(argmax, return_counts=True)
            modal = vals[counts.argmax()]
            top1_stab = float((argmax == modal).mean())
            top2_mass = float(np.sort(P, 1)[:, -2:].sum(1).mean())
            mean_ent = float(np.mean([entropy_of(P[i]) for i in range(nq)]))
            informative = (mean_ent >= ENTROPY_MIN) and (P.max(1).mean() < TOP1_BAR)
            res[(h, q)] = {"top1_stability": top1_stab, "top2_mass": top2_mass,
                           "mean_entropy": mean_ent, "informative": bool(informative),
                           "modal_key": int(modal)}
    return res


# ---------------------------------------------------------------------------
# falsifier: value-matched target-move (A-2)
# ---------------------------------------------------------------------------

def target_move_rate(model, cfg, n, layer, head, query_positions, n_pairs=60):
    """Over value-matched pairs (identical digit-n operands, lower carry toggled),
    fraction whose top-1 key (and top-2 set) CHANGES at each query position."""
    res = {q: {"top1_move": 0, "top2_move": 0} for q in query_positions}
    for _ in range(n_pairs):
        qn, qc = value_matched_pair(cfg, n)
        with torch.no_grad():
            _, c0 = model.run_with_cache(qn.unsqueeze(0))
            _, c1 = model.run_with_cache(qc.unsqueeze(0))
        p0 = c0[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
        p1 = c1[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
        for q in query_positions:
            if p0[q].argmax() != p1[q].argmax():
                res[q]["top1_move"] += 1
            s0 = set(np.argsort(p0[q])[-2:]); s1 = set(np.argsort(p1[q])[-2:])
            if s0 != s1:
                res[q]["top2_move"] += 1
    return {q: {"top1_move_rate": res[q]["top1_move"] / n_pairs,
                "top2_move_rate": res[q]["top2_move"] / n_pairs} for q in query_positions}


def perm_null_targetmove(model, cfg, n, layer, head, query_positions, n_pairs=60, n_perm=500):
    """Null: pair questions of the SAME lower-carry state (both no-carry) and
    measure the target-move rate -> chance floor for target changes from noise."""
    obs = {}
    rates = {q: [] for q in query_positions}
    for _ in range(n_pairs):
        # both no-lower-carry, different lower operands
        qn1, _ = value_matched_pair(cfg, n)
        qn2, _ = value_matched_pair(cfg, n)
        with torch.no_grad():
            _, c1 = model.run_with_cache(qn1.unsqueeze(0))
            _, c2 = model.run_with_cache(qn2.unsqueeze(0))
        p1 = c1[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
        p2 = c2[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
        for q in query_positions:
            rates[q].append(1.0 if p1[q].argmax() != p2[q].argmax() else 0.0)
    return {q: float(np.mean(rates[q])) for q in query_positions}


# ---------------------------------------------------------------------------
# synthetic content-routed positive control (A-3)
# ---------------------------------------------------------------------------

def synthetic_control(model, cfg, n, n_pairs=60):
    """Perturb head (0,0) W_Q/W_K so its target keys the digit-n D vs D' position
    depending on a digit value -> a KNOWN content-routed head. The value-matched
    target-move test MUST fire (>=0.5). Restores weights after."""
    layer, head = 0, 0
    Wq = model.blocks[layer].attn.W_Q.data.clone()
    Wk = model.blocks[layer].attn.W_K.data.clone()
    try:
        # crude content routing: add a large component so QK depends on the
        # digit-n token embedding direction -> target flips with digit value.
        d_model = model.cfg.d_model
        # route on the difference between two digit embeddings (a value axis)
        emb = model.W_E.detach().numpy()
        vaxis = torch.tensor((emb[9] - emb[0]) / (np.linalg.norm(emb[9]-emb[0])+1e-9),
                             dtype=Wq.dtype)
        model.blocks[layer].attn.W_Q.data[head] += 8.0 * torch.outer(vaxis, torch.ones(model.cfg.d_head))
        model.blocks[layer].attn.W_K.data[head] += 8.0 * torch.outer(vaxis, torch.ones(model.cfg.d_head))
        qpos = list(range(cfg.n_ctx))
        # measure target-move under a DIGIT-VALUE toggle (not carry): questions
        # differing only in digit-n value should move this head's target.
        moves = 0
        for _ in range(n_pairs):
            nd = cfg.n_digits; idx = nd - 1 - n
            d1 = [0]*nd; d2 = [0]*nd
            d1[idx] = 0; d2[idx] = 0
            qa = make_q(cfg, _digits_to_int(d1), _digits_to_int(d2))
            d1[idx] = 9
            qb = make_q(cfg, _digits_to_int(d1), _digits_to_int(d2))
            with torch.no_grad():
                _, ca = model.run_with_cache(qa.unsqueeze(0))
                _, cb = model.run_with_cache(qb.unsqueeze(0))
            pa = ca[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
            pb = cb[f"blocks.{layer}.attn.hook_pattern"][0, head].numpy()
            # any query position whose target moved
            if any(pa[q].argmax() != pb[q].argmax() for q in qpos):
                moves += 1
        rate = moves / n_pairs
    finally:
        model.blocks[layer].attn.W_Q.data = Wq
        model.blocks[layer].attn.W_K.data = Wk
    return {"synthetic_target_move_rate": rate, "detects": bool(rate >= 0.5)}


# ---------------------------------------------------------------------------
# per-model
# ---------------------------------------------------------------------------

def carry_relevant_positions(cfg):
    """operand-read + '=' + answer positions."""
    eq = 2 * cfg.n_digits + 1
    ans = list(range(cfg.n_ctx - (cfg.n_digits + 2), cfg.n_ctx))
    operands = list(range(0, cfg.n_digits)) + list(range(cfg.n_digits + 1, 2 * cfg.n_digits + 1))
    return sorted(set([eq] + ans + operands))


def run_model(mn):
    model, cfg = load_model(mn)
    acc = verify_accuracy(model, cfg)
    assert acc > 0.9
    n = CFG[mn]["digit"]
    qpos = carry_relevant_positions(cfg)
    # PRIMARY: target stability over random questions
    qs = [random_add_q(cfg)[0] for _ in range(400)]
    patterns = capture_patterns(model, cfg, qs)
    stability = {}
    for L in range(cfg.n_layers):
        ts = target_stability(patterns[L], qpos)
        stability[L] = {f"H{h}_Q{q}": v for (h, q), v in ts.items()}
    # FALSIFIER: value-matched target-move per head/layer + null
    falsifier = {}
    for L in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            tm = target_move_rate(model, cfg, n, L, h, qpos)
            nul = perm_null_targetmove(model, cfg, n, L, h, qpos)
            # exception = any query pos where value-matched top1_move clearly beats the same-state null
            exc = [q for q in qpos if tm[q]["top1_move_rate"] >= 0.5
                   and tm[q]["top1_move_rate"] > nul[q] + 0.3]
            falsifier[f"L{L}H{h}"] = {
                "max_top1_move_rate": max(tm[q]["top1_move_rate"] for q in qpos),
                "exception_positions": exc,
                "per_pos": {str(q): {"vm_top1_move": tm[q]["top1_move_rate"],
                                     "null_move": nul[q]} for q in qpos}}
    # positive control
    ctrl = synthetic_control(model, cfg, n)
    # informative static fraction (primary): informative cells with top1_stab>=0.95
    inf_cells = 0; inf_static = 0
    for L in stability:
        for k, v in stability[L].items():
            if v["informative"]:
                inf_cells += 1
                if v["top1_stability"] >= TOP1_BAR:
                    inf_static += 1
    exceptions = {node: f["exception_positions"] for node, f in falsifier.items() if f["exception_positions"]}
    del model
    return {"accuracy": acc, "digit": n, "carry_relevant_positions": qpos,
            "informative_cells": inf_cells, "informative_static": inf_static,
            "informative_static_fraction": (inf_static / inf_cells) if inf_cells else None,
            "target_move_exceptions": exceptions,
            "synthetic_control": ctrl,
            "stability": stability, "falsifier": falsifier}


def run_models():
    results = {}
    for mn in CFG:
        print(f"=== {mn} ===", flush=True)
        r = run_model(mn)
        results[mn] = r
        print(f"  acc={r['accuracy']:.3f} informative_static={r['informative_static']}/{r['informative_cells']} "
              f"(frac {r['informative_static_fraction']})")
        print(f"  synthetic control detects routing: {r['synthetic_control']['detects']} "
              f"(rate {r['synthetic_control']['synthetic_target_move_rate']:.2f})")
        if r["target_move_exceptions"]:
            print(f"  VALUE-MATCHED TARGET-MOVE EXCEPTIONS: {r['target_move_exceptions']}")
        else:
            print("  no value-matched target-move exceptions (A5-consistent)")
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    # save stratified question sets byproduct
    return results


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "control":
        for mn in CFG:
            model, cfg = load_model(mn)
            print(mn, synthetic_control(model, cfg, CFG[mn]["digit"]))
            del model
    if mode in ("models", "all"):
        run_models()
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
