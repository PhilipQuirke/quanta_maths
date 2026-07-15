"""Deep-cascade mechanism: deciding-digit patching and cascade tracing
(study-deep-cascade-mechanism.md).

Discriminates how multi-digit `...999` carry chains are physically resolved in
2-layer addition models:
  * R-tail (form alpha)  : resolved carry bit stored at the question tail
  * R-sequential (form beta): partial state accumulated across chain positions
  * R-selection (A9)     : one-hop L1-attention fetch of the deciding digit
  * R-widefetch          : static wide read + MLP resolves in one nonlinear step

Implements the pre-run design plus Gate-1 amendments D-1..D-6:
  * D-1: tail-joint provenance check (position-invariance + local-tail decomp)
  * D-2: R-sequential (form beta) row + intermediate all-9s digit patch
  * D-3: control 1 relabeled (harness liveness) + control 6 (computed-state CE5)
  * D-4: control 7 (consumer-class pattern-patch on a CE8 routing cell)
  * D-5: tracking = value-matched target-key move, not raw mass
  * D-6: all-depths accuracy reporting

CPU-only. Run:
    PYTHONPATH=. python3 scripts/deep_cascade_mechanism.py control
    PYTHONPATH=. python3 scripts/deep_cascade_mechanism.py models
    PYTHONPATH=. python3 scripts/deep_cascade_mechanism.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    patched_prediction, _digits_to_int, operand_attention, tristate_test,
    flip_signature, same_class_null,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-deep-cascade-mechanism")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
RNG = np.random.default_rng(SEED)

MODELS = ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]

# CE4/CE5 nodes for controls (per model). conduit = harness-liveness (D-3 C1);
# combiner = computed-state control (D-3 control 6).
CE4_CONDUIT = {  # (pos, layer) of the L0 MLP conduit; borderline in 5-digit per CE5
    "add_d5_l2_h3_t15K_s372001": (10, 0),
    "add_d6_l2_h3_t20K_s173289": (11, 0),
}
CE5_COMBINER = {  # (pos, layer) of the answer-position L1 MLP combiner
    "add_d5_l2_h3_t15K_s372001": (14, 1),
    "add_d6_l2_h3_t20K_s173289": (16, 1),
}
# digit each conduit/combiner serves (verified via tristate_test)
CE4_CONDUIT_DIGIT = {"add_d5_l2_h3_t15K_s372001": 2, "add_d6_l2_h3_t20K_s173289": 3}
CE5_COMBINER_DIGIT = {"add_d5_l2_h3_t15K_s372001": 2, "add_d6_l2_h3_t20K_s173289": 3}
# CE3 SA head for the node/bar control (pos, layer, head, digit)
CE3_SA_HEAD = {
    "add_d5_l2_h3_t15K_s372001": (14, 0, 1, 3),
    "add_d6_l2_h3_t20K_s173289": (20, 0, 1, 0),
}
# CE8 6-digit clean routing cell for Battery C2 (the head A9 predicts routes).
CE8_ROUTING_CELL = {
    "add_d6_l2_h3_t20K_s173289": {"layer": 1, "head": 1, "query": 11},
}
# Control-7 instrument (D-4): a consumer-position L1 head whose PATTERN is
# demonstrably causal, so the pattern-patch method is validated on the correct
# head class even though the specific CE8 cell (L1H1 Q11) turns out pattern-inert.
CONSUMER_L1_INSTRUMENT = {
    "add_d5_l2_h3_t15K_s372001": {"layer": 1, "head": 0, "query": 13},
    "add_d6_l2_h3_t20K_s173289": {"layer": 1, "head": 0, "query": 14},
}


# ===========================================================================
# position helpers (verified against maths_config: an_to_position_name=n_ctx-1-n)
# ===========================================================================

def dn_pos(cfg, n):
    """Position of operand digit Dn (first operand)."""
    return cfg.n_digits - 1 - n


def dpn_pos(cfg, n):
    """Position of operand digit D'n (second operand)."""
    return 2 * cfg.n_digits - n


def ak_pos(cfg, k):
    """Position of answer token A_k."""
    return cfg.n_ctx - 1 - k


def consuming_pos(cfg, k):
    """Residual position that PRODUCES A_k's logits (autoregressive: pos(A_k)-1).
    Leading digit A_top is consumed at the answer-sign position."""
    return ak_pos(cfg, k) - 1


# ===========================================================================
# chain stimulus C(n, k, class) with assertions
# ===========================================================================

def _rand_sum_le8():
    while True:
        a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
        if a + b <= 8:
            return a, b


def _rand_sum_ge10():
    while True:
        a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
        if a + b >= 10:
            return a, b


def _rand_sum9():
    a = int(RNG.integers(0, 10))
    return a, 9 - a


def build_chain(cfg, n, k, dec_class, shared=None):
    """Build one C(n,k,class) question.
      chain digits n-k+1..n : pair-sum == 9  (the ...999 chain)
      deciding digit d=n-k  : class 'lo' (sum<=8) or 'hi' (sum>=10)
      digits < d            : no carry (sum<=8)
      digit n+1             : no carry (sum<=8, != 9) so the top flip is clean
      digits > n+1          : 0

    `shared` lets a matched pair reuse identical fillers for every position EXCEPT
    the deciding operand positions {dn_pos(d), dpn_pos(d)} (so the only token diff
    between a matched hi/lo pair is at the deciding digit). Returns
    (a, b, info, shared_state).
    """
    nd = cfg.n_digits
    d = n - k
    assert d >= 0, f"deciding digit {d} < 0 (n={n},k={k})"
    d1 = [0] * nd; d2 = [0] * nd
    st = {} if shared is None else shared

    def pick(pos_key, gen):
        if pos_key in st:
            return st[pos_key]
        v = gen()
        st[pos_key] = v
        return v

    # deciding digit d: NOT shared (this is the toggled contrast)
    if dec_class == "lo":
        da, db = _rand_sum_le8()
    elif dec_class == "hi":
        da, db = _rand_sum_ge10()
    else:
        raise ValueError(dec_class)
    d1[nd - 1 - d] = da; d2[nd - 1 - d] = db

    # chain digits d+1..n : sum == 9 (shared fillers within a matched pair)
    for j in range(d + 1, n + 1):
        a, b = pick(("chain", j), _rand_sum9)
        d1[nd - 1 - j] = a; d2[nd - 1 - j] = b

    # digits below d : no carry (shared)
    for j in range(0, d):
        a, b = pick(("low", j), _rand_sum_le8)
        d1[nd - 1 - j] = a; d2[nd - 1 - j] = b

    # digit n+1 : no carry and != 9 (shared) so the top carry-out flip is clean
    if n + 1 < nd:
        def gen_np1():
            while True:
                a, b = _rand_sum_le8()
                if a + b != 9:
                    return a, b
        a, b = pick(("above", n + 1), gen_np1)
        d1[nd - 1 - (n + 1)] = a; d2[nd - 1 - (n + 1)] = b
    # digits > n+1 stay 0

    a_int = _digits_to_int(d1); b_int = _digits_to_int(d2)
    info = {"n": n, "k": k, "d": d, "dec_class": dec_class,
            "dec_sum": da + db, "d1": d1, "d2": d2}
    return a_int, b_int, info, st


def assert_matched_pair(cfg, info_hi, info_lo):
    """Verify a matched hi/lo pair differs at exactly the deciding operand
    positions and the chain digits sum to 9 in both (the u-resolution A-6 lesson)."""
    nd = cfg.n_digits
    n, k, d = info_hi["n"], info_hi["k"], info_hi["d"]
    diff_positions = set()
    for arr in ("d1", "d2"):
        for i in range(nd):
            if info_hi[arr][i] != info_lo[arr][i]:
                diff_positions.add((arr, i))
    expected = {("d1", nd - 1 - d), ("d2", nd - 1 - d)}
    # allow the deciding digit to also coincide by chance in one operand; require
    # the diff set to be a SUBSET of the deciding-digit positions.
    assert diff_positions.issubset(expected), \
        f"pair differs outside deciding digit: {diff_positions - expected}"
    for j in range(d + 1, n + 1):
        assert info_hi["d1"][nd - 1 - j] + info_hi["d2"][nd - 1 - j] == 9
        assert info_lo["d1"][nd - 1 - j] + info_lo["d2"][nd - 1 - j] == 9
    assert info_hi["dec_sum"] >= 10 and info_lo["dec_sum"] <= 8
    return True


def affected_digits(n, k):
    """CASCADE answer digits that flip between hi and lo: A_{d+1}..A_{n+1}.
    These are the discriminating digits for the mechanism batteries (the deciding
    digit's carry propagates up through the 9-chain to the top)."""
    d = n - k
    return list(range(d + 1, n + 2))


def behavioral_diff_digits(n, k):
    """Answer digits that differ between a matched hi/lo pair in the CLEAN model:
    the cascade digits A_{d+1}..A_{n+1} PLUS A_d itself (the deciding digit's own
    sum differs between hi and lo, so its answer digit changes too). Used by the
    behavioral gate; the mechanism batteries score only the cascade set."""
    d = n - k
    return list(range(d, n + 2))


# ===========================================================================
# per-depth behavioral gate (D-6: report all depths)
# ===========================================================================

def behavioral_gate(model, cfg, n, k, n_q=40):
    """Clean hi vs lo predictions must differ on exactly A_{d+1}..A_{n+1} and match
    ground truth. Returns pass fraction + detail."""
    ap = answer_positions(cfg); na = len(ap)
    aff = set(behavioral_diff_digits(n, k))
    ok = 0
    for _ in range(n_q):
        sh = {}
        ahi, bhi, ihi, sh = build_chain(cfg, n, k, "hi", shared=sh)
        alo, blo, ilo, _ = build_chain(cfg, n, k, "lo", shared=sh)
        try:
            assert_matched_pair(cfg, ihi, ilo)
        except AssertionError:
            continue
        qhi = make_q(cfg, ahi, bhi); qlo = make_q(cfg, alo, blo)
        phi = predict_answer(model, cfg, qhi); plo = predict_answer(model, cfg, qlo)
        thi = qhi[ap]; tlo = qlo[ap]
        # correctness
        if not (torch.equal(phi, thi) and torch.equal(plo, tlo)):
            continue
        # differ on exactly the affected answer digits
        diff = (phi != plo).numpy()
        diff_digits = set()
        for k2 in range(cfg.n_digits + 1):
            idx = na - 1 - k2
            if 0 <= idx < na and diff[idx]:
                diff_digits.add(k2)
        if diff_digits == aff:
            ok += 1
    return ok / n_q


# ===========================================================================
# flip metric on matched chain pairs (interchange source->target)
# ===========================================================================

def chain_flip(model, cfg, n, k, hooks_builder, n_pairs=40, direction="hi2lo",
               same_class=False):
    """Interchange patch on matched pairs. Reads each AFFECTED digit at its own
    consuming position. Returns per-affected-digit flip rates (vs target clean),
    the mean over affected digits, and the joint-flip rate (all affected flip)."""
    ap = answer_positions(cfg); na = len(ap)
    aff = affected_digits(n, k)
    per = {kk: [] for kk in aff}
    joint = []
    for _ in range(n_pairs):
        sh = {}
        if same_class:
            # both hi, different fillers -> same-class null
            sa, sb, si, sh = build_chain(cfg, n, k, "hi", shared=None)
            ta, tb, ti, _ = build_chain(cfg, n, k, "hi", shared=None)
        else:
            src_cls, tgt_cls = ("hi", "lo") if direction == "hi2lo" else ("lo", "hi")
            sa, sb, si, sh = build_chain(cfg, n, k, src_cls, shared=sh)
            ta, tb, ti, _ = build_chain(cfg, n, k, tgt_cls, shared=sh)
            try:
                assert_matched_pair(cfg, si, ti) if src_cls == "hi" \
                    else assert_matched_pair(cfg, ti, si)
            except AssertionError:
                continue
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        allf = True
        for kk in aff:
            idx = na - 1 - kk
            f = float(diff[idx]) if 0 <= idx < na else 0.0
            per[kk].append(f)
            if f < 0.5:
                allf = False
        joint.append(1.0 if allf else 0.0)
    per_mean = {str(kk): float(np.mean(v)) if v else float("nan") for kk, v in per.items()}
    allv = [x for v in per.values() for x in v]
    return {"per_digit": per_mean,
            "mean_affected_flip": float(np.mean(allv)) if allv else float("nan"),
            "joint_flip": float(np.mean(joint)) if joint else float("nan"),
            "n_pairs": len(joint)}


# ===========================================================================
# controls
# ===========================================================================

def run_controls(model_name):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg, n=64)
    out = {"model": model_name, "accuracy": acc, "seed": SEED}

    # --- control 5: per-depth behavioral separation (also a headline result) ---
    n_top = cfg.n_digits - 2  # main-battery chain top (mid-high digit)
    depth_gate = {}
    for k in range(1, n_top + 1):
        depth_gate[k] = behavioral_gate(model, cfg, n_top, k, n_q=40)
    out["control5_behavioral_gate"] = {"n_top": n_top, "per_depth_pass": depth_gate}

    # --- control 1 (D-3 relabeled): harness liveness on the CE4 conduit ---
    # Uses the exact CE4 depth-1 U counterfactual (fix sum=9 at the served digit,
    # toggle lower carry): a known U-flip TRANSMITTER registers an A_{n+1} flip.
    cp, cl = CE4_CONDUIT[model_name]
    c_dig = CE4_CONDUIT_DIGIT[model_name]
    def hb_conduit():
        return [{"name": f"blocks.{cl}.hook_mlp_out", "pos": cp}]
    c1 = tristate_test(model, cfg, c_dig, hb_conduit, n_pairs=40)
    out["control1_harness_liveness"] = {"conduit": f"P{cp}L{cl}MLP", "digit": c_dig,
                                        "A_n+1_flip": c1["A_n+1_flip"]}

    # --- control 6 (D-3): computed-state cell registers (CE5 combiner) ---
    # The CE5 combiner must show U-regime carry-dependence (tri-state flip) so an
    # R-tail/R-sequential null is interpretable (harness CAN register a state cell).
    mp, ml = CE5_COMBINER[model_name]
    m_dig = CE5_COMBINER_DIGIT[model_name]
    def hb_comb():
        return [{"name": f"blocks.{ml}.hook_mlp_out", "pos": mp}]
    c6 = tristate_test(model, cfg, m_dig, hb_comb, n_pairs=40)
    out["control6_computed_state"] = {"combiner": f"P{mp}L{ml}MLP", "digit": m_dig,
                                      "A_n+1_flip": c6["A_n+1_flip"]}

    # --- control 2 (bar): CE3 SA head z-patch flips A_n (base-add signature) ---
    sp, sl, shd, sdig = CE3_SA_HEAD[model_name]
    def hb_sa():
        return [{"name": f"blocks.{sl}.attn.hook_z", "pos": sp, "head": shd}]
    sig = flip_signature(model, cfg, sdig, hb_sa, "local", n_pairs=40,
                         src_cls=1, tgt_cls=0)
    null = same_class_null(model, cfg, sdig, hb_sa, "local", cls=0, n_pairs=40)
    out["control2_sa_head_bar"] = {"node": f"P{sp}L{sl}H{shd}", "digit": sdig,
                                   "A_n_flip": sig["A_n_flip"],
                                   "A_n+1_flip": sig["A_n+1_flip"],
                                   "same_class_null_A_n": null["A_n_flip"]}

    # --- control 3: readout control (patch consuming resid flips that digit) ---
    readout = {}
    for k in affected_digits(n_top, min(2, n_top)):
        cpos = consuming_pos(cfg, k)
        def hb_read(cpos=cpos):
            return [{"name": "blocks.1.hook_resid_post", "pos": cpos}]
        r = chain_flip(model, cfg, n_top, min(2, n_top), hb_read, n_pairs=30)
        readout[f"A{k}@{cpos}"] = r["per_digit"].get(str(k), float("nan"))
    out["control3_readout"] = readout

    # --- control 4 + 7: pattern-patch validity (D-4) ---
    out["control4_7_pattern_patch"] = pattern_patch_controls(model, cfg, model_name)

    with open(os.path.join(RESULT_DIR, f"control_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    del model
    return out


def _pattern_hook(layer, head, query_pos, src_pattern_row):
    """Hook that overwrites blocks.{layer}.attn.hook_pattern[:,head,query_pos,:]
    with a source attention row."""
    name = f"blocks.{layer}.attn.hook_pattern"
    def hook(act, hook):
        act[:, head, query_pos, :] = torch.tensor(src_pattern_row, dtype=act.dtype)
        return act
    return name, hook


def pattern_patch_prediction(model, cfg, source_q, target_q, layer, head, query_pos):
    """Capture source attention row at (layer,head,query_pos), patch it into
    target, return target predicted answers."""
    with torch.no_grad():
        _, sc = model.run_with_cache(source_q.unsqueeze(0))
    row = sc[f"blocks.{layer}.attn.hook_pattern"][0, head, query_pos, :].numpy()
    name, hook = _pattern_hook(layer, head, query_pos, row)
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=[(name, hook)])
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)


def synthetic_redirect_prediction(model, cfg, target_q, layer, head, query_pos, key_positions):
    """Force (layer,head,query_pos)'s attention onto `key_positions` (uniform),
    return target predicted answers. A causal pattern-only intervention that must
    move the answer if the head's pattern is load-bearing."""
    name = f"blocks.{layer}.attn.hook_pattern"
    def hook(act, hook):
        act[:, head, query_pos, :] = 0.0
        for kp in key_positions:
            act[:, head, query_pos, kp] = 1.0 / len(key_positions)
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=[(name, hook)])
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)


def pattern_patch_controls(model, cfg, model_name):
    """Control 4 (D-4): a causal pattern-patch on the CE3 SA head moves the answer
    predictably. A pattern-only patch can only matter if the KEY positions carry
    different values between source and target, so we contrast two questions whose
    operand digits differ everywhere; a genuinely causal head's pattern-patch then
    changes the target's answer.
    Control 7 (D-4): same on a CE8 clean routing cell (the head-class Battery C2
    patches)."""
    ap = answer_positions(cfg); na = len(ap)
    res = {}

    def redirect_move_rate(layer, head, query, target_digit, n_q=40):
        """Force the head to attend to `target_digit`'s operand positions; a
        load-bearing pattern must move the answer. This is the plan's control 4
        ('patch the pattern to a DIFFERENT digit's operands')."""
        keys = [dn_pos(cfg, target_digit), dpn_pos(cfg, target_digit)]
        moves = []
        lim = 10 ** cfg.n_digits
        for _ in range(n_q):
            a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2))
            tq = make_q(cfg, a, b)
            clean = predict_answer(model, cfg, tq)
            patched = synthetic_redirect_prediction(model, cfg, tq, layer, head, query, keys)
            moves.append(float((patched != clean).numpy().any()))
        return float(np.mean(moves))

    # control 4: CE3 SA head redirected to a DIFFERENT digit's operands (not the
    # one it already reads) -> must move answer.
    sp, sl, shd, sdig = CE3_SA_HEAD[model_name]
    redir4 = 3 if sdig != 3 else 1
    res["control4_sa_pattern_moves"] = {
        "node": f"P{sp}L{sl}H{shd}", "serves_digit": sdig, "redirect_to_digit": redir4,
        "any_move_rate": redirect_move_rate(sl, shd, sp, redir4)}
    # control 7 (D-4): validate the pattern-patch instrument on the CONSUMER L1
    # head class Battery C2 patches. Reports (a) the specific CE8 routing cell
    # (may be pattern-inert -> a real finding) and (b) a candidate consumer L1
    # instrument. If NO consumer L1 head validates, C2 is `invalid` for the model.
    c7 = {}
    if model_name in CE8_ROUTING_CELL:
        rc = CE8_ROUTING_CELL[model_name]
        c7["ce8_cell"] = {
            "node": f"L{rc['layer']}H{rc['head']}Q{rc['query']}", "redirect_to_digit": 3,
            "any_move_rate": redirect_move_rate(rc["layer"], rc["head"], rc["query"], 3)}
    inst = CONSUMER_L1_INSTRUMENT[model_name]
    c7["consumer_l1_instrument"] = {
        "node": f"L{inst['layer']}H{inst['head']}Q{inst['query']}", "redirect_to_digit": 3,
        "any_move_rate": redirect_move_rate(inst["layer"], inst["head"], inst["query"], 3)}
    c7["c2_valid"] = c7["consumer_l1_instrument"]["any_move_rate"] >= 0.30
    res["control7_routing_pattern_moves"] = c7
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("control", "controls", "all"):
        for mn in MODELS:
            print(f"=== CONTROLS {mn} ===", flush=True)
            c = run_controls(mn)
            print(f"  acc={c['accuracy']:.3f}")
            print(f"  [C5] behavioral gate per depth: {c['control5_behavioral_gate']['per_depth_pass']}")
            print(f"  [C1] harness-liveness conduit A_n+1_flip={c['control1_harness_liveness']['A_n+1_flip']:.2f}")
            print(f"  [C6] computed-state combiner A_n+1_flip={c['control6_computed_state']['A_n+1_flip']:.2f}")
            print(f"  [C2] SA-head A_n_flip={c['control2_sa_head_bar']['A_n_flip']:.2f} "
                  f"null={c['control2_sa_head_bar']['same_class_null_A_n']:.2f}")
            print(f"  [C3] readout {c['control3_readout']}")
            print(f"  [C4/7] {c['control4_7_pattern_patch']}")
    if mode in ("models", "all"):
        from scripts.deep_cascade_batteries import run_all_batteries
        run_all_batteries(MODELS, RESULT_DIR, RNG, SEED)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
