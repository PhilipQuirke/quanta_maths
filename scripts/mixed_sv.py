"""M1/M2/M3: replicate the addition SV interface on the mixed model's three
question classes (ADD / SUB / NEG), reusing the addition instruments.

Batteries (per class), each mapped to the addition claim it replicates:
  A  writer encoding (CE13)   -- the question-tail tri-state writer (ST/MT/NT)
                                 linearly encodes its class.
  B  combiner causal (CE5)    -- the last-layer answer MLP is a causal combiner
                                 (STC / MTC / NTC): ablating it flips A_k.
  C  resolved cascade (CE6/7) -- the resolved carry/borrow-in (SV/MV/NV) is a
                                 clean binary code at the combiner input.
  D  edge delivery (CE14)     -- a consumer head (SLT, L1) delivers the resolving
                                 carry/borrow into the L2 combiner, cascade-
                                 specifically (deciding-matched null must be ~0).

Positive controls: per-class accuracy (M0) + an untrained control that must fail
A/C; for D the stimulus validity (clean source A_k != clean target A_k) gates the
measurement, and the deciding-matched null is the specificity control.

Evidence integrity: every number is computed here and written to
results/study-mixed-sv/results.json.

Run: PYTHONPATH=. python scripts/mixed_sv.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf, make_untrained_control
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_probe import (
    sub_labels, neg_labels, site_hook_and_pos, first_layer, last_layer,
    probe_accuracy_with_null,
)
from quanta_maths.maths_edge_patch import (
    answer_positions, consuming_pos, run_multi_head_edge_patch,
    flip_rate_with_matched_null,
)
from quanta_maths.maths_batch import _combiner_is_causal

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
OUT_DIR = "results/study-mixed-sv"
CLASSES = ["ADD", "SUB", "NEG"]
OP = {"ADD": MathsToken.PLUS, "SUB": MathsToken.MINUS, "NEG": MathsToken.MINUS}
# SLT consumer/selector head from the verified map (L1 H1 at answer positions).
CONSUMER_LAYER, CONSUMER_HEAD = 1, 1


# ---------------------------------------------------------------------------
# class-aware question + label helpers
# ---------------------------------------------------------------------------

def class_question(cfg, rng, cls):
    lim = 10 ** cfg.n_digits
    if cls == "ADD":
        return int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
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


def class_labels(cfg, a, b, cls):
    if cls == "NEG":
        return neg_labels(a, b, cfg.n_digits)
    return sub_labels(a, b, cfg.n_digits, operation=OP[cls])


def to_q(cfg, a, b, cls):
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, OP[cls])
    return q[0]


# ---------------------------------------------------------------------------
# Battery A/C: collect activations + class-correct labels at a site/layer
# ---------------------------------------------------------------------------

def collect(model, cfg, cls, n_q, digits, site, layer, rng):
    acts = {n: [] for n in digits}
    labs = {t: {n: [] for n in digits} for t in ("ST", "SV", "SA")}
    hook0 = site_hook_and_pos(cfg, site, digits[0], layer=layer)[0]
    for _ in range(n_q):
        a, b = class_question(cfg, rng, cls)
        q = to_q(cfg, a, b, cls)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook0)
        SA, ST, SV = class_labels(cfg, a, b, cls)
        for n in digits:
            _, pos = site_hook_and_pos(cfg, site, n, layer=layer)
            acts[n].append(c[hook0][0, pos, :].numpy())
            labs["ST"][n].append(ST[n]); labs["SV"][n].append(SV[n]); labs["SA"][n].append(SA[n])
    acts = {n: np.asarray(v) for n, v in acts.items()}
    for t in labs:
        labs[t] = {n: np.asarray(v) for n, v in labs[t].items()}
    return acts, labs


def battery_A_C(model, cfg, cls, rng, n_q=400, digits=(1, 2, 3)):
    """A: tri-state (ST/MT/NT) decodable at writer (Dpn, first layer).
    C: resolved cascade (SV/MV/NV) binary at combiner input (ans, last layer)."""
    # A -- writer encoding at the operand-2 (question-tail) site, early layer
    aA, lA = collect(model, cfg, cls, n_q, list(digits), "Dpn", first_layer(cfg), rng)
    A = {}
    for n in digits:
        out = probe_accuracy_with_null(aA[n], lA["ST"][n], rng, n_perm=30)
        A[n] = {"tri_acc": out["observed_acc"], "null_p": out["null_p"]}
    # C -- resolved cascade at the answer-producing position, last layer
    aC, lC = collect(model, cfg, cls, n_q, list(digits), "ans", last_layer(cfg), rng)
    C = {}
    for n in digits:
        # SV/MV/NV is binary; require both classes present
        if len(np.unique(lC["SV"][n])) < 2:
            C[n] = {"carry_acc": float("nan"), "null_p": float("nan")}
            continue
        out = probe_accuracy_with_null(aC[n], lC["SV"][n], rng, n_perm=30)
        C[n] = {"carry_acc": out["observed_acc"], "null_p": out["null_p"]}
    return A, C


# ---------------------------------------------------------------------------
# Battery B: combiner causal (STC / MTC / NTC) at the last layer
# Uses the same validated single-step U stimulus as Battery D (non-degenerate,
# distinct operands) rather than the library's a=b convenience stimulus, so the
# three classes are compared on equal footing. A_k(U) is decided by the carry/
# borrow-in; zero-ablating the last-layer answer MLP must change A_k if that MLP
# is the combiner.
# ---------------------------------------------------------------------------

def _ablate_Ak(model, cfg, q, cpos, k, mlp_layer):
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - k

    def hook(act, hook):
        act[:, cpos, :] = 0.0
        return act
    with torch.no_grad():
        lg = model.run_with_hooks(q.unsqueeze(0),
                                  fwd_hooks=[(f"blocks.{mlp_layer}.mlp.hook_post", hook)])
    return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])


def _clean_Ak(model, cfg, q, k):
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - k
    with torch.no_grad():
        lg = model(q.unsqueeze(0))
    return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])


def battery_B(model, cfg, cls):
    ll = last_layer(cfg)
    causal = {}          # U-stimulus zero-ablation (this harness)
    causal_lib = {}      # library carry/borrow stimulus (_combiner_is_causal)
    for k in range(1, cfg.n_digits):
        pp = consuming_pos(cfg, k)
        a, b = _edge_pair_digits(cfg, cls, k, carry_in=1)  # U at k, resolved by cascade
        q = to_q(cfg, a, b, cls)
        causal[k] = bool(_clean_Ak(model, cfg, q, k) != _ablate_Ak(model, cfg, q, pp, k, ll))
        causal_lib[k] = bool(_combiner_is_causal(model, cfg, pp, k, ll, cls=cls))
    return {"u_stimulus": causal, "lib_stimulus": causal_lib}


# ---------------------------------------------------------------------------
# Battery D: cascade-specific consumer-head -> combiner edge delivery (CE14)
# single-step U test: digit k is a tri-state U; toggling carry/borrow-in flips A_k
# between 9 and 0. Patch the SLT consumer head's OV message src->tgt into the L2
# combiner input and measure the A_k flip vs a deciding-matched null.
# ---------------------------------------------------------------------------

def _edge_pair_digits(cfg, cls, k, carry_in):
    """Return (D, D') integer operands with digit k a 'U' and carry/borrow-in to k
    equal to carry_in (0/1), class-consistent, other digits definite filler."""
    nd = cfg.n_digits
    D = [0] * nd            # index 0 = units
    Dp = [0] * nd
    top = nd - 1
    if cls == "ADD":
        D[k], Dp[k] = 4, 5                 # sum 9 -> U
        D[k - 1], Dp[k - 1] = (9, 9) if carry_in else (0, 0)
        # other digits definite no-carry (0+0) already
    elif cls == "SUB":
        D[k], Dp[k] = 5, 5                 # equal -> U (borrow tricase)
        D[k - 1], Dp[k - 1] = (0, 9) if carry_in else (9, 0)
        D[top], Dp[top] = 9, 0             # ensure D > D'
        for j in range(nd):                # definite no-borrow filler
            if j not in (k, k - 1, top):
                D[j], Dp[j] = 9, 1
    else:  # NEG: compute on D'-D; ensure D < D'
        D[k], Dp[k] = 5, 5                 # equal -> U
        D[k - 1], Dp[k - 1] = (9, 0) if carry_in else (0, 9)  # D'<D -> neg-borrow into k
        D[top], Dp[top] = 0, 9             # ensure D < D'
        for j in range(nd):                # definite neg filler (D' > D, no neg-borrow)
            if j not in (k, k - 1, top):
                D[j], Dp[j] = 1, 9
    a = int("".join(str(d) for d in D[::-1]))
    b = int("".join(str(d) for d in Dp[::-1]))
    return a, b


def _resid_patch_pred(model, cfg, sq, tq, cpos, hook_name):
    """Patch the (source - target) residual delta at ``cpos`` into ``hook_name``."""
    with torch.no_grad():
        _, sc = model.run_with_cache(sq.unsqueeze(0), names_filter=lambda nm: nm == hook_name)
        _, tc = model.run_with_cache(tq.unsqueeze(0), names_filter=lambda nm: nm == hook_name)
    delta = sc[hook_name][0, cpos, :] - tc[hook_name][0, cpos, :]

    def hook(act, hook):
        act[:, cpos, :] = act[:, cpos, :] + delta
        return act
    ap = answer_positions(cfg)
    with torch.no_grad():
        lg = model.run_with_hooks(tq.unsqueeze(0), fwd_hooks=[(hook_name, hook)])
    return lg[0, [p - 1 for p in ap]].argmax(-1)


def _edge_flip(model, cfg, cls, k, a1, b1, a0, b0, recv_layer, patch_kind):
    """One (source carry_in=1 -> target carry_in=0) patch; A_k flip vs a
    deciding-matched null (both carry_in=0). ``patch_kind``:
      full_resid   -- whole combiner input (blocks.L.hook_resid_mid) [control]
      resid_pre    -- pre-last-layer residual (blocks.L.hook_resid_pre = L0+L1 out):
                      does the resolved cascade RIDE THE RESIDUAL into the combiner?
      lastlayer_attn -- all last-layer attention heads' OV -> combiner ln2:
                      does LAST-LAYER ATTENTION deliver it (the 2-layer CE14 edge)?"""
    cpos = consuming_pos(cfg, k)

    def patch_fn(sq, tq):
        if patch_kind == "full_resid":
            return _resid_patch_pred(model, cfg, sq, tq, cpos,
                                     f"blocks.{recv_layer}.hook_resid_mid")
        if patch_kind == "resid_pre":
            return _resid_patch_pred(model, cfg, sq, tq, cpos,
                                     f"blocks.{recv_layer}.hook_resid_pre")
        # lastlayer_attn: patch the last-layer heads' OV into the combiner ln2 input
        with torch.no_grad():
            _, sc = model.run_with_cache(
                sq.unsqueeze(0),
                names_filter=lambda nm: nm == f"blocks.{recv_layer}.attn.hook_z")
            _, tc = model.run_with_cache(tq.unsqueeze(0))
        heads = [(cpos, recv_layer, h,
                  sc[f"blocks.{recv_layer}.attn.hook_z"][0, cpos, h, :].numpy())
                 for h in range(cfg.n_heads)]
        return run_multi_head_edge_patch(model, cfg, tq, patches=heads,
                                         recv_layer=recv_layer, tgt_cache=tc)

    def pair_builder():
        return to_q(cfg, a1, b1, cls), to_q(cfg, a0, b0, cls)

    def null_builder():
        aa, bb = _edge_pair_digits(cfg, cls, k, carry_in=0)
        return to_q(cfg, aa, bb, cls), to_q(cfg, a0, b0, cls)

    return flip_rate_with_matched_null(model, cfg, pair_builder, patch_fn,
                                       answer_digit=k, n_pairs=24, null_builder=null_builder)


def battery_D(model, cfg, cls, k=2):
    """Cascade-specific delivery to the combiner (3-layer-correct arms)."""
    ll = last_layer(cfg)
    a1, b1 = _edge_pair_digits(cfg, cls, k, carry_in=1)   # source (A_k resolved one way)
    a0, b0 = _edge_pair_digits(cfg, cls, k, carry_in=0)   # target (resolved the other)
    src_Ak = _clean_Ak(model, cfg, to_q(cfg, a1, b1, cls), k)
    tgt_Ak = _clean_Ak(model, cfg, to_q(cfg, a0, b0, cls), k)
    out = {"answer_digit": k, "consuming_pos": consuming_pos(cfg, k),
           "stimulus_valid": bool(src_Ak != tgt_Ak), "src_Ak": src_Ak, "tgt_Ak": tgt_Ak}
    for kind in ("full_resid", "resid_pre", "lastlayer_attn"):
        out[kind] = _edge_flip(model, cfg, cls, k, a1, b1, a0, b0, ll, kind)
    return out


# ---------------------------------------------------------------------------

def run_all(model, cfg, tag, rng_seed=0):
    out = {}
    for cls in CLASSES:
        rng = np.random.default_rng(rng_seed)
        print(f"--- [{tag}] {cls} ---")
        A, C = battery_A_C(model, cfg, cls, rng)
        B = battery_B(model, cfg, cls)
        D = battery_D(model, cfg, cls)
        out[cls] = {"A_writer_encoding": A, "B_combiner_causal": B,
                    "C_resolved_cascade": C, "D_edge_delivery": D}
        print("  A tri-acc:", {n: round(A[n]["tri_acc"], 3) for n in A})
        print("  B combiner-causal (U-stim):", [k for k, v in B["u_stimulus"].items() if v],
              "| (lib-stim):", [k for k, v in B["lib_stimulus"].items() if v])
        print("  C carry-acc:", {n: round(C[n]["carry_acc"], 3) for n in C})
        print(f"  D A{D['answer_digit']} valid={D['stimulus_valid']} | "
              f"full_resid={D['full_resid']['flip']['rate']:.2f}/n{D['full_resid']['null']['rate']:.2f} "
              f"resid_pre={D['resid_pre']['flip']['rate']:.2f}/n{D['resid_pre']['null']['rate']:.2f} "
              f"lastlayer_attn={D['lastlayer_attn']['flip']['rate']:.2f}/n{D['lastlayer_attn']['null']['rate']:.2f}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")

    print("=== M1/M2/M3 [1/2] trained model ===")
    trained = run_all(model, cfg, "trained")

    print("=== M1/M2/M3 [2/2] untrained control (must fail A/C/D) ===")
    ctrl = make_untrained_control(cfg)
    control = {}
    for cls in CLASSES:
        rng = np.random.default_rng(0)
        A, C = battery_A_C(ctrl, cfg, cls, rng, n_q=250, digits=(2,))
        D = battery_D(ctrl, cfg, cls)
        control[cls] = {"A_writer_encoding": A, "C_resolved_cascade": C,
                        "D_edge_delivery": {"full_resid": D["full_resid"],
                                            "stimulus_valid": D["stimulus_valid"]}}
        print(f"  {cls}: A tri-acc(d2)={A[2]['tri_acc']:.3f} C carry-acc(d2)={C[2]['carry_acc']:.3f} "
              f"D full_resid flip={D['full_resid']['flip']['rate']:.2f}")

    result = {"model": MODEL, "consumer_head": f"L{CONSUMER_LAYER}H{CONSUMER_HEAD} (SLT)",
              "trained": trained, "untrained_control": control}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
