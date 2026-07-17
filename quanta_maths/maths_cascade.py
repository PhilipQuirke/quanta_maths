"""Deep carry/borrow cascade stimuli + combiner-delivery sweep (model-general).

Promoted from the mixed-model >=2-depth delivery study so the sweep can be run
across the model zoo (the delivery pathway may differ by model). Two reusable
pieces:

  * ``make_cascade_operands`` -- build (a, b) with a depth-k carry/borrow chain for
    a given question class (ADD / SUB / NEG). The deciding digit generates or does
    not generate the cascade; the digits between it and the read digit are
    tri-state ``U`` propagators; digits above are definite and set the class
    (ADD free; SUB D>D'; NEG D<D') and stop the chain (sign stable).
  * ``combiner_delivery_flip`` / ``combiner_delivery_sweep`` -- at a read digit's
    last-layer combiner input, measure whether a delivery arm carries the resolved
    cascade carry/borrow-specifically (deciding-matched null), per class and depth.

Arms: ``full_resid`` (whole resid_mid -- positive control / near-tautological),
``resid_pre`` (pre-last-layer residual), ``lastlayer_attn`` (all last-layer heads'
OV -- the class-discriminating arm). CPU-friendly.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_probe import last_layer
from quanta_maths.maths_edge_patch import (
    answer_positions, consuming_pos, run_multi_head_edge_patch,
    flip_rate_with_matched_null,
)

CLASS_OP = {"ADD": MathsToken.PLUS, "SUB": MathsToken.MINUS, "NEG": MathsToken.MINUS}
ARMS = ("full_resid", "resid_pre", "lastlayer_attn")


# ===========================================================================
# Deep-cascade stimulus
# ===========================================================================

def make_cascade_operands(cfg, cls: str, read_digit: int, carry_in: int,
                          variant: int = 0, deciding_digit: int = 0) -> Tuple[int, int]:
    """(a, b) with a depth-(read_digit - deciding_digit) carry/borrow chain.

    ``cls`` in {"ADD","SUB","NEG"}. ``carry_in`` 1 makes the deciding digit
    generate the cascade, 0 does not; ``variant`` (used with carry_in=0) picks a
    different non-cascading deciding operand for a deciding-matched null. Digits
    above ``read_digit`` are definite and set the class + stop the chain.
    """
    if cls not in CLASS_OP:
        raise ValueError(f"unknown class {cls!r}")
    nd = cfg.n_digits
    if not (0 <= deciding_digit < read_digit < nd):
        raise ValueError(f"need 0<=deciding<{read_digit}<{nd} (read_digit={read_digit})")
    D = [0] * nd
    Dp = [0] * nd
    top = nd - 1
    d = deciding_digit

    if cls == "ADD":
        D[d], Dp[d] = (9, 9) if carry_in else ((0, 0) if variant == 0 else (1, 1))
        for j in range(d + 1, read_digit + 1):
            D[j], Dp[j] = 4, 5                       # sum 9 -> U
        for j in range(read_digit + 1, nd):
            D[j], Dp[j] = 0, 0                       # definite, no carry
    elif cls == "SUB":                               # positive answer, D > D'
        D[d], Dp[d] = (0, 9) if carry_in else ((9, 0) if variant == 0 else (8, 1))
        for j in range(d + 1, read_digit + 1):
            D[j], Dp[j] = 5, 5                       # equal -> U (borrow tricase)
        for j in range(read_digit + 1, top):
            D[j], Dp[j] = 9, 1                       # definite, no borrow, D>D'
        D[top], Dp[top] = 9, 0                       # top: D > D'
    else:                                            # NEG: negative answer, D < D'
        D[d], Dp[d] = (9, 0) if carry_in else ((0, 9) if variant == 0 else (1, 8))
        for j in range(d + 1, read_digit + 1):
            D[j], Dp[j] = 5, 5                       # equal -> U
        for j in range(read_digit + 1, top):
            D[j], Dp[j] = 1, 9                       # definite, no neg-borrow, D<D'
        D[top], Dp[top] = 0, 9                       # top: D < D'

    a = int("".join(str(x) for x in D[::-1]))
    b = int("".join(str(x) for x in Dp[::-1]))
    return a, b


def cascade_answer_digit(cls: str, a: int, b: int, read_digit: int, n_digits: int) -> int:
    """The true emitted answer digit A_{read_digit} for the class (ground truth,
    model-independent -- used to validate a stimulus offline)."""
    val = (a + b) if cls == "ADD" else abs(a - b)
    return int(str(val).zfill(n_digits + 2)[::-1][read_digit])


def cascade_question(cfg, cls: str, a: int, b: int) -> torch.Tensor:
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, CLASS_OP[cls])
    return q[0]


# ===========================================================================
# Combiner-delivery sweep
# ===========================================================================

def _clean_answer_digit(model, cfg, q, k):
    ap = answer_positions(cfg)
    with torch.no_grad():
        lg = model(q.unsqueeze(0))
    return int(lg[0, [p - 1 for p in ap]].argmax(-1)[len(ap) - 1 - k])


def _arm_patch_fn(model, cfg, cls, read_digit, arm, recv_layer):
    cpos = consuming_pos(cfg, read_digit)
    ap = answer_positions(cfg)

    def patch_fn(sq, tq):
        with torch.no_grad():
            _, tc = model.run_with_cache(tq.unsqueeze(0))
        if arm in ("full_resid", "resid_pre"):
            hook = (f"blocks.{recv_layer}.hook_resid_mid" if arm == "full_resid"
                    else f"blocks.{recv_layer}.hook_resid_pre")
            with torch.no_grad():
                _, sc = model.run_with_cache(sq.unsqueeze(0), names_filter=lambda nm: nm == hook)
            delta = sc[hook][0, cpos, :] - tc[hook][0, cpos, :]

            def hk(act, hook):
                act[:, cpos, :] = act[:, cpos, :] + delta
                return act
            with torch.no_grad():
                lg = model.run_with_hooks(tq.unsqueeze(0), fwd_hooks=[(hook, hk)])
            return lg[0, [p - 1 for p in ap]].argmax(-1)
        if arm == "lastlayer_attn":
            zname = f"blocks.{recv_layer}.attn.hook_z"
            with torch.no_grad():
                _, sc = model.run_with_cache(sq.unsqueeze(0), names_filter=lambda nm: nm == zname)
            heads = [(cpos, recv_layer, h, sc[zname][0, cpos, h, :].numpy())
                     for h in range(cfg.n_heads)]
            return run_multi_head_edge_patch(model, cfg, tq, patches=heads,
                                             recv_layer=recv_layer, tgt_cache=tc)
        raise ValueError(f"unknown arm {arm!r}")
    return patch_fn


def combiner_delivery_flip(model, cfg, cls: str, read_digit: int,
                           arm: str = "lastlayer_attn", n_pairs: int = 24,
                           recv_layer: int = None) -> dict:
    """A_k flip rate for a delivery ``arm`` on a depth-``read_digit`` cascade vs a
    deciding-matched null (both carry_in=0, different deciding operand). Returns
    ``flip``/``null`` rates (+CIs), stimulus validity, and the clean endpoints.
    """
    if recv_layer is None:
        recv_layer = last_layer(cfg)
    a1, b1 = make_cascade_operands(cfg, cls, read_digit, carry_in=1)
    a0, b0 = make_cascade_operands(cfg, cls, read_digit, carry_in=0, variant=0)
    src = _clean_answer_digit(model, cfg, cascade_question(cfg, cls, a1, b1), read_digit)
    tgt = _clean_answer_digit(model, cfg, cascade_question(cfg, cls, a0, b0), read_digit)
    patch_fn = _arm_patch_fn(model, cfg, cls, read_digit, arm, recv_layer)

    def pair_builder():
        return cascade_question(cfg, cls, a1, b1), cascade_question(cfg, cls, a0, b0)

    def null_builder():
        an, bn = make_cascade_operands(cfg, cls, read_digit, carry_in=0, variant=1)
        return cascade_question(cfg, cls, an, bn), cascade_question(cfg, cls, a0, b0)

    res = flip_rate_with_matched_null(model, cfg, pair_builder, patch_fn,
                                      answer_digit=read_digit, n_pairs=n_pairs,
                                      null_builder=null_builder)
    res.update({"arm": arm, "read_digit": read_digit, "cls": cls,
                "stimulus_valid": bool(src != tgt), "src_Ak": src, "tgt_Ak": tgt})
    return res


def combiner_delivery_sweep(model, cfg, classes: Sequence[str] = None,
                            depths: Sequence[int] = (2, 3, 4),
                            arms: Sequence[str] = ARMS, n_pairs: int = 24) -> dict:
    """Run ``combiner_delivery_flip`` over classes x depths x arms. Returns a nested
    dict ``out[cls][depth][arm] = {flip, null, valid}``. Model-general: pass any
    loaded maths model + cfg (e.g. across the zoo) to compare delivery pathways.
    """
    classes = list(classes) if classes is not None else ["ADD", "SUB", "NEG"]
    out = {}
    for cls in classes:
        out[cls] = {}
        for k in depths:
            out[cls][k] = {}
            for arm in arms:
                r = combiner_delivery_flip(model, cfg, cls, k, arm, n_pairs=n_pairs)
                out[cls][k][arm] = {"flip": r["flip"]["rate"], "null": r["null"]["rate"],
                                    "valid": r["stimulus_valid"]}
    return out


def model_classes(cfg) -> List[str]:
    """Question classes a model supports, from its config: ADD if it does addition,
    SUB and NEG if it does subtraction."""
    out = []
    if getattr(cfg, "perc_add", 0) > 0:
        out.append("ADD")
    if getattr(cfg, "perc_sub", 0) > 0:
        out += ["SUB", "NEG"]
    return out


def delivery_route(model, cfg, cls: str, read_digit: int = 2,
                   n_pairs: int = 20, hi: float = 0.6, lo: float = 0.3) -> str:
    """Classify how the resolved carry/borrow reaches the combiner for ``cls``:
    ``resatt`` (residual + last-layer attention), ``res`` (residual only),
    ``att`` (attention only), ``none``, or ``na`` (stimulus invalid). CE25.
    Carry/borrow-specific: an arm counts only if flip>=hi AND deciding null<lo.
    """
    resid = combiner_delivery_flip(model, cfg, cls, read_digit, "resid_pre", n_pairs=n_pairs)
    if not resid["stimulus_valid"]:
        return "na"
    attn = combiner_delivery_flip(model, cfg, cls, read_digit, "lastlayer_attn", n_pairs=n_pairs)
    res_ok = resid["flip"]["rate"] >= hi and resid["null"]["rate"] < lo
    att_ok = attn["flip"]["rate"] >= hi and attn["null"]["rate"] < lo
    if res_ok and att_ok:
        return "resatt"
    if res_ok:
        return "res"
    if att_ok:
        return "att"
    return "none"


def tag_delivery_route_nodes(model, cfg, nodes, read_digit: int = 2, n_pairs: int = 20) -> int:
    """Tag the last-layer answer-MLP combiner nodes with the per-class delivery
    route ``Probe:DELIVERY.{ADD|SUB|NEG}={res|resatt|att|none|na}`` (CE25). A
    model-level fact replicated onto each combiner node; run across the zoo to see
    where the route differs by model. Returns tags added.
    """
    ll = last_layer(cfg)
    produce = {consuming_pos(cfg, k) for k in range(1, cfg.n_digits)}
    added = 0
    for cls in model_classes(cfg):
        route = delivery_route(model, cfg, cls, read_digit, n_pairs=n_pairs)
        for node in nodes.nodes:
            if (not node.is_head) and node.layer == ll and node.position in produce:
                added += node.add_tag("Probe", f"DELIVERY.{cls}={route}")
    return added
