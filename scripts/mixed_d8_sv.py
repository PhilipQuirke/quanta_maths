"""Generate the full d8 mixed SV dataset (model-general battery on the accurate
8-digit mixed model ``ins1_mix_d8_l3_h4_t70K_s572091``).

All batteries are map-free (stimuli + loci from the config), reusing the same
library + battery code validated on d6, so this is a genuine cross-size (d6->d8)
test of the mixed SV mechanism (A12):

  accuracy         per-class ADD/SUB/NEG (positive control)
  A writer-encode  tri-state (ST/MT/NT) decodable at the question-tail writer
  C resolved-casc  binary carry/borrow (SV/MV/NV) at the last-layer combiner input
  B combiner       last-layer MLP is a causal combiner (U-stim + library check)
  STEP             on-manifold alpha-sweep -> step function
  =-source         `=` is not the middle-digit value source
  canonical        resolved carry is a format-invariant code (cross-digit transfer)
  DELIVERY         class-dependent route across depths 2-4 (CE25)

Also emits the reusable d8 SV node-tag dataset: features.json (Algo:A{d}.STC/MTC/
NTC combiners) + behaviors.json (Probe:DELIVERY.* routes).

The d8 model has NO published verified map on HF, so the map-anchored pieces
(paper Fail%/Impact map, mechanism diagram, writer-necessity, SLT-sited
shared-engine) are out of scope here.

Run: PYTHONPATH=. python scripts/mixed_d8_sv.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from QuantaMechInterp import UsefulNode, UsefulNodeList

from quanta_maths import (load_maths_model_from_hf, make_untrained_control,
                          combiner_delivery_sweep, tag_delivery_route_nodes,
                          make_cascade_operands)
from quanta_maths.maths_search_add import add_stc_functions
from quanta_maths.maths_search_sub import sub_mtc_functions, neg_ntc_functions
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_edge_patch import answer_positions, consuming_pos
from quanta_maths.maths_probe import last_layer

# reuse the model-general batteries validated on d6 (same-dir imports)
from mixed_sv import battery_A_C, battery_B, class_question, to_q, CLASSES, OP
from mixed_sv_impl import battery_step, battery_eq_source, battery_canonical

MODEL = "ins1_mix_d8_l3_h4_t70K_s572091"
OUT_DIR = "results/study-mixed-d8"


def per_class_accuracy(model, cfg, n=300):
    ap = answer_positions(cfg)
    out = {}
    for cls in CLASSES:
        rng = np.random.default_rng(0)
        hit = 0
        for _ in range(n):
            a, b = class_question(cfg, rng, cls)
            q = to_q(cfg, a, b, cls)
            with torch.no_grad():
                lg = model(q.unsqueeze(0))
            hit += int((lg[0, [p - 1 for p in ap]].argmax(-1) == q[ap]).all())
        out[cls] = hit / n
    return out


TAG_FN = {"ADD": add_stc_functions.tag, "SUB": sub_mtc_functions.tag,
          "NEG": neg_ntc_functions.tag}


def _combiner_input_causal(model, cfg, cls, k):
    """Robust (redundancy-proof) combiner check for deep/large models: does the
    last-layer combiner INPUT causally set A_k? Patch the resid_mid at the
    producing position from a carry_in=1 cascade onto the carry_in=0 one and see
    if A_k flips to the carry_in=1 value. (Zero-ablating the whole MLP is
    redundancy-limited on d8; this input-causality criterion is what the STEP
    battery uses and it works.)"""
    ll = last_layer(cfg)
    pp = consuming_pos(cfg, k)
    ap = answer_positions(cfg)
    idx = len(ap) - 1 - k
    hook = f"blocks.{ll}.hook_resid_mid"
    a1, b1 = make_cascade_operands(cfg, cls, k, carry_in=1)
    a0, b0 = make_cascade_operands(cfg, cls, k, carry_in=0)
    q1, q0 = to_q(cfg, a1, b1, cls), to_q(cfg, a0, b0, cls)

    def ak(q):
        with torch.no_grad():
            lg = model(q.unsqueeze(0))
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    ak0, ak1 = ak(q0), ak(q1)
    if ak0 == ak1:
        return False  # invalid stimulus (no cascade reaches k)
    with torch.no_grad():
        _, c1 = model.run_with_cache(q1.unsqueeze(0), names_filter=lambda nm: nm == hook)
        _, c0 = model.run_with_cache(q0.unsqueeze(0), names_filter=lambda nm: nm == hook)
    delta = c1[hook][0, pp, :] - c0[hook][0, pp, :]

    def hk(act, hook):
        act[:, pp, :] = act[:, pp, :] + delta
        return act
    with torch.no_grad():
        lg = model.run_with_hooks(q0.unsqueeze(0), fwd_hooks=[(hook, hk)])
    return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx]) == ak1


def emit_node_tags(model, cfg):
    """Build the answer-position last-layer MLP node list and tag it with the
    combiner (STC/MTC/NTC, via the redundancy-proof combiner-input criterion) +
    delivery-route tags; save features/behaviors JSON."""
    from QuantaMechInterp import QType
    ll = last_layer(cfg)
    feats = UsefulNodeList()
    behav = UsefulNodeList()
    produce = {consuming_pos(cfg, k): k for k in range(cfg.n_digits + 1)}
    for pos in sorted(produce):
        feats.nodes.append(UsefulNode(pos, ll, False, 0, []))
        behav.nodes.append(UsefulNode(pos, ll, False, 0, []))
    counts = {"STC": 0, "MTC": 0, "NTC": 0}
    keymap = {"ADD": "STC", "SUB": "MTC", "NEG": "NTC"}
    for cls in CLASSES:
        for k in range(1, cfg.n_digits - 1):   # k with room for cascade + class-top
            if _combiner_input_causal(model, cfg, cls, k):
                pos = consuming_pos(cfg, k)
                for node in feats.nodes:
                    if node.position == pos and node.layer == ll and not node.is_head:
                        counts[keymap[cls]] += node.add_tag(QType.ALGO.value, TAG_FN[cls](k))
    n_del = tag_delivery_route_nodes(model, cfg, behav)
    os.makedirs(OUT_DIR, exist_ok=True)
    feats.save_nodes(os.path.join(OUT_DIR, "features.json"), "Algo")
    behav.save_nodes(os.path.join(OUT_DIR, "behaviors.json"), "")
    counts["DELIVERY"] = n_del
    return counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")
    print(f"=== {MODEL}: n_digits={cfg.n_digits} n_layers={cfg.n_layers} "
          f"n_heads={cfg.n_heads} n_ctx={cfg.n_ctx} ===")

    print("--- accuracy (positive control) ---")
    acc = per_class_accuracy(model, cfg)
    print(" ", acc)

    out = {"model": MODEL,
           "config": {"n_digits": cfg.n_digits, "n_layers": cfg.n_layers,
                      "n_heads": cfg.n_heads, "n_ctx": cfg.n_ctx,
                      "perc_add": cfg.perc_add, "perc_sub": cfg.perc_sub},
           "accuracy": acc, "classes": {}}

    for cls in CLASSES:
        print(f"--- {cls} ---")
        rng = np.random.default_rng(0)
        A, C = battery_A_C(model, cfg, cls, rng, n_q=250)
        B = battery_B(model, cfg, cls)
        step = battery_step(model, cfg, cls)
        eqs = battery_eq_source(model, cfg, cls)
        canon = battery_canonical(model, cfg, cls, np.random.default_rng(3), n_q=250)
        out["classes"][cls] = {
            "A_writer_encoding": A, "C_resolved_cascade": C,
            "B_combiner_causal": B, "STEP": step,
            "eq_source": eqs, "canonical": canon,
        }
        print(f"  A tri-acc: {{ {', '.join(f'{n}:{A[n]['tri_acc']:.2f}' for n in A)} }}")
        print(f"  C carry-acc: {{ {', '.join(f'{n}:{C[n]['carry_acc']:.2f}' for n in C)} }}")
        print(f"  B combiner-causal (lib): {[k for k,v in B['lib_stimulus'].items() if v]}")
        print(f"  STEP A_k {step['Ak0']}->{step['Ak1']} endpoints_ok={step['endpoints_ok']} "
              f"alpha*={step['alpha_star']} sweep={step['sweep']}")
        print(f"  =-source flip={eqs['eq_source_flip']} (combiner control={eqs['combiner_input_flip']})")
        print(f"  canonical transfer={canon['transfer_acc']:.2f} self={canon['self_acc']:.2f}")

    print("--- DELIVERY sweep (classes x depths 2-4 x arms) ---")
    delivery = combiner_delivery_sweep(model, cfg, classes=CLASSES, depths=(2, 3, 4), n_pairs=20)
    out["delivery"] = delivery
    for cls in CLASSES:
        for k in (2, 3, 4):
            v = delivery[cls][k]
            print(f"  {cls} d{k}: resid_pre={v['resid_pre']['flip']:.2f}/n{v['resid_pre']['null']:.2f} "
                  f"attn={v['lastlayer_attn']['flip']:.2f}/n{v['lastlayer_attn']['null']:.2f}")

    print("--- untrained control (writer-encoding + delivery must fail) ---")
    ctrl = make_untrained_control(cfg)
    rng = np.random.default_rng(0)
    A0, C0 = battery_A_C(ctrl, cfg, "ADD", rng, n_q=150, digits=(2,))
    out["untrained_control"] = {"ADD_tri_acc_d2": A0[2]["tri_acc"], "ADD_carry_acc_d2": C0[2]["carry_acc"]}
    print(f"  untrained ADD tri-acc(d2)={A0[2]['tri_acc']:.2f} carry-acc(d2)={C0[2]['carry_acc']:.2f}")

    print("--- emit d8 SV node-tag dataset ---")
    out["node_tags"] = emit_node_tags(model, cfg)
    print(" ", out["node_tags"])

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json + features.json + behaviors.json")


if __name__ == "__main__":
    main()
