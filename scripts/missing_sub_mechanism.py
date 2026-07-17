"""What do the CE30 map-missed SUBTRACTION heads COMPUTE? (study-missing-sub-mechanism.md)

CE30 localized the map-omitted subtraction-important nodes to the last-layer (L2)
attention heads at the answer-producing positions of the FAILING digits
(P15L2H* = produce-pos(A5), P18L2H* = produce-pos(A2)). This script characterizes
what those heads compute with three complementary readouts:

  READ   attention profile: over random class stimuli, where (which key positions)
         does each head at cpos attend? own-operands (D_k,D'_k) vs lower-operands
         (D_{<k}: the borrow source) vs higher / eq+sgn.
  WRITE  OV-encode probe: does the head-group (and each head) OV write `z @ W_O` at
         cpos decode the borrow-IN SV[k], the tri-state ST[k], the base-difference
         SA[k], or the final digit A_k?  (balanced-acc + permutation null)
  CARRY  causal interchange on a depth-k U cascade: toggle borrow-in, patch the
         last-layer heads' OV at cpos (group + per-head), A_k flip vs deciding null.

Discriminator: SV[k] (borrow INTO k) depends on LOWER digits only; SA[k]/ST[k]
depend on digit k's OWN operands only. attend-lower + encode-SV => delivery;
attend-own + encode-SA/ST => local difference/tri-state.

Controls: untrained twin (must fail WRITE+CARRY); permutation null (WRITE);
deciding-matched null (CARRY).

Run: PYTHONPATH=. python3 scripts/missing_sub_mechanism.py [model_name]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf, make_untrained_control
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_probe import sub_labels, neg_labels, last_layer, probe_accuracy_with_null
from quanta_maths.maths_edge_patch import (
    consuming_pos, head_group_ov, attention_mass_by_group,
    run_multi_head_edge_patch, flip_rate_with_matched_null)
from quanta_maths.maths_cascade import (
    make_cascade_operands, cascade_question, cascade_answer_digit,
    combiner_delivery_flip, CLASS_OP)
from quanta_maths.maths_sufficiency import class_question, load_keep_set

OUT_DIR = "results/study-missing-sub-mechanism"
DEFAULT_MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
CLASSES = ["SUB", "NEG"]
# CE30 (d6): A5 (10^5) and A2 (hundreds) are the failing digits. On another model
# pass the failing answer-digit indices as argv[2] (comma-separated), e.g.
#   python scripts/missing_sub_mechanism.py mix_d8_l3_h4_t60K_s173289 7,3
DEFAULT_FAIL_DIGITS = [5, 2]


# --------------------------------------------------------------------------- #
# class-aware helpers
# --------------------------------------------------------------------------- #

def class_labels(cfg, a, b, cls):
    if cls == "NEG":
        return neg_labels(a, b, cfg.n_digits)
    return sub_labels(a, b, cfg.n_digits, operation=CLASS_OP[cls])


def to_q(cfg, a, b, cls):
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, CLASS_OP[cls])
    return q[0]


def _pos(name):
    return int(name[1:])


def key_groups(cfg, k):
    """Accessible (causal) key-position groups for a query at produce-pos(k)."""
    nd = cfg.n_digits
    dp = lambda j: _pos(cfg.dn_to_position_name(j))
    dpp = lambda j: _pos(cfg.ddn_to_position_name(j))
    return {
        "own_operands": [dp(k), dpp(k)],
        "lower_operands": [p for j in range(k) for p in (dp(j), dpp(j))],
        "higher_operands": [p for j in range(k + 1, nd) for p in (dp(j), dpp(j))],
        "eq_sgn": [2 * nd + 1, _pos(cfg.an_to_position_name(nd + 1))],
    }


# --------------------------------------------------------------------------- #
# READ: attention profile
# --------------------------------------------------------------------------- #

def attention_profile(model, cfg, cls, k, layer, rng, n_q=80):
    cpos = consuming_pos(cfg, k)
    groups = key_groups(cfg, k)
    qs = torch.stack([to_q(cfg, *class_question(cfg, rng, cls), cls) for _ in range(n_q)])
    return {f"H{h}": attention_mass_by_group(model, cfg, qs, cpos, layer, h, groups)
            for h in range(cfg.n_heads)}


# --------------------------------------------------------------------------- #
# WRITE: OV-encode probe
# --------------------------------------------------------------------------- #

def collect_ov(model, cfg, cls, k, layer, rng, n_q=300):
    cpos = consuming_pos(cfg, k)
    heads = list(range(cfg.n_heads))
    qs = []
    labs = {"SV_borrow_in": [], "ST_tristate": [], "SA_basediff": [], "Ak_final": []}
    for _ in range(n_q):
        a, b = class_question(cfg, rng, cls)
        qs.append(to_q(cfg, a, b, cls))
        SA, ST, SV = class_labels(cfg, a, b, cls)
        labs["SV_borrow_in"].append(int(SV[k]))
        labs["ST_tristate"].append(int(ST[k]))
        labs["SA_basediff"].append(int(SA[k]))
        labs["Ak_final"].append(cascade_answer_digit(cls, a, b, k, cfg.n_digits))
    X_group, X_head = head_group_ov(model, cfg, torch.stack(qs), cpos, layer, heads)
    labs = {t: np.asarray(v) for t, v in labs.items()}
    return X_group, X_head, labs


def probe_all(X, labs, rng, n_perm=20):
    out = {}
    for t, y in labs.items():
        nc = len(np.unique(y))
        if nc < 2:
            out[t] = {"acc": float("nan"), "chance": float("nan"), "null_p": float("nan")}
            continue
        r = probe_accuracy_with_null(X, y, rng, n_perm=n_perm)
        out[t] = {"acc": r["observed_acc"], "null_mean": r["null_mean_acc"],
                  "null_p": r["null_p"], "chance": 1.0 / nc}
    return out


# --------------------------------------------------------------------------- #
# CARRY: per-head causal interchange (group = combiner_delivery_flip lastlayer_attn)
# --------------------------------------------------------------------------- #

def perhead_delivery_flip(model, cfg, cls, k, head, layer, n_pairs=24):
    cpos = consuming_pos(cfg, k)
    a1, b1 = make_cascade_operands(cfg, cls, k, carry_in=1)
    a0, b0 = make_cascade_operands(cfg, cls, k, carry_in=0, variant=0)
    zname = f"blocks.{layer}.attn.hook_z"

    def patch_fn(sq, tq):
        with torch.no_grad():
            _, sc = model.run_with_cache(sq.unsqueeze(0), names_filter=lambda nm: nm == zname)
            _, tc = model.run_with_cache(tq.unsqueeze(0))
        patches = [(cpos, layer, head, sc[zname][0, cpos, head, :].detach().numpy())]
        return run_multi_head_edge_patch(model, cfg, tq, patches=patches,
                                         recv_layer=layer, tgt_cache=tc)

    def pair_builder():
        return cascade_question(cfg, cls, a1, b1), cascade_question(cfg, cls, a0, b0)

    def null_builder():
        an, bn = make_cascade_operands(cfg, cls, k, carry_in=0, variant=1)
        return cascade_question(cfg, cls, an, bn), cascade_question(cfg, cls, a0, b0)

    return flip_rate_with_matched_null(model, cfg, pair_builder, patch_fn,
                                       answer_digit=k, n_pairs=n_pairs, null_builder=null_builder)


# --------------------------------------------------------------------------- #
# map cross-check
# --------------------------------------------------------------------------- #

def map_crosscheck(model_name, cfg):
    ll = cfg.n_layers - 1
    tagged = {}
    try:
        keep = load_keep_set(model_name)
        for (p, l, ih, num) in keep:
            if ih and l == ll:
                tagged.setdefault(num, []).append(p)
    except Exception as e:
        tagged = {"_error": str(e)}
    roles = {}
    try:
        from quanta_maths.maths_diagram import capture_role_registry
        _, reg = capture_role_registry(model_name)
        # only last-layer head roles are relevant here
        roles = {t: [loc for loc in locs if f"L{ll}H" in loc] for t, locs in reg["roles"].items()}
        roles = {t: v for t, v in roles.items() if v}
    except Exception as e:
        roles = {"_error": str(e)}
    return {"ll_head_tagged_positions": {f"H{h}": sorted(tagged.get(h, []))
                                         for h in range(cfg.n_heads)} if isinstance(tagged, dict)
            and "_error" not in tagged else tagged,
            "ll_head_roles": roles}


# --------------------------------------------------------------------------- #

def run_class_digit(model, cfg, cls, k, seed=0, n_read=80, n_write=300, n_pairs=24):
    ll = last_layer(cfg)
    heads = list(range(cfg.n_heads))
    rng = np.random.default_rng(seed)
    cpos = consuming_pos(cfg, k)

    read = attention_profile(model, cfg, cls, k, ll, rng, n_q=n_read)
    Xg, Xh, labs = collect_ov(model, cfg, cls, k, ll, rng, n_q=n_write)
    write = {"group": probe_all(Xg, labs, rng),
             "per_head": {f"H{h}": probe_all(Xh[h], labs, rng) for h in heads}}
    group_flip = combiner_delivery_flip(model, cfg, cls, k, arm="lastlayer_attn", n_pairs=n_pairs)
    carry = {"group": {"flip": group_flip["flip"]["rate"], "null": group_flip["null"]["rate"],
                       "valid": group_flip["stimulus_valid"]},
             "per_head": {}}
    for h in heads:
        r = perhead_delivery_flip(model, cfg, cls, k, h, ll, n_pairs=n_pairs)
        carry["per_head"][f"H{h}"] = {"flip": r["flip"]["rate"], "null": r["null"]["rate"]}
    return {"cpos": cpos, "READ": read, "WRITE": write, "CARRY": carry}


def _fmt_write(w):
    return {t: round(w[t]["acc"], 2) for t in w if not np.isnan(w[t]["acc"])}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    fail_digits = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
                   else DEFAULT_FAIL_DIGITS)
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(model_name, device="cpu")
    ll = last_layer(cfg)
    print(f"=== {model_name}: last layer L{ll}, heads {cfg.n_heads}, fail_digits={fail_digits} ===")

    out = {"model": model_name, "last_layer": ll, "fail_digits": fail_digits,
           "map_crosscheck": map_crosscheck(model_name, cfg), "classes": {}}
    print("  map: last-layer head tagged positions:",
          out["map_crosscheck"]["ll_head_tagged_positions"])
    print("  map: last-layer head roles:", out["map_crosscheck"]["ll_head_roles"])

    for cls in CLASSES:
        out["classes"][cls] = {}
        for k in fail_digits:
            r = run_class_digit(model, cfg, cls, k)
            out["classes"][cls][f"A{k}"] = r
            cp = r["cpos"]
            print(f"\n--- {cls} A{k} (produce-pos P{cp}) ---")
            for h in range(cfg.n_heads):
                rd = r["READ"][f"H{h}"]
                print(f"    READ  H{h}: own={rd['own_operands']:.2f} low={rd['lower_operands']:.2f} "
                      f"hi={rd['higher_operands']:.2f} eq/sgn={rd['eq_sgn']:.2f} self={rd['self']:.2f}")
            print(f"    WRITE group: {_fmt_write(r['WRITE']['group'])}")
            for h in range(cfg.n_heads):
                print(f"    WRITE H{h}:   {_fmt_write(r['WRITE']['per_head'][f'H{h}'])}")
            cg = r["CARRY"]["group"]
            print(f"    CARRY group: flip={cg['flip']:.2f} null={cg['null']:.2f} valid={cg['valid']}")
            print("    CARRY per-head:",
                  {h: f"{r['CARRY']['per_head'][h]['flip']:.2f}/{r['CARRY']['per_head'][h]['null']:.2f}"
                   for h in r["CARRY"]["per_head"]})

    # ---- untrained control (must fail WRITE + CARRY) on (SUB, first fail digit) ----
    print("\n=== untrained control (must fail WRITE+CARRY) ===")
    kc = fail_digits[0]
    ctrl = make_untrained_control(cfg)
    rng = np.random.default_rng(0)
    Xg, _, labs = collect_ov(ctrl, cfg, "SUB", kc, ll, rng, n_q=300)
    cw = probe_all(Xg, labs, rng)
    cflip = combiner_delivery_flip(ctrl, cfg, "SUB", kc, arm="lastlayer_attn", n_pairs=24)
    out["untrained_control"] = {"WRITE_group": cw,
                                "CARRY_group": {"flip": cflip["flip"]["rate"],
                                                "null": cflip["null"]["rate"]}}
    print(f"  WRITE group: {_fmt_write(cw)}  (chance SV=0.5, SA/Ak=0.1)")
    print(f"  CARRY group: flip={cflip['flip']['rate']:.2f} null={cflip['null']['rate']:.2f}")

    path = os.path.join(OUT_DIR, f"results_{model_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
