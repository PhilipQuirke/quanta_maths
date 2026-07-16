"""Entry 2(a): Mixed-model SV-implementation batteries (CE16/CE17) per class.

Tests the mechanism claims the addition SV account rests on, on the mixed model
`ins1_mix_d6_l3_h4_t40K_s372001`, for ADD / SUB / NEG:

  B1 step-combiner (CE17 / A10 iv) -- on-manifold alpha-sweep of the combiner
     input; STEP if A_k jumps at a threshold with endpoints gated.
  B2 class-necessity (CE16 iii / A6) -- mean-ablate the class's map-named
     tri-state writers; cascade accuracy collapses & carry-free spared, over an
     untagged-head baseline.
  B3 =-not-a-source (CE16 ii) -- patch the '=' residual into a middle-digit
     target; A_k flip ~0 means '=' is not the middle-digit value source.
  B4 message-canonical (CE16 i) -- resolved-carry probe transfer across digits.

Reuses the class helpers from mixed_sv.py and the M0 map registry. Evidence
integrity: numbers -> results/study-mixed-sv-impl/results.json.
Run: PYTHONPATH=. python scripts/mixed_sv_impl.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf
from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_probe import (
    first_layer, last_layer, sub_labels, neg_labels,
    fit_probe, probe_balanced_accuracy, balance_idx,
)
from quanta_maths.maths_edge_patch import answer_positions, consuming_pos

from mixed_sv import (  # same-dir import (scripts/ is on sys.path when run directly)
    class_question, to_q, _edge_pair_digits, _clean_Ak, OP, CLASSES,
)

MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
OUT_DIR = "results/study-mixed-sv-impl"
MAP = "results/study-mixed-map/results.json"


def _resid_at(model, cfg, q, hook, pos):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
    return c[hook][0, pos, :].clone()


# ===========================================================================
# B1 step-combiner (CE17 / A10 iv)
# ===========================================================================

def battery_step(model, cfg, cls, k=2, n_axis=24):
    ll = last_layer(cfg)
    pp = consuming_pos(cfg, k)
    hook = f"blocks.{ll}.hook_resid_mid"
    ap = answer_positions(cfg); idx = len(ap) - 1 - k

    a0, b0 = _edge_pair_digits(cfg, cls, k, 0)
    a1, b1 = _edge_pair_digits(cfg, cls, k, 1)
    q0, q1 = to_q(cfg, a0, b0, cls), to_q(cfg, a1, b1, cls)
    Ak0, Ak1 = _clean_Ak(model, cfg, q0, k), _clean_Ak(model, cfg, q1, k)
    r0 = _resid_at(model, cfg, q0, hook, pp)
    r1 = _resid_at(model, cfg, q1, hook, pp)

    # independent carry axis: mean (r1-r0) over re-randomised far filler
    diffs = []
    rng = np.random.default_rng(0)
    for _ in range(n_axis):
        # keep digit k a U and toggle carry-in; leave the rest to _edge_pair_digits
        aa0, bb0 = _edge_pair_digits(cfg, cls, k, 0)
        aa1, bb1 = _edge_pair_digits(cfg, cls, k, 1)
        d = (_resid_at(model, cfg, to_q(cfg, aa1, bb1, cls), hook, pp)
             - _resid_at(model, cfg, to_q(cfg, aa0, bb0, cls), hook, pp))
        diffs.append(d.numpy())
    axis = np.mean(diffs, 0)
    axis_u = axis / (np.linalg.norm(axis) + 1e-9)

    def patch_read(vec):
        def hk(act, hook):
            act[:, pp, :] = vec
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(q0.unsqueeze(0), fwd_hooks=[(hook, hk)])
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    sweep, proj = {}, {}
    for al in grid:
        r = r0 + al * (r1 - r0)
        sweep[al] = patch_read(r)
        proj[al] = float(np.dot((r - r0).numpy(), axis_u))
    endpoints_ok = (sweep[0.0] == Ak0) and (sweep[1.0] == Ak1)
    # step threshold: first alpha whose A_k == Ak1 (the resolved-1 readout)
    astar = next((al for al in grid if sweep[al] == Ak1), None)
    return {"Ak0": Ak0, "Ak1": Ak1, "sweep": sweep, "carry_proj": proj,
            "endpoints_ok": endpoints_ok, "alpha_star": astar}


# ===========================================================================
# B2 class-necessity (CE16 iii / A6)
# ===========================================================================

WRITER_TASKS = {"ADD": ["ST"], "SUB": ["MT"], "NEG": ["MT", "GT"]}


def _writer_heads(registry, cls):
    heads = []
    for task in WRITER_TASKS[cls]:
        for e in registry["roles"].get(task, []):
            if e["is_head"]:
                heads.append((e["position"], e["layer"], e["num"]))
    return sorted(set(heads))


def _all_tagged_heads(registry):
    tagged = set()
    for task, entries in registry["roles"].items():
        for e in entries:
            if e["is_head"]:
                tagged.add((e["position"], e["layer"], e["num"]))
    return tagged


def _cascade_question(cfg, rng, cls, cascade):
    """cascade=True -> long carry/borrow chain; False -> carry/borrow-free."""
    nd = cfg.n_digits
    if cls == "ADD":
        if cascade:  # 9..9 + small -> full carry chain
            a = int("9" * nd); b = int(rng.integers(1, 10))
        else:        # both digits <=4 -> no carry
            a = int("".join(str(rng.integers(0, 5)) for _ in range(nd)))
            b = int("".join(str(rng.integers(0, 5)) for _ in range(nd)))
        return a, b
    # SUB/NEG: borrow chain when minuend has low digits over high subtrahend
    if cascade:
        hi = int("9" * nd)
        lo = int("1" + "0" * (nd - 1))  # 10..0 -> borrow chain against 9..9-ish
        if cls == "SUB":
            return hi, hi - lo  # positive, long borrow
        return hi - lo, hi      # negative, long borrow
    else:  # no borrow: each minuend digit >= subtrahend digit (SUB) / <= (NEG)
        da = [rng.integers(5, 10) for _ in range(nd)]
        db = [rng.integers(0, 5) for _ in range(nd)]
        a = int("".join(map(str, da))); b = int("".join(map(str, db)))
        return (a, b) if cls == "SUB" else (b, a)


def _acc_under_ablation(model, cfg, cls, heads, rng, n=60, cascade=True, digits_only=False):
    ap = answer_positions(cfg)
    read = ap[1:] if digits_only else ap  # digits_only excludes the SGN token
    layers = sorted(set(l for _, l, _ in heads))
    by_layer = {l: [(p, h) for p, ll_, h in heads if ll_ == l] for l in layers}

    def hooks():
        fns = []
        for l in layers:
            def mk(ph):
                def hk(act, hook):
                    for (p, h) in ph:
                        act[:, p, h, :] = 0.0
                    return act
                return hk
            fns.append((f"blocks.{l}.attn.hook_z", mk(by_layer[l])))
        return fns

    hit = 0
    for _ in range(n):
        a, b = _cascade_question(cfg, rng, cls, cascade)
        q = to_q(cfg, a, b, cls)
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=hooks())
        pred = lg[0, [p - 1 for p in read]].argmax(-1)
        hit += int((pred == q[read]).all())
    return hit / n


def _acc_clean(model, cfg, cls, rng, n=60, cascade=True, digits_only=False):
    ap = answer_positions(cfg)
    read = ap[1:] if digits_only else ap
    hit = 0
    for _ in range(n):
        a, b = _cascade_question(cfg, rng, cls, cascade)
        q = to_q(cfg, a, b, cls)
        with torch.no_grad():
            lg = model(q.unsqueeze(0))
        hit += int((lg[0, [p - 1 for p in read]].argmax(-1) == q[read]).all())
    return hit / n


def battery_necessity(model, cfg, cls, registry, rng):
    heads = _writer_heads(registry, cls)
    # truly-untagged baseline: tail L0 heads that carry NO Algo tag at all.
    tagged_all = _all_tagged_heads(registry)
    all_tail = [(p, 0, h) for p in range(cfg.n_digits, 2 * cfg.n_digits + 2)
                for h in range(cfg.n_heads)]
    untagged = [x for x in all_tail if x not in tagged_all]
    rng.shuffle(untagged)
    untagged = untagged[:max(len(heads), 4)]
    # digit-only accuracy (exclude SGN) isolates cascade necessity from sign necessity
    out = {"writer_heads": [f"P{p}L{l}H{h}" for p, l, h in heads],
           "untagged_heads": [f"P{p}L{l}H{h}" for p, l, h in untagged]}
    for do, tag in [(True, "digits"), (False, "full")]:
        out[tag] = {
            "clean_cascade": _acc_clean(model, cfg, cls, np.random.default_rng(1), cascade=True, digits_only=do),
            "clean_carryfree": _acc_clean(model, cfg, cls, np.random.default_rng(2), cascade=False, digits_only=do),
            "ablW_cascade": _acc_under_ablation(model, cfg, cls, heads, np.random.default_rng(1), cascade=True, digits_only=do),
            "ablW_carryfree": _acc_under_ablation(model, cfg, cls, heads, np.random.default_rng(2), cascade=False, digits_only=do),
            "ablUntag_cascade": _acc_under_ablation(model, cfg, cls, untagged, np.random.default_rng(1), cascade=True, digits_only=do),
        }
    return out


# ===========================================================================
# B3 =-not-a-source (CE16 ii)  and  B4 message-canonical (CE16 i)
# ===========================================================================

def battery_eq_source(model, cfg, cls, k=2):
    ll = last_layer(cfg)
    eq_pos = 2 * cfg.n_digits + 1  # '=' position
    pp = consuming_pos(cfg, k)
    ap = answer_positions(cfg); idx = len(ap) - 1 - k
    a1, b1 = _edge_pair_digits(cfg, cls, k, 1)
    a0, b0 = _edge_pair_digits(cfg, cls, k, 0)
    q1, q0 = to_q(cfg, a1, b1, cls), to_q(cfg, a0, b0, cls)
    Ak0 = _clean_Ak(model, cfg, q0, k)

    def patch(hook, pos):
        src = _resid_at(model, cfg, q1, hook, pos)
        tgt = _resid_at(model, cfg, q0, hook, pos)
        delta = src - tgt

        def hk(act, hook):
            act[:, pos, :] = act[:, pos, :] + delta
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(q0.unsqueeze(0), fwd_hooks=[(hook, hk)])
        return int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx])

    eq_flip = int(patch(f"blocks.{first_layer(cfg)+1}.hook_resid_post", eq_pos) != Ak0) \
        if cfg.n_layers >= 2 else 0
    combiner_flip = int(patch(f"blocks.{ll}.hook_resid_mid", pp) != Ak0)  # positive control
    return {"eq_source_flip": eq_flip, "combiner_input_flip": combiner_flip}


def battery_canonical(model, cfg, cls, rng, k_train=2, k_test=3, n_q=350):
    ll = last_layer(cfg)
    hook = f"blocks.{ll}.hook_resid_mid"
    Xa = {k_train: [], k_test: []}
    ya = {k_train: [], k_test: []}
    for _ in range(n_q):
        a, b = class_question(cfg, rng, cls)
        SA, ST, SV = (neg_labels(a, b, cfg.n_digits) if cls == "NEG"
                      else sub_labels(a, b, cfg.n_digits, operation=OP[cls]))
        q = to_q(cfg, a, b, cls)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == hook)
        for kk in (k_train, k_test):
            Xa[kk].append(c[hook][0, consuming_pos(cfg, kk), :].numpy())
            ya[kk].append(int(SV[kk]))
    Xtr, ytr = np.asarray(Xa[k_train]), np.asarray(ya[k_train])
    Xte, yte = np.asarray(Xa[k_test]), np.asarray(ya[k_test])
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {"transfer_acc": float("nan"), "self_acc": float("nan")}
    bi = balance_idx(ytr, rng)
    clf = fit_probe(Xtr[bi], ytr[bi])
    transfer = probe_balanced_accuracy(clf, Xte, yte)
    bj = balance_idx(yte, rng)
    self_clf = fit_probe(Xte[bj], yte[bj])
    self_acc = probe_balanced_accuracy(self_clf, Xte, yte)
    return {"transfer_acc": transfer, "self_acc": self_acc}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    registry = json.load(open(MAP))
    model, cfg = load_maths_model_from_hf(MODEL, device="cpu")
    out = {"model": MODEL}
    for cls in CLASSES:
        print(f"--- {cls} ---")
        b1 = battery_step(model, cfg, cls)
        b2 = battery_necessity(model, cfg, cls, registry, np.random.default_rng(0))
        b3 = battery_eq_source(model, cfg, cls)
        b4 = battery_canonical(model, cfg, cls, np.random.default_rng(3))
        out[cls] = {"B1_step": b1, "B2_necessity": b2, "B3_eq_source": b3, "B4_canonical": b4}
        print(f"  B1 step: A_k {b1['Ak0']}->{b1['Ak1']} endpoints_ok={b1['endpoints_ok']} "
              f"sweep={b1['sweep']} alpha*={b1['alpha_star']}")
        print(f"  B2 necessity writers={b2['writer_heads']}")
        for tag in ("digits", "full"):
            d = b2[tag]
            print(f"     [{tag}] cascade clean={d['clean_cascade']:.2f} ablW={d['ablW_cascade']:.2f} "
                  f"ablUntag={d['ablUntag_cascade']:.2f} | carryfree clean={d['clean_carryfree']:.2f} "
                  f"ablW={d['ablW_carryfree']:.2f}")
        print(f"  B3 eq_source_flip={b3['eq_source_flip']} (combiner-input control flip={b3['combiner_input_flip']})")
        print(f"  B4 canonical transfer={b4['transfer_acc']:.2f} self={b4['self_acc']:.2f}")
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
