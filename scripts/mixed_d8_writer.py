"""Map-anchored pieces for a mixed d8 model that CE26 left MAP-BLOCKED (no verified
map on HF at the time). The map is now published + combiner-complete, so:

  writer-encoding  : does the MAPPED tri-state writer locus (Algo:ST/MT nodes)
                     linearly encode the class tri-state? (CE26 probed the Dpn site
                     and got untrained>=trained -> invalid; redo at the mapped locus.)
  writer-necessity : mean-ablate the mapped ST (add) vs MT (sub) writer heads and
                     measure the per-class accuracy drop -> class-specific necessity.
  shared-combiner  : the STC/MTC/NTC combiner nodes co-locate (one shared last-layer
                     answer MLP engine drives all three classes).

Reads roles from the ANALYSIS repo features.json (ins1_mix_d8 is analysis-repo-only,
not on the legacy flat repo). Run:
    PYTHONPATH=. python3 scripts/mixed_d8_writer.py [model_name]
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import torch

from huggingface_hub import hf_hub_download

from quanta_maths.maths_model_loader import load_maths_model_from_analysis_repo
from quanta_maths import make_untrained_control
from quanta_maths.maths_diagram import algo_task
from quanta_maths.maths_probe import sub_labels, neg_labels, probe_accuracy_with_null
from quanta_maths.maths_edge_patch import answer_positions
from quanta_maths.maths_sufficiency import make_batch, baseline_accuracy, class_question
from quanta_maths.maths_cascade import CLASS_OP, model_classes
from quanta_maths.maths_utilities import make_a_maths_question_and_answer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "ins1_mix_d8_l3_h4_t70K_s572091"
OUT_DIR = f"results/study-{MODEL}-writer"
# writer role per class (question-tail tri-state writers)
WRITER_ROLE = {"ADD": "ST", "SUB": "MT", "NEG": "NT"}


def read_roles(name):
    fea = json.load(open(hf_hub_download(f"PhilipQuirke/QuantaMaths_{name}", "features.json")))
    roles = collections.defaultdict(list)
    for n in fea:
        for t in n.get("tags", []):
            if t.startswith("Algo:"):
                roles[algo_task(t.split(":", 1)[1])].append(
                    (n["position"], n["layer"], n["is_head"], n["num"]))
    return roles


def to_q(cfg, a, b, cls):
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, CLASS_OP[cls])
    return q[0]


def tristate(cfg, a, b, cls, k):
    _, ST, _ = (neg_labels(a, b, cfg.n_digits) if cls == "NEG"
                else sub_labels(a, b, cfg.n_digits, operation=CLASS_OP[cls]))
    return int(ST[k])


# --------------------------------------------------------------------------- #
# writer-encoding: probe the class tri-state at each mapped writer (pos,layer)
# --------------------------------------------------------------------------- #

def writer_encoding(model, cfg, cls, writers, k, rng, n_q=300):
    """Best linear decode of the class tri-state ST[k] across the mapped writer
    positions (resid_post at each writer's layer)."""
    pls = sorted({(p, L) for (p, L, ih, _) in writers})
    if not pls:
        return {"best_acc": float("nan"), "note": "no mapped writers"}
    hooks = {f"blocks.{L}.hook_resid_post" for (_, L) in pls}
    acts = {pl: [] for pl in pls}
    ys = []
    for _ in range(n_q):
        a, b = class_question(cfg, rng, cls)
        with torch.no_grad():
            _, c = model.run_with_cache(to_q(cfg, a, b, cls).unsqueeze(0),
                                        names_filter=lambda nm: nm in hooks)
        for (p, L) in pls:
            acts[(p, L)].append(c[f"blocks.{L}.hook_resid_post"][0, p, :].detach().numpy())
        ys.append(tristate(cfg, a, b, cls, k))
    y = np.asarray(ys)
    if len(np.unique(y)) < 2:
        return {"best_acc": float("nan"), "note": "tri-state degenerate at k"}
    best = None
    for pl in pls:
        r = probe_accuracy_with_null(np.asarray(acts[pl]), y, rng, n_perm=20)
        if best is None or r["observed_acc"] > best["acc"]:
            best = {"acc": r["observed_acc"], "null_p": r["null_p"],
                    "pos_layer": f"P{pl[0]}L{pl[1]}"}
    return best


# --------------------------------------------------------------------------- #
# writer-necessity: mean-ablate a writer head-group, per-class accuracy drop
# --------------------------------------------------------------------------- #

def _mean_z(model, cfg, batch, layers, chunk=100):
    names = {L: f"blocks.{L}.attn.hook_z" for L in layers}
    acc = {L: None for L in layers}
    n = 0
    for i in range(0, batch.shape[0], chunk):
        with torch.no_grad():
            _, c = model.run_with_cache(batch[i:i + chunk],
                                        names_filter=lambda nm: nm in names.values())
        for L in layers:
            z = c[names[L]].sum(0)
            acc[L] = z if acc[L] is None else acc[L] + z
        n += batch[i:i + chunk].shape[0]
    return {L: acc[L] / n for L in layers}


def ablate_acc(model, cfg, clean, nodes_by_layer, means, chunk=200):
    ap = answer_positions(cfg)
    read = [p - 1 for p in ap]
    hooks = []
    for L, phs in nodes_by_layer.items():
        def zhook(act, hook, phs=phs, mz=means[L]):
            for (p, h) in phs:
                act[:, p, h, :] = mz[p, h, :]
            return act
        hooks.append((f"blocks.{L}.attn.hook_z", zhook))
    correct = 0
    for s in range(0, clean.shape[0], chunk):
        cl = clean[s:s + chunk]
        with torch.no_grad():
            lg = model.run_with_hooks(cl, fwd_hooks=hooks)
        correct += int((lg[:, read].argmax(-1) == cl[:, ap]).all(dim=1).sum())
    return correct / clean.shape[0]


def writer_necessity(model, cfg, writers_by_role, n=250):
    """For each writer role (ST/MT), mean-ablate its heads and report the per-class
    accuracy (vs baseline). Class-specific necessity = ST hurts ADD, MT hurts SUB/NEG."""
    classes = model_classes(cfg)
    out = {}
    for cls in classes:
        clean = make_batch(cfg, cls, n, np.random.default_rng(0))
        base = baseline_accuracy(model, cfg, clean)
        layers = sorted({L for role in writers_by_role for (_, L, ih, _) in writers_by_role[role] if ih})
        means = _mean_z(model, cfg, clean, layers) if layers else {}
        row = {"baseline": base}
        for role, nodes in writers_by_role.items():
            nbl = collections.defaultdict(list)
            for (p, L, ih, h) in nodes:
                if ih:
                    nbl[L].append((p, h))
            if not nbl:
                row[f"ablate_{role}"] = float("nan")
                continue
            row[f"ablate_{role}"] = ablate_acc(model, cfg, clean, dict(nbl), means)
        out[cls] = row
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_analysis_repo(MODEL, device="cpu")
    roles = read_roles(MODEL)
    kmid = cfg.n_digits // 2
    print(f"=== {MODEL}: n_digits={cfg.n_digits} n_layers={cfg.n_layers} probe-digit A{kmid} ===")

    # shared-combiner (from the fixed map)
    comb = {r: sorted({f"P{p}L{L}{'H' if ih else 'M'}{nu}" for (p, L, ih, nu) in roles.get(r, [])})
            for r in ("STC", "MTC", "NTC")}
    shared = set(comb["STC"]) & set(comb["MTC"]) & set(comb["NTC"])
    print("shared-combiner: STC=%d MTC=%d NTC=%d nodes | co-located(all 3)=%s"
          % (len(comb["STC"]), len(comb["MTC"]), len(comb["NTC"]), sorted(shared)))

    # writer-encoding at mapped nodes (trained vs untrained)
    print("--- writer-encoding at mapped writer locus (trained vs untrained) ---")
    ctrl = make_untrained_control(cfg)
    enc = {}
    for cls in model_classes(cfg):
        role = WRITER_ROLE[cls]
        writers = roles.get(role, [])
        tr = writer_encoding(model, cfg, cls, writers, kmid, np.random.default_rng(0))
        un = writer_encoding(ctrl, cfg, cls, writers, kmid, np.random.default_rng(0))
        enc[cls] = {"role": role, "n_writers": len({(p, L) for (p, L, ih, _) in writers}),
                    "trained": tr, "untrained": un}
        print(f"  {cls} ({role}): trained={tr.get('best_acc', tr.get('acc')):.2f}@{tr.get('pos_layer')} "
              f"untrained={un.get('best_acc', un.get('acc')):.2f}")

    # writer-necessity
    print("--- writer-necessity (mean-ablate mapped writers; per-class accuracy) ---")
    writers_by_role = {r: roles.get(r, []) for r in ("ST", "MT", "NT") if roles.get(r)}
    nec = writer_necessity(model, cfg, writers_by_role)
    for cls, row in nec.items():
        print(f"  {cls}: baseline={row['baseline']:.3f} | " +
              " ".join(f"ablate-{r.split('_')[1]}={row[r]:.3f}" for r in row if r.startswith("ablate_")))

    out = {"model": MODEL, "probe_digit": kmid,
           "shared_combiner": {"counts": {r: len(comb[r]) for r in comb},
                               "co_located_all3": sorted(shared)},
           "writer_encoding": enc, "writer_necessity": nec}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
