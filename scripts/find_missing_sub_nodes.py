"""Identify the SUBTRACTION-important nodes the verified map misses (CE29 follow-up).

Restoration is the group-aware complement of ablation: start from keep=map (which
retains only ~0.5 SUB / ~0.37 NEG under mean-ablation) and ADD map-omitted
(complement) nodes back, measuring recovery. Because the missing nodes are
individually redundant (that's why the ablation map omits them), we use group
restoration + cumulative top-k, not per-node ablation.

Phase 1  localize: restore the complement by structural group (layer / head-vs-MLP
         / question-vs-answer region / tagged-elsewhere-head-vs-untagged).
Phase 2  rank + minimal set: single-node restoration gain -> rank -> cumulative
         top-k recovery vs a same-size RANDOM-augment control (specificity).
Phase 3  characterize: per-answer-digit retention (target the failing 10^5 &
         hundreds digits, not the already-1.00 sign) + mean/resample bracket + seeds.

Run: PYTHONPATH=. python scripts/find_missing_sub_nodes.py [model_name]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from quanta_maths import load_maths_model_from_hf
from quanta_maths.maths_edge_patch import answer_positions
from quanta_maths.maths_sufficiency import (
    load_keep_set, masks_from_set, make_batch, ablate_accuracy, _precompute_means)

OUT_DIR = "results/study-missing-sub-nodes"
DEFAULT_MODEL = "ins1_mix_d6_l3_h4_t40K_s372001"
CLASSES = ["SUB", "NEG"]


def all_nodes(cfg):
    nL, nH, nC = cfg.n_layers, cfg.n_heads, cfg.n_ctx
    heads = {(p, l, True, h) for l in range(nL) for p in range(nC) for h in range(nH)}
    mlps = {(p, l, False, 0) for l in range(nL) for p in range(nC)}
    return heads | mlps


def acc(model, cfg, keepset, clean, source, means, method="mean"):
    hk, mk = masks_from_set(cfg, keepset)
    return ablate_accuracy(model, cfg, clean, hk, mk, method, source=source, means=means)


def perdigit_retention(model, cfg, keepset, clean, means, chunk=100):
    """Per-answer-token accuracy under mean-ablation of the complement of keepset."""
    nL = cfg.n_layers
    zn = [f"blocks.{L}.attn.hook_z" for L in range(nL)]
    mn = [f"blocks.{L}.hook_mlp_out" for L in range(nL)]
    hk, mk = masks_from_set(cfg, keepset)
    mean_z, mean_m = means
    ap = answer_positions(cfg)
    read = [p - 1 for p in ap]
    correct = np.zeros(len(ap)); total = 0
    for s in range(0, clean.shape[0], chunk):
        cl = clean[s:s + chunk]
        hooks = []
        for L in range(nL):
            def zhook(act, hook, L=L):
                return torch.where(hk[L][None, :, :, None], act, mean_z[L][None])
            def mhook(act, hook, L=L):
                return torch.where(mk[L][None, :, None], act, mean_m[L][None])
            hooks += [(zn[L], zhook), (mn[L], mhook)]
        with torch.no_grad():
            logits = model.run_with_hooks(cl, fwd_hooks=hooks)
        pred = logits[:, read].argmax(-1)
        correct += (pred == cl[:, ap]).float().sum(0).numpy()
        total += cl.shape[0]
    return (correct / total).tolist()  # [sign, A_top..A0]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    os.makedirs(OUT_DIR, exist_ok=True)
    model, cfg = load_maths_model_from_hf(model_name, device="cpu")
    keep = load_keep_set(model_name)
    comp = sorted(all_nodes(cfg) - keep)
    nL, nC = cfg.n_layers, cfg.n_ctx
    qmax = 2 * cfg.n_digits + 1  # '=' position; question region is <= qmax
    tagged_hl = {(l, h) for (p, l, ih, h) in keep if ih}  # heads tagged useful somewhere

    def group(name):
        if name.startswith("layer"):
            L = int(name[5:]); return [n for n in comp if n[1] == L]
        if name == "heads": return [n for n in comp if n[2]]
        if name == "mlps": return [n for n in comp if not n[2]]
        if name == "question_region": return [n for n in comp if n[0] <= qmax]
        if name == "answer_region": return [n for n in comp if n[0] > qmax]
        if name == "tagged_elsewhere_heads": return [n for n in comp if n[2] and (n[1], n[3]) in tagged_hl]
        if name == "untagged_heads": return [n for n in comp if n[2] and (n[1], n[3]) not in tagged_hl]
        return []

    out = {"model": model_name, "n_keep": len(keep), "n_complement": len(comp), "classes": {}}
    print(f"=== {model_name}: keep {len(keep)}, complement {len(comp)} ===")

    for cls in CLASSES:
        clean = make_batch(cfg, cls, 300, np.random.default_rng(0))
        src = make_batch(cfg, cls, 300, np.random.default_rng(1))
        means = _precompute_means(model, cfg, src)
        base = acc(model, cfg, keep, clean, src, means)
        full = acc(model, cfg, all_nodes(cfg), clean, src, means)  # ~baseline
        print(f"--- {cls}: keep-map={base:.3f}  keep-all={full:.3f} ---")

        # Phase 1: group restoration
        groups = {}
        for g in ["layer0", "layer1", "layer2", "heads", "mlps", "question_region",
                  "answer_region", "tagged_elsewhere_heads", "untagged_heads"]:
            G = group(g)
            if L := len(G):
                groups[g] = {"n": L, "restore_acc": acc(model, cfg, keep | set(G), clean, src, means)}
        for g, v in groups.items():
            print(f"    [P1] restore {g:24} (+{v['n']:3}) -> {v['restore_acc']:.3f}")

        # Phase 2: single-node restoration gain (mean), rank, cumulative top-k
        gains = []
        for c in comp:
            gains.append((acc(model, cfg, keep | {c}, clean, src, means) - base, c))
        gains.sort(reverse=True)
        ranked = [c for _, c in gains]
        cum = {}
        rng = np.random.default_rng(7)
        for k in [5, 10, 20, 40, 80, 160]:
            if k > len(ranked): continue
            topk = set(ranked[:k])
            randk = set(comp[i] for i in rng.choice(len(comp), size=k, replace=False))
            cum[k] = {"topk_acc": acc(model, cfg, keep | topk, clean, src, means),
                      "random_augment_acc": acc(model, cfg, keep | randk, clean, src, means)}
        for k, v in cum.items():
            print(f"    [P2] +top{k:3} -> {v['topk_acc']:.3f}   (random-augment {v['random_augment_acc']:.3f})")

        # Phase 3: per-digit retention (keep-map vs keep-map + top-40) + resample bracket
        top40 = set(ranked[:40])
        pd_map = perdigit_retention(model, cfg, keep, clean, means)
        pd_top = perdigit_retention(model, cfg, keep | top40, clean, means)
        resamp_top = acc(model, cfg, keep | top40, clean, src, means, method="resample")
        ap_names = ["SGN"] + [f"A{cfg.n_digits - i}" for i in range(cfg.n_digits + 1)]
        out["classes"][cls] = {
            "keep_map": base, "keep_all": full, "groups": groups, "cumulative": cum,
            "top20_nodes": [f"P{p}L{l}{'H' if ih else 'M'}{n}" for (p, l, ih, n) in ranked[:20]],
            "perdigit_keepmap": dict(zip(ap_names, pd_map)),
            "perdigit_keepmap_plus_top40": dict(zip(ap_names, pd_top)),
            "top40_resample": resamp_top,
        }
        print(f"    [P3] per-digit keep-map:     {dict(zip(ap_names, [round(x,2) for x in pd_map]))}")
        print(f"    [P3] per-digit +top40:       {dict(zip(ap_names, [round(x,2) for x in pd_top]))}")
        print(f"    [P3] top20 missing nodes: {out['classes'][cls]['top20_nodes']}")

    with open(os.path.join(OUT_DIR, f"results_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {OUT_DIR}/results_{model_name}.json")


if __name__ == "__main__":
    main()
