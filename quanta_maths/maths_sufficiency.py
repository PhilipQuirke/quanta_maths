"""Circuit-sufficiency test (model-general): keep ONLY the map-useful nodes and
destroy the position-specific complement (resample- or mean-ablation from
same-class inputs); measure retained accuracy. The sufficiency complement to the
per-node ablation-necessity evidence.

Keep-set = the model's published `behavior.json` useful nodes. Complement = every
`(pos, layer, head)` / `(pos, layer, MLP)` not in the keep-set. Embeddings /
positional / LayerNorm / unembed are always kept (not "nodes"). A keep-random
control (same size + head/MLP composition) must fail, proving the useful set is
specifically load-bearing.

CPU-friendly; batched. Reusable across the ~40-model zoo.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from quanta_maths.maths_constants import MathsToken
from quanta_maths.maths_utilities import make_a_maths_question_and_answer
from quanta_maths.maths_edge_patch import answer_positions
from quanta_maths.maths_cascade import CLASS_OP, model_classes
from quanta_maths.maths_model_loader import DEFAULT_HF_REPO


# --------------------------------------------------------------------------- #
# keep-set + masks
# --------------------------------------------------------------------------- #

def load_keep_set(model_name: str, hf_repo: str = DEFAULT_HF_REPO) -> set:
    """The set of map-useful nodes `(position, layer, is_head, num)` from the
    model's published `<name>_behavior.json` (each listed node is a useful node)."""
    from QuantaMechInterp.model_train_json import download_huggingface_json
    beh = download_huggingface_json(hf_repo, f"{model_name}_behavior.json")
    return set((n["position"], n["layer"], n["is_head"], n["num"]) for n in beh)


def masks_from_set(cfg, keep: set):
    """(head_keep, mlp_keep): per-layer boolean masks, True == keep clean."""
    nL, nH, nC = cfg.n_layers, cfg.n_heads, cfg.n_ctx
    head = [torch.zeros((nC, nH), dtype=torch.bool) for _ in range(nL)]
    mlp = [torch.zeros(nC, dtype=torch.bool) for _ in range(nL)]
    for (p, l, is_head, num) in keep:
        if l >= nL or p >= nC:
            continue
        if is_head:
            if num < nH:
                head[l][p, num] = True
        else:
            mlp[l][p] = True
    return head, mlp


def random_keep_set(cfg, n_heads: int, n_mlps: int, rng) -> set:
    """A random keep-set with the same head/MLP counts (the control)."""
    nL, nH, nC = cfg.n_layers, cfg.n_heads, cfg.n_ctx
    all_heads = [(p, l, True, h) for l in range(nL) for p in range(nC) for h in range(nH)]
    all_mlps = [(p, l, False, 0) for l in range(nL) for p in range(nC)]
    hi = rng.choice(len(all_heads), size=min(n_heads, len(all_heads)), replace=False)
    mi = rng.choice(len(all_mlps), size=min(n_mlps, len(all_mlps)), replace=False)
    return set(all_heads[i] for i in hi) | set(all_mlps[i] for i in mi)


# --------------------------------------------------------------------------- #
# stimuli
# --------------------------------------------------------------------------- #

def class_question(cfg, rng, cls: str):
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


def make_batch(cfg, cls: str, n: int, rng) -> torch.Tensor:
    op = CLASS_OP[cls]
    qs = torch.zeros((n, cfg.n_ctx), dtype=torch.int64)
    for i in range(n):
        a, b = class_question(cfg, rng, cls)
        make_a_maths_question_and_answer(cfg, qs[i:i + 1], 0, a, b, op)
    return qs


# --------------------------------------------------------------------------- #
# ablation-accuracy
# --------------------------------------------------------------------------- #

def _precompute_means(model, cfg, source, chunk=100):
    nL = cfg.n_layers
    zn = [f"blocks.{L}.attn.hook_z" for L in range(nL)]
    mn = [f"blocks.{L}.hook_mlp_out" for L in range(nL)]
    zsum, msum, n = [None] * nL, [None] * nL, 0
    for s in range(0, source.shape[0], chunk):
        with torch.no_grad():
            _, c = model.run_with_cache(source[s:s + chunk],
                                        names_filter=lambda nm: nm in zn or nm in mn)
        for L in range(nL):
            z, m = c[zn[L]].sum(0), c[mn[L]].sum(0)
            zsum[L] = z if zsum[L] is None else zsum[L] + z
            msum[L] = m if msum[L] is None else msum[L] + m
        n += source[s:s + chunk].shape[0]
    return [z / n for z in zsum], [m / n for m in msum]


def baseline_accuracy(model, cfg, clean, chunk=200) -> float:
    ap = answer_positions(cfg)
    read = [p - 1 for p in ap]
    correct = 0
    for s in range(0, clean.shape[0], chunk):
        cl = clean[s:s + chunk]
        with torch.no_grad():
            logits = model(cl)
        correct += int((logits[:, read].argmax(-1) == cl[:, ap]).all(dim=1).sum())
    return correct / clean.shape[0]


def ablate_accuracy(model, cfg, clean, head_keep, mlp_keep, method="resample",
                    source=None, means=None, chunk=100) -> float:
    """Fraction of `clean` questions fully correct while the COMPLEMENT of
    (head_keep, mlp_keep) is destroyed. method='resample' overwrites each complement
    node with the paired `source` question's activation; method='mean' overwrites
    with the source-distribution mean (`means`=(mean_z, mean_mlp))."""
    nL = cfg.n_layers
    zn = [f"blocks.{L}.attn.hook_z" for L in range(nL)]
    mn = [f"blocks.{L}.hook_mlp_out" for L in range(nL)]
    ap = answer_positions(cfg)
    read = [p - 1 for p in ap]
    correct, total = 0, 0
    for s in range(0, clean.shape[0], chunk):
        cl = clean[s:s + chunk]
        if method == "resample":
            with torch.no_grad():
                _, src = model.run_with_cache(source[s:s + chunk],
                                              names_filter=lambda nm: nm in zn or nm in mn)
            rz = {L: src[zn[L]] for L in range(nL)}
            rm = {L: src[mn[L]] for L in range(nL)}
        else:
            mean_z, mean_m = means
            rz = {L: mean_z[L][None] for L in range(nL)}
            rm = {L: mean_m[L][None] for L in range(nL)}
        hooks = []
        for L in range(nL):
            def zhook(act, hook, L=L, rz=rz):
                return torch.where(head_keep[L][None, :, :, None], act, rz[L])
            def mhook(act, hook, L=L, rm=rm):
                return torch.where(mlp_keep[L][None, :, None], act, rm[L])
            hooks += [(zn[L], zhook), (mn[L], mhook)]
        with torch.no_grad():
            logits = model.run_with_hooks(cl, fwd_hooks=hooks)
        correct += int((logits[:, read].argmax(-1) == cl[:, ap]).all(dim=1).sum())
        total += cl.shape[0]
    return correct / total


def circuit_sufficiency(model, cfg, model_name: str, hf_repo: str = DEFAULT_HF_REPO,
                        classes: Optional[List[str]] = None, n: int = 300,
                        n_random: int = 3) -> dict:
    """Full sufficiency test for a mapped model. Returns per-class
    {baseline, keep_useful_mean, keep_useful_resample, keep_random_mean}. High
    keep_useful_mean (≈ baseline) AND keep_random_mean ≈ 0 ⇒ the map-useful circuit
    is sufficient and specific; a low keep_useful_mean localizes map incompleteness.
    """
    keep = load_keep_set(model_name, hf_repo)
    nh = sum(1 for (_, _, ih, _) in keep if ih)
    nm = sum(1 for (_, _, ih, _) in keep if not ih)
    head_keep, mlp_keep = masks_from_set(cfg, keep)
    classes = classes or model_classes(cfg)
    out = {"model": model_name, "keep_heads": nh, "keep_mlps": nm,
           "total_heads": cfg.n_layers * cfg.n_ctx * cfg.n_heads,
           "total_mlps": cfg.n_layers * cfg.n_ctx, "classes": {}}
    for cls in classes:
        clean = make_batch(cfg, cls, n, np.random.default_rng(0))
        source = make_batch(cfg, cls, n, np.random.default_rng(1))
        means = _precompute_means(model, cfg, source)
        rand_res, rand_mean = [], []
        for seed in range(n_random):
            rk = random_keep_set(cfg, nh, nm, np.random.default_rng(100 + seed))
            hk, mk = masks_from_set(cfg, rk)
            rand_res.append(ablate_accuracy(model, cfg, clean, hk, mk, "resample", source=source))
            rand_mean.append(ablate_accuracy(model, cfg, clean, hk, mk, "mean", means=means))
        out["classes"][cls] = {
            "baseline": baseline_accuracy(model, cfg, clean),
            "keep_useful_mean": ablate_accuracy(model, cfg, clean, head_keep, mlp_keep, "mean", means=means),
            "keep_useful_resample": ablate_accuracy(model, cfg, clean, head_keep, mlp_keep, "resample", source=source),
            # matched control under BOTH ablation methods (skeptic-requested):
            "keep_random_mean": float(np.mean(rand_mean)),
            "keep_random_resample": float(np.mean(rand_res)),
        }
    return out
