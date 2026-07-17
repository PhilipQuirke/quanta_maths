"""Edge / OV-path patching for maths models (Stage 3 migration).

QuantaMechInterp's ``a_run_attention_intervention`` patches a whole head's ``z``
into a clean run and measures per-digit answer impact. It does NOT isolate a
sender->receiver *edge*, the OV path of a single head, or an attention *pattern*.
Those primitives are the backbone of CE2 (operand-fetch value path = linear
transport), CE5 (U-combiner = answer-position L1 MLP), and CE8 (hybrid attention
routing). They were hand-rolled in cascade_handoff_edge_patch.py, sv_compounding.py
and deep_cascade_mechanism.py; this module owns them once.

Per the migration decision this subject-specific layer lives in ``quanta_maths``
(not pushed down into QuantaMechInterp). It reuses TransformerLens hooks directly.

All functions are layer-parameterized (no hardcoded ``blocks.1``) so they work on
2-, 3- and 4-layer models. CPU-friendly.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


# ===========================================================================
# position helpers (delegate to MathsConfig algebra)
# ===========================================================================

def answer_positions(cfg) -> List[int]:
    """Token positions that hold the answer (sign + digits), top..units order."""
    na = cfg.n_digits + 2
    return list(range(cfg.n_ctx - na, cfg.n_ctx))


def consuming_pos(cfg, k: int) -> int:
    """Position whose logits PRODUCE answer digit A_k (= pos(A_k) - 1)."""
    return int(cfg.an_to_position_name(k)[1:]) - 1


def _read_answer(model, cfg, logits) -> torch.Tensor:
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)


# ===========================================================================
# LN-fair helper
# ===========================================================================

def ln_scale(vec: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(centered, std)`` for a d_model vector under LayerNorm."""
    xm = vec - vec.mean()
    std = torch.sqrt(xm.var(unbiased=False) + eps)
    return xm, std


# ===========================================================================
# OV path
# ===========================================================================

def head_ov(model, z_row: torch.Tensor, layer: int, head: int) -> torch.Tensor:
    """Map a head's ``z`` output through its output matrix: ``z @ W_O[layer,head]``.

    This is the head's contribution to the residual stream (its OV/output path).
    """
    return z_row @ model.blocks[layer].attn.W_O[head]


def head_group_ov(model, cfg, questions: torch.Tensor, cpos: int, layer: int,
                  heads: Sequence[int], chunk: int = 64):
    """Per-question OV writes (``z @ W_O``) at position ``cpos`` for a set of heads.

    Returns ``(group, per_head)`` where ``group`` is ``[n, d_model]`` (the sum of the
    heads' OV writes) and ``per_head[h]`` is ``[n, d_model]``, as numpy arrays. This
    is the "what does a head-group write into the residual at this position" feature
    collector (probe it for the quantity a consumer head delivers). Model-general,
    layer-parameterized; batched. ``questions`` is ``[n, n_ctx]``.
    """
    zname = f"blocks.{layer}.attn.hook_z"
    group_parts, head_parts = [], {h: [] for h in heads}
    for s in range(0, questions.shape[0], chunk):
        with torch.no_grad():
            _, c = model.run_with_cache(questions[s:s + chunk],
                                        names_filter=lambda nm: nm == zname)
        z = c[zname][:, cpos]  # [b, head, d_head]
        g = None
        for h in heads:
            ov = z[:, h] @ model.blocks[layer].attn.W_O[h]  # [b, d_model]
            head_parts[h].append(ov.detach().cpu().numpy())
            g = ov if g is None else g + ov
        group_parts.append(g.detach().cpu().numpy())
    group = np.concatenate(group_parts, 0)
    per_head = {h: np.concatenate(head_parts[h], 0) for h in heads}
    return group, per_head


def attention_mass_by_group(model, cfg, questions: torch.Tensor, query_pos: int,
                            layer: int, head: int, key_groups: dict,
                            chunk: int = 64) -> dict:
    """Mean attention mass from ``(layer, head, query_pos)`` onto named key-position
    groups, averaged over a batch of ``questions`` ``[n, n_ctx]``.

    ``key_groups`` maps ``name -> [key positions]`` (should be disjoint and exclude
    ``query_pos``). Returns ``{name: mean_mass, 'self': mass_on_query_pos,
    'other': remainder}`` (a distribution summing to ~1). Use to decide WHAT a
    consumer head reads (e.g. own-operands vs lower-operands = the borrow source).
    Model-general, layer-parameterized; batched.
    """
    name = f"blocks.{layer}.attn.hook_pattern"
    n = questions.shape[0]
    tot = {g: 0.0 for g in key_groups}
    tot["self"] = 0.0
    for s in range(0, n, chunk):
        with torch.no_grad():
            _, c = model.run_with_cache(questions[s:s + chunk],
                                        names_filter=lambda nm: nm == name)
        rows = c[name][:, head, query_pos, :]  # [b, key]
        tot["self"] += float(rows[:, query_pos].sum())
        for g, ps in key_groups.items():
            if ps:
                tot[g] += float(rows[:, list(ps)].sum())
    out = {g: tot[g] / n for g in tot}
    out["other"] = max(0.0, 1.0 - sum(out.values()))
    return out


# ===========================================================================
# Head -> resid edge delta + patch (raw / LN-fair / MLP-only arms)
# ===========================================================================

def head_edge_delta(model, src_cache, tgt_cache, cpos: int, layer: int, head: int) -> torch.Tensor:
    """The (head -> resid_mid) edge delta at ``cpos``: ``(z_src - z_tgt) @ W_O``.

    This is the change the receiver would see if ONLY this head's message on this
    edge were swapped from the target to the source question.
    """
    zs = src_cache[f"blocks.{layer}.attn.hook_z"][0, cpos, head, :]
    zt = tgt_cache[f"blocks.{layer}.attn.hook_z"][0, cpos, head, :]
    return (zs - zt) @ model.blocks[layer].attn.W_O[head]


def direct_path_delta(src_cache, tgt_cache, cpos: int, layer: int = 0, scale: float = 1.0) -> torch.Tensor:
    """The direct (resid_post of ``layer`` -> next resid_mid) edge delta at ``cpos``."""
    ds = src_cache[f"blocks.{layer}.hook_resid_post"][0, cpos, :]
    dt = tgt_cache[f"blocks.{layer}.hook_resid_post"][0, cpos, :]
    return (ds - dt) * scale


def run_edge_patch(model, cfg, target_q: torch.Tensor, cpos: int, delta: torch.Tensor,
                   recv_layer: int, arm: str = "raw", tgt_cache=None) -> torch.Tensor:
    """Add ``delta`` to the receiver-layer residual at ``cpos`` and predict.

    arm:
      * ``raw``      -> patch ``blocks.{recv_layer}.hook_resid_mid`` (MLP + skip both
                        see delta; ln2 renormalizes).
      * ``lnfair``   -> patch ``ln2.hook_normalized`` with delta rescaled by the CLEAN
                        std (removes LN whole-vector renormalization damping).
      * ``mlp_only`` -> patch the MLP input path only; the direct skip stays clean, so
                        (raw - mlp_only) isolates the skip contribution.
    """
    L = recv_layer
    if arm == "raw":
        def hook(act, hook):
            act[:, cpos, :] = act[:, cpos, :] + delta
            return act
        fwd = [(f"blocks.{L}.hook_resid_mid", hook)]
    elif arm in ("lnfair", "mlp_only"):
        if tgt_cache is None:
            raise ValueError(f"arm {arm!r} needs tgt_cache")
        rm = tgt_cache[f"blocks.{L}.hook_resid_mid"][0, cpos, :]
        xm, std = ln_scale(rm)
        gamma = model.blocks[L].ln2.w
        norm_clean = xm / std
        xm2 = (rm + delta) - (rm + delta).mean()
        norm_fair = xm2 / std  # freeze std to clean (LN-fair)
        add = (norm_fair - norm_clean) * gamma
        def hook(act, hook):
            act[:, cpos, :] = act[:, cpos, :] + add
            return act
        fwd = [(f"blocks.{L}.ln2.hook_normalized", hook)]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=fwd)
    return _read_answer(model, cfg, logits)


def run_multi_head_edge_patch(model, cfg, target_q: torch.Tensor,
                              patches: Sequence[Tuple[int, int, int, np.ndarray]],
                              recv_layer: int, tgt_cache) -> torch.Tensor:
    """Patch several heads' OV messages at once onto the receiver's MLP input.

    ``patches`` is a list of ``(cpos, layer, head, z_row)`` where ``z_row`` is the
    SOURCE head-z to inject. Deltas are summed PER consuming position and applied
    LN-fair at ``ln2.hook_normalized`` of ``recv_layer`` (supports patches spread
    across several positions).
    """
    L = recv_layer
    gamma = model.blocks[L].ln2.w
    # accumulate the OV delta per consuming position
    deltas = {}
    for (cpos, layer, head, z_row) in patches:
        zt = tgt_cache[f"blocks.{layer}.attn.hook_z"][0, cpos, head, :]
        zsrc = torch.as_tensor(z_row, dtype=zt.dtype)
        d = (zsrc - zt) @ model.blocks[layer].attn.W_O[head]
        deltas[cpos] = d if cpos not in deltas else deltas[cpos] + d

    # convert each position's delta into an LN-fair additive term at that position
    adds = {}
    for cpos, total in deltas.items():
        rm = tgt_cache[f"blocks.{L}.hook_resid_mid"][0, cpos, :]
        xm, std = ln_scale(rm)
        adds[cpos] = ((rm + total) - (rm + total).mean()) / std * gamma - xm / std * gamma

    def hook(act, hook):
        for cpos, a in adds.items():
            act[:, cpos, :] = act[:, cpos, :] + a
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(
            target_q.unsqueeze(0),
            fwd_hooks=[(f"blocks.{L}.ln2.hook_normalized", hook)])
    return _read_answer(model, cfg, logits)


# ===========================================================================
# Attention-pattern patching (CE8 routing)
# ===========================================================================

def pattern_patch_prediction(model, cfg, source_q: torch.Tensor, target_q: torch.Tensor,
                             layer: int, head: int, query_pos: int) -> torch.Tensor:
    """Copy the SOURCE attention row at ``(layer, head, query_pos)`` into the target
    run and predict. Isolates the causal effect of *where a head attends*.
    """
    with torch.no_grad():
        _, sc = model.run_with_cache(source_q.unsqueeze(0))
    row = sc[f"blocks.{layer}.attn.hook_pattern"][0, head, query_pos, :].clone()
    name = f"blocks.{layer}.attn.hook_pattern"

    def hook(act, hook):
        act[:, head, query_pos, :] = row.to(act.dtype)
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=[(name, hook)])
    return _read_answer(model, cfg, logits)


def synthetic_redirect_prediction(model, cfg, target_q: torch.Tensor, layer: int,
                                  head: int, query_pos: int,
                                  key_positions: Sequence[int]) -> torch.Tensor:
    """Force ``(layer, head, query_pos)`` to attend uniformly onto ``key_positions``
    and predict. A pattern-only intervention that must move the answer if the head's
    routing is load-bearing.
    """
    name = f"blocks.{layer}.attn.hook_pattern"
    w = 1.0 / len(key_positions)

    def hook(act, hook):
        act[:, head, query_pos, :] = 0.0
        for kp in key_positions:
            act[:, head, query_pos, kp] = w
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=[(name, hook)])
    return _read_answer(model, cfg, logits)


# ===========================================================================
# Head mean-ablation + causal flip-rate measurement (CE16 SV-implementation)
# ===========================================================================

def mean_ablate_heads_prediction(model, cfg, q: torch.Tensor, cpos: int,
                                 heads: Sequence[int], mean_z, layer: int) -> torch.Tensor:
    """Replace the given heads' ``z`` at ``cpos`` with their mean value and predict.

    ``mean_z`` maps head index -> mean z vector (numpy or tensor, shape [d_head]).
    Used for CLASS-necessity tests: mean-ablate a head pair jointly and see which
    question families break (CE16 class-necessity battery).
    """
    def hook(act, hook):
        for h in heads:
            act[:, cpos, h, :] = torch.as_tensor(mean_z[h], dtype=act.dtype)
        return act
    with torch.no_grad():
        logits = model.run_with_hooks(
            q.unsqueeze(0), fwd_hooks=[(f"blocks.{layer}.attn.hook_z", hook)])
    return _read_answer(model, cfg, logits)


def flip_rate_with_matched_null(model, cfg, pair_builder, patch_fn, answer_digit: int,
                                n_pairs: int = 40, null_builder=None):
    """Measure a causal patch's per-digit flip rate against a matched null.

    This is the generic form of the CE16 PC1 measurement: over ``n_pairs`` matched
    (source, target) pairs, apply ``patch_fn`` and count how often answer digit
    ``answer_digit`` changes vs the clean target prediction. Optionally repeat with
    a ``null_builder`` (e.g. a same-class / re-drawn-operand partner) to get the
    matched-null flip rate that a genuine effect must exceed.

    Args:
      pair_builder() -> (source_q, target_q)  matched intervention pair
      patch_fn(source_q, target_q) -> predicted answer tokens (uses maths_edge_patch)
      answer_digit: which answer digit A_k to score (0 = units)
      null_builder() -> (null_source_q, target_q)  matched-null pair (optional)

    Returns a dict with ``flip`` (mean_ci) and, if ``null_builder`` given, ``null``.
    """
    from quanta_maths.maths_stats import mean_ci

    ap = answer_positions(cfg)
    idx = len(ap) - 1 - answer_digit

    def clean_pred(tq):
        with torch.no_grad():
            logits = model(tq.unsqueeze(0))
        return logits[0, [p - 1 for p in ap]].argmax(-1)

    flips, nulls = [], []
    for _ in range(n_pairs):
        sq, tq = pair_builder()
        clean = clean_pred(tq)
        patched = patch_fn(sq, tq)
        flips.append(float(patched[idx] != clean[idx]))
        if null_builder is not None:
            nsq, ntq = null_builder()
            nclean = clean_pred(ntq)
            npatched = patch_fn(nsq, ntq)
            nulls.append(float(npatched[idx] != nclean[idx]))

    out = {"flip": mean_ci(flips), "answer_digit": answer_digit}
    if null_builder is not None:
        out["null"] = mean_ci(nulls)
    return out
