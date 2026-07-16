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
    SOURCE head-z to inject. Deltas are summed and applied LN-fair at
    ``ln2.hook_normalized`` of ``recv_layer``.
    """
    L = recv_layer
    total = None
    cpos_ref = patches[0][0]
    for (cpos, layer, head, z_row) in patches:
        zt = tgt_cache[f"blocks.{layer}.attn.hook_z"][0, cpos, head, :]
        zsrc = torch.as_tensor(z_row, dtype=zt.dtype)
        d = (zsrc - zt) @ model.blocks[layer].attn.W_O[head]
        total = d if total is None else total + d
    rm = tgt_cache[f"blocks.{L}.hook_resid_mid"][0, cpos_ref, :]
    xm, std = ln_scale(rm)
    gamma = model.blocks[L].ln2.w
    norm_clean = xm / std
    xm2 = (rm + total) - (rm + total).mean()
    add = (xm2 / std - norm_clean) * gamma

    def hook(act, hook):
        act[:, cpos_ref, :] = act[:, cpos_ref, :] + add
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
