"""Canonical loader for VerifiedArithmetic models hosted on HuggingFace.

Before this module, every experiment script and Colab notebook re-implemented its
own ``load_model`` (build a ``MathsConfig``, download the ``.pth``, override the
device, ``load_state_dict``). This module owns that path once so the CE1-CE8
findings can be re-run across the ~40 HF models from a single function.

Model-name grammar (see docs/hugging_models.md), e.g.::

    add_d5_l2_h3_t15K_s372001
    = addition, 5 digits, 2 layers, 3 heads, 15K epochs, seed 372001

The paired ``XXXXXX_train.json`` artifact supplies the exact config fields that the
name does not encode (``d_head``, ``d_mlp``, ``d_vocab``, ``final_loss``) and is the
source of truth for the positive-control loss check.

CPU-friendly: ``device`` defaults to ``"cpu"``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from quanta_maths.maths_config import MathsConfig

# Default HF repository holding every VerifiedArithmetic model.
DEFAULT_HF_REPO = "PhilipQuirke/VerifiedArithmetic"


def _strip_state_dict(sd: dict) -> dict:
    """Some checkpoints wrap the weights under a ``"model"`` key."""
    if "model" in sd and "embed.W_E" not in sd:
        return sd["model"]
    return sd


def build_maths_config(
    model_name: str,
    hf_repo: str = DEFAULT_HF_REPO,
    use_train_json: bool = True,
) -> MathsConfig:
    """Build a ``MathsConfig`` for ``model_name``.

    ``set_model_names`` parses digits/layers/heads/seed from the name. When
    ``use_train_json`` is set (default) the paired ``_train.json`` is downloaded and
    ``load_training_json`` overlays the exact stored config (``d_head``, ``d_mlp``,
    ``d_vocab``) and records ``final_loss`` / ``avg_final_loss`` for control checks.
    """
    cfg = MathsConfig()
    cfg.set_model_names(model_name)

    if use_train_json:
        # Imported lazily so unit tests can stub HF access.
        from QuantaMechInterp.model_train_json import (
            download_huggingface_json,
            load_training_json,
        )

        data = download_huggingface_json(hf_repo, f"{model_name}_train.json")
        load_training_json(cfg, data)

    return cfg


def load_maths_model_from_hf(
    model_name: str,
    device: str = "cpu",
    hf_repo: str = DEFAULT_HF_REPO,
    use_train_json: bool = True,
    strict: bool = False,
) -> Tuple[HookedTransformer, MathsConfig]:
    """Load a VerifiedArithmetic model + its ``MathsConfig`` from HuggingFace.

    Returns ``(model, cfg)`` with ``model`` in eval mode on ``device``.

    ``strict=False`` (default) tolerates benign key mismatches (e.g. missing
    ``IGNORE`` buffers); callers wanting a hard check can pass ``strict=True``.
    """
    cfg = build_maths_config(model_name, hf_repo=hf_repo, use_train_json=use_train_json)

    htc = cfg.get_HookedTransformerConfig()
    htc.device = device
    htc.init_weights = False
    model = HookedTransformer(htc)

    path = hf_hub_download(repo_id=hf_repo, filename=f"{model_name}.pth")
    sd = _strip_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(sd, strict=strict)
    model.eval()
    return model, cfg


def make_untrained_control(
    cfg: MathsConfig,
    device: str = "cpu",
    seed: Optional[int] = 0,
) -> HookedTransformer:
    """Return a randomly-initialised model with the SAME architecture as ``cfg``.

    This is the negative control for every study harness: any probe/patch/search
    that fires on the trained model must NOT fire (or must fall to chance) on this
    untrained twin. Keeping it in the library guarantees every experiment uses an
    identically-shaped control.
    """
    htc = cfg.get_HookedTransformerConfig()
    htc.device = device
    htc.init_weights = True
    if seed is not None:
        htc.seed = seed
    model = HookedTransformer(htc)
    model.eval()
    return model
