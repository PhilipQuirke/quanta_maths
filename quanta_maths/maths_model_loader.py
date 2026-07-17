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

# Legacy single-repo holding every VerifiedArithmetic model (flat <name>_*.json).
DEFAULT_HF_REPO = "PhilipQuirke/VerifiedArithmetic"

# Canonical per-model analysis repos: PhilipQuirke/QuantaMaths_<name> holding
# model.pth + training_loss.json + behaviors.json + features.json.
ANALYSIS_REPO_PREFIX = "PhilipQuirke/QuantaMaths_"


def analysis_repo_id(model_name: str) -> str:
    """Return the canonical per-model analysis repo id for ``model_name``."""
    return ANALYSIS_REPO_PREFIX + model_name


def _strip_state_dict(sd: dict) -> dict:
    """Some checkpoints wrap the weights under a ``"model"`` key."""
    if "model" in sd and "embed.W_E" not in sd:
        return sd["model"]
    return sd


def build_maths_config(
    model_name: str,
    hf_repo: str = DEFAULT_HF_REPO,
    use_train_json: bool = True,
    train_filename: Optional[str] = None,
) -> MathsConfig:
    """Build a ``MathsConfig`` for ``model_name``.

    ``set_model_names`` parses digits/layers/heads/seed from the name. When
    ``use_train_json`` is set (default) the paired training JSON is downloaded and
    ``load_training_json`` overlays the exact stored config (``d_head``, ``d_mlp``,
    ``d_vocab``) and records ``final_loss`` / ``avg_final_loss``.

    ``train_filename`` defaults to ``<name>_train.json`` (legacy VerifiedArithmetic).
    For the canonical per-model repos pass ``"training_loss.json"``.
    """
    cfg = MathsConfig()
    cfg.set_model_names(model_name)

    if use_train_json:
        # Imported lazily so unit tests can stub HF access.
        from QuantaMechInterp.model_train_json import (
            download_huggingface_json,
            load_training_json,
        )

        fname = train_filename if train_filename else f"{model_name}_train.json"
        data = download_huggingface_json(hf_repo, fname)
        load_training_json(cfg, data)

    return cfg


def load_maths_model_from_hf(
    model_name: str,
    device: str = "cpu",
    hf_repo: str = DEFAULT_HF_REPO,
    use_train_json: bool = True,
    strict: bool = False,
    weights_filename: Optional[str] = None,
    train_filename: Optional[str] = None,
) -> Tuple[HookedTransformer, MathsConfig]:
    """Load a model + its ``MathsConfig`` from HuggingFace.

    Returns ``(model, cfg)`` with ``model`` in eval mode on ``device``.

    Defaults target the legacy ``VerifiedArithmetic`` layout (``<name>.pth`` +
    ``<name>_train.json``). ``weights_filename`` / ``train_filename`` override the
    filenames, e.g. ``model.pth`` / ``training_loss.json`` for the per-model repos.

    ``strict=False`` (default) tolerates benign key mismatches (e.g. missing
    ``IGNORE`` buffers); callers wanting a hard check can pass ``strict=True``.
    """
    cfg = build_maths_config(model_name, hf_repo=hf_repo, use_train_json=use_train_json,
                             train_filename=train_filename)
    cfg.use_cuda = str(device).startswith("cuda")

    htc = cfg.get_HookedTransformerConfig()
    htc.device = device
    htc.init_weights = False
    model = HookedTransformer(htc)

    weights = weights_filename if weights_filename else f"{model_name}.pth"
    path = hf_hub_download(repo_id=hf_repo, filename=weights)
    sd = _strip_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(sd, strict=strict)
    model.eval()
    return model, cfg


def load_maths_model_from_analysis_repo(
    model_name: str, device: str = "cpu", strict: bool = False,
) -> Tuple[HookedTransformer, MathsConfig]:
    """Load a model from its canonical ``PhilipQuirke/QuantaMaths_<name>`` repo
    (``model.pth`` + ``training_loss.json``)."""
    return load_maths_model_from_hf(
        model_name, device=device, hf_repo=analysis_repo_id(model_name),
        weights_filename="model.pth", train_filename="training_loss.json",
        strict=strict)


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
