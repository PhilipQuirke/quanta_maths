"""Cross-model batch tagging driver (Stage 5, local proof-of-integration).

Reuses each model's ALREADY-PUBLISHED ``*_behavior.json`` useful-node list, adds
the new Stage-2/3/4 findings as INLINE tags (per the migration decision), and
writes the updated JSON to a LOCAL directory only (no HuggingFace upload). A full
refresh / HF push is deferred until more techniques land.

New inline tags produced:
  * ``Algo:A{d}.STC``          -- the answer-position L1 MLP is the ST-combiner (CE5),
                                  confirmed by a causal ablation-flip check.
  * ``Probe:A{d}.LINXFER=NN``  -- balanced probe accuracy (%) that the digit's
                                  operand value is linearly decodable at its fetch
                                  site (CE2 linear-transport evidence).

This is intentionally lightweight: it does NOT re-run full ablation discovery; it
trusts the published node list and only adds tags to nodes that pass a direct
causal / decodability check.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

# The 5 accurate models (Stage-5 scope decision).
ACCURATE_MODELS = [
    "add_d5_l2_h3_t15K_s372001",
    "add_d6_l2_h3_t15K_s372001",
    "add_d6_l2_h3_t20K_s173289",
    "add_d6_l2_h3_t20K_s572091",
    "ins1_mix_d6_l3_h4_t40K_s372001",
]


def _download_behavior_nodes(model_name: str, hf_repo: str, local_dir: str):
    """Download a model's published behavior.json and load it into a UsefulNodeList."""
    from QuantaMechInterp import UsefulNodeList
    from QuantaMechInterp.model_train_json import download_huggingface_json
    import json

    data = download_huggingface_json(hf_repo, f"{model_name}_behavior.json")
    raw_path = os.path.join(local_dir, f"{model_name}_behavior.json")
    with open(raw_path, "w") as f:
        json.dump(data, f)
    nodes = UsefulNodeList()
    nodes.load_nodes(raw_path)
    return nodes, raw_path


def _stc_is_causal(model, cfg, produce_pos: int, impact_digit: int,
                   mlp_layer: int) -> bool:
    """Return True if ablating the answer-position MLP flips answer digit A_k on a
    carry-bearing addition (the CE5 combiner signature)."""
    from quanta_maths.maths_edge_patch import answer_positions
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken

    base = int(("2" * cfg.n_digits))
    a = base + 7 * 10 ** max(0, impact_digit - 1)
    b = base + 7 * 10 ** max(0, impact_digit - 1)
    lim = 10 ** cfg.n_digits
    a %= lim
    b %= lim
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, MathsToken.PLUS)
    q = q[0]
    ap = answer_positions(cfg)
    with torch.no_grad():
        clean = model(q.unsqueeze(0))[0, [p - 1 for p in ap]].argmax(-1)

    def hook(act, hook):
        act[:, produce_pos, :] = 0.0
        return act
    with torch.no_grad():
        abl = model.run_with_hooks(
            q.unsqueeze(0),
            fwd_hooks=[(f"blocks.{mlp_layer}.mlp.hook_post", hook)]
        )[0, [p - 1 for p in ap]].argmax(-1)
    return not torch.equal(clean, abl)


def tag_stc_nodes(model, cfg, nodes, mlp_layer: Optional[int] = None) -> int:
    """Add ``Algo:A{d}.STC`` to answer-position MLP nodes that pass the causal check.

    ``mlp_layer`` defaults to the penultimate layer (the combiner layer in the
    2-layer models is L1). Returns the number of tags added.
    """
    from QuantaMechInterp import QType
    from quanta_maths.maths_search_add import add_stc_functions

    if mlp_layer is None:
        mlp_layer = cfg.n_layers - 1

    added = 0
    for k in range(cfg.n_digits + 1):
        produce_pos = int(cfg.an_to_position_name(k)[1:]) - 1
        for node in nodes.nodes:
            if node.is_head:
                continue
            if node.position != produce_pos or node.layer != mlp_layer:
                continue
            if _stc_is_causal(model, cfg, produce_pos, k, mlp_layer):
                added += node.add_tag(QType.ALGO.value, add_stc_functions.tag(k))
    return added


def _collect_operand_digit_labels(model, cfg, n_q, digits, rng):
    """Gather operand-2 residual activations at each digit's own L0 fetch site,
    labelled by the operand-2 DIGIT VALUE (0..9) -- the quantity CE2 says is
    linearly transported. Returns (acts, labs) keyed by digit index n.
    """
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken
    from quanta_maths.maths_probe import site_hook_and_pos

    acts = {n: [] for n in digits}
    labs = {n: [] for n in digits}
    lim = 10 ** cfg.n_digits
    hook = "blocks.0.hook_resid_post"
    for _ in range(n_q):
        a = int(rng.integers(0, lim // 2))
        b = int(rng.integers(0, lim // 2))
        q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(cfg, q, 0, a, b, MathsToken.PLUS)
        bd = [int(d) for d in str(b).zfill(cfg.n_digits)]
        with torch.no_grad():
            _, c = model.run_with_cache(
                q, names_filter=lambda nm: nm == hook)
        for n in digits:
            _, pos = site_hook_and_pos(cfg, "Dpn_L0", n)
            acts[n].append(c[hook][0, pos, :].numpy())
            labs[n].append(bd[cfg.n_digits - 1 - n])  # operand-2 digit value at n
    return ({n: np.asarray(v) for n, v in acts.items()},
            {n: np.asarray(v) for n, v in labs.items()})


def tag_linxfer_nodes(model, cfg, nodes, digits: Optional[List[int]] = None,
                      n_q: int = 800, seed: int = 0) -> int:
    """Add ``Probe:A{d}.LINXFER=NN`` to operand-fetch head nodes when the operand-2
    DIGIT VALUE is linearly decodable at its L0 fetch site (CE2 linear transport).
    ``NN`` is the balanced-accuracy percentage. Returns tags added.
    """
    from quanta_maths.maths_probe import probe_accuracy_with_null

    if digits is None:
        digits = list(range(1, cfg.n_digits - 1))
    rng = np.random.default_rng(seed)
    acts, labs = _collect_operand_digit_labels(model, cfg, n_q, digits, rng)

    added = 0
    for n in digits:
        out = probe_accuracy_with_null(acts[n], labs[n], rng, n_perm=50)
        # 10-way digit decode: chance 0.10; require a clear margin + significance.
        if out["observed_acc"] > 0.5 and out["null_p"] < 0.05:
            pct = int(round(out["observed_acc"] * 100))
            fetch_pos = int(cfg.ddn_to_position_name(n)[1:])
            for node in nodes.nodes:
                if node.position == fetch_pos and node.layer == 0 and node.is_head:
                    added += node.add_tag("Probe", f"A{n}.LINXFER={pct}")
    return added


def run_batch(models: Optional[List[str]] = None,
              hf_repo: str = "PhilipQuirke/VerifiedArithmetic",
              local_dir: str = "results/study-code-migration/batch",
              do_linxfer: bool = True) -> dict:
    """Run the local batch over ``models`` (default: the 5 accurate models).

    For each model: load it + its published behavior nodes, add STC (+ optional
    LINXFER) inline tags, and save updated ``*_maths.json`` / ``*_behavior.json``
    locally. Returns a summary dict.
    """
    from quanta_maths import load_maths_model_from_hf

    models = models or ACCURATE_MODELS
    os.makedirs(local_dir, exist_ok=True)
    summary = {}
    for name in models:
        model, cfg = load_maths_model_from_hf(name, device="cpu")
        nodes, _ = _download_behavior_nodes(name, hf_repo, local_dir)
        n_stc = tag_stc_nodes(model, cfg, nodes)
        n_lin = tag_linxfer_nodes(model, cfg, nodes) if do_linxfer else 0
        maths_path = os.path.join(local_dir, f"{name}_maths.json")
        behav_path = os.path.join(local_dir, f"{name}_behavior.json")
        nodes.save_nodes(maths_path)
        nodes.save_nodes(behav_path)
        summary[name] = {"stc_tags": n_stc, "linxfer_tags": n_lin,
                         "n_nodes": len(nodes.nodes)}
    return summary
