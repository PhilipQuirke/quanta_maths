"""Headless QMAnalyse discovery: build behaviors.json + features.json from scratch.

This is the library-shared extraction of the ``QMAnalyse`` notebook's discovery
orchestration (previously notebook-only). Both the notebook and the HF-refresh
pipeline (``maths_hf_update``) call these functions, so there is one code path.

Behaviour discovery (-> behaviors.json tags):
  make_maths_test_questions_and_answers -> a_set_ablate_hooks + a_calc_mean_values
  -> test_maths_questions_by_complexity (drop failed Qs) -> per-position
  test_maths_questions_by_impact (useful positions) -> ablate MLP/head + tag
  (Fail% / Impact / Math.Add|Sub|Neg) -> add_node_attention_tags (Attn) ->
  headless PCA tri-case tagging (Math.Add|Sub `.SP`/`.MP`).

Feature discovery (-> features.json tags):
  search_and_tag over the sub-task functions -> Algo:* role tags.

CRITICAL ORDERING: behaviors.json must be saved BEFORE the Algo search runs (it
must contain no ``Algo`` tags), exactly as the notebook does. Callers that produce
both files must save behaviors first, then run feature discovery.

All plotting is omitted; the PCA step reproduces the notebook's tag DECISION
(EVR% + silhouette + calinski-harabasz + label-agreement thresholds) headlessly.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch


@contextlib.contextmanager
def _cpu_cuda_shim():
    """Neutralise QMI's hardcoded ``tensor.cuda()`` calls (ablate_hooks.py) when no
    GPU is present. On a CPU-only box ``.cuda()`` otherwise RAISES; here it becomes a
    no-op returning the CPU tensor, restored on exit. No effect when CUDA exists.

    TODO(upstream): QuantaMechInterp.ablate_hooks should honour cfg.use_cuda / model
    device instead of hardcoding ``.cuda()``; this shim is a workaround.
    """
    if torch.cuda.is_available():
        yield
        return
    orig = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.Tensor.cuda = orig  # type: ignore[assignment]

# PCA tri-case tagging thresholds (identical to QMAnalyse.ipynb).
PCA_EVR_PERC_THRESHOLD = 30
PCA_SILHOUETTE_THRESHOLD = 20
PCA_CALINSKI_HARABASZ_THRESHOLD = 50
PCA_LABEL_AGREEMENT_THRESHOLD = 50


def tag_pca_nodes(cfg, operation, evr_thr: float = PCA_EVR_PERC_THRESHOLD,
                  sil_thr: float = PCA_SILHOUETTE_THRESHOLD,
                  cal_thr: float = PCA_CALINSKI_HARABASZ_THRESHOLD,
                  lab_thr: float = PCA_LABEL_AGREEMENT_THRESHOLD) -> int:
    """Headless port of the notebook's auto PCA tagger. For each useful HEAD and
    answer digit, tag ``Math.Add|Sub : A{d}.SP|MP`` when the head's attention
    output clusters into the tri-case (EVR% + cluster-quality thresholds).

    Requires ``cfg.tricase_questions_dict`` populated (make_maths_tricase_questions)
    and ``cfg.main_model`` set.
    """
    from QuantaMechInterp import calc_pca_for_an, pca_evr_0_percent, QType
    from quanta_maths.maths_constants import MathsToken
    from quanta_maths.maths_pca import _build_title_and_error_message, pca_op_tag

    major = QType.MATH_ADD if operation == MathsToken.PLUS else QType.MATH_SUB
    added = 0
    for node in list(cfg.useful_nodes.nodes):
        if not node.is_head:
            continue
        for answer_digit in range(cfg.n_digits + 1):
            if (answer_digit, operation) not in cfg.tricase_questions_dict:
                continue
            base_title, err = _build_title_and_error_message(
                cfg=cfg, node_location=node, operation=operation, answer_digit=answer_digit)
            pca, _outs, _title, cluster = calc_pca_for_an(
                cfg=cfg, node_location=node, title=base_title, error_message=err,
                test_inputs=cfg.tricase_questions_dict[(answer_digit, operation)])
            if pca is None or pca_evr_0_percent(pca) <= evr_thr:
                continue
            sil = max(cluster['silhouette_scores']['2_clusters'],
                      cluster['silhouette_scores']['3_clusters'])
            cal = max(cluster['calinski_harabasz_scores']['2_clusters'],
                      cluster['calinski_harabasz_scores']['3_clusters'])
            lab = max(cluster['label_agreement_scores']['2_clusters'],
                      cluster['label_agreement_scores']['3_clusters'])
            if sil >= sil_thr and cal >= cal_thr and lab > lab_thr:
                before = len(node.tags)
                cfg.add_useful_node_tag(node, major.value, pca_op_tag(answer_digit, operation))
                added += len(node.tags) - before
    return added


def discover_behaviors(cfg, model, varied_questions=None, include_pca: bool = True) -> "UsefulNodeList":
    """Run the QMAnalyse behaviour-discovery pipeline; populate cfg.useful_nodes
    with Fail% / Impact / Math.* / Attn (+ PCA .SP/.MP) tags. Returns the list.

    NOTE: run this and save behaviors.json BEFORE discover_features (behaviors.json
    must contain no Algo tags).
    """
    from QuantaMechInterp import (
        acfg, UsefulNodeList, a_set_ablate_hooks, a_calc_mean_values,
        ablate_mlp_and_add_useful_node_tags, ablate_head_and_add_useful_node_tags,
        add_node_attention_tags)
    from quanta_maths import (
        set_maths_vocabulary, set_maths_question_meanings,
        make_maths_test_questions_and_answers, test_maths_questions_by_complexity,
        test_maths_questions_by_impact, test_maths_questions_and_add_useful_node_tags,
        make_maths_tricase_questions)
    from quanta_maths.maths_constants import MathsToken

    cfg.main_model = model
    # Respect the model's device (notebook may run on GPU; pipeline runs on CPU).
    try:
        cfg.use_cuda = str(next(model.parameters()).device).startswith("cuda")
    except StopIteration:
        cfg.use_cuda = False
    set_maths_vocabulary(cfg)
    set_maths_question_meanings(cfg)

    if varied_questions is None:
        varied_questions = make_maths_test_questions_and_answers(cfg)

    with _cpu_cuda_shim():
        a_set_ablate_hooks(cfg)
        a_calc_mean_values(cfg, varied_questions)

        # Drop questions the model itself gets wrong (can't reveal useful nodes).
        varied_questions = test_maths_questions_by_complexity(cfg, acfg, varied_questions)

        # Useful positions: ablate all nodes at each position; keep positions that hurt.
        for position in range(cfg.n_ctx):
            if test_maths_questions_by_impact(cfg, acfg, varied_questions, position, ablate=True) > 0:
                cfg.add_useful_position(position)

        # Useful nodes + behaviour tags (Fail% / Impact / Math.Add|Sub|Neg).
        cfg.useful_nodes = UsefulNodeList()
        ablate_mlp_and_add_useful_node_tags(cfg, varied_questions, test_maths_questions_and_add_useful_node_tags)
        ablate_head_and_add_useful_node_tags(cfg, varied_questions, test_maths_questions_and_add_useful_node_tags)
        add_node_attention_tags(cfg, varied_questions)

        if include_pca:
            make_maths_tricase_questions(cfg)
            if cfg.perc_add > 0:
                tag_pca_nodes(cfg, MathsToken.PLUS)
            if cfg.perc_sub > 0:
                tag_pca_nodes(cfg, MathsToken.MINUS)

    cfg.useful_nodes.sort_nodes()
    return cfg.useful_nodes


def discover_features(cfg) -> "UsefulNodeList":
    """Run the QMAnalyse feature-discovery (algo) search; add Algo:* role tags to
    cfg.useful_nodes. Must run AFTER discover_behaviors (needs the useful nodes and
    their behaviour tags for the search prereqs). Returns the list."""
    from QuantaMechInterp import acfg, search_and_tag, QType
    from quanta_maths import (
        add_ss_functions, add_sc_functions, add_sa_functions, add_st_functions,
        sub_md_functions, sub_mb_functions, sub_mt_functions, sub_gt_functions,
        neg_nd_functions, neg_nb_functions, opr_functions, sgn_functions)

    with _cpu_cuda_shim():
        cfg.useful_nodes.reset_node_tags(QType.ALGO.value)

        if cfg.perc_add > 0:
            search_and_tag(cfg, acfg, add_ss_functions)
            search_and_tag(cfg, acfg, add_sc_functions)
            search_and_tag(cfg, acfg, add_sa_functions, do_pair_search=True, allow_impact_mismatch=True)
            search_and_tag(cfg, acfg, add_st_functions, do_pair_search=True, allow_impact_mismatch=True)

        if cfg.perc_sub > 0:
            search_and_tag(cfg, acfg, sub_md_functions, do_pair_search=True, allow_impact_mismatch=True)
            search_and_tag(cfg, acfg, sub_mb_functions, allow_impact_mismatch=True)
            search_and_tag(cfg, acfg, sub_mt_functions, do_pair_search=True)
            search_and_tag(cfg, acfg, sgn_functions)
            search_and_tag(cfg, acfg, neg_nd_functions, do_pair_search=True, allow_impact_mismatch=True)
            search_and_tag(cfg, acfg, neg_nb_functions, allow_impact_mismatch=True)
            search_and_tag(cfg, acfg, sub_gt_functions, allow_impact_mismatch=True)

        if cfg.perc_add > 0 and cfg.perc_sub > 0:
            search_and_tag(cfg, acfg, opr_functions)

    return cfg.useful_nodes
