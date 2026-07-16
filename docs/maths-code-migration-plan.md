# Maths Code Migration Plan: One-off Scripts → `quanta_maths` Library

## Executive Summary
The `scripts/` directory holds ~17 one-off experiment scripts (~6,600 LOC) written during
the CE1–CE8 study cycles. They repeatedly re-implement capability that either (a) already
exists in the `quanta_maths` / `QuantaMechInterp` (QMI) library, or (b) is genuinely new and
worth promoting into the library so it can run across the ~40 HF models and store results as
JSON on HuggingFace.

This plan answers four questions:
1. **What code should migrate into `quanta_maths`?**
2. **What findings does the migrated code provide?**
3. **How do those results integrate into the "store as JSON on HF" framework?**
4. **A concrete, staged migration plan.**

Guiding principle: the library already owns *position algebra*, *tri-case (ST 0/U/1)
question building*, *interchange-intervention patching*, *PCA tri-state tagging*, and
*UsefulNode JSON persistence*. Scripts should be refactored to **consume** those, and only
genuinely new machinery (a canonical HF model loader, a linear-probe/geometry toolkit, an
edge/path-patching layer, and a per-digit ST-combiner locator) should be promoted.

**Status: IMPLEMENTED (Stages 1–5 landed, uncommitted for review).** Decisions locked in:
findings stored as **inline tags** in existing `behavior.json`/`maths.json`; Stage-5 batch is a
**local proof-of-integration** on sample models (reusing published `behavior.json` nodes; no HF
upload — a full refresh is deferred until more techniques land). See "Implementation results"
at the bottom.

---

## 1. What code should migrate into `quanta_maths`?

Migration candidates fall into three tiers.

### Tier A — Promote to library (new, reusable, non-overlapping)

| Candidate | Source script(s) | Target module | Why it's new |
|---|---|---|---|
| **`load_maths_model_from_hf(model_name, device="cpu")`** | `confirm_st_node.py:37` (+ 2 dup variants in `pair_sum_sufficiency.py:212`, `digit_embedding_geometry.py:387`) | `quanta_maths/maths_model_loader.py` (new) | Library has **no** HF `.pth`→`HookedTransformer` loader; it lives only in notebooks and duplicated scripts. Highest-value gap. Must build `MathsConfig` from `_train.json`, force device override, handle `sd["model"]` wrapper + `strict=False`. |
| **Linear-probe + axis-decomposition toolkit** | `probe_transfer.py`, `node_output_encoding.py`, `digit_embedding_geometry.py`, `st_tristate_geometry.py` | `quanta_maths/maths_probe.py` (new) | Library PCA (`maths_pca.py`) only does KMeans-cluster tagging of the tri-case. Supervised linear probes (train/test split, cross-position transfer, permutation nulls, DFT circular-order geometry) are new and are the backbone of CE1/CE2/CE6/CE7. |
| **Edge / path patching (`edge_patch_pred`, `head_ov`, `_direct_patch_pred`)** | `cascade_handoff_edge_patch.py`, `sv_compounding.py`, `deep_cascade_mechanism.py` | `quanta_maths/maths_edge_patch.py` (new) | QMI `a_run_attention_intervention` patches *node z* and measures per-digit impact, but does **not** do sender→receiver *edge* patching or OV-path isolation. Needed for CE2/CE5/CE8 cascade claims. |
| **Per-digit ST-combiner locator** | `confirm_st_node.py`, `sv_compounding.py`, `combiner-vs-conduit` study | new `SubTaskBase` subclass `add_stc_functions` in `maths_search_add.py` | CE5 (U-combiner = answer-position L1 MLP) and CE8 (hybrid routing) are per-digit facts that belong in the auto-tagging search so they produce `Algo:` tags for all models. |

### Tier B — Refactor scripts to consume the library (delete duplicated helpers)

These helpers are copy-pasted across ≥3 scripts and **already exist** in the library. Delete
the local copies; import the library equivalent.

| Duplicated local helper | Library replacement |
|---|---|
| `answer_positions(cfg)`, inline digit-index math | `MathsConfig.{an,dn,ddn,op}_to_position_name` |
| `make_q` / `make_pair` / `build_class_question` | `make_maths_tricase_questions[_customized]`, `make_single_tricase_question` (case 8/9/10 → ST 0/U/1) |
| `_digits_to_int`, `predict_answer`, `verify_accuracy` | `SimpleQuestionDescriptor.from_tensor`, `test_correctness_on_num_questions`, `test_maths_questions_by_complexity` |
| ad-hoc permutation-null loops | promote **one** `permutation_null(...)` into `maths_probe.py`, reuse everywhere |
| ad-hoc "untrained-control" model builder | promote `make_untrained_control(cfg)` into `maths_model_loader.py` |
| per-script `load_model` | `load_maths_model_from_hf` (Tier A) |

### Tier C — Leave in `scripts/` (genuinely one-off / exploratory)

Narrative walkthroughs and single-model sanity probes with no multi-model reuse value
(`leading_digit_walkthrough`, `ln_aware_embedding` exploratory bits). Keep as study
artifacts; do not promote.

---

## 2. What findings does the migrated code provide?

Mapped to the existing claim-evidence ledger (CE1–CE8) so promotion preserves provenance:

- **`maths_model_loader`** → prerequisite for *every* CE re-run across models; enables the
  10-model replication the human wants.
- **`maths_probe` (linear probes + DFT geometry)** →
  - CE1: digit embeddings near-isotropic 9-D; LN-robust-but-seed-fragile circular ordering.
  - CE2: operand-fetch value path = linear transport (probe transfer across positions).
  - CE6/CE7: no dedicated `{0,1,U}` tri-state symbol; carry is binary, resolved ~L1.
- **`maths_edge_patch`** →
  - CE2: operand-fetch OV path isolation.
  - CE5: U-combiner = answer-position L1 MLP (edge sender→receiver).
  - CE8: hybrid attention routing (a few heads route by carry state; 6-digit only) —
    falsifies strong A5.
- **`add_stc_functions` search subtask** → per-digit `Algo:A{d}.STC` tags identifying the
  ST-combiner node in *any* model, making CE5/CE8 machine-checkable at scale.

The value of promotion is **cross-model generalization**: today these findings are proven on
1–2 models (primary `add_d5_l2_h3_t15K_s372001`, replication `add_d6_l2_h3_t20K_s173289`);
promoted, they become a batch job over all ~40 HF models with per-model JSON evidence.

---

## 3. How results integrate into the "store as JSON on HF" framework

The library already persists node facts via `UsefulNodeList.save_nodes(filename, major_tag)`
/ `load_nodes`, and the HF contract (`docs/hugging_models.md`) is:

- `XXXXXX.pth` — weights (input).
- `XXXXXX_train.json` — `{Config, TrainingLoss, AvgFinalLoss, FinalLoss}` (input; loaded by
  `load_training_json`). **Consumed by** `load_maths_model_from_hf`.
- `XXXXXX_behavior.json` — auto-learned behavior tags (`Fail%:`, `Impact:A…`,
  `Math.Add/Sub/Neg:`, `Attn:Pn=…`). Produced by
  `test_maths_questions_and_add_useful_node_tags`.
- `XXXXXX_maths.json` — algorithmic tags (`Algo:OPR`, `Algo:D4.GT`, `Algo:A5.SA/MD/ND`,
  `Algo:SGN`, …). Produced by the `algo_search` drivers.

**Integration rule (DECIDED — inline tags): the migrated code must emit its findings as new
minor tags on existing UsefulNodes inside the existing `XXXXXX_behavior.json` /
`XXXXXX_maths.json` artifacts, not as ad-hoc `results/study-*/results.json` and not as
separate sidecar files.** The node-list schema is explicitly extensible (`useful_tags.md:59`).
Concretely:

1. **ST-combiner** (`add_stc_functions`) → adds `Algo:A{d}.STC` to `XXXXXX_maths.json`
   (reuses the existing `search_and_tag_digit` driver; zero new file format).
2. **Probe / geometry facts** → new `Probe:` minor tags on nodes, e.g. `Probe:A{d}.LINXFER`,
   written **inline into `XXXXXX_behavior.json`** via `UsefulNodeList.save_nodes`.
3. **Edge-patch cascade facts** → node tags on the sender/receiver
   (`Edge:A{d}.STC<-L0Hk`), written **inline into `XXXXXX_maths.json`**.
4. **Model-level scalars** (probe accuracy, permutation-null p-values, EVR) → encoded as a
   compact numeric suffix on the relevant node tag (e.g. `Probe:A2.LINXFER=97`) so they ride
   along inside the same JSON rather than a parallel scalar store. Cross-model tables are
   built by reading these tags back across the model set.

This keeps everything inside the existing UsefulNode/JSON-on-HF pipeline; the Colab
`VerifiedArithmeticAnalysis` notebook gains a few extra tagging passes that mutate the
established `behavior.json` / `maths.json` artifacts in place.

**Evidence-integrity constraint (from the research contract):** headline numbers must be
computed *in the committed library function* and written to JSON — never hand-copied. The
migration must therefore keep the "compute → JSON → read back for the note" path intact.

---

## 4. Migration plan (staged)

Each stage ends green (`pytest tests`) before the next. Every promoted function gets a
positive **and** negative control test (untrained-control model must fail the probe/patch),
matching the study harness discipline that the skeptic gates enforce.

### Stage 0 — Prep (no behavior change)
- Add `docs/study-maths/study-code-migration.md` note with `## Executive Summary`.
- Snapshot current `pytest tests` green baseline.

### Stage 1 — `maths_model_loader.py` (Tier A, highest value)
- Implement `load_maths_model_from_hf(model_name, device="cpu")` and
  `make_untrained_control(cfg)`.
- Build `MathsConfig` from `_train.json` via `init_from_json`; force device; handle
  `sd["model"]` + `strict=False`.
- Refactor `confirm_st_node.load_model` and the 2 dup variants to call it.
- Un-comment `tests/test_maths.py::test_pca` model-load path against a small model.
- **Controls:** loaded model reproduces `_train.json` `FinalLoss`; untrained control does not.

### Stage 2 — `maths_probe.py` (Tier A)
- Promote linear-probe train/test, cross-position transfer, `permutation_null`, DFT circular
  geometry from `probe_transfer.py` / `digit_embedding_geometry.py` / `node_output_encoding.py`.
- Refactor those scripts to import; delete duplicated null loops.
- Emit `Probe:` tags via `UsefulNodeList` (Section 3.2).
- **Controls:** probe on shuffled labels ≈ chance; untrained control ≈ chance.

### Stage 3 — `maths_edge_patch.py` (Tier A)
- Promote `edge_patch_pred`, `head_ov`, `_direct_patch_pred` from
  `cascade_handoff_edge_patch.py` / `sv_compounding.py` / `deep_cascade_mechanism.py`.
- Reconcile with QMI `a_run_attention_intervention` (reuse its `acfg` hooks where possible;
  add only the edge/OV-path layer).
- Emit `Edge:` tags (Section 3.3).
- **Controls:** patching an irrelevant edge → no impact; known SV edge → expected per-digit
  impact.

### Stage 4 — `add_stc_functions` ST-combiner search (Tier A)
- Add `SubTaskBase` subclass to `maths_search_add.py` producing `Algo:A{d}.STC`.
- Wire into `search_and_tag_digit`; extend `tests/test_maths.py` prereqs/tag smoke test.
- **Controls:** finds the known combiner (P14.L1.MLP on primary; P16.L1.MLP on replication);
  finds nothing in untrained control.

### Stage 5 — Cross-model batch + JSON
- **Model scope (DECIDED — 5 accurate models):** the 4 accurate addition models
  (`add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`,
  `add_d6_l2_h3_t20K_s572091`) plus the accurate mixed model
  `ins1_mix_d6_l3_h4_t40K_s372001`.
- Add a driver that runs Stages 1–4 over those 5 models and writes the new tags **inline**
  into each model's existing `XXXXXX_behavior.json` / `XXXXXX_maths.json`.
- Build the cross-model comparison table by reading the `Algo:/Probe:/Edge:` tags back.
- **This delivers the human's stated goal:** re-run the validated SV explanation across the
  accurate models.

### Stage 6 — Cleanup
- Delete Tier-B duplicated helpers from scripts.
- Cross-doc + link audit; update `docs/maths-next-steps.md` agenda and
  `docs/maths-results-synthesis.md`.

---

## Implementation results (Stages 1–5)

All stages built with positive + negative controls; `pytest tests` = **47 passed, 10 skipped**
(HF-integration tests gated behind `RUN_HF_TESTS=1`; all pass when enabled).

| Stage | New library module | Tests | Refactored script (proof of drop-in) |
|---|---|---|---|
| 1 | `maths_model_loader.py` (`load_maths_model_from_hf`, `make_untrained_control`) | `test_model_loader.py` (4 offline + 3 HF) | `confirm_st_node.load_model`, `pair_sum_sufficiency.load_model` |
| 2 | `maths_probe.py` (labels, sites, probes+null, subspace angles, DFT geometry) | `test_probe.py` (11 offline) | `probe_transfer.py` |
| 3 | `maths_edge_patch.py` (OV path, edge/LN-fair patch, pattern patch) | `test_edge_patch.py` (3 offline + 4 HF) | `cascade_handoff_edge_patch.py` |
| 4 | `add_stc_functions` + `MathsTask.STC_TAG` | `test_stc_search.py` (3 offline + 2 HF) + `test_maths.py` | — |
| 5 | `maths_batch.py` (`run_batch`, STC + LINXFER inline tagging, local-only) | `test_batch.py` (1 offline + 1 HF) | — |

**Sample batch output (2 models, local):** `add_d5_l2_h3_t15K_s372001` → 4 `Algo:A{d}.STC` tags on
the L1 answer-position MLPs (P13–P17) + 1 `Probe:A1.LINXFER=89` on the P9 operand-fetch head;
`add_d6_l2_h3_t20K_s173289` → 5 STC + 2 LINXFER. Tags are written **inline** into local
`*_maths.json` / `*_behavior.json` copies (transient; regenerate via `run_batch`).

**Gate-caught correction:** an initial Stage-3 positive control (full L1 attention-pattern swap)
did NOT move the 5-digit answer — consistent with CE8 (hybrid carry-routing was 6-digit-only).
The control was rewritten to a `synthetic_redirect` on an L0 answer-position head, matching CE2/CE3
physics, rather than forcing a false assertion.

**Thread boundary:** experiments `#17/#18` and `scripts/sv_implementation.py` (+ its study/results
files) were NOT touched.

## Decisions (resolved)
1. **Inline tags** — extend existing `XXXXXX_behavior.json` / `XXXXXX_maths.json` with new
   `Probe:` / `Edge:` minor tags (with numeric `=NN` suffix for scalars). No sidecar files.
2. **Model scope for Stage 5** — 5 accurate models: 4 accurate add models +
   `ins1_mix_d6_l3_h4_t40K_s372001`.
3. **Scope** — **plan-only for now.** No code changes until the human reviews this plan.

4. **QMI ownership (DECIDED)** — the edge/OV-path layer lives in `quanta_maths`
   (subject-specific), i.e. `quanta_maths/maths_edge_patch.py`. It may *reuse* QMI's `acfg`
   hooks but is not pushed down into `QuantaMechInterp`.
5. **Thread boundary** — experiments `#17`, `#18`, … are owned by another thread. This
   migration must **not** touch those scripts.
