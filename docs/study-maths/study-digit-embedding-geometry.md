# Study: Digit-Embedding Geometry Audit (study-digit-embedding-geometry.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #3

**Verdict: AMBIGUOUS — no clean geometric digit code at the token level.** Across
four accurate addition models the digit **embedding** is near-isotropic in 9
dimensions (participation ratio ≈ 8.7/9), *not* the low-dimensional circle/helix
the direction hypothesised. A weak frequency-1 (circular) signal exists (~0.27 of
centered variance, permutation p < 0.01 in 3 of 4 models) but sits in the
sub-threshold 25–40% band and is only ~4 points above the untrained baseline
(0.23); the unembedding is unstructured and misaligned with the embedding.

The one training-induced signal is a weak circular **ordering** of digit values
(significant in 3/4 models, absent in the untrained control), provisional pending
the LN-aware close-out. Net: the strong "dominant digit geometry" reading (A1) is
refuted at the token level; this became claim CE1 and redirected the thread from
token-embedding geometry toward activation/MLP-level mechanism.

## Pre-run (write before the experiment)

- **Question**: Do the digit-token embeddings and unembeddings of accurate
  trained addition models carry functional numeric geometry — circular
  (mod-10, Fourier-like) structure, possibly plus an ordered magnitude
  component (a helix) — or are they an unstructured, near-orthogonal 10-way
  categorical code?

- **Motivation**: This is the cheapest entry in the agenda (weights-only, no
  forward-pass harness) and it settles the vocabulary every later geometry
  study is phrased in: the geometry-vs-lookup fork between
  [A1](../maths-conjectures-agent.md#a1-digit-embeddings-carry-functional-numeric-geometry-mod-10-circle-maybe-helix)
  and its lookup/heuristics alternatives, and the "near-linear" wording of
  [C1](../maths-conjectures-human.md#c1-representations-are-simple-near-linear-and-low-effective-dimension).
  It also calibrates the full-space geometry instruments that agenda entries
  3, 5, and 8 will reuse.

- **Hypothesis / competing reads** (neutral; all are live):
  - **R1 — Circular**: centered variance of the 10 digit vectors concentrates
    in one or two circular (Fourier) frequencies over digit value, with the
    digits in circular order and `9` adjacent to `0` (wrap-around). Predicted
    by A1's main branch.
  - **R2 — Helix**: a circular component plus a distinct ordered magnitude
    direction (value-linear). Predicted by A1's helix variant.
  - **R3 — Value-linear only**: a dominant ordered line `0→9` with no
    wrap-around. Partial A1 (magnitude without modular geometry); consistent
    with Zhou et al. 2024's finding that from-scratch models lean on
    low-frequency/magnitude features.
  - **R4 — Unstructured categorical**: near-equidistant, near-orthogonal
    vectors with no label-aligned ordering; effective dimension ≈ 9.
    Consistent with the lookup-table alternative (MLP key-value memories over
    digit pairs) and requires no numeric structure at the token level.
  - Mixtures are possible (partial structure plus idiosyncratic remainder);
    the design classifies by variance share rather than forcing a binary.
  - Note: R1–R3 are all "simple" in C1's sense; the discriminating question is
    *which* simple form, and whether any label-aligned form exists at all (R4).

- **Design**:
  - **Models** (weights from
    [PhilipQuirke/VerifiedArithmetic](https://huggingface.co/PhilipQuirke/VerifiedArithmetic),
    per [hugging_models.md](../hugging_models.md); loaded via `MathsConfig` /
    TransformerLens `HookedTransformer`):
    - Primary: `add_d5_l2_h3_t15K_s372001` (accurate 5-digit, 2-layer, 3-head).
    - Replication A (size): `add_d6_l2_h3_t15K_s372001` (accurate 6-digit).
    - Replication B (seed): `add_d6_l2_h3_t20K_s173289` (accurate 6-digit,
      different seed).
    - Optional descriptive extra, not verdict-driving:
      `add_d5_l1_h3_t30K_s372001` (inaccurate 1-layer Paper-1 reproduction) —
      does weaker task accuracy co-occur with weaker geometry?
  - **Objects**: the 10 digit-token rows of the embedding `W_E` (tokens `0-9`
    are vocabulary indices 0–9; `d_model = 510`) and the 10 digit-token
    columns of the unembedding `W_U`. Embed and unembed are analyzed and
    reported separately (digits play different roles as question inputs vs
    answer outputs). Operator/sign/equals tokens (indices 10–14) are out of
    scope.
  - **Instrument verification**: before analysis, verify each loaded model
    reproduces expected behavior on a small batch of addition questions
    (accuracy check), guarding against loading/config errors.
  - **Metrics** (all on the centered 10-vector set, full `d_model`
    dimensionality, no 2D/3D projection shortcuts):
    1. Effective dimension: singular value spectrum, variance shares,
       participation ratio.
    2. Circular-frequency decomposition: joint linear regression of the
       embedding matrix on a design matrix of `cos(2πkd/10), sin(2πkd/10)`
       for `k = 1..4`, the alternating term (`k = 5`), and a linear term `d`;
       report variance share per component. (With 10 points, `k` and `10−k`
       alias; `k ≤ 5` is the full basis. Shared variance between the linear
       term and low frequencies is reported explicitly, not double-counted.)
    3. Circular-order statistic: angular order of the 10 digits in the
       dominant 2D plane (top-2 principal components, and separately the
       fitted frequency-`k` plane); count adjacency violations; wrap-around
       check: distance(9, 0) versus mean adjacent-digit distance.
    4. Structured-model comparison: variance explained by circle-only (R1),
       circle+line (R2), line-only (R3) bases, versus the permutation null
       (R4 reference).
    5. Embed/unembed relation: principal angles between the dominant embed
       and unembed subspaces; whether the two geometries agree in form and
       digit ordering.
  - **Controls / nulls**:
    - Label-permutation null: recompute the frequency-concentration and
      ordering statistics under ≥10,000 random relabelings of the 10 digits.
      This tests *label alignment* — the point cloud's intrinsic low-rankness
      is unchanged by relabeling, so incidental low-dimensionality cannot
      pass. Empirical p-values from the permutation distribution.
    - Untrained control: an identically configured, freshly initialized model
      (via `MathsConfig.get_HookedTransformerConfig`, `init_weights = True`)
      run through the identical pipeline; expected to show no label-aligned
      structure. Calibrates the pipeline's baseline on real-scale random
      weights.
  - **LayerNorm caveat handled as secondary analysis**: these models use LN,
    so raw `W_E` geometry may differ from the effective geometry after the
    first LN. Primary analysis is raw weights (pre-registered); a secondary
    LN-aware variant is reported alongside. If the two disagree on the
    verdict, the result is classified ambiguous (instrument sensitivity), not
    silently resolved.
  - **Sample sizes and power**: n = 10 vectors per matrix is fixed by the
    task; 2 matrices per model; 3 verdict-driving models. Statistical power
    comes from the permutation null (resolution ~1e-4 with 10k draws) and
    from replication across models, not from n.
  - **Minimal effect of interest**: a structured basis (circular and/or
    linear) capturing **≥ 40%** of centered variance with permutation
    **p < 0.01**, replicated in at least 2 of 3 accurate models. The positive
    control (below) must demonstrate the pipeline detects planted structure
    at exactly this level; if it cannot, the study is underpowered and says
    so before interpreting real data. Structure between 25% and 40% is
    reportable as "partial" but does not drive a positive verdict.
  - **Scope**: weights-only; token embedding/unembedding geometry. Explicitly
    NOT tested here: whether the model *uses* any found geometry (causal —
    backlog item B1), node-level activation geometry (agenda entries 2–3),
    or operator/sign token structure.

- **Positive control**: a planted-geometry synthetic sweep. Construct
  synthetic 10×510 matrices with known ground truth: (a) pure frequency-1
  circle plus isotropic Gaussian noise, at graded structure-variance shares
  (~20% to ~90%); (b) helix (circle + line) plus noise, same grading; (c)
  pure noise. The pipeline must (i) recover planted variance shares within
  stated tolerance, (ii) detect planted structure at the 40%-share
  minimal-effect level with permutation p < 0.01, and (iii) report no
  significant structure on pure noise at the same α (false-positive check).
  A flat result on real embeddings with a failing positive control is
  `invalid`/`underpowered`, never `negative`.

- **Success condition** (defined now): in **≥ 2 of 3** accurate models, a
  structured basis captures ≥ 40% of centered variance at permutation
  p < 0.01 in the embedding (unembedding reported alongside), with consistent
  digit ordering in the dominant plane (≤ 1 adjacency violation). Verdict:
  "label-aligned numeric geometry present", sub-classified R1 / R2 / R3 by
  which basis dominates. Note the pre-stated asymmetry: R1/R2 (wrap-around
  present) confirms A1's circular prediction; R3 (line without wrap-around)
  is positive-for-structure but scores A1's *circular* prediction as refuted
  at the embedding level while confirming its magnitude component.

- **Failure condition** (defined now): with the positive control passing —
  in **all 3** accurate models, no structured basis reaches 25% of centered
  variance at p < 0.01, and ordering statistics are indistinguishable from
  the permutation null. Verdict: "no label-aligned numeric geometry" (R4);
  scores A1's embedding-level prediction refuted and strengthens the
  lookup-table alternative.

- **Ambiguous / invalid condition**:
  - Ambiguous: best structured basis lands between 25% and 40% variance
    share; or replication splits (structure in exactly 1 of 3 models); or
    embed and unembed disagree in form; or the raw-weights and LN-aware
    variants disagree on the verdict.
  - Invalid: positive control fails any of its three requirements; or a model
    fails the instrument-verification accuracy check (bad load voids that
    model's data, and fewer than 2 valid accurate models voids the run).

- **Skeptic review (pre-launch)**: Run 2026-07-14 in a separate skeptic thread
  that rehydrated only from version-controlled docs (this note, the two
  conjecture files, [thor-document-rules.md](../thor-document-rules.md),
  [thor-glossary.md](../thor-glossary.md),
  [maths-next-steps.md](../maths-next-steps.md),
  [hugging_models.md](../hugging_models.md)) and did not inherit the working
  thread's context.

  **Verdict: PASS WITH CONDITIONS.** The design architecture (weights-only,
  label-permutation null, LN-ambiguity → ambiguous, verdict-label integrity) is
  sound and compliant with the Evidence And Verdict Rules. Two BLOCKING
  statistical-specification gaps and several recommended tightenings were raised.

  BLOCKING findings:

  1. **≥40% "structured basis" threshold is degenerate at n=10.** The Metric-2
     regressor set (`cos/sin(2πkd/10)`, k=1..4 = 8 terms; k=5 alternating = 1;
     linear = 1) is 10 non-constant regressors spanning the full rank-9 centered
     space, so a *joint* fit explains ~100% of centered variance by
     construction. As written, Success ("structured basis captures ≥ 40%") is
     trivially true and Failure ("no basis reaches 25%") almost never fires;
     R1/R2/R3/R4 are not separable.
  2. **Linear-vs-low-frequency basis leakage has no defined convention.** Over
     10 points the linear term is fully expressible in the Fourier basis;
     marginal shares are not additive, so R2 (helix) vs R3 (line-only) vs R1
     (circle) cannot be adjudicated without a pre-registered partialling rule.

  Non-blocking / recommended:

  3. Positive-control pure-noise arm should exercise the *label-permutation*
     pipeline itself (confirm empirical false-positive rate ≈ α), not a
     different significance path.
  4. Add an anisotropic / spectrum-matched noise arm to the positive control;
     isotropic Gaussian under-stresses the incidental-low-rank failure mode.
  5. "3 verdict-driving models" is really ~2 seeds: primary and Replication A
     share seed `s372001`; only Replication B (`s173289`) is independent. Either
     swap Replication A to the unused accurate third seed
     `add_d6_l2_h3_t20K_s572091`, or state that replication is across *size* not
     *seed* and defer seed-universality to backlog B5.
  6. Success is stated on the embedding only, but A1 predicts structure in
     *both* embed and unembed. Either make embed↔unembed agreement (Metric 5) a
     named sub-condition of "A1 confirmed," or downgrade embed-only success to
     "partial."
  7. Define "raw-vs-LN disagreement" precisely (a change in classified read
     R1/R2/R3/R4 or a per-component share crossing the 40%/25% bands), so a
     marginal LN wobble cannot neutralize a strong raw signal.
  8. Tie the "≤ 1 adjacency violation" ordering criterion to the permutation
     null (p-value on the adjacency/wrap-around statistic) rather than an
     asserted count.
  9. Pre-register the top-2 PC plane as primary for the ordering verdict
     (theory-neutral); fitted-k plane secondary/confirmatory.
  10. Decision Impact says "raise A1 confidence" for R1/R2 — ensure that happens
      only after the post-result gate; the note itself only *scores* the
      prediction.
  11. Doc self-sufficiency adequate, two gaps: state the design's rank-9
      saturation in-note; and `d_model = 510` / digit-token indices 0–9 are
      asserted here but not verifiable from [hugging_models.md](../hugging_models.md)
      — add them to the model contract or cite the config source.

  Conditions before launch: **C-1** (Finding 1, per-component thresholds +
  R1/R2/R3 decision rule) and **C-2** (Finding 2, partialling convention) must
  be fixed or explicitly overruled. C-3, C-4, C-5 (Findings 3, 6, 7) should be
  fixed; Findings 4, 5, 8, 9, 10, 11 may be adopted or overruled each with a
  one-line justification.

  *Status: RESOLVED 2026-07-14 by the working thread. C-1 and C-2 fixed by
  amendment; C-3/C-4/C-5 fixed; Findings 4/5/8/9/11 adopted; Finding 10 noted.
  See [Amendments](#amendments-post-skeptic-pre-launch) below. Gate PASSED —
  launch authorized.*

## Amendments (post-skeptic, pre-launch)

Dated amendments made by the working thread to resolve the pre-launch skeptic
gate, before any data was analyzed. These supersede the conflicting pre-run text
above where noted.

**2026-07-14 — A-1 (resolves BLOCKING C-1, Finding 1): thresholds are
per-component, not joint.** The verdict statistic is the variance share of a
single named low-order component, never the joint span of all regressors (which
is rank-9 and trivially explains ~100%). Pre-registered per-read statistics on
the centered 10-vector set, expressed as fraction of total centered variance:

- **R1 (circular)** statistic = variance share of the **frequency-1 plane**
  (the `cos(2πd/10), sin(2πd/10)` pair, 2 dims). A distant-second frequency-2
  share is recorded as a descriptor but is not the R1 statistic.
- **R2 (helix)** statistic = frequency-1-plane share **plus** the *unique*
  (partialled) linear share from A-2 below.
- **R3 (line-only)** statistic = the unique linear share from A-2.
- Decision rule (evaluated only if the R1-or-R3 component clears the
  minimal-effect band and beats its permutation null at p < 0.01):
  - **R1** if freq-1 share ≥ unique-linear share **and** wrap-around present
    (A-4 ordering test passes);
  - **R3** if unique-linear share > freq-1 share **and** wrap-around absent;
  - **R2** if both freq-1 and unique-linear shares individually clear the 25%
    "partial" band and neither dominates the other by more than 2×;
  - **R4** if no component clears the 25% band at p < 0.01.
- Minimal effect of interest, restated per-component: the R1 (or R3) statistic
  ≥ **40%** of centered variance at permutation **p < 0.01**, in ≥ 2 of 3
  accurate models. 25–40% is "partial", not a positive verdict.

**2026-07-14 — A-2 (resolves BLOCKING C-2, Finding 2): fixed partialling
convention for linear vs low-frequency.** Over 10 points the linear ramp `d`
projects onto low Fourier frequencies, so shares are not additive. Convention,
pre-registered:

- Report the **complete marginal DFT spectrum** (orthonormal real Fourier basis,
  frequencies k = 0..5; non-constant shares sum to 100%). This is descriptor
  #1.
- Report the **unique linear share** = R² of the centered ramp `d` after
  orthogonalizing it against the frequency-1 plane (i.e. the part of the linear
  trend not already captured by the circle). This is the R3/helix statistic.
- R2 vs R3 vs R1 is decided by A-1's rule using the freq-1 share and this unique
  linear share. Both marginal and unique quantities are reported so the
  decomposition is auditable; neither is silently double-counted.

**2026-07-14 — A-3 (resolves C-3, Finding 3): positive-control calibration
arm.** The pure-noise arm of the positive control is run through the *exact
label-permutation pipeline* used on real data; the empirical false-positive rate
of the freq-1 and unique-linear statistics at α = 0.01 must be ≤ 0.02 over ≥ 200
independent isotropic-noise draws. A miscalibrated permutation test (FPR
materially above α) voids the run as `invalid`.

**2026-07-14 — A-4 (resolves Finding 8, refines ordering test): ordering tied to
its null.** The wrap-around / circular-order verdict uses a permutation-null
p-value, not the asserted "≤ 1 adjacency violation". Statistic: sum over the 10
digits of squared angular gap between consecutive-by-value digits in the primary
plane; wrap-around statistic: distance(9,0) / mean adjacent-digit distance.
"Ordering present" requires permutation p < 0.01 on the angular-order statistic.
The old "≤ 1 adjacency violation" is retained only as a human-readable
descriptor.

**2026-07-14 — A-5 (resolves Finding 9): primary ordering plane is the top-2 PC
plane** (theory-neutral); the fitted frequency-1 plane is secondary/confirmatory.
Disagreement between them is reported as a caveat, not a pass.

**2026-07-14 — A-6 (resolves C-4, Finding 6): A1 scoring requires embed AND
unembed.** "A1 confirmed" requires the structured verdict (R1/R2) to hold in
*both* `W_E` and `W_U` with consistent digit ordering (Metric 5 principal angles
small, orderings agree). Embed-only structure scores A1 **partially confirmed**,
not confirmed. Unembed reported with its own per-component statistics, not merely
"alongside".

**2026-07-14 — A-7 (resolves C-5, Finding 5): replication is across size, and
seed diversity is strengthened.** Replication A `add_d6_l2_h3_t15K_s372001`
shares seed `s372001` with the primary, so it is primarily a *size* replication.
To add genuine seed diversity, the accurate third seed
`add_d6_l2_h3_t20K_s572091` (per [hugging_models.md](../hugging_models.md)) is
added as **Replication C**. The verdict now requires the structured read to hold
in ≥ 2 of {primary, Rep-A size, Rep-B seed s173289, Rep-C seed s572091} accurate
models, of which at least one must be an independent seed (Rep-B or Rep-C). This
prevents a same-seed-only pass.

**2026-07-14 — A-8 (resolves C-5 / Finding 4): anisotropic positive-control
arm.** In addition to the isotropic-noise arms, the positive-control sweep
includes one arm with **spectrum-matched anisotropic noise** (noise covariance
matched to the untrained-model embedding spectrum) at the 40% planted-structure
level, to show detection and permutation-null calibration survive realistic
nuisance geometry.

**2026-07-14 — A-9 (resolves Finding 7): LN-disagreement defined.** "Raw-vs-LN
disagreement → ambiguous" fires only when the two pipelines produce a different
**classified read** (R1/R2/R3/R4) or when the R1/R3 statistic crosses a
threshold band boundary (the 25% or 40% line). Numeric drift that leaves the
classified read and band unchanged is not a disagreement.

**2026-07-14 — A-10 (Finding 11 / self-sufficiency): model dimensions now cited.**
`d_model = 510`, `d_vocab = 15` (indices 0–9 = digit tokens `0`–`9`; 10–14 =
`+ - * / =`), `d_head = 170`, `d_mlp = 2040`, `n_ctx = 19` for the 5-digit
model, all read directly from `MathsConfig` (`quanta_maths/maths_config.py`,
`get_HookedTransformerConfig`) after `set_model_names(...)`. Verified against the
downloaded `.pth` state dict (`embed.W_E` shape `(15, 510)`, `unembed.W_U` shape
`(510, 15)`). Finding 10 noted: Decision-Impact "raise A1 confidence" happens
only after the post-result gate; this note only *scores* predictions.

- **Decision impact**:
  - **R1/R2 (circular/helix)**: A1's embedding prediction `confirmed`;
    C1's "near-linear" wording gets its first refinement candidate (curved
    content inside a small subspace). After the post-result gate: raise A1
    confidence in [maths-conjectures-agent.md](../maths-conjectures-agent.md),
    add the thread's first claim to
    [maths-claim-evidence.md](../maths-claim-evidence.md), start the results
    summary story, keep agenda order (entry 2 proceeds with geometry
    vocabulary), and promote backlog B1 (causal test) toward the queue.
  - **R3 (line-only)**: A1 split verdict (magnitude `confirmed`, circle
    `refuted` at embedding level); Zhou et al.'s from-scratch caveat gains
    weight; entry 2's design should probe sum-linearity rather than rotation;
    agenda re-rank likely mild.
  - **R4 (unstructured)**: A1 embedding prediction `refuted`; lookup
    alternative strengthened; entry 3's interpretation shifts (categorical
    codes become the default expectation); B1 demotes; consider promoting B2
    (MLP mechanism zoom-in).
  - **Ambiguous**: no conjecture updates; define the follow-up (LN-folding
    resolution or more seeds — promotes backlog B5) before re-running.

- **Risks / confounds**:
  - **Small-n geometry**: any 10 points admit some low-dimensional
    description. Mitigation: the label-permutation null tests label
    alignment specifically; thresholds pre-stated above.
  - **LayerNorm distortion**: raw `W_E` may not be the effective geometry.
    Mitigation: paired raw + LN-aware analyses; disagreement → ambiguous.
  - **Basis leakage**: a linear trend over `0..9` projects onto low circular
    frequencies. Mitigation: joint regression with explicit shared-variance
    reporting rather than sequential fitting.
  - **Embed/unembed conflation**: digits-as-inputs and digits-as-outputs may
    differ. Mitigation: never averaged; reported separately.
  - **Loading errors masquerading as negative results**: mitigated by the
    per-model accuracy verification gate.
  - **Interpretation trap**: structure present ≠ structure used. The verdict
    is explicitly about representation, not causal use (B1's job); the study
    note and any claim wording must preserve this distinction.
  - **Aliasing**: with 10 points, frequencies above k = 5 are aliases;
    the basis stops at k = 5 and says so.

- **Expected artifacts**:
  - A small standalone analysis script (or notebook section), runnable
    locally on CPU — no Colab or GPU required.
  - Per model and per matrix (embed/unembed): singular-value spectrum plot,
    per-component variance-share table (frequencies k = 1..5 + linear),
    dominant-plane scatter with digit labels, permutation-null histograms
    with observed statistics marked, embed-vs-unembed principal-angle table.
  - Positive-control sweep results (detection curve over planted variance
    share).
  - A metrics JSON per model for machine-readable prediction scoring.
  - All artifacts land in a local results folder for this study and are
    referenced from this note's Run record; no Hugging Face uploads are
    needed (analysis-only; no change to the
    [hugging_models.md](../hugging_models.md) folder contract).

## Post-run (fill in after the experiment)

- **Executive summary**: Across all four accurate addition models the digit
  **embedding** carries a *weak, real, but sub-threshold* frequency-1 (circular)
  signal (~0.27 of centered variance, permutation p < 0.01 in 3 of 4) that sits
  in the pre-registered **25–40% "partial" band**, never reaching the 40%
  minimal-effect bar. The **unembedding** is unstructured everywhere (R4). The
  embedding spectrum is nearly **flat across 9 dimensions** (participation ratio
  ≈ 8.7 of a max 9) — the opposite of the "low effective dimension" the agenda
  expected — and an **untrained control shows freq-1 ≈ 0.23**, so training adds
  only ~4 points of circular structure above random-geometry baseline. Verdict:
  **AMBIGUOUS (partial structure)**, leaning against a clean geometric digit
  code at the token level. Most important caveat: this is raw-weights,
  pre-LayerNorm geometry of a tiny 10-token vocabulary; token-level embedding
  geometry is not the same as the geometry the *computation* uses (that is
  backlog B1, and agenda entries 2–3 look at activations, not weights).

- **Run record**:
  - Command: `PYTHONPATH=. python3 scripts/digit_embedding_geometry.py control`
    then `... models`.
  - Script: [`scripts/digit_embedding_geometry.py`](../../scripts/digit_embedding_geometry.py)
    (standalone, CPU-only, no Colab/GPU).
  - Environment: python 3.13.7, macOS-26.5.2-arm64, numpy 2.3.2, torch 2.8.0,
    scikit-learn 1.7.1, transformer_lens (installed). Repo commit `8c2b506`
    (working tree; this note + agenda + script + results untracked at run time).
  - Dates: 2026-07-14.
  - Weights: `PhilipQuirke/VerifiedArithmetic` via `huggingface_hub`
    (`add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t15K_s372001`,
    `add_d6_l2_h3_t20K_s173289`, `add_d6_l2_h3_t20K_s572091`,
    `add_d5_l1_h3_t30K_s372001`; untrained control freshly initialized via
    `MathsConfig`, `device=cpu`, `seed=999`).
  - Artifacts (local; no HF upload):
    `results/study-digit-embedding-geometry/positive_control.json`,
    `model_geometry.json`, `embed_variance_spectra.png`, `embed_pc_planes.png`.
  - `N_PERM = 10000`, `α = 0.01`, permutation RNG seed `20260714`.

- **Results**:
  - **Positive control PASSED** all three requirements (A-3, A-8):
    (i) recovers planted variance (circle: planted 0.40 → recovered freq1 0.518;
    monotone across 0.2–0.9); (ii) detects planted circle at the 40% level with
    p ≈ 0.0005 (detected from planted ≥ 0.30); (iii) calibration on pure noise
    within tolerance — isotropic FPR 0.005 (freq1) / 0.020 (unique-linear),
    anisotropic FPR 0.005, all ≤ 0.02 at α = 0.01. Note: planted *helix* arms
    classify as "partial/ambiguous" rather than R2 at low shares — the classifier
    is conservative about calling helix, which is acceptable for a
    detection/calibration control. The instrument works and is not trigger-happy.
  - **Embedding freq-1 (circular) share** (verdict statistic for R1): primary
    0.271 (p = 0.0001), repA_size 0.272 (p = 0.0001), repC_seed 0.275
    (p = 0.0002), repB_seed 0.237 (p = 0.0385). All in the 25–40% partial band;
    **none ≥ 40%**. Independent-seed repB is the weakest and only marginally
    beats the null.
  - **Unique-linear share** (R3/helix statistic): 0.07–0.11 everywhere,
    p ≈ 1.0 (never significant). No magnitude line. R2/R3 ruled out.
  - **Untrained control**: freq1 0.233 (p = 0.11, n.s.), unique-linear 0.105 —
    i.e. ~0.23 freq1 is the *baseline* from 10 random points in 510-dim; trained
    models add only ~+0.04.
  - **Effective dimension**: embedding variance shares are near-flat
    (~0.15 → 0.08 over 9 non-null dims); participation ratio 8.2–8.8 (max 9) for
    both trained and untrained. Digit embeddings are close to **isotropic in 9D**,
    not low-dimensional.
  - **Marginal DFT** is spread almost evenly across frequencies (k1 ≈ k2 ≈ k3 ≈
    k4 ≈ 0.21–0.27); the k1 excess in trained models is small.
  - **Ordering**: the *primary* model shows significant circular value-ordering
    in the top-2 PC plane (angular-order p = 0.0001) despite low freq1 — a weak
    real circular tendency — but this is inconsistent across models (repB
    p = 0.045, repC embedding "partial/ambiguous", untrained p = 0.34) and
    wrap-around ratios are erratic (0.41–1.45).
  - **Unembedding**: R4 in all trained models (freq1 0.21–0.30, mostly n.s.);
    the inaccurate 1-layer model's unembed is the most structured (0.301,
    p = 0.0015) — no monotone accuracy→geometry trend.
  - **Embed vs unembed principal angles** are large (36–88°): the two geometries
    do **not** agree in form or orientation (A-6 sub-condition for "A1 confirmed"
    fails).

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **Success condition NOT met**: it required a structured basis ≥ 40% at
    p < 0.01 in ≥ 2 of 3 accurate models with consistent ordering. Observed peak
    is ~0.27 (partial band), unembed unstructured, embed/unembed disagree.
  - **Failure condition NOT met either**: it required *no* basis reaching 25% at
    p < 0.01 in **all** accurate models. But freq1 does clear 25% at p < 0.01 in
    3 of 4 accurate embeddings. So this is not a clean R4 "no structure" verdict.
  - Therefore the pre-registered verdict is **AMBIGUOUS** (best structured
    component lands in the 25–40% band; A-1). Sub-classification: a **weak
    partial circular tendency in the embedding only**, not helical, not linear,
    not present in the unembedding. The positive control passed, so this is a
    genuine "partial" — not `invalid`/`underpowered`.
  - Substantively: the token-level digit code is closer to a **near-isotropic
    9-dimensional categorical code** than to a clean low-dimensional circle/helix.
    The "low effective dimension" premise behind C1/A8 is **not** supported at
    the embedding level.
  - **Post-gate correction (C-A/C-B).** The circular evidence splits into two
    metrics that must be read separately:
    - *Variance share* (freq1 ≈ 0.27): **at the noise floor.** The untrained
      control gets 0.233; trained adds only ~+0.03. The positive control shows
      the freq1 metric is inflated by +0.1–0.2 at low structure (planted 0.40 →
      recovered 0.518), so the trained ~0.27 lies below even a 20%-planted
      circle's recovered value (0.397). The share metric is **not** evidence of
      trained circular structure.
    - *Angular ordering* (top-2 PC plane): a **genuine, baseline-clean,
      replicated training signal.** Significant (p = 0.0001) in 3 of 4 accurate
      models (primary, repA, repC) and absent in the untrained control
      (p = 0.337). This is the one place A1's "circular order" prediction
      survives — as an ordering tendency, not as variance concentration. It is
      **provisional pending the LN-aware secondary analysis (A-9)**, which was
      not run this pass and to which ordering is the most sensitive metric.

- **Prediction scoring** (records evidence only; does NOT update conjectures —
  that waits for the post-result skeptic gate and the phase-6 update):
  - **A1** ("digit embedding geometry dominated by a few components with
    circular order; same structure in unembedding; sub-task outputs align")
    — re-scored post-gate (C-C):
    - **Strong / dominant-geometry form: REFUTED at the embedding-token level.**
      The spectrum is near-isotropic (participation ratio 8.7/9, not "a few
      components"); the freq1 variance share is at the noise floor (C-B); the
      unembedding is unstructured (R4) and misaligned with the embedding
      (principal angles 36–88°, A-6 fails). Helix/linear (R2/R3) refuted —
      unique-linear share never significant.
    - **Circular-*ordering* existence sub-claim: PARTIALLY CONFIRMED
      (provisional, raw-weights only, pending A-9).** Basis: significant angular
      ordering (p = 0.0001) replicated in 3/4 accurate models and absent in the
      untrained control — a real, baseline-free, training-induced circular
      *ordering*, distinct from (and not rescued by) the noise-floor share
      metric.
    - **Falsifier** ("near-orthogonal, no consistent circular structure"):
      **not met** — the ordering is consistent in 3/4 accurate models.
    - Net: A1's specific "circle *dominates* the geometry" reading is wrong at
      the embedding level; A1's weaker "digits carry a circular ordering" reading
      holds provisionally. Confidence in the dominant-geometry form should drop;
      low-medium confidence that *some* circular ordering exists is warranted.
  - **A8** ("sharp low-rank structure; low effective dimension"): **refuted at
    the embedding level** — participation ratio ≈ 8.7/9, near-flat spectrum.
    (A8 also makes activation-level claims this study did not test — those are
    **untouched**.)
  - **C1 (human)** ("simple, near-linear, low effective-dimension;
    ~one direction per feature"): the "low effective-dimension" clause is
    **refuted for digit embeddings**; the broader "simple/near-linear" claim is
    **untouched** (this study is weights-only, not a probe of computed features).
  - All other predictions (A2–A7, C2, C3): **untouched** — out of scope of a
    weights-only embedding study.
  - Streak note: this is the thread's first study; one refuted-leaning result,
    no streak yet.

- **Skeptic review (post-result)**: Run 2026-07-14 in a separate thread that
  rehydrated only from the version-controlled docs and the two result JSONs and
  did not inherit the working thread's context.

  **Verdict: PASS WITH CONDITIONS.** The reporting is factually accurate (all
  numbers verified against the JSONs), the positive control genuinely passed its
  three requirements, and "AMBIGUOUS" is correct by the pre-registered rules
  (Failure/R4 provably did not fire since freq1 clears 25% at p<0.01 in 3 of 4;
  invalid/underpowered excluded because the control passed). **But** the
  interpretation credited the wrong metric. Four blocking re-scoring conditions:

  - **C-A** — The freq1 *variance-share* and *angular-ordering* metrics tell
    opposite stories and the note leaned on the weaker one. freq1 share is
    **baseline-dominated**: untrained = 0.233, trained ≈ 0.26 (**+0.03**); the
    permutation p<0.01 largely reflects label-alignment of incidental
    low-frequency mass that 10 random points in 510-D already carry. Angular
    *ordering* is the genuine training signal: significant (p = 0.0001) in **3 of
    4** accurate models (primary, repA, **repC**) and **absent in the untrained
    control** (p = 0.337). The note mislabeled repC's ordering as
    "partial/ambiguous" — that is repC's per-model *classifier read*; repC's
    actual ordering p = **0.0001**. Fix the baseline split and the repC report.
  - **C-B** — Positive-control **inflation**: planted circle 0.40 → recovered
    freq1 0.518 (leakage +0.1–0.2 at low structure). So the 40% recovered bar
    corresponded to only ~0.25–0.30 *true* planted structure, and the trained
    ~0.27 sits **below** even a 20%-planted circle's recovered value (0.397) —
    i.e. the trained variance-share circular signal is **at the noise floor**.
    Say so; it makes the share-based read more negative.
  - **C-C** — Re-score A1: the surviving evidence is *ordering*, not *share*.
    Drop the ~0.27 share as support for the surviving claim.
  - **C-D** — The LN-aware secondary analysis (A-9) was not run, and ordering is
    the metric most sensitive to LN. The strong-form / A8 / share refutations are
    robust to LN and may be recorded durably, but the A1 ordering
    partial-confirmation must be marked **provisional pending A-9**.

  Non-blocking: A8 "refuted at embedding level" is fair and correctly scoped;
  the C1 split is defensible (mark the refutation narrow — digit embeddings, not
  feature activations). No causal/scope over-reach. "Next read" endorsed, and
  strengthened: since the only surviving signal is a correlational *ordering*,
  the activation/MLP story (A2/A3, entry 2/3) and the B1 causal test become more
  central; sequence B1 after A-9. Suggested close-out: entry 1 completes as
  `question answered (ambiguous; strong-form A1 refuted)`; add "complete A-9
  LN-aware" as a required entry-1 close-out before de-provisionalizing the claim.
  Minor artifact gap: per-model instrument-verification accuracy is asserted in
  Design but not stored in `model_geometry.json` (recommend adding).

  *Status: RESOLVED 2026-07-14 by the working thread. C-A, C-B, C-C, C-D applied
  as corrections to Interpretation and Prediction scoring below (goalposts
  unchanged — the run verdict stays AMBIGUOUS; only evidence attribution and
  prediction wording changed). Gate 2 PASSED. Router-doc updates authorized,
  with the A1 ordering sub-claim recorded provisionally pending A-9.*

  **Corrections applied (C-A..C-C):**
  - freq1 variance share is baseline-dominated (+0.03 over untrained) and, per
    the positive-control calibration, at the noise floor — it is **not** evidence
    of trained circular structure.
  - The baseline-clean, replicated training signal is **angular ordering**:
    p = 0.0001 in primary, repA, and **repC** (correcting the earlier repC
    mislabel), versus p = 0.337 untrained. repB is the lone accurate model
    without it (p = 0.045).
  - A1 re-scored on ordering, not share (see Prediction scoring).

- **Limitations**:
  - Weights-only: says nothing about whether the model *uses* any geometry
    (causal — B1) or about activation-level feature geometry (entries 2–3).
  - n = 10 vectors; power comes from the permutation null + replication, not n.
    The untrained baseline of ~0.23 freq1 shows 10 points in high-dim always
    carry incidental low-frequency mass; the pre-registered minimal effect was
    defined relative to 0, not relative to that baseline — a design choice worth
    revisiting (flagged for the post-result skeptic).
  - Raw pre-LN weights only for the verdict; the planned LN-aware secondary
    analysis (A-9) was **not** run this pass — noted as a follow-up, so the
    raw-vs-LN ambiguity check is incomplete. This does not change the verdict
    (raw already ambiguous) but should be completed before any strong claim.
  - Tri-state `ST`/`SV` and operator/sign geometry are explicitly out of scope.

- **Doc updates**: This study note only (results + interpretation + prediction
  scoring recorded). **No** conjecture, claim-evidence, results-summary,
  results-synthesis, ledger, or agenda updates yet — blocked on the post-result
  skeptic gate per the phase rules. The LN-aware secondary analysis is added as
  an outstanding item for this line.

- **Next read**: Do **not** freeze. (a) Run the post-result skeptic gate. (b)
  Complete the LN-aware secondary analysis (A-9) to finish the pre-registered
  design. (c) If the skeptic upholds the ambiguous/R4-leaning read, the natural
  next step is to test whether the *computation* uses digit magnitude/circularity
  regardless of a clean embedding geometry (promote B1 causal test and/or agenda
  entry 2 activation geometry) — a clean embedding circle was only ever one route
  to numeric computation, and its weakness makes the activation- and
  MLP-level story (A2/A3) more central, not less.
