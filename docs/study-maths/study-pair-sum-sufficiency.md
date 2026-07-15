# Study: Pair-Sum Sufficiency at an ST Node (study-pair-sum-sufficiency.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #1

**Verdict: `instrument failure` — A2 untested.** The assay mis-targeted
and its headline `R²_sum/R²_pair` ratio (~0.20)
proved non-discriminating.

The one durable by-product (replicated over 6 nodes, 2 models, recorded as CE2):
at those operand-fetch heads the value-path output is indistinguishable from
linear transport of the weakly-circular digit embeddings — extending the
digit-embedding "structure is a weak low-variance projection" motif to the value
path. This line was frozen.

## Pre-run (write before the experiment)

- **Question**: At an addition model's `ST` node (a head + adjacent MLP that
  Paper 2 finds *jointly* necessary), does the **head output** (value-path,
  pre-MLP) depend on the two operand digits `Dn, D'n` essentially *only through
  their sum* `Dn + D'n` (an ordered ~1-D arc over the 19 possible sums, with the
  `U`/sum-9 case at the arc's transition), while the discrete 3-cluster
  `{0, 1, U}` [ST](../thor-glossary.md#st) shape appears only **after the MLP**?
  I.e. is the circuit *aggregate-then-discretize*
  ([A2](../maths-conjectures-agent.md#a2-the-core-computation-is-aggregate-then-discretize))?

- **Motivation**: A2 is the second-sharpest fork in the agent conjectures:
  whether attention *performs* the per-digit addition in embedding space (linear
  value path sums the operands) or merely *transports* operands for later
  combination. It also promises a mechanism for Paper 2's otherwise-unexplained
  "head and MLP both necessary at ST nodes" observation. After the
  digit-embedding study found little token-level geometry, locating where the
  categorical `ST` shape is *created* is now the central open question, and it
  feeds entry 2 (full-space ST geometry).

- **Hypothesis / competing reads** (neutral; all live):
  - **H-A2 (aggregate-then-discretize)**: pre-MLP head output is a function of
    `Dn + D'n` to good approximation — equal-sum digit pairs (e.g. 3+6, 4+5,
    2+7) are near-identical there; the representation is a low-dimensional
    ordered arc in sum; the 3-way `{0,1,U}` clustering is weak/absent pre-MLP and
    strong post-MLP; `U` (sum=9) sits at the arc location between the sum≤8 and
    sum≥10 regions.
  - **H-early (attention already categorical)**: the head output is already
    organized by the tri-class `{0,1,U}` (or by carry bit), not by the ordered
    sum — attention/its value path does the discretization, not the MLP.
  - **H-transport (separate operands)**: the head output does *not* have
    `Dn + D'n` as a sufficient statistic — equal-sum pairs are *not* collapsed;
    operands are carried separately (e.g. the head mostly copies one operand, or
    keeps `Dn` and `D'n` on separable directions) and combination happens
    elsewhere.
  - Mixtures possible (partial sum-collapse + residual operand identity); the
    design quantifies by variance decomposition rather than forcing a binary.

- **Design**:
  - **Model**: primary `add_d5_l2_h3_t15K_s372001` (accurate 5-digit, 2-layer,
    3-head; the Hypothesis-3 node map names its `ST` nodes). Weights from
    [PhilipQuirke/VerifiedArithmetic](https://huggingface.co/PhilipQuirke/VerifiedArithmetic)
    via `MathsConfig` + TransformerLens `HookedTransformer`, `device=cpu`.
    Replication on `add_d6_l2_h3_t20K_s173289` (accurate, independent seed) to
    avoid a single-model verdict.
  - **Node selection (data-driven, not assumed)**: candidate `ST` nodes are the
    layer-0 heads Paper 2 App. Hypothesis-3 lists for this model class
    (`P8.L0.H1`, `P9.L0.H1`, `P11.L0.H2`, `P14.L0.H1`). We confirm a node is an
    `ST` node *before* analyzing it: at its token position, a PCA of the head
    output over tricase questions for the relevant digit must show the 3-cluster
    `{0,1,U}` structure (Paper-2 criterion). Only confirmed `ST` nodes drive the
    verdict. We report which digit `n` each confirmed node computes.
  - **Stimuli**: for the target digit `n` of a confirmed `ST` node, build a
    question set that **varies `Dn` and `D'n` across all pairs** while holding
    other digits fixed to a neutral pattern that does not itself induce carries
    into position `n` (controls below). Each of the 100 ordered pairs
    `(Dn, D'n) ∈ {0..9}²` is instantiated in ≥ 8 questions with randomized
    other digits (no lower carry into `n`), giving pair sums 0..18. This lets us
    separate "sum" from "identity of the operands".
  - **Captured activations**: `blocks.0.attn.hook_z[:, pos, head, :]` (head
    output, pre-MLP; the value-path signal A2 is about) and, for the post-MLP
    comparison, `blocks.0.mlp.hook_post[:, pos, :]` and
    `blocks.0.hook_resid_post[:, pos, :]` at the node's token position.
  - **Metrics**:
    1. **Sum-sufficiency (headline)**: fraction of head-output variance
       explained by a 19-level one-hot of `Dn + D'n` (between-sum variance /
       total), versus the fraction explained by the full 100-cell
       `(Dn, D'n)` one-hot. If sum is sufficient, the two are nearly equal and
       the *residual* (100-cell minus 19-sum) is small. Report
       `R²_sum`, `R²_pair`, and the **collapse ratio** `R²_sum / R²_pair`.
    2. **Equal-sum collapse vs matched control**: mean within-equal-sum pairwise
       distance of head outputs versus a matched control — pairs drawn to have
       *different* sums but the same |Dn − D'n| spread (guards against the
       collapse being a trivial artifact of averaging). Report the ratio.
    3. **Ordering / dimensionality of the sum arc**: PCA of the 19 sum-mean
       vectors; is variance dominated by 1–2 components ordered monotonically in
       sum? Participation ratio; monotonicity (Spearman of PC1 vs sum).
    4. **U location**: where the sum=9 mean sits on the arc relative to sum≤8 and
       sum≥10 means (between, or already offset off-axis?).
    5. **Pre-MLP vs post-MLP tri-cluster strength**: 3-way `{0,1,U}` cluster
       separation (silhouette / between-over-within class variance) computed on
       head output vs on MLP post vs on resid_post. A2 predicts monotone
       increase head → MLP.
  - **Controls / nulls**:
    - **No-lower-carry control**: other digits chosen so no carry propagates
      *into* position `n` — otherwise the effective local sum is `Dn+D'n+carry`
      and confounds the sum bins. Verified per question.
    - **Position control**: repeat the sum-sufficiency metric at a *non-ST*
      layer-0 head at the same token position (e.g. a head Paper 2 tags as `SA`
      or unused). If sum-sufficiency is just a generic property of any head
      output here, it will appear there too — that would weaken an A2-specific
      read.
    - **Label-shuffle null**: permute the `Dn+D'n` labels across questions and
      recompute `R²_sum`; the permuted value is the chance floor for the
      variance-explained metric given the sample.
    - **Fixed-other-digit replication**: rerun with a *second* neutral filler
      pattern for the non-target digits to confirm the verdict is not an artifact
      of one filler choice.
  - **Minimal effect of interest**: for H-A2 to be supported, `R²_sum ≥ 0.80`
    of `R²_pair` (sum explains ≥ 80% of what the full pair identity explains) at
    the confirmed `ST` node, **and** the pre→post-MLP tri-cluster separation
    increases by a pre-registered margin (post-MLP silhouette ≥ 0.5 while
    pre-MLP ≤ 0.3), replicated across the two models. A collapse ratio between
    0.5 and 0.8 is "partial" (operands partly but not fully summarized by sum).
  - **Sample sizes / power**: ≥ 8 questions per `(Dn,D'n)` cell × 100 cells =
    ≥ 800 questions per (model, node, filler); this resolves per-cell means with
    low error and gives the label-shuffle null a tight chance floor. Power comes
    from the large question count and the matched control + shuffle null, not
    from asymptotics.

- **Positive control**: two planted references run through the identical
  pipeline. (a) **Sum-sufficient reference**: a synthetic "head output" defined
  as `f(Dn+D'n) + small noise` for a smooth ordered `f` — the pipeline must
  report `R²_sum/R²_pair ≈ 1`, a 1–2 D ordered arc, and near-zero residual.
  (b) **Operand-identity reference**: a synthetic signal
  `g(Dn) + h(D'n)` with `g,h` independent random per-digit codes (sum is NOT
  sufficient because the pipeline cannot recover sum from independent codes
  unless it collapses) — the pipeline must report `R²_sum/R²_pair` clearly < 1
  and detect residual operand-identity variance. (c) **Categorical reference**:
  a signal that is a clean function of the tri-class `{0,1,U}` only — pre-MLP
  metric must classify it as "already categorical" (H-early), confirming the
  metrics can distinguish the three reads. A flat/ambiguous result on real data
  whose positive control also fails to separate the three references is
  `invalid`/`underpowered`, never a verdict.

- **Success condition** (defined now): at a confirmed `ST` node in **both**
  models, `R²_sum ≥ 0.80 · R²_pair`, equal-sum collapse ratio well below the
  matched control, the sum arc is 1–2 D and monotone in sum, and tri-cluster
  separation rises from pre-MLP (≤ 0.3) to post-MLP (≥ 0.5). Verdict:
  **H-A2 supported** — attention aggregates by sum, MLP discretizes.

- **Failure condition** (defined now): with the positive control passing —
  in both models the head output is *already* strongly tri-class-clustered
  pre-MLP (pre-MLP silhouette ≥ 0.5, no pre→post increase) → **H-early**; OR
  `R²_sum < 0.5 · R²_pair` with substantial residual operand-identity variance
  and no equal-sum collapse → **H-transport**. Either scores A2's core
  prediction refuted.

- **Ambiguous / invalid condition**:
  - Ambiguous: collapse ratio 0.5–0.8; or models disagree; or the
    pre→post-MLP separation change is present but below the pre-registered
    margins; or the position-control (non-ST head) shows equal sum-sufficiency
    (verdict not ST-specific).
  - Invalid: positive control fails to separate the three planted references;
    or no candidate node confirms as an `ST` node (cannot locate the assay's
    unit); or the loaded model fails an accuracy check.

- **Skeptic review (pre-launch)**: Run 2026-07-14 in a separate skeptic thread
  that rehydrated only from the version-controlled docs and the Paper-2 node
  facts and did not inherit the working thread's context.

  **Verdict: PASS WITH CONDITIONS.** Design is neutrally framed and compliant
  (positive control with false-positive arm, shuffle null, minimal-effect
  statement, flat+failing-control → invalid). Four BLOCKING gaps that would
  otherwise permit a non-updating "gameable pass":

  - **B1 (most important)** — The positive control does not fault-inject the one
    artifact that fakes H-A2 under H-transport: a **linear value path over
    circular digit embeddings**. Since `cos a + cos b = 2 cos((a−b)/2)
    cos((a+b)/2)`, merely *transporting* (equal-weight summing) two circular
    codes yields genuine sum-dependence with no aggregate-then-discretize
    computation. The prior digit-embedding study found exactly such a weak
    circular component, so a high `R²_sum/R²_pair` alone does **not** discriminate
    H-A2 from H-transport.
  - **B2** — `R²_sum/R²_pair` is biased: the 100-cell denominator is inflated
    ~5× more by sampling noise than the 19-level numerator; and the shuffle null
    was specified on `R²_sum`, not on the ratio.
  - **B3** — Selecting `ST` nodes by 3-cluster PCA on the *head output* and then
    measuring pre-MLP tri-cluster on the same object is circular; it can
    manufacture a false H-early verdict.
  - **B4** — Attention-equal-weight on the two operand positions is the
    precondition for the whole pair-sum frame but was only a "risk"; unequal
    attention depresses `R²_sum` for reasons unrelated to A2-vs-transport.

  Should-fix: N1 (add LN'd MLP-input as a measurement point), N4 (calibrate
  silhouette thresholds against positive-control reference values), N6
  (correlational-not-causal verdict wording). Doc gaps D1 (state tricase
  case→class mapping and `d_head`).

  *Status: RESOLVED 2026-07-14 by the working thread. B1–B4 fixed and N1/N4/N6
  adopted by the amendment below; goalposts (the run's competing reads) are
  unchanged — the amendment makes the assay *discriminating*, it does not tilt
  toward a desired answer. Gate 1 PASSED — launch authorized.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread to resolve the Gate-1 blockers, before
any data was analyzed. These supersede conflicting pre-run text where noted.

**2026-07-14 — A-1 (resolves B1): circular-transport reference is the decisive
control, and the tri-cluster increase (not the sum ratio) is the load-bearing
discriminator.** A fourth positive-control arm is added: synthesize a "head
output" as `W_V·embed(Dn) + W_V·embed(D'n)` using the model's *actual* `W_E`,
`W_V` (equal weights, no nonlinearity) — a pure H-transport signal built from
the real (weakly circular) embeddings. Pre-registered rule: if this transport
reference already yields `R²_sum/R²_pair ≥ 0.80` and a 1–2-D monotone sum arc,
then **the sum-sufficiency ratio cannot by itself distinguish H-A2 from
H-transport**. In that case (which the prior study makes likely) the
**load-bearing discriminator is the pre→post-MLP tri-cluster increase**
(Metric 5): H-A2 requires the categorical `{0,1,U}` shape to be *created by the
MLP* (weak pre-MLP, strong post-MLP), whereas H-transport predicts *no* such
increase (the MLP is not where discretization happens) and H-early predicts it
is already categorical pre-MLP. Sum-sufficiency is demoted to
necessary-not-sufficient evidence for A2.

**2026-07-14 — A-2 (resolves B2): held-out R² and correct null for the ratio.**
`R²_sum` and `R²_pair` are computed by **5-fold cross-validation** (fit group
means on train folds, score on held-out fold), so the 19-vs-100 parameter
asymmetry is not baked into the ratio; report both raw in-sample and CV values.
The **ratio's chance floors** are the operand-identity reference (independent
per-digit codes → sum not sufficient) and the new circular-transport reference
(A-1). The sum-label shuffle is retained only as the floor for `R²_sum ≠ 0`.
The 0.80 threshold is evaluated on the **held-out** ratio.

**2026-07-14 — A-3 (resolves B3): selection/measurement separation.** `ST` nodes
are selected on a **selection-independent** basis: the Paper-2 named 5-digit
positions (`P8.L0.H1`, `P9.L0.H1`, `P11.L0.H2`, `P14.L0.H1`) as the candidate
list, confirmed by **ablation impact** (does zeroing the node degrade the
digit-`n` answer?) and/or **post-MLP/resid** tri-cluster — never by pre-MLP
head-output clustering. The pre-MLP head-output tri-cluster silhouette is
measured **only as an outcome** (Metric 5). The H-early failure verdict requires
pre-MLP tri-structure **in excess of** the position-control non-ST head selected
the same way, not an absolute silhouette alone.

**2026-07-14 — A-4 (resolves B4): attention-weight gating.** For each confirmed
node, the post-softmax attention to the two operand positions (`Dn`, `D'n`) is
measured **first**. Pre-registered tolerance band: operand-attention ratio in
[0.40, 0.60] of the mass on the two operand positions. Nodes outside the band
are excluded from the headline equal-sum verdict; for them the **weighted**
sufficient statistic `w·Dn + (1−w)·D'n` (with `w` = measured attention share) is
tested and reported alongside. A node attending to neither operand position
voids the pair-sum frame for that node (reported as a finding).

**2026-07-14 — A-5 (adopts N1): LN decomposition.** Metric 5 adds a third
measurement point, so the pre→post transition is
`head output z` → `LN(resid_pre_mlp)` (the actual MLP input) → `mlp.hook_post`
→ `resid_post`. This attributes any discretization to LN vs the MLP rather than
lumping them.

**2026-07-14 — A-6 (adopts N4): thresholds calibrated to the control.** The
silhouette pass/fail is expressed **relative to the positive-control
references**: post-MLP tri-cluster must exceed the *categorical reference*
silhouette minus a margin, and pre-MLP must fall below the *sum-arc reference*
silhouette plus a margin (margins pre-registered after the control reports its
reference values, before real-data verdicts are read). The absolute 0.3/0.5
numbers become descriptors.

**2026-07-14 — A-7 (adopts N6): verdict wording is correlational.** "H-A2
supported" means *the pre-MLP representation has sum as a (near-)sufficient
statistic and the categorical shape is created downstream of the head*, a
representational claim. The causal claim "attention *performs* the addition" is
reserved for an intervention study (agenda backlog B1/B2) and must not be
asserted from this study.

**2026-07-14 — A-8 (D1 doc facts): tricase case→class mapping is
`test_case 8 → sum ≤ 8 (ST 0)`, `9 → sum = 9 (ST U)`, `10 → sum ≥ 10 (ST 1)`;
the generator enforces no lower-digit make-carry. `d_head = 170`, `d_model =
510` (from `MathsConfig`). The 6-digit replication model's `ST` node positions
are not pre-known from the docs; that model relies on data-driven
selection-independent confirmation (A-3), and failure to locate a confirmed `ST`
node makes it `invalid` for that model.**

- **Decision impact**:
  - **H-A2 supported**: score A2 confirmed; adds the thread's mechanism claim
    (attention sums, MLP discretizes) to claim-evidence; sharpens entry 2 (the
    `ST` categorical geometry is an MLP *output*, to be measured post-MLP);
    strengthens the C3 sharpening that "attention moves" undersells attention.
  - **H-early**: A2 refuted toward "attention itself discretizes"; entry 2 and
    the A3 `U`-geometry question relocate to the head output; C3 largely intact.
  - **H-transport**: A2 refuted toward late combination; raises the priority of
    finding *where* operands combine; revises the aggregate-then-discretize
    stage of the agent's overall picture.

- **Risks / confounds**:
  - Lower-digit carry leaking into the local sum (mitigated by the
    no-lower-carry control).
  - LayerNorm between head output and MLP: hook_z is pre-LN; the "MLP input" is
    LN(resid). Report head-output geometry as the pre-MLP object (what A2 names)
    and note LN as the nonlinearity boundary; do not conflate.
  - A "head output" here is `z` (pre-`W_O`); also report the `W_O`-projected
    contribution to resid to confirm the arc survives the output projection.
  - Attention pattern not actually equal-weight on `Dn`,`D'n`: measure the
    node's attention to the two operand positions; if it is not attending to
    both, the pair-sum frame does not apply and that is itself a finding.
  - Metric artifact: `R²_sum` mechanically ≤ `R²_pair` (sum is a coarsening of
    pair); the label-shuffle null and the operand-identity positive control
    guard against reading a high ratio as structure when it is averaging.

- **Expected artifacts**:
  - Standalone CPU script `scripts/pair_sum_sufficiency.py`.
  - Per (model, confirmed node, filler): `R²_sum`, `R²_pair`, collapse ratio,
    shuffle-null floor, sum-arc PCA (participation ratio, monotonicity),
    pre/MLP/post tri-cluster silhouettes, attention-to-operands.
  - Plots: head-output sum-arc (colored by sum, marked at U); pre-MLP vs
    post-MLP tri-cluster scatter; position-control comparison.
  - Positive-control separation table for the three planted references.
  - A metrics JSON for machine-readable prediction scoring.
  - Local results folder `results/study-pair-sum-sufficiency/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (rewritten post-Gate-2): The assay **mis-targeted** and
  its core metric was **non-discriminating**, so it did **not** test A2. The
  operand-fetch heads it confirmed are **answer-position** heads (plausibly `SA`
  base-add readout), not the question-position `ST` carry-compute nodes A2 is
  about (those Paper-2 candidates failed single-node ablation confirmation). And
  the headline `R²_sum/R²_pair` ratio (~0.20) is **quantitatively
  indistinguishable from a pure-transport null** built noise-free from the
  model's own `W_E·W_V` (ratio 0.185, same monotone sum arc, same weak
  tri-cluster): summing two weakly-circular digit embeddings *automatically*
  yields exactly this signature, so the ratio cannot separate "attention
  aggregates" from "attention transports". **Verdict: A2 untested (assay
  mis-targeted + non-discriminating metric); this line freezes as `instrument
  failure`.** The one durable, positive by-product is a replicated finding
  (6 nodes, 2 models): at these operand-fetch heads the value-path output *is*
  indistinguishable from linear transport of the (weakly-circular) embeddings —
  extending the digit-embedding study's "structure is a weak low-variance
  projection" motif to the value path.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/pair_sum_sufficiency.py control`
    then `... models`.
  - Script: [`scripts/pair_sum_sufficiency.py`](../../scripts/pair_sum_sufficiency.py)
    (standalone, CPU).
  - Env: python 3.13.7, macOS-26.5.2-arm64, numpy 2.3.2, torch 2.8.0,
    scikit-learn 1.7.1, scipy 1.16.1. Repo commit `43e0ac5` (working tree).
    Date 2026-07-14.
  - Models: `add_d5_l2_h3_t15K_s372001` (accuracy check 1.000),
    `add_d6_l2_h3_t20K_s173289` (1.000), weights via `huggingface_hub`.
  - Artifacts (local; no HF upload):
    `results/study-pair-sum-sufficiency/positive_control.json`,
    `model_results.json`, `head_output_pca.png`.

- **Results**:
  - **Positive control (incl. the A-1 circular-transport arm) PASSED and
    separates the three reads**: ref_sum_sufficient ratio_cv 1.00 /
    tricluster 0.43; ref_categorical ratio 1.00 / **tricluster 0.96**;
    ref_operand_identity ratio_cv **0.06**; **ref_circular_transport (real
    W_E·W_V, summed, no nonlinearity) ratio_cv 0.20**. The circular-transport
    null did *not* fake sum-sufficiency (the embedding's circular component is
    too weak, per the prior digit-embedding study), so the 0.80 sum bar remains
    a valid discriminator and the pre→post tri-cluster increase remains the
    load-bearing metric where sum is ambiguous.
  - **Primary model**: of the four Paper-2 candidates, `P14.L0.H1` confirmed
    (attends D3/D'3 at 0.47/0.50, ablation impact 0.31, in band). `P8/P9/P11`
    did not meet the joint confirm bar under this stimulus/ablation (P9 attends
    D2/D'2 with mass 0.97 but zero single-node ablation impact on the naive
    answer-token test — a known redundancy per Paper 2; noted as a limitation of
    the single-node ablation confirm, not a claim that P9 is not ST).
  - `P14.L0.H1`: `R²_sum_cv = 0.195`, `R²_pair_cv = 1.00` → **ratio 0.20**;
    sum-arc PC1 monotonicity 0.998, participation ratio 3.0, PC1 var share 0.52;
    equal-sum collapse-vs-control 0.95 (≈1 = little extra collapse beyond the
    matched control); tri-cluster silhouette z 0.036 → LN 0.060 → mlp 0.060 →
    resid 0.104. Shuffle floor for R²_sum ≈ 0.02.
  - **Replication (6-digit, seed s173289)**: the answer-position layer-0 heads
    `P15..P20.L0.H1` all confirmed (near-equal operand attention 0.42–0.49,
    ablation impact 0.88–0.90, in band). Every one gives **ratio_cv 0.22–0.24**,
    monotonic sum arc (mono 1.00), and weak tri-cluster rising modestly toward
    resid (z ≈ 0.01 → resid 0.06–0.14). Highly consistent across 6 nodes and 2
    models.
  - Cross-check: repeating the tri-cluster measurement on the library's own
    `make_maths_tricase_questions` stimuli (rather than the custom set) at
    `P14.L0.H1` gave the same profile (z 0.046, mlp 0.057, resid 0.158),
    confirming the weak-tricluster finding is not a stimulus artifact.

- **Interpretation** (rewritten post-Gate-2; goalposts unchanged, verdict
  corrected):
  - **The assay does not adjudicate A2.** Two independent reasons: (1)
    *mis-targeting* — the confirmed nodes sit at answer positions (`P14` for the
    5-digit model; `P15–P20` for the 6-digit model), i.e. operand-fetch heads
    consistent with `SA` base-add readout, not confirmed `ST` carry-compute nodes
    at question positions; the question-position ST candidates (P8/P9/P11) failed
    single-node ablation confirmation (a known Paper-2 redundancy limitation of
    single-node ablation, not proof they are not ST). (2) *non-discriminating
    metric* — with the no-lower-carry stimulus `R²_pair = 1.0`, so ratio =
    `R²_sum`, a fixed geometric quantity for equal-weight-summed circular codes;
    the noise-free transport null reproduces the real ratio (0.185 vs 0.195),
    arc (mono 1.0, PC1 0.55), and tri-cluster (0.04). The metric therefore cannot
    separate H-A2 from H-transport under this design.
  - **Not a clean H-A2, H-early, or H-transport verdict.** H-A2 (ratio ≥ 0.80 +
    tri-cluster rise) is not met, but that is uninformative given the above.
    H-early is not met (head output not tri-clustered). H-transport *is* fully
    consistent with the data — but "consistent with transport" is exactly what a
    non-discriminating assay yields, so it cannot be scored as a positive
    H-transport finding either.
  - **Freeze classification: `instrument failure`.** The unit of analysis
    (attention+ablation-selected operand-fetch heads) plus the no-lower-carry
    stimulus (pins `R²_pair=1`) cannot discriminate A2 from transport. Reopen
    with (a) an independently confirmed `ST` node at a *question* position
    (path-patching to the `SV` cascade, or post-MLP tri-cluster at a question
    position), and (b) a stimulus/metric that does not over-determine the ratio.
  - Durable positive by-product: at the operand-fetch heads reached, the
    value-path output is quantitatively indistinguishable from linear transport
    of the models' weakly-circular digit embeddings — a replicated (6 nodes, 2
    models) extension of the CE1 "weak low-variance projection" motif from the
    embedding to the value path. This does **not** locate where `ST`
    discretization happens.

- **Prediction scoring** (rewritten post-Gate-2; records evidence only):
  - **A2** ("pre-MLP head output depends on operands almost solely through
    `Dn+D'n`; equal-sum pairs near-identical; 3-cluster only post-MLP"):
    **untouched / weakly-challenged — NOT refuted.** The assay never reached a
    confirmed `ST` compute node, and its ratio metric cannot distinguish A2 from
    transport (the noise-free transport null reproduces the whole signature). At
    the answer-position operand-fetch heads it did reach, the output is
    indistinguishable from transport, which *weakly* disfavors reading attention
    as doing aggregation-as-extra-structure, but does not test A2's actual
    pre-MLP-at-an-ST-node prediction. No confidence change to "refuted".
  - **A3** ("ST is a well-separated categorical code, `U` off-axis"):
    **untouched.** Measured at the wrong locus (answer positions) with a
    conservative full-space silhouette; deferred to entry 2, which measures ST
    geometry at a confirmed ST node.
  - **C3 (human)** ("attention moves, MLP transforms"): **untouched.** The
    MLP-transform half was not tested at an ST node; the attention-fetch half is
    weakly consistent (heads do fetch both operands) but that is also what
    transport predicts.
  - **A5** ("attention is static positional wiring"): **untouched** (the
    near-identical answer-position heads across digits are descriptively
    consistent, no more).
  - All other predictions: **untouched**.
  - Net: this study scores **no prediction confirmed or refuted**; it is an
    `instrument failure` that produced one durable side-claim (value-path
    transport signature) and a sharpened design requirement for the real A2 test.

- **Skeptic review (post-result)**: Run 2026-07-14, separate thread, rehydrated
  only from the docs + the two result JSONs. **Verdict: BLOCK** — upheld by the
  working thread; the original interpretation was wrong and has been rewritten.

  Key findings (all verified by the working thread):
  - **F1/F2 (blocking)**: the decisive circular-transport control had been
    corrupted by a 0.05 noise term that deflated its `R²_pair` to 0.44, making
    its ratio *look* discriminating. Recomputed **noise-free from the model's
    real `W_E·W_V`**, the pure-transport null reproduces the real head output on
    *every* metric: ratio 0.185 vs real 0.195, `R²_pair` 1.0 vs 1.0, sum-arc
    monotonicity 1.0 vs 0.998, PC1 var 0.55 vs 0.52, participation ratio 2.8 vs
    3.0, tri-cluster 0.043 vs 0.036. So the real head output is **statistically
    indistinguishable from equal-weight transport of the model's weakly-circular
    digit embeddings**. The "real but minority sum arc = extra structure beyond
    transport" claim is **false** — the arc *is* the transport signature
    (`cos a + cos b = 2cos((a−b)/2)cos((a+b)/2)`). "Partial/mixed" is the one
    read the data do not support.
  - **F3 (blocking)**: the confirmed nodes are **answer-position** heads
    (primary `P14` and rep `P15–P20` are all answer tokens), plausibly doing
    `SA` base-add operand-fetch, **not** confirmed `ST` carry-compute nodes at
    question positions. The Paper-2 question-position ST candidates (P8/P9/P11)
    all failed single-node ablation confirmation. So **A2's actual prediction
    (pre-MLP sum-sufficiency at an ST node, with MLP-created tri-state) was never
    tested** — the assay mis-targeted.
  - **F4**: the no-lower-carry stimulus pins `R²_pair = 1.0`, which with the
    circular-embedding geometry over-determines ratio ≈ 0.2 independent of any
    A2-vs-transport question — so the ratio strand is `invalid`, not `negative`.
  - **F5**: full-space silhouette (170–510 D) is conservative; the weak
    tri-cluster at answer positions cannot adjudicate whether an MLP discretizes
    at a real ST node.

  Re-scoring mandated (applied below): A2 → **untouched / weakly-challenged**
  (not refuted); A3 → **untouched** (deferred to entry 2 at the right locus);
  C3 → **untouched** (MLP-transform half untested; attention-fetch half weakly
  consistent). Freeze this assay line as **`instrument failure`**.

  *Status: BLOCK RESOLVED 2026-07-14 by the working thread — control fixed
  (noise-free transport arm, `model_results`/`positive_control.json`
  regenerated), and the Executive summary, Interpretation, and Prediction
  scoring below rewritten to the honest read. Gate 2 now PASSED on the corrected
  interpretation. Router-doc updates use the corrected read.*

- **Limitations**:
  - Full-space silhouette is a conservative clustering metric in high dimension;
    weak silhouette does not prove the absence of a low-dimensional categorical
    code (entry 2 measures ST geometry directly and should be read alongside).
  - Single-node ablation under-detects redundant nodes (Paper 2 documents
    duplicate ST nodes); P8/P9 non-confirmation is likely this, not evidence
    they are not ST. A path-patching or joint-ablation confirm would be stronger.
  - The no-lower-carry stimulus isolates the local pair but removes cascade
    context; the head output studied is the "clean local" regime.
  - hook_z is pre-`W_O`; the W_O-projected check (planned) was not separately
    tabulated this pass — the resid_post tri-cluster (post-everything) is the
    downstream proxy reported.
  - Correlational (representation), not causal: does not show the head *performs*
    the addition (A-7 wording honored).

- **Doc updates** (after the corrected Gate 2 passed): ledger (new bundle);
  claim-evidence (new **CE2** — value-path transport signature, hedged; and a
  note strengthening CE1's projection motif); results-synthesis + summary (add
  the value-path finding and the `instrument failure` on the A2 line);
  conjectures (A2 annotated "first attempt was an instrument failure; untested",
  **confidence held**, not lowered; A3/C3 unchanged); agenda (freeze this assay
  line `instrument failure`; re-scope entry 2 node selection to a confirmed
  question-position ST node; promote a causal/path-patching A2 test).

- **Next read**: The real A2 test needs (a) an independently **confirmed ST node
  at a question position** (path-patching to the `SV` cascade, or post-MLP
  tri-cluster at a question position; single-node ablation is too weak given
  Paper-2 redundancy), and (b) a stimulus + metric that do not pin `R²_pair=1`
  and over-determine the ratio (e.g. include cascade-varying context; compare the
  head output against the explicit transport null as the H-transport baseline
  rather than against `R²_pair`). Entry 2 should carry the confirmed-ST-node
  requirement. The "structure is a weak low-variance projection" motif (digit
  circle → value-path transport) is now strong enough to state as an explicit
  cross-study claim.
