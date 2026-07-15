# Study: Full-Space ST Tri-State Geometry at the L1-MLP Combiner (study-st-tristate-geometry.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #6

**Verdict: no off-axis tri-state at the combiner input — the carry is a clean
binary code (A3's off-axis-third-symbol prediction refuted at this locus).** In
both models the `U`-mean is not significantly off the committed-0↔committed-1
line (off-axis 0.19–0.23, permutation p 0.85–0.92); the `U` cases are split by
their resolution (`U→0`≈committed-0, `U→1`≈committed-1), and a control shows the
split is the resolved carry, not lower-operand identity (a committed digit's
input barely moves under a lower-carry toggle: ~1.2–1.7 vs the ~28–29 U-split).

Recorded as CE6. Caveat: the raw `carry_in` ingredient is also linearly present
here, so this shows the resolved carry is *present* at the L1-MLP input (probed
post-L1-attention) but does not establish where the decision is computed. It left
open whether a genuine `{0,1,U}` tri-state exists at an *earlier* site.

## Pre-run (write before the experiment)

- **Question**: At the input to the confirmed `U`-combiner (the answer-position
  L1 MLP's residual input, `blocks.1.hook_resid_mid` / `ln2.hook_normalized`),
  what is the full-space geometry of the three `ST_n` classes `{0, 1, U}`? Which
  of the pre-registered discrete shapes ([A3](../maths-conjectures-agent.md#a3-the-st-tri-state-is-a-2d-categorical-code-with-u-off-the-01-axis)):
  - **(a) 1D ordered scalar** — `U` on the segment between the `0` and `1`
    centroids (one direction, two thresholds);
  - **(b) two near-independent binary directions** — a "definitely-carry" bit
    and a "sum-is-9" bit (three used corners of a square);
  - **(c) 2D simplex-like categorical code** — three well-separated centroids,
    `U` substantially off the 0–1 axis;
  - or an explicit fourth (e.g. effectively 1D but `U` off-axis in a small way;
    or high-dimensional / no clean centroid structure)?

- **Motivation**: This converts the thread's best-known observation (the Paper-2
  3-cluster PCA, which the digit-embedding study warned is only a *projection*)
  into the thread's first real **representation** claim, at a *confirmed*
  functional locus rather than an arbitrary node. The shape decides how a linear
  downstream consumer reads `U` (A3) and refines C1's "near-linear" wording (is
  `U` a third symbol or a scalar midpoint?).

- **Where the tri-state lives (design rationale)**: CE5 showed the combiner
  *output* is the resolved binary `carry_out` (`U` collapses to 0/1). So the
  three-way `{0,1,U}` distinction must live at the combiner's **input**: the L1
  MLP must be able to tell `U` apart from `0` and `1` to resolve it using the
  lower carry. We therefore probe the residual **feeding** the L1 MLP at the
  answer position, and (for comparison) the make-carry head's output region
  (CE3), where only the binary `{0,1}` split is expected.

- **Hypothesis / competing reads** (neutral; all live):
  - R-scalar (a): `U` centroid lies on the 0–1 segment; the three-class variance
    is dominated by **one** direction; `U`-vs-{0,1} needs two thresholds on that
    axis, not a separate direction.
  - R-square (b): `0`, `1`, `U` occupy three corners of a rectangle spanned by
    two near-orthogonal binary directions (carry-bit ⟂ sum-is-9-bit).
  - R-simplex (c): three centroids roughly equidistant / an equilateral-ish
    triangle in 2D, `U` off the 0–1 line by a large angle.
  - R-other: effectively >2 dimensions, or no clean centroid separation in the
    full space (echoing the digit-embedding "low-variance projection" caution).

- **Design**:
  - **Models**: `add_d5_l2_h3_t15K_s372001` (combiner `P14.L1.MLP`, digit 2),
    `add_d6_l2_h3_t20K_s173289` (combiner `P16.L1.MLP`, digit 3). Weights via
    `MathsConfig`+TransformerLens, CPU, accuracy-verified (invalid if < 0.99).
  - **Stimuli**: for the combiner's target digit `n`, build a large, balanced set
    of questions in each `ST_n` class — `0` (sum ≤ 8), `1` (sum ≥ 10), `U`
    (sum == 9) — with the confound controls below. Reuse the validated
    `build_U_question`/tricase constructions.
  - **Probe activation**: the residual input to the L1 MLP at the answer
    position that predicts `A_{n+1}` (the combiner's read site;
    `blocks.1.hook_resid_mid` and `ln2.hook_normalized`, reported separately),
    full `d_model`. Also probe the make-carry head output (CE3) as a
    binary-only comparison.
  - **Metrics** (full space; no 2D/3D projection shortcut):
    1. **Centroid geometry**: the three class-mean vectors; pairwise distances
       `d(0,1)`, `d(0,U)`, `d(1,U)`; the **angle** of `U` off the 0–1 line
       (project `U`−midpoint onto the `1`−`0` axis; report the perpendicular
       fraction). Scalar (a) → `U` near the 0–1 segment (small perpendicular).
       Simplex (c) → large perpendicular, near-equal pairwise distances.
    2. **Effective dimension of the class structure**: PCA of the three
       centroids (rank ≤ 2) *and* of the full per-question activations
       restricted to between-class variance; participation ratio. Scalar → 1D
       dominant; square/simplex → 2D.
    3. **Square-vs-simplex discriminator**: fit the best rectangle (two
       orthogonal binary axes) vs the best equilateral simplex to the three
       centroids; report which fits the between-class variance better, and the
       angle `∠(0→1, 0→U)` (square ≈ 90°, simplex ≈ 60°, scalar ≈ 0°/180°).
    4. **Linear separability / probe (C1)**: train linear probes (a) `0` vs `1`,
       (b) `U` vs {0,1}; report accuracy in the full space. If `U`-vs-{0,1}
       needs a direction not on the 0–1 axis, that refutes the scalar read and
       supports a genuine third symbol.
    5. **Within-class spread vs between-class**: silhouette / between-over-within
       variance, to state how *clean* the categorical structure is (the
       digit-embedding lesson: a 2D PCA can show clusters that are weak in full
       space).
  - **Controls / nulls**:
    - **Operand/base-add confound**: `ST_n` class correlates with `Dn+D'n`, which
      also sets `SA_n = (Dn+D'n) mod 10`. The combiner input may encode `SA_n`
      too. Control: within each `ST` class, **balance `SA_n`** across its
      allowed values, and report the class geometry after regressing out a
      `SA_n` one-hot (residualized class means), so the `{0,1,U}` geometry is not
      an artifact of the base-add digit.
    - **Lower-carry balance**: for the `U` class, include both lower-carry
      resolutions (U→0 and U→1) in balanced proportion, since the combiner input
      pre-resolution should still read `U` regardless; report whether the U
      cluster splits by its eventual resolution (if it does, the "input" already
      encodes the resolved value and the tri-state is really binary there — an
      informative negative).
    - **Label-permutation null**: recompute all centroid/separation statistics
      under shuffled class labels (≥ 1000 draws) → chance floor for
      separation/off-axis-angle, guarding the digit-embedding "any-few-points-
      look-structured" trap.
    - **Positive control**: the make-carry head output (CE3) is a known **binary**
      `{0,1}` code; run it through the identical pipeline — it must show a clean
      1D/2-corner binary structure and **no** distinct `U` centroid (`U` should
      fall with `0` since a make-carry head treats sum==9 as no-carry). This
      demonstrates the pipeline can (i) find a clean categorical code and
      (ii) report `U` collapsed when it is collapsed. A synthetic planted simplex
      and planted scalar are also run to confirm the square-vs-simplex
      discriminator resolves them correctly.
  - **Minimal effect / bars**: a genuine **third symbol** (`U` distinct) requires
    the `U`-vs-{0,1} off-axis perpendicular fraction ≥ 0.40 AND `U`-vs-{0,1}
    linear-probe accuracy ≥ 0.9 using a direction ≥ 30° off the 0–1 axis, above
    the permutation null, in both models. Otherwise `U` is scalar/collapsed.
    Square vs simplex decided by the centroid angle (90° ± 20° → square;
    60° ± 20° → simplex).
  - **Sample sizes**: ≥ 300 questions per class per model; probes
    cross-validated (5-fold) to avoid overfit separation.

- **Success condition** (defined now): the three-class geometry at the combiner
  input is classified into one of R-scalar / R-square / R-simplex / R-other in
  **both** models (or the models are reported as disagreeing), with full-space
  statistics above the permutation null and the `SA_n` confound removed. A3 is
  scored by which shape wins; C1 by whether `U`-vs-{0,1} is linearly separable in
  the full space.

- **Failure condition** (defined now): with the positive control passing (the
  make-carry head shows clean binary structure) — the combiner-input class
  structure is **not** separable above the permutation null in the full space
  (no clean centroids), i.e. the tri-state is a low-variance projection here too.
  A substantive negative that would extend the CE1/CE2 "structure is a weak
  projection" motif to the combiner input.

- **Ambiguous / invalid condition**:
  - Ambiguous: the off-axis perpendicular lands between the scalar and simplex
    bands (0.2–0.4); or models disagree; or the `SA_n`-residualized and raw
    geometries give different shapes.
  - Invalid: positive control fails (make-carry head shows no binary structure,
    or the planted simplex/scalar are mis-classified); or accuracy check fails.

- **Skeptic review (pre-launch)**: Run 2026-07-15 in a separate skeptic thread
  (docs + five prior study notes + Paper-2 facts only; no working-thread
  context).

  **Verdict: BLOCK** — a fatal structural confound the working thread missed:

  - **F2 (fatal)**: `U` is **perfectly confounded with `SA_n = 9`**. Since `U`
    requires `Dn+D'n = 9`, `SA_n = (Dn+D'n) mod 10 = 9` — and `SA_n = 9` occurs
    for `U` *and only* `U` (classes 0 and 1 span `SA_n ∈ {0..8}`). So "`U` is an
    off-axis third symbol" is **indistinguishable** from "the `SA_n` code places
    digit-9 off the 0–1 line". Worse, residualizing an `SA_n` one-hot is
    **collinear with the `U` indicator** — it removes the very signal. Balancing
    is impossible (`U` is `SA_n=9` with probability 1). This kills the naive
    centroid/off-axis/probe metrics for the A3 question.
  - **F1**: the L1-MLP residual input is a superposition (SA_n, operands,
    position, carry) — probe `ln2.hook_normalized` (what the MLP reads) as
    primary; enumerate all confounds, not just SA_n.
  - **F3**: positive control (make-carry head, `d_head` 170) doesn't calibrate a
    `d_model` 510 residual probe; and it should be a *discriminator* (U must
    collapse at the make-carry site *despite* being SA_n=9).
  - **F4**: 3 centroids are generically ~equidistant (simplex is the *default*,
    not evidence); null off-axis/angle against permutation + structured
    baselines with bootstrapped CIs, drop "eff dim ≤ 2" as a finding.
  - **F5**: the U-vs-{0,1} probe passes trivially via an SA_n=9 detector (inherits
    F2).

  **Rescue (skeptic)**: break the `U ≡ SA_n=9` identity using the
  **lower-carry / resolution** variable, which `SA_n=9` cannot explain (SA_n=9 is
  a single point regardless of the lower carry). Make the primary A3 test:
  does the `U` state (a) *split by eventual resolution* (U→0 vs U→1), (b) sit off
  the committed-0/committed-1 line in a way that *tracks carry_in*, and/or (c)
  align with the CE5 `carry_out` axis — none producible by an SA_n=9 code.

  *Status: RESOLVED 2026-07-15 by the working thread — redesigned around the
  resolution/lower-carry discriminator (A-1), fixed the probe site (A-2),
  positive control (A-3), nulls (A-4), and probe (A-5). Goalposts unchanged
  (still A3: is `U` a genuine third symbol vs a scalar midpoint). Gate 1 PASSED
  on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-15 — A-1 (resolves F2/F5, the core fix): the primary A3 discriminator
is the RESOLUTION variable, not the raw off-axis angle.** Because `U ≡ SA_n=9`
structurally, "U off-axis" is un-attributable. Instead, build `U` questions
*with a real lower carry* so the same `SA_n=9` digit resolves to carry_out 0 or
1 depending on carry_in. The A3 shape is then read from **four** class means at
the combiner input: committed-`0` (definite, sum≤8), committed-`1` (definite,
sum≥10), `U→0` (sum=9, no lower carry), `U→1` (sum=9, lower carry). Since all `U`
cases share `SA_n=9`, any structure that *distinguishes U→0 from U→1* or places
the `U` means off the committed-0↔committed-1 line **cannot** come from the
`SA_n=9` code (which is a single point) — it is genuine ST/carry structure. Verdicts:
  - **R-scalar (a)**: `U→0` sits with committed-`0`, `U→1` with committed-`1`,
    all on one axis — `U` is not a third symbol, just the resolved value on the
    0–1 line (A3 falsified).
  - **R-third-symbol / simplex (c)**: the `U` means (pre-resolution direction)
    sit **off** the committed 0–1 line by a large, permutation-significant angle
    and are separable from both committed states — `U` is a genuine deferred
    symbol (A3 supported).
  - **R-square (b)**: two orthogonal binary axes (carry-committed ⟂ is-U);
    `U→0`,`U→1` differ from committed-0,committed-1 along the is-U axis.
  - Alignment check: project the committed-0→committed-1 direction and test
    whether it aligns with the **CE5 `carry_out` axis** (external anchor); the
    `U` off-axis component must be *orthogonal* to `SA_n=9`'s local slot — tested
    by whether it is shared across digits (cascade fact) vs position-local
    (SA_n fact).

**2026-07-15 — A-2 (resolves F1): probe site + confound list.** Primary site =
`blocks.1.ln2.hook_normalized` at the `A_{n+1}`-predicting position (what the L1
MLP reads); `hook_resid_mid` secondary. Enumerated confounds each get a control:
`SA_n` (broken by A-1's resolution contrast, since U→0/U→1 share SA_n=9),
operand distribution (committed classes sampled to match U's sum-9-like operand
spread where feasible; report residual), lower-carry state (the *variable of
interest* in A-1, not a nuisance). Verdict framed as "class-discriminative
geometry at the combiner input, confounds addressed", not "the ST feature's
intrinsic shape".

**2026-07-15 — A-3 (resolves F3): positive controls.** (1) Make-carry site run
through the **same `d_model` residual** (make-carry head's residual contribution
/ residual at its position), and used as a **discriminator**: the `U` cases must
**collapse onto committed-0** there (make-carry is carry-in-inert, treats sum=9
as no-carry) *despite* being `SA_n=9` — if `U` is off-axis at the make-carry
site, the combiner-input off-axis signal is attributed to `SA_n=9`, not ST.
(2) Planted-shape controls (simplex / scalar / square) with pre-registered
variance share (grade 0.2–0.9), isotropic + spectrum-matched noise, and the
90°/60°/0° angle recovered within a stated tolerance — per the digit-embedding
A-3/A-8 precedent.

**2026-07-15 — A-4 (resolves F4): nulls.** Report the off-axis fraction and the
committed-vs-U separation as **p-values against the label-permutation null**
(≥ 1000 draws) AND against the structured make-carry baseline — not absolute
bars. Bootstrap all centroid angles over questions; a CI spanning two shape
bands → ambiguous. Drop "effective dim ≤ 2" as a reported finding (trivial for
3–4 centroids); report instead the between-class variance share vs the
permutation null.

**2026-07-15 — A-5 (resolves F5): probe.** The C1 linear-separability test is
**U→0 vs U→1** (resolution) and **U(pre-resolution) vs committed states**, NOT
U-vs-{0,1} (which an SA_n=9 detector passes trivially). Report whether the U→0/1
separating direction aligns with the committed 0→1 (carry_out) axis; CV 5-fold.

**2026-07-15 — A-6 (F6): power + scope.** Minimal effect stated vs the
permutation+structured null (not absolute). 6-digit `n_ctx=22`, combiner
`P16.L1.MLP` digit 3; 5-digit `n_ctx=19`, combiner `P14.L1.MLP` digit 2 (both
have a valid lower digit for the carry). ≥ 300 questions per class; the
residualized-SA geometry is expected to be *degenerate* (collinear with U), not
an independent ambiguity signal.

- **Decision impact**:
  - **R-simplex (c)**: A3 confirmed (`U` a genuine off-axis third symbol);
    strengthens the "categorical, not scalar" reading; C1 refined (simple but
    the natural unit is a 2D categorical code, not one direction).
  - **R-scalar (a)**: A3 refuted (its falsifier — `U` between 0 and 1); C1's
    near-linear one-direction reading supported.
  - **R-square (b)**: A3's alternative (b) confirmed; suggests the code inherits
    the legacy `SC`/`SS` two-bit structure.
  - **R-other / not-separable**: extends the low-variance-projection motif;
    weakens A3's clean-centroid premise and C1's clean-separability premise.

- **Risks / confounds**:
  - **`SA_n` correlation with `ST` class** (main confound) — handled by
    balancing and residualizing `SA_n`.
  - **Which activation is "the tri-state"**: the combiner *output* is binary
    (CE5); probing the wrong site (output) would trivially show binary. Probe the
    *input* residual; report both input and output so the read is unambiguous.
  - **Silhouette conservatism in high-D** (digit-embedding lesson): report
    against the permutation null, not absolute silhouette.
  - **Small number of classes (3)**: centroid geometry from 3 points is
    inherently ≤ 2D; the permutation null and the probe accuracy guard against
    over-reading 3-point structure.
  - 2-layer, addition-only; single-digit `U`.

- **Expected artifacts**:
  - Standalone CPU script `scripts/st_tristate_geometry.py`.
  - Per (model, site): centroid distances/angles, off-axis fraction, effective
    dim, square-vs-simplex fit, probe accuracies (CV), silhouette + permutation
    null; make-carry binary comparison; planted-shape control results.
  - Plots: centroid triangle + off-axis illustration; probe-accuracy bars.
  - A shape-verdict registry JSON.
  - Local results folder `results/study-st-tristate-geometry/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (corrected post-Gate-2): **At the L1-MLP combiner
  input there is no distinct off-axis tri-state `U` — the carry is a clean
  *binary* code on the committed-0↔committed-1 axis, and the `U` cases are split
  by their resolution (`U→0`≈committed-0, `U→1`≈committed-1).** In both models
  the `U`-mean is not significantly off the 0–1 line (off-axis 0.19–0.23,
  permutation p 0.85–0.92), and the per-class distances confirm the split:
  d(c0,U→0)=10.1 vs d(c1,U→0)=29.5; d(c1,U→1)=13.3 vs d(c0,U→1)=30.5 (5-digit;
  6-digit similar). A control shows the split is the *resolved carry*, not
  lower-operand identity (a committed-0 digit's input moves only 1.21/1.65 under
  a lower-carry toggle, vs the ~28–29 `U`-split). **A3's "off-axis third symbol"
  is refuted at this locus.** Caveat (Gate-2 F3): the raw `carry_in` bit is also
  ~98%/94% linearly decodable here (on committed digits), so the site holds the
  decision *ingredients* — this does **not** establish that the U→{0,1}
  *decision* is completed upstream of the MLP; it only shows the resolved carry
  is *linearly present at the L1-MLP input* (post-L1-attention), narrowing CE5's
  output-level finding. No attribution to L0 is made (the probe is downstream of
  L1-attention).

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/st_tristate_geometry.py control`
    then `... models`.
  - Script:
    [`scripts/st_tristate_geometry.py`](../../scripts/st_tristate_geometry.py)
    (standalone CPU; reuses confirm-ST-node helpers). Env: python 3.13.7,
    macOS-26.5.2-arm64, torch 2.8.0, scikit-learn 1.7.1. Repo commit `368f3a9`
    (working tree). Date 2026-07-15.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000, combiner `P14.L1.MLP`,
    digit 2), `add_d6_l2_h3_t20K_s173289` (acc 1.000, combiner `P16.L1.MLP`,
    digit 3). Probe site: `blocks.1.ln2.hook_normalized` (what the MLP reads).
  - Artifacts (local; no HF): `results/study-st-tristate-geometry/`:
    `results.json`, `shape_registry.json`, `combiner_input_pca.png`.

- **Results**:
  - **Planted-shape controls PASS**: the discriminator recovers scalar
    (off-axis 0.11, perm p 1.0), simplex (0.85, d(u0,u1)-norm 0.2, angle 58°),
    square (0.98, d-norm 0.95, perm p 0.004). Metric verified on signal-only
    data (scalar off-axis 0.00 / simplex 0.87 / square 1.00), so the metric is
    sound and the permutation null (not absolute off-axis) is the significance
    gate — exactly the digit-embedding lesson.
  - **Combiner input (both models)**: off-axis fraction 0.20 (6-digit) / 0.23
    (5-digit), permutation p 0.93 / 0.85 → `U`-mean is **not** significantly off
    the 0–1 line. Between-class variance share is high (0.85–0.91), i.e. the
    classes *are* separated — but along the committed 0–1 axis, not a third
    dimension.
  - **`U` is pre-resolved (decisive)**: `U→0`-vs-`U→1` linear probe accuracy
    **1.00** and `d(U→0, U→1) = 28.8` (larger than any other pair). Class-mean
    distances (5-digit): d(c0,c1)=32.3; d(c0,U→0)=10.0, d(c1,U→0)=29.4 (U→0≈c0);
    d(c1,U→1)=13.3, d(c0,U→1)=30.7 (U→1≈c1).
  - **Confound control (decisive)**: for a *committed-0* digit, toggling the
    lower carry moves the combiner input by only **1.3** — negligible vs the 28.8
    `U`-split — so the split is the **resolved carry_out**, not lower-digit
    operand identity.
  - **Make-carry site**: off-axis 0.09–0.12 (U collapsed toward committed-0), as
    the positive-control discriminator predicts for a carry-in-inert binary node.

- **Interpretation** (corrected post-Gate-2; goalposts unchanged):
  - **A3 falsifier met at this locus → A3 refuted *at the combiner input***
    (locus-scoped): `U` is not a distinct off-axis third symbol here; off-axis is
    n.s. vs the permutation null and the `U` cases sit on the committed 0–1 axis
    routed by resolution. This does **not** refute a tri-state existing at some
    *earlier* site (out of scope; see limitations). Shape: 6-digit R-scalar;
    5-digit lands in the pre-registered **ambiguous off-axis band** (0.23), same
    direction.
  - **Narrows CE5 (not "upstream decision")**: the resolved carry_out is
    linearly present at the L1-MLP *input*, not only its *output* (CE5). But the
    raw `carry_in` ingredient is *also* ~98% decodable here on committed digits,
    so the assay **cannot** show the U→{0,1} *decision* is completed upstream vs
    within the MLP — both "resolved-upstream" and "ingredients-present,
    MLP-decides" produce a linearly-separable input. CE5's causal "L1 MLP is the
    combiner" is untouched; this study only adds that the input already linearly
    carries the resolved carry (and its ingredients). No L0 attribution (probe is
    post-L1-attention).
  - **C1**: at this locus the carry is a clean **binary**, linearly-separable
    code on one axis — a narrow datapoint consistent with C1's near-linear
    reading; no curved/off-axis third state here (but this is one locus, not
    broad support).

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A3** ("`ST` tri-state is a 2D categorical code with `U` off the 0–1 axis"):
    **refuted at the combiner input (locus-scoped)** — `U` is on the axis
    (off-axis n.s.) and split by resolution, not a distinct off-axis symbol.
    **Untouched** on whether a tri-state exists at any *earlier* site (this study
    probes one locus only).
  - **C1 (human)** ("simple, near-linear; `U`-vs-{0,1} linearly separable"):
    **narrowly supported at this locus** — clean binary linear carry code here;
    not broad support (one site; the digit-embedding study refuted C1's
    low-dimensionality clause elsewhere).
  - **A2** ("MLP does the nonlinear step"): **untouched** — the assay cannot show
    the U→{0,1} decision is upstream of vs within the MLP (both `carry_in` and
    the class ingredients are linearly present at the input). Consistent with
    CE5's "A2 supported at the U-combine step"; no new A2 evidence.
  - **A6**: **untouched** (static geometry read, not a cascade/tie-break test).
  - Others untouched.

- **Skeptic review (post-result)**: Run 2026-07-15, separate thread (docs + JSON
  + script). **Verdict: BLOCK** — upheld; two serious issues, both fixed below.

  Findings (all verified/fixed by the working thread):
  - **F1 (fatal, fixed)**: the headline per-class distances (d(c0,U→0) etc.) and
    the committed-lower-carry control (d≈1.3) were computed in ad-hoc snippets,
    **not by the committed script** — so they were absent from `results.json`
    (evidence-integrity failure). **Fixed**: the script now computes and stores
    `d_c0_u0/d_c1_u0/d_c0_u1/d_c1_u1` and a `committed_lowercarry_move`; re-run
    values: d(c0,u0)=10.1, d(c1,u0)=29.5, d(c1,u1)=13.3, d(c0,u1)=30.5 (5-d);
    committed-0 lower-carry move 1.21 (5-d) / 1.65 (6-d). Numbers hold.
  - **F3 (major, corrected)**: the "resolution is complete upstream / the MLP
    merely applies" claim was **asserted, not measured**. The skeptic-prescribed
    discriminator (is the raw `carry_in` bit linearly present at this site?) was
    run: **`carry_in` is ~98%/94% decodable on *committed* digits** — so the
    site holds the *ingredients* (both `carry_in` and the digit's class are
    linearly present), NOT a proven completed-upstream decision. Claim dropped.
  - **F4 (major, corrected)**: the probe is `blocks.1.ln2.hook_normalized` =
    resid *post-L1-attention*, so "resolution complete before the L1-MLP input"
    is supportable but attributing it to **L0** is not (L1-attn is included). All
    "L0 conduit" attributions removed; scoped to "by the L1-MLP input".
  - **F2 (corrected)**: "U pre-resolved" is largely a restatement of "this site
    encodes carry_out" (u0≈c0 / u1≈c1 because both have carry_out 0/1) — the
    genuinely new content is narrow (resolved carry linearly present at the MLP
    *input*, refining CE5's *output* finding). Reframed.
  - **F6/F7 (corrected)**: "A3 refuted" → "refuted **at the combiner input**
    (locus-scoped)"; the 5-digit model is in the pre-registered **ambiguous
    off-axis band** (0.23 ∈ [0.2,0.4]) → reported as ambiguous-band, not clean
    R-scalar; verdict = "6-digit R-scalar; 5-digit ambiguous-band, same
    direction".
  - **F8/F9 (hedged)**: make-carry resolution probe is 0.79/0.80 (NOT chance) —
    reported honestly; planted-simplex control reached only perm p 0.076 (not
    α=0.01) — a weak-simplex might be missed, noted.
  - Scoring corrections: A2 → **untouched** (not "weakly cautioned" — the assay
    can't show the decision is upstream; consistent with CE5); C1 → **narrowly
    supported at this locus**; A6 untouched.

  *Status: BLOCK RESOLVED 2026-07-15 — script fixed so all headline numbers are
  reproducible in `results.json`; interpretation re-scoped (locus-only; no L0
  attribution; ingredients-not-upstream-decision); A3 locus-scoped; 5-digit
  reported ambiguous-band. Gate 2 PASSED on the corrected read.*

- **Limitations**:
  - This locates where `U` is *already resolved* (combiner input); it does not
    itself find the earliest site where a genuine tri-state `{0,1,U}` might exist
    (if any) — that would need probing progressively earlier residual sites /
    the L0 conduit. The A3 "third symbol" question is answered *negatively at the
    combiner input*, not everywhere.
  - Single-digit `U`; 2-layer; addition only. The combiner-input probe is the
    LN'd residual (a superposition); the U-split control (committed-digit
    lower-carry insensitivity, d=1.3) addresses the main operand confound but not
    every possible one.
  - Verdict labels differed cosmetically across models (5-digit "R-other" at
    off-axis 0.23 vs 6-digit "R-scalar" at 0.20) but agree substantively (n.s.
    off-axis; U pre-resolved).

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (**CE6** —
  at the L1-MLP combiner input the carry is a clean binary code, no off-axis
  tri-state; hedged, narrows CE5); synthesis + summary; conjectures (A3
  refuted-at-locus / untouched-elsewhere; C1 narrowly-supported; A2/A6
  untouched); agenda (complete entry 1 as a locus-scoped negative; promote the
  earliest-tri-state / upstream-site sweep).

- **Next read**: the open question is *where, if anywhere, a genuine `{0,1,U}`
  tri-state exists* before it collapses to binary — probe **earlier** residual
  sites (pre-L1-attention, the L0 conduit output) and test whether `carry_in`
  and the sum==9 flag are separately present there (ingredients) vs a resolved
  bit. Also test whether the make-carry axis and this resolved-carry axis are the
  same direction (alignment). B11 (L0→L1 path-patch) is the causal complement.
