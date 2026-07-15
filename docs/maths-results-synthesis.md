# Maths Results Synthesis (maths-results-synthesis.md)

Read role and rules: [Results Synthesis](thor-document-rules.md#results-synthesis).

Mid-level empirical synthesis for the `maths` thread's primary results themes.
Organize by empirical question, not by experiment chronology. Keep only figures
that change interpretation or priority. This is the normal home for detailed
mechanism interpretation that would overload
[maths-results-summary.md](maths-results-summary.md).

## Q: How are digit values represented at the token level (embedding / unembedding)?

Status after the 2026-07-14 digit-embedding geometry audit (weights-only; see
[claim CE1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix)
and the [study note](study-maths/study-digit-embedding-geometry.md)).

- The digit-token embedding is **near-isotropic in 9 dimensions** (variance
  spread ≈ 0.15→0.08 across components, participation ratio ≈ 8.7/9) for both
  trained and untrained models. There is **no dominant low-rank circle or
  helix**: the frequency-1 variance share (~0.27) barely exceeds the untrained
  baseline (0.23) and, by the positive control's own calibration (planted 0.40 →
  recovered 0.52), sits at the noise floor.
- The one **training-induced** effect is a weak **circular ordering** of digit
  values in the top-2 PC plane: permutation p = 0.0001, absent in the untrained
  control (p = 0.34). This is an ordering tendency, not variance concentration.
  The LN-aware close-out (A-9, 2026-07-16) settled it: the ordering **survives
  the LN-effective geometry** the computation reads — but LN is near-isometric
  here (γ std ~0.005), so this is *weak* robustness, and the signal is
  **seed-fragile** (holds in 2 of 3 independent seeds; fails s173289). No longer
  provisional; now Low / LN-robust-weakly / seed-fragile.
- The **unembedding** carries no comparable structure (R4 everywhere) and does
  not align with the embedding geometry (principal angles 36–88°).
- **Decision-relevant read**: a clean geometric digit code was only ever one
  route to numeric computation, and it is largely absent here at the token level.
  This makes the **activation- and MLP-level** story (aggregate-then-discretize;
  tri-state `ST` geometry) and a **causal** test of whether the model uses digit
  magnitude/circularity the more central questions — not less. See
  [maths-next-steps.md](maths-next-steps.md).

Figure: `results/study-digit-embedding-geometry/embed_pc_planes.png`
(top-2 PC plane per model, digit-labeled) is the decision-relevant plot; the
near-flat `embed_variance_spectra.png` is the evidence for isotropy.

## Q: Where is the ST carry computed, and does attention aggregate or transport?

Status after the 2026-07-14 pair-sum study (an `instrument failure` for its
primary question; see [CE2](maths-claim-evidence.md#ce2-at-layer-0-operand-fetch-heads-the-value-path-output-is-indistinguishable-from-linear-transport-of-the-weakly-circular-digit-embeddings)
and the [study note](study-maths/study-pair-sum-sufficiency.md)).

- The A2 "aggregate-then-discretize" question is **still open**. The first assay
  mis-targeted (it confirmed answer-position operand-fetch heads, not
  question-position `ST` compute nodes) and used a metric that the no-lower-carry
  stimulus rendered non-discriminating.
- Durable by-product: at those operand-fetch heads, the value-path output is
  **indistinguishable from linear transport** of the weakly-circular digit
  embeddings — the same "weak low-variance projection" pattern seen for the
  embedding itself (CE1). The monotone sum arc there is the arithmetic identity
  `cos a + cos b = 2cos((a−b)/2)cos((a+b)/2)`, not evidence of an aggregation
  computation.
- **Method lesson (decision-relevant)**: full-space geometry keeps showing that
  the tidy low-D structures (digit circle, ST tri-cluster, sum arc) are
  *low-variance projections*; verdicts must compare against explicit nulls
  (permutation, transport) and confirm the node's role independently, not read
  a projection at face value.

## Q: Where is the carry computed, and is it the tri-state ST or a binary carry?

Status after the 2026-07-14 confirm-ST-node study (causal path-patching; see
[CE3](maths-claim-evidence.md#ce3-the-carry-is-computed-by-binary-make-carry-heads-at-answer-positions-dissociated-from-base-add-heads-the-tri-state-u-resolution-is-a-separate-unlocated-path)).

- The per-digit carry is computed by **specific attention heads at the answer
  token positions** (5-digit: `P13/P14/P15.L0.H0`; 6-digit: `P14.L0.H1`,
  `P15–P19.L0.H2`), each causal for the *next-higher* answer digit `A_{n+1}`
  (interchange flip 1.00, `A_n` untouched, null 0.00). A **clean head-role
  dissociation** holds: at each answer position one head does the carry and a
  different head does base-add (`SA`, flips `A_n`).
- But these are **binary make-carry (`SC`) heads, not tri-state `ST`**: the
  genuine `U` (sum=9) test flips nothing through them. The model *does* resolve
  `U` correctly, so the **tri-state resolution runs on a separate path that is
  not yet located** — the most important open thread now.
- **Method wins**: the flip-signature (does patching move `A_n` vs `A_{n+1}`?)
  cleanly dissociates SA from carry where the earlier variance-ratio could not;
  and a node-level positive control + genuine-`U` battery were essential
  (without the latter a binary carry node reads as tri-state).

## Q: Where is the tri-state U-resolution computed? (partial answer)

Status after the 2026-07-15 U-resolution study (positive-but-ambiguous; see
[CE4](maths-claim-evidence.md#ce4-the-tri-state-u-resolution-flip-is-transmitted-by-an-mlp-heavy-l0l1-path-distinct-from-the-make-carry-heads-whether-it-is-combined-or-merely-relayed-is-unresolved)).

- The U-resolution flip is causally **transmitted by an MLP-heavy L0/L1 path**
  (5-digit `P10.L0.MLP`+`P14.L1.MLP`; 6-digit adds head `P11.L0.H2`), **distinct
  from the CE3 make-carry heads** (U-flip 0.00) — so the "binary carry and
  tri-state U run on separate components" picture is reconfirmed with named
  nodes.
- **But** whether any of these *combines* digit-`n`'s sum==9 flag with the lower
  carry, or merely *relays* the resolved carry, is **unresolved**: the intended
  U-regime-specificity discriminator was vacuous (a definite digit's `A_{n+1}`
  has no lower-carry dependence, so the trivial readout also "passes").
- **Method lesson (decision-relevant)**: an interaction gate only works if the
  control arm has a signal for a *non*-target node to transmit. A real
  combiner-vs-conduit test needs a *corruption* control (inject a lower-carry
  inconsistent with a definite digit and check whether the answer is corrupted).

## Q: Combiner vs conduit — answered (the L1 MLP combines)

Status after the 2026-07-15 combiner-vs-conduit study (see
[CE5](maths-claim-evidence.md#ce5-the-tri-state-u-combiner-is-the-answer-position-layer-1-mlp-layer-0-nodes-relay-the-running-carry-conduit)).

- The **answer-position layer-1 MLP** (`P14.L1.MLP` 5-digit, `P16.L1.MLP`
  6-digit) is the **tri-state `U`-combiner**: its output is invariant to
  `carry_in` for a definite digit but varies in the `U` regime, and it routes to
  the class-determined `carry_out` centroids — i.e. its output *is* the resolved
  carry. Replicated in both models, and it survives a carry_out-centroid test
  (rules out a "U-detector" artifact).
- **Layer-0 nodes are conduits** (relay the running carry regardless of
  digit-`n`'s class) — cleanly in the 6-digit model; the 5-digit L0 candidate is
  borderline. So the emerging picture: **L0 conducts/computes the running carry,
  the answer-position L1 MLP combines it with digit-`n`'s class to resolve `U`.**
- **Full mechanistic chain so far**: digit embeddings (weak geometry, CE1) →
  layer-0 attention heads compute base-add (`SA`) and binary make-carry
  (CE3) → layer-0 nodes conduct the running carry (CE5) → **answer-position L1
  MLP combines to resolve the tri-state `U`** (CE5) → answer readout. Distinct
  binary-carry and tri-state paths (CE3/CE4/CE5).
- **Method arc**: it took three iterations (instrument failure → vacuous cascade
  battery → vacuous interaction gate) to find a *node-level, endpoint-independent*
  discriminator that works — the lesson being that definite-digit
  carry-independence defeats any endpoint-based test.

## Q: ST tri-state geometry at the combiner input — answered (binary, no third symbol)

Status after the 2026-07-15 tri-state-geometry study (see
[CE6](maths-claim-evidence.md#ce6-at-the-u-combiners-input-the-carry-is-a-clean-binary-code--no-distinct-off-axis-tri-state-the-resolved-carry-is-already-linearly-present-there)).

- At the L1-MLP combiner **input**, the carry is a clean **binary** linear code
  on the committed-0↔committed-1 axis; the `U` cases are split by their
  resolution (U→0≈committed-0, U→1≈committed-1) and the `U`-mean is **not**
  off-axis (n.s. vs permutation null). **A3's "off-axis third symbol" is refuted
  at this locus** — but only at this locus (a tri-state could exist upstream).
- The resolved carry is already linearly present at the MLP *input* (narrowing
  CE5's output finding), and the raw `carry_in` ingredient is ~98% decodable
  here too — so the site holds ingredients; the assay cannot say whether the
  U→{0,1} *decision* is completed upstream or in the MLP.
- **Method lesson**: `U ≡ SA_n=9` is a fatal structural confound for the naive
  geometry read; the *resolution variable* (U→0 vs U→1 via the lower carry) is
  what makes the question answerable. And a site linearly separating an outcome
  ≠ that site *computing* it (ingredients vs decision).
- **Chain so far** (updated): embeddings (weak geometry, CE1) → L0 heads
  (base-add + binary make-carry, CE3) → L0/L1 conduit + combiner path resolves
  the carry (CE5) → by the L1-MLP combiner input the carry is a resolved binary
  code (CE6) → answer readout.

## Q: Does a dedicated `{0,1,U}` tri-state symbol exist? — answered (no, at answer positions)

Status after the 2026-07-16 earliest-tri-state-site study (see
[CE7](maths-claim-evidence.md#ce7-no-dedicated-01u-tri-state-symbol-at-any-answer-position-residual-site-u-is-resolved-to-binary-around-l1-attention)).

- Sweeping 7 residual sites (embedding → combiner) at the answer position, there
  is **no site where `U` is a dedicated, resolution-independent off-axis third
  symbol** (is-U axis never significant vs the null). The carry is **binary
  throughout**; the `U` cases are resolved toward the committed 0/1 axis, with
  the binary resolution applied **around L1-attention** (U→1 flips from
  committed-0-leaning to committed-1 at `L1.resid_mid`).
- So the tri-state `U` of the Paper-2 *algorithm* is a **functional** description;
  the model's **representation** is binary carry, resolving `U` implicitly rather
  than storing a third symbol. This is *simpler* than A3 predicted.
- **Method arc (7 studies)**: this is the fourth consecutive A3-family
  refuted/untouched result (CE6 + CE7). Per the Evidence Rules streak signal, the
  "U as a distinct symbol" frame is stale at answer positions — the productive
  next move is question-position / multi-digit-cascade, not more answer-position
  geometry.
- **Full chain (updated)**: embeddings (weak geometry, CE1) → L0 heads
  (base-add + binary make-carry, CE3) → L0/L1 conduit + L1-MLP combiner resolve
  the carry (CE5) → carry is a **binary** code, `U` resolved around L1-attention,
  no tri-state symbol (CE6, CE7) → readout.

## Q: Is attention static positional wiring (A5)? — no, hybrid

Status after the 2026-07-16 attention-invariance census (see
[CE8](maths-claim-evidence.md#ce8-attention-routing-is-hybrid--a-few-heads-relocate-their-target-with-carry-state-6-digit-only-most-cells-are-target-static)).

- **A5's strong "static positional wiring" form is falsified** (6-digit model):
  a few heads — `L1.H1` (operand-read Q11 and answer Q14) and `L0.H0` (answer
  Q17) — **relocate their attention target with the carry state** under a
  value-matched contrast (strict null ≤0.08, Bonferroni-safe). This is A5's own
  pre-registered falsifier. Routing is therefore **hybrid**: most (head,position)
  cells target-static, a few content-routed by carry.
- The cleanest cell is an **operand-read** position (Q11), so it is *not* the
  trivial "answer digit depends on carry" — carry-state routing genuinely occurs
  where the carry is being read, not only where it is emitted.
- Scoped: clean in **6-digit only** (the 5-digit top-1-argmax is too tie-noisy —
  null 0.40–0.53 — to adjudicate); representational, not causal. The routing
  cells are candidate content-routed nodes for the cascade-tracing entry, to be
  **confirmed causally** there.
- **Method note**: value-matched contrast + strict null was essential — raw
  pattern-variance and 3-way stratum TV are A5-consistent (a static head still
  has value-dependent softmax weights), so only a *target-move under a
  value-matched contrast* falsifies A5.

## Q: How is the deep `...999` cascade resolved (A6 sequential vs A9 selection)? — not localizable at this granularity

Status after the 2026-07-16 deep-cascade mechanism study (see
[CE9](maths-claim-evidence.md#ce9-at-nodeattention-pattern-granularity-the-deep-999-cascade-mechanism-is-not-localizable--no-single-cell-selection-no-sequential-per-digit-state-real-graded-tail-state)).

- The deciding-digit patch across graded chain depths **ran**, but the
  node/attention-pattern instrument is **too blunt** to resolve the fork.
- **A9's predicted convergence is absent**: a causally deciding-selective consumer
  head (6-digit `L1.H0` Q14 — redirect-to-deciding 0.93/0.90 beats both wrong and
  irrelevant baselines) and a deciding-digit-*tracking* head (`L1.H2` Q15, top-2
  key set follows the deciding position at ≥2 non-degenerate depths) are
  **different cells**; the CE8 routing cell `L1.H1` is causally **inert** (0.00).
  A9 needs one head that both tracks and delivers — not found here.
- **Sequential per-digit state** is disfavored where testable (5-digit `=` k=3:
  no intermediate-digit identity carried) but the test is untestable in 6-digit,
  so this is not a refutation.
- Real **graded** computed state sits near the question tail (pure-state cells,
  null 0.00, joint-flip 0.00) — operand-adjacent, not a single stored resolved bit.
- **Method note (two-directional over-reach — a strong instance of the skeptic
  gate working)**: the first write-up over-claimed a *negative* (sequential
  refuted, selection not found) on uncontrolled/unimplemented assays; the corrected
  second write-up over-claimed a *positive* (A9 confirmed via `L1.H1`) on a
  value-matched metric that was asserted in a docstring but never coded, a
  boundary-artifact tracking gap, and three signatures pulled from three different
  cells. Both were caught by post-result skeptic rounds. Durable lessons: a null
  needs a positive control *on the same locus/unit*; the metric named must be the
  metric coded; require a mechanism's signatures to converge on the *same* cell
  before scoring it; single-head uniform pattern-redirect confounds selectivity
  with generic disruption — use an **edge path-patch** for a hand-off question.

### Q: Does an L1 head's output causally drive the combiner (A9 hand-off)? — one-depth crumb (CE10)

Status after the 2026-07-16 edge-patch study
([CE10](maths-claim-evidence.md#ce10-at-the-combiner-edge-a-single-l1-head-carries-the-top-cascade-digits-computed-carry-at-one-depth--a-causal-crumb-but-the-single-position-edge-instrument-is-underpowered)).

- The edge path-patch (`L1.head → combiner MLP` edge, isolated via the exact
  `resid_mid = resid_post(L0) + Σ_h z@W_O + b_O` decomposition) resolves CE9's
  null: at **one depth** (6-digit k=3) **`L1.H1`** — the CE8 routing cell CE9
  found inert — **causally drives the combiner**, carrying the top cascade digit's
  **computed, deciding-selective** carry through the MLP input (dec 1.00 /
  same-class-operand null 0.00 / wrong-digit 0.00).
- But the **single-position edge instrument is underpowered** at most cells (6/9
  6-digit, 3/9 5-digit — a single-head-magnitude direct edge cannot flip them, LN
  renormalizes a single head's edge against the whole residual), **no head clears
  the ≥2-depth bar**, and the **5-digit direct residual path is causally live**.
  So A9's attention-delivery is **circumstantial, not confirmed**, and A6's
  residual-carry is **not refuted**.
- **Method note (third skeptic catch in this line, twice positive-direction)**: the
  first edge-patch draft over-claimed "attention-delivered, not residual (0/9); A9
  confirmed" — the power control was **inverted** (scaled the direct path UP ~20×,
  disabling the underpower gate). Corrected (scale DOWN to per-cell head-edge norm),
  6/9 cells are underpowered. Durable lesson: a power/underpower control must be
  constructed at the *magnitude of the effect being tested* and its direction
  verified; and LN renormalization makes single-head edge patches systematically
  underpowered — use a multi-position or less-damped instrument.

## Q: Do sub-tasks share a template across positions and stay orthogonal across sub-tasks (C2/A4/A8)? — no, for ST (CE11)

Status after the 2026-07-16 probe-transfer study
([CE11](maths-claim-evidence.md#ce11-the-tri-state-carry-st-is-position-specific-at-question-positions-no-cross-position-probe-transfer-and-geometrically-entangled-with-sv--c2a4-template-sharing-and-a8-interference-challenged-for-st)).

- **Template sharing (C2/A4): falsified for `ST` at question positions.** An `ST`
  probe trained at digit `i` does not transfer to digit `j` (retention 0.10/0.12 ≪
  0.6; off-diagonal ≈ chance), and mean-centering does not restore it — so not even
  A4's per-position-offset form. The tri-state carry representation is
  position-specific at the question site, despite architectural weight sharing.
- **Orthogonality (C2) / interference (A8): challenged.** `ST` and `SV` are
  geometrically **entangled** (principal angle 21° ≪ 64°/50° label-correlation
  null, both models) despite label-independence at the same digit — a genuine shared
  direction, not a label near-identity.
- **Scope**: `ST`-only (SA/SV are diag-weak at the question site; SA decodes 1.00 at
  the *answer* position, confirming CE2/CE3). Both models agree. Linear probes.
- **Method note (Gate-2 discipline held)**: the first draft over-claimed "all
  sub-tasks position-specific + entangled" and leaned on "position decodes at 1.00";
  Gate 2 narrowed it to `ST` (the only validly-assessable sub-task at the site) and
  demoted the position-1.00 gloss (a trivial positional-embedding fact) in favor of
  the centering-fails discriminator. The falsification is earned for `ST`.

## Q: Answer-phase binding — tape or register (A4)? — split (CE12)

Status after the 2026-07-16 answer-binding study
([CE12](maths-claim-evidence.md#ce12-answer-phase-layout-is-split--sa-is-a-just-in-time-register-absent-at---sv-is-resolvedpresent-at--but-its-per-digit-slots-are-not-orthogonal-tape-refuted-both-share-an-answer-side-template)).

- **`SA` (sum digit) = just-in-time register**: absent at `=` (raw acc ≈ chance),
  present only at its own answer position — A4's just-in-time fetch supported.
- **`SV` (resolved carry) = present at `=` but not an orthogonal tape**: `SV_n`
  decodes at `=` for all digits (CE7-consistent resolution locus), but the per-digit
  slots are **not orthogonal** (6-digit 12–27°; 5-digit 2/3 pairs < 60°) → the
  orthogonal-tape alternative is refuted. Whether SV is present *beyond a full
  operand recompute* is untested (weak isolated-operand baseline) — no storage/bus
  claim; A6/C3 untouched.
- **Answer-side template**: both `SA`/`SV` transfer across answer positions (SV
  beats the operand floor), unlike the question-side `ST` (CE11) — so **template
  sharing is position-of-computation dependent** (question side position-specific,
  answer side shared).
- **Method note (Gate-2 discipline)**: the first draft over-claimed a "coexistent
  carry bus at `=` beyond re-derivation"; Gate 2 showed the SV isolated-operand
  baseline is too weak (SV depends on all lower digits) so the claim restates CE7's
  locus, not new binding — reframed CE7-consistent; entanglement scoped 6-digit.

## Open empirical questions

- **Is the `=` resolved carry actually USED downstream** (causal), or recomputed at
  each answer position? And is `SV` present at `=` *beyond a full operand recompute*
  (a fair digits-`0..n` baseline)? — the causal/fair-baseline follow-ups to CE12.
- **Raise power on the hand-off**: a less LN-damped / multi-position edge
  instrument + a second depth (6-digit k=4) to test whether `L1.H1`→combiner
  delivery holds at ≥2 depths (CE10 got one depth). Neuron-level (B2) on how
  `L1.H1` + the combiner MLP compute `carry_out`.
- Why do 5-digit and 6-digit **differ** in delivery (5-digit direct path live,
  6-digit not)? Cross-seed/size (B5).
- Does a *transient* `{0,1,U}` tri-state exist at **question positions** (D'n)
  before the answer position (the only regime A3 can still live)? Needs a
  purpose-built question-position construction.
- Does the tie-break economy (A6) hold for **multi-digit `...999` cascades**?
  CE9 showed the *mechanism* is not localizable at node/pattern granularity;
  sequential accumulated state is disfavored where testable. Two-site L0→L1
  single-digit hand-off (B11) and the deep-chain edge path-patch (entry 1) remain.
- Does attention aggregate or transport operands at the carry node? (A2
  re-test.)
- Does the model *use* digit magnitude / circular ordering in its computation,
  regardless of a weak embedding geometry? (Causal; backlog B1.)
- Is the tri-state `ST` code well-separated at a *confirmed* ST node in the full
  space, or is it (like the digit circle) a low-variance projection? (Entry 2,
  re-scoped to a confirmed ST node.)
- Does the weak embedding circular ordering survive LayerNorm folding? (A-9.)
