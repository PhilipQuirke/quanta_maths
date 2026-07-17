# Maths Results Synthesis (maths-results-synthesis.md)

Read role and rules: [Results Synthesis](thor-document-rules.md#results-synthesis).

Mid-level empirical synthesis for the `maths` thread's primary results themes.
Organize by empirical question, not by experiment chronology. Keep only figures
that change interpretation or priority. This is the normal home for detailed
mechanism interpretation that would overload
[maths-results-summary.md](maths-results-summary.md).

## Q: How are digit values represented at the token level (embedding / unembedding)?

Status after the 2026-07-14 digit-embedding geometry audit (weights-only; see
[claim CE1](maths-claim-evidence.md#ce1)
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
primary question; see [CE2](maths-claim-evidence.md#ce2)
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
[CE3](maths-claim-evidence.md#ce3)).

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
[CE4](maths-claim-evidence.md#ce4)).

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
[CE5](maths-claim-evidence.md#ce5)).

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
[CE6](maths-claim-evidence.md#ce6)).

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
[CE7](maths-claim-evidence.md#ce7)).

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
[CE8](maths-claim-evidence.md#ce8)).

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
[CE9](maths-claim-evidence.md#ce9)).

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
([CE10](maths-claim-evidence.md#ce10)).

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
([CE11](maths-claim-evidence.md#ce11)).

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
([CE12](maths-claim-evidence.md#ce12)).

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

## Q: How does each map-named output-only node (ST/SA/SC) encode its value (C5 step 1)? — local class + single-step U-resolution; CE3 = redundancy (CE13)

Status after the 2026-07-16 node-output-encoding study
([CE13](maths-claim-evidence.md#ce13),
revising [CE3](maths-claim-evidence.md)). First study under the human C5 refocus
(start from the paper's HF verified node maps).

- **ST/SC nodes encode their class** at their write (paper output-only tags
  confirmed). The **ST write co-carries single-step local U-resolution** (on a `U`
  pair the write depends on the immediate carry-in) — so not a *pure* local class
  code, but **not** multi-digit compounding (single-step; carry-out = cin by
  definition on `U`). A10's "ST-local / L1-compounds" division is **refined**, not
  settled.
- **CE3 refined to redundancy (baseline-controlled)**: single-node interchange
  flips nothing (reproduced), but low-digit ST-node mean-ablation exceeds an
  untagged-head baseline (map causal tags **vindicated** — a C5 win); high-digit ST
  nodes redundant. Matches the paper's "redundant ST node" caveat.
- **Map-named `SA` L0 heads do not write the answer digit** (except the leading
  digit) — the sum is an answer-position computation (CE11/CE12); the `SA` L0 tag
  marks operand-fetch.
- **Method note (two Gate-2 rounds, both directions)**: first draft over-read "ST
  write is local" (definite-pair artifact); the fix over-swung to "compounding
  begins at the ST write"; settled on "single-step U-resolution co-located, not
  multi-digit compounding". Durable lessons: locality tests must use the case where
  the toggled variable can matter (U, not definite) and report a ratio; ablation
  claims need an untagged-head baseline; single-step cin ≠ multi-digit compounding.

## Q: How is the deep carry compounded and delivered to the combiner (C5 steps 2-4 / A10)? — carry-specific distributed delivery, selection unshown (CE14)

Status after the 2026-07-16 SV-compounding study
([CE14](maths-claim-evidence.md#ce14)).

- **A10 core partially confirmed**: patching the map-named answer-position L1
  consumer heads' output edge into the L1-MLP combiner flips the top cascade digit at
  ≥ 2 depths, **carry-specifically** (deciding-matched null = 0.00) and
  **consumer-head-specifically** — so attention (not the direct residual path)
  delivers the deep carry to the combiner. This strengthens CE10's one-depth crumb.
- **But**: single-head **selection is not shown** (tracking head H2 ≠
  single-depth-edge-causal H1); it is **sufficiency not necessity** (ablating the
  sufficient head H1 is harmless; H2 carries necessity + tracking → redundant pair);
  the **direct path is not excluded** (underpowered arm); value content is not
  head-specific. A9 (single-head selection) **not supported**; A6 economy supported
  at H2.
- **Method note (THREE Gate-2 rounds on the flagship conjecture A10)**: round 1
  over-claimed "R-A10-selection at L1.H1" (a three-attribution smear); round 2
  over-corrected to "not carry-specific / A10 not confirmed" on a **broken null**
  (it toggled the deciding carry, so null=real=1.0 was the *expected* signature of a
  carry-specific head); round 3 fixed the null to deciding-matched (null=0.00) →
  carry-specific, landing the calibrated **R-A10-distributed-delivery**. Durable
  lesson: a specificity null must hold the tested variable FIXED — a "same-class
  null" that re-toggles the deciding carry proves nothing; and edge SUFFICIENCY
  (patching flips) is not NECESSITY (ablation) — report both.

## Q: How is the leading digit produced in the hard case (C5 step 5)? — same as middle digits (CE15)

Status after the 2026-07-16 leading-digit-walkthrough study
([CE15](maths-claim-evidence.md#ce15)).

- The leading digit `A_top` in `99..9+1` is produced by the **same carry-specific
  L1-head-edge delivery** to the answer-position (sign) L1-MLP combiner as the
  middle digits (CE14): the sign-position L1 head edge flips `A_top` carry-specifically
  (real 1.00, deciding-matched null 0.00). Verified 5-digit across a genuine depth
  spread; 6-digit at deep chains only (shallow leading cascades carried by an
  unadjudicated path).
- **A10 consolidated across ALL answer digits including the hardest — but NOT
  raised**: this extends CE14's *partial* confirmation to the leading locus with the
  same open questions (necessity, selection, direct-path) plus leading-specific
  limits (economy uninformative at the sign bottleneck; 6-digit depth-restriction).
  Link 4 (readout) is quarantined; the A10 verdict rests only on Link 3.
- **C5's 5-step program is now COMPLETE for the addition model**: CE13 (output
  encodings) → CE14 (SV compounding) → CE15 (leading-digit hard case), all landing
  on: carry-specific attention-edge delivery to answer-position L1-MLP combiners,
  redundant and partially confirmed.
- **Method note**: nearly clean Gate 2 (the discipline from the CE14 3-round
  ordeal paid off — Link 4 quarantined, consolidate-not-raise honored structurally).
  One fix: two adjacent maximal chains (k=4,5) are not an independent 2-depth spread
  → 6-digit re-labeled "verified deep-chains only".

## Q: What does the SV mechanism actually implement (the 4 details)? — canonical carry, distributed-ST source, SV-path effective, class-necessary (CE16)

The SV-implementation sprint estimated A10's four implementation parameters on the
confirmed wiring ([CE16](maths-claim-evidence.md)):
- **(i) Message = canonical resolved carry** — the head→combiner edge is
  format-invariant across deciding positions (within-chain carry-probe transfer
  1.00 = within-acc); a deciding-position co-rider is *decodable* (non-causal).
- **(ii) Source = distributed question-tail ST cluster, NEVER `=`** — the `=`
  value arm flips 0.00 with carry-axis OV-projection ≈ 0, so **`=` is a depot, not
  a value source** (adjudicates the old source fork toward distributed ST, against
  the "`=` carry depot read as a value" lean). Deciding-ST is carry-specific and
  dominant at 6d k3; chain-ST carries the mass elsewhere.
- **(iii) Path = head-pair (SV) effective; skip carries negligible carry** — real
  head-pair patch flips 1.00; the measured real-skip carry is ≈200× smaller and a
  power injection at that magnitude (1× and 2×) flips 0.00, so the model does not
  route carry through the skip (**not** a formal exclusion — closes CE14's
  direct-arm gap in the "is it used" sense). The pair is **class-necessary**
  (joint H1+H2 ablation: cascade collapses, carry-free spared, over a ~0 untagged
  baseline; necessity-over-baseline 1.07 6d / 0.85 5d).
- **(iv) Combiner form = OPEN** — Battery F instrument invalid (α-sweep produced no
  flips). Sole remaining A10 implementation item (B2).
- **Dispositions**: A10 core held at medium-high, items i–iii resolved; A9
  selection **stays retired** (deciding-ST dominant at only 1 independent depth);
  A6 economy **raised to class level**.
- **Method note**: dual-gated combined sprint pass — pre-launch
  PASS-WITH-CONDITIONS caught two coordinate-space traps (skip power control in
  the wrong residual space; M projection needing LN-normalization) resolved as
  SI-1..SI-10; post-result PASS-WITH-CORRECTIONS downgraded two over-claims ("skip
  excluded" → "carries ≈0"; "deciding-ST source" → "distributed ST"). Two design
  bugs (value-only R patch under-read; raw residual scale-confound) surfaced in a
  fast smoke run and fixed before the full run.

## Q: Where is the multi-digit carry compounded (L0 relay vs L1 read), and what is the combiner's form? — combiner is a STEP; locus is R-mixed (CE17)

Building on CE16's puzzle (a canonical resolved carry on the wire, assembled from
sources that individually hold only single-step resolution, CE13):
- **The combiner is a STEP function** (A10 item iv, the last A10 detail): an
  on-manifold α-sweep — interpolating the *real* captured head-pair edge
  contribution and substituting it PRE-LN, endpoint-gated to CE16's 0/1 (fixing
  CE16's dead Battery F) — flips the digit sharply at threshold **α*≈0.75** in
  both models while the carry-axis projection rises linearly. A hard
  discretization, not a graded dial.
- **The L0 tail-relay (A11) is only a single-site representational trace.** In
  the 6d model one question-tail ST site (P11H2, horizon m=1) decodes the resolved
  carry (bacc 1.00) exactly at the depth where it can see the deciding digit — the
  A11 horizon prediction — but it fails at an adjacent depth and does not replicate
  in 5d.
- **The relay is causally undetermined.** Twin-interchange of the tail-ST writes
  flips the leading digit 0.00 at every depth, both models. The instrument is
  valid (reproduces CE13's ablation load-bearing) but tests a *different
  unit/target* than the interchange, so the null cannot distinguish redundancy
  (CE13 interchange=0.00) from interchange-too-weak-for-this-target (the CE16-F1
  trap). And the L1 edge output reconstructs from **local class alone**
  (r²≈0.95–0.99), so nothing forces the relay account.
- **Dispositions**: A10 iv RESOLVED (step, α*≈0.75); A11 → **low** (representational
  hint, unreplicated, causally undetermined); A9 **stays retired** (no causal
  selection at either locus); A6/C3 human sequential-cascade lean **not
  adjudicated** (only signal is the single unreplicated cell). **R-mixed, leaning
  L1-local-sufficient.** This is the last mechanism study before the paper
  hand-off (agenda entry 2). Dual-gated (CA-1..CA-6 pre-launch; F1/F2 post-result
  downgraded H-"replication" and Y-"redundancy-masked" over-claims).
- **Method note**: a fast smoke run caught two real bugs before the full run (a
  broken 5d horizon-inversion at the `=`/sign positions; a Battery-Y ablation
  control that under-measured until switched to CE13's exact `_mean_ablate_acc`
  unit); the pre-launch gate caught an invalid depth (k=4→d=−1 at n=3) fixed by
  re-pinning the chain-top to n_top=4.

## Q: Does the SV mechanism generalize across model sizes, and does its redundancy thin at scale? — skeleton+combiner generalize; redundancy is intrinsic (CE18)

Testing A12 (the layer-general SV interface) and the human's C6 prior (redundancy
= small-model slack, stripped at n≥10) on d5/d6/d10/d13 (all accurate):
- **The role skeleton is size-general**: question-tail/sign ST writers and
  high-Fail combiner MLPs are in the published d10/d13 maps, and a causal
  answer-position consumer head is identifiable at every size (though NOT map-tagged
  at d10/d13 — a role-transfer datapoint in itself).
- **The combiner is a STEP function from d5 to d13** — the one robust cross-size
  causal result (on-manifold α-sweep, endpoint-gated to each model's real 0/1,
  α*≈0.5–0.75). CE17's combiner finding generalizes.
- **But the causal SOURCE signatures do not reproduce at large n**: the carry axis
  collapses (sep ~6 vs ~30) and both the `=` and deciding-ST per-key contribution
  arms flip 0.00 at d10/d13, so `=`-not-a-source sits on a null background and the
  source-fork is probe-limited/untested at n≥10.
- **C6 is NOT SUPPORTED**: single-node ST ablation is ~0 at every size and the
  class-vs-single redundancy gap does NOT shrink across d5→d6→d10
  ({0.056, 0.116, 0.324} — increasing); d13 is inconclusive (whole-class ST
  ablation 0.040 ≈ CE13 single-node magnitude — instrument-weak at n_ctx 43). So
  redundancy **persists (does not thin) with size** — it reads as intrinsic to the
  learned algorithm, not capacity slack. Consequently scale did **not** rescue the
  A11 compounding-locus lever (Battery L untriggered); A11 stays low.
- **Dispositions**: A12 role-transfer + step-combiner → **medium-high**;
  A12-tightening / C6 → **low (not supported)**; A10 iv step **generalizes**; A11
  **unchanged (low)**. This is the last mechanism input before the paper hand-off:
  the paper can state the SV mechanism as a **size-general skeleton with intrinsic,
  non-thinning redundancy** and a step combiner, while noting the large-n causal
  source probes are instrument-limited.
- **Method note**: dual-gated. Pre-launch caught that the reused CE13/CE16
  registries are hard-coded to d5/d6 (rebuilt d10/d13 from the published maps + a
  PC2b anchor) and that the tightness index needed an interchange-independent leg
  (the class-vs-single ablation gap). Post-result downgraded two over-claims
  ("transfer confirmed" → "role+combiner confirmed, source probe-limited"; "C6
  refuted" → "not supported, d13 inconclusive"). A fast smoke run caught a broken
  5d/large-n carry-axis config and a consumer-ID attention-threshold that rejected
  the real d13 causal head (fixed to select by causal flip + carry-specificity).

## Q: Where is the multi-digit carry compounded — an L0 positional relay (A11) or the L1 read? — CONCLUSIVELY the L1 read (CE24, superseding CE19)

**Update (CE24, conclusive):** a redesign — reading the FULL residual at the
see-everything gather position and discriminating via **cross-depth transfer** (a
canonical carry transfers across chain-depths; raw input-reading cannot), on
**9-free** stimuli (the human's `66666+33334`/`66666+33433` insight, so no literal
`9` token leaks into the carry decode) — **layer-localizes** the compounding, both
models: the answer-agnostic carry transfers cross-depth **1.00 at the L1 combiner
input** (answer-top does NOT ride the carry axis, 0.12 vs 0.78 full-residual) while
**L0's output has no carry-specific canonical code** (extreme-pair transfer ~chance
with CIs, high within-depth ceiling; weak transfer = magnitude nuisance). This fixes
CE19's two flaws (OV-write washout; the ill-posed invisible-cell — MSD-first layout
+ causal mask make an L0 carry-direction relay architecturally impossible) and,
with CE16's causal head-pair delivery, places the emergence of the canonical carry
in the **L1 read**. Key methodological lesson: **decode ≠ computation** (the
resolved carry is a deterministic function of the inputs, so raw decodability tracks
input-visibility) — only cross-context transfer / causal tests localize a *computed*
abstraction. A11 rejected/low. Caveats: linear probe; L0 at `=` (ST/sign via
CE17/CE19); own causal battery invalid (locus per CE16). The CE19 account below is
retained for provenance.

**Token-time (CE24 TF addendum):** having settled the *layer* (L1 read), TF closes
the *when-by-token* question. Using 9-free graded chains with a run-break (so the
top carry is decorrelated from the local make-carry) and cross-deciding-position
transfer at each (token position × layer): **LAZY (representational) propagation +
eager local**, both models. The propagated canonical carry appears only at the
**sign token, L1** (~chance at every operand token and at `=`, never at L0) — a
**~2-token deferral past full input availability (D'_0)** and past the `=` gather
(chance; consistent with `=`-is-a-depot). The **local single-step make-carry is
eager, in-place at L0 at its own token** (CE13). The make-carry-token→D'_0 span is
information-availability (MSD-first layout, CE19), not laziness. Net picture:
**single-step resolution computed eagerly in place (L0); multi-digit propagation
represented lazily at the answer read (L1).** Reusable cross-model tool shipped in
the package (`quanta_maths/maths_temporal_finalization.py`) — the eager/lazy split
may differ by model, so run the zoo. Linear-probe; coarse tail; two small models.

### (superseded) Q: … — L1-read, not a relay (CE19)

CE17 left this R-mixed; CE18 showed scale won't sharpen it. CE19 settled it with a
better instrument:
- **The decorrelation lever**: a chain-ST site sitting inside the 999-run has its
  local digit-sum fixed at 9 (local class ≡ U, uninformative) while the resolved
  carry reaching it still flips 0/1 — so decoding the carry there cannot be
  local-class leakage (the CE17 Battery-L confound is removed; verified, local-class
  decode = 0.50 exactly).
- **The discriminator**: split decorrelated cells into VISIBLE (the site can see the
  deciding digit — it could compute the carry itself) vs **INVISIBLE** (the deciding
  digit is below the site's visibility horizon — a carry there could ONLY have been
  relayed to it). The invisible cells are the only test that separates an L0 relay
  from local computation.
- **Result**: at the invisible-decorrelated cells the resolved carry decodes at
  **chance** in both models. Where the write is readable at that depth (5d P9H1 k3,
  a per-depth readability control passes), that chance = a genuine **"no relayed
  carry"** → **R-L1-read**: the multi-digit compounding is completed in the L1
  consumer read, not by an L0 positional relay. At 6d the deep-chain writes wash out
  (per-depth control fails) → underpowered. The ST write carries the resolved carry
  **only when the deciding digit is its own digit** (local single-step, CE13) — no
  cross-position relay anywhere.
- **Knock-out** confirms the ST cluster is carry-**necessary** (ablating the
  sufficient set breaks the digit differentially over a 0.00 specificity-null and
  0.00 untagged baseline) but does not itself discriminate relay from local
  resolution — DH is the load-bearing battery.
- **Dispositions**: **A11 (positional L0 relay) → low (not supported, not
  rejected)**; compounding locus = **L1-read**; **A9 retired**; **A6** ST-class
  necessity again; the **human C3/sequential-cascade lean is not supported at the
  L0 tail**. Caveats: linear probe (non-linear relay not excluded); the refutation
  rests on one readable invisible cell (5d); 6d underpowered. This substantially
  settles the last open SV-mechanism question for the paper hand-off.
- **Method note**: dual-gated. Pre-launch was a BLOCK — the auditor caught that
  decoding on VISIBLE-decorrelated cells cannot separate A11-relay from "each site
  sums what it sees" (both predict decode-when-visible); fixed by making the
  INVISIBLE cells the discriminator. Post-result added a per-depth readability
  control that correctly downgraded 6d to underpowered (the shallow-depth instrument
  control did not license readability at the deep invisible depth) and softened
  "refuted" to "not supported" (linear probe).

## Mixed model — addition→mixed generalization (CE20–CE23, CE25)

*(Scope: the mixed add/sub model `ins1_mix_d6_l3_h4_t40K_s372001`; 3 layers, 4
heads; initialised from a 6-digit addition model. Distinct from the addition-model
Q&A above.)*

- **Q: Does the addition SV mechanism replicate on the mixed model, across ADD /
  SUB / NEG? — Yes, the representation; delivery is class-dependent (CE20).** The
  question-tail tri-state writers encode their class (ST/MT/NT ~1.00 vs untrained
  chance) and the resolved carry/borrow is a clean **binary** code at the last-layer
  (L2) combiner input (SV/MV/NV ~1.00). Delivery is carry/borrow-specific
  (deciding-null 0.00) but the *route* differs by class: **ADD residual-only;
  SUB/NEG residual + last-layer attention** — the inserted addition circuit resolves
  early and rides the residual; the freshly-learned subtraction cascades also use
  last-layer-attention delivery.
- **Q: Does the SV *implementation* (combiner form, message, source) replicate? —
  Yes for STEP/canonical/`=`-not-source; necessity redundancy-blurred (CE22).** The
  combiner is a **STEP** for all three classes (α-sweep, endpoints gated); the
  delivered carry is a **canonical format-invariant** code (cross-digit transfer
  1.00); `=` is **not** the middle-digit source. Class-necessity is not scored
  (writer-class ablation ≈ untagged baseline — redundancy, as in CE13/CE18).
- **Q: How is the answer sign `SGN` computed? — the `D≥D'` comparison delivered to
  the `=` combiner, the CE15 leading-digit analog (CE21).** Boundary crossing flips
  SGN 1.00; the sign is binary-decodable 1.00 at `=`; a sign-combiner edge patch
  flips it comparison-specifically.
- **Q: Is the operator a low-rank control selecting a shared engine (A7) or are
  add/sub separate circuits (C2)? — hybrid; A7's low-rank-control form refuted
  (CE21, CE23).** The engine is **shared at the combiner** (patching the L1 state
  ADD→SUB emits the correct ADD digit 0.96) and `SA`/`MD`/`ND` share head nodes,
  but the operator is **not** an additive rank-1 knob at either the combiner or the
  SLT selector, and no single SLT head selects — the add/sub selection is a
  **distributed, high-dimensional** L1 transformation (leans C2 on selection).
- **Q: Does the class-dependent delivery hold at multiple cascade depths? — Yes,
  depths 2/3/4 (CE25).** ADD residual-only, SUB/NEG residual + last-layer attention
  at every depth (deciding-null 0.00; untrained control delivers nothing). Clears
  the CE14 ≥2-depth bar on the mixed model. Reusable cross-model sweep in
  `quanta_maths.maths_cascade`.

## Where the story is still ambiguous

Standing ambiguities that constrain interpretation (the ranked plan and backlog
IDs live in the [experiment agenda](maths-next-steps.md); this list is only the
open empirical uncertainty, not the schedule):

- Whether the `=` resolved carry is causally USED downstream or recomputed per
  answer position, and whether `SV` at `=` survives a fair operand-recompute
  baseline — the open half of CE12.
- Whether the hand-off delivery holds at ≥2 depths (CE10 established only one),
  and how the consumer head + combiner MLP compute `carry_out` — under-powered,
  not settled.
- Why delivery differs by size (5-digit direct path live, 6-digit not).
- Whether a *transient* `{0,1,U}` tri-state exists at question positions before
  the answer position — the only regime A3 could still live in.
- Whether the tie-break economy (A6) holds for multi-digit `...999` cascades; CE9
  found the mechanism non-localizable at node/pattern granularity.
- Whether the model causally *uses* digit magnitude / circular ordering despite a
  weak embedding geometry, and whether that ordering survives LayerNorm folding.
