# Study: geometry-certificate (study-geometry-certificate.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

**Status: DESIGNED 2026-07-16 (evening) — pre-run only. Awaiting the sprint
combined skeptic gate, then implementation + overnight run. No code has been
written or executed for this study yet.** Owner: `maths` thread (latent-geometry
stream). Paired study: [study-geometry-factorization.md](study-geometry-factorization.md)
(shares the stimulus grids and activation caches; one collection pass, two
analysis batteries).

## Pre-run (write before the experiment)

- **Question**: Is the resolved-carry computation implemented as a
  **place-value dominance code along a single carry rail**? Concretely: at a
  consumer's combiner input, per-source-site contributions along the CE16/CE17
  carry axis should show (a) per-site class values ordered
  `p_i < u_i < q_i` (write for local class 0 < write for `U` < write for
  local class 1), with the `U` value additionally leaning toward the incoming
  single-step carry (CE13); and (b) attention-weighted class gaps
  `g_i = w_i·(q_i − p_i)` that form a dominance hierarchy in digit
  significance (each site's gap exceeding the sum of all lower sites'
  gaps, up to measured margins) — so that a **thresholded** linear read
  (the CE17 STEP at α*) provably computes TriAdd for every reachable class
  configuration.

- **Motivation**: The mechanistic account is settled to working-axiom
  standard (ST writers → L1 consumer read where the canonical carry emerges
  (CE24) → STEP combiner (CE17)). What is missing — and what the human has
  directed this stream at — is the *geometric* account: what arrangement of
  the per-site write manifolds makes a mostly-static attention average yield
  a depth-invariant resolved carry. There is a sharp a-priori argument that
  shapes this study: **an exact linear read of local-class-only writes cannot
  equal the resolved carry at depth ≥ 2** (two-site counterexample: configs
  (U,1)/(U,0) force a nonzero low-site gap on the read, configs (1,0)/(1,1)
  force that same gap to zero). A **thresholded** linear read escapes the
  contradiction iff the class values are monotone with `U` strictly between
  the committed values and the weighted gaps form a dominance (place-value)
  hierarchy. If the measured geometry satisfies those inequalities, the STEP
  combiner (CE17) stops being an incidental detail and becomes the thing that
  makes local-class writes + a linear attention read *sufficient* — a
  geometry-explains-mechanism result the paper's representation section can
  use directly. It would also convert two standing puzzles into predictions:
  the asymmetric threshold α*≈0.75, and the CE18 large-n axis collapse
  (see G3: bounded norms + dominance ⟹ gap compression with digit count).

- **Hypothesis / competing reads** (stated neutrally):
  1. **G2 dominance code**: per-site rail values ordered (`p_i < u_i < q_i`,
     `u_i` split by cin per CE13), weighted gaps super-increasing in
     significance, measured α* inside the feasible-threshold interval; the
     `=`-token arm contributes a class-independent bias (the "depot" =
     the threshold unit's bias anchor).
  2. **Interaction/attention-selected read**: no per-site static ordering or
     dominance; the resolved carry instead comes from prompt-dependent
     attention reallocation (CE8-style routing) or non-additive LN/OV
     interactions. Predicts additivity (T3) fails and per-prompt dominance is
     absent while the canonical carry (CE24) still transfers.
  3. **Ordered-but-uncertified**: ordering holds (T1) but gaps do not
     dominate (T2 fails) — the model then relies on the rarity of adversarial
     lower-digit configurations rather than a worst-case-correct geometry;
     the certificate (T4) fails on enumerable configs the model actually gets
     right, implicating non-additive rescue.

- **Design**:
  - **Models**: `add_d6_l2_h3_t15K_s372001` (primary), `add_d5_l2_h3_t15K_s372001`
    (replication). Stretch: `add_d10_l2_h3_t40K_s572091` + d13 (CE18's empirically
    identified consumer registry — needed for G3's compression law).
    Negative control: `make_untrained_control` of the primary.
  - **Sites/wiring (from the maps + CE14/CE16 registries — do not re-locate)**:
    the answer-position consumer head pair (H1/H2-type, per model registry),
    the combiner MLP input at ≥2 consumers per model — at minimum the
    **sign-position consumer** (leading digit; full-depth cascade, CE15) and
    **one middle-digit consumer** (CE14). Source sites: the map-named
    question-tail/sign ST writer positions (CE13/CE17 strong writers).
  - **Carry rail** ĉ: refit per model exactly as CE16/CE17 — unit vector
    between committed-0 and committed-1 class centroids of the combiner
    input; α coordinates normalized to those endpoints.
  - **Stimuli** (reuse `build_carry_chain` / `make_cascade_operands`
    families): (i) per-site class grids — for each source site `i`, fix its
    local class ∈ {0 (pair ≤ 8), 1 (pair ≥ 10), U (pair = 9)} while drawing
    other digits from controlled families, including both values of the
    site's incoming single-step carry on the U arm (CE13); (ii) chain
    families depth k = 1..4 at the two consumers; (iii) 9-free variants
    (CE24's `66666+33334`-style) to break class/digit-identity correlation;
    (iv) a random-question holdout for T3. Target N ≈ 300–500 per cell;
    report per-cell N.
  - **Per-prompt decomposition** (the core measurement): for each prompt and
    consumer, decompose the combiner-input along ĉ into per-source-position
    contributions `w_i · OV_h(LN(resid_i))·ĉ` using the tested
    `maths_edge_patch.head_ov` + `ln_scale` (LN folded per prompt, per head,
    summed over the H1/H2 pair as in CE16), plus the direct/residual arm.
    This gives, per prompt: per-site rail contributions, the `=`-arm
    contribution, and the measured α.
  - **Batteries**:
    - **T1 ordering & transparency**: class-conditional per-site rail values
      `p_i, u_i, q_i` with bootstrap CIs; test `p_i < u_i < q_i` and the CE13
      cin-split `u_i(cin=0) < u_i(cin=1)`, both within `(p_i, q_i)`; report a
      midpointness index `|u_i − (p_i+q_i)/2| / (q_i − p_i)`.
    - **T2 dominance profile**: `g_i = w̄_i(q_i − p_i)` vs `Σ_{j<i} g_j` per
      consumer (with the attention-weight dispersion across prompts shown, not
      assumed static). **T2b irrelevance**: sites *above* the consumer's digit
      must show ≈ 0 class-dependent rail contribution at that consumer (else
      they must enter the certificate).
    - **T3 additivity / reconstruction**: predict α on held-out real prompts
      from the per-site static class values + measured per-prompt attention
      (α̂ = [Σ w_i v_i(class_i) + bias]/(endpoint gap)); report R²(α̂, α) and
      the agreement rate of `1[α̂ > α*]` with the model's actual carry-out.
    - **T4 certificate**: from measured (p_i, u_i±δ_i, q_i) and per-prompt
      attention ranges, compute the **feasible threshold interval** for exact
      TriAdd at depth ≤ k (lower bound: worst-case deciding-bit-0 config;
      upper bound: worst-case deciding-bit-1 config, enumerated analytically
      over class configurations). Report: interval non-empty? Does the
      measured α* lie inside it? Margins in combiner-input-noise SD units.
    - **T5 (stretch, G3)**: gap profiles on d10/d13; fit `g_i ~ ρ^i`; compare
      the smallest gaps to the noise floor; relate to CE18's axis-separation
      collapse (sep ~30 → ~6).
  - **Minimal effect of interest**: per-site class-gap `q_i − p_i` ≥ 2× its
    bootstrap SE at the map-named strong writers; dominance margin
    `g_i − Σ_{j<i} g_j` resolvable at ≥ 2× SE for at least the top two sites
    of each consumer. If per-cell N cannot resolve this, say so before
    interpreting T2 as negative.

- **Positive control**:
  - **PC1**: with this study's caches, reproduce CE16's joint head-pair edge
    flip (real-pair 1.00 / deciding-matched null 0.00) at one consumer per
    model — proves sites, heads, LN folding, and the rail are wired as in the
    accepted studies.
  - **PC2**: ĉ separates committed-0/1 combiner inputs at the CE16/CE17
    magnitude for that model (d5/d6 sep ≈ 30-class; d10/d13 expected weaker
    per CE18 — record, feeds T5 rather than failing the control at large n).
  - **PC3 (wrong-axis null)**: replacing ĉ with a random unit vector must
    destroy T1 ordering and T2 structure (guards the recurring
    low-variance-projection trap — the rail here is causally anchored by
    CE16/CE17, but the analysis must still show its structure is not generic).

- **Success condition** (defined now): T1 ordering holds (all strong-writer
  sites, both consumers, CIs clear) AND T2 dominance holds at the sign-position
  consumer for the top ≥ 3 sites AND T3 R² ≥ 0.7 with threshold-agreement
  ≥ 0.9 on chain families AND T4 interval non-empty with α* inside at k ≤ 3 on
  d6. → G2 confirmed (place-value dominance code); write CE entry; feed the
  paper hand-off section (b).

- **Failure condition** (defined now, requires PCs passing and power met):
  T1 ordering violated (u_i outside (p_i,q_i) beyond CI at ≥ 2 strong sites)
  OR per-prompt deciding-site contribution is not the largest class-dependent
  term on depth-≥2 chains (median across prompts) → G2 refuted; report which
  competing read (2 or 3) the residual structure supports.

- **Ambiguous / invalid condition**: PCs fail → invalid (fix harness, do not
  interpret). T1/T2 pass but T3 fails (R² < 0.4) → "ordered rail,
  non-additive read": G2 scoped down to ordering-only; certificate withheld;
  the non-additivity itself becomes the finding. Attention dispersion so large
  that w̄_i is unrepresentative → report per-prompt dominance only, skip T4.
  d10/d13 gaps below noise floor → T5 reports the floor (that is G3 *content*,
  not a failure).

- **Skeptic review (pre-launch)**: PENDING — sprint-mode combined gate
  (single separate-thread pass covering pre+post per the 2026-07-16 sprint
  protocol) must run before launch. Specific questions for the skeptic:
  (1) is the per-prompt OV/LN decomposition faithful enough that "per-site
  contribution" is well-defined (cf. CE10's LN-damping lesson)? (2) does the
  T4 enumeration assume independence between sites' write values that the
  stimulus grid can't license? (3) is the deciding-site-dominance metric
  (T2/failure condition) redundancy-safe under axiom 2?

- **Decision impact**: G2 confirmed → new CE entry; A10/A6 gain a geometric
  implementation account; α*'s value and the `=`-depot get explanations;
  paper section (b) gains a constructive geometry story; G3 becomes the
  cross-size narrative for CE18's collapse. G2 refuted → the geometry stream
  pivots to the interaction/routing read (competing read 2) and the paper
  keeps the current representation section unchanged. Either way score G2,
  G3 (T5), and touch A3 (U-placement), A5 (attention dispersion data), A10.

- **Risks / confounds**: low-variance-projection trap (mitigated: causally
  anchored rail + PC3); LN folding subtleties (reuse the tested
  `maths_edge_patch` helpers verbatim); head-pair asymmetry (report per-head
  and summed); `=`-arm class-dependence (test it — if the `=` contribution is
  class-dependent, the bias-anchor reading is wrong and CE16's depot needs a
  caveat, which is itself informative); d5's weaker/borderline sites (CE13);
  cross-config additivity assumed by T4 but tested by T3; deep-chain cells at
  d10/d13 may wash out as in CE19 (then restrict T5 to shallow k).

- **Expected artifacts**: `scripts/geometry_certificate.py` (to be written at
  launch; reuses `maths_probe.collect_site_activations`, `maths_edge_patch.head_ov`
  / `ln_scale`, `maths_temporal_finalization.build_carry_chain`,
  `maths_cascade.make_cascade_operands`) → `results/study-geometry-certificate/results.json`
  + figures: per-consumer gap-profile bars (g_i vs Σ_{j<i} g_j), α̂-vs-α scatter,
  feasible-threshold interval with α* marked, d10/d13 gap-compression curve.

## Post-run (fill in after the experiment)

*(not run yet)*
