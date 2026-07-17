# Study: geometry-certificate (study-geometry-certificate.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

**Status: RUN 2026-07-16 (overnight), dual-gated (see Post-run). Verdict:
G2 partial (low-medium — ordered dominance geometry supported; worst-case-certificate
+ U-midpoint forms scoped/refuted); G3 low (consistent, not discriminated).** Owner:
`maths` thread (latent-geometry stream). Paired study: [study-geometry-factorization.md](study-geometry-factorization.md)
(shares the stimulus grids and activation caches; one collection pass, two
analysis batteries).

## Executive summary

**Question:** is the resolved-carry computation implemented as a **place-value
dominance code on a single carry rail** — per-site class values ordered
`p_i < u_i < q_i` (U leaning to the incoming carry, CE13), super-increasing
attention-weighted gaps in digit significance, and the CE17 STEP threshold
sitting inside the feasible interval that makes TriAdd provably correct?

**Verdict (`add_d6_l2_h3_t20K_s173289` + `add_d5_l2_h3_t15K_s372001`, plus d10/d13
for G3; all acc 1.000; per-prompt LN-fair OV decomposition, reconstruction
exact to ~5e-8):**

- **G2 → LOW-MEDIUM (partial).** SUPPORTED: an ordered, trained-specific carry
  rail (class-1 > class-0 at every source site), super-increasing weighted gaps
  within the cascade region, cin-dependent U (never a static third symbol,
  reinforcing A3-low), a static-attention additive read (d5) that a single rail
  threshold turns into carry-out (agreement 0.92–1.00), and a **met worst-case
  feasibility certificate at the d6 middle consumer** (non-empty + α\* inside +
  config-accuracy 1.00). NOT established / refuted: U-as-neutral-midpoint (fails
  at the dominant deciding site — a full cin-gate there); additivity at the
  primary d6 consumer (static R² 0.24–0.42, ambiguous zone); worst-case
  certification at the leading digit (empty on both shared and local rails —
  a genuine compressed top-gap); α\* strictly inside at 3/4 consumers. So the
  dominance code is real and does computational work, but "provably computes
  TriAdd everywhere via one STEP" holds only at a middle digit.
- **G3 → LOW (consistent, not discriminated).** The carry-axis dynamic-range
  collapse (sep 28.6/32.2 → 5.6/6.0 at d10/d13) reproduces CE18's 30→6 as
  physics, not instrument failure; but the strict geometric ρ^i decay law is not
  discriminated from plain irrelevance / per-consumer renormalization.
- **Touches** A3 (cin-dependent U on the resolution axis), A5 (static-attention
  additive read beats per-prompt-attention — supports mostly-static routing),
  A10 (gives the combiner STEP a geometric implementation: a threshold on a
  super-increasing dominance rail).

**Controls (all PASS):** PC1 reproduces CE16 (head-pair flip 1.00 / null 0.00);
PC2 carry-axis sep 28.6/32.2; PC3 wrong-axis gap collapse ~50–100×; untrained
control sep 2.67/2.68 (trained-specific); local-rail control refutes the
shared-rail-misalignment confound. Artifacts:
`scripts/geometry_certificate.py`, `results/study-geometry-certificate/results.json`.
Full detail in [Post-run](#post-run-run-2026-07-16-overnight).

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

## Post-run (run 2026-07-16, overnight)

**Status: RUN + dual-gated (combined skeptic pass BLOCKed the first interpretation;
re-run with the two demanded controls; corrected read below).** Owner: `maths`
thread (latent-geometry stream). Artifacts:
`results/study-geometry-certificate/results.json` + `run_full.log`. Code:
`scripts/geometry_certificate.py` (reuses `maths_edge_patch.head_ov`/`ln_scale`,
`sv_implementation.carry_axis`/`pair_at_top`/`twin_pair`, `deep_cascade_mechanism`
chain builders, `node_output_encoding.ST_NODES`, `sv_compounding.CONSUMER_HEADS`).

**Models actually used (disclosure):** `add_d6_l2_h3_t20K_s173289` (primary) and
`add_d5_l2_h3_t15K_s372001` (replication) — the **CE13–CE17 registry models**
(the wired `ST_NODES`/`CONSUMER_HEADS`/`GEO_CFG` carry-rail exist only for these
seeds). The pre-run's "`add_d6_..._t15K_s372001`" was a pre-registration slip vs
the actual SV-implementation study model; using the wired-registry models is
required to reuse the CE16/CE17 sites and rail. T5/G3 on `add_d10_..._s572091`
and `add_d13_..._s572091` (CE18 empirical consumer registry). All acc 1.000.

### Core measurement is exact (skeptic Q1 answered)

The per-prompt LN-fair OV decomposition is **exact**: summing the per-source-site
rail contributions + the skip/bias baseline reconstructs the measured combiner-input
α to `recon_err ~5e-8` (both models). The baseline had to include the attention
output bias `b_O`, the LN2 bias, AND the non-consumer heads' output at the position
(their omission was the only source of error). "Per-site contribution" is therefore
well-defined and additive under the frozen (this-prompt) LN std.

### Controls (all PASS)

- **PC1** (reproduce CE16): joint head-pair edge flip **1.00** / deciding-matched
  null **0.00**, both models.
- **PC2** carry-axis committed-class separation: **28.6** (d6) / **32.2** (d5) —
  CE16/CE17 magnitude.
- **PC3** wrong (random) axis: T1 ordering flag drops (1.0→0.4–0.75) and, decisively,
  class-gap **magnitude collapses ~50–100×** (0.14→~0.002). (Lean on gap magnitude,
  not the ordering flag, whose random-axis null is only ~0.4–0.6.)
- **Untrained negative control** (added at skeptic request): on an untrained twin the
  "carry axis" sep collapses to **2.67/2.68** (vs 28.6/32.2) and the max class-gap to
  **~0.006** (vs ~0.14) — the geometry is **trained-specific, not a pipeline / 10-points-
  in-high-D artifact**.
- **Local-rail control** (added at skeptic request; the decisive one): refitting the
  rail at **each consumer's own consuming position** from its delivered-carry twins
  (PC2_local 23–27, all strong) reproduces the same per-digit gap profile — in
  particular the **leading consumer's top-digit gap stays compressed on its own rail**
  (d6 local: digit3=0.12 ≫ top digit4=0.02; d5 local: digit2=0.073 ≫ top digit3=0.003)
  and its certificate stays empty, while the **middle consumer's certificate stays
  non-empty (cfg-acc 0.89–1.00)**. So the leading-vs-middle contrast is **genuine, not
  shared-rail misalignment** (the skeptic's decisive confound is refuted).

### Battery results (final, full-N)

**T1 ordering & transparency — largely holds, one scoped exception.** Class-1 rail >
class-0 at **every** source site both models (pos_gap 1.00); p<u<q and U-inside hold
at ~all moderate-weight sites; U carries the **CE13 cin lean**. Exception (skeptic
Q on transparency): at the single **highest-weight deciding site** (d5 middle, w=0.56)
U **overshoots** the committed range (u0=−0.198 < p=−0.027; u1=+0.235 > q=+0.003;
ordered=False) — there U is not a neutral midpoint but a **full cin-controlled gate**.
Net: U is **cin-dependent on the carry-resolution axis** (never a static third symbol
— consistent with A3-low/CE6/CE7); "U as neutral midpoint" holds only at
moderate-weight sites.

**T2 dominance — super-increasing within the cascade region; top digit compressed at
the leading consumer.** Per-digit weighted gaps are super-increasing in significance
up to the deciding region (d6 middle 0.009→0.034→0.139; d5 middle 0.112→0.147), on
**both** the shared and the local rail. The **top** digit each **leading** consumer
serves has a **compressed** gap (d6 digit4=0.02 after digit3=0.093; d5 digit3=0.003
after digit2=0.047) — persistent on the local rail (genuine).

**T3 additivity (STATIC-attention reconstruction — the correct metric).** d5 leading
R²=**0.79**, middle **0.96** → additive; d6 leading **0.24**, middle **0.42** → in the
non-additive/ambiguous zone (the pre-registered R²<0.4 → "ordered rail, non-additive
read"). The **per-prompt-attention** reconstruction is *worse* (negative R²) — i.e. a
**static-attention** additive read is the better model (consistent with A5
mostly-static routing). Threshold-agreement (1[read>α*]==carry-out) is **0.92–1.00**
everywhere. (separation_bacc=1.00 is demoted to a consistency check — near-tautological
since the axis is the class-mean difference.)

**T4 feasibility certificate.** Non-empty **and α\* inside with config-accuracy 1.00**
at the **d6 middle** consumer (geo and local rail) — the geometry provably certifies
TriAdd there. Non-empty but α\* outside (cfg-acc 0.78–0.89) at the **d5 middle**;
**empty** at **both leading** consumers (cfg-acc 0.49–0.68) on both rails. The
operating α\* (rail coordinate ≈0.49–0.54) is **not** CE17's interpolation-α\*≈0.75
(different coordinate — flagged, not "explained").

**T5 / G3 dynamic-range law.** Carry-axis sep collapses **28.6/32.2 (d5/d6) → 5.57/6.04
(d10/d13)** — reproducing CE18's 30→6 as **dynamic-range compression** (physics), not
instrument failure. Gap profiles are super-increasing (d10 ρ≈6.9, r²=0.78; d13 ρ≈1.9,
r²=0.41 on a non-contiguous digit set), with low-significance gaps at the noise floor.
**Caveat (skeptic):** the near-zero low-digit gaps are not discriminated from **plain
irrelevance / per-consumer renormalization** (those digits are far below the consumer),
so the sep-collapse re-explanation of CE18 is the durable part; the strict ρ^i decay
law is suggestive at d10, weak at d13.

### Verdict

- **G2 → LOW-MEDIUM (partial; constructive geometry supported, strong certificate form
  scoped/refuted).** SUPPORTED: an ordered place-value carry rail (class-1>class-0,
  trained-specific, rail-robust), super-increasing weighted gaps within the cascade
  region, cin-dependent U (CE13), a static-additive read (d5) that a single rail
  threshold turns into carry-out (agreement 0.92–1.00), and a **met worst-case
  certificate at the d6 middle consumer** (non-empty + α\* inside + cfg-acc 1.00).
  NOT established / refuted: U-as-neutral-midpoint (fails at the dominant deciding
  site); additivity at the **primary d6** (static R² 0.24–0.42); worst-case
  certification at the **leading digit** (empty on both rails — compressed top-gap,
  genuine); α\* strictly inside the interval at 3 of 4 consumers. The place-value
  dominance code is real and does computational work, but "provably computes TriAdd
  everywhere via a single STEP" is met only at a middle digit.
- **G3 → LOW (consistent, not discriminated).** The carry-axis dynamic-range collapse
  (CE18 re-explanation) is real; the geometric-decay law is not discriminated from
  renormalization/irrelevance.
- **Touches:** A3 (U is cin-dependent on the resolution axis, never a static third
  symbol — reinforces A3-low; adds the dominant-site cin-gate overshoot); A5
  (static-attention additive read beats per-prompt-attention — supports mostly-static
  routing); A10 (gives the combiner STEP a geometric implementation: a threshold on a
  super-increasing dominance rail; α\*-frame and `=`-depot-as-bias-anchor notes).

### Skeptic review (combined gate — run 2026-07-16, separate subagent thread)

- **Round 1: post-result BLOCK.** The gate flagged a decisive confound (rail fit at the
  middle consumer's position, reused for the leading consumer → the compressed-top-gap
  could be rail-misalignment/G1, not compression) plus over-claims on U-transparency,
  the additivity metric (report STATIC R², honoring the d6 ambiguous condition), the T4
  α\*-inside sub-condition (α\* outside the interval at 3/4 consumers; α\*≠0.75),
  separation_bacc being near-tautological, the primary-model swap, and the missing
  untrained control.
- **Resolution:** added the **local-rail** control (refutes the confound — compression
  persists on the consumer's own rail) and the **untrained** control (structure is
  trained-specific); re-ran full-N; rewrote every flagged claim to the scoped wording
  above (STATIC R² as the additivity metric; T4 reported as met-at-d6-middle-only;
  α\*-frame flagged; sep_bacc demoted; model choice disclosed). Post-correction the
  interpretation is the scoped G2-partial / G3-consistent read.

### Immediate next read

Post-deadline: the paired [study-geometry-factorization.md](study-geometry-factorization.md)
(G1/G4) shares these caches; and a genuine multi-depth certificate at n=10/13 (where
G3 predicts the sharpest gap structure) would test whether the leading-digit
compression is the same physics as CE18's collapse. Figures (gap-profile bars, α̂-vs-α
scatter, feasible-interval plot, d10/d13 compression curve) deferred — numbers in
`results.json`.
