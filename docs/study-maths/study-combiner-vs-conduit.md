# Study: Combiner vs Conduit at the U-Flip Transmitters (study-combiner-vs-conduit.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #5

**Verdict: the tri-state `U`-combiner is the answer-position layer-1 MLP —
cleanly and replicably.** Using a node-level activation-invariance discriminator
(a combiner's output is invariant to `carry_in` for a *definite* digit but must
vary in the `U` regime), the answer-position L1 MLP passes the combiner signature
in both models (5-digit `P14.L1.MLP`, 6-digit `P16.L1.MLP`: definite-regime
activation diff ~0.004, U-regime ~1.2–1.3). Post-result skepticism further
verified the L1-MLP output routes to the class-determined `carry_out` centroids,
ruling out a "U-detector" artifact. Layer-0 nodes are **conduits** (carry the
running carry regardless of digit class — clean in the 6-digit model; the single
5-digit L0 candidate is borderline).

Recorded as CE5. This confirms the two-site mechanism — layer-0 conducts the
running carry, the answer-position L1 MLP combines it with the digit's sum-class
to resolve `U` — and is the U-resolver the prior (CE4) study could not pin down.

## Pre-run (write before the experiment)

- **Question**: Among the CE4 U-flip transmitters, which (if any) are true
  **combiners** — nodes that compute the correct `carry_out(n)` by reading both
  digit-`n`'s sum-class and the lower carry — versus **conduits** that merely
  relay the lower carry regardless of digit-`n`? And does the two-site
  hypothesis (a low-value-digit L0 MLP computes/holds the running carry; an
  answer-position L1 MLP applies it) hold, tested by an L0→L1 path-patch?

- **Motivation**: This is the last gate between "we located an MLP path that
  moves the U-flip" and a real mechanistic claim about *where the tri-state
  carry is computed*. It settles A6 (genuine carried/combined state vs
  transport), gives A3/entry-2 a *confirmed* tri-state locus, and directly tests
  A2's "MLP does the nonlinear combination" at the right node.

- **The discriminator (corruption control)**: the correct carry-out at digit `n`
  is `1 if (Dn+D'n > 9) or (Dn+D'n == 9 and carry_in == 1) else 0`. For a
  **definite** digit (sum ≤ 8 or sum ≥ 10) the carry-out is *independent* of
  `carry_in`. So:
  - Take a **target** with digit `n` **definite** and a known correct answer.
  - Patch the candidate node's activation from a **source** whose lower carry is
    the *opposite* of the target's (so the source relays a carry_in inconsistent
    with the target's true state), holding digit `n`'s class matched between
    source and target where possible.
  - **Conduit** prediction: it relays the source's (wrong-for-this-target)
    carry; if a downstream reader uses it, `A_{n+1}` is **corrupted** away from
    the correct answer.
  - **Combiner** prediction: it recomputes the correct carry-out for the
    definite digit (which ignores `carry_in`), so `A_{n+1}` stays **correct** —
    no corruption.
  This is non-vacuous because for a *definite* digit the two hypotheses predict
  *different* outcomes (corruption vs no corruption), unlike the earlier
  U-regime gate.

- **Hypothesis / competing reads** (neutral):
  - **R-combiner**: the node keeps `A_{n+1}` correct under the corruption control
    for a definite digit (recomputes), AND transmits the U-flip in the U regime.
    It is the U-resolver.
  - **R-conduit**: the node corrupts `A_{n+1}` under the corruption control
    (relays the wrong carry) — it carries the carry bit but does not combine.
  - **R-readout**: the node is the `A_{n+1}` readout site itself (downstream of
    resolution); patching it corrupts trivially (it is where the value is read),
    so it fails to distinguish and is excluded as the resolver by position.
  - Multiple transmitters may split roles (e.g. L0 = conduit/compute, L1 =
    combiner/apply); the design reports each node's classification, not a single
    winner.

- **Design**:
  - **Models**: `add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`
    (accuracy-verified). Candidate nodes = the CE4 U-flip transmitters (registry
    `results/study-u-resolution-path/u_resolution_registry.json`) plus the CE3
    make-carry heads (known binary, expected conduit-like or U-inert) as
    reference points.
  - **Corruption metric** (per node, per definite class lo/hi): over many
    matched pairs, `corruption_rate` = fraction where patching the node
    (source lower-carry ≠ target lower-carry, digit `n` definite in both) makes
    the target's `A_{n+1}` prediction **differ from its clean-correct value**.
    - High corruption (≥ bar) → **conduit** (relays the inconsistent carry).
    - Low corruption (≤ null) → **combiner** OR inert-for-definite (recomputes /
      ignores carry_in).
  - **Combiner confirmation** requires BOTH: (i) transmits the U-flip in the U
    regime (from CE4, re-verified here), AND (ii) low corruption under the
    definite corruption control. A node that transmits the U-flip but *also*
    corrupts the definite case is a **conduit** (it is moving the carry bit, not
    combining). A node that neither transmits U nor corrupts is inert.
  - **Same-state corruption null**: patch between source and target with the
    **same** lower carry (both consistent with the target) and digit `n`
    definite → should NOT corrupt (false-positive floor).
  - **L0→L1 path-patch (two-site hypothesis)**: patch the L0-MLP's output *and
    then* read whether the L1-MLP's U-flip contribution changes — operationally,
    (a) does patching L0-MLP alone in the U counterfactual transmit the flip
    (already CE4: yes), and (b) does freezing L0-MLP to a no-carry source while
    the rest of the target has a carry *remove* the L1-MLP's ability to resolve
    U (i.e. is L1's resolution *downstream of* L0)? Report the dependency.
  - **Positive control**: (1) node-level — the CE3 make-carry head under a
    *binary* carry corruption (inject inconsistent carry into a definite digit
    via the make-carry head) — this is expected to corrupt (it carries the
    binary carry), demonstrating the corruption metric can register a conduit.
    (2) A recompute reference — patching an operand-fetch (`SA`) head that does
    not carry the cross-digit carry should NOT corrupt `A_{n+1}`. If the
    corruption metric cannot separate these two references, the assay is
    `invalid`.
  - **Bar / null**: conduit if corruption ≥ 0.50 (tied to the make-carry
    reference's corruption rate minus a margin); combiner if corruption ≤ 0.10
    AND U-flip ≥ 0.50; both over ≥ 60 pairs per (node, class), both definite
    classes, both models.

- **Success condition** (defined now): each CE4 U-flip transmitter is classified
  combiner / conduit / inert by the corruption control (non-vacuous: the two
  references separate), and the L0→L1 dependency is reported, in both models.
  A node is confirmed the **U-combiner** iff it transmits the U-flip AND does not
  corrupt the definite case, replicated across both models (or the model-general
  pattern is stated).

- **Failure condition** (defined now): with the positive control separating the
  references — all U-flip transmitters corrupt the definite case (all conduits;
  no combiner among them) → the true combiner is elsewhere/distributed, or the
  "combiner" abstraction does not apply (the model may resolve U by a
  distributed sum rather than a localized combine). A substantive finding.

- **Ambiguous / invalid condition**:
  - Ambiguous: corruption rates land between null and bar (partial), or models
    disagree, or a node is combiner in one battery and conduit in another.
  - Invalid: positive-control references do not separate (make-carry reference
    fails to corrupt, or SA reference corrupts); or accuracy check fails.

- **Skeptic review (pre-launch)**: Run 2026-07-15 in a separate skeptic thread
  (docs + prior notes + Paper-2 facts only; no working-thread context).

  **Verdict: BLOCK.** The corruption idea is a real improvement (for a definite
  digit, combiner and conduit predictions genuinely differ *in principle*), but
  the assay as specified is not launchable — it repeats, a third time, the
  definite-digit carry-independence trap, now in the positive control and the
  endpoint metric:

  - **F1 (fatal)**: the positive control is mis-specified. The CE3 make-carry
    head computes `Dn+D'n≥10` (digit-`n` only) and is carry-in-*inert*
    (U-flip 0.00), so for a definite digit its activation is identical across
    opposite-`carry_in` sources → it will **not** corrupt → it is combiner-*like*
    under this metric, not a conduit. So "positive control (1)" fails, which
    trips the study's own `invalid` condition. No known conduit exists to serve
    as the positive reference.
  - **F2 (fatal)**: corruption is read at the endpoint `A_{n+1}` *through the
    whole downstream path*. Because the model is correct on definite digits
    regardless of `carry_in`, a downstream combiner can absorb an upstream
    conduit's wrong relayed carry → "low corruption" can reflect *downstream*
    recompute, not that the *patched* node combines. Requirement (ii) (transmits
    U-flip) doesn't separate them — conduits transmit the U-flip too.
  - **F3**: the lower-operand-identity confound isn't closed, and the same-state
    null is specified two incompatible ways ("≤ null floor" vs "subtract it").
  - **F4**: by the study's own `invalid` condition (positive control must
    separate references), the assay is invalid as written (make-carry and SA
    references both ~0 corruption for a definite digit → don't separate).

  Non-blocking: F5 (L0→L1 path-patch hand-wavy — no metric/null), F6 (state a
  minimal net-effect), F7 (report what the patched activation encodes).

  **Core prescription (skeptic)**: stop reading the endpoint `A_{n+1}`; read a
  **node-attributable** effect (does the candidate's *own activation* differ
  under an opposite `carry_in` for a definite digit?).

  *Status: RESOLVED 2026-07-15 by the working thread — redesigned around a
  node-level activation-invariance discriminator (A-1) with a properly-planted
  conduit reference (A-2), fixed the null (A-3), and demoted the path-patch
  (A-4). Goalposts unchanged (still combiner vs conduit). Gate 1 PASSED on the
  amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede the conflicting pre-run text where noted.

**2026-07-15 — A-1 (resolves F1/F2, the core fix): NODE-LEVEL activation
invariance is the primary discriminator, not endpoint corruption.** The correct
carry-out of a definite digit is independent of `carry_in`. So a **combiner**
(which outputs `carry_out(n)`) has an activation that is **invariant** to a
`carry_in` toggle when digit `n` is definite; a **conduit** (which relays
`carry_in`) has an activation that **differs**. This is measured *at the node*,
so a downstream recompute cannot mask it (F2). Procedure, per candidate node and
definite class (lo/hi):
- Build many pairs identical at digit `n` (same definite operands) but with
  **opposite lower carry** (digit `n-1` sum ≥10 vs <10). Record the node's
  activation vector in each.
- **Activation-difference statistic**: mean cosine distance (and normalized L2)
  between the node's activation under `carry_in=1` vs `carry_in=0`, for the
  definite digit. Compare to the **U-regime** activation-difference at the same
  node (where a combiner *must* differ, since U depends on `carry_in`).
- **Combiner signature**: activation **invariant** in the definite regime
  (diff ≤ null) but **variant** in the U regime (diff ≥ bar) — i.e. the node's
  output depends on `carry_in` *only when digit `n` is U*. That is exactly
  "reads digit-`n`'s class AND the carry, and combines".
- **Conduit signature**: activation **variant in both** regimes (it relays
  `carry_in` regardless of digit-`n`'s class).
- **Inert signature**: invariant in both (doesn't encode the carry at all).
- The endpoint `A_{n+1}` corruption test is retained as *secondary
  confirmation* only, not the primary verdict.

**2026-07-15 — A-2 (resolves F1/F4): planted conduit + inert positive controls
for the activation metric.** (1) **Conduit reference**: take the *residual
stream at the digit-`n-1`/lower-carry position* (which by construction carries
the resolved lower carry) — its activation MUST differ under the `carry_in`
toggle in *both* regimes (definite and U). This proves the activation-difference
metric registers a carry-dependent signal. (2) **Inert reference**: a digit-`n`
operand-fetch (`SA`) head's activation at a definite digit should be invariant
to the lower `carry_in` (it reads digit-`n` operands only) — proving the metric
reports ~0 when there is no carry dependence. If these two do not separate
(conduit-ref diff ≫ inert-ref diff), the assay is `invalid`. The make-carry head
is NOT used as the positive control (it is carry-in-inert; per F1 it would read
as combiner-like/inert — that non-corruption is itself a recorded datapoint,
not the control).

**2026-07-15 — A-3 (resolves F3): single net rule + matched marginals.** The
endpoint corruption secondary uses **net** corruption
`corr_opposite − corr_samestate` with matched lower-operand marginals; the
primary activation metric uses the definite-vs-U activation-difference contrast
(A-1), which is inherently confound-robust because digit `n`'s operands are held
*identical* within each pair (only the lower carry toggles), so lower-operand
identity is matched by construction within a pair. Per-candidate lower-operand
attention mass is reported.

**2026-07-15 — A-4 (F5): path-patch demoted.** The L0→L1 hand-off is an
exploratory secondary (reported descriptively), removed from the success
condition. A proper direct-effect edge patch is deferred to a follow-up.

**2026-07-15 — A-5 (F6): minimal effect.** Combiner requires definite-regime
activation diff ≤ 0.5× the U-regime diff at the same node AND U-regime diff
≥ the inert-reference diff + a margin; conduit requires definite-regime diff
≥ 0.5× U-regime diff. ≥ 60 pairs per (node, class, regime). Bars finalized
against the two A-2 references before reading candidate verdicts.

**2026-07-15 — A-6 (scope/self-suff): `d_model=510, d_head=170, n_ctx=19`
(5-digit); hooks `blocks.{L}.attn.hook_z`, `blocks.{L}.hook_mlp_out`. 2-layer
addition-only; single-digit U. Candidate set = CE4 registry transmitters +
CE3 make-carry heads (as datapoints).**

- **Decision impact**:
  - **Combiner found**: scores A6 (genuine combined/carried state, not
    transport), A2 (MLP does the nonlinear combination), gives entry 2 a
    confirmed tri-state anchor; adds a claim-evidence entry naming the combiner.
  - **All conduits / distributed**: A6's "carried state" reframed as transport
    + distributed combine; A3's single-locus premise weakened; redirects entry 2
    to a distributed-readout analysis.

- **Risks / confounds**:
  - **The corruption control still injects lower-digit info** (source/target
    differ in lower digits). Mitigation: patch ONLY the candidate node's
    activation (not resid), and use the same-state null; a node that corrupts
    only because of injected lower-digit *operands* (not the carry bit) would
    also corrupt the same-state null — subtract it.
  - **Readout trivially corrupts**: the `A_{n+1}` readout position corrupts by
    construction; exclude it as the resolver by position (it is downstream).
  - **Definite-digit answer must be unambiguous**: construct so the target's
    `A_{n+1}` is well-defined and the clean model predicts it correctly (verify
    per pair).
  - **A combiner might still corrupt if the patch overwrites the sum-class flag
    too**: patch the node output only; if the node encodes both the flag and the
    carry, patching swaps both — report this interpretation limit.
  - 2-layer, addition-only scope stated up front; cross-model deferred (B5).

- **Expected artifacts**:
  - Standalone CPU script `scripts/combiner_vs_conduit.py`.
  - Per (model, node, definite class): corruption rate, same-state null, U-flip
    (re-verified), classification; L0→L1 dependency; positive-control reference
    separation.
  - A combiner/conduit registry JSON; a corruption heatmap.
  - Local results folder `results/study-combiner-vs-conduit/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: **The tri-state `U`-combiner is the answer-position
  layer-1 MLP** — cleanly and replicably. Using the node-level activation-
  invariance discriminator (a combiner's output is invariant to `carry_in` for a
  *definite* digit but must vary in the `U` regime), the answer-position L1 MLP
  passes the combiner signature in both models: 5-digit **`P14.L1.MLP`**
  (definite-regime activation diff **0.004**, U-regime **1.18**, ratio 0.00);
  6-digit **`P16.L1.MLP`** (**0.004** / **1.31**, ratio 0.00). By contrast the
  **question-position layer-0 nodes are conduits** (carry the running lower
  carry regardless of digit-`n`'s class — both-regime variation): 6-digit
  `P11.L0.MLP` (0.51/0.59) and head `P11.L0.H2` (0.95/0.90); 5-digit `P10.L0.MLP`
  is **borderline** (0.35/0.73, ratio 0.48 — varies substantially for definite
  digits too, so mixed, not a clean combiner). The CE3 make-carry heads are
  mostly `carry_in`-inert (as expected). The controls separate cleanly (planted
  conduit reference varies in both regimes ~0.5–0.75; inert reference flat 0.00),
  so the discriminator is non-vacuous — unlike the prior interaction gate. This
  **confirms the two-site mechanism**: layer-0 nodes conduct/compute the running
  cascade carry; the **answer-position L1 MLP combines** it with digit-`n`'s
  sum-class to resolve `U`. This is the U-resolver CE4 could not pin down.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/combiner_vs_conduit.py control`
    then `... models`.
  - Script:
    [`scripts/combiner_vs_conduit.py`](../../scripts/combiner_vs_conduit.py)
    (standalone CPU; reuses confirm-ST-node helpers). Env: python 3.13.7,
    macOS-26.5.2-arm64, torch 2.8.0, numpy 2.3.2. Repo commit `368f3a9` (working
    tree). Date 2026-07-15.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000).
  - Artifacts (local; no HF): `results/study-combiner-vs-conduit/`:
    `control_*.json`, `results.json`, `combiner_registry.json`,
    `combiner_vs_conduit.png`.

- **Results**:
  - **Positive controls PASSED and separate** (the fix over the vacuous prior
    gate): planted **conduit reference** (layer-0 resid at the `A_{n+1}` token
    position, which carries the raw lower context) varies in **both** regimes
    (5-digit 0.75/0.75; 6-digit 0.52/0.56); planted **inert reference** (early
    head `P0.L0.H0`) is flat (0.00/0.00) in both. So the metric registers a
    carry-dependent signal and reports ~0 when there is none.
  - **Combiner (U-only variation)**: `P14.L1.MLP` (5-digit) def 0.004 /U 1.18;
    `P16.L1.MLP` (6-digit) def 0.004 /U 1.31 — activation depends on `carry_in`
    *only* when digit `n` is `U`. Clean combiner in both models.
  - **Conduit (both-regime variation)**: 6-digit `P11.L0.MLP` (0.51/0.59),
    `P11.L0.H2` (0.95/0.90) — the CE4 "U-flip transmitters" at layer-0 turn out
    to be **conduits**, not combiners.
  - **Borderline**: 5-digit `P10.L0.MLP` def 0.35 /U 0.73 (ratio 0.48, just under
    the 0.5 combiner boundary); it varies materially for definite digits, so it
    is a mixed/conduit-leaning node, not a clean combiner. Recorded as
    borderline, not claimed as the combiner.
  - **Make-carry heads**: mostly `carry_in`-inert (0.000/0.000), consistent with
    CE3; one units-position head (`P15.L0.H0`, 5-digit) reads as conduit (it sits
    at the last answer position with maximal lower context).

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **Success condition MET**: each candidate is classified by a **non-vacuous**
    control (the conduit and inert references separate cleanly), and a node is
    confirmed the **U-combiner** iff it transmits the U-flip (CE4) AND is
    definite-invariant / U-variant. The answer-position **L1 MLP** meets this in
    **both** models. Verdict: **U-combiner located — the answer-position layer-1
    MLP.**
  - **Two-site mechanism — combiner replicates, L0-conduit half hedged (C3)**:
    the **combiner** (answer-position L1 MLP) replicates cleanly in *both* models.
    The **L0-conduit** companion is clean only in the **6-digit** model
    (`P11.L0.MLP`, `P11.L0.H2` — both-regime variation, no carry_out separation);
    the single 5-digit L0 candidate (`P10.L0.MLP`) is **borderline/mixed** (ratio
    0.48; partial carry_out separation + partial carry_in relay), so the full
    "two-site: L0 conducts, L1 combines" story is fully evidenced only in the
    6-digit model. The `U`-resolution is a genuine *combination at a specific
    MLP* (not mere transport), distinct from the CE3 binary make-carry heads.
  - The node-level metric avoids the endpoint/downstream-absorption confound that
    sank the prior gate: it reads whether the node's *own output* is carry-
    dependent-only-in-U, which a downstream recompute cannot fake.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A6** ("cascade is carried state, consulted to resolve `U`"): **supported
    (refined) on the carried+combined sub-claim; untouched on tie-break-economy.**
    There is a genuine carried carry (layer-0 conduits, clean in 6-digit) that is
    *combined* at the L1 MLP specifically to resolve `U` — carried state consulted
    for the `U` tie-break, matching A6's core. `=`-resolution refined to "resolved
    at the answer-position L1 MLP" (consistent with CE3's `=`-null). A6's
    **tie-break-economy** sub-claim (selective harm to multi-digit `...999`
    cascades, sparing carry-free) is **untested** (single-digit U only).
  - **A2** ("MLP does the nonlinear step"): **supported at the U-combine step
    only** — the `U`-resolution (nonlinear, carry-class-dependent combination) is
    localized to the L1 MLP output, not attention. A2's broader pre-MLP
    sum-sufficiency prediction remains untested (pair-sum freeze).
  - **A3** ("tri-state geometry lives at a locus"): the `U`-combiner locus is now
    **confirmed** (`P14.L1.MLP` / `P16.L1.MLP`) — entry 2 should measure tri-state
    geometry *there*. Scored untouched (geometry not measured here) but anchored.
  - **A5** ("data-dependence in MLPs, not moved attention"): **supported
    (partial)** — the U branch is in the MLP.
  - Others untouched.

- **Skeptic review (post-result)**: Run 2026-07-15, separate thread (docs + JSON
  artifacts + script), with three *additional* discriminating probes the working
  thread did not run. **Verdict: PASS WITH CONDITIONS** — breaks the three-BLOCK
  streak; the combiner claim survives a harder test than run.

  Key findings (all verified by the working thread):
  - **F1 (decisive, positive)**: the node-level metric genuinely escapes the
    prior endpoint vacuity, and the "combiner vs U-detector" alternative is
    **refuted**: the L1-MLP fires *equally hard* for definite and U (norms
    30.7/28.9/31.9 — not silent for definite, so `def_diff≈0.004` is a real
    invariance of a live output), and the output **routes to the class-determined
    `carry_out` centroids** (definite carry_out=0 vs =1 centroids ~orthogonal,
    cos 1.27; U-resolved outputs land on the matching definite centroid, cos
    0.058 / 0.045; cross 1.24). So the L1-MLP output *is* the resolved
    `carry_out` — "combiner" is earned, not overclaimed. (Working thread
    independently reproduced these numbers.)
  - **F2 (positive)**: the conduit reference genuinely relays `carry_in` even for
    a definite digit (cos 0.83); inert reference flat; refs separate → metric
    valid and non-vacuous.
  - **F3 (fix required)**: `P10.L0.MLP` (ratio 0.48) was labeled `combiner` in
    the registry but "borderline" in prose — a hard-0.5-cutoff artifact.
  - **F4 (hedge required)**: the *combiner* (L1 MLP) replicates in both models,
    but the *L0-conduit* half is clean only in the 6-digit model; 5-digit has one
    borderline L0 candidate. "Two-site mechanism" is fully evidenced only in the
    6-digit model.
  - Prediction-scoring corrections: A6 → supported on "carried+combined state
    consulted for U", **untouched** on the tie-break-economy / multi-digit
    sub-claim; A2 → supported *at the U-combine step* only; A5 partial; A3
    untouched (anchored).

  *Status: RESOLVED 2026-07-15 by the working thread — C1 (norm/centroid
  evidence recorded below and verified), C2 (P10.L0.MLP relabeled `borderline`
  in registry + results.json), C3 (two-site/L0 claim hedged). A-3/A-4 secondaries
  were not run (A-3 secondary corruption demoted; A-4 path-patch deferred). Gate
  2 PASSED on the amended read.*

  **C1 evidence (verified):** `P14.L1.MLP` output — norms def/U ≈ 30.7/31.9
  (fires equally, not a U-detector); `cos(carry_out=0 centroid, carry_out=1
  centroid) = 1.27` (class-separated); `cos(U-resolved-to-1, carry_out=1) =
  0.058`, `cos(U-resolved-to-0, carry_out=0) = 0.045` (U output lands on the
  matching `carry_out` centroid). These — not `def_diff≈0.004` alone — are what
  earn "combiner".

- **Limitations**:
  - The combiner metric reads the node's *output activation* carry-dependence;
    it shows the L1 MLP's output is carry-dependent-only-in-`U` (the combiner
    signature), but does not decompose *which* neurons implement it (B2).
  - "Combiner" here means "output = the resolved, class-gated carry"; patching
    swaps the whole MLP output, so if it bundles other info that is included.
  - 2-layer, addition-only, single-digit `U` (not multi-digit `...999`
    cascades). The layer-0 conduit vs the make-carry head distinction is
    position-dependent and not exhaustively mapped.
  - The 5-digit `P10.L0.MLP` borderline case shows the L0/L1 split is not perfectly
    clean at every position.

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (**CE5**
  U-combiner = answer-position L1 MLP; L0 nodes conduits, 6-digit-clean —
  upgrades CE4); synthesis + summary; conjectures (A6 supported-refined +
  untouched-on-tie-break, A2 supported-at-U-combine, A5 partial, A3 anchored);
  agenda (complete entry 1; promote entry 2 with the confirmed `P14/P16.L1.MLP`
  anchor; backlog an L0→L1 path-patch for the two-site hand-off).

- **Next read**: entry 2 (tri-state geometry) now has a **confirmed** anchor:
  measure the `{0,1,U}` *input/latent* geometry at `P14.L1.MLP` / `P16.L1.MLP`
  (distinct from the output carry_out geometry F1 already shows is well-
  separated). A neuron-level decomposition of the L1-MLP combiner (B2) and the
  L0→L1 path-patch (two-site hand-off) are the natural deepenings.
