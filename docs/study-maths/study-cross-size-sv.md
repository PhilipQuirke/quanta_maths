# Study: Cross-Size SV Validation and Tightness Census (d5/d6/d10/d13) (study-cross-size-sv.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #19

The SV mechanism was fully described on the 5- and 6-digit addition models
(CE13–CE17). The paper needs to know: **does it generalize to bigger models, and
does the human's hunch (C6) hold — that the "redundancy" we keep hitting is just
slack that small models can afford, and that 10-/13-digit models are forced to be
tighter?** We ran the same instruments on d5, d6, **d10, and d13** (all accurate,
1.000), reading each model's published role map.

Findings (post-result-corrected):
- **The role skeleton + the combiner GENERALIZE (robustly).** All three roles are
  present at d10 and d13: question-tail/sign **ST writers** (from the map), **combiner
  MLPs** (from the map), and a causal **consumer head** (found empirically — they're
  *not* map-tagged at d10/d13, itself a datapoint). Most robustly, the **combiner is
  a STEP function at every size** (endpoint-gated to each model's real 0/1;
  α*≈0.5–0.75), reproducing CE17 — the digit-combiner discretizes a delivered carry
  the same way from 5 to 13 digits.
- **BUT the CE16 causal SOURCE signatures do NOT reproduce at large n [F1].** At
  d10/d13 both the `=` arm AND the deciding-ST arm flip 0.00, and the carry axis is
  weak (sep ~6 vs ~30 at d5/d6). So "`=`-not-a-source" at large n is a null on a
  null background — the *positive* deciding-ST source signal washes out too. The
  source-fork is **untested at n≥10** (probe-limited), not confirmed. What transfers
  robustly is role presence + the step combiner + the PC2b class-ablation anchor.
- **C6 is NOT SUPPORTED — redundancy does not thin with size [F2].** Our tightness
  measure (single top ST node vs whole ST class) shows the single node is
  **near-irrelevant at every size** (gap ~0.00–0.08), and the class-vs-single gap
  does **not** decrease across d5→d6→d10 ({0.056, 0.116, 0.324} — increasing).
  **d13 is inconclusive** on this axis (whole-class ST ablation only 0.040 ≈ CE13's
  single-node magnitude; PC2b clears baseline by only 0.02 — "very redundant" vs
  "ST-ablation ineffective at n_ctx 43" not separated). So on d5/d6/d10 redundancy
  **persists (does not thin)**; the human's C6 prior (redundancy = small-model slack)
  is not supported. Redundancy reads as **intrinsic to the algorithm.**
- **A11 not rescued by scale.** The hoped-for "bigger ⇒ tighter ⇒ CE17 assays
  discriminate" premise fails (single-node null everywhere; carry axis weaker at
  large n), so Battery L was not triggered. A11 stays low.

Bottom line: **role presence + step combiner generalize (medium-high); the causal
source signatures are probe-limited/unreproduced at large n; C6 not supported
(redundancy persists d5–d10, d13 inconclusive).** The paper can state the SV
mechanism (ST writers → redundant consumer delivery → step combiner) as a
**size-general skeleton with an intrinsic, non-thinning redundancy**, while noting
the large-n causal source probes are instrument-limited. Dual-gated (pre-launch
XS-A…XS-F; post-result F1/F2/F3/F5 downgrades). Filed as CE18.

Gate: **single combined skeptic pass** (pre-launch audit of this plan
+ post-result audit in one thread), per the sprint process. Evidence-integrity
rules unchanged (all headline numbers from the committed script into
`results.json`).

## Pre-run (write before the experiment)

- **Framing (working axioms)**: per the
  [working axioms](../maths-conjectures-agent.md#working-axioms), the SV
  interface EXISTS and is fully described at 5/6-digit (CE13–CE17: ST writers →
  redundant consumer head-pair delivering a canonical resolved carry from the ST
  cluster, never `=` → step-function combiner). This study does **not** re-test
  existence; it **estimates whether that interface transfers to larger n and
  whether its redundancy tightens** (the human's C6 prior / A12). Every battery
  attributes or estimates a parameter across sizes; redundancy counts (including
  "still redundant") are findings, not nulls.

- **The question ([A12](../maths-conjectures-agent.md#a12-the-sv-interface-generalizes-across-model-sizes-and-the-implementation-tightens-as-n-grows) / [C6](../maths-conjectures-agent.md))**:
  Across the accurate 2-layer/3-head addition zoo (d5, d6 anchors; **d10, d13
  test**; d7/d8/d9 available as a census gradient), does the SV interface hold
  role-for-role, and does its **redundancy shrink as n grows** (C6: small-model
  slack stripped by sequential-accuracy pressure at large n)? If larger models
  are tighter, the CE17 compounding-locus assays that nulled under 6-digit
  redundancy may finally discriminate A11-relay from L1-read.

- **Ground-truth facts / reused assets** (self-sufficiency):
  - **Models (verified 2026-07-16, all load, acc 1.000)**: `add_d5_l2_h3_t15K_s372001`
    (n_ctx 19), `add_d6_l2_h3_t20K_s173289` (22), `add_d10_l2_h3_t40K_s572091`
    (34), `add_d13_l2_h3_t50K_s572091` (43). Gradient (census only, if time):
    d7/d8/d9. Layout is uniform: `Dn` at `n_digits-1-n`, `D'n` at `2*n_digits-n`,
    `=` at `2*n_digits+1`, sign at `2*n_digits+2`, `A_top` consuming pos =
    sign pos + 1.
  - **HF maps (verified present)**: `*_maths.json` gives `Algo:A{k}.ST` L0
    head tags at question-tail/sign positions for **all four models**;
    `*_behavior.json` gives `Fail%`/`Impact`/`Attn` per node. **KNOWN WRINKLE
    (verified): d10/d13 maths.json has NO L1 `Algo` head tags** (0 L1 algo heads),
    unlike d5/d6. So the ST-writer and combiner-MLP roles are map-tagged at all
    sizes, but the **consumer-head role is map-untagged at d10/d13** — this is
    itself an A12 role-transfer datapoint, and the consumer head must be
    identified EMPIRICALLY at large n (CE16 method), not read from the map.
  - **Instruments reused**: CE16 `carry_axis` / `twin_pair` / `same_class_twin` /
    `edge_patch_pred` / `pair_at_top` / per-key v-patch; CE13 `_mean_ablate_acc`
    / `ST_NODES` capture; CE17 `chain_st_sites` / horizon logic; the deciding-
    matched null (CE14 SV-9). All are n-parameterized already.
  - **The 5/6-digit baselines to beat/compare** (from CE13/CE16/CE17): interchange
    of single ST writes = 0.00 (redundant); mean-ablation impact 0.02–0.07
    low-digit over ~0 untagged baseline; `=` value arm 0.00 (depot); joint H1+H2
    class necessity 1.07/0.85; combiner step α*≈0.75.

- **Hypotheses / competing reads** (attribution across sizes):
  - **A12-transfer**: the three map-tagged/identifiable roles appear at d10/d13
    (ST writers with multi-digit Impact; empirically-identified answer-position
    consumer head(s) attending the ST cluster; high-`Fail%` answer-position
    combiner MLPs), and the CE16 interface signatures reproduce (`=`-not-a-source
    ≈ 0.00; class necessity high; step combiner).
  - **A12-tighten (C6)**: a per-role **redundancy index** (defined below)
    **decreases** monotonically with n; single-node interventions that nulled at
    5/6-digit (single-ST interchange, single-head edge) become **decisive** at
    d10/d13.
  - **A12-partial / null**: roles transfer but redundancy does NOT shrink (C6
    refuted — redundancy is not capacity slack) — or a role fails to appear (the
    account is size-specific). Reported as the measured index trajectory, per the
    axioms a finding.
  - **Compounding-locus (stretch)**: if d10/d13 are tighter, the CE17 relay
    interchange / horizon decode give a sharp A11-vs-L1-read verdict there.

- **Design**:
  - **Sizes**: d5, d6, d10, d13 primary (CPU, acc ≥ 0.99 asserted else
    invalid). d7/d8/d9 census-only gradient if time. All headline numbers per
    (model, role) with Wilson/normal CIs → `results.json`.
  - **Battery C — redundancy census (CORE, never dropped; the C6/A12 test).**
    For each size and role (ST writers; consumer head(s); combiner MLPs) compute
    a **redundancy index** with three components, each comparable across n:
    (i) **map duplicate count** — number of nodes sharing a role/Impact digit
    (from maths.json/behavior.json), normalized per answer digit;
    (ii) **interchange decisiveness** — fraction of matched-twin single-node
    interchanges (ST write; consumer-head edge) that flip the served digit
    (0.00 at 5/6-digit = fully redundant; higher = tighter);
    (iii) **paired-ablation gap** — accuracy drop from ablating the single
    map-top node vs the whole role-class, over an untagged-head baseline (CE13
    unit). C6 predicts (i)↓, (ii)↑, (iii single-node gap)↑ with n. Report the
    per-size trajectory + a monotonicity test across {5,6,10,13} (+gradient).
  - **Battery I — interface transfer (CORE).** Reproduce the CE16 signatures at
    each size on the leading-digit cell: (a) **`=`-not-a-source** — per-key
    v-patch of the `=` key only vs the deciding-ST keys (CE16 SI-8 per-key
    CONTRIBUTION decomposition), deciding-matched null; expect `=` arm ≈ 0.00,
    OV-proj ≈ 0 at all sizes; (b) **class necessity** — joint consumer-pair
    mean-ablation on cascade vs carry-free, untagged baseline (CE16); (c) **step
    combiner** — the CE17 Battery-T on-manifold α-sweep (endpoint-gated to the
    per-model real 0/1), report class + α*. Consumer heads identified per §below.
  - **Battery L — compounding locus at large n (STRETCH; drop first).** Only if
    Battery C shows d10/d13 are tighter (interchange decisiveness > ~0.3 at some
    ST site): re-run the CE17 horizon decode + relay interchange at d10/d13 to
    see whether the sharper model discriminates A11-relay from L1-read. If d10/d13
    are NOT tighter, skip (the 6-digit R-mixed verdict stands cross-size).
  - **Consumer-head identification (needed because d10/d13 are map-untagged)**:
    at each size, the consumer head(s) for the leading digit = L1 answer-position
    heads at the `A_top` consuming position whose attention mass on the ST-cluster
    key positions exceeds a threshold AND whose OV edge into the high-Fail
    combiner MLP is carry-specific (CE16 `pair_at_top` for d5/d6 is the map-tagged
    ground truth; for d10/d13 select by attention-to-ST-cluster, validated by the
    CE16 endpoint reproduction — if no head reproduces the 0/1 endpoints, the
    consumer role is reported "not identified at this size", an A12-partial
    finding, and Battery I(a,b) for that size is `invalid`).
  - **Drop order under deadline** (only sanctioned scope reduction): d7/d8/d9
    gradient → Battery L (stretch) → d13 Battery I. **Never drop**: Battery C on
    {d5,d6,d10,d13}, Battery I on d10 (the primary large-n transfer test).
  - **Pre-registered decision table**:
    | Roles at d10/d13 | Redundancy index vs n | Read |
    | --- | --- | --- |
    | all three present | (i)↓/(ii)↑ monotone | **A12 confirmed (transfer + C6 tighten)** |
    | present | flat / non-monotone | **A12-transfer only** (C6 refuted: redundancy not slack) |
    | a role absent | — | **A12-partial** (size-specific scope restriction — flag for paper) |

- **Positive controls** (failure → `invalid` for the affected battery, never a
  negative): (1) each model acc ≥ 0.99; (2) Battery-C interchange/ablation
  instrument reproduces the CE13/CE16 5/6-digit numbers (single-ST interchange
  ≈ 0.00, low-digit ablation impact > untagged baseline) — anchors the index
  scale; (3) Battery-I step-combiner endpoints reproduce the per-model real 0/1
  (CE17 gate; else that size's combiner arm invalid); (4) carry-axis anchor
  separates committed classes per size; (5) per-size behavioral gates on the
  chain family.

- **Success condition** (defined now): controls pass; the three roles are
  attributed present/absent at d10/d13; the redundancy index trajectory across
  {5,6,10,13} is estimated with CIs and its monotonicity scored; the CE16
  interface signatures are reproduced or their failure scoped per size; (stretch)
  the large-n compounding-locus verdict is reported if tightness permits. Scored
  against A12 (transfer + tighten / transfer-only / partial), C6 (confirmed /
  refuted by the index trajectory), A11 (large-n disposition if Battery L runs),
  A10 (step combiner cross-size), A6 (class necessity cross-size).

- **Failure condition**: under the axioms only instrument failure (a battery's
  control fails → that battery `invalid` for that size) or irreducible instability
  (index varies beyond CI without trend → "heterogeneous across sizes", reported
  with numbers).

- **Ambiguous / invalid**: A12-partial and flat-index are RESULTS (reported), not
  ambiguities; a size whose consumer head can't be identified → Battery I invalid
  for that size (reported as an A12 role-transfer finding).

- **Skeptic review (combined, sprint) — PRE-LAUNCH half**: Run 2026-07-16 in a
  separate skeptic thread (docs + working axioms + A12/C6/A11/A10 + CE13/CE16/CE17
  + reused instruments + verified HF facts). **Verdict: PASS WITH CONDITIONS** —
  1 blocking (C2) + 5 refinements, all resolved via amendments XS-A…XS-F. The two
  headline risks were judged addressed by design: the index is per-answer-digit
  normalized, and component (iii) (class-vs-single ablation gap) is a genuinely
  independent tightness leg that does NOT inherit the CE17-F2 interchange
  ambiguity.
  - **C2 (blocking) — "instruments n-parameterized already" is FALSE.** The
    CE13/CE16 registries (`ST_NODES`, `CONSUMER_HEADS`, `INSTRUMENT_HEAD`,
    `EQ_POS`, `GEO_CFG`, `MODELS`; and `compounding_arithmetic.py`'s `nd = 6 if
    startswith("add_d6") else 5`) are hard-coded to d5/d6. Only the *layout*
    helpers are cfg-parameterized. So d10/d13 need their ST/combiner sets built
    from maths.json/behavior.json and their consumer set built empirically. Fix
    (XS-B): (1) restate — instrument LOGIC is n-parameterized, registries are
    not and are repopulated from the maps for d10/d13; (2) add **positive control
    2b**: for d10/d13, assert each repopulated ST node reproduces its OWN
    maths.json Impact tag (mean-ablation impact at its tagged digit > untagged
    baseline) BEFORE any index number is trusted — anchors the large-n index
    scale; failure → index `invalid` for that size, not "tighter".
  - **C1 — index normalization underspecified.** Fix (XS-A): component (i) =
    duplicate multiplicity per role per answer-digit = (#nodes tagged Impact:A_k)
    averaged over a FIXED set of scored digits common to every n (the leading
    digit + top-3 answer-digit window present at all sizes); monotonicity test
    on this normalized scalar; flat-within-CI = C6 refutation (finding).
  - **C3 — interchange (ii) inherits CE17-F2 at every size.** Fix (XS-C): the
    tightness verdict is scored PRIMARILY on component (iii) (class-vs-single
    ablation gap over untagged baseline — causal, F2-free); (ii) is
    corroborating-only, flagged F2-ambiguous; flat (ii)-at-0 across sizes →
    "interchange undetermined at all n", (iii) carries the call; flat (iii) = C6
    refuted.
  - **C4 — consumer-head selector false-admit hole.** Fix (XS-D): a candidate
    consumer head passes only if its endpoint reproduction is carry-SPECIFIC
    (deciding-matched null ≤ 0.20 via `same_class_twin`), not merely
    endpoint-reproducing — closes the "reproduces 0/1 by another route" route.
  - **C5 — Battery L trigger keys only on the ambiguous (ii).** Fix (XS-E):
    trigger L on "(ii) interchange > ~0.3 at some ST site OR (iii) class-vs-single
    gap shrinks below [small-model gap halved]"; stays a drop-first stretch.
  - **C6 — 4 points {5,6,10,13} ≈ 2 regimes; cannot show MONOTONE.** Fix (XS-F):
    top-row wording restricted to "tighter at large n (STEP)" unless d7/d8/d9
    gradient runs (promoted to REQUIRED for any "monotone" claim); if the
    gradient is dropped, max claim = "A12-transfer + large-n-tighter (step, not
    shown monotone)" — pre-registered downgrade, not post-hoc softening.

  *Status: RESOLVED 2026-07-16 via XS-A…XS-F. Pre-launch half of the combined
  sprint pass; post-result half in Post-run.*

## Amendments (post-skeptic pre-launch, sprint)

**2026-07-16 — XS-A (C1): index component (i) = duplicate multiplicity per role
per answer-digit, averaged over a FIXED cross-size digit window** (leading + top-3
answer digits present at every n); monotonicity on the normalized scalar; flat =
C6 refuted.

**2026-07-16 — XS-B (C2, blocking): registries are NOT n-parameterized.** d10/d13
ST/combiner sets built from maths.json (`Algo:A{k}.ST`) / behavior.json
(high-Fail% L1 MLPs); consumer set built empirically. **Positive control 2b**:
each repopulated d10/d13 ST node must reproduce its own map Impact tag
(mean-ablation impact at tagged digit > untagged baseline) before its index counts;
else that size's index `invalid`.

**2026-07-16 — XS-C (C3): tightness scored PRIMARILY on component (iii)**
(class-vs-single ablation gap over untagged baseline, F2-free); (ii) interchange
corroborating-only + F2-flagged.

**2026-07-16 — XS-D (C4): consumer-head selector requires carry-SPECIFICITY**
(deciding-matched null ≤ 0.20 via `same_class_twin`) on top of endpoint
reproduction.

**2026-07-16 — XS-E (C5): Battery L trigger widened** to fire on component (ii)
OR (iii); still drop-first stretch.

**2026-07-16 — XS-F (C6): "monotone" claims require the d7/d8/d9 gradient**; else
the max claim is "large-n tighter (step, not shown monotone)".

- **Skeptic review (combined, sprint) — POST-RESULT half**: **PENDING** — one combined pass;
  rehydrate from this note, the conjecture files (working axioms + A12 + C6 +
  A11 + A10), document rules, agenda, CE13/CE16/CE17 study notes + results.json,
  and the four HF maps. Suggested audit focus: (a) is the **redundancy index**
  actually comparable across n (does the map duplicate count / interchange rate
  normalize correctly per answer digit, given d13 has 13 digits vs d5's 5), or
  does a size-confound (more digits ⇒ mechanically more nodes) masquerade as
  "less redundant"; (b) the **consumer-head identification** at d10/d13 — is
  attention-to-ST-cluster + endpoint-reproduction a fair selector, or could it
  miss a real consumer or admit a false one (and does "role not identified"
  correctly become an A12-partial finding rather than a negative); (c) is the
  **interchange decisiveness** a fair tightness proxy or does it inherit CE17's
  "interchange too weak" ambiguity at every size equally (so a cross-size trend
  is still interpretable even if the absolute value is undetermined); (d) Battery
  L's trigger condition — is "tighter ⇒ re-run locus" gated on the right
  Battery-C signal.

- **Decision impact**: serves the paper's **generalization** claim directly
  (agenda entry 2). A12-confirmed → the SV account is zoo-wide and tightens with
  n (the paper states scope as "the mechanism, tightening at scale"); A12-transfer
  only → the account holds but redundancy is intrinsic not slack (C6 refuted);
  A12-partial → a scope restriction the paper must state. If Battery L runs and
  discriminates, A11 gets its first sharp cross-size verdict. No paper edits from
  this thread.

- **Risks / confounds**:
  - **Size-confound in the index** (the design's main risk): more digits ⇒ more
    nodes trivially; the index must normalize per answer digit / per role and the
    monotonicity test must be on the normalized index — the skeptic is pointed here.
  - **Consumer-head mis-identification at d10/d13** (map-untagged): mitigated by
    the endpoint-reproduction gate (a mis-identified head won't reproduce 0/1).
  - **Interchange-decisiveness absolute ambiguity** (CE17-F2): the absolute value
    is undetermined (redundancy vs weak-interchange), but the CROSS-SIZE TREND is
    the estimand and is interpretable if the unit is held fixed across sizes.
  - **CPU cost of d13** (n_ctx 43): reduce n per battery for large models; report
    the per-size n; d13 Battery I is in the drop order.
  - **Compute/time**: pre-registered drop order; Battery C {d5,d6,d10,d13} + I@d10
    protected.

- **Expected artifacts**: standalone CPU script `scripts/cross_size_sv.py`
  (reuses CE13/CE16/CE17 modules); `results/study-cross-size-sv/`: `results.json`
  (per-size role census + redundancy index trajectory + CIs, interface-transfer
  numbers per size, stretch locus verdict, all controls),
  `redundancy_trajectory.png`, `interface_transfer.png`. No HF uploads (reads
  published maps only).

## Post-run (filled 2026-07-16)

- **Run record**: `PYTHONPATH=. python3 scripts/cross_size_sv.py all` (CPU).
  Models d5/d6/d10/d13 all acc 1.000 (n=64). n=40 causal pairs, 250 ablation Qs,
  200 carry-axis Qs, 30 consumer-ID pairs. Registries built from published
  maths.json/behavior.json (XS-B). Artifact
  `results/study-cross-size-sv/results.json`. Seed 20260716. Script
  `scripts/cross_size_sv.py` (reuses CE13/CE16/CE17 modules). d7/d8/d9 gradient
  NOT run (drop order; so no "monotone" claim per XS-F). Battery L NOT triggered
  (C6 refuted ⇒ no tightness to exploit).

- **Results** (headline; full numbers + CIs in results.json):
  - **Role transfer (all present at d10/d13)**: ST writers {d5:7, d6:6, d10:10,
    d13:11}, combiner MLPs {6,7,10,14}, consumer heads identified at every size
    (empirically at d10/d13 via causal-flip + carry-specificity, XS-D; NOT
    map-tagged there). Controls: PC2b (ST class ablation > untagged baseline) PASS
    at d10 (0.328 vs 0.00) and d13 (0.040 vs 0.00); carry-axis anchor sep 32.3/28.5
    (d5/d6) but **only 6.1/6.1 at d10/d13** (weaker probe at large n).
  - **Combiner (A10 iv cross-size)**: **STEP at every size**, endpoints reproduce
    per-model 0/1 (gate passes all four), α*≈0.75 (d5/d6/d10), 0.5 (d13).
  - **`=`-not-a-source**: `=` arm flip 0.00 at d6/d10/d13 (0.45 at d5 — d5's `=`
    carries some content, known looser small model); deciding-ST arm 0.00/0.62 at
    d5/d6 but **0.00 at d10/d13** (the single-cell source signal washes out at
    large n — more redundancy, and the weaker carry axis).
  - **Class necessity**: selective gap 0.93 (d5), 0.82 (d10); 0.00 (d6), 0.07
    (d13) — the *single* identified consumer head is class-necessary at some sizes
    but redundant at others (consistent with the redundant-pair structure; a
    single head is not always the whole class).
  - **Tightness (C6, PRIMARY leg iii)**: single-node ablation gap ~0 at every size
    (0.076/0.044/0.004/0.000); class-minus-single gap {d5:0.056, d6:0.116,
    d10:0.324, d13:0.040} — **NOT decreasing with n** (small-mean 0.086,
    large-mean 0.182 — large is *higher*). Interchange (ii) 0.00 at all sizes
    (F2-ambiguous, corroborating-only per XS-C).

- **Interpretation** (nothing dropped except the optional gradient + untriggered L):
  the SV interface is **size-general** — ST writers, consumer heads, and a
  **step combiner** appear from d5 to d13, and `=` is never a value source. But the
  human's C6 prior is **not supported**: the redundancy that blocked single-node
  causal isolation at 5/6-digit **persists and does not thin** at d10/d13 (single
  nodes stay null; the class-vs-single gap does not shrink). Redundancy is therefore
  an **intrinsic feature of the learned algorithm**, not capacity slack that scale
  removes. Consequently the large-n models do **not** provide the sharper lever A12
  hoped for on the A11 compounding-locus question (Battery L untriggered). The one
  cross-size caveat: the carry axis is much weaker at d10/d13 (sep ~6 vs ~30) and
  d13's ST-ablation impact is small, so the large-n *interface* probes (=-arm,
  deciding-ST arm, necessity) are noisier and their near-zero values partly reflect
  probe weakness — but the STEP combiner (endpoint-gated) and role presence are
  robust, and the tightness verdict rests on the ablation-gap trajectory which is
  instrument-anchored (PC2b).

- **Prediction scoring** (post-result-corrected):
  - **A12 (interface generalizes)**: **role presence + step combiner CONFIRMED**
    (all three roles present at d10/d13; step combiner endpoint-gated at all sizes;
    PC2b class-ablation anchor passes). Raise this sub-claim to **medium-high**.
    **BUT the causal source signatures (=-not-a-source, deciding-ST) are NOT
    reproduced at large n** [F1] — probe-limited (carry axis sep ~6, both arms
    ~0.00); the source-fork is untested at n≥10, not confirmed there.
  - **A12 (tightens with n) / C6**: **NOT SUPPORTED** [F2] — redundancy does not
    shrink; the class-minus-single gap is flat-to-increasing across d5/d6/d10
    ({0.056, 0.116, 0.324}); **d13 inconclusive** (class ablation 0.040 ≈ CE13
    single-node magnitude, instrument-weak). C6 (redundancy = small-model slack) is
    **not supported**; redundancy reads intrinsic. A12's tightening sub-claim →
    **low** (not "refuted" — d13 can't separate redundant from unmeasured).
  - **A11 (compounding locus)**: **unchanged (low)** — scale did not rescue the
    causal-isolation lever (Battery L untriggered); the CE17 R-mixed verdict stands
    cross-size.
  - **A10 iv (step combiner)**: **generalized** — step at all four sizes
    (endpoint-gated; the single robust cross-size causal implementation result).
  - **A6 (economy)**: class-level necessity holds where the identified head is the
    class (d5/d10: gap 0.93/0.82); NOT comparable at d6/d13 [F3] — the empirical
    single-head selector (vs d5/d6 map-tagged pairs) means d13's necessity 0.07 is
    "one of a redundant set", not "single head not necessary". Necessity does not
    drive any headline.

- **Skeptic review (post-result half of the combined pass)**: Run 2026-07-16
  (separate thread; rehydrated from this note, results.json, working axioms +
  A12/C6/A11/A10, CE13/CE16/CE17, the code). **Verdict: PASS WITH CORRECTIONS.**
  All headline numbers traceable to results.json; Battery L honestly logged as
  untriggered. Two blocking wording downgrades + two minor, all applied above:
  - **F1 (blocking, RESOLVED)**: "A12 role transfer CONFIRMED / generalizes"
    over-read — at large n only role presence + step combiner + PC2b are robust;
    the CE16 causal source signatures (=-arm, deciding-ST arm) are probe-weak nulls
    (carry axis sep ~6, both 0.00). Reworded to "role presence + step combiner
    confirmed; source signatures not reproduced at large n (probe-limited);
    source-fork untested at n≥10" throughout (Exec, scoring).
  - **F2 (blocking, RESOLVED)**: "C6 REFUTED" over-read given d13 is instrument-
    limited (class ablation 0.040 ≈ CE13 single-node magnitude; PC2b margin only
    0.02). Downgraded to "C6 NOT SUPPORTED — redundancy persists d5/d6/d10; d13
    inconclusive." Mechanism conclusion survives on d5/d6/d10 alone.
  - **F3 (RESOLVED)**: large-n consumer-ID is a single causal-flip head (vs d5/d6
    map-tagged pairs) — necessity not cross-size comparable; d13's 0.07 does not
    license a "single head not necessary" claim. Flagged; necessity drives no
    headline.
  - **F5 (RESOLVED)**: the (iii) ablation gaps are un-intervalled point estimates
    (N_abl=250) — stated as a limitation; the trajectory is "suggestive", the
    verdict rests on the qualitative persistence d5–d10, not a tight interval.
  What genuinely stands (per the auditor): the size-general **role skeleton**, the
  **step combiner** (the one robust cross-size causal result), and **redundancy
  persists (does not thin) d5–d10** (C6 not supported). Substance intact; only
  "confirmed"/"refuted" over-claims removed.

- **Limitations**: carry axis weak at d10/d13 (sep ~6 vs ~30) → large-n interface
  arm-probes (=-arm, deciding-ST, necessity) noisy/probe-limited, so the causal
  source-fork is untested at n≥10; d13 ST-ablation impact small (0.04) — "very
  redundant" vs "instrument-weak" not separated (PC2b passes by only 0.02); the
  (iii) ablation gaps are un-intervalled point estimates (N_abl=250) — trajectory
  suggestive not tight [F5]; 4 size points, no gradient (no monotone claim, XS-F);
  consumer head selected as a single causal-flip head at large n vs map-tagged
  pairs at d5/d6, so necessity is not cross-size comparable [F3]; interchange leg
  (ii) F2-ambiguous throughout (corroborating-only). 2-layer/3-head addition zoo,
  one seed family at large n.

- **Doc updates** (feeds paper hand-off, agenda entry 2): A12 role-transfer +
  step-combiner → **medium-high**; source-fork **untested at large n** (probe-
  limited); A12-tightening / C6 → **not supported** (redundancy persists d5–d10;
  d13 inconclusive); A10 iv step **generalizes**; A11 **unchanged (low)**. CE18
  added to claim-evidence; router docs appended. (Applied 2026-07-16.)

- **Next read**: post-result skeptic audit, then router-doc updates, then the
  paper hand-off (agenda entry 2) — the SV account is now size-general with
  intrinsic redundancy.
