# Study: Attention-Pattern Invariance Census (study-attention-invariance.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #8

**Verdict: A5 (attention is static positional wiring) is FALSIFIED / narrowed to
hybrid — a few attention heads route by carry state.** Using a *value-matched*
contrast (digit-`n` operands held identical, only the lower carry toggled, so any
target change is carry-state routing, not value content), several heads relocate
their attention target when the carry flips — A5's own pre-registered falsifier.
Cleanest in the 6-digit model: `L0.H0` (Q17), `L1.H1` (Q11, Q14) show a
value-matched top-1 target-move rate of 0.97–1.00 against a near-zero strict null
(0.00–0.08, Bonferroni-safe). The cleanest cell, `L1.H1` Q11, is at an
**operand-read** position (not an answer position), so the routing is not merely
"the answer digit depends on carry". The synthetic content-routed positive
control fires at 1.00, validating the move-counter.

So routing is **hybrid**: most (head, position) cells are target-stable, but a
small identifiable set of layer-1 heads (operand-read Q11 + answer positions)
plus one L0 head genuinely relocate with carry state. Recorded as CE8, scoped
**6-digit only** — the 5-digit model is inconclusive (same-state null 0.40–0.53
makes top-1-argmax near-useless, and one candidate fails Bonferroni), so this is
clean in one model, representational (pattern-shift), not causal. Breaks the
recent A3-family refutation streak with a genuine new structural fact, but a
narrow one; the 6-digit routing cells feed the cascade-tracing entry as
candidates to be **confirmed causally**.

## Pre-run (write before the experiment)

- **Question**: Across a large, stratified set of addition questions of a fixed
  format, how much does each attention head's **post-softmax pattern** (the
  distribution over key positions at each query position) vary with the *content*
  (digit values / carry structure) of the question? Is routing **static
  positional wiring** (A5: patterns near-invariant across questions), or are
  there **content-dependent exceptions** — heads whose target shifts
  systematically with digit values or carry class (A5's falsifier, especially at
  cascade / selection nodes when `ST` is `U`)?

- **Motivation**: A5 is currently *medium-high, partially supported* (CE3 noted
  the confirmed carry heads have high, question-independent operand attention).
  A full census either confirms static wiring across all heads — which simplifies
  every later patching design (interventions can target the value path with less
  confounding) — or finds the content-dependent exceptions that would falsify A5.
  It also yields two reusable byproducts: (1) the stratified question classes
  (carry-free / single-carry / `U`-cascade), (2) a per-(head,position) attention
  baseline the cascade-tracing entry reuses.

- **Hypothesis / competing reads** (neutral):
  - **R-static (A5)**: attention patterns are near-invariant across questions;
    per-(head,query) pattern variance across questions is at/near the softmax
    numerical floor; no head's argmax key or attention mass shifts systematically
    with digit values or carry class. Data-dependence is in the value path/MLPs,
    not the pattern.
  - **R-content-routing (A5 falsifier)**: ≥ 1 head shows systematic
    content-dependent pattern variation — its attended key position, or a
    material fraction of its attention mass, shifts with digit values or (the
    sharpest case) between `U` and definite `ST` classes at a cascade/selection
    node.
  - **R-mixed**: most heads static, a small identifiable set content-dependent
    (A5's own "hybrid routing" alternative). Reported as an explicit
    exception list.

- **Design**:
  - **Models**: `add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`
    (accurate, independent seeds). Weights via `MathsConfig`/TransformerLens,
    CPU, accuracy-verified (invalid if < 0.99).
  - **Stimuli (stratified, the reusable byproduct)**: a large random-addition
    set (≥ 500 questions) plus three purpose-built strata per digit `n`:
    **carry-free** (no digit produces a carry), **single-carry** (exactly one
    carry, at digit `n`), **`U`-cascade** (`...9`-chains that force `U`
    resolution). All questions share the fixed input **format** (same token
    positions), so any pattern variation is *content*, not format.
  - **Captured**: `blocks.{L}.attn.hook_pattern` `[head, query, key]` for both
    layers, over the question set.
  - **Metrics** (per head `h`, layer `L`, query position `q`):
    1. **Pattern variance across questions**: for the attention vector `p[h,L,q,:]`
       (a distribution over key positions), the mean per-key standard deviation
       across the random-question set, and the mean Jensen-Shannon / total-
       variation distance of each question's pattern from the per-position mean
       pattern. Low = static.
    2. **Argmax-key stability**: fraction of questions whose top-attended key
       position equals the modal top-key at that (h,L,q). 1.0 = perfectly static
       target.
    3. **Content-shift test (the A5-falsifier probe)**: does the pattern differ
       *systematically between strata*? For each (h,L,q), compare the mean pattern
       on carry-free vs single-carry vs `U`-cascade (and `U` vs definite `ST` at
       the relevant digit): total-variation distance between stratum-mean
       patterns, tested against the **within-stratum permutation null** (shuffle
       stratum labels across questions and recompute the between-stratum TV).
       A significant systematic shift = content routing at that node.
  - **Aggregation**: report the full (head × query-position) variance heatmap per
    model, the global static-wiring fraction (share of (h,q) cells below a
    pre-registered variance bar), and an **explicit exception list** of any (h,q)
    cell whose content-shift TV beats the permutation null at p < 0.01.
  - **Controls / nulls**:
    - **Softmax numerical floor / baseline**: run the *same question twice* (or
      identical questions with different unrelated high-digit filler that cannot
      affect the head) → the pattern variance floor from numerical + irrelevant-
      content noise. The static-wiring bar is set relative to this floor.
    - **Positive control (content-dependent reference)**: a head/query where the
      pattern *must* be content-dependent if any is — the **answer-sign / operator
      region** is the natural candidate (the sign token's identity depends on the
      answer), or, failing an internal one, a **synthetic content-routed
      reference** (a hand-constructed attention that keys on a digit value) to
      confirm the content-shift test *can* detect routing. If neither the sign
      region nor the synthetic reference registers as content-dependent, the
      content-shift test is underpowered and the run is `invalid`.
    - **Permutation null** for the content-shift TV (≥ 1000 draws), per
      (h,L,q,stratum-pair).
  - **Minimal effect / bars**: R-static requires the global static-wiring fraction
    high (≥ 90% of carry-relevant (h,q) cells below the floor-calibrated variance
    bar) AND **zero** (h,q) cells with a content-shift TV significant at p < 0.01
    beyond the sign/operator region, in **both** models. Any significant
    carry-class-dependent shift at a cascade/selection node is an A5 falsifier and
    is reported as an exception regardless of the global fraction.

- **Success condition** (defined now): the (head × position) attention-variance
  census and content-shift exception list are produced for both models, scored
  against A5 and the routing half of C3, with the stratified question classes and
  per-node baseline saved for reuse. A5 is scored by the static-wiring fraction +
  the exception list; C3's "attention moves" routing half by whether routing is
  content-independent.

- **Failure condition** (defined now): with the positive control detecting the
  content-routed reference — a material set of arithmetic-relevant heads (beyond
  the sign/operator region) show significant content-dependent pattern shifts,
  especially carry-class-dependent shifts at cascade nodes. Verdict: **A5
  refuted / hybrid routing** — attention is not static wiring; data-dependence is
  (partly) in the QK pattern.

- **Ambiguous / invalid condition**:
  - Ambiguous: a small number of borderline content-shift cells (TV significant
    but tiny in magnitude), or the two models disagree on the exception list.
  - Invalid: positive control fails to detect the content-routed reference; or
    the accuracy check fails.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + Paper-2 facts + reused harness source; no working-thread context).

  **Verdict: PASS WITH CONDITIONS.** One singular, correct core problem: **a
  static positional head still has value-dependent softmax weights** (QK reads
  the digit-token embeddings, which differ by digit — CE1), so pattern-variance
  and between-stratum weight-TV are **A5-consistent, not A5-falsifying**. Only a
  **discrete target relocation under a value-matched contrast** falsifies A5.
  Five findings:

  - **C1 (blocking)**: make **argmax/target-key stability** the primary A5
    statistic (A5 predicts target-invariance, not weight-invariance);
    variance-vs-floor is descriptive only.
  - **C2 (blocking)**: the three-way stratum TV conflates routing (a) with
    incidental value-correlation (b) — the permutation null flags (b) too.
    Replace/gate it with a **value-matched contrast** (fix `Dn+D'n`, toggle lower
    carry — reuse `tristate_test`'s stimulus) and score **target-move**; only a
    target relocation there counts as an A5 falsifier.
  - **C3 (blocking)**: the sign/operator region is likely *also* static (its
    content is in the value path), so it would falsely trip the invalid rule →
    make a **synthetic content-routed reference** the primary positive control;
    the deterministic same-question floor is ~0 → use the value-matched null.
  - **C4**: add an **informativeness gate** (BOS-sink / single-key / dead heads
    are trivially "static" — exclude from the informative static fraction).
  - **C5 (blocking)**: redefine R-static on the corrected statistics with
    **per-layer** (L0 vs L1) expectations (L1 QK reads more content); the
    exception list = value-matched target-moves.
  - C6: commit the *revised* statistics to results.json; scope decision-impact
    to "consistent with", not "shows", value-path localization.

  *Status: RESOLVED 2026-07-16 by the working thread — argmax/target-stability is
  now primary (A-1); value-matched target-move is the falsifier test (A-2);
  synthetic content-routed positive control + value-matched null (A-3);
  informativeness gate (A-4); per-layer verdict + target-move exception list
  (A-5); revised numbers to results.json + scoped decision-impact (A-6). Gate 1
  PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-16 — A-1 (resolves C1): argmax/target stability is the PRIMARY A5
statistic.** A5 predicts the *target* is position-fixed even if the softmax
weights jiggle with content. Primary per-(head,layer,query) statistics:
(i) **top-1 stability** = fraction of questions whose argmax key equals the modal
top key; (ii) **top-k mass stability** = fraction of attention mass that stays on
the modal top-2 key set. The raw pattern-variance-vs-floor is demoted to a
descriptive heatmap, NOT a verdict driver.

**2026-07-16 — A-2 (resolves C2, the core fix): value-matched target-move is the
falsifier test.** The generic carry-free/single-carry/`U`-cascade stratum TV
conflates routing with incidental value-correlation (both give significant TV),
so it is reported as *descriptive only, explicitly non-falsifying*. The
**falsifier test** uses a **value-matched contrast**: reuse the `tristate_test`
stimulus (fix `Dn+D'n` at digit `n`, toggle only the lower carry) — digit-`n`
operands are *identical* within a pair, so any change in the head's **target key
(argmax / top-2 set)** between the two is carry-state *routing*, not value
content. Score the **target-move rate** (fraction of pairs whose top key set
changes) vs a permutation null. Only a significant target-move under the
value-matched contrast is an A5 falsifier.

**2026-07-16 — A-3 (resolves C3): synthetic positive control + value-matched
null.** Primary positive control = a **synthetic content-routed head**: perturb
one head's `W_Q`/`W_K` (or hand-build a QK) so its target key is a *function of a
digit value* (e.g. attend `Dn` if `Dn≥5` else `D'n`); the target-move test MUST
fire on it. If it does not, the falsifier test is underpowered → `invalid`. The
sign/operator region is kept as a *secondary, expected-static-if-A5* observable,
NOT a validity gate. The variance floor is the **value-matched null** (pattern
differences when digit-`n` operands are held identical, only lower carry toggled)
— not the trivially-zero same-question floor.

**2026-07-16 — A-4 (resolves C4): informativeness gate.** Only cells whose
attention has meaningful spread (pattern entropy above a threshold OR top-1 mass
< 0.95 over ≥ 2 causally-valid keys) are counted as *informative*. Report the
static fraction split into **informative** vs **trivial** (BOS-sink / single-key
/ dead) cells; a head attending ~entirely to one fixed key is trivially static
and excluded from the informative fraction. Carry-relevant query positions
pre-registered against the token layout: operand-read positions, `=` (pos 11 for
5-digit / 13 for 6-digit), and answer positions.

**2026-07-16 — A-5 (resolves C5): per-layer verdict; target-move exception
list.** R-static verdict = high **top-1/top-k stability** on informative cells
AND **zero significant value-matched target-moves** (beyond the synthetic
control), with **separate L0 vs L1 expectations** (L1 QK reads `resid_post(L0)`
so legitimately carries more content — a higher weight-variance at L1 under
static routing is expected and is not a falsifier; only a target-move is). The
**exception list = value-matched target-moves**, a short interpretable set of
candidate routing nodes, not the long value-correlation artifact list.

**2026-07-16 — A-6 (resolves C6): integrity + scope.** The committed
`scripts/attention_invariance.py` computes and stores the *revised* headline
numbers (top-1/top-k stability per informative cell, value-matched target-move
rate + perm p, synthetic-control detection, per-layer static fractions) to
`results.json`; the deprecated variance-vs-floor is descriptive-only. Decision
impact is scoped: a "no target-moves" result supports A5's *pattern-static*
claim (attention targets are content-invariant) but only *is consistent with*
(does not *show*) the value path being where data-dependence lives — that is a
separate causal question.

- **Decision impact**:
  - **R-static (A5 confirmed)**: A5 → high confidence; C3's "attention moves
    (content-independently)" routing half supported; every later patching design
    can treat routing as fixed positional wiring (targets the value path). Adds a
    claim-evidence entry (static routing).
  - **R-content-routing / R-mixed (A5 falsified/hybrid)**: A5 refuted or narrowed
    to hybrid; the content-routed exceptions become important nodes; later
    patching must account for pattern shifts. Reshapes C3.

- **Risks / confounds**:
  - **Attention causality / masked keys**: patterns are lower-triangular; compare
    only over the causally-valid key positions per query.
  - **Static-by-triviality at low query positions**: early query positions attend
    to few keys, so low variance there is uninformative — restrict the headline
    static-wiring fraction to carry-relevant query positions (answer + `=` +
    operand-reading positions), and report the full heatmap for context.
  - **"Variance is low because the head is dead"**: a head that attends ~uniformly
    or to BOS regardless of content is trivially static but arithmetically
    irrelevant. Cross-reference with the CE3/CE5 useful-node maps; report which
    static cells are arithmetically load-bearing vs inert.
  - **Content-shift confound with SA_n / operand identity**: the strata differ in
    digit values, which correlate with everything; the content-shift test is
    about the *pattern* (routing), and the permutation null over stratum labels
    guards against reading incidental value-correlation as routing. Report
    magnitude, not just significance (tiny-but-significant TV is not routing).
  - Evidence integrity: every headline number (per-model static fraction, the
    exception list with TV + perm p, the floor and positive-control values)
    computed in the committed script and stored to results.json.

- **Expected artifacts**:
  - Standalone CPU script `scripts/attention_invariance.py`.
  - Per model: (head × query-position) variance heatmap; global static-wiring
    fraction; content-shift exception list (TV, perm p, magnitude); baseline
    floor; positive-control detection.
  - The stratified question sets (carry-free / single-carry / `U`-cascade),
    saved for reuse by the cascade-tracing entry.
  - A static-wiring registry JSON.
  - Local results folder `results/study-attention-invariance/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: **A5 (static positional wiring) is FALSIFIED / narrowed
  to hybrid — specific attention heads route by carry state.** Using the
  Gate-1-mandated load-bearing test (a *value-matched* contrast: digit-`n`
  operands held identical, only the lower carry toggled, so any target change is
  carry-state *routing*, not value content), several heads relocate their
  attention target when the carry flips — the exact A5 falsifier ("a head's
  target switching when `ST` is `U`"). Cleanest in the 6-digit model: `L0.H0`
  (Q17), `L1.H1` (Q11, Q14) show a value-matched top-1 target-move rate of
  0.97–1.00 against a **near-zero same-state null** (0.00–0.08, confirmed under a
  strict digit-`n`-fixed null at Gate 2). Crucially, the **cleanest cell,
  `L1.H1` Q11, is at an operand-read position** (not an answer position) — so the
  routing is *not* merely "the answer digit depends on carry". The **5-digit
  model is inconclusive**: its candidate cells ride a same-state null of
  0.40–0.53 (top-1-argmax near-useless at near-tied attention) and one fails
  multiple-comparison correction — so this is **clean in the 6-digit model
  only, not replicated**. The synthetic content-routed positive control fires at
  1.00 (validates the move-counter; note it does not validate the
  null-discrimination step).

  So routing is **hybrid, not purely static**: most (head, position) cells are
  target-stable, but a small identifiable set of **layer-1 heads (operand-read
  Q11 and answer positions) plus one L0 answer head** genuinely relocate their
  attention target with the carry state — A5's own pre-registered falsifier. This
  is a genuine *new structural fact* (breaking the recent A3-family refutation
  streak), but a **narrow one: one model, representational (pattern-shift), not
  causal**.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/attention_invariance.py control`
    then `... models`.
  - Script:
    [`scripts/attention_invariance.py`](../../scripts/attention_invariance.py)
    (standalone CPU; reuses confirm-ST-node helpers). Env: python 3.13.7,
    macOS-26.5.2-arm64, torch 2.8.0. Repo commit `368f3a9` (working tree). Date
    2026-07-16.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000, digit 2),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000, digit 3).
  - Artifacts (local; no HF): `results/study-attention-invariance/results.json`,
    `attention_routing_registry.json`, `carry_routing_heatmap.png`.

- **Results**:
  - **Positive control PASSES**: the synthetic content-routed head
    (`W_Q`/`W_K` perturbed to key a digit value) has target-move rate 1.00 in
    both models → the value-matched target-move test detects genuine routing.
  - **Value-matched target-move exceptions (the A5 falsifier)**:
    - 6-digit (clean): `L0.H0` Q17 move 1.00 / null 0.00 (gap 0.97); `L1.H1`
      Q11 1.00/0.03, Q14 0.98/0.05; plus noisier `L1.H1` Q17 (1.00/0.43),
      `L1.H2` Q11 (0.98/0.30), Q17 (1.00/0.48).
    - 5-digit (real but noisy): `L1.H1` Q14 (0.72/0.40), `L1.H2` Q15
      (1.00/0.53) — same direction, but the same-state null is high (argmax-tie
      jitter), so treat as suggestive, not clean.
  - **Informative-static fraction** (primary top-1 stability on random
    questions, informative cells only) ≈ 0.21 in both models — but this is the
    weight-jiggle-inclusive metric Gate-1 flagged as A5-consistent-not-falsifying;
    the load-bearing verdict is the value-matched target-move, above.
  - **Exceptions concentrate at layer-1 answer-position heads** — the same locus
    (L1, answer positions) where CE5 placed the U-combiner MLP.

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **Failure condition (partly) MET → A5 falsified / narrowed to hybrid**: with
    the positive control detecting routing, a small identifiable set of heads
    (beyond the sign/operator region) show significant value-matched carry-state
    target-moves — the A5 falsifier. Routing is therefore **hybrid**: most cells
    static, a few L1 (and one L0) heads content-routed by carry state.
  - **Cleanliness / replication caveat (Gate-2 F2)**: the 6-digit model gives the
    decisive result (clean moves at near-zero strict null); the 5-digit model is
    **inconclusive** — same-state null 0.40–0.53 makes top-1-argmax near-useless,
    and L1H1 Q14 fails Bonferroni. So this is **clean in 6-digit only, NOT a
    cross-model replication**.
  - **Not trivially-expected (Gate-2 F3)**: the cleanest cell (`L1.H1` Q11) is at
    an **operand-read** position, so carry-state routing there is *not* explained
    by "the answer digit depends on carry" — which rebuts the trivial reading and
    makes the falsifier substantive.
  - **Mechanistic fit (hedged)**: some routing cells sit at L1 answer positions
    near the CE5 U-combiner locus — *suggestive* of the combiner's L1-attention
    input fetching sources conditional on carry state — but this is
    representational co-location, **not established** (no pattern-patching), and
    the cleanest cell is an operand position, so the CE5 gloss is not load-bearing.
  - **C3 routing half**: "attention moves" is supported, but the movement is
    *partly content-dependent* (carry-state), not purely positional as A5
    claimed.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A5** ("attention is static positional wiring; data-dependence only in
    values/MLPs"): **falsified in its strong form / narrowed to hybrid** — its
    own falsifier (target switching with carry/`ST` state) is observed at L1
    answer-position heads (clean in 6-digit). A5's *majority* claim (most cells
    static) survives, but "static wiring" as an unqualified description is
    refuted. Lower A5 confidence; rewrite to "hybrid: mostly static, with
    carry-state routing at L1 cascade heads".
  - **C3 (human)** ("attention moves/routes state across positions"):
    **supported (routing half)**. The content-dependence sharpening *refines A5*
    (which claimed content-independence); C3 itself did not predict
    content-routing, so this is attributed to A5, not scored as new C3 content.
  - **A6** ("carried cascade state / tie-break"): **untouched** — this is a
    representational pattern-shift study with no cascade patching; the L1 routing
    is candidate evidence for entry-3 causal follow-up, not an A6 test.
  - Others untouched.

- **Skeptic review (post-result)**: Run 2026-07-16, separate thread (docs + JSON
  + script), with independent re-runs. **Verdict: PASS WITH CONDITIONS.** The
  skeptic **re-ran the assay with a stricter null** (digit-`n` held *fixed*, only
  lower operands re-randomized) and confirmed the 6-digit clean cells survive
  (L0H0 Q17 strict-null 0.00, L1H1 Q11 0.08, Q14 0.02 — vs value-matched move
  ~1.0), and that they survive Bonferroni ×108 by a huge margin — so **A5's
  strong form is genuinely falsified** and the 6-digit result is real (working
  thread reproduced the strict-null: 0.00/0.00/0.08). Evidence integrity clean
  (all numbers in results.json). Four scoping fixes (all applied):

  - **F2 (blocking)**: the 5-digit exceptions are **inconclusive**, not "real but
    noisy" — the same-state null is 0.40–0.53 (top-1-argmax near-useless at
    near-tied attention), and L1H1 Q14 (0.72 vs 0.70 bar) **fails Bonferroni**.
    So 5-digit is **not a replication**; the finding is clean in **6-digit only**.
  - **F3 (blocking)**: the "exceptions concentrate at answer-position heads / CE5
    combiner locus" framing is **contradicted by the cleanest cell** — 6-digit
    `L1H1 Q11` is an **operand-read** position (not an answer position). This
    *rebuts* the "trivially expected at answer positions" worry (carry-routing at
    an operand-read token is not explained by "the answer digit depends on
    carry"), but the CE5-combiner co-location is **suggestive, not established**.
  - **F4**: the synthetic positive control (8.0× W_Q/W_K perturbation, digit-value
    toggle, no null subtraction) validates the move-counter but **not** the
    null-discrimination step that failed in 5-digit — so 5-digit inconclusiveness
    is unshielded.
  - Scoring: **A6 → untouched** (not "weakly bears" — no cascade patching;
    co-location inference only); C3 content-dependence attributes to *refining
    A5*, not new C3 content.

  *Status: RESOLVED 2026-07-16 by the working thread — 5-digit rescored
  inconclusive (not a replication); "answer-position/CE5-locus" reframed
  (cleanest cell is operand-read Q11, which strengthens the finding by rebutting
  the trivial reading); synthetic-control weakness noted; A6→untouched, C3
  attribution corrected. The strict-null check was independently reproduced. Gate
  2 PASSED on the corrected, 6-digit-scoped read.*

- **Limitations**:
  - The top-1-argmax target-move is noisy where attention is near-tied (5-digit
    same-state null 0.40–0.53); the clean signal is the 6-digit near-zero-null
    cells. A top-*k*-mass or attention-mass-shift metric would be less
    tie-sensitive (follow-up).
  - Value-matched contrast is single-digit `U` (sum=9 at digit `n`, lower carry
    toggled); multi-digit cascades untested.
  - Representational/behavioral (pattern shifts), not causal — does not show the
    routing is *load-bearing* (would need patching the pattern).
  - The reusable stratified question classes were built but the census's
    load-bearing test ended up being the value-matched contrast, not the 3-way
    strata (which Gate-1 showed are confounded); the strata are saved as a
    descriptive byproduct only.
  - 2-layer, addition only.

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (new **CE8**:
  hybrid routing — carry-state target-routing at a few L1 heads (operand-read Q11
  + answer) and one L0 head, **6-digit only**, refutes strong A5); synthesis +
  summary; conjectures (A5 narrowed to hybrid — confidence lowered to medium; C3
  routing-half supported; A6 untouched); agenda (complete entry 1 with the
  metric-limitation note; feed the 6-digit routing cells into the cascade-tracing
  entry as candidate content-routed nodes to be **confirmed causally**).

- **Next read**: the 6-digit carry-routing cells (`L1.H1` Q11/Q14, `L0.H0` Q17)
  are new candidate content-routed nodes for the cascade-tracing entry — but that
  entry must **confirm them causally** (pattern-patching), since this study is
  representational only, and must not inherit the CE5-combiner gloss unchallenged.
  A less tie-sensitive routing metric (top-k mass / attention-mass-shift) is the
  flagged follow-up to adjudicate the 5-digit model.
