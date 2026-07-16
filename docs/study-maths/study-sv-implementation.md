# Study: SV Implementation — Edge Message, Source, Path Shares, Necessity (study-sv-implementation.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

Status: **pre-run written 2026-07-16; SPRINT study** (paper-revision deadline
~2026-07-18). Gate: per the sprint process proposed in
[maths-next-steps.md](../maths-next-steps.md#ranking-logic), a **single combined
skeptic pass** (pre-launch audit of this plan + post-result audit in one Opus
thread) — awaiting the human's sign-off/run; if the human prefers, the standard
two-gate flow applies. Evidence-integrity rules unchanged (all headline numbers
from the committed script into `results.json`).

## Pre-run (write before the experiment)

- **Framing (working axioms)**: this study follows the
  [working axioms](../maths-conjectures-agent.md#working-axioms). The SV
  mechanism **exists** and its wiring is established at medium-high
  ([A10](../maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster)
  consolidation; CE13–CE15). Nothing here is an existence test: every battery
  **estimates a parameter** of the known mechanism, and each estimate — including
  a zero — is a finding provided its power control passes. Redundancy is
  expected; all causal arms run jointly (head pair) as first-class, with
  single-head nulls scored underpowered-uninformative (CE14 SV-1 convention).

- **Question**: Four parameters of the confirmed SV delivery (A10 items i–iv):
  1. **Message** — what does the causal consumer-head→combiner edge
     contribution encode: the *canonical resolved carry* (same code as a
     committed carry, invariant to chain depth and deciding position), or a
     *source-tagged / U-flagged* variant (differs by deciding position or
     chain-ness)?
  2. **Source** — from which attended key position(s) do the consumer heads
     read that message: `=` (carry depot; compounding pre-L1 — the human/paper
     lean, consistent with CE12's all-digit `SV`-at-`=`), the **deciding ST
     site** (would revive A9 selection), or distributed?
  3. **Path shares + class necessity** — how much of the delivery flips via
     the head-pair edge vs the direct `resid_post(L0)` skip (with a *powered*
     direct arm this time), and is the consumer head **class** (H1+H2 jointly)
     necessary on cascade questions?
  4. **(Stretch) Combiner transfer function** — how does the L1-MLP map
     (delivered carry, local sum class) → `carry_out`/digit: step-like
     discretization or pass-through; how neuron-sparse?

- **Motivation**: These four numbers are exactly what upgrades the paper's SV
  section from a wiring diagram (CE13–CE15) to an **implementation
  description**, due in the revision (~2026-07-18). They are also the three
  follow-ups CE14's gate mandated (powered direct arm; paired necessity;
  same-cell selector question — the latter resolved here as a by-product of
  parameter 2) plus the B2-lite combiner form.

- **Ground-truth facts / reused assets** (self-sufficiency):
  - **Cells** (from CE14's registry, `results/study-sv-compounding/`): 6-digit
    `add_d6_l2_h3_t20K_s173289` (primary): consumer pair `P16.L1.H1/H2` →
    combiner `P16.L1.MLP` → digit `A4` (chain top `n=3`); sign-position pair
    `P14.L1.H0/H1` → `A6` (leading digit, `n=5`) as the secondary locus.
    5-digit `add_d5_l2_h3_t15K_s372001` (replication): `P13.L1.H2` /
    `P14.L1.H2` → `A4`/`A3`. `=` is P13 (6-digit) / P11 (5-digit).
  - **Stimuli**: chain family `C(n,k,class)` with matched hi/lo pairs
    (`build_chain` etc., reused); depths k ∈ {1 (anchor), 2, 3}; plus a
    **committed-carry family** (single-digit sum ≥ 10 vs ≤ 8 at digit `n−1`,
    no chain) for the message-invariance comparison. Per-family behavioral
    gate (clean hi/lo separation ≥ 38/40) inherited.
  - **Known axes**: the combiner-input committed-carry axis and centroids
    (CE6: `c0`/`c1` separated, `U→x` lands on `c_x`); the combiner-output
    `carry_out` centroids (CE5). Both recomputed in-script, not hand-copied.
  - **Instrument conventions inherited from CE14**: OV-edge patch through
    `ln2.hook_normalized` with clean-frozen LN std ("lnfair"); joint = head
    pair at the consuming position and same head index across ≥ 2 consuming
    positions; MLP-only routing (skip frozen); deciding-matched null (same
    carry class, different deciding operands) = 0.00 expected; per-head
    per-key value patching via `hook_v[batch, pos, head]`.
  - **CE14 anchor numbers** (regression checks): 6-digit `joint_pair` flip
    1.00 at k=2,3 with `NULL=0.00`; control-3 instrument validity 1.00;
    `direct_full`/`direct_scaled` 0.00 **without** a same-arm power control —
    the gap this study closes.

- **Hypothesis / competing reads per parameter** (estimation, not existence):
  - **Message (M)**: (M-canonical) the edge contribution lies on the one
    committed-carry axis, invariant across depth, deciding position, and
    chain-vs-committed families; (M-tagged) it varies systematically with
    deciding position or chain-ness (a positional/U-flagged code the combiner
    must normalize); (M-mixed) canonical component + reproducible tag.
  - **Source (R)**: (R-equals) the carry-specific effect rides the `=` key's
    value; (R-deciding) it rides the deciding ST-site keys (revives A9);
    (R-distributed) spread across keys with no dominant source; measured as a
    **share vector**, not a binary.
  - **Path (P)**: head-pair share vs direct-skip share of the causal flip,
    both arms powered; (P-heads) pair ≈ 1, skip ≈ 0; (P-skip) reverse;
    (P-split) both material. Class necessity: paired H1+H2 ablation degrades
    cascade accuracy ≫ untagged-pair baseline (a number, with CI).
  - **Combiner (F, stretch)**: (F-step) thresholded discretization along the
    delivered-carry axis; (F-linear) pass-through; neuron top-k share.

- **Design**:
  - **Models**: 6-digit primary, 5-digit replication (CPU, acc ≥ 0.99 else
    invalid). All headline numbers per (model, cell, depth) with binomial CIs;
    n ≥ 40 pairs per causal cell (n ≥ 200 questions per decode).
  - **Battery M — message identity.** Collect the consumer pair's edge
    contribution at the combiner input over {chain k=2, chain k=3 (differing
    deciding positions), committed-carry} × {carry=1, carry=0}. Statistics:
    (i) projection onto the committed-carry axis (per family: does carry=1
    land on the `c1` side at the same coordinate across families?);
    (ii) cross-family transfer decode (train carry probe on committed edges,
    test on chain edges — transfer ≈ within-family acc → M-canonical);
    (iii) residual-variance test: after removing the carry-axis component,
    does deciding-position/family remain decodable from the edge (→ tag)?
    **Subtlety stated up-front**: on chain stimuli the compound carry *equals*
    the deciding digit's make-carry bit, so M cannot be settled by value —
    only by **format invariance** across families and deciding positions;
    that is what (i)–(iii) measure.
  - **Battery R — source attribution (causal).** With attention patterns
    intact, patch the consumer head's **value input per key group** from the
    matched twin: (arm 1) `=` key only; (arm 2) deciding ST-site keys only
    (`Dd`, `D'd` positions and, 6-digit, the sign-token ST writes when they
    serve the cell); (arm 3) all remaining keys. Read the answer-digit flip
    per arm; deciding-matched null per arm (expected 0.00 — carry-specificity
    of the *source*, the same-carry twin differs in operand values, so a flip
    on the null means operand content, not carry). Report the **source share
    vector** per (cell, depth). Also the descriptive per-key OV contribution
    decomposition (`attn[key]·v[key]@W_O` projected on the carry axis) as the
    corroborating estimate. A9 disposition: R-deciding at ≥ 2 depths on the
    same cell revives selection; R-equals retires it toward the depot reading.
  - **Battery P — path shares + class necessity.** (i) **Powered direct arm**:
    before measuring, inject a *synthetic* carry-flip along the CE6 carry axis
    into the skip (`resid_post(L0)` at the consuming position, heads' inputs
    frozen) at the magnitude of a real carry difference — this **must** flip
    (power control for the arm; if it cannot, the arm is `invalid`, and no
    skip-share claim is made). Then the real twin-patch on the skip (heads
    frozen) and on the head pair (skip frozen), same pairs, same depths →
    share estimates that are interpretable *because* the arm is powered.
    (ii) **Class necessity**: mean-ablate H1+H2 jointly (and the sign-position
    pair at the leading-digit cell) on cascade vs carry-free families, against
    an untagged-pair baseline with the CE14 SV-6 absolute floor → necessity
    number + selective-economy confirmation at class level.
  - **Battery F (stretch, drop first under time pressure)** — sweep the
    delivered carry along the axis at α ∈ {−2,…,+2}× at the combiner input
    (local class fixed committed-lo), read `carry_out` projection and the
    answer digit → transfer-function class; top-k neuron share of the effect.
  - **Drop order under the deadline**: F, then M's committed-family arm
    (retain chain-only invariance), then the 5-digit replication (6-digit
    primary verdicts stand model-scoped). Battery R and P(i) are the core and
    are not dropped.

- **Positive controls** (failure → `invalid` for the affected battery, never a
  negative): (1) regression — reproduce CE14's joint-pair flip 1.00 / null
  0.00 at 6-digit k=3 under this script; (2) v-patch instrument — patching
  **all** keys of the consumer pair must reproduce the full edge effect
  (arm-sum sanity: source arms must roughly compose); (3) the synthetic
  carry-injection power control for the skip arm (P-i above); (4) axis
  anchors — the in-script CE6 carry axis separates committed classes (known
  positive); (5) per-family behavioral gates.

- **Success condition** (defined now): controls pass and the four parameters
  are **estimated with stated uncertainty** in the primary model (replication
  or explicit model-scoping): message identity (canonical/tagged/mixed with
  the invariance statistics), source share vector (with the causal per-arm
  flips and nulls), path shares from powered arms + the class-necessity
  number, and (stretch) the combiner transfer class. That estimate set *is*
  the paper's implementation description; A10 items i–iv updated; A9 revived
  or retired; A6 economy raised to class-level.

- **Failure condition** (defined now): under the axioms, only two genuine
  failures exist — (a) instrument failure (a battery's own power control
  fails → that battery `invalid`); (b) irreducible mixing (source or path
  shares unstable across depths/pairs beyond CI, no consistent estimate) →
  reported as "implementation heterogeneous at this granularity", itself a
  paper-grade statement with the numbers shown.

- **Ambiguous / invalid condition**: shares mid-range with overlapping CIs →
  report the split as the estimate (that is an answer, not an ambiguity);
  models disagree → model-scoped; accuracy gate or controls fail → invalid
  for the affected battery only.

- **Skeptic review (combined, sprint)**: **PENDING** — one combined pass in the
  human's separate Opus thread, auditing this plan now and the
  interpretation after results, rehydrating only from: this note, the two
  conjecture files (incl. working axioms), document rules, glossary, agenda,
  claim-evidence (CE5/CE6/CE13–CE15), the CE14 study note + its
  `results.json`/registry, and the HF `behaviors.json` maps. Blocking
  concerns resolved or explicitly overruled here before launch. Suggested
  audit focus: (a) does the M-battery invariance logic really separate
  canonical-vs-tagged given compound-carry ≡ deciding-bit on chains; (b) is
  the synthetic-injection power control for the skip arm a fair power match
  (magnitude calibration); (c) do the R-battery arms compose (leakage between
  key groups via LN).

- **Decision impact**: feeds agenda entry 2 (paper hand-off) directly — the
  four estimates slot into the SV-mechanism section; R settles A9 (revive or
  retire); P settles A10 item iii ("not excluded"/"not necessary" replaced by
  shares + a class-necessity number); M settles what the combiner receives
  (and whether `=` is a genuine carry depot, updating A6's carried-state
  reading and C3's propagation clause toward whichever side wins); F seeds B2.
  No paper edits from this thread (paper thread owns them).

- **Risks / confounds**:
  - **Compound ≡ deciding-bit on chains** (the central subtlety): handled by
    format-invariance design (M-i/ii/iii) and stated in-note; no value-level
    claim is made.
  - **Key-group leakage**: v-patches change one key group's values while LN
    and attention renormalize nothing (pattern intact, values additive) — but
    `W_O` mixing means arms may not compose exactly; the arm-sum control (2)
    quantifies leakage; shares reported net of it.
  - **Operand-content vs carry at deciding keys**: the deciding-matched null
    per arm (same carry, different operands) isolates carry-specificity —
    inherited CE14 convention.
  - **Skip-arm fairness**: the synthetic injection is calibrated to the
    magnitude of a real carry difference at that site (measured, not
    assumed); both under- and over-powering are reported.
  - **Redundancy**: pair-level arms are primary (SV-1); single-head numbers
    reported but never verdict-driving.
  - **Sign-position cell** may behave differently (CE15 noted economy
    uninformative at the sign bottleneck) — leading-digit cell reported
    separately, not pooled.
  - **Time**: pre-registered drop order (F → M-committed arm → 5-digit
    replication) so deadline pressure cannot silently move goalposts.

- **Expected artifacts**: standalone CPU script `scripts/sv_implementation.py`
  (reuses the CE14 harness modules); `results/study-sv-implementation/`:
  `results.json` (every headline number: per-cell/per-depth flips + nulls +
  CIs, source share vectors, path shares + power-control results, necessity
  numbers, M-invariance statistics, F sweep), `control_*.json`,
  `message_geometry.png`, `source_shares.png`, `path_shares.png`. No HF
  uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: *(pending)*
- **Run record**: *(pending)*
- **Results**: *(pending)*
- **Interpretation**: *(pending — read against the pre-stated conditions; the
  drop-order list is the only sanctioned scope reduction)*
- **Prediction scoring**: *(pending — A10 items i–iv, A9, A6, A5)*
- **Skeptic review (post-result half of the combined pass)**: *(pending)*
- **Limitations**: *(pending)*
- **Doc updates**: *(pending — feeds agenda entry 2, the paper hand-off)*
- **Next read**: *(pending)*
