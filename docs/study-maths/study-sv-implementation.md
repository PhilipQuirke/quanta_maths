# Study: SV Implementation — Edge Message, Source, Path Shares, Necessity (study-sv-implementation.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #17

We already know *that* the model adds by fetching a carry from the question-tail
and combining it at each answer position (the "SV mechanism", established over the
previous studies). This study doesn't re-test that it exists — it measures four
concrete **details** of how it's built, treating each as a number to estimate (even
a zero counts, as long as a power-check says the measurement could have moved).

The four questions and what we found (both models):
- **What message travels on the wire?** A **clean, resolved carry** — the same code
  regardless of which digit is deciding it (a "carry probe" trained at one depth
  reads the carry perfectly at another). Alongside it, the deciding *position* is
  also readable off the wire, but only as passive information (we did not show the
  combiner uses it).
- **Where does the carry come from?** From the **question-tail tri-state (ST)
  sites, spread across several of them — and never from the `=` token.** Patching
  the `=` key changes nothing (it's a pass-through *depot*, not a value source).
  Among the named sites the deciding-digit's ST cluster is the carry-specific
  source and dominates at one depth; the other chain sites carry it elsewhere.
- **Which path carries it, and is it needed?** The **attention-head pair is the
  real carrier** (patching it flips the answer every time); the alternative
  "skip"/residual route carries a **negligible** amount of carry — when we inject a
  carry of the size the skip actually carries, nothing moves, so the model isn't
  using the skip (this doesn't prove the skip *couldn't* carry a bigger signal). And
  the head pair is **necessary as a class**: knock both heads out and cascade sums
  break while carry-free sums are fine.
- **How does the combiner turn the carry into a digit?** **Unanswered** — that
  probe (Battery F) didn't work (its knob produced no change), so we honestly mark
  it invalid rather than a result. This is the one remaining open detail.

Method note: this was a **single combined skeptic pass** (deadline sprint). The
pre-launch audit caught two subtle "measuring in the wrong coordinate space"
traps; a quick smoke run caught two more implementation bugs; and the post-result
audit made us soften two claims ("skip excluded" → "skip carries ≈0"; "the source
is the deciding digit" → "the source is the ST cluster, spread out"). Net:
**A10's implementation items i–iii are now described; item iv (the combiner's
formula) stays open.** Filed as CE16.

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

- **Skeptic review (combined, sprint) — PRE-LAUNCH half**: Run 2026-07-16 in a
  separate skeptic thread (docs + working axioms + CE5/CE6/CE13–CE15 + the CE14
  harness/registry + HF maps, verified). **Verdict: PASS WITH CONDITIONS** — 4
  blocking, all coordinate-space / power-control coherence issues (calibrated to the
  working axioms: estimation not existence; the risk is the sprint framing licensing
  over-claim on the causal arms). Resolved via amendments SI-1…SI-7 below:
  - **C1 (blocking) — P-i skip power control is in the wrong space / under-specified
    magnitude** (the CE14-round-2 trap). The CE6 carry axis lives at the combiner
    *input* (`ln2.hook_normalized`, post-L1-attn/LN); injecting it into
    `resid_post(L0)` (skip origin, pre-L1/LN) tests a direction the real skip may not
    carry. Fix (SI-1): inject along the **measured real-skip carry direction** (twin
    `resid_post(L0)` carry-difference, non-carry variance regressed out), at 1× and
    2× the *measured* real-skip carry norm; report injected/real norm ratio; if the
    real-skip carry component ≈ 0, skip arm = `invalid` (no share); cross-check the
    injected delta lands on CE6 `c1` at the combiner input after propagating L1.
  - **C2 (blocking) — Battery M projects the RAW edge delta onto the CE6 (post-LN)
    axis** — space mismatch; LN std differs across families → false "tag". Fix
    (SI-2): project the **LN-normalized (lnfair)** edge contribution; report per-family
    LN std.
  - **C3 (blocking) — M cross-family transfer confounds tag with domain shift.** Fix
    (SI-3): primary M invariance = **within-chain, cross-deciding-position** transfer
    (k=2↔k=3, holds chain-ness); committed-family cross-transfer is secondary + a
    **domain-shift null** (a carry-*irrelevant* attribute probe); M-tagged only if the
    carry probe fails transfer WHILE the irrelevant-attribute null transfers.
  - **C4 (blocking) — Battery R arms may not compose (W_O/LN mixing).** Fix (SI-4):
    report the arm-sum residual `flip(all) − Σ flip(arm)`; add **leave-one-group-out**
    complement arms; a group's share = bracket [single-group, full−complement];
    if |residual| > 0.15, report **ordinal dominance**, not a normalized share.
  - **C5 (non-blocking) — A9 revive rule over-reads.** Fix (SI-5): R-deciding at ≥ 2
    *genuinely independent* depths (non-adjacent, CE15) → "A9 **source premise**
    supported"; the selection *mechanism* still needs the CE14 same-cell tracking+edge
    bar (unmet). R-equals → retire toward the depot reading.
  - **C6 (non-blocking) — name the correct null.** Fix (SI-6): per-arm null reuses
    `same_class_diff_operand` (CE14 SV-9: same carry class, deciding operand re-drawn,
    chain fillers shared); `NULL ≤ 0.20` per-arm gate else that arm `invalid`.
  - **C7 (non-blocking) — retained-core power gating.** Fix (SI-7): R and P-i are
    retained in *scope* but their headline numbers are gated on their own power
    controls; a retained battery with a failed control reports `invalid`, not a share.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments SI-1…SI-7. This
  is the pre-launch half of the combined sprint pass; the post-result half is below.*

## Amendments (post-skeptic pre-launch, sprint)

**2026-07-16 — SI-1 (C1): skip power control in the skip's own space/magnitude.**
Battery P-i injects along the **measured real-skip carry direction** — the twin
`resid_post(L0)` carry-difference on matched chain pairs, with non-carry variance
regressed out — at 1× and 2× the measured real-skip carry norm (report the ratio).
If the real-skip carry component's norm is ≈ 0, the skip arm is `invalid` and no
skip-share is claimed. Validity cross-check: the injected delta must land on the
CE6 `c1` side at the combiner input after L1 propagation.

**2026-07-16 — SI-2 (C2): Battery M projects the LN-normalized edge.** The edge
contribution is pushed through the clean-frozen-std LN (lnfair) before projecting
onto the CE6 `c1−c0` axis; per-family LN std reported.

**2026-07-16 — SI-3 (C3): M primary invariance is within-chain cross-deciding-
position.** Train the carry probe on k=2 edges, test on k=3 edges (chain-ness held,
deciding position varied) → a transfer drop is a tag. The committed-family
cross-transfer is secondary + a domain-shift null (carry-irrelevant attribute);
M-tagged only if carry-probe transfer fails while the irrelevant-null transfers.

**2026-07-16 — SI-4 (C4): Battery R composition brackets.** Report the arm-sum
residual and leave-one-group-out complements; a group's share is the bracket
[single-group flip, full − complement flip]; |residual| > 0.15 ⇒ report ordinal
dominance, not a normalized share.

**2026-07-16 — SI-5 (C5): A9 disposition reworded.** R-deciding at ≥ 2 non-adjacent
depths on the same cell → **A9 source premise supported** (not the full selection
mechanism, which still needs CE14's unmet same-cell tracking+edge bar); R-equals →
retire toward the `=` depot reading.

**2026-07-16 — SI-6 (C6): null named.** Per-arm deciding-matched null =
`same_class_diff_operand` (CE14 SV-9); `NULL ≤ 0.20` per-arm gate, else `invalid`.

**2026-07-16 — SI-7 (C7): retained-core power gating.** R and P-i headline numbers
are gated on their own power controls (SI-1 skip validity; SI-4 arm-sum tolerance;
SI-6 null gate); a retained battery with a failed control reports `invalid`.

**2026-07-16 — SI-8 (implementation, post-smoke): Battery R uses per-key
CONTRIBUTION decomposition, not value-only.** The fast smoke run showed a value-only
patch (swap twin `v`, keep target attention pattern) under-reads: the full all-keys
v-patch flipped 0.00 at k=3 while PC1's full-`z` swap flipped 1.00 — because CE14's
carry delivery is partly **attention-pattern-borne**, not value-borne. Fix: attribute
via `z = Σ_key pattern·v`; each key-group arm patches that group's FULL contribution
(source pattern AND source v for the group's keys), so all-keys reproduces PC1's
full-`z` swap and the arms sum to the full edge effect (the SI-4 arm-sum control now
bites correctly). Approved by human 2026-07-16.

**2026-07-16 — SI-9 (implementation, post-smoke): Battery M verdict = within-chain
transfer + SCALE-NORMALIZED residual; raw stat(iii) dropped.** The smoke run showed
the raw residual-family-decode (stat iii) returns 1.00 trivially because families
differ ~50× in edge projection scale (chain sep ≈ 29/20 vs committed ≈ 0.6), so it
cannot separate tag from scale; and the committed↔chain transfer is pure domain shift
(domain-shift null transfer = 0.50 = chance, confirming skeptic C3). Fix: decide
M-canonical/tagged on (a) the within-chain cross-deciding-position transfer (holds
chain-ness — the valid SI-3 invariance signal) plus (b) a residual-family test on
**z-scored / norm-matched** edge vectors so scale cannot fake a tag. The committed
cross-transfer is reported DESCRIPTIVELY only, flagged domain-shift-confounded.
Approved by human 2026-07-16.

**2026-07-16 — SI-10 (interpretation, post-smoke): M reported as two separable
facts, not a binary.** The smoke run gave a subtle honest result: within-chain
cross-deciding-position carry transfer = 1.00 (the CARRY message is format-invariant /
canonical) WHILE the deciding POSITION remains decodable from the same edge after
carry-axis removal + norm-matching (a positional co-rider). Collapsing this to
"M-tagged" under-states the canonical carry. Fix: the M parameter reports
**M-canonical-carry + position co-rider** — (1) the carry message is canonical
(transfer stat), (2) a positional co-rider co-exists in the edge (residual stat).
The combiner receives a resolved carry PLUS positional context. Approved by human
2026-07-16.

- **Skeptic review (combined, sprint) — POST-RESULT half**: **PENDING** — one combined pass in the
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

## Post-run (filled 2026-07-16)

- **Executive summary**: Four SV-implementation parameters estimated on both
  models (acc 1.000). Controls PASS: PC1 regression reproduces CE14 (joint-pair
  flip 1.00, deciding-matched null 0.00, 6d/5d k=3); PC4 carry-axis anchor
  separates committed classes (sep 28.6 / 32.3). **M = M-canonical-carry +
  position co-rider** (SI-10):   the carry message is format-INVARIANT
  (within-chain cross-deciding-position transfer 1.00 = within-acc), while the
  deciding-*position* is additionally DECODABLE from the edge after carry-axis
  removal (norm-matched residual family decode 1.00 — existence only, NON-causal;
  whether the combiner uses it is untested [post-result F3]); committed↔chain
  transfer is domain-shift-confounded
  (domain-shift null transfer 0.50) so used descriptively only. **R = source is
  the question-tail ST cluster and is NEVER `=`; deciding-ST is carry-specific at
  all depths, dominant among NAMED sources at 6d k=3** (SI-8 per-key-contribution
  decomposition) [F2 correction]: full all-keys reproduces the effect
  (1.00 6d / 0.93–1.00 5d); the deciding-ST arm is carry-specific everywhere
  (null 0.00) and dominant among named sources at 6d k=3 (0.60), but the `rest`
  catch-all — which CONTAINS the chain-ST sites — carries the mass at k=2/k=4
  (source DISTRIBUTED across ST sites, redundancy-consistent; not localized to
  deciding-ST). The `=` arm flips 0.00 at all 6d depths with carry-axis OV
  projection ≈ 0 — `=` is a depot (CE13), NOT a carry-value source. **P = the
  head-pair (SV) path is the effective carrier; the skip/direct residual carries
  NEGLIGIBLE carry** (SI-1) [F1 correction]: the measured real-skip carry
  magnitude is tiny (0.14 6d / 0.077 5d, ≈200× below the head-pair carry signal
  ~29), the synthetic power injection AT THAT MAGNITUDE at 1× and 2× flips 0.00,
  so the model does NOT route carry through the skip (this does not formally
  EXCLUDE the skip channel — a sufficient-magnitude injection was not tested;
  SI-1 was matched to the skip's own magnitude by design); per SI-1/SI-7 no
  skip-share is claimed; the head-pair real patch flips 1.00. **Class
  necessity** (joint H1+H2 mean-ablate): cascade accuracy collapses (0.00 6d /
  0.15 5d) while carry-free is spared (1.00), against a ~0 untagged-pair
  baseline → necessity-over-baseline 1.07 (6d) / 0.85 (5d) at CLASS level.
  **F (stretch) = INVALID (instrument)**: the combiner α-sweep produced zero
  answer flips at every α (spread 0.00) → Battery-F instrument failed; per the
  positive-control rule it is reported `invalid`, not a transfer class. F was
  the pre-registered drop-first battery, so this does not affect the core
  verdicts.

- **Run record**: `PYTHONPATH=. python3 scripts/sv_implementation.py all`
  (CPU). Both models acc 1.000 (n=64). n=40 causal pairs/cell/depth,
  n=120 M-pairs/family, carry axis n=250. Artifact:
  `results/study-sv-implementation/results.json`. Seed 20260716. HEAD at run:
  `02b17c3` (script uncommitted). Depths: 6d k∈{2,3,4}, 5d k∈{2,3}.

- **Results** (headline; full numbers + Wilson/normal CIs in results.json):
  - **PC1** joint-pair flip 1.00, deciding-matched null 0.00 (both models, k=3).
    **PC4** axis sep 28.55 (6d) / 32.28 (5d) > 0.
  - **M** transfer (within-chain k2↔k3) 1.00 = within-acc 1.00 (both);
    norm-matched residual family decode 1.00 (deciding position DECODABLE from
    the edge — existence only, non-causal [F3]);
    RAW residual 1.00 (scale-confounded, descriptive); committed transfer 1.00
    but domain-shift null 0.50 (confounded).
  - **R 6d**: full {k2 1.00, k3 1.00, k4 1.00}; deciding-ST arm {0.00, 0.60,
    0.07}, null-ST 0.00 throughout; `=` arm 0.00 throughout (OV-proj ≈ 0);
    rest {1.00, 0.28, 0.93}; arm-sum residual {0.00, 0.12, 0.00} (not leaky).
    **R 5d**: full {k2 0.93, k3 1.00}; `=` arm {0.00, 0.50}, rest {0.93, 0.00};
    k3 leaky (residual 0.50) → k3 5d reported as ORDINAL dominance only (SI-4).
  - **P** skip carry-mag 0.14 (6d) / 0.077 (5d); power 1×/2× = 0.00/0.00 →
    skip arm INVALID (SI-1); head-pair real flip 1.00; class
    necessity-over-baseline 1.07 (6d, cascade-abl acc 0.00, carry-free 1.00,
    untagged gap −0.07) / 0.85 (5d, cascade-abl 0.15, carry-free 1.00, untagged
    gap 0.00).
  - **F** spread 0.00 at all α → `invalid (instrument)`.

- **Interpretation** (against the pre-stated conditions; drop-order was the only
  sanctioned scope reduction, and only F fell — via instrument failure, not
  deadline): controls passed and all four parameters were estimated with
  uncertainty in the primary (6d) model and replicated in 5d for M, R-shape, P,
  and necessity. The implementation description for the paper: the consumer head
  pair delivers a **canonical (format-invariant) resolved carry** to the
  combiner, with the deciding **position additionally decodable** from the edge
  (existence only, non-causal — M) [F3]; the carry's causal **source is the
  question-tail ST cluster and is never `=`** — deciding-ST is carry-specific at
  all depths and dominant among named sources at 6d k3, while the chain-ST `rest`
  sites carry the mass at other depths (source DISTRIBUTED across ST sites) — and
  the `=` site is a **depot, not a value source** (R) [F2]; the effective **path
  is the head-pair (SV) route** — the skip/direct residual carries **negligible
  carry** (≈200× below the head-pair signal), so the model does not route carry
  through the skip, though this does not formally EXCLUDE the skip channel (SI-1
  power-matched to the skip's own magnitude; a sufficient-magnitude injection was
  not tested — this closes CE14's direct-arm gap in the *"is it used"* sense, not
  the *"could it be used"* sense) [F1]; and the pair is **necessary at class
  level** (selective cascade collapse over an untagged baseline). Redundancy-
  consistent (working axioms): the pair is class-necessary while no single node
  is; `rest` (chain-ST) shares mass with deciding-ST as expected.

- **Prediction scoring**:
  - **A10 i (map-named answer-position L1 fetch)**: SUPPORTED — consumer pair
    edge is the effective path; skip carries negligible carry [F1].
  - **A10 ii (fetch-and-combine over question-tail ST cluster)**: SUPPORTED —
    source is the ST cluster (carry-specific; deciding-ST dominant at 6d k3,
    chain-ST elsewhere); `=` is a depot [F2].
  - **A10 iii ("direct path not excluded / not necessary")**: RESOLVED (in the
    "is it used" sense) — replaced by shares: the skip carries negligible carry
    (0.14/0.077, ≈200× below the head-pair signal), head-pair path carries 1.00,
    so the model does not route carry through the skip. This STRENGTHENS but does
    not formally exclude the skip channel — a sufficient-magnitude injection was
    not tested (SI-1 matched the skip's own magnitude). Still a genuine advance
    over CE14's fully-underpowered direct arm [F1].
  - **A10 iv (combiner transfer)**: NOT ESTIMATED — Battery F instrument
    invalid; A10 iv remains an open parameter (B2 seed stands).
  - **A9 (selection)**: source premise supported at only 1 independent depth
    (6d k=3 deciding-ST dominant+specific); < 2 independent depths and the
    CE14 same-cell tracking+edge bar remains unmet → **A9 stays retired at the
    selection-mechanism level** (SI-5); the depot reading of `=` is reinforced.
  - **A6 (economy)**: RAISED to class level — joint-pair ablation is selective
    (cascade collapse, carry-free spared) over an untagged baseline.
  - **A5**: unaffected.

- **Skeptic review (post-result half of the combined pass)**: Run 2026-07-16 in
  a separate thread (rehydrated from this note, results.json, working axioms,
  CE5/CE6/CE13–CE15, the code). **Verdict: PASS WITH CORRECTIONS.** All headline
  numbers verified traceable to results.json (only rounding differences); PC1
  reproduces CE14; the SI-6 null (`same_class_twin`, deciding operand re-drawn)
  is the CORRECT null, not the CE14-round-2 carry-toggling trap. Two blocking
  wording over-claims + one soft downgrade, all resolved above:
  - **F1 (blocking, RESOLVED)**: "skip path excluded (powered)" overstated a
    "skip carries ≈0 carry" result — the power injection was matched to the
    skip's OWN tiny magnitude (≈200× below the head-pair signal), so failing to
    flip proves the model does not USE the skip, not that the skip COULDN'T
    carry. Reworded to "skip carries negligible carry; does not formally exclude
    the channel" throughout (Exec, Interpretation, A10 iii).
  - **F2 (blocking, RESOLVED)**: "source is the deciding-ST cluster"
    over-generalized from 6d k3 (the only depth where deciding-ST dominates;
    `rest`/chain-ST dominates at k2/k4). Reworded to "source is the ST cluster,
    distributed; deciding-ST dominant among named sources at 6d k3; never `=`".
  - **F3 (non-blocking, RESOLVED)**: M "position co-rider" downgraded to
    "position decodable from the edge (existence only, non-causal)".
  - F4–F7 PASS: `=`-depot consistent with CE13; null correct; A9 retirement
    fair (1 qualifying depth); F-invalid correctly quarantined (A10 iv OPEN);
    evidence integrity clean. The numbers and analysis stand; corrections were
    interpretive only.

- **Limitations**: F instrument failed (combiner transfer unestimated). R's
  `rest` group is a catch-all (chain-ST sites) not decomposed per-position; the
  named contrast (`=` vs deciding-ST) is clean but "rest" is only bounded. 5d
  k=3 R is leaky (ordinal only). M's position co-rider is a decode existence
  result, not a causal one. Skip-INVALID rests on the measured-direction power
  control (SI-1); a direction mis-estimate would masquerade as underpower
  (mitigated but not eliminated by the 2× arm).

- **Doc updates** (feeds agenda entry 2, the paper hand-off): A10 i/ii
  SUPPORTED, iii RESOLVED in the "is it used" sense (skip carries negligible
  carry; NOT formally excluded), iv OPEN (F invalid); A9 retired at selection
  level; A6 class-level economy. CE16 to add to claim-evidence; append
  results-by-time / summary / synthesis. (Applied 2026-07-16.)

- **Next read**: agenda entry 2 (paper hand-off consolidation / referee
  checkpoint) — the four estimates slot into the SV-mechanism section with the
  F1/F2 corrected wording.
