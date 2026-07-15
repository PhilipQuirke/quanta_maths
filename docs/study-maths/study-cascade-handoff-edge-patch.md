# Study: Deep-Cascade Hand-off — L1-head→Combiner Edge Path-Patch (study-cascade-handoff-edge-patch.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #11

The previous study couldn't tell whether one attention "head" delivers the carry to
the place that finishes the answer digit. Here we built a sharper tool — cutting one
specific internal wire and swapping just that signal — to test it directly.

The result is a small but real **clue, not a proof**: in the 6-digit model, at one
chain depth, a single head (the same one an earlier study had flagged) genuinely
hands the correct carry to the answer step. But for most of the other cases the tool
was too weak to see anything (a single wire's signal gets washed out), so we can't
make the general claim. So the "attention selects and delivers the carry" idea gets
its first genuine supporting crumb, but stays unconfirmed. A review pass caught a
bug in our fairness check that had made us over-claim; fixing it pulled the verdict
back to this modest crumb.

## Pre-run (write before the experiment)

- **Question**: Does a single answer-position L1 attention head's output *causally
  drive* the answer-position L1-MLP combiner ([CE5](../maths-claim-evidence.md))
  along the direct **head→MLP edge**, and does that edge carry the **deciding
  digit's** carry information in a multi-digit `...999` chain? Concretely: if we
  path-patch only the `L1.head_h → L1.MLP` edge at the combiner position (freezing
  every other input to the MLP at the target's clean value) from a source chain
  whose deciding digit is in the opposite carry class, does the combiner's output
  flip the cascade answer digits — and is that flip **specific to the deciding
  digit** (A9 selection→combine) rather than generic?

- **Motivation**: This is the agenda-1 follow-up mandated by CE9. The deciding-digit
  study established the *ingredients* of A9 selection exist but do not converge at
  node/pattern granularity: a causally deciding-selective consumer head (6-digit
  `L1.H0` Q14), a separate deciding-digit-*tracking* head (`L1.H2` Q15), and the
  CE8 routing cell `L1.H1` causally inert to single-head pattern-redirect. A9's
  mechanism specifically claims one head relocates to the deciding digit and *its
  output drives the combiner* (CE5). A single-head pattern-redirect could not test
  the head→combiner **edge** (it perturbs the whole downstream path and confounds
  selectivity with generic disruption). An edge path-patch isolates exactly that
  causal link: it either shows a head whose output carries the deciding bit into
  the combiner (A9's hand-off confirmed) or shows no single head does (A9's
  selection→combine form fails; the carry reaches the combiner distributed across
  heads or via the direct `resid_post(L0)` path).

- **Ground-truth facts the design uses** (self-sufficiency):
  - **Architecture (TransformerLens, 2-layer)**: the L1-MLP reads
    `blocks.1.ln2.hook_normalized = LN(blocks.1.hook_resid_mid)`, where
    `resid_mid = resid_post(L0) + Σ_h (z[h] @ W_O[h])` at each position. The
    combiner's inputs are therefore (i) the **direct path** `resid_post(L0)` and
    (ii) each **L1 head's** contribution `z[h] @ W_O[h]`. The `L1.head_h→MLP` edge
    is head `h`'s additive term in `resid_mid` at the combiner position.
  - **CE5 combiners** (answer-position L1 MLP): 5-digit `P14.L1.MLP` (digit 2),
    6-digit `P16.L1.MLP` (digit 3). Consuming position of answer digit `A_k` is
    `pos(A_k) − 1 = n_ctx − 2 − k` (verified in the deep-cascade study).
  - **CE9 candidate heads** (6-digit): causally deciding-selective consumer
    `L1.H0` Q14; deciding-digit-tracking `L1.H2` Q15; CE8 routing `L1.H1` Q11
    (pattern-inert). All L1 heads are measured — no selection-on-outcome.
  - **Chain stimulus `C(n,k,class)`** and the matched-pair machinery are reused
    verbatim from the deep-cascade study (deciding digit `d=n−k`; chain digits
    `d+1..n` sum-9; class `hi`/`lo`; matched pairs differ only at the deciding
    operand positions). Affected answer digits `A_{d+1}..A_{n+1}`.

- **What an "edge path-patch" is here (design rationale)**: To isolate the
  `L1.head_h → L1.MLP(combiner)` edge, run the **target** question and cache its
  clean activations; run the **source** question and cache `z_source[h]` at the
  combiner position; then re-run the target with a hook that, **at the combiner
  position only**, replaces the MLP's input residual contribution of head `h` with
  the source's while holding everything else (the direct `resid_post(L0)` path and
  all other heads' contributions) at the target's clean value. Implementation:
  hook `blocks.1.hook_resid_mid` at the combiner position and add
  `(z_source[h] − z_target[h]) @ W_O[h]` (equivalently patch the head's `hook_z`
  slice but *only where it feeds the combiner position's resid_mid*). The MLP then
  recomputes from a residual that differs from clean by exactly head `h`'s edge.
  Reading the answer at the combiner's own consuming position keeps the effect
  scoped to that digit. Two-sided (hi↔lo).

- **Hypothesis / competing reads** (neutral; the edge either carries the deciding
  bit, carries a non-selective signal, or carries nothing):
  - **R-edge-selective (A9 selection→combine)**: patching one L1 head's edge into
    the combiner flips the cascade answer, and does so **specifically for the
    deciding digit** — the flip appears when (and only when) the source and target
    differ in the deciding class, tracks the deciding digit across depths, and a
    same-class (deciding-matched) source does not flip. The head→combiner edge is
    the physical selection→combine hand-off.
  - **R-edge-nonselective**: the edge is causal (patching it flips the answer) but
    **not deciding-specific** — it flips for wrong-digit sources too, or a
    same-class null also flips. The head feeds the combiner but does not carry the
    *selected* deciding bit (generic contribution).
  - **R-direct-path**: no single L1 head's edge flips the cascade answer; the
    deciding carry reaches the combiner through the **direct `resid_post(L0)`
    path** (freeze all L1 heads → still flips when the direct path is patched).
    A9's "attention selects the deciding digit" is not the hand-off; the combiner
    reads the resolved/running carry from the residual directly.
  - **R-distributed**: no single head's edge is sufficient, but a **joint** patch
    of ≥2 head edges is — declared only after the joint escalation, else
    `underpowered`.
  - **R-model-scoped / ambiguous**: 5-digit and 6-digit disagree, or effects sit
    between null and bar — reported with explicit scope (CE8/CE9 precedent:
    5-digit may again be inconclusive).

- **Design**:
  - **Models**: 6-digit `add_d6_l2_h3_t20K_s173289` (primary — the model where
    CE8/CE9 signals are clean), 5-digit `add_d5_l2_h3_t15K_s372001` (replication —
    inconclusive under the blunt instrument, re-tested here). Weights via
    `MathsConfig`/TransformerLens, CPU; accuracy-verified (invalid if < 0.99).
  - **Stimulus**: reuse `build_chain`/matched pairs from the deep-cascade study,
    main battery depths (6-digit n=4, k∈{1,2,3,4}; 5-digit n=3, k∈{1,2,3}) and the
    leading-digit battery (consumer = answer-sign position). Per-depth behavioral
    gate (≥ 38/40) reused as the precondition (report all depths).
  - **Battery E1 — per-head edge patch (primary)**: for each L1 head `h`, edge-patch
    `L1.h→combiner` at the combiner position, matched hi↔lo, read the affected-digit
    flip vector + joint-flip. Same-class (deciding-matched, different-filler) null
    per head ≤ 0.10. Deciding-selectivity: also patch the edge from a **wrong-digit**
    source (deciding class toggled at a *non-chain* digit) — R-edge-selective
    requires deciding-source flip ≥ bar AND wrong-source flip ≪ that.
  - **Battery E2 — direct-path edge**: edge-patch the **direct `resid_post(L0)`→
    combiner** contribution (freeze all L1 heads at target, patch only the direct
    residual term at the combiner position) matched hi↔lo. Distinguishes
    R-direct-path (this flips, no single head needed) from R-edge-selective (a head
    edge flips).
  - **Battery E3 — joint head edges**: if no single head clears bar, patch all L1
    heads' edges jointly (still freezing the direct path) before any
    R-distributed/negative verdict (CE9 joint-escalation convention).
  - **Battery E4 — leading-digit locus**: E1/E2 with the consumer = answer-sign
    position (human C4). Reports whether the sign-position combiner has the same
    edge structure.
  - **Cell classification**: the deciding operand positions are excluded from the
    "selectivity" contrast by construction (matched pairs differ only there); the
    edge patch transmits *computed* head output, not token identity, because the
    head's `z` is read at the combiner position from the source's forward pass.
  - **Metric / bars**: flip bar = `max(0.50, control_rate − 0.10)` (deep-cascade
    convention); same-class null ≤ 0.10; deciding-vs-wrong selectivity gap ≥ 0.20
    (both directions); ≥ 40 matched pairs per verdict cell. Verdict requires
    consistency across ≥ 2 passing depths (incl. k ≥ 2) and an explicit per-model
    scope (5-digit may be inconclusive — pre-accepted).
  - **Pre-registered decision table**:
    | Single head edge flips? | Deciding-selective? | Direct-path edge flips? | Read |
    | --- | --- | --- | --- |
    | yes (≥1 head, ≥2 depths) | yes (dec ≫ wrong, null ≤ 0.10) | — | **R-edge-selective (A9 hand-off)** |
    | yes | no (wrong also flips / null high) | — | **R-edge-nonselective** |
    | no single head | — | yes | **R-direct-path** |
    | no single head; joint flips | — | — | **R-distributed** |
    | nothing ≥ bar after joint | — | — | **R-none (underpowered)** |
    | models disagree / mid-band | — | — | **R-ambiguous (scoped)** |
  - **Scope**: 2-layer addition, edge granularity at the combiner position; no
    neuron decomposition; no subtraction/mixed models.

- **Positive controls** (each exercises the actual edge-patch unit; any failure →
  `invalid`, never "negative"):
  1. **Edge-patch validity (the key control)**: edge-patch a head that is *known*
     to feed the answer at the combiner position from a source with a **different
     answer digit** — the answer must move. Concretely: patch the L1 head with the
     largest clean attention to the combiner's own operands, hi↔lo at that digit;
     must flip ≥ bar. Proves an edge patch can causally change the answer under
     this harness (the specific gap CE9's blunt redirect left).
  2. **Direct-path validity**: patching the full `resid_mid` at the combiner
     position (direct + all heads) reproduces the deep-cascade tail flip (~0.75)
     — proves the combiner-position residual is causal and the edge decomposition
     sums correctly (edge patches of direct + all heads ≈ full resid_mid patch).
  3. **Same-class null**: deciding-matched source (same class, different filler)
     edge-patched into target flips ≤ 0.10 — the false-positive floor.
  4. **Per-depth behavioral separation**: clean hi/lo predictions differ on exactly
     the expected digits (≥ 38/40 per depth per model) — reused from deep-cascade.
  5. **Additivity check**: sum of per-head edge patches + direct-path edge patch
     ≈ full resid_mid patch (within tolerance) — validates the edge decomposition
     is faithful (no missing path / double-counting).

- **Success condition** (defined now): controls 1–5 pass; a single L1 head's
  edge-patch flips the cascade answer ≥ bar at ≥ 2 depths (incl. k ≥ 2),
  deciding-selectively (dec ≫ wrong, null ≤ 0.10), in the primary model — selecting
  **R-edge-selective** and confirming A9's selection→combine hand-off. Or the
  direct-path edge flips while no single head does (**R-direct-path**), refuting
  A9's hand-off in favor of a residual-read combiner.

- **Failure condition** (defined now): with controls passing — no single head edge
  and no direct-path edge clears bar even after the joint escalation → **R-none**
  (the combiner-position edge decomposition does not localize the carry hand-off;
  finer methods needed). Or edge patches flip but never deciding-selectively across
  models → **R-edge-nonselective** (A9's *selective* hand-off refuted).

- **Ambiguous / invalid condition**:
  - Ambiguous: single-head edge flips at one depth only, or models disagree, or
    deciding-vs-wrong gap in the mid-band; report scoped.
  - Invalid: control 1 or 2 fails (edge patch can't move the answer / decomposition
    doesn't sum); accuracy < 0.99; behavioral gate fails at k ≥ 2 (re-scope to the
    accuracy-cliff note as in the deep-cascade study).

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  that rehydrated only from version-controlled docs (this note; the deep-cascade
  study incl. both its Gate-2 rounds; document rules; conjectures; claim-evidence
  CE5/CE8/CE9; the reused harness scripts) and independently verified the
  `resid_mid = resid_post(L0) + Σ_h z@W_O + b_O` decomposition, LN, and `W_O` shape.

  **Verdict: BLOCK.** The edge-patch decomposition is factually correct and the
  plan has internalized both prior Gate-2 failure modes, but three blocking design
  flaws would make the result uninterpretable:

  - **C1 (blocking) — single-position combiner patch affects only ONE digit.** A
    combiner sits at one answer position; the causal mask prevents it from changing
    other answer digits (verified: full `resid_mid` patch at pos 16 moves only
    `A_4`, 0.90, others 0.00). So the pre-registered "cascade flip vector /
    joint-flip" metric is structurally pinned at ~0 — reading it either way repeats
    the CE9 error. Fix: make E1 a **per-digit** combiner edge patch, patching each
    affected digit `A_j` at *its own* consuming position `consuming_pos(j)` and
    scoring only `A_j`; drop "joint-flip / cascade answer flips".
  - **C2 (blocking) — severe LN-damping underpower.** Single-head edges move the
    answer only 0.10/0.15/0.33 (heads 0/1/2) vs 0.90 for the full `resid_mid` patch
    and 0.70 for the direct-path edge — so "no single head clears bar → R-direct-
    path" is nearly guaranteed *by the instrument*, not the mechanism (an
    underpowered artifact). Fix: add an LN-fair arm (freeze `ln2` scale to clean)
    and a scaled-direct-path control; forbid an R-direct-path/R-none verdict unless
    a same-magnitude control shows single-head edges are detectable in principle.
  - **C3 (blocking) — the patch measures head→(MLP + direct skip), not head→MLP.**
    `resid_mid` feeds both the MLP *and* the direct residual skip to the logits, so
    a flip could be the linear skip contribution, not the combiner — a
    transports-vs-computes confound. Fix: add an **MLP-only routing arm** (patch so
    the skip stays clean, or decompose the flip into MLP-out delta vs skip delta);
    A9 "hand-off confirmed" requires the flip to survive MLP-only routing.
  - **C4 (non-blocking)** — import the D-9 **irrelevant-token baseline** and report
    per-source edge norms; deciding must beat wrong AND irrelevant by ≥ 0.20.
  - **C5 (non-blocking)** — specify the **same-cell logic**: full A9 form requires
    the edge-carrying head to be the CE9 selective/tracking head (`L1.H0` Q14 /
    `L1.H2` Q15); an edge on a *different* head is a distinct weaker result.
  - **C6 (non-blocking)** — drive E1 off the physical `consuming_pos(j)`, not CE5's
    off-by-one "digit" labels; E4 sign-position patch produces the top digit `A_6`.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments E-1…E-6 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-16 — E-1 (resolves C1): per-digit combiner edge patch.** E1 is
redefined: for each affected digit `A_j` (`j ∈ d+1..n+1`), patch the
`L1.head_h → resid_mid` edge at **that digit's own consuming position**
`consuming_pos(j)` and score only `A_j`'s flip (matched hi↔lo). The output is a
**per-(head, digit) edge-flip profile**, not a one-patch cascade vector. "joint-
flip" is redefined as *the same head index carries the edge at ≥ 2 of the affected
digits' combiners* (a per-position replication), not a single patch flipping all
digits. Decision-table row 1 and the success condition are reworded accordingly:
R-edge-selective requires a head whose edge flips **its own digit** at ≥ 2
consuming positions (incl. one from a k ≥ 2 chain), deciding-selectively.

**2026-07-16 — E-2 (resolves C2): LN-fair arm + power gate; no underpowered
R-direct-path.** Two arms per edge patch: (a) **raw** (patch `resid_mid`
contribution, let `ln2` renormalize) and (b) **LN-fair** (freeze the `ln2` scale
factor at the target's clean value so a single head's additive edge is not damped
by the whole-vector norm). The LN-fair arm is primary for the *power* question.
Add a **scaled-direct-path control**: patch the direct `resid_post(L0)` edge scaled
to a typical single-head edge norm; if that is also sub-bar, the single-edge
battery is **`underpowered`** and *no* R-direct-path or R-none verdict may be
drawn. Pre-registered: R-direct-path requires (direct-path edge flips) AND
(single-head edges are detectable-in-principle by the scaled control).

**2026-07-16 — E-3 (resolves C3): MLP-only routing arm.** For any edge that flips,
decompose the effect: patch the edge and measure (i) the change in
`blocks.1.hook_mlp_out` at the combiner (the head→MLP→logits path) vs (ii) the
change carried by the direct skip (`resid_mid` delta projected past the MLP).
**R-edge-selective (A9 hand-off) requires the flip to be carried by the MLP-out
delta**; a flip that is purely skip-carried is scored **R-direct/linear, not A9**.
Implementation: run an arm that patches the head edge into the MLP input path only
(freeze the skip contribution of `resid_mid` to `resid_post` at clean), and
compare to the raw arm.

**2026-07-16 — E-4 (resolves C4): D-9 selectivity discipline.** The deciding-source
edge flip must exceed **both** the wrong-digit-source flip **and** an
irrelevant-token (`+`/`=`) edge baseline by ≥ 0.20, with per-source edge norms
reported so a norm mismatch cannot masquerade as (non)selectivity. If
deciding ≈ irrelevant → **R-edge-nonselective / underpowered**, never "A9 refuted".

**2026-07-16 — E-5 (resolves C5): same-cell logic pre-registered.** **Full A9
form (R-edge-selective)** requires the edge-carrying head to be the CE9
causally-selective consumer (`L1.H0` Q14, 6-digit) and/or the tracking head
(`L1.H2` Q15). An edge-selective flip on a *different* head is scored as a
distinct, weaker result ("an unflagged head carries the selected bit"), reported
separately and **not** folded into "A9 confirmed".

**2026-07-16 — E-6 (resolves C6): physical consuming positions.** E1 is driven off
`consuming_pos(j)` per affected digit, not CE5's "digit N" labels. Position table
(verified): 6-digit n=4 → A_1@19, A_2@18, A_3@17, A_4@16, A_5@15, A_6@14 (sign
position, leading digit); 5-digit n=3 → A_1@16, A_2@15, A_3@14, A_4@13, A_5@12
(sign). E4 (leading-digit locus) patches at the sign position and reads the top
digit `A_top`.

- **Decision impact**:
  - **R-edge-selective**: A9's selection→combine hand-off **confirmed** (first
    edge-level causal evidence); A6 sequential further disfavored; A5's CE8/CE9
    routing cells gain a causal role at the combiner edge; C3 propagation recast as
    selection→combine. Promotes A9 toward medium-high.
  - **R-direct-path**: A9's *attention*-selection hand-off refuted; the combiner
    reads the carry from the residual directly (closer to a resolved-state read) —
    reshapes A9 and partially rehabilitates a residual-carried reading (A6),
    though still not sequential-accumulated.
  - **R-edge-nonselective / R-distributed / R-none**: A9's specific hand-off not
    supported; the mechanism is distributed or needs neuron-level tools; hold A9,
    escalate to B2 (neuron decomposition of the combiner MLP).

- **Risks / confounds** (with mitigations):
  - **Edge patch not truly isolating the edge**: patching `hook_z` globally also
    changes the head's output at *other* positions. Mitigation: patch the head's
    contribution into `resid_mid` **at the combiner position only** (positional
    hook), and validate with the additivity check (control 5).
  - **LN nonlinearity**: `ln2` normalizes `resid_mid`, so a head-edge change is
    rescaled by the whole residual norm; a small edge may be damped. Mitigation:
    report the raw edge-patch flip and note LN damping; the direct-path arm (E2)
    and the additivity check bound this.
  - **Operand-content leakage**: a head that fetches the deciding *operand* (not a
    computed carry) would flip for the right reason but via content, not selection.
    Mitigation: matched pairs differ only at the deciding operand; the wrong-digit
    selectivity contrast + same-class null separate computed selection from content.
  - **Downstream recompute masking** (CE9 lesson): reading at the combiner's own
    consuming position, with the combiner MLP recomputing from the patched
    residual, is exactly the intended causal path — control 1 proves the edge can
    move the answer, so a null is interpretable.
  - **5-digit near-tie / blunt-instrument residue**: pre-accepted possibly
    inconclusive; model-scoped verdicts allowed (CE8/CE9 precedent).
  - **Multiple comparisons** over head × depth × direction: per-cell nulls + both
    directions + control-tied bar + ≥ 2-depth consistency + two-model replication.

- **Expected artifacts**:
  - Standalone CPU script `scripts/cascade_handoff_edge_patch.py` (reuses the
    deep-cascade `build_chain`/matched-pair machinery and confirm-st-node helpers).
  - `results/study-cascade-handoff-edge-patch/`: `results.json` (every headline
    number — per-head edge flips + nulls + selectivity, direct-path flips, joint
    flips, control results, additivity check, per-depth accuracy); `control_*.json`;
    `edge_flip_registry.json`; plots (per-head edge-flip × depth heatmap,
    deciding-vs-wrong selectivity bars). No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (rewritten post-Gate-2, after the power control was
  corrected): **A scoped, partial positive. The full combiner-position residual is
  causal, and at a single depth a single L1 head (`L1.H1`, the CE8 carry-routing
  cell) carries the top cascade digit's *computed, deciding-selective* carry through
  the combiner MLP input — a real advance over CE9's null. But the single-position
  edge battery is largely *underpowered*, so it cannot broadly discriminate
  attention-delivery from residual-carry, and no head meets the pre-registered
  ≥2-depth bar — so A9's attention-delivery is *circumstantially* supported, not
  confirmed.** With the edge decomposition exact (`resid_mid = resid_post(L0) +
  Σ_h z@W_O + b_O`, additivity 3.3e-6) and control 2 (full `resid_mid` patch)
  flipping the digit 1.00, the per-digit edge patch (corrected per-cell power gate,
  E-7) shows:
  - **6-digit**: 3/9 cells carry via a **single L1 head edge** (k=3 `L1.H1` flips
    A4/A5 at 1.00; k=1 `L1.H2` flips A5 at 0.82); **0/9 via the direct residual
    path**; and **6/9 are underpowered** (a single-head-magnitude edge is not
    detectable at those positions — the corrected power control flips 0.00). So the
    direct-path-vs-attention question is *unresolved at 6/9 cells*, not "attention
    wins 9/9".
  - **Where a head does carry it, the bit is computed and deciding-selective**:
    `L1.H1` k=3 flips 1.00 when the deciding *class* toggles, 0.00 when only the
    deciding operand *value* changes (same-class null) and 0.00 for a wrong-digit
    source — it delivers the *computed* carry, not operand content. This is a clean
    sub-result at that cell.
  - **`L1.H1` is the CE8 carry-routing cell** that CE9's blunt pattern-redirect
    found causally inert; the finer edge path-patch shows its output *does* drive
    the combiner (at k=3) — the CE9→edge-patch escalation was worth it.
  - **No head meets the ≥2-depth bar** (`L1.H1` k=3 only; `L1.H2` k=1 only), so the
    pre-registered R-edge-selective condition (a head carrying at ≥2 depths incl.
    k≥2) is **not met** — this is a one-depth localization, honestly not a confirmed
    universal selector.
  - **5-digit**: mixed — 3/9 head-carried (`L1.H2` k=2,3 computed+selective), **3/9
    via the direct residual path** (direct_full 0.70–0.93), 3/9 underpowered. The
    direct path *does* contribute here, so the two models differ.

  Net: a **modest but real advance over CE9** — the combiner residual is causal, and
  a specific head (`L1.H1`) carries the computed deciding carry at the top digit
  (k=3), reconciling the CE8 representational routing with a causal role. But the
  battery is underpowered at most cells and the effect is one-depth, so **A9's
  attention-delivery is circumstantially supported, not confirmed**, and A6's
  residual-carry is **not refuted** (only unsupported at the powered 6-digit
  top-digit cells; live in 5-digit). Scoped: 2-layer addition, edge granularity,
  two models.

  *(Process note: the first draft over-claimed "attention-delivered, not the
  residual path (0/9); A9 attention-delivery confirmed → medium". Gate 2 [BLOCK]
  found the power control was inverted [scaling the direct path UP ~20×, disabling
  the underpower gate]; corrected, 6/9 6-digit cells are underpowered, so the anti-A6
  "0/9" and the A9 upgrade were withdrawn. This corrected read supersedes it — the
  third consecutive time the skeptic gate caught a real over-reach in this cascade
  line, twice positive, once negative.)*

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/cascade_handoff_edge_patch.py control`
    then `... models`.
  - Script:
    [`scripts/cascade_handoff_edge_patch.py`](../../scripts/cascade_handoff_edge_patch.py)
    (standalone CPU; reuses the deep-cascade `build_chain`/matched-pair machinery
    and confirm-st-node helpers). Env: python 3.13.7, macOS-26.5.2-arm64, torch
    2.8.0. Repo commit `92edb5f` (working tree). Date 2026-07-16. Seed 20260716.
  - Models: `add_d6_l2_h3_t20K_s173289` (acc 1.000, n_top digit 4),
    `add_d5_l2_h3_t15K_s372001` (acc 1.000, n_top digit 3).
  - Artifacts (local; no HF): `results/study-cascade-handoff-edge-patch/results.json`,
    `control_*.json`.

- **Results** (corrected per-cell power gate, E-7):
  - **Controls**: additivity `max_abs_reconstruction_error = 3.3e-6` (edge
    decomposition exact); **control 2 (edge-patch validity)** — full `resid_mid`
    edge patch flips the target digit **1.00** at every affected digit/depth (an
    edge patch can causally move the answer). The pre-registered "control 1"
    (direct-path validity) **diverged and failed** (`direct_full`=0.00 in 6-digit);
    validity is instead established by control 2 and the single-head 1.00 flips
    (F6-post). Behavioral gate: 6-digit {k1 0.925, k2 0.975, k3 0.925, k4 0.875 →
    k4 skipped (<0.90)}; 5-digit {k1 0.95, k2 0.975, k3 0.95 — all ≥ 0.90, none
    skipped}.
  - **6-digit edge flips** (lnfair arm): k=1 `L1.H2` A5 0.82; k=3 `L1.H1` A4 1.00,
    A5 1.00; k=2 all < 0.5. `direct_full` = 0.00 at all 9 cells. **Corrected
    `direct_scaled_percell` = 0.00 at all cells** → single-head-magnitude edges
    are NOT detectable at these positions. Verdict cell-count (E-7): **3/9
    head-carried, 0/9 direct-path, 0/9 neither-powered, 6/9 UNDERPOWERED**.
  - **Selectivity (E-8)**: `L1.H1` k=3 deciding-flip **1.00**, same-class-operand
    null **0.00**, wrong-digit **0.00** → computed + deciding-selective. `L1.H2`
    k=2 dec 0.23 (weak). Other heads/depths ~0.
  - **5-digit edge flips**: k=1 `L1.H2` 0.72 with `direct_full` 0.93; k=2 `L1.H2`
    0.93 with `direct_full` 0.80; k=3 `L1.H2` 1.00 with `direct_full` 0.70.
    Selectivity: `L1.H2` k=2,3 dec 0.93/1.00, null 0.00, wrong 0.00 (computed +
    selective). Verdict cell-count: **3/9 head-carried, 3/9 direct-path, 3/9
    underpowered**.

- **Interpretation** (against pre-stated conditions; goalposts unchanged; corrected
  post-Gate-2):
  - **6-digit: R-attention-partial, but underpowered at most cells.** A single L1
    head carries the top-digit carry (`L1.H1` k=3; `L1.H2` k=1), computed and
    deciding-selective. But **6/9 cells are underpowered** (single-head-magnitude
    edges undetectable), so the direct-path-vs-attention discrimination holds only
    at the powered top-digit cells; the pre-registered R-edge-selective row
    (≥ 2 depths per head) is **not met** (`L1.H1` k=3 only). So A9's attention
    delivery is **circumstantially supported at one depth**, not confirmed.
  - **A6 not refuted**: the "direct path inert 0/9" of the first draft was an
    artifact of the inverted power control; with the corrected gate the 6-digit
    direct-path cells are underpowered (not shown inert), and the 5-digit direct
    path is causally live (0.70–0.93). A6's residual-carry is *unsupported at the
    powered 6-digit top cells*, not refuted.
  - **Computed, deciding-selective at the carrying cells**: where a head does carry
    it (`L1.H1` 6-digit k=3; `L1.H2` 5-digit k=2,3), dec ≈ 1.0, null 0.00, wrong
    0.00 — a clean computed-carry sub-result, ruling out operand-content leakage.
  - **CE9 reconciliation (honest)**: the edge path-patch shows `L1.H1` (the CE8
    cell CE9 found inert) *does* causally drive the combiner at k=3 — a genuine
    resolution of the CE9 null. But the edge-carrier (`L1.H1`/`L1.H2`) is **not**
    CE9's selective (`L1.H0`) or tracking (`L1.H2` was tracking) head in the way A9
    predicts one converged cell — so "selection" and "combine-drive" are still not
    cleanly the same cell; this is a *partial, reconfigured* A9, not a clean
    confirmation.
  - **Models differ**: 5-digit has live direct-path cells; the 6-digit
    attention-leaning reading is model-scoped.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2 passes):
  - **A9** (one-hop selection→combine): **circumstantially supported at one depth,
    NOT confirmed.** A single computed-carry head (`L1.H1` 6-digit k=3; `L1.H2`
    5-digit) drives the combiner for the top digit, but no head meets the ≥2-depth
    bar and most cells are underpowered. **Hold A9 at low-medium** (do not raise);
    the one-depth `L1.H1` edge is the first *causal* crumb but not a confirmation.
  - **A6** (sequential/residual-carried): **not refuted** — the corrected power gate
    shows the 6-digit direct-path cells are underpowered (not inert), and the
    5-digit direct path is causally live. Hold A6; the residual-carry form is
    unsupported only at the powered 6-digit top cells.
  - **A5** (CE8 routing causal?): **partially upgraded** — `L1.H1` (the CE8 cell) is
    causally load-bearing at the combiner edge **at k=3 (one depth)**; the edge
    path-patch resolves CE9's null there, but the one-depth scope keeps this
    tentative. Note A5 (medium) rather than raise.
  - **C3 (human)** (cascade rides the residual stream): **not adjudicated** — the
    direct-path arm is underpowered in 6-digit and live in 5-digit, so the study
    does not cleanly bear on the residual-carry clause. (Human-owned; noted.)

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + control JSONs + script), with independent re-runs.
  **Verdict: BLOCK** (resolved below). The skeptic verified the edge decomposition
  and controls but caught three over-reaches in the exact CE9 directions:
  - **F1-post (BLOCKING, linchpin) — the power control was inverted.** The
    `direct_scaled` control scaled the direct path *up* ~16–23× (I averaged
    `head_norm/direct_norm`, which is > 1), disabling the E-2 underpower gate. Run
    *as E-2 specified* (scale the direct edge *down* to each cell's own head-edge
    norm), **8/9 6-digit cells are underpowered**, not "powered/neither". Fix E-7.
  - **F2-post (BLOCKING) — the anti-A6 "direct path inert 0/9" is not earned.** It
    conflates no-signal cells (tiny direct delta at top digits) with underpowered
    cells; and the 5-digit direct path *does* flip (0.70–0.93), proving the harness
    detects direct-path flips when signal exists. A6 residual-carry is not cleanly
    refuted. Fix: rescore.
  - **F3-post (BLOCKING) — the single-head result fails its own ≥2-depth bar.** The
    3/9 head-carried cells are k=1 (`L1.H2`) and k=3 (`L1.H1`); **no single head
    carries at ≥2 depths** — the pre-registered R-edge-selective condition is not
    met (the CE9 round-2 one-depth trap). Fix: rescore A9 down; require ≥2 depths.
  - **F4-post (non-blocking)** — the E-4 selectivity arms (wrong-digit, irrelevant,
    edge norms) were not implemented. Fix E-8.
  - **F5/F6-post (non-blocking)** — `mlp_only` arm was redundant with `lnfair`
    (both patch `ln2.hook_normalized`, which *is* MLP-input-only, so E-3's substance
    is met by lnfair); the failed pre-registered "control 1" (direct-path validity)
    should be stated as a divergence+failure subsumed by control 2, not just "a
    finding".

  *What Gate 2 confirmed is right: the edge decomposition is exact (additivity
  3.3e-6); control 2 (full resid_mid) flips 1.00 (edge patches can move the answer);
  the `L1.H1` k=3 top-digit flip (1.00, same-class null 0.00, via the MLP input) is
  a real strong MLP-routed effect and a genuine advance over CE9's null; the
  per-cell verdict framework (no blind max) is the right structure.*

  *Status: RESOLVED 2026-07-16 by the working thread via amendments E-7/E-8 and a
  full rescore (below). The power control was corrected (per-cell scale-down),
  confirming 6/9 6-digit cells are underpowered; the anti-A6 and A9 claims are
  softened accordingly; the honest read is a scoped partial positive. A Gate-2
  round-2 re-review returned **PASS WITH CONDITIONS** — verifying the power-control
  fix is real (`sc = min(1, head/direct) ≤ 1`), the rescore is honestly calibrated
  in both directions, and the retained `L1.H1` k=3 positive is genuine (isolated
  single-head, MLP-routed, dec 1.00 / null 0.00 / wrong 0.00). Three non-blocking
  conditions applied: F1 5-digit gate prose corrected (none skipped); F2
  "underpowered" reworded as direct-path-discrimination-specific; F3 dead
  `irrelevant()` arm removed + E-8 wording fixed. **Gate 2 PASSED.***

## Amendments (post-Gate-2, before rescoring)

**2026-07-16 — E-7 (resolves F1/F2-post): per-cell power gate, scaled DOWN.** The
power control is `direct_scaled_percell`: at each (pair, digit) the direct-path
edge is scaled by `min(1, mean_head_edge_norm / direct_norm)` so its magnitude
matches a single head's edge. A cell counts as **underpowered** (no
direct-path/none verdict permitted) unless this scaled-down control flips ≥ bar.
Recomputed: 6-digit **6/9 underpowered**, 0/9 direct-path; 5-digit 3/9 underpowered,
3/9 direct-path. The anti-A6 "direct path inert" headline is withdrawn — A6
residual-carry is *unsupported at the powered 6-digit top-digit cells*, not refuted.

**2026-07-16 — E-8 (resolves F4-post): selectivity arms implemented.** The
selectivity battery now reports deciding-flip, same-class-diff-operand null, and
wrong-digit-flip (a below-deciding non-chain digit toggled), and both
`carries_computed_carry` (dec ≥ 0.5, null ≤ 0.2) and `deciding_selective`
(dec − wrong ≥ 0.2 AND dec − null ≥ 0.2). (An above-chain "irrelevant" arm was
dropped as redundant with the deciding arm — F3 round-2.) Result at the carrying
cells: 6-digit `L1.H1` k=3 dec 1.00 / null 0.00 / wrong 0.00 (computed +
selective); 5-digit `L1.H2` k=2,3 dec 0.93/1.00, null 0.00, wrong 0.00. So *where*
a single head carries the top-digit carry, it carries the **computed, deciding-
selective** bit — but this is confined to specific (head, depth, top-digit) cells.

**2026-07-16 — E-9 (rescore, per F3-post): ≥2-depth bar enforced.** No single head
carries at ≥ 2 depths in either model, so the pre-registered R-edge-selective
condition is **not met**; A9's attention-delivery is scored *circumstantial /
one-depth*, not confirmed.

- **Limitations**:
  - **The attention-vs-residual discrimination is underpowered at most cells** (6/9
    6-digit, 3/9 5-digit): "underpowered" here means the *direct-path* control at
    single-head magnitude cannot flip the cell, so we cannot tell whether the
    direct residual path contributes there — a head may still carry it (and at 3
    cells does). A single head's edge is LN-renormalized against the whole residual,
    damping it below the flip threshold at most positions, so the study localizes
    the carrier only where the effect is strong (top digits, specific depths).
  - One-depth localization: the carrier (`L1.H1` 6-digit k=3; `L1.H2` 5-digit) does
    not replicate across ≥2 depths, so it does not meet the pre-registered
    R-edge-selective bar; a second depth (e.g. 6-digit k=4, currently gate-skipped)
    is needed.
  - `lnfair` doubles as the MLP-only arm (patching `ln2.hook_normalized` leaves the
    skip clean), so the carrying flip does route through the MLP input; a full
    skip-vs-MLP-out decomposition was not separately computed.
  - Single seed per size; the two models differ (6-digit no powered direct-path,
    5-digit live direct-path), so nothing here is a universal claim.
  - 2-layer addition; deep chains rare in training (gate softness at depth; k=4
    6-digit / k=2 5-digit skipped at <0.90).

- **Doc updates** (after Gate 2 passes): ledger; claim-evidence (new **CE10**:
  at the combiner edge, a single L1 head [`L1.H1` 6-digit k=3; `L1.H2` 5-digit]
  carries the top cascade digit's computed, deciding-selective carry through the
  combiner MLP — a one-depth causal crumb resolving CE9's null on `L1.H1`; the
  single-position edge battery is underpowered at most cells, so attention-vs-
  residual delivery is not broadly discriminated and A6 is not refuted);
  synthesis + summary; conjectures (**A9 held at low-medium** — one-depth causal
  crumb, not confirmed; **A6 held** — not refuted; **A5** `L1.H1` causal at k=3,
  noted not raised; C3 not adjudicated); agenda (complete entry 1; follow-ups =
  a less LN-damped / multi-position edge instrument, k=4 second depth, neuron-level
  B2 on `L1.H1`+combiner, and the 5-digit/6-digit divergence B5).

- **Next read**: the edge path-patch gives a one-depth causal crumb — `L1.H1` (the
  CE8 cell CE9 found inert) drives the combiner for the top digit at k=3, computed
  and deciding-selective — but the instrument is underpowered at most cells and the
  effect is one-depth, so A9 is not confirmed and A6 is not refuted. The productive
  follow-ups: (a) a less LN-damped edge instrument (patch across the affected
  digit's *whole* consuming path, or several positions jointly) to raise power; (b)
  a second depth (6-digit k=4) to meet the ≥2-depth bar; (c) neuron-level B2 on how
  `L1.H1` + the combiner MLP compute `carry_out`; (d) the 5-digit-vs-6-digit
  delivery divergence (B5). No new training needed.
