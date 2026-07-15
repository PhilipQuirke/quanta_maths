# Study: Deep-Cascade Mechanism — Deciding-Digit Patching and Cascade Tracing (study-deep-cascade-mechanism.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #10

**Verdict: the deep `...999` cascade mechanism is NOT localizable at
node/attention-pattern granularity — R-hybrid/ambiguous (instrument-limited) in
both models; A9 one-hop selection is circumstantially supported but not
confirmed, sequential carried state is disfavored where testable but not
refuted.** With all seven positive controls passing, A9's own prediction — one L1
head that both relocates its target to the deciding digit AND causally delivers
its carry bit to the answer-position combiner — is **not met**: a causally
deciding-selective consumer head (6-digit `L1.H0` Q14: redirect-to-deciding moves
0.93/0.90 at k=2,3, beating both the wrong-digit and irrelevant-token baselines)
and a deciding-digit-*tracking* head (`L1.H2` Q15, top-2 key set follows the
deciding position at ≥2 non-degenerate depths) are **different cells**, and the
CE8 routing cell `L1.H1` is causally **inert** (0.00 under single-head redirect —
so CE8's routing stays representational, not load-bearing). Sequential per-digit
state is disfavored where the transmitting-`=` locus has a passing deciding-digit
control (5-digit k=3: no intermediate-digit identity carried) but the test is
untestable in the 6-digit model, so it is not refuted. Real computed state sits
near the question tail but is graded (joint-flip 0.00), not a single stored bit;
the 5-digit model is fully inconclusive (no causal L1 pattern).

Recorded as CE9. The finding is an **instrument limit**, not a mechanism verdict:
single-site interchange and single-head uniform pattern-redirect are too blunt for
a mechanism split across heads or graded across positions. The discriminating
follow-up is an **edge path-patch** of the candidate L1-head→L1-MLP-combiner edge
(promoted to agenda entry 1). Two post-result skeptic rounds were decisive — the
first caught an over-claimed *negative* (unearned refutations on
uncontrolled/unimplemented assays), the second caught the symmetric over-claimed
*positive* (A9 "confirmed" on a value-matched metric that was never actually coded,
a boundary-artifact tracking gap, and three signatures assembled from three
different cells). No conjecture confidence was raised. Scope: 2-layer addition,
single seed per size.

Status: **complete — Gate 1 and Gate 2 both passed** (Gate 2 after two BLOCK
rounds that corrected over-reach in both directions; see the skeptic-review
sections). Pre-run written 2026-07-15; run and gated 2026-07-16.

## Pre-run (write before the experiment)

- **Question**: How are multi-digit `...999` carry chains physically resolved in
  accurate 2-layer addition models? Specifically: (a) does a *resolved* carry
  state for the chain exist anywhere as causally-patchable stored state (and if
  so, where); (b) is the attention of the carry-consuming positions
  content-dependent on the chain's **deciding digit** (the highest digit below
  the chain top with pair-sum ≠ 9), and causally so; and (c) does one site's
  patch flip all cascade answer digits jointly, or does each consuming position
  resolve independently? Includes the **leading-digit locus** (human C4): the
  top answer digit is predicted from the answer-sign position, where no later
  position can rescue an unresolved carry — never probed by studies #1–#9.

- **Motivation**: This is agenda entry 1 and the thread's top fork
  ([sharpest forks](../maths-conjectures-agent.md#sharpest-forks)). All `U`
  evidence so far (CE4–CE7) is single-digit `U` at a middle answer digit, where
  `carry_in` is one make-carry bit; the human C4 reflections
  ([maths-conjectures-human.md](../maths-conjectures-human.md#c4-reflections-on-experiments-1-to-7))
  point out this cannot distinguish "the cascade rides the stream" from "there
  is no cascade to ride", and that the CE5 combiner story is implausible as the
  *whole* `U` mechanism for deep chains. Human lean (recorded 2026-07-15):
  sequential carried state, paper-style. Agent lean:
  [A9](../maths-conjectures-agent.md#a9-deep-u-cascades-are-resolved-by-one-hop-attention-selection-not-sequential-propagation)
  one-hop selection. These, plus wide-fetch, make divergent causal predictions
  testable with the existing patching harnesses.

- **Ground-truth facts the design uses** (self-sufficiency):
  - For a chain with digits `d+1..n` all `Dj+D'j = 9` and deciding digit `d`
    with `Dd+D'd ≠ 9`: `carry_out(n) = [Dd+D'd ≥ 10]`, *independent of any
    carry into `d`* (sum ≤ 8 absorbs it; sum ≥ 10 already carries). Toggling
    the deciding class flips answer digits `A_{d+1}..A_{n+1}` simultaneously
    (each 9-digit answers 9 vs 0; the top gets +0 vs +1).
  - **Token/position maps** (from
    [study-confirm-st-node.md](study-confirm-st-node.md) A-7, re-verified in
    code): 5-digit (`n_ctx=19`): pos 0–4 = `D4..D0`, 5 = `+`, 6–10 = `D'4..D'0`,
    11 = `=`, 12 = answer sign, 13–18 = `A5..A0`. `Dn` at pos `4−n`, `D'n` at
    `10−n`, `A_k` at `18−k`. 6-digit (`n_ctx=22`): pos 0–5 = `D5..D0`, 6 = `+`,
    7–12 = `D'5..D'0`, 13 = `=`, 14 = sign, 15–21 = `A6..A0`.
  - **Consuming position** of answer digit `A_k` (the position whose residual
    produces `A_k`'s logits) = `pos(A_k) − 1`. 5-digit: consuming(`A_k`) =
    `17−k`; the leading digit `A5` is consumed at pos 12 (the answer-sign
    token). 6-digit: consuming(`A_k`) = `20−k`; `A6` consumed at pos 14.
    Cross-check: CE5's combiner for digit 2 sits at pos 14 = consuming(`A3`) ✓.
  - **Causal-mask constraint** (architectural, shapes the hypothesis space):
    answers are emitted high→low, so the consuming position of a *higher*
    answer digit comes *earlier* than those of lower digits — the top consumer
    cannot read anything computed at later answer positions. And lower digits'
    operand tokens appear *later* in the question than higher digits', so a
    stepwise low→high cascade across question positions is impossible under
    the causal mask. In a 2-layer model, cross-position state readable by a
    consumer's L1 attention must live in `resid_post(L0)` at positions ≤ the
    consumer (L1 outputs at earlier positions are never readable — that would
    need an L2). Hence "sequential carried state" concretely operationalizes
    as **resolved carry state stored at the question tail** (the `D'0` / `=` /
    sign region), which is also where CE4 found its depth-1 conduits
    (`P10.L0.MLP` 5-digit, `P11.L0.MLP` 6-digit).

- **Hypothesis / competing reads** (neutral; taxonomy is by *storage locus* ×
  *fetch behavior*, mapped from observables, not story names):
  - **R-tail (carried state; operationalizes the human/paper lean)**: the
    resolved chain carry is computed and stored at question-tail positions in
    `resid_post(L0)`; consumers fetch the stored resolved bit with effectively
    static attention. Signature: tail *pure-state* patches (see Design) flip
    all affected answer digits jointly at every depth; consumer pattern-patches
    sub-bar; no deciding-digit target tracking.
  - **R-selection (A9)**: no stored resolved state; each consumer's L1
    attention relocates to the deciding digit's information (operand positions
    / their L0-written signals) and delivers one bit to the local combiner
    (CE5). Signature: consumer pattern-patches ≥ bar and value-matched target
    tracking of the deciding position; question-side pure-state patches do not
    transmit deep-chain flips (beyond the depth-1 adjacency CE4 already
    showed); flip topology is per-consumer.
  - **R-widefetch**: no stored resolved state and no content-dependent
    routing; each consumer reads all lower operand content with static wide
    attention and its MLP resolves the chain in one nonlinear step. Signature:
    only operand-content cells are causal; no pure-state cells anywhere; no
    tracking; pattern-patches sub-bar.
  - **R-hybrid / R-mixed**: different mechanism by depth (e.g. tail state for
    shallow, something else for deep) or by model — reported with explicit
    scope, not forced.
  - **R-distributed**: nothing localizes even under joint patches — declared
    only after the joint-patch escalation (CE4 A-3 convention), else the
    result is `underpowered`.
  - **Accuracy-cliff branch**: if the models are substantially *inaccurate* on
    deep chains (per-depth gate below), the mechanism question is moot at
    those depths and the cliff itself becomes the headline finding.

- **Design**:
  - **Models**: primary `add_d5_l2_h3_t15K_s372001`, replication
    `add_d6_l2_h3_t20K_s173289` (independent seed). Weights from
    [PhilipQuirke/VerifiedArithmetic](https://huggingface.co/PhilipQuirke/VerifiedArithmetic)
    via `MathsConfig` + TransformerLens, CPU; accuracy-verified on random
    additions (invalid if < 0.99).
  - **Stimulus family `C(n, k, class)`**: chain top digit `n`; digits
    `n−k+1..n` all pair-sum = 9; deciding digit `d = n−k` with class `lo`
    (sum ≤ 8) or `hi` (sum ≥ 10); digits below `d` no-carry (sum ≤ 8) and
    identical within a matched pair; digits above `n` no-carry, ≠ 9 (chain
    does not extend), with digit `n+1`'s sum ≤ 8 so the top flip is clean.
    Depth-1 reduces to the Gate-2-validated CE4 `U` counterfactual
    (continuity/regression check). **Code assertions per matched pair**
    (the u-resolution-path A-6 lesson): source/target question tokens differ
    at exactly positions `{4−d, 10−d}` (5-digit; analogous 6-digit); chain
    digits sum = 9 in both; deciding sums in opposite classes, both ≠ 9;
    ground-truth answers differ on exactly `A_{d+1}..A_{n+1}`.
  - **Main batteries** (5-digit: n = 3, k ∈ {1,2,3}; 6-digit: n = 4,
    k ∈ {1,2,3,4}) and **leading-digit batteries** (5-digit: n = 4,
    k ∈ {1..4}, consumer = pos 12 = sign; 6-digit: n = 5, k ∈ {1..5},
    consumer = pos 14).
  - **Per-depth behavioral gate** (precondition, also a reported result):
    clean (unpatched) predictions on `C(n,k,hi)` vs `C(n,k,lo)` must differ on
    exactly `A_{d+1}..A_{n+1}` and match ground truth in ≥ 38/40 questions per
    (depth, model). A failing depth is flagged **model-inaccurate regime**:
    its mechanism cells are reported but non-verdict-driving; if k = 2 already
    fails, the study re-scopes to the accuracy-cliff finding.
  - **Battery A — spatial causal map (primary)**: interchange patches on
    matched pairs (target question run with target's clean answer prefix
    teacher-forced; source activation patched in; each affected digit read at
    its own consuming position; both directions hi↔lo).
    - **Cell classification (the core discriminator)**: a site is a
      **pure-state cell** iff its own token content is identical between
      source and target (everything except the deciding operand positions
      `{4−d, 10−d}`); any flip it transmits is *computed state*, not local
      token identity. The deciding operand positions are **operand-content
      cells** — causal under every mechanism, reported but non-discriminating.
    - **Sites, coarse → fine triage** (confirm-st-node A-5 convention): coarse
      unit = `blocks.0.hook_resid_post` per question-side position (the only
      cross-position bus per the causal-mask note) at every position 0..sign;
      hits decomposed into per-head `hook_z`, `hook_mlp_out`, and joint
      head+MLP at that position. Consumer-side cells: L1 `hook_z` per head and
      L1 `hook_mlp_out` at each consuming position.
    - **Metric**: per (site, depth, direction): the **flip vector** — flip
      rate of each affected answer digit `A_{d+1}..A_{n+1}` — plus a
      same-class null (source/target same deciding class, different fillers;
      expected ≤ 0.10). Topology summarized as joint-flip (one site flips all
      digits), per-consumer, or graded.
    - **Joint-patch escalation** (before any distributed/negative verdict):
      tail-region joint (`resid_post(L0)` at pos {10,11,12} 5-digit /
      {12,13,14} 6-digit), chain-region joint (all chain-digit operand
      positions' L0 sites), head+MLP joints.
  - **Battery B — leading-digit locus**: Battery A run with n = top digit; the
    consumer is the answer-sign position. Also reports whether a CE5-style
    combiner signature exists at the sign position's L1 MLP (never tested).
  - **Battery C — consumer attention**:
    - **C1 (representational tracking)**: across `C(n,k,·)` with k varied at
      fixed n (consumer's own operands fixed), per consumer L1 head: attention
      mass on the deciding positions `{4−d, 10−d}` as a function of d, against
      a **role-vs-position null** (mass on the same physical positions in
      family members where that digit is *not* deciding). Top-2-mass metrics,
      not raw argmax (the CE8 tie-noise lesson); CE8's strict value-matched
      null protocol for class-toggle target moves at fixed d. The CE8 registry
      cells ([attention_routing_registry.json](../../results/study-attention-invariance/attention_routing_registry.json))
      are prioritized candidates but all consumer-position L1 heads are
      measured (no selection-on-outcome).
    - **C2 (causal pattern-patch — the selection discriminator)**: patch a
      consumer L1 head's `hook_pattern` row at the consuming position from a
      donor with the deciding digit at a *different* position (minimal-diff
      donor, code-asserted); read the flip vector. Null donor (same deciding
      position/class, different fillers) must not flip. R-selection predicts
      ≥ bar flips; R-tail/R-widefetch predict sub-bar.
  - **Battery D (secondary, non-verdict-driving)**: (i) tie-break economy —
    the same patches/ablations applied on carry-free questions must be
    harmless if cascade dependence is selective (A6's untested sub-claim);
    (ii) tracing — linear decodability of `carry_out(j)` and the resolved
    chain carry at `resid_post(L0)` per question position, with permutation
    nulls; descriptive only (presence ≠ causation — the CE6
    ingredients-vs-decision lesson).
  - **Minimal effects / bars / power**: flip bar = `max(0.50,
    node_control_rate − 0.10)` (confirm-st-node A-6 convention), same-class
    null ≤ 0.10, both directions; ≥ 40 matched pairs per verdict cell (coarse
    map ≥ 20, hits re-run at 40); tracking effect = deciding-position mass
    minus role-vs-position null ≥ 0.40; pattern-patch flip ≥ bar with null
    donor ≤ 0.10. Prior flip metrics in this harness family are
    near-deterministic (1.00 vs 0.00 nulls), so n = 40 separates 0.5 from 0.1
    at > 5σ (binomial); power comes from gap + null + two-model replication,
    not asymptotics. A verdict requires consistency across ≥ 2 depths
    (including k ≥ 2) and both models, or an explicit model-scoped verdict
    (CE8 precedent: 5-digit near-tie attention may again be inconclusive —
    pre-accepted, reported honestly).
  - **Pre-registered decision table** (observable triple → read):
    | State locus (pure-state cells) | Consumer pattern causal? | Flip topology | Read |
    | --- | --- | --- | --- |
    | tail region flips all digits, k ≥ 2 | no | joint from one tail site | **R-tail** |
    | none beyond depth-1 adjacency | yes (+ tracking) | per-consumer | **R-selection** |
    | none anywhere | no (no tracking) | operand-content cells only | **R-widefetch** |
    | inconsistent / partial | — | — | **R-hybrid/ambiguous** (scoped) |
    | nothing ≥ bar after joint escalation | — | — | **R-distributed** |
  - **Scope**: 2-layer addition models, interchange at node/site granularity;
    representational probes secondary; no neuron decomposition (B2), no
    subtraction/mixed models, no claim of a complete verified circuit.

- **Positive controls** (each exercises the actual unit/metric; any failure →
  `invalid`, never "negative"):
  1. **Known state cell, depth 1**: the CE4 conduits (`P10.L0.MLP` 5-digit,
     `P11.L0.MLP` 6-digit) must reproduce their ~1.00 `A_{n+1}` U-flip under
     this harness — proves a genuine stored-state cell registers as one.
  2. **Known node patch**: the CE3 SA head (`P14.L0.H1` 5-digit /
     `P20.L0.H1` 6-digit) z-patch flips `A_n` (~0.95 expected); sets the bar.
  3. **Readout control**: patching residual at each consuming position
     (including the sign position) flips that consumer's digit at ~1.00 —
     the metric registers flips at every consumer, including pos 12/14.
  4. **Pattern-patch validity**: patching the SA head's attention pattern to a
     different digit's operands must move `A_n` predictably — proves a
     pattern-only patch can causally change an answer under this harness
     (closes the gap CE8 left: its synthetic control validated only the
     move-counter, not causal patching).
  5. **Per-depth behavioral separation**: clean hi/lo predictions differ on
     exactly the expected digits (≥ 38/40 per depth per model) — the
     precondition that the target metric can differ at all.

- **Success condition** (defined now): controls 1–5 pass; the observable
  triple (state locus, consumer-pattern causality, flip topology) is
  internally consistent across ≥ 2 depths (k ≥ 2 included) in both models —
  or explicitly model-scoped — and selects exactly one row of the decision
  table (R-tail / R-selection / R-widefetch, or a cleanly-scoped hybrid).
  Scored against A6, A9, C3, A4 (just-in-time fetch), A5 (whether the routing
  exceptions are load-bearing).

- **Failure condition** (defined now): with controls passing — no mechanism
  cell clears bar even after the joint-patch escalation, or the observables
  are irreconcilably inconsistent → **no localizable deep-cascade mechanism at
  this granularity** (R-distributed). A substantive negative: both A6's
  localized carried-state form and A9's localized selection form take damage,
  and finer methods (path patching, neuron-level) are needed.

- **Ambiguous / invalid condition**:
  - Ambiguous: cells between null and bar; depth- or model-inconsistency
    without a clean scoping; tracking present but pattern-patch sub-bar
    (representational-only routing — echoes CE8's caveat and would leave
    R-selection unconfirmed).
  - Invalid: any positive control 1–4 fails; random-question accuracy < 0.99.
  - Special branch (not invalid): per-depth behavioral gate fails at k ≥ 2 →
    re-scope to the accuracy-cliff finding (mechanism claims limited to
    accurate depths; the cliff itself is a headline result bearing on all
    deep-cascade conjectures and, eventually, the paper thread).

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  that rehydrated only from version-controlled docs (this note; both conjecture
  files; document rules; glossary; agenda; claim-evidence; the CE3/CE4/CE5/CE8
  prior study notes and the attention-routing registry JSON) with no
  working-thread context, and verified the token/position layout in
  `maths_config.py` independently.

  **Verdict: PASS WITH CONDITIONS.** The plan has internalized the line's four
  recurring traps (pair-sum instrument-failure, CE6 ingredients-vs-decision,
  `U ≡ SA_n=9`, CE8 argmax-tie) and pre-registers a decision table, control-tied
  bars, joint-patch escalation, and per-pair token-diff assertions. Six findings;
  three blocking:

  - **C1 (blocking) — R-tail/R-selection alias on the tail-region joint patch.**
    The tail slab `resid_post(L0)` at pos {10,11,12} (5-digit) is a *downstream*
    residual that has already accumulated the deciding digit's L0 contribution via
    attention, so swapping it transports the deciding bit whether the mechanism
    *stores* a resolved state there (R-tail) or carries a *per-digit* bit later
    fetched by selection (R-selection) — both flip all cascade digits jointly. The
    tail slab is therefore not "pure-state" under strict own-token-identity. This
    is the "transports vs computes" trap that sank the pair-sum assay, inverted.
  - **C2 (blocking) — the causal-mask argument silently narrows the human
    lean.** The architecture argument (a literal layer-iterated low→high cascade
    is impossible in 2 layers; cross-position state must live in `resid_post(L0)`)
    is *sound* (skeptic verified the layout). But collapsing "sequential carried
    state" to "resolved state stored at the tail" strawmans the human lean: a
    legitimate causal-mask-compatible form is **running accumulation across every
    chain position's `resid_post(L0)`** (A6's "accumulating bus"), which a
    tail-only patch under-detects and the current table has no row for.
  - **C3 (blocking) — positive control 1 mis-cites its instrument.** CE5
    reclassified the CE4 `P10.L0.MLP` (5-digit) as borderline/conduit-leaning and
    the L0 nodes as **conduits**, not stored-resolved-state cells. Control 1 thus
    validates that a *conduit* registers a flip (a harness-liveness control), NOT
    that a *computed-resolved-state* cell registers — which the R-tail verdict
    needs. Without a computed-state positive control an R-tail null is
    uninterpretable.
  - **C4 (non-blocking)** — control 4 correctly closes the CE8 causal-pattern-patch
    gap for Battery C2, but validates the SA head, not the L1-consumer head class
    C2 actually patches; extend it to a CE8 6-digit routing cell.
  - **C5 (non-blocking)** — Battery C1's role-vs-position null is right in
    principle, but raw attention *mass* wobbles with digit value even for a
    target-static head (the CE8 lesson); tie the tracking metric to the CE8
    value-matched top-k target-key-set move, not mass magnitude.
  - **C6 (non-blocking)** — the accuracy-cliff branch is legitimate and
    pre-registered (not an escape hatch); just report the per-depth accuracy table
    for all attempted depths and flag any skipped failing intermediate depth.

  *Status: RESOLVED 2026-07-16 by the working thread — C1 (provenance
  discriminator + local-tail decomposition, D-1), C2 (accumulated-sequential form
  added with its own decision-table row + intermediate-digit stepwise test, D-2),
  C3 (control 1 relabeled harness-liveness + new computed-state control 6, D-3),
  C4 (consumer-class pattern-patch control 7, D-4), C5 (value-matched tracking
  metric, D-5), C6 (all-depths accuracy reporting, D-6). Gate 1 PASSED on the
  amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-16 — D-1 (resolves C1, the core discriminator fix): tail joint flips
require a provenance check before scoring R-tail.** The tail-region joint
(`resid_post(L0)` at pos {10,11,12} 5-digit / {12,13,14} 6-digit) can alias
because that slab has already accumulated the deciding digit's L0 output. Any tail
cell that flips jointly is subjected to two additional tests before it can score
**R-tail**, both reusing the CE5 activation-invariance idea:
- **(a) Position-invariance at fixed resolved carry**: across family members with
  the *same* resolved chain-carry value but the deciding digit at *different*
  positions (vary `k` at fixed `n` and fixed deciding class), is the tail
  activation invariant (R-tail: it stores the resolved bit, position-agnostic) or
  does it co-vary with which lower digit is deciding (R-selection: the slab still
  carries the source's identity)? Metric: tail-activation variance across
  deciding-position at fixed resolved carry, vs the variance across resolved-carry
  at fixed deciding-position; R-tail requires the former ≪ the latter.
- **(b) Local-tail decomposition**: re-run the tail patch holding the *lower*
  positions' `resid_post(L0)` at target and swapping only what the tail tokens
  (pos 10/11/12) wrote *locally* (patch the tail positions' `hook_mlp_out` /
  `hook_z` while freezing the accumulated lower-position contribution). Only a
  flip from the locally-written tail component supports R-tail.
- **Decision-table change**: a joint tail flip that fails (a) or (b) is scored
  **R-selection or ambiguous, not R-tail** (see the amended table in D-2).

**2026-07-16 — D-2 (resolves C2): "sequential carried state" has two admissible
physical forms, each with its own row; the intermediate-digit test is
pre-registered.** "Sequential carried state" (the human/paper lean) is admitted in
two causal-mask-compatible forms: **(form α) resolved bit stored at the tail**;
**(form β) partial cascade state progressively accumulated across the chain
positions' `resid_post(L0)`** (A6's accumulating bus). Form β is detected by the
**chain-region joint** (already in the design, line ~146) plus an **intermediate
all-9s digit patch** (A9's own falsifier, agent-conj line ~509): patch an
intermediate chain position's question-side L0 nodes (a digit strictly between the
deciding digit and the chain top, whose pair-sum = 9).
- Under **form β (sequential accumulated)** patching an intermediate all-9s
  position breaks the answer digits *above* it (stepwise dependence).
- Under **R-selection** it does ~nothing — only the deciding digit matters.
- Under **form α / R-tail** it does ~nothing at the intermediate position and the
  flip comes from the tail cell.

Amended pre-registered decision table (supersedes lines 188–194):

| State locus (pure-state cells) | Intermediate-digit patch | Consumer pattern causal? | Flip topology | Read |
| --- | --- | --- | --- | --- |
| tail cell flips all digits, k ≥ 2, passes D-1 (a)+(b) | inert | no | joint from one tail site | **R-tail (form α)** |
| chain-region joint flips; per-position contributions cumulative | breaks digits above it (stepwise) | no | graded/cumulative | **R-sequential (form β)** |
| none beyond depth-1 adjacency | inert | yes (+ value-matched tracking) | per-consumer | **R-selection (A9)** |
| none anywhere | inert | no (no tracking) | operand-content cells only | **R-widefetch** |
| inconsistent / partial | — | — | — | **R-hybrid/ambiguous** (scoped) |
| nothing ≥ bar after joint escalation | — | — | — | **R-distributed** |

Decision-impact mapping (lines 260–279): R-tail form α → the "resolved by `=`"
picture in operationalized form; R-sequential form β → the human C3/C4 "cascade
rides the residual stream" lean vindicated as *physical accumulation* (the
strongest possible confirmation of the human lean, and a distinct outcome from
form α); both α and β **refute A9**. Only R-selection confirms A9.

**2026-07-16 — D-3 (resolves C3): control 1 relabeled; a computed-state positive
control added.** Control 1 is relabeled to what it proves — **a known U-flip
*transmitter* (CE4/CE5 conduit) registers an `A_{n+1}` flip at depth 1 (harness
liveness)**; the 5-digit `P10.L0.MLP` citation is softened to "borderline per CE5"
and not asserted at ~1.00 as a state cell. New **control 6 (computed-resolved-state
cell)**: the CE5 combiner `P14.L1.MLP` (5-digit) / `P16.L1.MLP` (6-digit) must
reproduce its CE5 activation-invariance signature (definite-regime invariant to
`carry_in`; U-regime variant) under this harness — proving a genuine
*computed*-state cell registers, so an R-tail/R-sequential null is interpretable
(distinguishes "no stored resolved state" from "harness can't register one").
**If control 6 fails, the state-locus batteries are `invalid`, not "state
absent".**

**2026-07-16 — D-4 (resolves C4): consumer-class pattern-patch control.** Control
4 (SA-head pattern-patch) is kept, and **control 7** is added: a causal
pattern-patch on one CE8 6-digit routing cell (`L1.H1` Q11 or Q14, registry
`clean:true`) must move its read predictably — validating the actual head class
Battery C2 patches. If control 4 passes but control 7 fails, **C2 is `invalid` for
the consumer battery specifically** (not scored "no selection").

**2026-07-16 — D-5 (resolves C5): tracking metric is the value-matched target-key
move, not mass magnitude.** Battery C1's ≥ 0.40 tracking bar is on the **top-2
target-key-set move under the CE8 value-matched contrast** (hold the deciding
digit's *value* fixed, move only its *position* across family members; require the
top-2 key set to follow), not on raw attention mass on a physical position — so a
positionally-static but value-sensitive head cannot trivially pass. The
role-vs-position null stands as the floor.

**2026-07-16 — D-6 (resolves C6): all-depths accuracy reporting.** The per-depth
behavioral-gate accuracy table is reported for **every** attempted depth
regardless of pass/fail; a mechanism verdict requires ≥ 2 *passing* depths
(including k ≥ 2), and any skipped failing intermediate depth is flagged
explicitly (no cherry-picking non-contiguous passing depths silently).

- **Decision impact**:
  - **R-tail**: A9 refuted; A6's carried-state core strengthened and
    *relocated* (state lives at the question tail, not per-position en route);
    C3/C4 human lean vindicated as physical; B12 gains a sharp target (the
    tail state's format — per-digit binary vector vs anything tri-state — is
    the last place A3's remnant could live); the paper's "resolved by `=`"
    survives in operationalized form.
  - **R-selection**: A9 confirmed; A6 rewritten (stored per-digit bits +
    data-dependent fetch; no ridden state); A5's hybrid routing exceptions
    promoted from curiosity to mechanism; C3's propagation clause scored
    refuted-as-physical (the cascade is a functional description); relevant to
    the paper thread eventually (no paper edits — guardrail stands).
  - **R-widefetch**: A9 refuted; A6 alternative (a) wins; the C4
    "8-way combine is implausible" intuition itself takes damage —
    interesting either way.
  - **R-distributed / ambiguous**: conjectures held; agenda reranks toward
    finer-grained methods (edge path-patching, B2 neuron decomposition).
  - **Accuracy cliff**: all A6/A9 claims scoped to shallow chains; new
    question opened (how deep-chain failures look; training-distribution
    rarity), flagged for eventual paper-thread attention.

- **Risks / confounds** (with mitigations):
  - **Operand-content vs computed state** (the central trap): only
    pure-state cells (own-token-identical) drive state verdicts; the deciding
    operand positions are excluded by construction; code asserts the token
    diff set per pair.
  - **Redundancy / single-site under-detection** (Paper-2 redundancy; CE4
    F4): joint-patch escalation is mandatory before R-distributed or any
    "state absent" reading.
  - **Downstream recompute masking**: a consumer reading operand tokens
    directly can absorb an upstream state patch — under this design that is
    not a failure mode but the R-selection/R-widefetch *signature*;
    interpretable only because control 1 proves a genuine state cell
    registers (the combiner-vs-conduit F2 lesson, inverted).
  - **`U ≡ SA_n=9`** (the fatal historical confound): chain digits are
    sum-9 in *both* pair members — matched, never a contrast; no
    U-vs-committed comparison is used as a discriminator anywhere.
  - **Teacher-forcing / prefix effects**: each affected digit is read at its
    own consuming position with the target's clean answer prefix; prefix
    tokens through each read position are asserted identical between the
    clean and patched runs.
  - **Argmax-tie noise in tracking** (CE8's 5-digit failure): top-2-mass
    metrics with strict nulls; 5-digit pre-accepted as possibly inconclusive;
    model-scoped verdicts allowed and reported honestly.
  - **Pattern-patch donor incompatibility**: minimal-diff donors within the
    family, null donors as the floor, and control 4 validating the method on
    a known-causal pattern.
  - **Rare-pattern memorization** (Nikankin-style heuristics): deep chains
    are rare in random training data; erratic per-depth behavior would
    surface in the behavioral gate and per-depth inconsistency — reported,
    not forced into a mechanism row.
  - **Multiple comparisons** over the site × depth grid: per-cell nulls +
    both directions + control-tied bar + cross-depth consistency + two-model
    replication (CE4 A-4 convention); coarse-map hits confirmed at higher n.
  - **Tail-cell superset effects**: `resid_post(L0)` at `=`/sign feeds *all*
    later positions' L1 reads; a tail hit localizes state to the region, and
    the head/MLP decomposition plus flip topology (which digits move) refine
    it. The CE3 F2 caution about the `=`-resid patch (layer-0-only,
    autoregressively confounded, returned 0.00) is inherited as a caveat on
    interpreting tail *nulls* at that single position — hence the region-wide
    joint patch.
  - **2-layer scope**: deeper models have more options; findings are scoped
    to these models (B5/B10 are the generalization follow-ups; B10's
    one-layer contrast becomes especially informative if R-selection wins,
    since a 1-layer model lacks the L1 fetch stage).

- **Expected artifacts**:
  - Standalone CPU script `scripts/deep_cascade_mechanism.py` (reuses
    confirm-st-node helpers, the u-resolution-path chain construction, and
    the attention-invariance metrics/registry).
  - `results/study-deep-cascade-mechanism/`: `results.json` carrying **every
    headline number** (per-depth behavioral gate, all flip vectors + nulls,
    tracking masses + nulls, pattern-patch rates, control results — the
    recurring evidence-integrity rule: no number exists only in prose);
    `control_*.json`; `site_flip_registry.json` (per-site classification);
    plots: position × depth flip heatmaps per model, flip-topology summaries,
    tracking-mass-vs-depth curves, pattern-patch table; per-depth accuracy
    table. No HF uploads (analysis-only).

## Post-run (fill in after the experiment)

- **Executive summary** (final, after two Gate-2 BLOCK rounds forced correction of
  over-reach in *both* directions): **At node/pattern granularity the deep-cascade
  mechanism cannot be cleanly localized — R-hybrid/ambiguous / underpowered for a
  positive verdict in both models. A9 selection gets suggestive but not confirmed
  support; sequential accumulated state is not supported where testable; no single
  mechanism is established.** With every positive control passing, the corrected
  assays (controlled `=`-locus intermediate test D-7; genuine value-matched top-2
  key-set tracking with a units-end control D-8; C2 selectivity with an
  irrelevant-token baseline D-9; same-cell requirement) give a consistent but
  *non-convergent* picture:
  - **A9's two signatures do NOT land on the same cell (6-digit).** A consumer L1
    head *is* causally deciding-selective (`L1.H0` Q14: redirect-to-deciding moves
    0.93/0.90 at k=2,3 vs wrong 0.42/0.55 and irrelevant 0.38/0.50 — beats both
    baselines), and a *different* head tracks the deciding digit's position across
    depths (`L1.H2` Q15, 2/3 non-degenerate depths) — but the tracking head is
    **causally non-selective** (its redirect moves 0.28/0.00/0.05) and the causal
    head does **not** track. A9 predicts one head that both relocates to the
    deciding digit AND causally delivers its bit; that convergence is **absent**.
  - **The specific CE8 routing cell (`L1.H1`) is causally inert** (redirect move
    0.00 at every depth) — confirming CE8's "representational, not causal" caveat;
    it is *not* the deep-cascade mechanism.
  - **Sequential per-digit state: not supported where testable, not refuted.**
    Where the `=`-locus intermediate test has a passing deciding-digit control
    (5-digit k=3, control 0.75), the cascade state does **not** carry
    intermediate-digit identity (consistent with selection / against accumulation);
    in the 6-digit model the `=` control fails at all depths, so the test is
    *untestable* there. One passing depth in one model cannot carry a refutation.
  - **Real computed state near the tail, but graded:** pure-state cells flip near
    the question tail (null 0.00) but with joint-flip 0.00 — not a single stored
    resolved bit.
  - **5-digit is fully inconclusive:** no causal L1 pattern (C2 inert 0.00), no
    tracking — matching CE8's 5-digit `clean:false`.

  Net (honest): these node- and attention-pattern-level interventions are
  **underpowered to confirm a single deep-cascade mechanism**. The evidence is
  *weakly consistent with* A9-style selection (a causally-selective consumer head
  exists; a head tracks the deciding digit; no sequential identity is carried) but
  falls short of A9's own prediction of a single head doing both — so A9 stays
  **circumstantial, not causally confirmed**. Sequential carried state is
  disfavored where testable but not refuted. The productive follow-up is a finer
  instrument (edge path-patching) that can test the selection→combiner hand-off
  directly; a blunt uniform-redirect on single heads is the wrong tool. Scope:
  2-layer addition; node/pattern granularity.

  *(Process note: the first draft over-claimed a substantive negative ["sequential
  refuted; selection not found"]; Gate 2 round 1 [BLOCK] showed those refutations
  rested on uncontrolled/unimplemented assays. The second draft over-corrected to
  "A9 selection confirmed in 6-digit via L1.H1"; Gate 2 round 2 [BLOCK] showed that
  rested on a value-matched metric that was never actually implemented, a boundary
  -artifact tracking gap, and three signatures assembled from three *different*
  cells. This final read — genuine value-matched tracking, same-cell requirement,
  honest "underpowered" — supersedes both.)*

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/deep_cascade_mechanism.py control`
    then `... models`.
  - Scripts:
    [`scripts/deep_cascade_mechanism.py`](../../scripts/deep_cascade_mechanism.py)
    (controls + harness) and
    [`scripts/deep_cascade_batteries.py`](../../scripts/deep_cascade_batteries.py)
    (Batteries A/C, D-1, D-2). Standalone CPU; reuses confirm-st-node helpers
    (`load_model`, `patched_prediction`, `tristate_test`, `flip_signature`,
    `same_class_null`, `operand_attention`) and the CE8 registry. Env: python
    3.13.7, macOS-26.5.2-arm64, torch 2.8.0. Repo commit `92edb5f` (working tree).
    Date 2026-07-16. Seed 20260716.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000, n_top digit 3),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000, n_top digit 4).
  - Artifacts (local; no HF): `results/study-deep-cascade-mechanism/results.json`,
    `control_<model>.json`, `flip_heatmap_<model>.png`.

- **Results** (final corrected assays D-7/D-8/D-9, genuine value-matched tracking +
  same-cell requirement):
  - **Positive controls (all pass; 5-digit C2 pre-registered invalid)**:
    control 1 harness-liveness conduit `A_{n+1}` flip 1.00 (both); control 6
    computed-state CE5 combiner tri-state flip 1.00 (both); control 2 SA-head
    `A_n` flip 0.95–0.97, same-class null 0.00; control 3 readout 1.00 at every
    consuming position (incl. the sign position); control 4 SA-head
    pattern-redirect 0.875–0.90; control 5 per-depth behavioral gate (see F7-post
    reconciliation note below). Control 7: 6-digit consumer-L1 instrument
    (`L1.H0` Q14) causal → C2 valid; 5-digit 0.00 → **C2 invalid for 5-digit**.
  - **Battery A (spatial map)**: pure-state hits (null ≤ 0.10, joint-flip 0.00) are
    `resid_post(L0)` at positions that **move units-ward with depth** (5-digit pos
    10→11; 6-digit pos 10→11→12), i.e. one below the deciding digit, not a fixed
    `=` cell (the `=` token itself, pos 11/13, is inert in Battery A). Corrected per
    F5/F6-post: this is *not* a fixed `=` stored bit; it is graded and
    operand-adjacent.
  - **D-8 genuine value-matched tracking** (top-2 key-set follows the deciding
    position across depths, excluding the units-end degenerate depth, vs a
    units-end control): 5-digit **no** tracking head (0/2 and 1/2 depths). 6-digit
    **one** head tracks — `L1.H2` Q15 (2/3 non-degenerate depths); `L1.H1` (the CE8
    cell) only 1/3; `L1.H0` 0/3.
  - **D-9 C2 selectivity** (deciding vs wrong vs irrelevant redirect): 6-digit
    `L1.H0` Q14 (instrument) deciding 0.93/0.90/0.55 vs wrong 0.42/0.55/0.82 vs
    irrelevant 0.38/0.50/0.47 → **selective at k=2,3, inverts at k=4**. The
    **tracking head `L1.H2` Q15 is causally NON-selective** (deciding 0.28/0.00/0.05
    — barely moves the answer). The **CE8 cell `L1.H1` Q11 is inert** (0.00 all
    depths). 5-digit: all 0.00.
  - **Same-cell check**: **no cell both tracks (D-8) AND is causally
    deciding-selective (D-9)** in either model (`same_cell = []`). The causal head
    (`L1.H0`) and the tracking head (`L1.H2`) are different, and neither is the CE8
    cell.
  - **D-7 controlled intermediate test**: deciding-digit control passes only at
    5-digit k=3 (0.75); 6-digit `=` control fails at all depths (0.00). Where
    controlled (5-digit k=3), the cascade state does **not** carry
    intermediate-digit identity.
  - **D-1(a) tail position-invariance**: **downgraded to uninformative** (F5-post) —
    outlier/near-noise-floor variances; not leaned on.

- **Interpretation** (against pre-stated conditions; goalposts unchanged; final,
  post two Gate-2 rounds):
  - **Both models: R-hybrid/ambiguous — underpowered for a positive verdict.** The
    observable triple does not select a single decision-table row, and critically
    A9's own prediction (one head that both tracks the deciding digit *and* causally
    delivers its bit) is **not met**: tracking and causal selectivity land on
    *different* heads, and the CE8 candidate cell is causally inert.
  - **A9 (selection): circumstantially supported, NOT causally confirmed.** The
    ingredients A9 needs exist separately (a causally-selective consumer head; a
    head that tracks the deciding position; no carried per-digit identity) but do
    not converge on one cell at this granularity. This neither confirms nor refutes
    A9 — it says the node/pattern instrument is too blunt to close it.
  - **Sequential accumulated state (human lean): disfavored where testable, not
    refuted** — one passing controlled depth (5-digit k=3) shows no
    intermediate-digit identity; the 6-digit test is untestable.
  - **Tail state**: real graded computed state near the tail, operand-adjacent, not
    a single stored resolved bit.
  - **Instrument lesson (mirrors the pair-sum instrument-failure and CE8 tie-noise
    lessons)**: single-head attention-pattern redirect is a blunt causal probe;
    the selection→combiner hand-off needs an *edge* path-patch, not a node sweep.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2 passes):
  - **A9** (one-hop selection of the deciding digit): **untouched → weakly
    suggestive, NOT confirmed.** The pieces exist (selective causal head; tracking
    head; no sequential identity) but A9's predicted single-cell convergence is
    absent at this granularity, and the CE8 candidate cell is causally inert. Hold
    A9 at **low-medium**; do not raise. The edge path-patch (B11) is the test that
    could actually confirm it.
  - **A6** (carried state, tie-breaker): the **sequential-accumulated multi-digit**
    form is **weakly disfavored** where testable (no per-digit identity, 5-digit
    k=3), not refuted; A6's single-digit CE5 core untouched.
  - **C3 (human)** (cascade rides the stream): the physical position-to-position
    accumulation clause is **weakly disfavored where testable**, not refuted.
    (Human-owned; noted, not edited.)
  - **A5** (are the CE8 routing exceptions load-bearing?): the CE8 `L1.H1` cell is
    **causally inert** under single-head pattern redirect → **stays
    representational** (the round-2 over-claim that it was load-bearing is
    withdrawn). CE8's caveat stands.
  - **A4** (just-in-time fetch): **untouched** (not cleanly tested).

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + control JSONs + both scripts), with independent re-runs.
  **Verdict: BLOCK.** The R-hybrid/ambiguous top-line is defensible on the sound
  evidence (real pure-state computed state at the `=` token; graded per-digit flip
  topology; blunt-but-causal L1 patterns), but the three specific *refutations*
  that gave the first write-up its force are **not earned**:

  - **F2-post (BLOCKING) — the D-2 intermediate test locus is uncontrolled and
    provably inert.** Patching `resid_post(L0)` at operand positions returns 0.00
    *even for the deciding digit that provably drives the whole cascade* — only the
    `=` token transmits under this harness. So an intermediate 0.00 cannot separate
    "no sequential state" from "this locus never transmits"; the
    positive-control-failure ⇒ `invalid`-not-`negative` rule applies. "R-sequential
    (form β) refuted" is unearned.
  - **F3-post (BLOCKING) — Battery C1 tracking (D-5) was not actually
    implemented.** `deciding_target_tracking` recorded raw attention *mass*, the
    exact metric D-5 forbade; the value-matched key-set-move-vs-role-null (≥ 0.40
    bar) never ran. A9's tracking prediction is therefore **untouched**, not
    "refuted", and the raw masses actually show tracking-like signal at some
    depths (6-digit L1H1@15 = 0.998 at k=4) that the note folded into
    "non-selective".
  - **F4-post (BLOCKING) — the C2 "non-selective" reading is not separable from
    intervention crudeness.** Redirecting the 6-digit consumer head to *irrelevant*
    (`+`/`=`) tokens also moves the answer (0.30–0.675), comparable to the
    wrong-digit rates — so "wrong ≈ deciding" measures generic pattern-disruption
    sensitivity, not non-selectivity. No selectivity positive control. A9's
    selection mechanism is **not supported / underpowered**, not "refuted".
  - **F5-post (non-blocking)** — D-1a position-invariance ratio (1.34, 1.18) is
    outlier-over-outlier / noise-floor (6-digit vars ~1e-3); downgrade to
    "uninformative at this power"; lean on the graded joint-flip=0 topology instead.
  - **F6-post (non-blocking)** — the tail slab is deciding-operand-contaminated at
    deep depths (D'd inside the slab); report R-tail provenance only from the clean
    `=` single cell.
  - **F7-post (non-blocking)** — two divergent behavioral-gate tables (results.json
    min 0.90 vs control JSON min 0.85) from unsynced RNG; reconcile and report one,
    re-confirm the D-6 "≥ 2 passing depths incl. k ≥ 2" precondition.

  *What Gate 2 confirmed is right: controls genuinely pass and are correctly wired;
  the `=`-tail pure-state finding is real and honestly hedged; the graded topology
  soundly rejects a single clean stored bit; the R-hybrid top-line does not
  over-claim a mechanism.*

  *Status: RESOLVED 2026-07-16 by the working thread. Round-1 BLOCK resolved via
  amendments D-7/D-8/D-9 (below); a Gate-2 round-2 re-review then returned a second
  BLOCK, catching a symmetric over-reach in the positive direction (a value-matched
  tracking metric asserted in the docstring but never implemented; a
  boundary-artifact tracking gap; three A9 signatures assembled from three
  different cells; the CE8 cell actually causally inert). Round-2 findings resolved
  by: implementing genuine value-matched top-2 key-set tracking with a units-end
  control and a ≥2-non-degenerate-depth requirement; requiring tracking AND causal
  selectivity on the SAME cell for an R-selection verdict; correcting the
  tail-cell localization; withdrawing the A5 "load-bearing" and A9 "confirmed"
  claims. The re-run gives R-hybrid/ambiguous (underpowered) in both models — see
  the final Executive summary. Gate 2 PASSED on the corrected, honest read (the
  verdict is now a scoped ambiguous/underpowered result with no over-claim in
  either direction).*

  **Gate-2 round-2 findings (recorded):** F1 tracking gap was a units-end boundary
  artifact; F2 value-matched metric was never implemented (raw mass relabeled); F3
  the "converge on L1.H1" claim was false (tracking L1.H2, causal L1.H0, CE8 cell
  L1.H1 inert); F4 C2 k=4 inversion; F5 tail hits mislocalized (operand-adjacent,
  not `=`); F6 gate tables still divergent; F7 D-1a wording. All applied in the
  final read.

## Amendments (post-Gate-2, before rescoring)

**2026-07-16 — D-7 (resolves F2-post): D-2 re-targeted to a transmitting locus
with a matched positive control.** The intermediate-digit test no longer patches
`resid_post(L0)` at operand positions (proven inert even for the deciding digit).
Instead it patches the intermediate digit's contribution **as read at the `=`
token** (the one question-side position shown to transmit the cascade), via a
path-style patch of the intermediate position → `=` `resid_post(L0)`, AND adds a
**matched positive control**: the *deciding*-digit version of the same locus must
flip ≥ bar (proving the locus transmits). If the deciding-digit control is
sub-bar the test is `invalid` for that depth. Only with a passing control does an
intermediate 0.00 count as "no sequential state". Verdict language: the
`=`-only-transmission observation is reported as a *hint* against
position-to-position accumulation, not a "refuted" until the controlled test runs.

**2026-07-16 — D-8 (resolves F3-post): D-5 value-matched tracking implemented.**
Battery C1 now computes, per consumer L1 head and depth, the **top-2 key-set move**
under the value-matched contrast (deciding digit's value fixed, position varied)
against the **role-vs-position null** (mass on the same physical position when it
is NOT the deciding digit), with the pre-registered ≥ 0.40 bar. Raw mass is kept
descriptive only. A9's tracking prediction is scored from this metric.

**2026-07-16 — D-9 (resolves F4-post): C2 selectivity positive control.** Battery
C2 adds an **irrelevant-token redirect baseline** (redirect the consumer head to
`+`/`=` non-operand positions): non-selectivity may only be concluded if
redirect-to-deciding materially exceeds BOTH redirect-to-wrong AND
redirect-to-irrelevant. If deciding ≈ irrelevant, the instrument is too blunt →
C2 `underpowered`, not "selection refuted". A6/A9/C3 are rescored only after D-7/
D-8/D-9 runs; F5/F6/F7 applied as wording/reporting fixes.

- **Limitations**:
  - **Node/attention-pattern granularity is underpowered for this question** —
    that is the study's main finding. A graded/distributed mechanism, and any
    mechanism where the necessary computation is split across heads, evades
    single-site interchange and single-head pattern redirect.
  - The C2 pattern-patch is a *uniform-over-2-keys* redirect; it is causal on
    validated heads (controls 4/7) but too blunt to resolve selectivity cleanly
    (k=4 inversion; 5-digit fully inert). A natural-pattern donor would be less
    blunt.
  - The D-7 intermediate test transmits only where the `=` deciding-control passes
    (5-digit k=3 only); it is untestable in the 6-digit model, so the
    sequential-state question is only weakly addressed.
  - 2-layer addition, single seed per size; deep chains are rare in training data;
    mild accuracy softness at depth (behavioral gate 0.85–0.98).

- **Doc updates** (after Gate 2 passes): ledger (new bundle); claim-evidence (new
  **CE9**: at node/pattern granularity the deep `...999` mechanism is
  **not localizable** — R-hybrid/ambiguous/underpowered; A9's predicted single-cell
  tracking+causation convergence is absent [tracking `L1.H2`, causal `L1.H0`, CE8
  `L1.H1` inert]; sequential per-digit state disfavored where testable [5-digit
  k=3]; graded operand-adjacent tail state; 5-digit fully inconclusive);
  synthesis + summary; conjectures (**A9 held at low-medium — pieces present but
  convergence unshown**; A6 sequential-accumulated deep form weakly disfavored;
  **A5 CE8 cell stays representational** — the round-2 "load-bearing" over-claim
  withdrawn; C3 physical-cascade clause weakly disfavored where testable); agenda
  (complete entry 1 as **instrument-limited**; promote **B11 L0→L1 edge
  path-patch** as the finer instrument that can actually close A9).

- **Next read**: the node/pattern instrument cannot resolve the fork; the finding
  is an **instrument limit**, not a mechanism verdict. The productive follow-up is
  **B11** (edge path-patch: freeze the answer-position L1-MLP combiner's other
  inputs, vary only the candidate L1-head→combiner edge) — an edge-level causal
  test that can show whether the selective consumer head's output actually drives
  the combiner, closing the selection→combine hand-off that single-head redirects
  cannot. B10 (one-layer contrast) tests whether selection is forced by the
  2-layer architecture. A less-blunt (natural-donor) pattern instrument is needed
  to adjudicate the 5-digit model.
