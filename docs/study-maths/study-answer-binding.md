# Study: Answer-Position Binding — Tape vs Register (study-answer-binding.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #13

When the model writes out the answer, how does it keep track of each digit's result
and each carry? Two pictures: a "tape" (every digit's value stored in its own tidy
slot, all at once) or a "register" (it fetches each digit's value just when it needs
it). We checked what's readable at the `=` sign and at each answer position.

The answer is a **mix, and it's consistent across two models**: the *sum digits* are
fetched just-in-time (register — you only find a digit's value at the moment it's
written), while the *carries* are already worked out and readable at the `=` sign —
but they're stored in an overlapping, tangled way, **not** in tidy separate slots
(so "tape" is wrong). Unlike the input side (previous study), the answer side *does*
reuse a shared template across positions. A review pass stopped us from over-claiming
that the carries are "stored" there — we can only say they're computed/available by
then, which fits earlier findings.

## Pre-run (write before the experiment)

- **Question**: At the answer phase (`=` and the answer token positions), how are
  the resolved per-digit states `SA_n = (Dn+D'n+carry)` digit and `SV_n` (resolved
  carry into digit n) laid out? (a) **Coexistence**: at a single position, how many
  *different* digits' resolved states are simultaneously linearly decodable — few
  (a "register"/just-in-time fetch, A4) or many orthogonal slots (a "tape")? (b)
  **Cross-answer-position transfer**: does an `SA`/`SV` probe trained at answer
  position `i` transfer to answer position `j` (a shared answer-side template),
  unlike the position-specific question-side `ST` (CE11)? (c) **Slot geometry**: are
  the co-present per-digit slots orthogonal?

- **Motivation**: CE11 showed the question-side `ST` is position-specific and that
  `SA` decodes perfectly at the answer position (not the question site). A4
  explicitly relocates the coexistence/binding question to the answer phase and
  predicts **just-in-time fetch** (each answer position holds mainly *its own*
  digit's state; few others coexist), against a **tape** layout (all resolved digits
  stored in orthogonal slots at `=`). The pre-check already hints at a nuanced
  answer: at `=`, `SV` (carry) decodes for all digits (0.69–0.86) but `SA` (sum
  digit) does not (≈ chance) — carries coexist, sums do not. This study makes that
  precise with the coexistence census + transfer + a baseline that separates genuine
  storage from operand-token re-reading.

- **Ground-truth facts the design uses** (self-sufficiency):
  - **Labels** (deterministic from `(a,b)`): `SA_n` = the n-th **answer** digit =
    `(Dn+D'n+carry_in_n) % 10`; `SV_n` = `carry_in_n` (resolved carry into digit n).
    Both computed by the cascade in `sub_labels` (reused from probe-transfer).
  - **Positions** (6-digit, `n_ctx=22`): `=` at `2·n_digits+1 = 13`; answer sign at
    14; answer token `A_k` at `n_ctx-1-k`; the residual that **produces** `A_k`'s
    logits (its "consuming position") is `A_k`'s token position − 1 = `n_ctx-2-k`
    (e.g. A_0@20, A_1@19, …, A_5@15, A_6=sign@14). Read `blocks.1.hook_resid_post`.
  - **Causal-mask fact (the key confound)**: at answer position for `A_k` (a late
    token), the model can still attend to *all* operand tokens (they are earlier),
    so decoding a *lower* digit's `SA`/`SV` at a *higher* answer position could be
    the position **re-reading the operands**, not a *stored* resolved state. The
    baseline must separate "resolved state is bound/stored here" from "operands are
    re-derivable here". (This is the CE9/pair-sum transports-vs-computes lesson in
    probe form.)

- **Hypothesis / competing reads** (neutral):
  - **R-register (A4 just-in-time)**: each answer position holds mainly *its own*
    digit's resolved `SA`/`SV`; the coexistence count (how many *other* digits'
    states decode there beyond operand re-derivation) is low; `SA`/`SV` transfer
    across answer positions after a positional offset (a shared answer template).
  - **R-tape**: many resolved per-digit states are simultaneously decodable in
    **orthogonal** slots at `=` / a single position — high coexistence, orthogonal
    slot angles, beyond operand re-derivation.
  - **R-split (the pre-check hint)**: carries (`SV`) coexist at `=` (a partial
    carry-tape) but sums (`SA`) are fetched just-in-time (register) — a genuinely
    mixed layout; reported explicitly if the two sub-tasks differ.
  - **R-operand-rederivation (null-ish)**: apparent coexistence is entirely operand
    re-reading — decodability at a foreign position matches the operand-only
    baseline; nothing is *stored/bound*, so neither tape nor register in the
    representational sense.

- **Design**:
  - **Models**: `add_d6_l2_h3_t20K_s173289` (primary), `add_d5_l2_h3_t15K_s372001`
    (replication). CPU; accuracy-verified (invalid if < 0.99).
  - **Read sites**: `blocks.1.hook_resid_post` at (i) the `=` position and (ii) each
    answer consuming position `n_ctx-2-k`. Balanced classes, ≥ 4000 questions,
    70/30 split, fixed seed (reuse the probe-transfer harness).
  - **Battery C — coexistence census**: at each read position `p`, train a probe for
    every digit `n`'s `SA_n` and `SV_n`. The **coexistence count** at `p` = number
    of digits `n` whose state decodes at `p` **above the operand-rederivation
    baseline** (below). Report the full `position × digit` decodability grid for
    `SA` and `SV`.
  - **Operand-rederivation baseline (the critical control)**: for each (position `p`,
    digit `n`), a probe that predicts `SA_n`/`SV_n` from the *operand tokens'* own
    activations must be matched. Concretely: (1) a **shuffled-context control** —
    hold digit `n`'s operands fixed but re-randomize *all other* digits; if `SA_n`
    still decodes at `p`, the info is bound to `n` not re-derived from a
    position-specific mix; and (2) an **operand-only reference** — the decodability
    of `SA_n`/`SV_n` from the `Dn`/`D'n` operand positions themselves (upper bound on
    "just re-read the operands"). A foreign-digit state "coexists" at `p` only if it
    decodes there at a level not explained by operand re-derivation.
  - **Battery T — cross-answer-position transfer**: for `SA` and `SV`, the
    answer-position transfer matrix (probe trained at answer pos `i`, tested at `j`),
    raw + position-mean-centered (the CE11 protocol). Contrast with the CE11
    question-side result (`ST` did not transfer). A4 predicts answer-side transfer
    works (shared template) — a *direct C2/A4 discriminator* at the answer phase.
  - **Battery A — slot angles**: at `=` (and a representative answer position),
    principal angles between the per-digit `SV_n` slots (and `SA_n` where decodable),
    on class-mean activation subspaces (CE11 protocol), vs a label-correlation null.
    Orthogonal slots (≥ 60°) ⇒ tape-like; overlapping ⇒ shared/register.
  - **Controls / baselines**: chance floors (SA 0.10, SV 0.50); the diagonal
    (own-position) probe as the positive control (must decode ≫ chance — the
    pre-check shows SA/SV = 1.00 at their own answer positions); the
    operand-rederivation baseline above; per-position class balance.
  - **Minimal effects / bars**: a foreign-digit state "coexists" if it decodes
    ≥ chance + 0.15 AND ≥ operand-baseline + 0.10 at position `p`. Answer-side
    transfer "works" if off-diagonal retention_gain ≥ 0.6 (the CE11 bar, so the
    contrast is apples-to-apples). Slot orthogonality ≥ 60° vs the label-corr null.
    ≥ 4000 questions (≈ 1200 test); both models or explicitly model-scoped.
  - **Pre-registered decision table**:
    | Coexistence at `=` (beyond operand baseline) | Answer-side transfer | Slot angles | Read |
    | --- | --- | --- | --- |
    | low (few digits) | works | — | **R-register (A4 just-in-time)** |
    | high (many digits) | — | orthogonal | **R-tape** |
    | SV high, SA low | — | — | **R-split (carry-tape + sum-register)** |
    | only own-digit decodes; foreign = operand baseline | — | — | **R-operand-rederivation** |
  - **Scope**: 2-layer addition, answer-phase linear probes; no causal patching;
    no subtraction/mixed models.

- **Positive control**: the diagonal (own-position) `SA`/`SV` probe must decode
  ≫ chance at each answer position (pre-check: 1.00). A flat coexistence result
  whose diagonal also fails is `invalid`, not "no coexistence".

- **Success condition** (defined now): the coexistence grid (vs the
  operand-rederivation baseline), the answer-side transfer matrix, and the slot-angle
  table are produced for both models with passing diagonal controls, and select one
  row of the decision table for `SA` and for `SV` (they may differ → R-split).

- **Failure / ambiguous / invalid**:
  - The substantive outcomes are the decision-table rows (register / tape / split /
    operand-rederivation).
  - Invalid: diagonal probe at chance (read site wrong); accuracy < 0.99; the
    operand-rederivation baseline cannot be estimated (control fails).
  - Ambiguous: coexistence between the bar and the operand baseline; models disagree.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + CE2/CE3/CE7/CE11 + the reused probe-transfer harness; position map + SV
  cascade dependence verified in code). **Verdict: PASS WITH CONDITIONS** — five
  findings, four blocking, all sharpening the decodable-vs-bound distinction:
  - **C1 (blocking, linchpin) — the shuffled-context control is ill-posed for SV.**
    `SV_n` is by definition a function of digits ≤ n, so "fix digit n's operands,
    re-randomize others" scrambles the label being probed. Fix: a **label-preserving
    shuffle** — hold digits ≤ n fixed, re-randomize only digits > n (which cannot
    affect SV_n / carry_in_n); for SA_n likewise condition on carry_in_n.
  - **C2 (blocking) — the coexistence *count* is fragile near ceiling.** Replace the
    thresholded count with a **per-(position,digit) margin table** (decode gain over
    the label-preserving operand baseline, with CIs); add an explicit
    **"recompute-everything"** outcome (if operands re-derive the state at foreign
    positions, tape-vs-register is *ill-posed* for that sub-task — a distinct row
    from R-operand-rederivation); mark a sub-task `underpowered` for the census if
    diagonal − baseline < the 0.10 bar.
  - **C3 (blocking interpretation guard) — "carry-tape" must not restate CE7.**
    CE7 already places carry resolution around L1-attn/`=`, so "SV decodes at `=`"
    alone does not establish *storage*. Gate the "tape" label on SV beating the
    label-preserving operand baseline AND orthogonal per-digit slots; else report
    "SV is *resolved/computable* at `=` (consistent with CE7)", not "carry-tape",
    and do **not** update A6/C3 toward stored cascade state.
  - **C4 (non-blocking) — answer-side transfer can pass trivially via operand
    re-derivation.** Require it to beat the operand floor AND the position-centered
    arm, scored on chance-relative gain (CE11 T-7), before claiming a shared
    template.
  - **C5 (blocking, cheap) — SV_0 is a constant class (always 0).** Exclude SV_0
    (and any degenerate class at the read position); restrict to middle digits;
    import CE11's balanced-train+test and `lowvar_frac<3%` guards.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments AB-1…AB-5 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-16 — AB-1 (resolves C1): label-preserving binding control.** The binding
control is a **label-preserving shuffle**: for `SV_n` (and `SA_n`), hold digits
`0..n` fixed and re-randomize only digits `> n` — this leaves `carry_in_n` and hence
`SV_n`/`SA_n` unchanged by construction (asserted in code: the target label is
identical between the two runs of each pair). A foreign-position decode counts as
*bound/stored* only if `SV_n`/`SA_n` decodes at position `p` above the
**operand-only reference** under this label-preserving shuffle. The old
"fix-n-operands, re-randomize-all-others" control is withdrawn (it redefines SV_n).

**2026-07-16 — AB-2 (resolves C2): margin table + recompute-everything row.** The
coexistence census reports a full **per-(position, digit) margin** = decode gain
over the label-preserving operand baseline (with a bootstrap CI), all to
`results.json`; the verdict reads the margin *distribution*, not a thresholded
count. New pre-registered outcome **R-recompute-everything**: if the operand-only
reference already decodes `SA_n`/`SV_n` at ≈ ceiling at foreign positions, the
tape-vs-register distinction is declared **ill-posed for that sub-task** (distinct
from R-operand-rederivation, which is "foreign decode = baseline, nothing extra").
A sub-task with diagonal − baseline < 0.10 is `underpowered` for the census.

**2026-07-16 — AB-3 (resolves C3): storage-vs-computation gate on the tape label.**
The "tape"/"carry-tape" label (and any A6/C3 update toward stored cascade state)
requires BOTH (i) `SV_n` decodes at `=` above the **label-preserving** operand
baseline (beyond what visible operands re-derive) AND (ii) the per-digit `SV_n`
slots are **orthogonal** at `=` beyond the label-correlation null. Absent both, the
result is reported as "SV is resolved/computable at `=` (consistent with CE7)",
explicitly not storage, and does not move A6/C3.

**2026-07-16 — AB-4 (resolves C4): answer-side transfer must beat the operand
floor.** The answer-side `SA`/`SV` transfer matrix is scored on **chance-relative
gain** (CE11 T-7) and must beat BOTH the position-mean-centered arm AND the
operand-rederivation floor to count as a *shared answer-side template*; otherwise
it is reported as "state re-derivable at each answer position", not a template.

**2026-07-16 — AB-5 (resolves C5): degenerate-class + power hygiene.** `SV_0`
(constant 0) and any class that is degenerate at a read position are **excluded**
(reported N/A, not chance). The census/transfer/angle batteries run on middle
digits (units and top reported separately). CE11's balanced-train+test and the
`lowvar_frac < 3%` demotion flag are imported. Detectable margin: ≥ 4000 questions
→ ~1200 test → a 0.10 margin is resolvable at the near-balanced answer positions.

- **Decision impact**:
  - **R-register**: A4 just-in-time fetch supported; C2/A4's answer-phase binding is
    positional; contrasts cleanly with CE11's position-specific question side.
  - **R-tape**: A4 just-in-time falsified — resolved digits are stored in orthogonal
    slots at `=` (the alternative A4 itself names); a "binding at `=`" picture.
  - **R-split (carry-tape + sum-register)**: a new structural fact — the model keeps
    the *resolved carries* available at `=` (consistent with CE7's "U resolved by
    `=`") but fetches *sum digits* just-in-time; refines both A4 and C3.
  - **R-operand-rederivation**: the "coexistence" is illusory (re-reading operands);
    A4's storage framing is the wrong lens — the answer positions recompute.

- **Risks / confounds** (with mitigations):
  - **Operand re-derivation** (central): the operand-rederivation baseline
    (shuffled-context + operand-only reference) is mandatory before any coexistence
    claim; a foreign-digit decode that matches the baseline is *not* coexistence.
  - **Label correlation** across digits (carries are cascade-linked): the slot-angle
    label-correlation null (CE11 protocol) and the per-digit balancing guard against
    reading correlation as a stored orthogonal slot.
  - **Causal mask / autoregressive teacher-forcing**: answer positions are computed
    with the true answer prefix; the read is at the consuming position of each digit
    — consistent across probes. Position identity is decodable (CE11) — report the
    centered transfer arm so a positional offset is separated from no-transfer.
  - **`SV` at `=` may reflect the L1-attention resolution locus (CE7), not storage**:
    decodability ≠ binding; the shuffled-context control (does `SV_n` survive
    re-randomizing other digits?) is what distinguishes a bound slot from a
    transient computation. Reported explicitly.
  - **Low-variance-projection trap**: report the class-mean subspace variance
    fraction (CE11), flag < 3%.

- **Expected artifacts**:
  - Standalone CPU script `scripts/answer_binding.py` (reuses probe-transfer's
    `sub_labels`, `collect`-style extraction, logistic-regression + subspace-angle
    helpers).
  - `results/study-answer-binding/`: `results.json` (every headline number — the
    coexistence grid, operand-rederivation baselines, answer-side transfer matrices
    raw + centered, slot-angle table + null, chance/diagonal controls); plots
    (coexistence heatmap position×digit for SA/SV, answer-side transfer heatmaps).
    No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (corrected post-Gate-2): **The answer phase is a SPLIT
  layout, consistent across both models: the resolved carries `SV` are present and
  linearly decodable at `=` for every digit (consistent with CE7's resolution
  locus), whereas the sum digits `SA` are absent at `=` and present only at their
  own answer position — fetched JUST-IN-TIME (A4's register). And unlike the
  position-specific question-side `ST` (CE11), both `SA` and `SV` share an
  answer-side template that transfers across answer positions. Where testable, the
  orthogonal-"tape" layout is refuted: the per-digit `SV` slots at `=` are NOT
  orthogonal.**
  - **`SV` present at `=`**: `SV_n` decodes at `=` for every middle digit (6-digit
    ~0.60–0.86; 5-digit similar), above the *isolated-operand* reference (which is
    near-chance because `SV_n` depends on the lower digits, not just `Dn`/`D'n`).
    This confirms the carries are resolved/available at `=` — CE7's locus fact — but
    (per Gate 2, F1) does **not** establish presence "beyond a full operand
    recompute"; a digits-`0..n` baseline is the future confirmatory test. No
    "coexistent carry bus" claim is made.
  - **`SV` slots NOT orthogonal (tape refuted)**: at `=` the per-digit `SV_n` pairs
    are 12–27° in 6-digit (all non-orthogonal) and 27°/47°/63° in 5-digit (2 of 3
    non-orthogonal; one borderline pair at 63°) — so the clean orthogonal-tape
    layout (A4's own named alternative) is **refuted in 6-digit and mostly in
    5-digit**. The stronger "entangled below the label-correlation null" reading is
    **6-digit-only** and comes with a 1-D binary-subspace caveat (F3).
  - **`SA` is a register (just-in-time, A4)**: `SA_n` does **not** decode at `=`
    above chance (raw acc 0.10–0.31 ≈ chance 0.10) — absent at `=`, present at 1.00
    only at its own answer position. (One exception: 5-digit `SA_1` margin +0.19,
    units-adjacent, uncorroborated by 6-digit — footnoted.)
  - **Answer-side template (the C2/A4 discriminator vs CE11)**: both `SA` and `SV`
    **transfer across answer positions** (SA retention_gain 1.00; SV 0.76–0.79 ≥ the
    0.6 bar) — a shared answer-side template, unlike the question-side `ST` which did
    not transfer (CE11, 0.10–0.12). For `SV` this beats the (weak) operand-ref and
    the centered arm → a genuine template (as strong as that floor); for `SA` the
    transfer is likely operand re-derivation (SA is fully operand-determined), so
    the SA "template" is caveated.

  Net (both models agree on the split): the answer phase is a **just-in-time sum
  register + carries resolved-and-present at `=` in a non-orthogonal (not-tape)
  layout**, with a shared answer-side template. A4's just-in-time fetch is supported
  for `SA`; the orthogonal-tape alternative is refuted for `SV`. Scoped: 2-layer
  addition, answer-phase linear probes, middle digits; SV presence-at-`=` is
  CE7-consistent (not a new binding claim); A6/C3 untouched (AB-3 gate).

- **Run record**:
  - Command: `PYTHONPATH=. python3 scripts/answer_binding.py all`.
  - Script: [`scripts/answer_binding.py`](../../scripts/answer_binding.py)
    (standalone CPU; reuses probe-transfer `sub_labels`/`class_mean_subspace`/
    `principal_angle_deg`; sklearn). Env: python 3.13.7, macOS-26.5.2-arm64, torch
    2.8.0. Repo commit `92edb5f` (working tree). Date 2026-07-16. Seed 20260716,
    n_q = 4000, balanced train+test.
  - Models: `add_d6_l2_h3_t20K_s173289`, `add_d5_l2_h3_t15K_s372001` (both acc 1.000).
  - Artifacts (local; no HF): `results/study-answer-binding/results.json`,
    `coexistence_<model>.png`.

- **Results**:
  - **Positive control (diagonal)**: `SA`/`SV` decode ≈ 1.00 at their own answer
    position (both models) — the read site holds the resolved states.
  - **Coexistence at `=` (margin = acc − operand-only-ref)**: SV +0.10/+0.22/+0.30/
    +0.18 (6-digit d1–4 region), +0.26–0.40 (5-digit) — all positive; SA ≈ 0 or
    negative (6-digit 0.03/0.01/−0.03/−0.23; 5-digit 0.19/−0.26/−0.24). Operand-ref
    SV 0.53–0.70.
  - **SV slot angles at `=` vs label-corr null**: 12–27° observed, null 36–81° for
    most pairs (6-digit) — entangled below the null; 5-digit noisier (one pair 63°).
    No pair reaches the 60° orthogonality bar cleanly across the board.
  - **Answer-side transfer (retention_gain)**: SA 1.00 (both); SV 0.79/0.76 — both
    ≥ 0.6; SV offdiag 0.88–0.90 > operand-ref (genuine template); SA transfer likely
    re-derivation (fully operand-determined).

- **Interpretation** (against pre-stated conditions; goalposts unchanged; corrected
  post-Gate-2):
  - **`SA` → R-register (A4 just-in-time)**: SA is **absent at `=`** (raw acc ≈
    chance) and present only at its own answer position → just-in-time fetch. (One
    5-digit exception, SA_1, footnoted.)
  - **`SV` → present at `=`, not an orthogonal tape**: `SV_n` is resolved/decodable
    at `=` for all digits (CE7-consistent), but the per-digit slots are **not
    orthogonal** (6-digit all pairs 12–27°; 5-digit 2/3 pairs < 60°) → the clean
    orthogonal-**tape** alternative is **refuted** (6-digit; mostly 5-digit).
    Whether SV presence at `=` is *beyond a full operand recompute* is **untested**
    (the isolated-operand baseline is too weak, F1); no binding/"bus" claim is made,
    and A6/C3 are untouched (AB-3).
  - **Answer-side template**: both sub-tasks transfer across answer positions
    (unlike question-side `ST`, CE11) — a genuine shared template for `SV` (beats
    the weak operand-ref + centering), caveated re-derivation for `SA`. So C2/A4's
    template-sharing holds at the *answer* phase even though it failed at the
    *question* phase for `ST` (CE11) — a position-of-computation-dependent template.
  - **Both models agree on the split**; the entanglement-below-null sub-finding is
    6-digit-only.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2;
  corrected):
  - **A4**: **just-in-time fetch SUPPORTED for `SA`** (register, absent at `=`); the
    **orthogonal-tape alternative REFUTED for `SV`** (present at `=` but slots not
    orthogonal). A4's *answer-side* template-sharing is supported (transfer works),
    complementing CE11's question-side falsification → A4's template claim is
    **position-of-computation-dependent**. Net: raise A4's just-in-time-fetch
    sub-claim; note the tape alternative is refuted; the "stored-bus" reading is not
    claimed.
  - **C2 / A8**: the `SV` carry slots are **not orthogonal** at `=` (no pair ≥ 60°),
    and (6-digit only) more aligned than the label-correlation null — reinforcing
    CE11's challenge to C2's cross-subtask orthogonality and A8's low-interference,
    now also *within* the per-digit `SV` family at the answer phase (with the 1-D
    binary-subspace caveat).
  - **A6/C3**: **untouched** — per AB-3, SV presence at `=` is CE7-consistent
    resolution, not shown to be *stored* orthogonal cascade state (tape gate not
    met), and no causal test was run.

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + script), independently verified. **Verdict: PASS WITH
  CONDITIONS.** The AB-3 storage gate did its job (tape label withheld, A6/C3
  untouched), the SA/SV split and the answer-side-template contrast are robust, but
  the SV framing over-reached:
  - **F1-post (linchpin) — the SV operand-ref baseline is structurally too weak.**
    It decodes `SV_n` from only `Dn`/`D'n`, but `SV_n = carry_in_n` depends on *all
    lower digits* — so that baseline is near-chance (0.53–0.70) *by construction*,
    and beating it at `=` does **not** show "beyond operand re-derivation"; it only
    restates CE7's resolution locus. Drop "carry bus / beyond re-derivation /
    coexistence fact"; a fair test needs the digits-`0..n` baseline (future pass).
  - **F2-post — SA "register" is earned by raw acc ≈ chance at `=`** (SA simply
    absent there), not the margin logic. Reword.
  - **F3-post — the SV entanglement is a 1-D binary-subspace artifact and 5-digit
    doesn't replicate** (2/3 pairs not below null; one *more* orthogonal). Scope to
    6-digit + caveat; the "not orthogonal → tape withheld" conclusion still holds in
    both models regardless.
  - **F4-post — AB-2 bootstrap CIs and AB-5 lowvar flag were promised but not
    implemented.** Retract those provisions in the note (point estimates only).
  - **F5-post — "both models agree" over-smooths** a 5-digit SA_1 margin (+0.19 at
    `=`); footnote it.

  *What Gate 2 confirmed is right: the AB-3 gate correctly withheld the tape label
  and left A6/C3 untouched; the SA-register vs SV-present-at-`=` split is a real
  cross-model structural fact; the answer-side vs question-side template contrast is
  sound (SV transfer beats the centered arm + operand floor); every number traces
  to results.json.*

  *Status: RESOLVED 2026-07-16 by the working thread via prose amendments AB-6/AB-7/
  AB-8 (below) — SV reframed as CE7-consistent (resolved/present at `=`, not a
  coexistent bus); SA-register reworded to absent-at-`=`; entanglement scoped to
  6-digit with the 1-D caveat; AB-2/AB-5 provisions retracted; 5-digit SA_1
  footnoted. The corrected Executive summary / scoring below supersede the first
  draft. Gate 2 PASSED on the corrected read.*

**2026-07-16 — AB-6 (resolves F1/F2-post): SV reframed CE7-consistent; SA-register
reworded.** `SV` is reported as **resolved and linearly present at `=`** (beyond
what the *isolated* per-digit operands `Dn`/`D'n` encode — consistent with CE7's
resolution locus), **not** "coexistent beyond operand re-derivation" (the SV
operand-ref omits the carry-determining lower digits, so that stronger claim is
unearned; a digits-`0..n` baseline is the future confirmatory test). `SA` is a
**register** because it **does not decode at `=` above chance** (raw acc 0.10–0.31 ≈
chance 0.10) — absent at `=`, present only at its own answer position — independent
of any baseline.

**2026-07-16 — AB-7 (resolves F3-post): entanglement scoped to 6-digit + 1-D
caveat.** The `SV` slot-angle "entangled below null" finding is **6-digit-scoped**
(5-digit: 2/3 pairs not below null, one more orthogonal). Since `SV_n` is binary its
class-mean subspace is 1-D, so the principal angle is just the angle between two
mean-difference vectors — small angles for cascade-correlated binary carries can be
a low-dimensional artifact. The **tape-withheld** conclusion ("no `SV` pair reaches
60°", both models) stands regardless; "entangled beyond null" is a weaker,
6-digit-only, artifact-vulnerable positive.

**2026-07-16 — AB-8 (resolves F4/F5-post): retract AB-2 CI / AB-5 lowvar; footnote
5-digit SA_1.** The AB-2 bootstrap CIs and AB-5 `lowvar_frac` flag were not
implemented; margins are point estimates (borderline cells — 6-digit SV digit-4
+0.10, 5-digit SA_1 +0.19 — are not distinguished from small values). The 5-digit
`SA_1` shows a small positive margin at `=` (+0.19, units-adjacent where operand
leakage is highest, uncorroborated by 6-digit) — an exception to the SA-register
pattern, flagged not smoothed.

- **Limitations**:
  - Linear decodability, not causal use; "coexists beyond operand-ref" shows info is
    present beyond re-reading operands, not that it is *used* from `=`.
  - The operand-only reference (decode from Dn/D'n@L0) is one baseline; a probe at
    `=` could exploit intermediate computation not captured by the raw operand
    residual — the margin is a conservative "beyond raw operands" measure.
  - SA answer-side "transfer" is confounded with operand re-derivation (SA fully
    operand-determined); only the SV template beats the operand-ref cleanly.
  - Middle digits only; SV_0 excluded (constant); 2-layer addition; two models
    (agree).

- **Doc updates** (after Gate 2 passes): ledger; claim-evidence (new **CE12**:
  answer-phase layout is split — `SA` just-in-time register (absent at `=`); `SV`
  resolved/present at `=` (CE7-consistent) but its per-digit slots are **not
  orthogonal** → orthogonal-tape refuted; both share an answer-side template that
  transfers across answer positions, unlike question-side `ST`); synthesis +
  summary; conjectures (A4 just-in-time-fetch raised, tape alternative refuted,
  template claim position-dependent; C2/A8 non-orthogonality reinforced [6-digit
  entanglement caveated]; A6/C3 untouched); agenda (complete entry 1; the
  effective-dimensionality breadth study is next; a fair digits-`0..n` SV baseline +
  a causal test of whether the `=` carries are *used* downstream are backlog
  candidates).

- **Next read**: the answer phase splits into a just-in-time sum register and
  carries resolved/present at `=` in a non-orthogonal (not-tape) layout, and
  (unlike the question side) shows a shared answer-side template. The next agenda
  item is the **task-wide effective-dimensionality** breadth study (entry 2, A8/C1)
  — which the non-orthogonal `SV`/CE11 findings now directly inform (interference is
  real, so a clean low-rank inventory is less likely). Backlog candidates: a fair
  digits-`0..n` SV baseline (to test presence *beyond a full operand recompute*),
  and a causal test of whether the `=` carries are read downstream (vs recomputed).
