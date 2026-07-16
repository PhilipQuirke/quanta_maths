# Study: Mixed model — SLT-sited A7-vs-C2 shared-engine test — study-mixed-shared-engine.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #4 (CE23)

The decisive add-vs-subtract "one engine or two?" test, run at the selector stage
the model's own map points to. Mixed #2 (CE21) hinted the operator is not a simple
knob at the combiner; here we test that directly.

Findings: the final assembler (the combiner) **is shared** — copying an addition
run's selector-stage state onto a subtraction run makes the model emit the correct
*addition* digit 96% of the time. But you **cannot** flip add↔subtract with a
single simple direction (a rank-1 "operator knob" does nothing, at either the
selector or the combiner), and no single "selector" head carries the choice. So
the operation is chosen by a **distributed, high-dimensional** transformation, not
a compact switch — arguing against the clean "shared engine + low-rank control"
picture (A7) and toward "the selection needs the full machinery" (C2), even though
the assembler itself is shared.

Entry 2 step (b). Scope: **mixed model** `ins1_mix_d6_l3_h4_t40K_s372001`.
Follows [study-mixed-opr-sgn.md](study-mixed-opr-sgn.md) (CE21: operator
broadcast-decodable but NOT additively steerable at the combiner — consumed
upstream). Scores A7, C2.

## Pre-run

- **Question**: is the operator a **low-rank control that selects the readout of a
  shared engine at the SLT selector stage** (A7), or does the add/sub selection
  require the full selector machinery / break digits (C2)? CE21 showed a rank-1
  steer at the *combiner input* (L2) does nothing; the map says the **SLT** head
  (L1 H1) "selects the S/M/N outputs based on OPR/SGN", so the control must act at
  or before L1.
- **Competing reads** (neutral): **A7** — a compact operator direction at the L1
  selector flips the SA↔MD readout for the same operands, preserving the digit
  computation; **C2** — no single low-rank direction flips the family cleanly, or
  steering yields garbage (separate circuits multiplexed only at output).
- **Design** (matched operands `a≥b`; ADD `a+b` vs SUB `a-b`; middle digit k=2,
  producing pos P18 where the map has SLT `P18L1H1`; selector layer L1):
  - **Arm 1 rank-1 steer**: `op_dir = mean_add − mean_sub` of `resid_post(L1)` at
    the producing position over matched pairs; add `op_dir` to a SUB run there;
    success if A_k becomes the **ADD digit** (digit k of `a+b`) — a rank-1 control
    flipping the family. Also record whether the output is a valid digit.
  - **Arm 2 SLT-head patch**: replace the SLT head `L1H1` z at the producing
    position with its ADD-run value on a SUB run (full downstream propagation);
    does A_k flip toward the ADD digit? Tests whether the selector head carries
    the operation selection.
  - **Arm 3 full-L1-resid patch** (positive control): patch the whole
    `resid_post(L1)` at the producing position ADD→SUB; A_k must flip to the ADD
    digit (the selector-stage state is decisive).
  - Both directions (SUB→ADD and ADD→SUB) reported; digit-discriminating pairs
    only (add-digit ≠ sub-digit).
- **Positive control**: Arm 3 must flip (selector-stage state decisive); M0
  accuracy.
- **Success**: **A7** if Arm 1 (rank-1) flips to the correct ADD digit at a high
  rate with valid digits; **C2** if Arm 1 fails/garbles while Arm 3 (full) flips.
  Arm 2 localizes to the SLT head.
- **Failure/ambiguous**: Arm 3 does not flip → instrument void (selector locus
  wrong).
- **Skeptic review (pre-launch)**: rank-1 mean-difference is one specific
  low-rank form; a null does not refute *all* low-rank control (a learned rank-r
  subspace could still work) — scope the A7 claim to "rank-1 additive at L1".
  Digit-validity guards against "flips to garbage" being counted as a flip.
- **Decision impact**: the decisive A7-vs-C2 verdict for the mixed shared engine;
  feeds the paper's shared-engine framing.
- **Risks/confounds**: wrong selector locus (Arm 3 control guards); operand-value
  confound (matched a,b); k=2 only.
- **Expected artifacts**: `scripts/mixed_shared_engine.py`,
  `results/study-mixed-shared-engine/results.json`.

## Post-run

- **Executive summary**: **Decisive against A7's low-rank-control form; the shared
  engine exists at the combiner but the operation selection is a distributed,
  non-low-rank L1 transformation.** The full selector-stage residual is causal —
  patching `resid_post(L1)` at the producing position ADD→SUB flips A_k to the
  correct **ADD digit 96%** of the time (so the shared L2 combiner produces the
  right digit for either operation given the L1 state). But a **rank-1 operator
  steer at L1 changes the digit 0%** (op_dir norm 20), and patching the **single
  SLT head** changes it only 22% and **never** to the correct ADD digit (0%). So
  the operator is not a compact additive control at the selector (as it was not at
  the combiner, CE21), and no single head carries the selection.
- **Run record**: `PYTHONPATH=. python scripts/mixed_shared_engine.py`, CPU,
  2026-07-16. Artifact: `results/study-mixed-shared-engine/results.json`.
  54 digit-discriminating matched pairs, k=2, selector L1H1.
- **Results** (flip-to-correct-ADD-digit rate): **full_l1 0.96**, rank1_steer
  **0.00**, slt_head **0.00** (slt_head changed-at-all 0.22; rank1 changed 0.00).
- **Interpretation** (vs pre-stated): Arm 3 (positive control) flips → the L1
  selector-stage state is decisive (instrument valid). Arm 1 (rank-1) fails → the
  operator is **not** a rank-1 additive control at L1. Arm 2 → the SLT head alone
  does not carry the selection. Net: **A7's "function-vector / low-rank control
  selects the readout" mechanism is refuted** at the selector (and combiner,
  CE21); the engine is **shared at the L2 combiner** (Arm 3 produces the correct
  add digit through the same downstream), but the operation-specific selection is
  a **distributed, high-dimensional** L1 transformation — leaning **C2** on the
  *selection mechanism* while retaining a shared *combiner*.
- **Prediction scoring**:
  - **A7**: **function-vector/low-rank-control form REFUTED** (rank-1 steer null
    at both combiner and selector; single SLT head does not select). The
    "shared machinery exists" half survives (shared combiner; shared SA/MD/ND head
    locations). → A7 lowered to **low** on the control mechanism, hybrid overall.
  - **C2**: **partially supported** — the operation selection needs the full
    distributed L1 state (not a compact switch); combined with M6 (ADD readout
    more separated than random) this leans C2 on selection. But the combiner is
    shared (against pure "separate circuits"). → C2 hybrid, partially up.
- **Skeptic review (post-result)**: (1) *rank-1 is one low-rank form* — a learned
  rank-r subspace could still steer; the claim is scoped to "rank-1 additive at
  L1/L2 does not select", which is what A7's function-vector prediction literally
  says. (2) *changed=0.00 for rank-1* — the ADD−SUB L1 difference is
  high-dimensional (Arm 3 full delta flips 0.96); the rank-1 mean-component is in
  a subspace the downstream is robust to. (3) Arm 3 control passes → locus valid.
  (4) k=2 only; matched operands guard the operand confound.
- **Limitations**: single model/seed; k=2; rank-1 (not rank-r subspace) steer;
  one selector head (L1H1) for Arm 2.
- **Doc updates**: append **CE23** (Mixed model); A7/C2 conjecture updates +
  reflection-log; results bullets. Append-only.
- **Next read**: paper hand-off (entry 3). Post-deadline: a learned rank-r
  operator-subspace steer (to bound how low-rank the control can be) and the
  SGN-side analog.
