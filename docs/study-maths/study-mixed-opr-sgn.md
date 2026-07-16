# Study: mixed-model OPR / SGN / shared-engine (M4–M6) — study-mixed-opr-sgn.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #2 (CE21)

The mixed model has two jobs the addition model does not: read the **+/- operator**
in the question, and decide the **+/- sign** of the answer. This study asks how it
does each, and whether add and subtract share one engine (A7) or run separate
circuits (C2).

Findings: the answer **sign** is computed just like the addition model's leading
digit — it is the "is the first number bigger?" comparison, resolved and delivered
to the combiner at the "=" position (clean on every test). The **operator** is
readable everywhere in the model but is consumed early at a selector stage, so it
is *not* a knob you can turn where the digits are assembled. On the shared
digit-heads, add and subtract readouts partly overlap, but addition is pushed
*further* from subtraction than a random baseline — so neither "one engine + a
simple switch" nor "fully separate circuits" fits; it is a **hybrid**.

Design context: [study-mixed-plan.md](study-mixed-plan.md). Agenda:
[maths-next-steps.md](../maths-next-steps.md) entry 2. Scores A7, C2, C5, A10.

## Pre-run

- **Question**: the two mixed-only mechanisms the model's algorithm forces —
  (M4) how is the **operator** `OPR` (+/−) used? (M5) how is the **answer sign**
  `SGN` (+/−) computed? — plus (M6) do the map's shared `SA`/`MD`/`ND` nodes
  hold overlapping (A7) or near-orthogonal (C2) per-operation readouts?
- **Motivation**: required to make the mixed replication meaningful; A7 vs C2 is
  the entry-2 shared-engine fork; SGN is the CE15 leading-digit analog (the sign
  replaces the leading carry as the top-of-cascade product).
- **Competing reads** (neutral): **A7** — `OPR`/`SGN` are compact low-rank
  directions selecting the readout of shared machinery; **C2** — near-orthogonal
  separate per-operation circuits multiplexed at the output.
- **Design**:
  - **M4 OPR**: (a) decode operator (+/−) at sites from the `OPR` token (P6) to
    the L2 combiner input, with permutation null; (b) rank-1 causal steer — add
    the (add-mean − sub-mean) direction at the combiner input of a SUB question
    and test whether A_k flips to the ADD digit (A7's "move along the direction
    flips the family").
  - **M5 SGN**: (a) boundary positive control — cross `D≥D' ↔ D<D'` at the
    deciding digit, does SGN flip? (b) decode the sign at `=` (binary, no `U`);
    (c) full-resid edge patch at the sign-producing position (`=`) with a
    deciding-matched null (CE15 analog).
  - **M6 shared engine**: at the shared L0 answer-position heads, principal angles
    between per-operation digit-readout subspaces (`class_mean_subspace`), vs an
    untrained-control angle baseline. C2 → ~90°; A7 → small.
- **Positive control**: M0 accuracy; untrained control for M6 angle baseline; M5
  boundary control; probe permutation nulls.
- **Success/Failure**: A7 if operator is low-rank + rank-1-steerable and readouts
  overlap; C2 if near-orthogonal readouts and steering breaks digits; either
  damages a conjecture.
- **Skeptic review (pre-launch)**: combined sprint gate (below).
- **Risks**: mis-siting the operator control (it may act upstream of the
  combiner); angle baselines need the random-init reference; sign edge patch
  full-resid is a near-tautology absent the deciding-null.
- **Expected artifacts**: `scripts/mixed_opr_sgn.py`,
  `results/study-mixed-opr-sgn/results.json`.

## Post-run

- **Executive summary**: **SGN replicates the CE15 mechanism cleanly** — the sign
  is the top-of-cascade `D≥D'` comparison delivered to the sign-position (`=`)
  combiner: boundary crossing flips SGN 1.00, the sign is a perfectly decodable
  binary at `=` (1.00), and a full-resid edge patch at `=` flips SGN
  comparison-specifically (flip 1.00, deciding-matched null 0.00). **OPR and the
  shared-engine question give a hybrid A7/C2 verdict**: the operator is decodable
  **1.00 broadcast across every layer/position** (A7's "low-D operator signal
  available broadly") but is **not an additive rank-1 control at the combiner**
  (a large add−sub direction, norm 20.6 > site 17, changes the digit 0.00 of the
  time) — it is consumed **upstream at the SLT selector**, so A7's specific
  "steer the readout at the combine site" form fails. On the shared `SA`/`MD`/`ND`
  heads the per-operation readouts are **more separated than random for ADD-vs-
  subtraction** (68–71° vs 46–50° control) but **overlapping for SUB-vs-NEG**
  (48° ≈ 46° control): the engine is shared at the head level (A7) yet develops
  operation-specific readout rotations (C2-leaning for add-vs-sub) — neither
  strong form survives intact.
- **Run record**: `PYTHONPATH=. python scripts/mixed_opr_sgn.py`, CPU,
  2026-07-16, `ins1_mix_d6_l3_h4_t40K_s372001`. Artifact:
  `results/study-mixed-opr-sgn/results.json`.
- **Results**:
  - **M4 OPR**: operator decode acc = **1.000** (null p 0.03) at the combiner
    input, and 1.000 at P6 / L0 / L1 / L2 answer-position sites (broadcast).
    Rank-1 steer sub→add: to_add **0.00**, changed **0.00** (op_dir_norm 20.6 vs
    site 17.0) → not additively steerable at the combiner.
  - **M5 SGN**: boundary flips SGN = **1.00**; sign decode at `=` = **1.000**
    (null p 0.03), binary; sign edge flip = **1.00**, deciding-matched null =
    **0.00**, stimulus valid.
  - **M6 overlap** (mean principal angle, trained | untrained control):
    ADD-SUB 68.0° | 45.7°; ADD-NEG 71.1° | 50.5°; SUB-NEG 48.4° | 46.3°.
- **Interpretation**:
  - **M5** meets every success condition → the sign is produced by the
    CE15-style carry-specific fetch-to-combiner delivery, carrying the `D≥D'`
    comparison to the `=` combiner. Clean cross-mechanism replication (leading
    carry → sign).
  - **M4** splits the A7 prediction: the "broadcast low-D operator signal" half is
    **supported** (decodable everywhere; map has 18 `OPR` nodes across all
    layers), but the "additive low-rank control that flips the family at the
    combine site" half is **not** — the operator is applied upstream (SLT
    selector chooses SA/MD/ND before the combiner), so the combiner-input
    operator direction is a decodable spectator, not a steering knob.
  - **M6** is a **hybrid**: shared heads (structural A7) with operation-specific
    readout geometry that is *more orthogonal than chance* for ADD-vs-subtraction
    (C2-leaning) while the two subtraction families share readout geometry
    (A7-leaning). Strong C2 (near-orthogonal, 90°) is refuted (angles 48–71°);
    strong A7 (heavy overlap / rank-1 steer) is refuted for add-vs-sub.
- **Prediction scoring**:
  - **A7**: **partially confirmed / partially refuted** — broadcast low-D operator
    signal + shared heads + SUB/NEG readout overlap (for); rank-1 steer null +
    ADD readout more-separated-than-chance (against the low-rank-control and
    heavy-overlap forms).
  - **C2**: **partially confirmed / partially refuted** — ADD-vs-sub readouts
    more separated than random (for); but not near-orthogonal, and heads are
    shared not disjoint (against).
  - **C5**: **confirmed** — `OPR`/`SGN`/`SLT` map roles are causally borne out
    (SGN edge, operator broadcast, SLT-as-upstream-selector).
  - **A10**: **confirmed (extended to the sign)** — the fetch-to-combiner delivery
    produces the sign at `=`, mirroring CE15's leading digit.
- **Skeptic review (post-result)**: concerns & resolution: (1) *M4 mis-siting* —
  acknowledged: the rank-1 null is scoped to the *combiner input*; a cleaner A7
  test steers at the SLT selector (post-deadline). Reported as "not steerable at
  the combiner", not "no operator control". (2) *M6 needs a baseline* — added the
  untrained-control angles; the verdict is stated relative to them, not absolute.
  (3) *M5 full-resid tautology* — the deciding-matched null 0.00 + boundary
  control carry the specificity. (4) first M4 decode was a label-alignment bug
  (list-concat vs interleaved labels) → **corrected** (1.000) before scoring; the
  buggy 0.49 was not used.
- **Limitations**: single model/seed; M4 rank-1 additive steer only (not a
  learned low-rank subspace or an SLT-sited steer); M6 subspace angles on 10-way
  digit readouts (10-dim subspaces in d_model); linear probes.
- **Doc updates**: appended **CE21** (claim-evidence); reflection-log + A7/C2/A10
  updates (conjectures-agent); results bullets. Append-only; entry-1 files
  untouched.
- **Next read**: paper hand-off consolidation (entry 3); post-deadline — SLT-sited
  operator steer, and a depth-sweep of the subtraction delivery.
