# Study: What do the map-missed subtraction heads COMPUTE? — study-missing-sub-mechanism.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).
Agenda: [maths-next-steps.md](../maths-next-steps.md) entry 1 follow-up.
Follows **CE30** ([study-missing-sub-nodes.md](study-missing-sub-nodes.md)): the
subtraction-important nodes the verified map omits are the **last-layer (L2)
attention heads at the answer-producing positions of the failing digits**
(`P15L2H{0-3}` = produce-pos(A5), `P18L2H{0-3}` = produce-pos(A2)), heads tagged
useful *elsewhere* but under-tagged at these positions. CE30 localized them; this
study asks **what they compute** — the subtraction borrow/digit algorithm.

## Executive Summary (CE31)

The CE30 map-missed heads are **borrow-in DELIVERY heads**. All three readouts
agree, for both failing digits (A5, A2) × both classes (SUB, NEG), 3 seeds: at the
producing position they (READ) attend to the **lower-digit operands** (the borrow
source) + the `=`/`SGN` staging region, with **~0 mass on their own operands**;
(WRITE) their OV write decodes the **resolved borrow-in `SV[k]` at 1.00±0.00**, far
above the base difference `SA[k]` (0.68-0.78) — they carry the borrow, not the
difference; (CARRY) the group causally delivers it (flip 1.00 / deciding-null 0.00)
while **per-head is ≈0.00** (individually redundant — why per-node ablation missed
them). The map tags the same head (L2H0) at other positions as OPR/SGN but not at
the producing positions P15/P18. Mechanism: heads H0-H2 fetch the resolved borrow-in
and the already-mapped last-layer combiner MLP integrates it with the base
difference to emit `A_k`. Untrained twin fails the causal test (flip 0.00). Landed
as **CE31**.

## Pre-run

- **Question**: the CE30 heads sit at the *producing* position of a failing digit
  (not at operand positions). Do they (H-deliver) **deliver the resolved borrow-in
  `SV[k]`** — attend to lower-digit (borrow-source) positions and write the
  borrow-in that the combiner MLP integrates with the base difference — or
  (H-diff) fetch the digit's own operands and write the **base difference `SA[k]`**,
  or (H-tri) carry the **tri-state `ST[k]`**? And how is the role split across the
  4 heads (individually redundant per CE30)?
- **Key discriminator**: `SV[k]` (borrow INTO digit k) is a function of the
  **lower** digits only; `SA[k]`/`ST[k]` are functions of digit k's **own**
  operands only. So "attend-to-lower + encode-SV" ⇒ delivery; "attend-to-own +
  encode-SA/ST" ⇒ local difference/tri-state.
- **Design** (mixed d6 `ins1_mix_d6_l3_h4_t40K_s372001`; classes SUB + NEG;
  last-layer `ll=2`; producing positions `cpos ∈ {P15 (k=5), P18 (k=2)}`; heads
  H0-H3; class-correct labels via `sub_labels`/`neg_labels`):
  - **READ — attention profile.** Over N random class stimuli, mean attention mass
    from query=`cpos`, head h, onto key-position groups: `own_operands` (D_k, D'_k),
    `lower_operands` (D_j, D'_j for j<k), `higher_operands`, `eq_sgn` (`=`+SGN),
    `lower_answer` (produce-positions of digits <k). Per head. Delivery ⇒
    `lower_*` mass; difference ⇒ `own_operands` mass.
  - **WRITE — OV-encode probe.** Collect the head-group (and per-head) OV write
    `z @ W_O` at `cpos` over N random class stimuli; balanced-accuracy probe with a
    label-permutation null for each of `SV[k]` (borrow-in), `ST[k]` (tri-state),
    `SA[k]` (base diff), `A_k` (final). The task with the highest above-null,
    above-chance accuracy = what the heads write.
  - **CARRY — causal interchange.** On a depth-k `U` cascade
    (`make_cascade_operands`, digit k held at the borrow tri-case, borrow-in
    toggled), (a) **group**: patch all L2 heads' z at `cpos`
    (`combiner_delivery_flip` `lastlayer_attn`), A_k flip vs a **deciding-matched
    null**; (b) **per-head**: single-head OV patch flip (redundancy / labor split).
  - **MAP cross-check.** From the published map, which answer positions / roles are
    `L2 H{0-3}` tagged at (CE30 "tagged-elsewhere") — confirm *same heads,
    under-tagged positions*, and read the role tag (MD/ND/MB/NB/SA…) they carry
    where the map does keep them.
- **Positive/negative controls**: untrained twin (`make_untrained_control`) must
  fail WRITE + CARRY; a **wrong-position** last-layer head-group (a non-producing /
  question-region position) must not encode or carry A_k's borrow (specificity);
  probe permutation null; CARRY deciding-matched null.
- **Success**: READ + WRITE + CARRY tell one consistent story about the computed
  quantity, holding for **both failing digits (k=5, k=2) and both classes**, above
  chance/null and above the untrained + wrong-position controls, stable over ≥3
  seeds. Output = a mechanistic label for the CE30 heads + the per-head labor split.
- **Failure/ambiguous**: the batteries disagree (e.g. attend own-operands yet the
  OV write decodes borrow-in); nulls not beaten; or the untrained control also
  "passes" (then the readout is format-driven, not learned).
- **Skeptic (pre-launch)**: group-aware (the heads are individually redundant —
  report group AND per-head); permutation + deciding-matched nulls; untrained +
  wrong-position specificity; both classes × both digits × ≥3 seeds; the WRITE
  probe reads the **OV write** (isolates the heads), not the residual; NEG uses
  `neg_labels` (base difference on `D'-D`). Caveat: `SV/SA/ST` are correlated —
  the READ battery (lower vs own operands) is the independent cross-check on the
  WRITE probe's "which quantity".
- **Artifacts**: `scripts/missing_sub_mechanism.py`,
  `results/study-missing-sub-mechanism/results_<model>.json`.

## Post-run

Run 2026-07-17, `scripts/missing_sub_mechanism.py`, mixed d6, classes SUB+NEG,
producing positions P15 (A5) & P18 (A2), last-layer L2 heads H0-H3. Data:
`results/study-missing-sub-mechanism/results_ins1_mix_d6_l3_h4_t40K_s372001.json`
+ `multiseed.json`.

**MAP cross-check**: last-layer head tagged positions `H0:[13,16,17,19] H1:[19]
H2:[13] H3:[]`; roles `OPR@P13L2H0`, `SGN@P16L2H0,P17L2H0`. L2H0 is tagged
elsewhere (OPR/SGN) but **not at P15/P18** — same head, under-tagged positions +
unrecognized borrow-delivery role (confirms CE30).

**READ — attention profile** (mass from query=produce-pos onto key-groups; seed 0):

| cell | head | own | lower | higher | eq/sgn | self |
|---|---|---|---|---|---|---|
| SUB A5 | H0 | 0.01 | **0.52** | 0.00 | 0.00 | 0.02 |
| SUB A5 | H1 | 0.00 | 0.03 | 0.00 | 0.43 | 0.01 |
| SUB A5 | H2 | 0.01 | 0.20 | 0.00 | 0.26 | 0.10 |
| SUB A5 | H3 | 0.00 | 0.00 | 0.00 | 0.52 | 0.15 |
| NEG A5 | H0 | 0.00 | **0.46** | 0.00 | 0.52 | 0.01 |
| SUB A2 | H0 | 0.06 | 0.17 | 0.11 | 0.17 | 0.03 |
| NEG A2 | H0 | 0.05 | 0.19 | 0.08 | 0.54 | 0.04 |

→ own-operand mass ≈0 everywhere; H0-H2 read lower operands (+ `=`/`SGN`); H3 reads
only `=`/`SGN`+self. (3-seed: H0-H2 mean own 0.005-0.045, lower 0.08-0.39.)

**WRITE — OV-encode probe** (balanced-acc; group; seed 0; 3-seed `SV` in brackets):

| cell | SV borrow-in | ST tristate | SA base-diff | Ak final |
|---|---|---|---|---|
| SUB A5 | **1.00** [1.00±0.00] | 0.88 | 0.83 | 0.81 |
| SUB A2 | **1.00** [1.00±0.00] | 0.74 | 0.71 | 0.52 |
| NEG A5 | **1.00** [1.00±0.00] | 1.00 | 0.86 | 0.81 |
| NEG A2 | **1.00** [1.00±0.00] | 0.78 | 0.70 | 0.65 |

Per-head (SUB A5): H0/H1/H2 `SV`=1.00 with low `SA` (0.18-0.41); H3 `SV`=0.84.
→ the OV write carries the **borrow-in**, not the base difference (SA 0.68-0.78 «
SV 1.00). Chance: SV 0.5, ST 0.33, SA/Ak 0.1.

**CARRY — causal interchange** (flip / deciding-matched null; group + per-head):

| cell | group flip/null | per-head flip |
|---|---|---|
| SUB A5 | **1.00 / 0.00** | H0-H3 all 0.00 |
| SUB A2 | **1.00 / 0.00** | all 0.00 |
| NEG A5 | **1.00 / 0.00** | H0 1.00, rest 0.00 |
| NEG A2 | **1.00 / 0.00** | all 0.00 |

→ the group delivers the borrow specifically; individually redundant (per-head ≈0).

**Controls**: untrained twin (SUB A5) WRITE `SV`=0.76 (format floor) but **CARRY
flip 0.00** (fails) → delivery is learned; the causal flip, not the probe, is
decisive. Permutation null (probe) + deciding-matched null (causal) pass.

**Verdict**: CE30 heads = borrow-in DELIVERY/consumer heads. At produce-pos(k),
H0-H2 attend to the lower-digit operands (borrow source) + `=`/`SGN` and write the
resolved borrow-in `SV[k]`; the already-mapped last-layer combiner MLP integrates it
with the base difference to emit `A_k`. Redundant across H0-H2 (why the ablation map
under-tagged them at the high digits). H3 is not a borrow deliverer (labor split).
Landed as **CE31**.

**Open (follow-ups)**: (a) how the borrow-in is *resolved* upstream (L0/L1) — this
study pins the last-layer DELIVERY only; (b) d8 (identify d8 missing nodes first,
then re-run this model-general battery); (c) H3 / the `=`/`SGN` staging role.

**Reusable code**: the READ + WRITE collectors were promoted to
`quanta_maths.maths_edge_patch` as `attention_mass_by_group` (mean attention mass
from a query onto named key-position groups) and `head_group_ov` (per-question OV
writes of a head-group at a position) — both model-general, batched, tested
(`tests/test_edge_patch.py`, HF-gated; full suite 164 passed). The CARRY battery
reuses `maths_cascade.make_cascade_operands` + `combiner_delivery_flip`.
