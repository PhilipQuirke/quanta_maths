# Plan: Mixed-model SV replication (add & sub) — study-mixed-plan.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes),
[Experiment Agenda](../thor-document-rules.md#experiment-agenda).

> **Status: EXECUTED (2026-07-16).** M0–M6 ran; full three-class NEG library build
> landed with tests. Results: **CE20** (SV representation replicates across
> ADD/SUB/NEG; delivery class-dependent) + **CE21** (SGN = comparison→sign; OPR
> broadcast-but-upstream; hybrid A7/C2). Records:
> [study-mixed-sv-replication.md](study-mixed-sv-replication.md) (M0–M3),
> [study-mixed-opr-sgn.md](study-mixed-opr-sgn.md) (M4–M6). The plan below is the
> pre-registration; deviations are noted in the study notes (esp. Battery D
> re-sited to residual/last-layer arms for the 3-layer architecture; M4 operator
> steer re-sited finding).

This is the **planning / design doc** for [maths-next-steps.md](../maths-next-steps.md)
**entry 2** (owned by this `maths` thread, running in parallel with the other
`maths` thread's entry 1). It is not itself a study note: each numbered study
below gets its own `study-maths/study-mixed-<name>.md` pre-registration at
pick-up, per the [study-note rules](../thor-document-rules.md#study-notes). It
sequences those studies, states the library work each needs, and fixes scope.

Owns: `study-maths/study-mixed-*.md`, `scripts/mixed_*.py`,
`results/study-mixed-*/`. Router docs are **append-only per thread** during the
parallel window (add my CE / confidence blocks; do not rewrite entry-1's).

## 1. Goal and scope

**Goal**: replicate the established addition SV mechanism (CE13–CE18) on the
accurate mixed add/sub model `ins1_mix_d6_l3_h4_t40K_s372001`, extending the
library where subtraction needs it, and add the two mixed-only experiments the
model's algorithm forces — the **question operator `OPR`** (P6, `+`/`-`) and the
**answer sign `SGN`** (P14, `+`/`-`).

**Directive (human, 2026-07-16)**: *prioritize reproduction over new
exploration of mixed-specific features.* So the ranking front-loads re-running
the addition instruments on the mixed model's three question classes, and treats
`OPR`/`SGN` mechanism and the A7-vs-C2 shared-engine geometry as required-but-later.

**Conjectures in scope** (entry 2): [C5](../maths-conjectures-human.md#c5-the-paper-empirical-results-are-reliable)
(paper-map reliability on a new model/architecture),
[A10](../maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster)
/ [A12](../maths-conjectures-agent.md#a12-the-sv-interface-generalizes-across-model-sizes-and-the-implementation-tightens-as-n-grows)
(SV interface generality onto a different architecture and task-mix),
[A7](../maths-conjectures-agent.md#a7-mixed-models-are-one-engine-under-low-rank-oprsgn-control-not-parallel-circuits)
vs [C2](../maths-conjectures-human.md) (shared engine under low-rank `OPR`/`SGN`
control vs near-orthogonal separate circuits). New mixed-specific predictions
(OPR store/route, SGN = magnitude-comparison-to-sign) are logged in the study
notes and, if they earn it, promoted to the agent conjecture file after the gate.

**Explicitly out of scope** (belongs to entry-1 / other thread or is deferred):

- The **compounding-locus / A11** fork (deep `...999` relay vs L1-read, CE9/CE10/CE17
  Battery-L). Entry-1 owns it; I reuse the *SV-interface* instruments (delivery,
  combiner form, source, necessity), not the compounding-locus assays.
- **New training runs** (paused, per the agenda). Everything here runs on existing
  HF artifacts + CPU/Colab forward passes.
- **Neuron-level combiner decomposition (B2)** and **digit-embedding causal use
  (B1)** — post-deadline.

## 2. The mixed model (verified plumbing)

`ins1_mix_d6_l3_h4_t40K_s372001`: initialised from addition model
`add_d6_l2_h3_t15K` then trained on 6-digit mixed add/sub (`perc_sub≈80`,
`perc_add≈20`). `n_layers=3` (L0/L1/**L2**), `n_heads=4`, `n_digits=6`,
`n_ctx=22`. Accurate on all three classes (assumed; **Study M0 verifies**).
See [mixed_model.md](../mixed_model.md).

Token layout (verified against `MathsConfig`):

| Tokens | Positions |
| --- | --- |
| `D5..D0` (operand 1) | P0–P5 |
| `OPR` (`+`/`-`) | **P6** |
| `D'5..D'0` (operand 2) | P7–P12 |
| `=` | **P13** |
| `SGN` (`+`/`-`, = `A7`) | **P14** (produced at P13, the `=`) |
| `A6..A0` (answer digits) | P15–P21 (`A_n` produced at `pos(A_n)-1`) |

**Critical architecture note (multi-layer)**: the addition studies used
2-layer wiring (writer = L0, consumer + combiner = L1). Here the model has **3
layers**; frame everything layer-generally (writer / consumer / combiner) and
resolve the combiner to the **last layer L2** (`last_layer(cfg)`), never a
literal L1. `maths_probe.site_hook_and_pos(..., layer=last_layer(cfg))` and the
`maths_batch` combiner tagger already take an explicit/depth-relative layer — the
migration fixed this exact bug. Per [mixed_model.md](../mixed_model.md) the
answer-digit base task (`SA/MD/ND`) is computed by **L0** heads at the
answer-producing position (e.g. `P18L0H1`+`P18L0H2` for A2), with class-specific
consumer heads at L1 and the combiner MLP at L2 — read the real wiring from the
map, do not assume it.

The three question classes (peers, per [mixed_model.md](../mixed_model.md)
Hypothesis 2):

| Class | Definition | `SGN` | Digit sub-tasks | Cascade |
| --- | --- | --- | --- | --- |
| **ADD** | `OPR=+` | `+` | `SA`,`SC`,`ST` (`SS` optimised out) | carry (`SV`) |
| **SUB** | `OPR=-`, `D ≥ D'` | `+` | `MD`,`MB`,`MT` (`MZ` optimised out) | borrow (`MV`) |
| **NEG** | `OPR=-`, `D < D'` | `-` | `ND`,`NB`,`NT` (`NZ` optimised out) | neg-borrow (`NV`) |

## 3. Reflection: which addition experiments to replicate

Triaging CE1–CE18 by value-for-the-paper-deliverable and by cost. The paper
revision needs (a) the SV mechanism and (b) the latent representation of
intermediate results ([entry 3](../maths-next-steps.md#3-paper-hand-off-consolidation-and-referee-checkpoint)),
so the SV-mechanism chain (CE13→CE14→CE15→CE16→CE17, resting on CE5) is the
priority; representation-geometry (CE6/CE7, some CE11/CE12) is the secondary
deliverable; the compounding-locus line (CE8/CE9/CE10) is entry-1's.

| Addition claim | What it established | Mixed-model action | Priority |
| --- | --- | --- | --- |
| **CE13** (node-output-encoding, C5 step 1) | map-named `ST`/`SC` nodes encode class + single-step U-resolution; `SA` L0 heads don't write the digit | **Replicate per class**: capture the mixed map; probe `ST`/`MT`/`NT` writers encode their tri-state; check the base heads don't pre-write the digit | **P1** |
| **CE5** (combiner-vs-conduit) | U-combiner = answer-position last-layer MLP | **Replicate**: STC/MTC/**NTC** causal combiner at the L2 answer MLP, all three classes | **P1** |
| **CE14** (SV-compounding, C5 steps 2–4) | carry-specific consumer-head→combiner **edge delivery**, ≥2 depths, deciding-matched null 0.00 | **Replicate + mirror**: carry/borrow-specific edge delivery to the L2 combiner for ADD, then SUB, then NEG | **P1** |
| **CE17** (compounding-arithmetic) | combiner is a **STEP** function (α*≈0.75), on-manifold sweep | **Replicate**: step-combiner endpoint-gated α-sweep per class | **P2** |
| **CE16** (SV-implementation) | canonical resolved-carry **message**; **source = ST cluster, never `=`**; head-pair **class-necessary** | **Replicate (ADD), mirror (SUB/NEG)**: message/source/necessity batteries | **P2** |
| **CE15** (leading-digit, C5 step 5) | leading digit via edge delivery to the **sign-position** combiner | **Reframed as the SGN study (M5)**: in the mixed model the sign position resolves the `D≥D'` comparison — the analog of the leading carry, now producing `SGN` | **P2 (new)** |
| **CE6/CE7** (no dedicated `{0,1,U}` symbol; binary, resolved ~last layer) | representation geometry of the resolved carry | **Replicate (light)**: is the borrow / neg-borrow also binary-resolved with no dedicated `U` symbol? Serves deliverable (b) | **P3** |
| **CE11/CE12** (`ST` position-specific; `SV` split layout at `=`/answer) | representation binding | **Optional**: only the `SGN`/`SV`-at-`=` slice, folded into M5 if cheap | **P4** |
| **CE1** (digit embeddings near-isotropic, weak circular order) | weights-only embedding geometry | **Optional quick calibration** (one forward pass): confirms the loader + probe on this model; low deliverable value | **P4** |
| CE2 (pair-sum) | `instrument failure`; A2 untested | **Skip** (do not inherit a broken assay) | — |
| CE3/CE4 | make-carry heads; superseded by CE5/CE13 | **Covered** by the search functions + M1; no standalone study | — |
| **CE8/CE9/CE10** (attention routing, deep-cascade localization, edge path-patch) | compounding-locus fork | **Skip — entry-1 owns it** | — |
| **CE18** (cross-size SV) | role skeleton + step combiner d5→d13 | **Not size — the mixed generalization IS the cross-axis test here**; the A12 "different architecture/task-mix" leg is exactly M1–M4 | (subsumed) |

## 4. Library gap analysis

The code-migration work already built substantial subtraction support (in the
`quanta_maths` library — `sub_labels(operation=MINUS)`, `sub_mtc_functions`, and
the subtraction-aware combiner/probe paths, with tests under `tests/`). What
**exists** vs what **must be built** for the mixed model:

**Exists and reusable** (validated on pure `sub_d6`/`add_d10`, tests in
`tests/test_scaling_and_sub.py`):

- `maths_probe.sub_labels(a,b,nd,operation=MINUS)` — SA/ST/SV **borrow cascade**
  (positive-answer, `D≥D'`).
- `maths_search_sub`: `sub_mt_functions` (MT tri-state), `sub_gt_functions`
  (`GT` = `D≥D'`, drives the sign — the M5 backbone), `sub_md_functions`,
  `sub_mb_functions`, `neg_nd_functions`, `neg_nb_functions`,
  **`sub_mtc_functions`** (MTC combiner, MLP at the answer position).
- `maths_search_mix`: **`opr_functions`** (OPR), **`sgn_functions`** (SGN) node
  filters — the M4/M5 detectors.
- `maths_probe.site_hook_and_pos(..., layer=)`, `first_layer`/`last_layer` —
  depth-correct on 3-layer models (bug fixed).
- `maths_edge_patch` (OV path, LN-fair edge patch, pattern patch), `maths_stats`
  (Wilson/mean CI), `maths_batch._combiner_is_causal` / `tag_stc_nodes`
  (operation-aware: STC for PLUS, MTC for `perc_sub==100`).
- CE16/CE17 primitives in `scripts/sv_implementation.py` /
  `scripts/compounding_arithmetic.py` (carry-axis, LN-fair projection,
  on-manifold α-sweep) — copy the harness into `scripts/mixed_*.py`.

**Must build / extend** (each ships with a positive+negative control test, per
the migration discipline):

1. **`neg_labels` (or `sub_labels` NEG branch).** Confirmed gap:
   `sub_labels(100,201,MINUS)` returns ten's-complement digits `[9,0,9,…]`, not
   the `-101` answer digits `{1,0,1}`. NEG answer digits come from `D'−D`; add
   labels for the **negative-answer** digit (`ND`), borrow (`NB`), neg-cascade
   (`NV`), computed on the swapped operands. Unit-test mirrors the `sub_labels`
   borrow test.
2. **`neg_ntc_functions` (NTC combiner).** There is `add_stc_functions` (STC) and
   `sub_mtc_functions` (MTC) but **no NTC**. Add the negative-answer combiner
   (parallel of MTC) with a `MathsTask.NTC_TAG`; wire class-aware selection into
   `maths_batch.tag_stc_nodes` (currently PLUS-vs-MINUS only, defaults a mixed
   model to PLUS).
3. **Class-aware stimulus builders + per-class accuracy.** Helpers to force ADD
   / SUB (`D≥D'`) / NEG (`D<D'`) questions and report per-class accuracy — the
   **positive control** gating every mixed study. `_combiner_is_causal` needs an
   explicit `operation` + class-appropriate borrow/neg stimuli (it currently
   picks operation from `perc_sub==100`, wrong for a `perc_sub=80` mixed model).
4. **Borrow / neg carry-axis extraction.** CE16/CE17's `carry_axis` is the ADD
   resolved-carry direction; add the borrow-in (`MV`) and neg-borrow-in (`NV`)
   class-mean axes at the combiner input for the SUB/NEG batteries.

These are surgical parallels of existing code, not new method classes — keeping
this a **replication**.

## 5. Ranked study plan

Each study: `study-mixed-<name>.md` pre-reg + `scripts/mixed_<name>.py` +
`results/study-mixed-<name>/results.json`. Combined skeptic pass (pre+post) per
study, per the sprint process. Evidence-integrity rule: **every headline number
computed in the committed script and written to `results.json`**, never
hand-copied.

### M0 — Calibration & map capture (prerequisite; cheap)

- **Question**: is the model accurate on ADD/SUB/NEG separately, and what does
  its verified map name for each class?
- **Do**: `load_maths_model_from_hf`; per-class accuracy (≥ several hundred Qs
  each) as the standing **positive control**; download + parse the model's
  `behaviors.json`/`features.json`/`maths.json` (only `_train.json` is cached —
  the maps need a pull, HF auth confirmed as `PhilipQuirke`). Emit a node
  registry: `ST/SC/OPR/SGN` (add), `MT/MB/GT` (sub), `NT/NB` (neg) writers;
  candidate consumer heads at answer positions; `Fail%` combiner MLPs at L2.
- **Positive control**: accuracy ≥ ~0.99 per class (else the class is
  `underpowered`, not interpretable). **Fail**: a class the model can't do →
  descope that class.
- **Scores**: C5 (does the map name the roles on this architecture at all).
- **Artifacts**: `results/study-mixed-map/` node registry JSON.

### M1 — ADD SV interface replicates inside the mixed model  *(highest value)*

- **Question**: does the addition SV signature (CE13 encoding → CE5 combiner →
  CE14 carry-specific edge delivery) reproduce on the **ADD** class of the
  3-layer/4-head mixed model?
- **Instruments (reused)**: node-output encoding probe (CE13); STC combiner
  causal check at **L2** (CE5); carry-specific consumer-head→L2-combiner **edge
  patch** on `blocks.2.ln2.hook_normalized` with the **deciding-matched null**
  (CE14).
- **Positive/neg controls**: `make_untrained_control` fails the probe/patch;
  self-patch null = 0.00; untagged-head baseline for ablation.
- **Success**: encoding ~1.00, combiner causal, carry-specific edge flip at ≥2
  depths (null 0.00). **Fail/partial** scored explicitly (delivery
  sufficient-not-necessary is the expected CE14 texture).
- **Scores**: C5, A10 core, A12 (architecture/task-mix transfer).
- Because the model was **initialised from an addition model**, ADD is where
  reproduction should be cleanest — the sharpest single test of the paper's
  addition→mixed generalization claim.

### M2 — SUB (positive-answer) borrow mirror

- **Question**: does the same interface carry the **borrow** cascade (`MT`
  writers → MTC combiner → borrow-specific edge delivery) for SUB (`D≥D'`)?
- **Instruments**: `sub_labels` borrow ST/SV; `sub_mtc_functions` MTC; the M1
  edge patch with a **borrow**-deciding-matched null and the **MV** carry-axis.
- **Success/Fail/controls**: as M1, borrow-specific.
- **Scores**: C5, A10/A12 (interface generality onto a mirrored task); A7
  (same wiring reused) vs C2 (separate borrow circuit) — first evidence.

### M3 — NEG (negative-answer) family  *(needs library build 1–2)*

- **Question**: does the interface extend to the **NEG** family (`ND`/`NB`/`NV`
  → **NTC** combiner) for `D<D'`?
- **Blocked on**: `neg_labels` + `neg_ntc_functions` (§4.1–4.2) landing green.
- **Instruments/controls**: M1/M2 batteries with NEG labels, NTC combiner, NV
  axis, neg-deciding null.
- **Scores**: C5, A10/A12; and A7-vs-C2 for the third class.
- If the build slips the deadline, NEG is the **first thing cut** to stretch
  (M1+M2 already carry the addition→subtraction generalization).

### M4 — `OPR` store & route  *(new, required)*

- **Question (neutral)**: how is the operator represented and used? Two live
  reads: **(A7)** `OPR` is a compact low-rank direction, available early/broadly,
  that *selects* the add/sub readout of shared machinery; **(C2)** operator
  routing gates between near-separate circuits.
- **Do**: locate `OPR` heads via `opr_functions` (attend P6); probe operator
  decodability + rank across positions/layers; **causal** test — steer the
  operator direction at the combiner input (reuse CE16 `steer_along_axis`) and
  ask whether the answer flips add↔sub *without destroying digit content* (A7)
  or breaks digits (C2).
- **Positive control**: a known operator-flip stimulus (`+`↔`-`) moves the
  answer; untrained control shows no low-rank operator direction.
- **Success**: a low-rank steer flips class-readout, digits preserved (A7-lean),
  or the opposite (C2-lean) — either way a conjecture takes damage.
- **Scores**: A7, C2, C5.

### M5 — `SGN` = magnitude comparison → sign  *(new, required; CE15 analog)*

- **Question**: how is the answer sign computed? Hypothesis (mixed_model.md H3):
  at `=` (P13, producing `SGN`@P14) the model resolves `D≥D'` (a `GT`
  borrow-cascade-to-the-top) into the sign. Is this the **same carry-specific
  edge-delivery-to-combiner mechanism as CE15's leading digit**, delivering a
  *comparison result* to the sign-position combiner?
- **Instruments**: `sub_gt_functions` (GT), sign-position (P13→P14) edge patch
  into the L2 combiner, deciding-matched null on the deciding comparison digit
  (the analog of CE15's deepest-chain). Representation slice (CE12): is `SGN`
  a resolved binary at `=` with no dedicated `U` symbol (CE7)?
- **Positive control**: a `D≥D'`↔`D<D'` toggle at the deciding digit flips
  `SGN` cleanly.
- **Success**: sign resolves via the same fetch-to-combiner edge (CE15 mirror)
  carrying the comparison. **Scores**: C5, A10 (mechanism generality to the sign),
  A7 (`SGN` control direction).

### M6 — Shared engine vs separate circuits (A7 vs C2)  *(replication-plus; lowest)*

- **Question**: on the polysemantic `SA/MD/ND` nodes the map names (e.g.
  `P18L0H1`+`P18L0H2`), how much do the per-operation readouts overlap, and does
  a compact `OPR`/`SGN` control direction select among them?
- **Instruments**: per-operation readout subspaces + principal angles
  (`maths_probe` subspace-angle tool); low-rank control-direction selection
  (M4/M5 axes).
- **Success**: **C2** predicts near-orthogonal per-operation subspaces; **A7**
  predicts heavy overlap steered by a low-rank control. Either way a conjecture
  is damaged. **Scores**: A7, C2.

## 6. Cross-cutting discipline

- **Positive controls / power honesty**: every study gates on M0 per-class
  accuracy and an in-assay positive control; a flat result without a passing
  positive control is `invalid`/`underpowered`, never `negative`. Redundancy is
  the norm — necessity tested at the **class** level (paired/grouped ablation)
  over an **untagged-node baseline**, not per single node.
- **Skeptic gate**: one combined pre+post pass per study in a separate thread
  that rehydrates only from docs; findings + resolution recorded in the study
  note's Skeptic Review sections before conjecture/agenda updates.
- **Evidence integrity**: committed `scripts/mixed_*.py` → `results/study-mixed-*/results.json`;
  notes cite JSON, never hand-typed numbers.
- **Doc order after each study**: study note → append CE block (claim-evidence) →
  results-by-time/summary/synthesis → conjecture scoring → agenda rerank —
  **append-only per thread** during the parallel window; on any dirty-file
  collision in `git status`, assume it is entry-1's and leave it.

## 7. Deadline triage (sprint mode, hand-off ≥ T-12h)

- **Minimum shippable** (addition→mixed generalization claim): **M0 + M1**
  (ADD SV replicates in the 3-layer mixed model) + **M2** (borrow mirror). This
  alone answers entry 2's first coupled question and feeds entry 3.
- **Strong**: + **M4/M5** (OPR store/route, SGN = comparison→sign) — the
  mixed-only mechanism the paper must describe.
- **Stretch**: **M3** (NEG family, needs the library build) and **M6** (A7-vs-C2
  overlap). NEG and M6 are the first cuts if time is short.

## 8. Risks, confounds, open questions

- **Multi-layer mis-probing** — the single biggest trap; always resolve the
  combiner to `last_layer(cfg)` = **L2**, never literal L1. Guarded by the
  migration's bounds-check, but assert it in every `mixed_*` script.
- **Class contamination** — probing the borrow cascade on NEG questions (or vice
  versa) silently mixes label families. Every collector must filter to the exact
  class; M0 accuracy is per class.
- **NEG label correctness** — the `neg_labels` build must be unit-tested against
  worked examples (`100-201=-101` → digits `1,0,1`) before any NEG probe is
  trusted; a wrong label would manufacture a false null.
- **Redundancy blur** — CE18 showed redundancy is intrinsic (not size-slack); the
  3-layer mixed model may be *more* redundant, so lean on class-level
  sufficiency+specificity+replication, not single-node necessity.
- **`OPR`/`SGN` may not be low-rank** — A7's cheap-switch story could fail; M4/M6
  are designed so a C2-style result is equally publishable.

### Open questions for the human (before executing)

1. **Go-ahead to run M0** (load the model, per-class accuracy, pull the maps)?
   The session opened with "take no actions"; this plan MD is the requested
   deliverable. M0 is the natural first execution step and de-risks everything.
2. **NEG scope**: build `neg_labels`+`NTC` now (M3 in-scope), or ship
   add+sub (M1/M2) first and treat NEG as stretch given the deadline?
3. **Skeptic model/thread** availability for the combined gate during the sprint.
