# Study: mixed-model SV replication (M0–M3) — study-mixed-sv-replication.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #1 (CE20)

The addition model's "SV" carry mechanism was fully worked out on the pure
addition models. This study asks whether the **same machinery appears inside the
mixed add/sub model** — a deeper 3-layer model — and whether it behaves the same
for all three kinds of question: addition (ADD), positive-answer subtraction
(SUB), and negative-answer subtraction (NEG). We reused the addition instruments
and built the missing negative-answer tooling first.

Findings: the **representation replicates cleanly for all three classes** — the
early "tri-state" writer nodes encode their carry/borrow class, and the resolved
carry/borrow shows up as a clean yes/no (binary) code exactly where the answer
digit is assembled (the trained model scores ~1.0; an untrained copy is at
chance). **How** that resolved carry is *delivered* to the assembler differs by
class: addition delivers it via the residual stream only (the inherited addition
circuit resolves it early), while both subtraction classes *also* deliver via
last-layer attention. Everything is carry/borrow-specific — a matched control
that changes unrelated digits does not move it.

Design context: [study-mixed-plan.md](study-mixed-plan.md). Agenda:
[maths-next-steps.md](../maths-next-steps.md) entry 2 (`maths` thread, parallel
window). Scores C5, A10, A12, A7/C2.

## Pre-run

- **Question**: does the addition SV interface (CE13 writer encoding → CE5
  last-layer-MLP combiner → CE6/CE7 binary resolved cascade → CE14 carry-specific
  delivery) reproduce on the accurate mixed add/sub model
  `ins1_mix_d6_l3_h4_t40K_s372001` (3 layers, 4 heads), across all three question
  classes ADD / SUB / NEG?
- **Motivation**: the paper revision needs the addition→mixed generalization of
  the SV mechanism; the mixed model is initialised from `add_d6_l2_h3_t15K`, so
  ADD should reproduce cleanest and the subtraction families test genuine
  extension. Reproduction is prioritised over new mixed-specific features.
- **Competing reads** (neutral): (i) the interface replicates unchanged across
  classes; (ii) it replicates for ADD only (inserted circuit) and the
  subtraction families use a different mechanism; (iii) the extra layer changes
  the delivery pathway (residual vs last-layer attention).
- **Design**: four batteries per class, reusing the addition instruments, at the
  3-layer-correct loci (writer = first layer L0 question tail; combiner = last
  layer **L2** answer-position MLP):
  - **A writer encoding (CE13)**: linear probe of the tri-state class
    (ST/MT/NT ∈ {0,1,U}) at the operand-2 (question-tail) site, L0; permutation
    null. Digits 1–3.
  - **B combiner causal (CE5)**: zero-ablate the L2 answer-position MLP on a
    single-step U stimulus; does A_k flip? Reported alongside the library
    `_combiner_is_causal` (carry/borrow stimulus). Digits 1–5.
  - **C resolved cascade (CE6/CE7)**: linear probe of the resolved carry/borrow-in
    (SV/MV/NV, binary) at the L2 combiner input; permutation null. Digits 1–3.
  - **D cascade-specific delivery (CE14)**: single-step U pair (carry/borrow-in to
    digit k toggled), patch the combiner input src→tgt via three arms —
    `full_resid` (whole L2 resid_mid, positive control), `resid_pre` (pre-last-layer
    residual = L0+L1 output), `lastlayer_attn` (all L2 heads' OV) — scored as A_k
    flip vs a **deciding-matched null** (both carry-in=0). k=2.
  - Class-correct labels: `sub_labels` (ADD/SUB), new `neg_labels` (NEG). NEG
    needed the library build (`neg_labels`, `neg_ntc_functions`, NTC/NT tags,
    class-aware `_combiner_is_causal`) — landed with tests before this run.
- **Positive control**: M0 per-class accuracy (must be ~1.0 to interpret a class);
  an untrained control (`make_untrained_control`) must fail A/C; D gates on
  stimulus validity (clean src A_k ≠ clean tgt A_k) and the deciding-matched null.
- **Success**: A/C decode ≫ null and ≫ untrained; D flips with deciding-null ≈ 0.
- **Failure**: A/C at untrained level, or D flips = null (non-specific).
- **Ambiguous**: zero-ablation B null under redundancy (known crude instrument) —
  reported, not treated as absence of a combiner.
- **Skeptic review (pre-launch)**: combined sprint gate — see Skeptic Review below.
- **Decision impact**: supports/【refines】 C5 (map roles on a new architecture),
  A10/A12 (interface generality onto 3 layers + borrow/neg tasks), A7/C2 (shared
  vs separate) at the representation level.
- **Risks / confounds**: multi-layer mis-probing (guarded: combiner = `last_layer`);
  class contamination (guarded: class-filtered generation + class-correct labels);
  NEG label correctness (unit-tested vs emitted digits); redundancy blur (class-
  level reads, untagged baseline); LN damping of single-head edges (use residual
  + joint arms).
- **Expected artifacts**: `scripts/mixed_map.py`, `scripts/mixed_sv.py`,
  `results/study-mixed-map/results.json`, `results/study-mixed-sv/results.json`.

## Post-run

- **Executive summary**: **The SV *representation* replicates cleanly across all
  three classes**; the *delivery pathway* is class-dependent. Writers encode the
  tri-state class (ST/MT/NT decode ~1.00 at digits 1–2 vs untrained ~chance), and
  the resolved carry/borrow-in is a clean binary code at the L2 combiner input
  (SV/MV/NV decode ~1.00 vs untrained ~0.6) — CE13 + CE6/CE7 reproduce for ADD,
  SUB and NEG. Delivery to the combiner is **carry/borrow-specific** (deciding-
  matched null 0.00 on every arm) and arrives via the **residual for all classes**
  (`resid_pre` flip 1.00); **last-layer attention additionally delivers it for
  SUB/NEG (1.00) but NOT ADD (0.00)** — the inserted addition circuit resolves
  earlier and rides the residual, whereas the freshly-learned subtraction cascades
  also use last-layer-attention delivery (the 2-layer CE14 picture). Whole-MLP
  zero-ablation of the combiner is redundancy-limited (clean only at scattered
  digits), as in the addition CE5.
- **Run record**: `PYTHONPATH=. python scripts/mixed_map.py` (M0),
  `PYTHONPATH=. python scripts/mixed_sv.py` (M1–M3), CPU, 2026-07-16, model
  `ins1_mix_d6_l3_h4_t40K_s372001`. Artifacts: `results/study-mixed-map/results.json`,
  `results/study-mixed-sv/results.json`. Library build tested in
  `tests/test_scaling_and_sub.py` (21 passed incl. HF mixed-model NEG).
- **Results**:
  - **M0** per-class accuracy (500 Qs each): ADD 1.000, SUB 1.000, NEG 1.000
    (positive control passes). Verified map (98 nodes): L0 question-tail writers
    `ST`(P9–P14)/`MT`/`GT` sharing locations; L0 answer-position `SA`/`MD`/`ND`
    sharing heads H1/H2; L1 selector `SLT` (H1); 6 L2 answer-MLP combiners
    (P15–P20). `OPR`@P6, `=`@P13, `SGN`@P14.
  - **A** tri-acc (trained | untrained-control@d2): ADD d1/d2/d3 = 1.00/1.00/0.44 |
    0.55; SUB 1.00/1.00/0.46 | 0.45; NEG 1.00/1.00/0.45 | 0.60. (d3 weaker — a
    writer-locus/position caveat; d1–d2 clean.)
  - **B** combiner-causal digits — U-stim zero-ablation: ADD [], SUB [4], NEG [];
    library carry/borrow stimulus: ADD [], SUB [1,5], NEG [1,2,3,4,5]. Whole-MLP
    zero-ablation is redundancy-limited (esp. ADD).
  - **C** resolved-cascade carry-acc (trained | control@d2): ADD 1.00/1.00/1.00 |
    0.63; SUB 1.00/0.99/1.00 | 0.58; NEG 1.00/0.99/1.00 | 0.60.
  - **D** (A2; all deciding-matched nulls = 0.00): `full_resid` flip 1.00 (all);
    `resid_pre` flip 1.00 (all); `lastlayer_attn` flip **ADD 0.00, SUB 1.00,
    NEG 1.00**. Stimuli valid (src≠tgt A2 in every class).
- **Interpretation** (against pre-stated conditions):
  - Success met for **A** and **C** in all three classes (decode ≫ null and ≫
    untrained) → CE13 writer-encoding and CE6/CE7 binary-resolved-cascade
    **replicate on the mixed model for ADD, SUB and NEG**.
  - **D** meets the carry-specific success (deciding null 0.00) with a class-
    dependent pathway: the resolved cascade rides the **residual** into the
    combiner for all classes; last-layer attention is an **additional** delivery
    route for the subtraction families only. This is read (iii) + a partial (ii):
    the interface replicates, but ADD (inserted) delivers residual-only.
  - **B** is ambiguous-by-instrument for ADD (zero-ablation redundancy-limited),
    not evidence against a combiner — C already shows the resolved carry is
    present and causal (D `full_resid`) at the L2 combiner input.
- **Prediction scoring**:
  - **C5** (map roles reliable on a new architecture): **confirmed** — the map's
    ST/MT/NT writers, SLT selector, and L2 combiner MLPs carry the predicted
    encodings/causal roles on the 3-layer mixed model.
  - **A10** (writers→resolved-carry-at-combiner interface): **confirmed at the
    representation level, refined on delivery** — the resolved cascade is present
    and causal at the combiner input for all classes; "last-layer attention
    delivers" holds for SUB/NEG but ADD delivers via the residual (the extra
    layer relocates delivery earlier).
  - **A12** (interface generalises across architecture/task-mix): **confirmed** —
    onto 3 layers/4 heads and onto the borrow (MV) and neg-borrow (NV) cascades.
  - **A7 / C2**: **untouched here** (shared-vs-separate is M4–M6); the map-level
    head-sharing is noted but scored in study-mixed-opr-sgn.
- **Skeptic review (post-result)**: *(combined sprint gate.)* Concerns raised &
  resolved: (1) *full_resid is tautological* — yes; its value is the deciding-
  matched **null = 0.00** (specificity) + it localises delivery when contrasted
  with the inert `lastlayer_attn` ADD arm; not counted as trained-structure
  evidence on its own (A/C carry that). (2) *untrained control also flips
  full_resid* — expected (replacing the whole input changes output); the negative
  control that matters is A/C at ~chance, which holds. (3) *B zero-ablation
  weakness* — flagged as instrument-limited, not scored as "no combiner". (4)
  *d3 encoding 0.44* — scoped as a writer-locus caveat, not a failure (d1–d2
  clean, both subtraction families agree). (5) *NEG labels* — unit-tested against
  emitted digits before use (`(SA−SV)%10`). No goalposts moved.
- **Limitations**: single mixed model, one seed; k=2 for delivery (not a depth
  sweep — the CE14 ≥2-depth bar not attempted here, this is single-step U
  delivery); zero-ablation combiner underpowered; d3 writer locus not pinned;
  representational probes are linear; delivery arms use LN-fair OV patching.
- **Doc updates**: appended **CE20** (claim-evidence); reflection-log + A10/A12/C5
  confidence updates (conjectures-agent); results-by-time / results-summary
  bullets. Append-only per the parallel protocol; entry-1 files untouched.
- **Next read**: OPR/SGN + shared-engine ([study-mixed-opr-sgn.md](study-mixed-opr-sgn.md));
  then the paper hand-off consolidation (entry 3). A depth-sweep of delivery
  (CE14 ≥2-depth bar) on the subtraction classes is a post-deadline stretch.
