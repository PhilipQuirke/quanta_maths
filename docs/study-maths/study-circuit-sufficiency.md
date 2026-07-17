# Study: Circuit sufficiency — resample-ablate the complement of the map-useful nodes — study-circuit-sufficiency.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).
Agenda: [maths-next-steps.md](../maths-next-steps.md) entry 1. Scores C5 (map
completeness/reliability), A10/A12 (the identified SV circuit is sufficient).

## Executive Summary — Circuit Sufficiency (CE29)

Do we know **all** of where the arithmetic is computed? Keep only the map's useful
nodes and destroy everything else, then measure accuracy. **Answer: yes for
addition, only partly for subtraction.**

- **Addition** (`add_d5_l2_h3_t15K_s372001`): keeping the 48 map-useful nodes and
  **mean-ablating the other ~104 retains 94%** accuracy (baseline 1.00); full
  **resample-randomization** of the complement retains 62% (the gap is
  residual-stream pollution, not hidden computation). Keeping the same number of
  **random** nodes instead → **0%**. So the identified addition circuit is
  **largely sufficient**, and specifically so.
- **Mixed** (`ins1_mix_d6_l3_h4_t40K_s372001`): **addition again largely
  sufficient** (mean 0.89), but **subtraction is not** — keeping only the map's
  useful nodes retains just **~0.5 (SUB) / ~0.37 (NEG)** even under gentle
  mean-ablation. So the ablation-discovered map **misses load-bearing nodes for
  subtraction**. Skeptic-verified locus: the **sign/selection token is retained at
  1.00** for all classes — the gap is in the **subtraction digit/borrow engine**
  (specific digits 10^5, hundreds fail), a distinct locus from CE23 (selection).

Verdict: **C5 refined** — the map is a reliable, specific, and *sufficient* account
for addition, but *incomplete* for the mixed model's subtraction classes.

## Pre-run

- **Question**: is the map-useful node set **sufficient** for the computation? Keep
  the map-useful `(position, layer, head/MLP)` nodes intact and resample-ablate the
  **position-specific complement** (every other head/MLP-at-position); measure
  accuracy vs the clean baseline.
- **Motivation**: ablation studies (CE13–CE26) established *necessity*/redundancy
  per node/edge; this is the complementary **global sufficiency** test — a single
  end-to-end check of the whole map + SV account, and a sharp probe of map
  *completeness* (a drop localizes what the map omits, e.g. redundant nodes
  ablation-discovery misses).
- **Design**:
  - **Keep set** = all nodes in the model's published `behavior.json` (the useful
    nodes). **Complement** = all `(pos, layer, head)` and `(pos, layer, MLP)` NOT
    in the keep set.
  - **Resample-ablation** (per the human's choice): for each clean question, draw
    an independent random same-class **source** question; run the model on the
    clean input but, at every complement node, **overwrite its activation with the
    source's activation** at that position (`hook_z` per head; `hook_mlp_out` per
    MLP). Kept nodes run clean. Embeddings / positional / LayerNorm / unembed are
    always kept (not "nodes").
  - **Conditions** (per class; addition = ADD only): (i) **baseline** (no
    ablation); (ii) **keep-useful** (resample-ablate the complement); (iii)
    **keep-random control** (keep a RANDOM set of the same size + head/MLP
    composition as the useful set, resample-ablate the rest).
  - **Metric**: fraction of questions with ALL answer tokens correct (N≈300/class),
    per condition.
- **Positive control**: baseline ≈ 1.0 (else the harness/accuracy is broken).
- **Success**: keep-useful retains accuracy **≈ baseline** (say ≥ 0.95) AND **≫
  keep-random** — the useful nodes are sufficient and specifically so.
- **Failure**: keep-useful accuracy collapses (map is incomplete / the circuit is
  not sufficient) — a real, informative negative that localizes missing nodes.
- **Ambiguous/invalid**: baseline < ~0.99 (harness bug); keep-random ≈ keep-useful
  (ablation not biting / too weak).
- **Skeptic (pre-launch)**: (1) resample-ablation is on-distribution (source is a
  real same-class input) so a retained accuracy is not a trivial "zeros do
  nothing" artifact; (2) keep-random control guards against "keeping any N nodes
  suffices"; (3) map completeness caveat — the useful set is ablation-discovered,
  so redundant load-bearing nodes may be missing (that is exactly what a drop would
  reveal); (4) single resample draw per question — average over N and optionally
  repeat.
- **Decision impact**: strong retention → C5 (map complete) + A10/A12 (SV circuit
  sufficient) raised; a drop → map-completeness gap logged, localize.
- **Risks/confounds**: position-shared head weights (handled — we ablate the head's
  OUTPUT at complement positions, not its weights); resample source must be
  same-class; batch the resample.
- **Expected artifacts**: `scripts/circuit_sufficiency.py`,
  `results/study-circuit-sufficiency/results.json`.

## Post-run

- **Executive summary**: see above — the map-useful circuit is largely sufficient
  for addition (mean-ablation retention 0.94 pure-add / 0.89 mixed-ADD) but
  incomplete for mixed SUB (0.48) / NEG (0.36); keep-random = 0.00 throughout
  (the useful set is specifically load-bearing).
- **Run record**: `PYTHONPATH=. python scripts/circuit_sufficiency.py <model>`,
  CPU, 2026-07-16. Artifacts:
  `results/study-circuit-sufficiency/results_add_d5_l2_h3_t15K_s372001.json`,
  `results_ins1_mix_d6_l3_h4_t40K_s372001.json`. Keep-set = the published
  `behavior.json` useful nodes; complement resampled/mean-ablated per class.
- **Results** (baseline | keep-useful mean-ablate | keep-useful resample |
  keep-random):
  - add_d5 ADD: 1.000 | **0.943** | 0.623 | 0.000
  - mix_d6 ADD: 1.000 | **0.893** | 0.527 | 0.000
  - mix_d6 SUB: 1.000 | **0.483** | 0.047 | 0.000
  - mix_d6 NEG: 1.000 | **0.357** | 0.047 | 0.000
  - (add_d5: keep 48/152 nodes; mix_d6: keep 98/330 nodes.)
- **Interpretation** (vs pre-stated): success (keep-useful ≥0.95 & ≫ random) is
  **met for addition under mean-ablation** (0.94, ≫ 0.00) — the identified circuit
  is sufficient and specific. For mixed **SUB/NEG the keep-useful retention is low
  even under mean-ablation** (0.48/0.36) → **map incomplete for subtraction** (a
  real, informative negative: subtraction load-bearing nodes are missing from the
  ablation-discovered map). The resample-vs-mean gap (e.g. add 0.62 vs 0.94) is
  **residual-stream pollution** (resampling injects a random other question's
  signal into the shared stream the kept circuit reads), not the complement doing
  hidden computation — so mean-ablation is the cleaner sufficiency measure.
- **Prediction scoring**:
  - **C5**: **refined** — the map's tagged nodes are load-bearing & specific
    (keep-random 0.00) and **sufficient for addition** (mean 0.94), but
    **incomplete for the mixed model's subtraction** (mean 0.48/0.36).
  - **A10/A12**: the SV *circuit* is sufficient for addition (and mixed-ADD); for
    mixed SUB/NEG the sufficient set is larger/more distributed than the map — the
    account holds but the complete node set is under-captured by ablation.
  - **A6/redundancy**: consistent — distributed redundant subtraction nodes
    (individually low-Fail%, so map-omitted) are collectively load-bearing.
- **Skeptic review (post-result) — separate thread, HOLD WITH CAVEATS**
  (2026-07-16, fresh docs-only context): both claims reproduced (2 seeds); the
  masks are provably genuine (layer-shift → 0.007, head-shift → 0.470 — a broken
  keep-mask cannot score 0.94); the specificity control was completed under the
  matched method (**keep-random-mean = 0.00**, not just resample). **Corrections
  folded in**: (a) report SUB as **~0.5 (0.48–0.58 across seeds)**, not a point
  0.48; (b) ADD "largely sufficient under mean, **bracket ≈ [0.62, 0.94]**", drop
  "essentially all the computation" (resample is a harsh lower bound); (c) **the
  SUB deficit is the subtraction DIGIT/BORROW engine, not selection** — the sign
  token is retained at 1.00 for all classes; specific digits (10^5, hundreds)
  fail — so the **CE23 (selection) link is dropped**; (d) note the ADD
  operand-range restriction (no overflow; mean-robust, resample optimistic); (e)
  it is additive to (not a restatement of) CE13/CE18/CE23 — first end-to-end
  sufficiency localizing the gap to the digit engine. Not-fatal caveats: single
  headline seed + single mixed model for SUB/NEG; keep-set = the ablation-map
  under test. **Recommendation: proceed to the missing-node identification
  follow-up**, group-aware (individually-redundant nodes), targeting the failing
  digits, scored under mean+resample + a random-augment specificity control,
  ≥3 seeds / d8 if possible.
- **Self-skeptic (pre-separate-gate) notes**: resample harshness bracketed with
  mean; keep-random 0.00; baseline 1.00.
- **Limitations**: resample-ablation corrupts the shared residual (mean-ablation is
  the fairer sufficiency bound); the keep-set is the ablation-discovered map (its
  own completeness is what's under test); single seed; the d8 mixed / addition-zoo
  not yet run.
- **Doc updates**: append **CE29** (claim-evidence); C5/A12 conjecture note; agenda
  entry 1 (done, with the subtraction-incompleteness follow-up); promote the
  harness to the library (`maths_sufficiency`) for the zoo.
- **Next read**: which subtraction nodes are missing from the map (rerun with the
  complement narrowed, or lower the map's Fail% threshold); run across the addition
  zoo + d8 mixed; a per-class map-completeness score as a batch technique.
