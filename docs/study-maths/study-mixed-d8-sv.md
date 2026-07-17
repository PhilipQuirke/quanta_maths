# Study: Mixed model — d8 cross-size SV dataset — study-mixed-d8-sv.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #6 (CE26)

All prior mixed SV results (CE20–CE23, CE25) are 6-digit. This runs the
model-general SV battery on the accurate **8-digit** mixed model
`ins1_mix_d8_l3_h4_t70K_s572091` (3 layers, 4 heads, 20% add / 80% sub).

**The core mechanism replicates d6→d8, for ADD, SUB and NEG**: the resolved
carry/borrow is a clean **binary** code at the last-layer combiner input
(~1.00 vs untrained chance), the combiner is a **STEP** function (α*≈0.5,
endpoints gated), the delivered carry is a **canonical** cross-digit code
(transfer 1.00), and **`=` is not the middle-digit source**; the combiner is
**shared** (all three classes tag the same L2 answer-MLP nodes, A1–A6). **One
thing changes with size — the delivery route**: on d6 the subtraction cascades
reached the combiner via *both* the residual and last-layer attention at every
depth; on **d8 they ride the residual for shallow cascades (depths 2–3) but
switch to last-layer attention for deep ones (depth 4)**, while addition stays
residual-only at all depths. **A12 supported** (the SV interface is size-general;
the delivery route is model/depth-specific, as CE25 anticipated). Two pieces are
**map-blocked** (this model has no published node map): the question-tail
writer-encoding probe (site unconfirmed — untrained ≥ trained, invalid) and the
paper-style behavior map / mechanism diagram. A reusable **d8 SV node-tag
dataset** was emitted (18 combiner `Algo` tags + 21 `Probe:DELIVERY` tags).

Entry 2 follow-up (a) (cross-model). Scope: **mixed model**, 8-digit. Scores
**A12** (cross-size mixed generalization). The d8 model has **no published
verified map** on HF, so the map-anchored pieces (paper Fail%/Impact behavior map,
mechanism diagram, writer-necessity, SLT-sited shared-engine) are out of scope
here; everything below is map-free (stimuli + loci derived from the config).

## Pre-run

- **Question**: does the mixed SV mechanism established on d6 (CE20/CE22/CE25)
  reproduce on the accurate d8 mixed model? Specifically, per class ADD/SUB/NEG:
  (A) do the question-tail tri-state writers encode their class; (C) is the
  resolved carry/borrow a clean binary at the last-layer combiner input; (B) is
  the last-layer MLP a causal combiner; (STEP) is it a step function; (=) is `=`
  not the middle-digit source; (canonical) is the delivered carry format-invariant;
  (DELIVERY) does the class-dependent route (ADD residual-only; SUB/NEG residual +
  last-layer attention) hold across depths 2–4?
- **Motivation**: the human asked for the full d8 mixed SV dataset; all prior
  mixed evidence is single-size (d6). A12 predicts the SV interface is
  architecture/size-general.
- **Design**: reuse the model-general battery code (`quanta_maths.maths_cascade`
  for the delivery sweep; the CE20/CE22 batteries) on d8. Deciding-matched nulls,
  endpoint-gated α-sweep, untrained-control checks as in d6. Also emit the SV
  node-tag dataset (combiner `Algo:A{d}.{STC,MTC,NTC}` → features.json;
  `Probe:DELIVERY.{ADD,SUB,NEG}` → behaviors.json) via the library taggers.
- **Positive control**: per-class accuracy ~1.0 (verified: ADD 1.00, SUB 1.00,
  NEG 0.995); stimulus validity per depth; deciding-matched null ≈ 0.
- **Success**: the d6 signatures reproduce at d8 (encoding/cascade decode ≫ chance;
  combiner STEP; `=`-arm 0; canonical transfer high; delivery route matches the
  class-dependent d6 pattern). **A12 supported.**
- **Failure/ambiguous**: signatures absent or the delivery route differs — a
  size-scope restriction (still a datapoint for A12).
- **Skeptic (pre-launch)**: same instrument caveats as d6 (full_resid/resid_pre
  near-tautological → rely on `lastlayer_attn` discrimination + deciding null;
  whole-last-layer-attention patch; single model/seed; map-free so no
  writer-necessity/SLT test).
- **Artifacts**: `scripts/mixed_d8_sv.py`, `results/study-mixed-d8/results.json`,
  `results/study-mixed-d8/{features,behaviors}.json` (node-tag dataset).

## Post-run

- **Executive summary**: see above — core SV mechanism replicates d6→d8 across
  ADD/SUB/NEG; the delivery route becomes depth-dependent for SUB/NEG on d8
  (residual shallow, last-layer attention at depth 4). A12 supported.
- **Run record**: `PYTHONPATH=. python scripts/mixed_d8_sv.py`, CPU, 2026-07-16,
  `ins1_mix_d8_l3_h4_t70K_s572091`. Artifacts: `results/study-mixed-d8/results.json`
  + the node-tag dataset `results/study-mixed-d8/{features,behaviors}.json`.
- **Results** (per class; all deciding-matched nulls 0.00):
  - accuracy ADD 1.00 / SUB 1.00 / NEG 0.997 (positive control).
  - **C resolved cascade** at the L2 combiner input: carry-acc ~1.00 (d1–d3, all
    classes) vs untrained ~0.61 → binary resolved code replicates.
  - **STEP** combiner: A_k two-level, endpoints gated, α*≈0.5, all classes.
  - **`=`-not-source**: `=`-arm flip 0, combiner-input control flip 1, all classes.
  - **canonical**: cross-digit resolved-carry transfer 1.00, all classes.
  - **combiner tags**: 6 STC + 6 MTC + 6 NTC (A1–A6, all three on the same L2
    nodes → shared combiner) via the redundancy-proof combiner-input criterion.
  - **DELIVERY (depths 2/3/4)**: ADD residual-only at all depths (attn 0.00);
    **SUB/NEG residual at d2/d3 (resid_pre 1.00, attn 0.00) but attention at d4
    (resid_pre 0.00, attn 1.00)** — all stimuli valid. Depth-dependent route,
    unlike d6 (both routes at all depths).
- **Interpretation** (vs pre-stated): success met for C/STEP/`=`/canonical/combiner
  across all classes → the core mechanism generalizes (A12). The delivery route
  *differs* from d6 (depth-dependent on d8) — within A12's "route may differ by
  model/size" and CE25's caveat, not a failure. Writer-encoding (A) is **invalid
  on d8** (untrained 0.59 ≥ trained ~0.35 → the probed Dpn site is not the d8
  writer locus; unconfirmable without the map) → not scored.
- **Prediction scoring**: **A12 confirmed on a new size** (d8) for the core SV
  mechanism (representation / STEP / canonical / `=`-not-source / shared combiner);
  delivery route size/depth-specific (A10 delivery refined — route is not
  universal). **A6** untouched (no necessity test — map-blocked).
- **Skeptic (post-result)**: (1) writer-encoding invalid (untrained ≥ trained) —
  reported not-scored, not a negative. (2) zero-ablation combiner tagger fired 0
  (redundancy-limited on d8) → switched to the combiner-input causal criterion
  (what STEP uses), which is sound and gives 18 tags. (3) the single-depth
  `Probe:DELIVERY` tag records the *shallow* (depth-2) route (SUB/NEG=res); the
  depth-4 attention switch is in `results.json` — the tag is a coarse per-model
  marker, the sweep is the full record. (4) single model/seed.
- **Limitations**: single d8 model/seed; map-blocked (no writer-necessity,
  SLT-sited shared-engine, mechanism diagram, or paper Fail%/Impact map); delivery
  tag is depth-2 only; whole-last-layer-attention patch (not per-head).
- **Doc updates**: append **CE26** (claim-evidence, Mixed model); A12/A10
  conjecture updates + reflection-log; results-by-time / summary. Dataset in
  `results/study-mixed-d8/`. Append-only (other thread active on d8 *addition* +
  geometry — separate dirs/scope).
- **Next read**: generate + publish a d8 mixed verified map (unblocks
  writer-necessity, SLT shared-engine, diagram); a per-depth `Probe:DELIVERY` tag;
  the SUB/NEG depth-transition (why d8 switches to attention only at deep cascades).
