# Study: Mixed model — ≥2-depth carry/borrow delivery sweep (SUB/NEG, ADD) — study-mixed-delivery-depth.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #5 (CE25)

Mixed #1 (CE20) showed the resolved carry/borrow reaches the digit-assembler, and
that *how* it gets there differs by question type — but that was a single-step
(depth-1) test. This study drives a **deep** carry/borrow chain (the carry has to
ripple 2, 3, or 4 digits up) and checks the delivery route at each depth.

Findings: **the class-dependent picture holds at every depth (2, 3, 4).** For
**addition**, the resolved carry reaches the assembler purely through the residual
stream — last-layer attention plays no part (swapping only the last-layer attention
message changes nothing). For **both kinds of subtraction**, the resolved borrow is
delivered *both* through the residual and through last-layer attention. Every
effect is carry/borrow-specific (a matched control that keeps the carry the same
but changes other digits moves nothing), and an untrained copy delivers nothing.
So the class-dependent delivery route is a genuine multi-digit-cascade property,
not a one-step artifact. The reusable sweep is now in the library
(`quanta_maths/maths_cascade.py`) so it can be run across other models, where the
route may differ.

Entry 2 follow-up (i). Scope: **mixed model** `ins1_mix_d6_l3_h4_t40K_s372001`.
Builds on [study-mixed-sv-replication.md](study-mixed-sv-replication.md) (CE20).
Scores A10 (delivery), depth-generality. **Reusability**: the deep-cascade builder
and the delivery sweep are written model-general and promoted to `quanta_maths`
(`maths_cascade.py`) so the sweep can be run across the model zoo (delivery may
differ by model).

## Pre-run

- **Question**: Mixed #1 (CE20) established single-step (k=1, U-digit) delivery of
  the resolved carry/borrow to the L2 combiner, and found the pathway
  class-dependent (ADD residual-only; SUB/NEG residual + last-layer attention).
  Does this hold across **cascade depths k=2,3,4** — i.e. when the deciding digit
  is 2–4 positions below the read digit — for each class?
- **Motivation**: the addition CE14/CE15 bar for "delivery" was ≥2 genuine depths
  with a deciding-matched null; CE20's single-step result did not clear it. The
  paper's SV claim needs the delivery to be a real multi-digit cascade, not a
  one-step artifact. Also the first mixed study whose code is built for cross-model
  reuse.
- **Competing reads** (neutral): (i) the CE20 class-dependent pathway holds at all
  depths (ADD residual-only; SUB/NEG residual + attention); (ii) attention
  delivery for SUB/NEG only appears at shallow depth and residual takes over deep
  (or vice versa); (iii) delivery degrades with depth (a shallow-only mechanism).
- **Design** (per class ADD/SUB/NEG; deciding digit fixed at units d=0; read digit
  = depth k ∈ {2,3,4}; combiner = last layer L2 at the read digit's producing
  position):
  - **Deep-cascade stimulus** `make_cascade_operands(cfg, cls, read_digit,
    carry_in, variant)`: deciding digit 0 generates (carry_in=1) or not
    (carry_in=0) the cascade; digits 1..k are tri-state **U** (propagators) so the
    carry/borrow ripples up; digits above k are definite and set to control the
    class (ADD unconstrained; SUB D>D'; NEG D<D') and to stop the chain (sign
    stable). A_k then reads 9↔0 with the deciding carry.
  - **Delivery arms** at the read digit's combiner input (`recv_layer=L2`):
    `full_resid` (whole resid_mid — positive control), `resid_pre` (pre-last-layer
    residual = does it ride the residual?), `lastlayer_attn` (all L2 heads' OV =
    does last-layer attention deliver?).
  - **Metric**: per (class, depth, arm), the A_k flip rate on real pairs
    (carry_in 1→0) vs a **deciding-matched null** (both carry_in=0, different
    non-carrying deciding operands). Carry/borrow-specific delivery ⇒ real flip
    high, null ≈ 0.
- **Positive control**: stimulus validity per depth (clean A_k(carry 1) ≠
  A_k(carry 0)); `full_resid` must flip (combiner input causal); untrained control
  (arms should not deliver a learned carry). M0 per-class accuracy.
- **Success**: SUB/NEG show carry/borrow-specific delivery (real flip high, null 0)
  at **≥2 depths** on the `resid_pre` and/or `lastlayer_attn` arms; ADD shows it on
  `resid_pre` (residual) but not `lastlayer_attn` — i.e. the CE20 class-dependent
  pattern holds with depth.
- **Failure/ambiguous**: stimulus invalid at a depth (can't build the chain);
  `full_resid` doesn't flip (instrument void at that depth); flip == null
  (non-specific).
- **Skeptic review (pre-launch)**: (1) `full_resid`/`resid_pre` are near-tautological
  for whole-vector patches — the **deciding-matched null** carries specificity, and
  the `lastlayer_attn` vs `resid_pre` contrast carries the pathway attribution.
  (2) deciding fixed at units → the "depth" is the read distance; state that
  clearly. (3) chain must not flip the sign (stops above read digit) — assert.
  (4) redundancy: don't read a single-head null as "no delivery".
- **Decision impact**: A10 delivery scored at depth on mixed; confirms/【refines】
  CE20's class-dependent pathway; produces the reusable cross-model sweep.
- **Risks/confounds**: multi-layer loci (combiner=last layer); class contamination
  (class-correct builders); LN damping of the attention arm (report vs the
  residual arm, not in isolation).
- **Expected artifacts**: `scripts/mixed_delivery_depth.py`;
  `quanta_maths/maths_cascade.py` (promoted) + tests; `results/study-mixed-delivery-depth/results.json`.

## Post-run

- **Executive summary**: the CE20 class-dependent delivery pathway **holds across
  cascade depths 2, 3, 4** with all deciding-matched nulls at 0.00: ADD delivers
  via the residual only (`resid_pre` flip 1.00, `lastlayer_attn` 0.00 at every
  depth), while SUB and NEG deliver via **both** the residual and last-layer
  attention (both arms 1.00 at every depth). Clears the CE14 ≥2-depth bar on the
  mixed model. The reusable sweep is promoted to `quanta_maths.maths_cascade`.
- **Run record**: `PYTHONPATH=. python scripts/mixed_delivery_depth.py`, CPU,
  2026-07-16, `ins1_mix_d6_l3_h4_t40K_s372001`. Artifact:
  `results/study-mixed-delivery-depth/results.json`. Library:
  `quanta_maths/maths_cascade.py` (+ `tests/test_cascade.py`, 5 offline + 3 HF
  pass; full suite 73 passed).
- **Results** (trained; flip/null; all depths k=2,3,4; stimulus valid at every
  depth):
  - **ADD**: full_resid 1.00/0.00, resid_pre 1.00/0.00, **lastlayer_attn
    0.00/0.00** — residual-only delivery, no last-layer attention.
  - **SUB**: full_resid 1.00/0.00, resid_pre 1.00/0.00, **lastlayer_attn
    1.00/0.00** — residual + last-layer attention.
  - **NEG**: full_resid 1.00/0.00, resid_pre 1.00/0.00, **lastlayer_attn
    1.00/0.00** — residual + last-layer attention.
  - **Untrained control** (discriminating arm): lastlayer_attn flip 0.00 for all
    classes (no learned delivery).
- **Interpretation** (vs pre-stated): read (i) confirmed — the class-dependent
  pathway holds at all tested depths. `lastlayer_attn` cleanly separates ADD (0.00)
  from SUB/NEG (1.00) at every depth; the deciding-matched null (0.00) supplies
  carry/borrow-specificity; the untrained control (attn 0.00) shows the effect is
  learned. `full_resid`/`resid_pre` are near-tautological whole-vector patches
  (positive controls); their specificity comes from the null.
- **Prediction scoring**:
  - **A10 (delivery)**: **confirmed at depth on the mixed model** — carry/borrow
    delivery to the combiner is carry-specific across ≥2 depths per class; the
    delivery *route* is class-dependent (ADD residual; SUB/NEG residual +
    last-layer attention), a layer-general refinement of CE14's "attention
    delivers" (holds for the freshly-learned subtraction families; ADD, the
    inserted circuit, uses the residual).
  - **A12**: further supported (the delivery property is depth-general on the new
    architecture/tasks).
- **Skeptic review (post-result)**: (1) *`resid_pre`/`full_resid` tautological* —
  acknowledged; the class-discriminating claim rests on `lastlayer_attn` (ADD 0 vs
  SUB/NEG 1) + the deciding-matched null + the untrained control (first control
  used `resid_pre`, which spuriously "delivered" on the untrained model → switched
  to `lastlayer_attn`). (2) *deciding fixed at units* — "depth" = read distance;
  stated. (3) sign stable (chain stops above the read digit) — by construction.
  (4) single model/seed; depths 2–4 (not 5+, no room for the class-control top).
- **Limitations**: single mixed model/seed; depths 2–4; deciding digit fixed at
  units; `lastlayer_attn` is a whole-last-layer-attention patch (not per-head).
- **Doc updates**: append **CE25** (claim-evidence, Mixed model); A10/A12
  conjecture scoring + reflection-log; results-by-time / summary; next-steps entry
  2 (mark (i) done). Library `maths_cascade.py` promoted + exported. Append-only.
- **Next read**: entry 2 remaining follow-ups (ii clean necessity instrument,
  iii rank-r operator subspace, iv binding/d3) are lower priority; or run the
  promoted `combiner_delivery_sweep` across the mixed-model zoo (cross-model
  delivery comparison — the human's stated future use).
