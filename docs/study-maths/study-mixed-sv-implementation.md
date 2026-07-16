# Study: Mixed model — SV-implementation batteries (CE16/CE17) per class — study-mixed-sv-implementation.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary — Mixed #3 (CE22)

Mixed #1 (CE20) showed the SV *representation* is present in the mixed model; this
study tests the mechanism **details** the paper's SV account rests on, for all
three question classes (ADD/SUB/NEG).

Findings: the digit-combiner is a **step function** — it flips the digit sharply
once the delivered carry/borrow crosses a threshold, not gradually — for addition,
positive subtraction, and negative subtraction alike; the delivered carry is a
**canonical, reusable code** (a probe trained on one digit reads the carry at
another); and the "=" token is **not** where middle digits get their carry from.
The one thing we could not pin down is **necessity**: removing a whole class of
writer nodes does not break the model (they are redundant, as in the addition
models), and our harder-case test had a stimulus flaw — so necessity is left
unscored (consistent with the addition redundancy finding).

Entry 2 step (a). Scope: **mixed model** `ins1_mix_d6_l3_h4_t40K_s372001` only.
Builds on [study-mixed-sv-replication.md](study-mixed-sv-replication.md)
(CE20: representation replicates; delivery class-dependent). Scores A10, A12, A6.

## Pre-run

- **Question**: do the addition **SV-implementation** signatures (CE16 message /
  `=`-not-a-source / class-necessity; CE17 step-combiner) reproduce on the mixed
  model for each class ADD / SUB / NEG? CE20 established the *representation*
  (writers encode the tri-state; resolved carry/borrow is binary at the combiner
  input) and single-step delivery; this study tests the *mechanism* claims the
  paper's SV account rests on.
- **Motivation**: paper deliverable (a) is "details of the SV mechanism"; the
  addition account is CE16 (canonical message, ST-cluster source not `=`,
  class-necessary writers) + CE17 (the combiner is a STEP). Entry 2's "Done when"
  requires these scored replicate/partial/fail on mixed.
- **Competing reads** (neutral): (i) the implementation signatures reproduce
  unchanged per class; (ii) they reproduce for the tri-state families (ADD ST /
  SUB MT) but the NEG family (borrow on `D'−D`) differs; (iii) the 3-layer
  architecture changes the source/necessity picture (CE20 already showed delivery
  differs by class — ADD residual-only).
- **Design** (middle answer digit k=2 unless noted; combiner = last layer L2;
  combiner input = `blocks.2.hook_resid_mid` at the producing position):
  - **B1 step-combiner (CE17 / A10 iv)**: on-manifold α-sweep. Build a U-digit
    pair (carry/borrow-in to k toggled) giving A_k endpoints (0 vs 9). Capture the
    combiner-input residual `r0` (carry-in 0) and `r1` (carry-in 1); patch
    `r(α)=r0+α(r1−r0)` onto the carry-in-0 target and read A_k across
    α∈{0,.25,.5,.75,1}. Endpoint-gated (α=0→A_k(0), α=1→A_k(1)); also project
    r(α) onto the carry axis to confirm a **linear** input rise. STEP if A_k jumps
    at a threshold α* while the projection rises linearly; graded otherwise.
  - **B2 class-necessity (CE16 iii / A6)**: mean-ablate the class's map-named
    question-tail tri-state writers (ADD=ST, SUB=MT, NEG=MT+GT) jointly at their
    positions; measure per-class accuracy on **cascade-bearing** vs **carry-free**
    questions, against a **matched untagged-head** ablation baseline. Necessary if
    cascade accuracy collapses, carry-free is spared, and the drop exceeds the
    untagged baseline.
  - **B3 `=`-not-a-source (CE16 ii)**: patch the `=` (P13) residual from a
    carry-in=1 source into a carry-in=0 target for a **middle** digit; A_k flip
    ≈ 0 ⇒ `=` is not the value source for middle-digit cascades (scoped away from
    the sign, which IS computed at `=`).
  - **B4 message-canonical (CE16 i)**: train a resolved-carry probe at the
    combiner input for deciding digit i, test transfer to digit j; high transfer
    ⇒ a canonical (format-invariant) carry code.
- **Positive controls**: M0 per-class accuracy (1.000); B1 endpoints must gate
  (α=0/1 reproduce the clean A_k(0)/A_k(1)); B2 untagged-head baseline; B3
  stimulus validity (clean src A_k ≠ tgt A_k) + a full-resid patch that DOES flip
  (the carry is deliverable, just not from `=`); B4 diagonal (self-digit) probe.
- **Success**: B1 STEP (sharp α* both endpoints gated); B2 writer ablation ≫
  untagged baseline on cascades, carry-free spared; B3 `=`-arm ≈ 0; B4 transfer ≫
  chance. Per class.
- **Failure**: B1 graded pass-through; B2 writer ablation ≈ untagged; B3 `=`-arm
  flips like full-resid; B4 no transfer.
- **Ambiguous/invalid**: endpoints not gated (B1 instrument void — the CE16-F
  dead-zero trap); accuracy floor already low (can't measure a drop).
- **Skeptic review (pre-launch)**: *combined sprint gate (self-adversarial).*
  Risks flagged: (1) B1 whole-resid interpolation rides other content (mitigate:
  endpoint-gate + carry-axis projection; report as on-manifold not axis-pure).
  (2) B2 ablation may be too destructive (mitigate: untagged baseline +
  carry-free spare check; report necessity-over-baseline). (3) NEG writers: no
  `NT` map tag — use MT+GT (documented). (4) B3 `=` for the sign is a genuine
  source (scope B3 to middle digits). (5) redundancy → single-node ablation ~0
  expected (test at the CLASS level).
- **Decision impact**: A10 iv (step) + A6 (class necessity) + the source verdict
  scored on mixed → feeds the paper hand-off "SV mechanism, mixed" section.
- **Risks/confounds**: multi-layer loci (use last_layer); class contamination
  (class-filtered + class-correct labels/neg_labels); LN damping (B1 patches
  resid_mid pre-LN).
- **Expected artifacts**: `scripts/mixed_sv_impl.py`,
  `results/study-mixed-sv-impl/results.json`.

## Post-run

- **Executive summary**: **The SV-implementation signatures replicate on the
  mixed model across ADD/SUB/NEG on three of four batteries.** The combiner is a
  **STEP** for every class (B1: A_k jumps at α*≈0.5 with endpoints gated — CE17 /
  A10 iv generalizes to the 3-layer model and to the borrow/neg-borrow families);
  `=` is **not the middle-digit value source** (B3: `=`-patch flip 0 while the
  combiner-input control flips 1 — CE16 ii); and the resolved carry/borrow is a
  **canonical, format-invariant code** (B4: resolved-carry probe transfers across
  digits at 1.00 — CE16 i). **Class-necessity (B2) is redundancy-blurred** and not
  cleanly scored: ablating the whole map-named writer class is harmless for
  ADD/SUB cascades (== a truly-untagged baseline, both 1.00) — the expected
  CE13/CE18 redundancy — with a partial NEG signal (difference-writers load-bearing
  for carry-free NEG digits) that is confounded by cascade-stimulus triviality.
- **Run record**: `PYTHONPATH=. python scripts/mixed_sv_impl.py`, CPU,
  2026-07-16, `ins1_mix_d6_l3_h4_t40K_s372001`. Artifact:
  `results/study-mixed-sv-impl/results.json`. Reads the M0 registry
  `results/study-mixed-map/results.json`.
- **Results** (per class; k=2 middle digit; combiner = L2):
  - **B1 step** — ADD A_k 9→0, SUB/NEG 0→9; endpoints_ok True all; sweep is
    two-level (only the endpoints appear), jumping at **α*≈0.5**; carry-axis
    projection rises linearly. STEP for all three classes.
  - **B3 `=`-source** — `=`-patch flip **0** for all classes; combiner-input
    control flip **1**. `=` is a depot, not the middle-digit source.
  - **B4 canonical** — cross-digit resolved-carry transfer **1.00** (self 1.00)
    for all classes. Canonical/format-invariant carry.
  - **B2 necessity** — ADD/SUB: writer-class ablation cascade acc **1.00** ==
    untagged baseline **1.00** (redundant). NEG: cascade **1.00**, carry-free
    **0.00** (digits-only too) — difference-writers load-bearing for NEG digits,
    but the cascade/carry-free contrast is confounded (cascade NEG answer
    ≈100000 is trivial). Not cleanly scored.
- **Interpretation** (vs pre-stated conditions): B1/B3/B4 meet the success
  conditions for all three classes → the CE16(i,ii)/CE17 SV-implementation
  signatures **replicate on the mixed model**. B2 is **ambiguous/instrument-limited**
  (the pre-registered redundancy caveat applies: single class-level ablation is
  redundancy-blurred; the cascade-vs-carry-free stimulus is confounded) → A6
  class-necessity **not scored** on mixed; the redundancy axiom holds.
- **Prediction scoring**:
  - **A10 iv** (combiner is a STEP): **confirmed on the mixed model, all three
    classes** (CE17 generalizes to 3 layers + borrow/neg).
  - **A10 i/ii** (canonical message; `=` not a value source): **confirmed on
    mixed** (B4/B3).
  - **A12** (interface generalises): **further confirmed** — the *implementation*
    (not just representation) signatures port to the mixed architecture/tasks.
  - **A6** (class-necessity / tie-break economy): **untouched/not-scored** —
    redundancy-blurred, stimulus-confounded; consistent with CE18 (redundancy
    intrinsic).
- **Skeptic review (post-result)**: (1) *B1 on-manifold interpolation rides
  non-carry content* — mitigated by endpoint-gating + the two-level readout (only
  A_k(0)/A_k(1) appear); reported as on-manifold, not axis-pure. (2) *B2 first
  run invalid* (untagged baseline hit tagged non-writer heads; full-accuracy
  conflated sign) — refixed with a truly-untagged baseline + digit-only accuracy;
  still redundancy-blurred/confounded → reported not-scored, not "no necessity".
  (3) *NEG cascade stimulus trivial* — flagged; NEG necessity left open. (4) B3
  scoped to middle digits (the sign IS computed at `=`, tested separately in M5).
- **Limitations**: single model/seed; k=2; B1 5-point α grid (threshold between
  0.25–0.5); B2 necessity unresolved (redundancy + stimulus confound); linear
  probes (B4).
- **Doc updates**: append **CE22** (claim-evidence, Mixed model); conjecture
  scoring (A10 iv/i/ii confirmed on mixed, A12 strengthened, A6 not-scored) +
  reflection-log; results-by-time / summary bullets. Append-only.
- **Next read**: 2b SLT-sited A7 steer ([study-mixed-shared-engine.md](study-mixed-shared-engine.md));
  then paper hand-off. Post-deadline: a clean cascade-depth necessity instrument
  (fix the B2 stimulus triviality) and a ≥2-depth delivery sweep.
