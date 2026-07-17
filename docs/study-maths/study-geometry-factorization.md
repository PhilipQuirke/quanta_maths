# Study: geometry-factorization (study-geometry-factorization.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

**Status: DESIGNED 2026-07-16 (evening) — pre-run only. Awaiting the sprint
combined skeptic gate, then implementation + overnight run. No code has been
written or executed for this study yet.** Owner: `maths` thread (latent-geometry
stream). Paired study: [study-geometry-certificate.md](study-geometry-certificate.md)
(shares stimulus grids and activation caches on the addition models).

## Pre-run (write before the experiment)

- **Question**: Do the per-digit `ST` storage manifolds **factor as one shared
  1-D carry rail ⊕ private per-site address subspaces** — so that CE11's two
  negatives (no cross-digit probe transfer; ST–SV entanglement at 21°) are two
  views of a single geometry — and, on the mixed model, do the resolved
  carry/borrow codes (`SV`/`MV`/`NV`) share **one** unit-adjust rail across
  ADD/SUB/NEG?

- **Motivation**: The stream's directive is a geometric account of storage.
  Two accepted results currently read as *obstacles* to any tidy geometry:
  (CE11) question-side `ST` probes do not transfer across digit positions
  (retention 0.10/0.12), and `ST`/`SV` subspaces are entangled (principal
  angle ~21° vs 50–64° label-null). G1 proposes both are *design*: each
  site's write = (its tri-state coordinate on a **shared carry rail** — the
  component the consumer OV projects out and the combiner thresholds) +
  (a **private address component** that dominates raw variance and defeats
  raw probe transfer). Under G1 the ST–SV 21° entanglement *is* the shared
  rail: the carry-bearing component of the ST write is literally the quantity
  SV accumulates. If confirmed, the paper's representation section flips two
  puzzling negatives into one constructive claim. The mixed-model half (G4)
  asks whether "subtract borrows" reuses the same geometric rail as "add
  carries" — the geometric substance behind the shared-engine result, and a
  clean contrast with CE23: the *data* being carried may be 1-D even though
  the *operator control* is high-dimensional.

- **Hypothesis / competing reads** (stated neutrally):
  1. **G1 factorization**: projecting write-site activations onto the
     consumer-OV pre-image of the carry rail restores cross-site transfer
     (gaps transfer; per-site offsets don't and are centered out); removing
     the rail direction from the ST and SV subspaces raises their principal
     angle toward the label-null.
  2. **Genuinely position-specific codes**: the no-transfer result is not
     rescued by any shared 1-D component; canonicalization into a common
     carry code happens only *inside* the consumer read (per-source OV
     rotations aligning distinct site codes). Predicts F2 retention stays at
     the CE11 floor even after rail projection, while CE24's canonical carry
     still emerges downstream.
  3. **Partial sharing**: a shared rail exists for some sites (e.g. the
     strong map-named writers) but not all; retention rescue is intermediate
     and site-dependent.
  - Mixed-model fork (G4): (a) one shared rail (|cos| high between
    carry/borrow/neg-borrow axes, one direction explains most class-axis
    variance); vs (b) class-specific rails (~orthogonal axes read by the
    shared combiner through different input directions).

- **Design**:
  - **Models**: `add_d6_l2_h3_t15K_s372001`, `add_d5_l2_h3_t15K_s372001`
    (both, for the CE11 parity); mixed `ins1_mix_d6_l3_h4_t40K_s372001` for
    F4. Negative control: `make_untrained_control` (rail fit at chance;
    angles at null).
  - **Sites**: the map-named question-tail/sign ST writer sites (CE13/CE17
    strong writers; restrict shape claims to strong writers), read at the
    same residual read-sites CE11 used (harness parity is a positive
    control, below). Mixed: the L2 combiner-input site per CE20/CE22.
  - **Rail pre-image**: per consumer head `h` in the CE14/CE16 pair,
    `r_h = (LN∘OV_h)ᵀ ĉ` at the source site (LN folded as in
    `maths_edge_patch.head_ov`/`ln_scale`); report per-head and pair-summed.
    ĉ refit per model as in the paired study.
  - **Batteries**:
    - **F1 write-site manifold shape**: per strong-writer site, the tri-state
      class-mean structure (`maths_probe.class_mean_subspace`): collinear
      (1-D ordered, U between the committed classes) vs simplex (2-D, U with
      a genuine off-rail component). Metric: U's off-rail height / (0–1 base
      length), against a permutation null. Also record the cin-split of the
      U centroid (CE13). This settles A3's descoped question-position
      remnant *descriptively* at the sites that matter.
    - **F2 OV-aligned transfer (the G1 killer test)**: cross-site tri-state
      probe retention — raw (CE11 reproduction, expected ≈ 0.10–0.12) vs on
      the 1-D rail-projection coordinate with per-site offset centering
      (1-D threshold probe trained at site i, tested at site j). Pre-stated
      bar: retention ≥ 0.6 (the CE11 transfer criterion) = rescue;
      ≤ 0.2 = no rescue.
    - **F3 entanglement re-read**: reproduce CE11's ST–SV principal angle
      (~21°, both models); then remove span(rail pre-image) from both
      subspaces and re-measure (`maths_probe.principal_angles_deg`). G1
      predicts the residual angle rises at least halfway toward the
      label-correlation null (50–64°); competing read 2 predicts ~no change.
    - **F4 mixed rail identity**: fit class axes at the L2 combiner input on
      matched stimuli (CE20 harness): carry axis (ADD), borrow axis (SUB),
      neg-borrow axis (NEG). Report pairwise |cos|, the variance explained by
      the best single shared direction, and (descriptively) |cos| to the SGN
      axis (CE21). Null: label-permutation axes + untrained control.
      Pre-stated bar: shared rail if pairwise |cos| ≥ 0.7 AND one direction
      explains ≥ 70% of the three axes' class-separation; class-specific if
      all pairwise |cos| ≤ 0.3.
  - **Stimuli**: shared caches with the paired study on the addition models
    (per-site class grids + chain families + 9-free variants); mixed model
    reuses the CE20/CE22 matched ADD/SUB/NEG cascade grids
    (`maths_cascade.make_cascade_operands`). N ≈ 300–500 per cell; report
    per-cell N and probe train/test splits (`maths_probe` helpers with
    permutation nulls).
  - **Minimal effect of interest**: F2 — the rescue-vs-floor gap (0.6 vs 0.2)
    is far above probe SE at these Ns; F3 — an angle change of ≥ 10° (CE11's
    21° vs 50–64° null leaves ~30–40° of headroom); F4 — |cos| distinguishable
    from the permutation null (expected ~0.1 at these dims) at 2× SE.

- **Positive control**:
  - **PC1 (harness parity with CE11)**: reproduce, with this study's caches,
    (a) within-site tri-state probe accuracy at the CE11 level (≥ 0.9), and
    (b) the raw cross-site NO-transfer (≤ 0.2). If raw transfer comes out
    high, the harness is not measuring what CE11 measured — invalid, fix
    before interpreting F2. (The rescue claim only means something against a
    reproduced floor.)
  - **PC2 (mixed)**: reproduce CE20's binary carry/borrow decodability
    (~1.00 vs untrained chance) at the L2 combiner input before comparing
    axes.
  - **PC3 (wrong-axis null)**: a random unit vector in place of the rail
    pre-image must NOT rescue transfer (F2) nor move the F3 angle — guards
    the low-variance-projection trap.

- **Success condition** (defined now): F2 retention ≥ 0.6 after rail
  projection (both models, majority of strong-writer site pairs) AND F3
  residual angle rises ≥ 10° toward the null with PC1/PC3 passing → **G1
  confirmed**. Independently: F4 meets the shared-rail bar → **G4 confirmed**
  (one unit-adjust rail, sign/rotation conventions recorded).

- **Failure condition** (defined now, PCs passing): F2 retention ≤ 0.2 after
  rail projection at the strong writers → **G1 refuted** (no shared stored
  rail; canonicalization is a read-time rotation — competing read 2; itself a
  paper-worthy statement: "the common currency exists only on the wire, not
  in storage"). F4 all pairwise |cos| ≤ 0.3 → **G4 refuted** (class-specific
  rails into a shared combiner).

- **Ambiguous / invalid condition**: PC1 fails either half → invalid.
  F2 retention in (0.2, 0.6) or strongly site-dependent → partial sharing
  (competing read 3); scope by site and hand off as such. F3 moves < 10°
  with F2 rescued → the entanglement is not (only) the rail; report both.
  F4 intermediate cosines → report the spectrum; no binary verdict.

- **Skeptic review (pre-launch)**: PENDING — sprint-mode combined gate.
  Specific questions for the skeptic: (1) is per-site offset centering in F2
  licensed, or does it smuggle in position information the consumer doesn't
  have? (2) does removing a 1–2D rail from a ~2D ST subspace in F3 leave
  enough dimensions for the residual angle to be meaningful? (3) are the F4
  axes comparable given the class-dependent delivery routes (ADD residual vs
  SUB/NEG attention, CE20/CE25) — should axes be fit per route?

- **Decision impact**: G1 confirmed → CE entry; A4's template story gets its
  mechanism (shared rail = the transferable part; address = why raw probes
  fail); A8's "entanglement" caveat is re-framed as computation; CE11/CE12
  reinterpretation feeds the paper hand-off section (b) directly. G1 refuted
  → the read-time-rotation account goes in instead (equally concrete). G4
  either way sharpens the shared-engine paragraph (C2/A7) with a geometric
  statement. Score G1, G4; touch A3 (F1 shape), A4, A8, C1/C2.

- **Risks / confounds**: rail pre-image differs per head (report both;
  pair-sum is the CE16-accepted unit); weak writers dilute site-pair stats
  (restrict to strong writers, pre-named from CE13/CE17); CE11 used specific
  read sites/depths — parity enforced by PC1 rather than assumed; mixed-model
  axes may live at slightly different read depths per class (skeptic Q3);
  1-D probes are linear (a nonlinear shared code would be missed — state as
  scope, consistent with the thread's linear-probe caveats).

- **Expected artifacts**: `scripts/geometry_factorization.py` (to be written
  at launch; reuses `maths_probe` probe/angle/subspace helpers, `maths_edge_patch`
  OV/LN helpers, `maths_cascade` stimuli) →
  `results/study-geometry-factorization/results.json` + figures: retention
  bars (raw vs rail-projected, per site pair), ST–SV principal angle before/after
  rail removal vs null band, mixed-model axis |cos| matrix, per-site manifold
  scatter (rail coordinate vs top address PC, classes colored).

## Post-run (fill in after the experiment)

*(not run yet)*
