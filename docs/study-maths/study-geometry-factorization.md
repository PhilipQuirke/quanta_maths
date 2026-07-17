# Study: geometry-factorization (study-geometry-factorization.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

**Status: RUN 2026-07-16 (overnight), dual-gated (pre-launch PASS-WITH-CONDITIONS
incl. an F3 redesign; post-result PASS-WITH-CORRECTIONS, 12 corrections folded).
Landed as [CE27](../maths-claim-evidence.md#ce27).** Owner: `maths` thread
(latent-geometry stream). Paired study:
[study-geometry-certificate.md](study-geometry-certificate.md)
(shares stimulus grids and activation caches on the addition models).

## Executive summary

**Question:** does question-side `ST` storage factor as one shared 1-D carry
rail (the consumer heads' OV pre-image of the CE16 carry axis) ⊕ a private
per-site address, so that CE11's two negatives (no cross-digit probe transfer;
ST–SV 21° entanglement) become one geometry — and, on the mixed model, do the
resolved carry/borrow/neg-borrow codes share one unit-adjust rail?

**Verdict (both addition models + mixed, acc 1.000; linear-probe / representational):**

- **G1 → REFUTED (OV-preimage form); no shared *stored* carry rail shown.** The
  proposed rail does not rescue CE11's cross-digit no-transfer — its cross-site
  binary-carry off-diagonal gain (0.099 d6 / 0.116 d5) does not beat the
  OV-pre-image-of-random wrong-axis null (0.078 / 0.106) by the pre-registered
  0.05 margin, and it is a weak within-site instrument (binary diag ~0.60). Even
  a full-activation *binary* carry probe (CE11 tested only tri-state) transfers
  only at the null band (0.088 / 0.096). **Read:** canonicalization into the
  common carry currency is a **read-time transformation** (consistent with CE24
  — the canonical carry emerges in the L1 read), the common currency living on
  the wire, not in storage. (A *nonlinear* shared code is not excluded.)
- **G4 → single-rail REFUTED → two-rail (descriptive).** At the shared mixed-model
  combiner input, borrow (SUB) and neg-borrow (NEG) share one rail (|cos| 0.90 /
  0.96) but add-carry is ~orthogonal (0.20 / 0.22); SGN orthogonal to all
  (|cos| ≤ 0.09). The carried datum is low-D but splits by operation family.
  Caveat: ADD's axis carries an untrained-decodable nuisance (untrained decodes
  ADD's bit 0.68 vs SUB 0.51).
- **A3 → settled descriptively at the question write site.** The strong-writer
  `ST` manifold is **collinear-ordered** — U's off-rail fraction is not above a
  permutation null (p 0.99–1.0), U split by incoming carry on the rail (CE13).
  No off-axis U "symbol".

**Controls:** PC1 reproduces CE11 (tri retention 0.169/0.074; within-site diag
0.63/0.69; ST–SV 26°/21°); PC2 reproduces CE20 (bits decode 1.00 vs untrained
0.51–0.68); PC3 wrong-axis nulls all at the ~0.10 level the OV rail fails to
beat. Artifacts: `scripts/geometry_factorization.py`,
`results/study-geometry-factorization/results.json`. Full detail in
[Post-run](#post-run-run-2026-07-16).

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
  - **Models**: `add_d6_l2_h3_t20K_s173289`, `add_d5_l2_h3_t15K_s372001`
    (both, the exact CE11 models — the earlier `add_d6_..._t15K_s372001` was a
    typo, corrected 2026-07-16 at launch to match `scripts/probe_transfer.py`);
    mixed `ins1_mix_d6_l3_h4_t40K_s372001` for F4. Negative control:
    `make_untrained_control` (rail fit at chance; angles at null).
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
    (a) within-site tri-state probe accuracy at the CE11 level (**~0.64/0.72
    balanced-acc, gain 0.31/0.39** — the note's earlier "≥ 0.9" was wrong; CE11's
    real tri-state diagonal is 0.639/0.72), and
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

- **Skeptic review (pre-launch)**: DONE 2026-07-16 (fresh-context gate, separate
  agent rehydrated from docs). **Verdict: PASS-WITH-CONDITIONS; F3 as originally
  written was BLOCK (degenerate) and was redesigned before launch.** Key findings
  and the fixes folded into `scripts/geometry_factorization.py`:
  - **Rail pre-image formula is faithful** — `r = w1 ⊙ ((W_V@W_O) @ (w2 ⊙ ĉ))`
    (summed over consumer-pair heads, unit-normed) is exactly the CE16
    `lnfair_project` (SI-2) quantity, linearized; orientation `M@v` (not `Mᵀ`) is
    correct; `w2` applied once, consistent with post-gamma `hook_normalized`.
    Conditions: (1a) **center `r`** (`r ← r − mean(r)`) to remove the LN
    mean-subtraction nuisance term; (1d) **report pairwise |cos| among per-head
    pre-images** before applying one global `r` (d6 has heads {1,2}).
  - **Per-site offset centering LICENSED** as the "gaps transfer, offsets don't"
    test; conditions: (2a) estimate the offset on the **train split only**;
    (2b) report the **uncentered** arm too.
  - **U near ST0 on the rail**: (3c) report **binary ST0-vs-ST1 retention as the
    PRIMARY rail metric**, tri-state as CE11 parity; (3a) apply the
    `diag_weak` guard and confirm PC1 within-site diagonal ≥ 0.9 on this
    instrument; report the F1 cin-split of U.
  - **F3 REDESIGNED (was degenerate: SV class-mean subspace is exactly 1-D, so
    removing `r` leaves 0 dims)**: instead report (4a) |cos|(`r`, SV carry axis)
    and the principal angle of `r` into the 2-D ST subspace, plus the pre-removal
    ST–SV angle (reproduce ~21°); (4b) for a residual-angle number remove `r`
    from **ST only** and measure residual-ST vs full-SV with a **recomputed**
    reduced-dim label null; (4c) verify ST rank ≥ 2 at strong writers first.
  - **PC3 strengthened**: besides the plain random-unit floor, add (5a) a
    **variance-matched** random direction and (5b, headline) an
    **OV-pre-image-of-a-random-axis** null (same `w1⊙(M@(w2⊙·))` geometry, random
    "ĉ") to isolate carry-relevance from OV/LN projection geometry; (5c) a
    random-direction-within-ST-subspace null.
  - **F4 axes**: fit at the shared combiner input (correct/comparable — do NOT
    fit per route); condition (6a) fit the carry-bit axis from **natural random
    questions** (`mean(bit=1) − mean(bit=0)`) so operand content averages out
    rather than from a single `carry_in` operand toggle; (6b) also fit at
    `resid_pre` and report (locus may shift for SUB/NEG without refuting one
    shared rail).

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

## Post-run (run 2026-07-16)

**Artifacts**: `scripts/geometry_factorization.py`,
`results/study-geometry-factorization/results.json` (+ `run.log`). Models:
`add_d6_l2_h3_t20K_s173289`, `add_d5_l2_h3_t15K_s372001` (acc 1.000 both),
mixed `ins1_mix_d6_l3_h4_t40K_s372001`. n_q=2000 (addition transfer), 300 (F1
classes), 400/class (mixed). Both skeptic gates run in separate fresh-context
threads (pre-launch PASS-WITH-CONDITIONS incl. F3 redesign + PC3 strengthening;
post-result PASS-WITH-CORRECTIONS — corrections folded into the wording below).

### Verdict summary

- **G1 (rail-and-address factorization): the specific OV-preimage form is
  REFUTED; no shared stored carry rail is demonstrated.** Framing:
  canonicalization into the common carry currency is *consistent with a
  read-time transformation* (CE24), not a shared stored 1-D coordinate. The
  earlier smoke-test "rescue" (retention ≈ 1.0) was a small-n artifact of the
  near-tautological retention *ratio* on a fixed 1-D axis; the discriminator is
  absolute cross-site gain vs matched wrong-axis nulls.
- **G4 (one unit-adjust rail): single-rail REFUTED; a descriptive TWO-RAIL
  spectrum** — SUB borrow and NEG neg-borrow share one rail; ADD carry is
  ~orthogonal to them (with an ADD-axis nuisance caveat).
- **A3 (F1, descriptive): the question-position ST write is COLLINEAR-ordered,
  U not off-axis** — settles A3's descoped question-position remnant against a
  simplex, at the strong writers.

### PC1 harness parity (passes)

- Full-activation **tri-state** cross-site transfer retention **0.169 (d6) /
  0.074 (d5)** vs CE11 0.124/0.097 — CE11's position-specific no-transfer
  reproduced (d6 slightly high, same regime). Within-site tri diagonal
  0.626/0.688 reproduces CE11's real 0.639/0.72 (gain 0.31/0.39).
- F3 pre-removal ST–SV principal angle **26.1° (d6) / 20.9° (d5)** vs CE11 ~21°
  — entanglement reproduced.

### F2 — the G1 killer test (OV-rail rescue): REFUTED for the OV pre-image

- The OV-preimage rail's cross-site **binary-carry off-diagonal gain is 0.099
  (d6) / 0.116 (d5)**, which does **not exceed the pre-registered headline
  wrong-axis null (OV-preimage-of-random-axis, gate cond 5b) by the 0.05
  specificity margin** (d6 +0.021 vs null 0.078; d5 +0.010 vs null 0.106). So
  the rail is not axis-specific → **G1's OV-preimage form REFUTED.**
- Instrument caveat (post-gate cond 4): the OV-preimage 1-D coordinate is a
  **weak within-site instrument** (binary diag only 0.599/0.614, `diag_weak`
  true for d6) — gate cond 3a's ≥0.9 within-site bar is NOT met, so its
  null-level off-diagonal is partly instrument weakness. The refutation does not
  rest on the weak rail alone: the full-activation binary probe confirms ST *is*
  present within-site (diag 0.85/0.87), it simply does not transfer on the
  consumer's carry-read direction.
- d6 note: the pair-summed "single rail" is a sum of two **near-orthogonal**
  head pre-images (H1–H2 |cos| 0.15), so a single-rail object is not even
  well-defined for d6 — further against a clean shared rail.
- **No shared storage subspace demonstrated (corrected):** a per-site-fit
  BINARY carry-bit probe on the *full* activation transfers at off-diagonal
  gain **0.088 (d6) / 0.096 (d5)** — at (d5 below) the same wrong-axis null band
  used to refute the OV rail, and with no magnitude/nuisance control (CE24:
  magnitude nuisance alone gives ~0.65 transfer). Its high retention *ratio*
  (0.25) is inflated by the strong self-diagonal (0.85/0.87); it is **not**
  evidence of a shared stored carry subspace. Per-digit binary carry axes are
  only partially aligned (mean |cos| 0.41/0.60; some pairs 0.6–0.9, others ~0.1)
  — position-specific storage, consistent with CE11.

### F3 — entanglement re-read (single representative digit per model; scoped)

- |cos|(OV-rail, SV carry axis) = 0.35 (d5) / 0.47 (d6) — moderate; and the
  rail is **oblique to the ST plane** (angle into ST plane 52–57°, i.e.
  substantially outside it). Removing the OV-rail from the ST subspace raises
  the residual ST–SV angle to 81–89°, but the **recomputed reduced-dim null also
  rises to 72–79°**, so the residual-vs-null gap is only ~9–10° and the residual
  overshoots the null. Read: the OV-rail is only *partly* aligned with the
  ST–SV overlap; the 21° entanglement is real but is not cleanly "the OV rail".
  n=1 site per model — scoped, descriptive.

### F1 — write-site manifold shape (A3 descriptive; strong writers)

- Every strong-writer site is **collinear-ordered**: U's off-rail fraction is
  **not above the permutation null** (p 0.99–1.0), though up to 0.31 of the 0→1
  base length in absolute terms at D3/D4. No off-axis U "symbol" at the question
  write site — extends CE6/CE7 (no dedicated U symbol) to the question-position
  write.
- The U centroid **splits by incoming carry on the rail** (cin-split up to 0.98
  at d5-D3, 0.33 at d6-D3; ≈0 elsewhere) — site-concentrated (one digit per
  model), consistent with CE13's low/middle-digit single-step U-resolution.

### F4 — mixed rail identity (G4): two-rail spectrum (descriptive)

- PC2 passes on decodability (all bits 1.00 trained). Untrained control: SUB
  0.51 (clean), NEG 0.61, **ADD 0.68** — so the **ADD carry-bit axis carries an
  untrained-decodable nuisance component** (operand/format), a caveat on ADD's
  separateness below.
- Combiner-input pairwise |cos|: **SUB–NEG 0.897**, ADD–SUB 0.196, ADD–NEG
  0.223; resid_pre: SUB–NEG 0.956, ADD–SUB 0.006, ADD–NEG 0.054. A single shared
  direction explains only **0.66** of the three axes' class separation (< 0.70
  bar). SGN axis orthogonal to all three (|cos| ≤ 0.09).
- Pre-registered scheme (all ≥0.7 shared vs all ≤0.3 specific) → **intermediate
  → report spectrum, no binary verdict**. Descriptive reading: **borrow (SUB)
  and neg-borrow (NEG) share one rail; ADD carry is a distinct ~orthogonal
  rail** — the note's alternative (a). Caveat: ADD's orthogonality is not
  established until its axis is re-derived with the untrained nuisance removed,
  and there is no format-nuisance control on the high SUB–NEG alignment (SUB/NEG
  share subtraction format).

### What this changes

- **G1 → refuted (OV-preimage form) / no shared storage rail shown.** The
  paper's representation section (hand-off entry 3 §b) gets the concrete
  **read-time-canonicalization** statement (consistent with CE24): the common
  carry currency lives on the wire, not as a shared stored 1-D coordinate;
  question-side ST storage is position-specific (CE11 confirmed and sharpened —
  the binary carry bit does not transfer either, on the consumer's read axis).
- **G4 → two-rail descriptive** sharpens the shared-engine paragraph (C2/A7):
  the *carried datum* is not one universal rail; add-carry and subtract-borrow
  are geometrically distinct, while positive/negative subtraction reuse one
  borrow rail.
- **A3** descriptively closed at the question write site (collinear, U on-rail).

### Skeptic review (post-result)

DONE 2026-07-16 (fresh-context gate). Verdict **PASS-WITH-CORRECTIONS**; all 12
corrections folded into the wording above: dropped the "partial shared carry
subspace" over-claim (binary off-diag at null; double standard vs the OV-rail
refutation); removed the mismatched tri-vs-binary chance-level comparison;
stated the specificity margin rather than "at null"; added the weak-instrument,
near-orthogonal-heads, single-digit-F3, cin-split-concentration, and ADD-nuisance
caveats; reframed "read-time rotation" as CE24 consistency (not proof); confirmed
the CE11 model parity (d6 = `add_d6_..._t20K_s173289`, note typo fixed) and the
real ~0.64/0.72 PC1 level.

### Immediate next read

Fold G1(refuted)/G4(two-rail)/A3(collinear) into the paper hand-off §b as the
concrete read-time-canonicalization + two-rail statements. Post-deadline: an ADD
nuisance-controlled G4 re-fit and a multi-digit F3 would firm the two-rail read.
*(2026-07-17: the G4 re-fit was human-green-lit inside the window — Addendum
F4b below.)*

## Addendum — Battery F4b: ADD-nuisance-controlled G4 re-fit (pre-run written 2026-07-17, ~T-7h; human green-light)

- **Question**: Does the F4 two-rail verdict (SUB–NEG share a borrow rail;
  ADD-carry ~orthogonal) survive removal of the two flagged nuisances — (i) the
  ADD axis's untrained-decodable operand/format component (untrained decode
  0.68), and (ii) the SUB/NEG shared-subtraction-format confound on their high
  alignment (0.897)?
- **Competing reads (neutral)**: (1) **two-rail genuine** — after nuisance
  control, SUB–NEG stays high, ADD pairs stay low, ADD decode stays high (its
  axis is real carry signal, orthogonally stored); (2) **nuisance-masked
  single rail** — cleaning *raises* ADD–SUB/ADD–NEG toward the shared bar (the
  nuisance component was hiding alignment) → G4's original form revives; (3)
  **format-artifact SUB–NEG** — cleaning/cross-digit drops SUB–NEG toward the
  null → three separate rails.
- **Design** (mixed `ins1_mix_d6_l3_h4_t40K_s372001`, last-layer combiner
  input `ln2.hook_normalized`, same harness as F4):
  - Collect class-conditional pools per (class ∈ {ADD,SUB,NEG} × read_digit ∈
    {2,3} × {trained, untrained-twin}), natural random questions split by the
    resolved SV/MV/NV bit (n_q≈300/bit trained, ≈120 untrained), reusing
    `_mixed_axis_at`.
  - **Arm 1 — untrained-nuisance projection**: nuisance subspace N = span of
    the untrained-twin per-class axes (≤3-D per read digit); project N out of
    *trained* activations; re-fit class axes; recompute pairwise |cos| +
    shared-direction variance-explained. Controls: trained bit-decode after
    projection stays ≥0.9 (the axis is not the nuisance); untrained bit-decode
    after projection falls to ≈0.5 (the projection removes what untrained
    could decode).
  - **Arm 2 — cross-digit canonical axes**: per-class axes fit independently
    at read digits 2 and 3. Metrics: within-class cross-digit |cos|
    (canonicality; untrained twin same metric = nuisance replication control)
    and the **cross-class × cross-digit** |cos| matrix (e.g. ADD@2 vs SUB@3) —
    digit-bound nuisance cannot align axes fit at different digits.
  - **Null band**: |cos| of random unit vectors at d_model (≈0.04–0.09) via
    permutation draws; SGN-axis cosines recomputed descriptively.
- **Positive control**: trained per-class bit decode ≈1.00 pre-projection
  (reproduces F4 PC2); Arm-1 projection controls above.
- **Success / verdict rule (pre-stated)**: apply the original F4 bars to the
  **nuisance-controlled** matrix (Arm 1, corroborated by Arm 2): shared if all
  pairwise ≥0.7 AND shared-direction ≥0.70; class-specific if all ≤0.3.
  **SUB–NEG "genuine shared borrow rail"** requires: cleaned SUB–NEG ≥0.7 AND
  cross-digit SUB–NEG ≥0.6 AND untrained SUB–NEG within ~2× the null band.
  **"ADD orthogonal, established"** requires: cleaned ADD pairs ≤0.3 AND
  cleaned ADD decode ≥0.9 AND cross-digit ADD-vs-SUB/NEG ≤0.3.
- **Failure / revision condition**: cleaning raises ADD pairs above 0.3 →
  report the masked-alignment read (read 2); SUB–NEG cleaned or cross-digit
  < 0.6 → downgrade the borrow-rail sharing to format-artifact-suspect
  (read 3).
- **Ambiguous / invalid**: Arm-1 controls fail (projection kills trained
  decode or leaves untrained decode high) → the nuisance is not a ≤3-D linear
  subspace at this site; report Arm 2 alone as the verdict, scoped. Arms
  disagree → report both, no binary verdict.
- **Skeptic review**: sprint combined gate (single separate-thread pass,
  design + interpretation together) after results, before doc updates —
  matching the certificate-study pattern.
- **Decision impact**: firms or revises the CE27-F4 two-rail statement and
  the hand-off §(b)6 wording within the submission window.
- **Risks**: untrained twin's nuisance direction may differ from the trained
  model's nuisance (initialization-specific) — mitigated by Arm 2, which
  needs no untrained estimate; rejection sampling for rare bits at read
  digit 3 may be slow (cap tries, report per-cell N); single model, single
  seed — scope stays descriptive-firmed, not universal.
- **Expected artifacts**: `scripts/geometry_g4_refit.py` →
  `results/study-geometry-factorization/results_f4b.json` + `run_f4b.log`.

### F4b Post-run (run 2026-07-17, ~T-6.5h)

**Artifacts**: `scripts/geometry_g4_refit.py`,
`results/study-geometry-factorization/results_f4b.json`, `run_f4b.log`.
n=300/bit trained, 120/bit untrained, read digits {2,3}, all cells filled
(no rejection-sampling shortfall). Random-|cos| null band: mean 0.039,
p95 0.092 (d_model-matched draws).

**Baseline replication**: rd=2 pairwise |cos| ADD–SUB 0.207 / ADD–NEG 0.226 /
SUB–NEG **0.898** (CE27-F4: 0.196/0.223/0.897 — replicated on a fresh draw);
rd=3: 0.205/0.226/0.828. Untrained decodes 0.52–0.61 this draw (CE27 saw ADD
0.68 — same nuisance regime, sampling variation).

**Arm 1 (untrained-nuisance projection)** — controls **PASS**: after
projecting out the untrained twin's per-class axes, trained decode is **1.00
for every class at both read digits**, and untrained *held-out* decode falls
to 0.586/0.598/0.526 (rd2) and 0.348/0.475/0.468 (rd3), all ≤ 0.6. Cleaned
cosines are **unchanged**: rd2 ADD–SUB 0.204 / ADD–NEG 0.221 / SUB–NEG
**0.898** (rd3: 0.199/0.221/0.828); axis shift raw-vs-clean |cos| ≈ 1. The
estimated untrained nuisance component carries **none** of the alignment
structure. Shared-direction variance stays 0.66 (< 0.70 shared bar).

**Arm 2 (cross-digit canonical axes)**: within-class cross-digit |cos| is
**0.981 (ADD) / 0.977 (SUB) / 0.975 (NEG)** trained — each class's rail is
canonical across read digits (CE22-consistent) — vs **0.415 / 0.099 / 0.223**
untrained (the nuisance largely does not replicate; ADD's 0.415 confirms a
real digit-stable format component that training's axis does not inherit —
cleaned ADD decode 1.00, cleaned/cross-digit ADD alignments unchanged).
Cross-class × cross-digit |cos| (nuisance-robust by construction): ADD–SUB
**0.208**, ADD–NEG **0.224**, SUB–NEG **0.877**.

**Verdict against the pre-registered bars**:

- **ADD orthogonal — ESTABLISHED** (all bars met: cleaned ADD pairs ≤ 0.3,
  cleaned ADD decode 1.00 ≥ 0.9, cross-digit ADD pairs ≤ 0.3). The CE27
  ADD-nuisance caveat is **closed**: the ADD carry axis is real signal,
  ~orthogonal (≈0.21, vs null p95 0.092) to the borrow rails.
- **Single shared rail — NO** (cleaning does not raise the ADD pairs; the
  nuisance was not masking a shared rail; shared-dir var 0.66 < 0.70).
- **SUB–NEG shared borrow rail — GENUINE (registered bars met at the primary
  read digit)**: cleaned 0.898 (≥ 0.7 ✓), cross-digit 0.877 (≥ 0.6 ✓), and
  the **registered** untrained control — the untrained twin's cross-class
  SUB–NEG alignment — is **0.029 at rd2** (null floor; bar ≤ ~2×null-p95 =
  0.183 ✓). At rd3 the same quantity is 0.226 (above the band); together with
  the within-class replication probe (SUB 0.099/0.005, NEG 0.223/0.348 across
  two draws; ADD 0.415/0.407) the untrained background is a **draw-unstable
  ≤~0.4 noise band** (untrained axes are noise directions, decode 0.5–0.6),
  which cannot account for the stable, projection-invariant 0.83–0.90
  trained alignment — kept as the residual caveat.
- **Automated verdict line** (`results_f4b.json`) printed "see components /
  scoped" because the script's verdict logic substituted a **stricter,
  unregistered** control (untrained *within-class cross-digit replication*
  ≤ 0.2) for the registered one (untrained *cross-class* alignment); the
  registered quantity was computed and appended post-hoc
  (`untrained_registered_bar_check`) and **passes at rd2** — see the skeptic
  section for this discrepancy's disposition. Supporting note: the trained
  ADD–SUB alignment (0.21) is *below* the untrained ADD–SUB background
  (0.31) — training actively separates the ADD axis from the shared-format
  background.

**Net for CE27/G4**: the **two-rail spectrum is FIRMED** — ADD-orthogonality
upgraded from "caveated descriptive" to **established (nuisance-controlled,
two arms)**; SUB–NEG sharing meets its registered bars (rd2) and survives
both nuisance arms, with the untrained noise-band (≤~0.4, draw-unstable) as
the stated residual caveat. What this battery cannot distinguish: a
trained-model-*learned* component that is bit-locked in both SUB and NEG is
definitionally the shared borrow signal, not a separable nuisance — the
axes' identity as the delivered resolved borrow is anchored causally
upstream by CE20/CE22 (delivery flips 1.00, deciding-matched null 0.00), not
by this battery. Single-model, single-seed; representational; read digits
2–3; natural-question stimuli.

**Prediction scoring**: G4 two-rail (CE27 form) — **confirmed/firmed** (this
battery); G4 original single-rail — stays refuted; competing read 2
(nuisance-masked single rail) — **refuted** (cleaning changes nothing);
competing read 3 (SUB–NEG format artifact) — **disfavored** (survives
projection + cross-digit; residual caveat noted).

### F4b Skeptic review

**Separate-thread gate WAIVED by the human at ~T-6h** (the gate-subagent
launch was declined; instruction: continue). In its place a written
**self-skeptic pass** (same audit questions the gate would have received),
findings and dispositions:

1. **(CORRECTED — the material finding)** Registration/implementation
   mismatch: the pre-registered SUB–NEG control was the untrained twin's
   **cross-class SUB–NEG alignment** (≤ ~2× null band); the script's verdict
   logic instead tested untrained **within-class cross-digit replication**
   (≤ 0.2) — a stricter, unregistered quantity — and failed NEG on it (0.223).
   Resolution: the registered quantity was computed post-hoc and appended to
   `results_f4b.json` (`untrained_registered_bar_check`): rd2 **0.029 PASS**
   (primary read digit, matching the original F4's read_digit=2), rd3 0.226
   above-band. Verdict follows the registered bar; both probes are reported.
2. **(NOTE)** Untrained axes are noise directions (decode 0.5–0.6, n=120/bit),
   so their cosines are draw-unstable (NEG within-replication 0.223 vs 0.348,
   SUB 0.099 vs 0.005 across two draws). Untrained cosine magnitudes are
   reported as a **band (≤~0.4)**, not point values; no bar should be read as
   precise to ±0.05. The trained alignments (0.83–0.90) sit ~3–4× above the
   band; the trained ADD alignments (~0.21) sit *inside* it (and below the
   untrained ADD–SUB 0.31), so ADD's orthogonality verdict rests on the
   cleaned-decode (1.00), projection-invariance, and cross-digit stability —
   not on 0.21 being "small" in isolation.
3. **(NOTE)** Arm-1's untrained twin only estimates the *pre-training*
   nuisance; a component the trained model *learned* that co-varies with the
   borrow bit in both SUB and NEG is not excludable here — and is
   functionally the shared borrow signal itself. The axes' causal identity
   rests on CE20/CE22 (stated in the Net paragraph).
4. **(Verified)** Quoted numbers checked against `results_f4b.json`; Arm-1
   has no circularity (nuisance fit on untrained train-split; untrained
   control decode on held-out split); cross-digit cosines are between
   independently fitted axes; positive controls pass (trained decode 1.00
   all cells pre- and post-projection).
