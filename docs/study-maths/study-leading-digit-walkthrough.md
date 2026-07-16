# Study: Leading-Digit Hard-Case Walkthrough (study-leading-digit-walkthrough.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #16

When you compute `99999 + 1`, the answer's *leading* digit (the new `1` in
`100000`) can only be decided at the very last question token (the `=`/sign
position) — a carry has to ripple through all five 9s and arrive there. This study
writes an end-to-end, node-by-node **worked example** of exactly how the model does
that, using the causal tools built in the previous two studies, and marks each step
as *causally verified* or *inferred*. It's the capstone of the paper's 5-step
program (C5): a documented trace of the hardest single case.

Agenda entry **C5 step 5**. A documentation/synthesis study
applying the CE13/CE14 instruments to the **leading-digit locus**; scores
[A10](../maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster)
and [A6](../maths-conjectures-agent.md) at that locus. Frames cautiously: A10 is
*partially* confirmed (CE14 — carry-specific distributed delivery, selection
unshown), so the trace documents carry-specific *involvement* of the redundant L1
pair, not a clean single-head selector.

## Pre-run (write before the experiment)

- **Question**: In a hard leading-digit edge case (`99999+00001=`, `999999+000001=`),
  produce a **per-link causal trace** of how the leading answer digit `A_top` is
  generated at the **sign-token position** (which is `A_top`'s consuming position):
  which map-named nodes carry each step (sign-position L0 `ST` heads → sign-position
  L1 head edge → sign-position L1-MLP combiner → `A_top` logits), and is each link
  **causally verified** (by the CE13/CE14 instruments) or **inferred**?

- **Motivation**: C5 step 5 and the C4 leading-digit argument: the top answer digit
  is predicted from the answer-sign position, where no later position can rescue an
  unresolved carry — a locus no study probed until now. It is the sharpest stress
  case for A10 (the *full* cascade must resolve at one position) and the eventual
  paper-thread worked example. CE13 (ST nodes write local class + single-step
  U-resolution), CE14 (carry-specific attention-edge delivery to the combiner at
  ≥ 2 depths, redundant H1/H2 pair, selection unshown) supply the instruments and
  the mechanism template; this study applies them at the sign position and documents
  the end-to-end path.

- **Ground-truth facts the design uses** (self-sufficiency; HF maps + verified in
  code 2026-07-16):
  - **Leading-digit locus**: `A_top` (5-digit `A5`, 6-digit `A6`) is produced at the
    **sign-token position** = `2·n_digits+2` (5-digit pos 12, 6-digit pos 14); this
    IS `A_top`'s consuming position (`n_ctx-2-n_top` = the sign token). Verified:
    `99999+1=100000` and `999999+1=1000000` predicted correctly; patching the
    sign-position `resid_mid` from a `+1` source into a `+0` target flips `A_top`
    0→1 (pre-check).
  - **Sign-position wiring** (HF `features.json`/`behaviors.json`):
    - 5-digit: sign-position L0 ST heads `P12L0H1`(A3.ST), `P12L1H2`(A4.ST); the `=`
      (pos 11) L0 ST head `P11L0H2`(A0.ST). Sign-position L1 heads `P12L1H0`,
      `P12L1H2` (Impact:A5, attend {`=`/`P6`, ST-cluster}); sign-position combiner
      `P12L1MLP` (Fail 13%, Impact:A5).
    - 6-digit: sign-position L0 ST heads `P14L0H1`(A5.ST), `P14L0H2`(A4.ST).
      Sign-position L1 heads `P14L1H0`, `P14L1H1` (Impact:A6, attend {`=`=P13,
      ST-cluster P10/P11}); combiner `P14L1MLP` (Fail 3%, Impact:A6).
  - **Hard-case contrast**: `99...9 + 00...01` (leading = 1, full cascade) vs
    `99...9 + 00...00` (leading = 0, no cascade). The single deciding input is the
    units `+1`; the resolved carry into `A_top` differs (1 vs 0). This is the
    matched pair for causal links (differs only at the units operand).
  - **Instruments** (reuse): CE10/CE14 edge-patch (`ln2.hook_normalized` MLP-only,
    less-damped), deciding-matched specificity null, mean-ablation vs untagged
    baseline, value-matched tracking — from `sv_compounding.py`/
    `cascade_handoff_edge_patch.py`; `build_chain` for graded-depth variants.

- **Hypothesis / competing reads** (neutral; the trace documents which links are
  causal):
  - **R-A10-at-leading (expected)**: the leading-digit path mirrors CE14 at the
    sign position — the sign-position L1 head(s)' edge carry-specifically drives the
    sign-position L1-MLP combiner, which produces `A_top`; the L0 ST heads supply the
    inputs; the delivery is redundant (H-pair). Each link causally verified.
  - **R-direct-at-leading**: the resolved carry reaches the sign-position combiner
    via the direct residual path, not the L1 head edge (as the underpowered CE14
    direct arm left open).
  - **R-partial/inferred**: some links (esp. L0-ST → L1-head value read) are only
    *inferred* (content present but not head-attributable, per CE13/CE14), documented
    honestly as inferred, not verified.
  - **R-different-at-leading**: the leading digit uses a *different* mechanism than
    middle digits (e.g. its lower Fail% combiner suggests a more static/degenerate
    path) — a genuinely new fact if the sign-position links behave unlike CE14.

- **Design**:
  - **Models**: 5-digit `add_d5_l2_h3_t15K_s372001`, 6-digit
    `add_d6_l2_h3_t20K_s173289`. CPU; acc-verified.
  - **The trace (per model)**, each link tested by the named instrument:
    - **Link 4 (combiner → `A_top` logits)**: patch the sign-position `resid_mid`
      (whole combiner input) from the `+1` source into the `+0` target on the hard
      pair → does `A_top` flip 0→1? (readout/combiner causality). Verified pre-check.
    - **Link 3 (L1 head edge → combiner)**: the CE14 less-damped edge patch of the
      sign-position L1 head(s) (single + joint pair) into the combiner, on the hard
      matched pair, with the **deciding-matched null** (same leading class, filler
      differs) → carry-specific edge flip? at graded cascade depths (units `+1` with
      k nines below the leading digit).
    - **Link 2 (L0 ST heads → L1 head)**: value content — is the resolved/ST carry
      readable in the L1 head's value at the attended ST/`=` sites (CE14 Battery V,
      with the same-position wrong-role baseline)? Likely *inferred* (position-level).
    - **Link 1 (operands → L0 ST heads)**: the sign-position L0 ST heads encode
      their local class (CE13 Battery E) and their ablation impact (CE13 Battery Ab
      vs untagged baseline) at the leading locus.
    - **Economy (A6)**: mean-ablate the sign-position L1 head(s)/combiner — harms the
      hard cascade case vs a carry-free leading case (`00000+00000`-style)? vs
      untagged baseline.
  - **Depth-graded hard cases**: `k` nines below the leading digit (k=1..n_top),
    units `+1`, to check the leading-digit edge delivery at ≥ 2 depths (CE14
    standard).
  - **Verified vs inferred labeling**: each link tagged `verified` (instrument
    clears its pre-registered bar with its null/baseline) or `inferred` (consistent
    but not independently causal at this locus — e.g. content not head-specific).
  - **Bars**: edge flip ≥ 0.5 with deciding-matched null ≤ 0.20 at ≥ 2 depths
    (carry-specific, CE14 standard); combiner-residual patch flips `A_top`
    (readout); ablation impact > untagged baseline (CE13/CE14). Both models or
    model-scoped.
  - **Scope**: 2-layer addition, the single leading-digit hard case + graded depths;
    a documented trace, not a new mechanism claim beyond CE14 applied at the sign
    locus.

- **Positive controls**: (1) the hard case is answered correctly (`A_top`=1) — the
  trace describes a real computation; (2) the combiner-residual patch flips `A_top`
  (readout control, pre-checked True); (3) the CE14 edge instrument validity on a
  known-causal sign-position head; (4) untagged-head ablation baseline (economy).

- **Success condition** (defined now): a per-model end-to-end trace of `A_top` in
  the hard case exists, each of links 1–4 + economy tagged verified/inferred with
  its instrument result, and the leading-digit path is scored against A10 (does it
  mirror CE14's carry-specific distributed delivery?) and A6 (economy).

- **Failure / ambiguous / invalid**: the substantive outcomes are the per-link
  verified/inferred tags and whether the leading locus mirrors CE14
  (R-A10-at-leading) or differs (R-different-at-leading / R-direct-at-leading).
  Invalid: hard case mispredicted; controls fail.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + HF maps + reused code, all verified). **Verdict: PASS WITH CONDITIONS** —
  four blocking findings, all closing over-read risks inherited from the cascade
  line's history:
  - **C1 (blocking) — Link-4/Link-3 direct-path slide.** The Link-4 whole-`resid_mid`
    patch is the *readout* (trivially flips `A_top`); it must be tagged
    **`readout-only` and NEVER counted toward A10**. Carry-*delivery* is scored ONLY
    by Link-3 (the head edge) with the deciding-matched null AND the CE14 `direct_
    scaled` power gate (a sub-bar head edge = R-direct-at-leading only if the scaled
    direct arm is supra-bar; else "direct not excluded / underpowered").
  - **C2 (blocking) — verified-vs-inferred bar.** Link-3 `verified` requires
    joint-or-single edge flip ≥ 0.5 at **≥ 2 non-degenerate depths** AND
    deciding-matched null ≤ 0.20; a single-head-only or one-depth-only pass is
    **`inferred (underpowered, CE14 SV-1)`**, never `verified`. Tag field admits
    `{verified, inferred, different, underpowered}`.
  - **C3 (blocking) — behavioral gate.** Per graded depth, record the model's
    accuracy on the leading-digit hard case; a depth < 1.0 (stated floor) is
    **excluded from the trace** (`behaviorally-invalid at depth k`); "≥ 2 depths"
    counts only behaviorally-passing depths.
  - **C4 (blocking) — degenerate/different rule.** The leading combiner has much
    lower Fail% (13% / 3% vs 24–59%) → a genuine degenerate-mechanism candidate.
    Pre-register: **R-different/degenerate** is selected when Link-3's carry-specific
    head edge is sub-bar while Link-4 (readout) fires and a **static-output control**
    shows near-constant `A_top` except in the rare cascade. Record
    `leading_mechanism ∈ {mirrors_CE14, different, degenerate, inconclusive}`.
  - **C5 (non-blocking) — A10 ceiling.** This single-locus worked example can only
    **consolidate CE14** or record R-different; it **cannot raise A10** above medium
    or add confirmation beyond CE14 (`a10_delta ∈ {consolidate, different, none}`;
    `confirm` forbidden).
  - **C6 (non-blocking) — wiring label.** `P12L1H2` is an **L1** node (A4.ST at L1),
    not an L0 ST head; the sign-position **L0** ST heads (5-digit) are `P12L0H1`
    (A3.ST) and `P11L0H2` (A0.ST at `=`).

  *Status: RESOLVED 2026-07-16 by the working thread via amendments LW-1…LW-6 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

**2026-07-16 — LW-1 (resolves C1): Link 4 is readout-only; delivery = Link 3 only.**
Link 4 (whole-`resid_mid` patch) is tagged `readout-only` — it establishes the
sign position is `A_top`'s readout, NOT that the L1 head delivers the carry; it is
**never counted toward A10**. Carry-delivery is scored ONLY by Link 3 (the
sign-position L1 head edge) against the deciding-matched null at ≥ 2 depths, with the
CE14 `direct_scaled` power gate. results.json keys: `link4_readout_flip`,
`link3_headedge_flip`, `link3_deciding_matched_null`, `direct_full_signpos`,
`direct_scaled_signpos`.

**2026-07-16 — LW-2 (resolves C2): Link-3 verified bar = ≥ 2 non-degenerate depths.**
`verified` iff (joint-or-single edge flip ≥ 0.5 at ≥ 2 non-degenerate,
behaviorally-passing depths) AND (deciding-matched null ≤ 0.20). Single-head-only or
one-depth-only → `inferred (underpowered, CE14 SV-1)`. Every link carries a tag in
`{verified, inferred, different, underpowered}` in results.json.

**2026-07-16 — LW-3 (resolves C3): per-depth behavioral gate.** For each graded
depth `k` (units `+1`, `k` nines below the leading digit), record leading-digit
accuracy (`behavioral_gate_signpos[k]`); a depth below 1.0 is excluded from the
trace and flagged `behaviorally-invalid at depth k`. "≥ 2 depths" counts only
passing depths.

**2026-07-16 — LW-4 (resolves C4): degenerate/different decision rule + static
control.** Add a **static-output control**: `A_top` output variance at the combiner
across the graded hard cases; near-constant-except-cascade → `degenerate`.
Pre-registered rule: **R-different/degenerate** if Link-3 carry-specific head edge is
sub-bar while Link-4 (readout) fires; **R-A10-at-leading (mirrors CE14)** only if
Link-3 is carry-specific at ≥ 2 depths. Record `leading_mechanism`.

**2026-07-16 — LW-5 (resolves C5): A10 scoring ceiling.** This study
**consolidates CE14 at the sign locus or records R-different — it does NOT raise
A10** (ceiling = medium; `confirm` forbidden). `a10_delta ∈ {consolidate, different,
none}`; doc-updates must not change A10's confidence.

**2026-07-16 — LW-7 (design fix, found on first run): graded builder connects the
chain to the leading digit at every depth.** The first builder placed the 9-chain at
the top k digits but the deciding `+1` at the fixed units, so the carry reached the
leading digit ONLY at full depth (k = nd−1) — giving a spurious "one relevant depth"
(and a wrongly `underpowered` Link-3 tag). Corrected: the 9-chain is the top k
digits and the **deciding digit sits immediately below the chain** (index k), so the
carry always connects to the leading digit and A_top flips 0→1 at **every** k
(verified: k=1..4 all correct in 5-digit). "≥ 2 depths" now counts genuine
graded-cascade depths into the leading digit. (Note: the leading digit is
cascade-affected only when the chain connects — this graded family is the correct
way to exercise it, vs the middle-digit `build_chain`.)

**2026-07-16 — LW-6 (resolves C6): wiring corrected.** Sign-position **L0** ST heads
(5-digit): `P12L0H1`(A3.ST); the `=`-position `P11L0H2`(A0.ST). `P12L1H2`(A4.ST) is
an **L1** node (ST-at-L1 special case), treated as a consumer/L1 site, not an L0 ST
read site. 6-digit L0 ST at sign: `P14L0H1`(A5.ST), `P14L0H2`(A4.ST).

- **Decision impact**:
  - **R-A10-at-leading**: C5 step 5 done — the leading digit uses the same
    carry-specific distributed-delivery mechanism (CE14) at the sign position; A10
    consolidated across all answer digits incl. the hardest; A6 economy at the
    leading locus. The paper-thread worked example is ready (with the CE14 caveats).
  - **R-different-at-leading**: a new fact — the leading digit is special (more
    static / different path); reshapes the A10 "uniform mechanism" reading.
  - **R-partial/inferred**: honest documentation that some links are inferred, not
    causal at this locus — bounds the worked example's rigor.

- **Risks / confounds** (with mitigations):
  - **Single hard case** = low n: use graded depths + matched pairs + the
    deciding-matched null (not a one-question anecdote).
  - **Carry-specificity** (the CE14 lesson): deciding-matched null mandatory before
    any "delivers the carry" link is tagged verified.
  - **Content ≠ delivery** (CE13/CE14): the L0-ST→L1-head link (value content) is
    tagged *inferred* unless it beats the same-position wrong-role baseline.
  - **Sufficiency ≠ necessity** (CE14): edge-patch verifies sufficiency; ablation
    (vs untagged baseline) tests necessity; report both, they may differ (redundant
    pair).
  - **Leading-digit degeneracy**: the leading digit is usually static (0/1); the
    hard case is the non-static instance — report the graded-depth behavioral gate.

- **Expected artifacts**:
  - Standalone CPU script `scripts/leading_digit_walkthrough.py` (reuses the CE14
    edge-patch, deciding-matched null, ablation, and `build_chain` at the sign
    position).
  - `results/study-leading-digit-walkthrough/`: `results.json` (per-link results +
    verified/inferred tags, per-model; graded-depth edge flips + nulls; economy vs
    baseline; controls); a rendered per-model trace table. No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: **The leading digit is produced by the SAME carry-specific
  attention-edge delivery mechanism as the middle digits (CE14), at the sign
  position — C5 step 5 done, A10 CONSOLIDATED (not raised) across all answer digits
  including the hardest.**   In the hard graded-cascade case (`99..9 + 00..01`, chain
  of `k` nines reaching the leading digit, deciding `+1` just below), the
  sign-position L1 consumer head edge patched into the sign-position L1-MLP combiner
  flips the leading digit `A_top` **carry-specifically** (real flip 1.00,
  deciding-matched null 0.00) — Link 3 **verified across a genuine depth spread in
  5-digit (k=1–4) and at deep chains only in 6-digit (k=4,5, adjacent near-maximal —
  not an independent 2-depth spread)**.
  Honest per-model + per-link nuances:
  - **Link 3 (delivery) — 5-digit clean, 6-digit deep-chains-only**: 5-digit the
    joint head edge carries at **all 4 depths** (real 1.00, null 0.00) — a genuine
    depth spread. 6-digit carries only at the **deep** chains **k=4,5** (two adjacent
    near-maximal chains, so *not* an independent 2-depth spread) and **not at k=1–3**,
    where the answer still flips via Link 4 (readout) but **not through these two
    heads' edge** — so in 6-digit the L1 head demonstrably does NOT deliver for
    shallow leading cascades; that path is unadjudicated (direct arm underpowered).
    5-digit meets the verified bar cleanly; 6-digit is "verified deep-chains only".
  - **Link 4 (readout) — readout-only, NOT counted toward A10** (LW-1): the
    whole-`resid_mid` patch flips `A_top` 1.00, confirming the sign position is
    `A_top`'s readout — nothing more.
  - **Link 1 (L0 ST encoding/ablation)**: verified in 5-digit (`P11L0H2` A0.ST
    ablation impact 0.045 > 0) but **inferred/redundant** in 6-digit (sign-position
    L0 ST heads impact 0.00 — redundant, matching CE13).
  - **Economy (A6) — uninformative here**: ablating the sign-position L1 heads harms
    the hard cascade (gap 0.47–0.53) BUT an **untagged head at the same position also
    harms it** (baseline gap 0.49) → **either** the sign position is a general
    bottleneck for the leading digit **or** mean-ablation is too destructive at this
    fragile single locus — **either way selective economy is not demonstrable here**
    (an instrument limit, not a refutation of A6).
  - **Direct path NOT excluded** (`direct_scaled` = 0, underpowered — inherited CE14).

  Net: **C5 step 5 complete — the leading digit `A_top` in the hard case is produced
  by carry-specific L1-head-edge delivery to the sign-position combiner, mirroring
  CE14** (5-digit across a genuine depth spread; 6-digit at deep chains only, with
  shallow leading cascades carried by an unadjudicated path). A10 is **consolidated**
  across all answer digits *including the hardest* (with the 6-digit-deep-only rider;
  per LW-5, this does NOT raise A10 —
  it extends CE14's partial confirmation to the leading locus with the same caveats:
  sufficiency-not-necessity, direct-path-not-excluded, selection-not-shown, plus the
  leading-specific economy-bottleneck and 6-digit depth-restriction). The paper-thread
  worked example is ready with these caveats. Scoped: 2-layer addition, sign locus.

- **Run record**:
  - Command: `PYTHONPATH=. python3 scripts/leading_digit_walkthrough.py all`.
  - Script: [`scripts/leading_digit_walkthrough.py`](../../scripts/leading_digit_walkthrough.py)
    (standalone CPU; reuses CE14 `edge_patch_pred`/`_direct_patch_pred`/`cache_qs`/
    `head_ov`, graded leading-hard builder). Env: python 3.13.7, torch 2.8.0. Repo
    commit `d285797` (working tree). Date 2026-07-16. Seed 20260716. Wiring from HF
    `features.json`/`behaviors.json`.
  - Models: `add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289` (both acc 1.000).
  - Artifacts (local; no HF): `results/study-leading-digit-walkthrough/results.json`.

- **Results**:
  - **Behavioral gate**: leading-digit hard case correct at every depth k=1..n_top−1
    (1.00) in both models.
  - **Link 4 (readout-only)**: whole-`resid_mid` patch flips `A_top` 1.00 (both).
  - **Link 3 (delivery, carry-specific)**: 5-digit joint edge real 1.00 / null 0.00
    at k=1,2,3,4; 6-digit real 1.00 / null 0.00 at k=4,5, real 0.00 at k=1,2,3.
    `direct_full`/`direct_scaled` 0.00 all depths (direct arm underpowered).
  - **Link 1 (L0 ST ablation)**: 5-digit `P11L0H2` impact 0.045, `P12L0H1` 0.00;
    6-digit `P14L0H1`/`P14L0H2` 0.00 (redundant).
  - **Economy**: tagged head gaps 0.47–0.53; untagged baseline gap max 0.49 → not
    selective. Static-output: `A_top` takes 2 distinct values across graded cases
    (0 and 1) — as expected for a leading digit (not degenerate-constant here since
    the hard family exercises both).

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **R-A10-at-leading (mirrors CE14)** selected: Link 3 carry-specific — 5-digit
    across a genuine depth spread, 6-digit at deep chains only. The leading digit
    uses the same mechanism as the middle digits (CE14) — NOT a different/degenerate
    path (R-different not selected: the head edge IS carry-specific, not a static
    readout). (6-digit shallow leading cascades are carried by an unadjudicated path,
    so "the L1 head delivers" is 6-digit-deep-scoped, not uniform.)
  - **6-digit depth-restriction** is a genuine nuance: the sign-position head edge
    carries only for deep chains (k≥4); shallow leading cascades flip via Link 4
    (readout) but not these heads' edge — consistent with a redundant/distributed
    delivery (CE14) where a different path covers shallow cases, unadjudicated by the
    underpowered direct arm.
  - **Economy is an instrument limit** at the sign position (general bottleneck), not
    an A6 result either way.
  - **A10 consolidated, not raised** (LW-5): this extends CE14's partial confirmation
    to the leading locus; it does not resolve the CE14 open questions (necessity,
    selection, direct-path).

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A10**: **consolidated at the leading locus** — carry-specific L1-head-edge
    delivery to the sign-position combiner (5-digit genuine depth spread; 6-digit
    deep-chains only), mirroring CE14. Per LW-5 this does **not raise A10** (hold at
    medium); it extends the partial confirmation to all answer digits incl. the
    hardest, with the same caveats + the 6-digit depth-restriction + economy-
    uninformative-at-this-locus.
  - **A6**: **not testable at the leading locus** (the sign position is a general
    bottleneck; untagged baseline ≈ tagged) — economy neither supported nor refuted
    here.
  - **A9**: **untouched** (no same-cell selection test at the leading locus; the
    delivery is via the head pair, selection not isolated).

- **Skeptic review (post-result)**: Run 2026-07-16, separate thread (docs +
  results.json + script), independent cross-checks. **Verdict: PASS WITH
  CONDITIONS** — after seven prior cascade-line studies each needing correction,
  this one is nearly clean. The skeptic confirmed: **Link 4 is genuinely quarantined**
  (the `mirrors_CE14`/consolidate verdict is a pure function of Link 3, never the
  readout patch); **"consolidate not raise" is honored structurally** (`derive_trace`
  cannot emit `confirm`; A10 held at medium; net strictly more-caveated than CE14);
  **carry-specificity is earned** where claimed (real 1.0 / deciding-matched null 0.0);
  the **builder fix (LW-7) was legitimate** (behavioral gate passes at every depth).
  One substantive condition + two wording nudges, all applied:
  - **F1 (applied) — 6-digit "verified ≥2 depths" over-reads two adjacent maximal
    chains (k=4,5) as an independent 2-depth spread.** Re-labeled: 5-digit verified
    across a genuine spread (k=1–4); 6-digit "verified deep-chains only" — and the
    "consolidated across all answer digits" phrase now always carries the
    6-digit-deep-only rider.
  - **F2 (applied) — 6-digit shallow (k=1–3) flips via readout but NOT the head
    edge**, so "the L1 head delivers" is 6-digit-deep-scoped, not uniform; shallow
    path unadjudicated (direct arm underpowered).
  - **F6 (applied) — economy**: "general bottleneck" OR "mean-ablation too
    destructive at this fragile locus" — either way A6 not testable here.

  *Status: RESOLVED 2026-07-16 by the working thread (F1/F2/F6 wording applied; the
  science — mirrors_CE14, consolidate-not-raise — was confirmed sound). Gate 2
  PASSED.*

- **Limitations**:
  - Worked-example scope: a documented trace of the leading digit, not a new
    mechanism beyond CE14 applied at the sign locus (per LW-5).
  - 6-digit head-edge delivery is depth-restricted (deep chains only); shallow
    leading cascades are carried by an unadjudicated path (direct arm underpowered).
  - Economy (A6) is uninformative at the sign position (general bottleneck).
  - Inherits all CE14 caveats: sufficiency-not-necessity, direct-path-not-excluded,
    selection-not-shown, redundant pair, value content not head-specific.
  - 2-layer addition, two models; graded leading-hard family.

- **Doc updates** (after Gate 2 passes; APPEND to the committed C5 docs): ledger;
  claim-evidence (new **CE15**: the leading digit `A_top` in the hard case is
  produced by carry-specific L1-head-edge delivery to the sign-position combiner,
  mirroring CE14 [5-digit genuine depth spread; 6-digit deep-chains only, shallow
  unadjudicated]; Link-4 readout-only; economy uninformative [instrument limit];
  direct path not excluded);
  synthesis + summary; conjectures (**A10 consolidated at leading locus, NOT raised**
  — hold at medium; A6 not testable here; A9 untouched); agenda (complete C5 step 5
  entry 1 → the **C5 program is DONE for the addition model**; entry 1 becomes the
  mixed-model shared-engine geometry).

- **Next read**: **C5's five-step program is complete for the addition model** —
  output encodings (CE13), SV compounding (CE14), and now the leading-digit hard
  case (CE15) all land on the same picture: carry-specific attention-edge delivery
  to answer-position L1-MLP combiners, redundant and (partially) confirmed, with
  open questions on necessity/selection/direct-path. The next agenda item is the
  **mixed-model shared-engine geometry** (C5 "then repeat for a mixed model"; A7 vs
  C2). A consolidation checkpoint / adversarial referee report on the addition-model
  story (per the contract's major-checkpoint rule) is warranted before the mixed
  model, given the A10 evidence is *partial* and took heavy skeptic correction.
