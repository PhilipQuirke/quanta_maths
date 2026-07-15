# Study: LN-Aware Digit-Embedding Close-Out (A-9) (study-ln-aware-embedding.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #9

**Verdict: the CE1 circular-ordering signal persists in the LN-effective geometry
— but this is a *weak* robustness result and the signal is seed-fragile.** Two
qualifiers lead: (1) the learned γ is near-constant (std ~0.005, β≈0), so LN is
**near-isometric** on the digit geometry (`normalize-only` ≡ `full-LN` ordering
p), meaning LN was structurally near-incapable of reshaping an angular statistic —
"survives LN" is weak robustness, not a passed stress test; (2) the ordering
holds in **2 of 3 independent seeds** (primary + repA both `s372001`, repC
`s572091`), while the remaining seed repB (`s173289`) fails raw AND LN (ordering p
0.048→0.089). With those caveats, the angular-ordering permutation p is nearly
identical raw vs LN in every model (primary/repA/repC 0.0001↔0.0001), freq-1 is
unchanged (~0.24–0.28, no dominant geometry), and the model-true
`ln1.hook_normalized` (Option B) reproduces the pattern at the actual digit
positions.

Recorded as a CE1 revision: the ordering half is de-provisionalized to **"Low;
LN-robust (weakly, near-isometric LN) but seed-fragile (2/3 seeds)"** —
robustness rises modestly, magnitude and seed-generality do not. The near-isotropic
/ no-dominant-geometry half of CE1 is confirmed under LN. The raw-vs-LN
`disagree=True` flags are a classifier-boundary artifact (R1↔partial label flip on
the `wraparound_ratio` 2.0 cutoff at identical freq-1 and ordering-p), not a
geometry change. This closes the embedding-geometry line at the representational
level; the only deeper embedding question left is *causal* (does the model use the
ordering — backlog B1).

## Pre-run (write before the experiment)

- **Question**: When the digit-token embeddings are transformed by the first
  LayerNorm (the operation the model actually applies before layer-0 attention
  reads them), does the CE1 verdict hold? Specifically: (a) does the near-
  isotropic / no-dominant-geometry result survive; (b) does the weak circular
  *ordering* signal (the provisional half of CE1) survive, strengthen, or vanish;
  and (c) does the raw-vs-LN comparison agree or disagree per the pre-registered
  A-9 rule (disagreement = a change in classified read R1/R2/R3/R4 or a crossing
  of the 25%/40% variance band)?

- **Motivation**: This is the cheapest queued item (weights-only, reuses the
  digit-embedding harness) and it **de-provisionalizes or revises the CE1
  ordering claim** — currently the only Low/provisional part of an otherwise
  settled claim. LN is a genuine nonlinearity (centering + normalization + learned
  γ/β) that can reshape a weak signal, so "the effective geometry the computation
  sees" may differ from raw `W_E`.

- **What "LN-effective embedding" means (design rationale)**: the model computes
  `resid_pre = W_E[digit] + W_pos[pos]`, then `ln1` applies center-over-`d_model`
  + normalize-by-std + scale-by-`γ` + shift-by-`β` before layer-0 attention reads
  it. Two principled LN-effective representations, both reported:
  - **(A) position-free LN of `W_E`**: apply the LN operation directly to the 10
    raw `W_E` digit rows (center each row over `d_model`, normalize, apply
    `γ,β`). Directly comparable to the raw-`W_E` analysis (position-independent),
    isolating the LN transform of the digit code itself.
  - **(B) model-true `ln1.hook_normalized`**: run each digit token at its natural
    `D_n` input position through the model and capture
    `blocks.0.ln1.hook_normalized` at that position — exactly what the model
    sees (includes the positional contribution). Report per a couple of digit
    positions to check position sensitivity.
  - The **primary verdict** uses (A) (the clean apples-to-apples comparison with
    the raw analysis); (B) is the reality check that the position term does not
    overturn it.

- **Design**:
  - **Models**: the same set as the digit-embedding study — accurate
    `add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t15K_s372001`,
    `add_d6_l2_h3_t20K_s173289`, `add_d6_l2_h3_t20K_s572091`; plus the untrained
    control (freshly-initialized primary config). Weights via
    `MathsConfig`/TransformerLens, CPU.
  - **Metrics** (reuse `scripts/digit_embedding_geometry.py` `analyze_matrix`
    verbatim on the LN-effective 10×`d_model` matrix): centered variance spectrum
    / participation ratio; freq-1-plane variance share; angular-ordering statistic
    with the 10k-draw **label-permutation null** (the CE1 ordering signal); the
    R1/R2/R3/R4 classification.
  - **Comparison**: for each accurate model, the raw verdict (from CE1) vs the
    LN-effective verdict. Report agreement/disagreement per the **A-9 rule**
    (classified-read change or 25%/40% band crossing).
  - **Untrained control**: LN-effective untrained embedding must show the
    baseline (no significant ordering) — confirms LN does not *manufacture*
    ordering.
  - **Positive control**: the positive-control machinery in the harness (planted
    circle detected at the pre-registered level; permutation null calibrated on
    noise) is inherited from the digit-embedding study; re-affirm the permutation
    null's false-positive rate on the untrained LN-effective matrix.
  - **Minimal effect**: same as the digit-embedding study — the ordering signal
    is "present" if the angular-ordering permutation p < 0.01 in ≥ 2 of the
    accurate models AND absent in the untrained control; freq-1 variance is a
    dominant-geometry signal only at ≥ 40% (unchanged bar).

- **Success condition** (defined now): the LN-effective geometry is measured for
  all accurate models + untrained control, the raw-vs-LN agreement is reported
  per the A-9 rule, and the CE1 ordering claim is **de-provisionalized**
  (if LN agrees: ordering survives / vanishes consistently) **or revised**
  (if LN disagrees: mark ambiguous per A-9). A1's ordering sub-claim is re-scored.

- **Failure / ambiguous / invalid**:
  - Not a hypothesis test with a "failure" verdict per se — it is a close-out.
    The substantive outcomes are: **ordering survives LN** (de-provisionalize CE1
    ordering as a real weak effect), **ordering vanishes under LN** (revise CE1:
    the ordering was a raw-weights artifact not seen by the computation), or
    **raw-vs-LN disagree** (A-9 → mark ordering ambiguous).
  - Invalid: model load/accuracy failure, or the untrained LN-effective control
    shows spurious significant ordering (would indicate LN manufactures the
    signal, voiding the comparison).

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + the original digit-embedding study incl. its gates + the reused harness
  source; no working-thread context).

  **Verdict: PASS WITH CONDITIONS** — two make-or-break findings:

  - **C-1 (F2, sharpest)**: the pre-registered A-9 disagreement rule is defined
    on the **classified read (R1/R2/R3/R4) / freq1 band** — but the *provisional*
    quantity this study closes is the **angular-ordering permutation p**, which
    the classified read is blind to (freq1 is at the noise floor, so the read is
    pinned regardless of ordering). Ordering could vanish under LN while the rule
    reports "agree". Must extend disagreement to cover an ordering-significance
    crossing.
  - **C-2 (F1)**: LN centers **per-vector over d_model**, which composes with the
    harness's **per-digit centering over the 10 rows** (double-centering); and γ
    is a non-constant per-dimension rescale, so the freq1 *variance share* is
    **not** LN-scale-invariant (my note's claim was wrong). Need a
    planted-circle-through-LN sanity control + report normalize-only vs full-γ/β.
  - **C-3**: recompute BOTH raw and LN headline numbers in the committed script →
    `results.json` (don't hand-copy raw from CE1 — the recurring evidence-integrity
    trap).
  - **C-4**: commit the untrained-LN ordering check + permutation-null FPR.
  - **C-5/C-8**: foreground that "ambiguous / LN-sensitive" is a legitimate modal
    outcome for a boundary signal in a 4-model sample; on survival, CE1 ordering
    stays **Low but LN-robust** (de-provisionalized, not strengthened).

  *Status: RESOLVED 2026-07-16 by the working thread — A-9 rule extended to the
  ordering signal (A-11), LN transform validated via planted-circle-through-LN +
  normalize-only/full-γ/β split (A-12), both-arms-in-script committed (A-13),
  untrained-LN control committed (A-14), ambiguous-is-fine + no-upgrade framing
  (A-15). Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed.

**2026-07-16 — A-11 (resolves C-1, the sharpest fix): the disagreement rule
covers the ORDERING signal.** For this study, raw-vs-LN **disagreement** fires if
EITHER (i) the classified read changes / a freq1 25%/40% band is crossed (the
original A-9), OR (ii) the **angular-ordering permutation p crosses the p<0.01
boundary in any accurate model**, OR (iii) the **count of accurate models with
significant ordering changes** (raw = 3/4). The **ordering-p table (raw vs LN,
per model)** is the PRIMARY comparison object; the classified-read table is
secondary. This closes the gap where the freq1-pinned read would mask an
ordering flip.

**2026-07-16 — A-12 (resolves C-2): LN transform validated + γ isolated.**
- Correction: the freq1 *variance share* is NOT LN-scale-invariant, because γ
  (`ln1.w`) is a non-constant per-dimension rescale that can move variance
  between the freq1 plane and the rest. So the study reports two LN variants:
  **normalize-only** (center per-vector + unit-std, no γ/β) and **full LN**
  (+ γ/β), isolating γ/β's contribution.
- **Planted-circle-through-LN sanity control**: apply the exact Option-A LN
  pipeline to (a) a planted freq1 circle (must still be detected at the
  positive-control level) and (b) the untrained `W_E` (must NOT gain ordering).
  This validates the LN-*composed* pipeline, since the inherited positive control
  only validated the raw pipeline.
- The double-centering (LN per-vector, then harness per-digit) is stated
  explicitly; (A) position-free is primary for comparability, (B) model-true
  `ln1.hook_normalized` is the faithfulness check. If (A) and (B) disagree on the
  classified read OR ordering significance → **ambiguous** (not "(A) wins").
- `eps = 1e-5` (TransformerLens `LN` default); γ (`ln1.w`) and β (`ln1.b`) read
  from the same raw `.pth` state dict `load_WE_WU` uses. Report γ spread.

**2026-07-16 — A-13 (resolves C-3): both arms in the committed script.**
`scripts/ln_aware_embedding.py` recomputes, in one run, the **raw** and the
**LN-effective** statistics (freq1 share, angular-ordering p, participation
ratio, classified read) for all 4 accurate models + untrained control, and
writes all headline numbers + the raw-vs-LN agreement table to
`results/study-ln-aware-embedding/results.json`. The raw arm calls the identical
`analyze_matrix` on raw `W_E`, reproducing CE1's numbers as a cross-check; if the
reproduced raw numbers do not match CE1 within permutation noise, the run is
`invalid` (load/config drift). No headline number exists only as prose. RNG
re-seeded deterministically per matrix so numbers are order-independent; seed +
N_PERM recorded.

**2026-07-16 — A-14 (resolves C-4): untrained-LN control committed.** The
untrained LN-effective matrix is run through the permutation null (FPR at α=0.01
must be ≤ 0.02) and its angular-ordering p must NOT be < 0.01 — confirming LN
does not manufacture ordering. Stored to `results.json`.

**2026-07-16 — A-15 (resolves C-5/C-8): ambiguous is fine; survival ≠ upgrade.**
Given a signal already at the significance boundary in a 4-model sample
(raw: 3/4 significant, repB at p=0.045), **"ordering remains ambiguous / LN-
sensitive" is a legitimate and possibly-modal outcome** and the study will not
force de-provisionalization. A single-model significance flip with the other
three unchanged is scored **ambiguous/LN-sensitive**, not "confirmed". On the
survival branch, CE1 ordering is de-provisionalized to **remains Low, now
LN-robust** (confidence rises in *robustness*, not *magnitude*) — NOT to Medium.
Option-B positions: the operand digit input positions actually used
(`Dn`/`D'n`); position-inconsistent ordering significance in (B) is reported as
position-sensitivity, a caveat on the (A) primary.

- **Decision impact**:
  - **Ordering survives LN**: CE1 ordering de-provisionalized to Low→Medium
    (a real, weak, LN-robust training signal); A1's "some circular ordering
    exists" sub-claim confirmed (weak).
  - **Ordering vanishes under LN**: CE1 revised — the ordering was raw-`W_E` only,
    not in the effective geometry; A1's ordering sub-claim refuted; strengthens
    the "no functional digit geometry" reading and C1's simple-code view.
  - **Disagree (A-9)**: CE1 ordering marked ambiguous; note the LN sensitivity.

- **Risks / confounds**:
  - **Position term (B)**: `ln1` sees `W_E + W_pos`, so the model-true LN-effective
    embedding mixes in position. Handled by reporting the position-free (A) as
    primary and (B) as a position-sensitivity check across ≥ 2 digit positions.
  - **LN normalization changes scale, not just direction**: the freq-1 *variance
    share* and the *angular ordering* are both scale-invariant-ish (share is a
    ratio; ordering is angular), so LN's rescale should not by itself change the
    verdicts — but centering over `d_model` (a genuine projection) can. Report
    the spectrum so any centering-induced rank change is visible.
  - **Small n (10 vectors)**: same as the digit-embedding study; the permutation
    null is the significance gate, not asymptotics.
  - **γ/β near-degenerate**: if `γ` is near-constant the LN transform is ~a pure
    normalize; report `γ` spread so the transform's non-triviality is documented.

- **Expected artifacts**:
  - Standalone CPU script `scripts/ln_aware_embedding.py` (reusing the
    digit-embedding harness metrics).
  - Per model: raw vs LN-effective (A) freq-1 share, angular-ordering perm p,
    participation ratio, classified read; the model-true (B) check at ≥ 2
    positions; untrained LN-effective control; raw-vs-LN agreement per A-9.
  - A small comparison table (raw vs LN verdict per model) + the CE1 disposition.
  - Local results folder `results/study-ln-aware-embedding/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: **The CE1 circular-ordering signal persists in the
  LN-effective geometry — but this is a *weak* robustness result and the signal
  remains seed-fragile.** Two important qualifiers lead the verdict: (1) the
  learned γ is near-constant (std ~0.005, β≈0), so **LN is near-isometric on the
  digit geometry** — `normalize-only` and `full-LN` give identical ordering-p —
  meaning LN was structurally near-incapable of reshaping an angular statistic,
  so "survives LN" is a *weak* form of robustness, not a passed stress test.
  (2) The ordering holds in **2 of 3 independent seeds** (primary + repA both
  seed `s372001`; repC `s572091`); the one remaining independent seed, repB
  (`s173289`), **fails raw AND LN** (ordering p 0.048→0.089) — so the signal is
  seed-fragile. With those caveats: the angular-ordering permutation p is nearly
  identical raw vs LN in every model, the same models significant under both
  (primary/repA/repC 0.0001↔0.0001), freq-1 unchanged (~0.24–0.28, no dominant
  geometry), and the model-true `ln1.hook_normalized` (Option B) reproduces the
  pattern at the actual digit positions. **Verdict: CE1 ordering
  de-provisionalized to "Low; LN-robust (weakly, near-isometric LN) but
  seed-fragile (2/3 seeds)"** — robustness rises modestly, magnitude does not.
  The raw-vs-LN `disagree=True` flags are a classifier-boundary artifact (the
  R1↔partial/ambiguous label flips on the `wraparound_ratio` 2.0 cutoff) at
  *identical* freq-1 and ordering-p — not a geometry change (see F4 below).

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/ln_aware_embedding.py control`
    then `... models`.
  - Script: [`scripts/ln_aware_embedding.py`](../../scripts/ln_aware_embedding.py)
    (reuses the digit-embedding harness metrics). Env: python 3.13.7,
    macOS-26.5.2-arm64, torch 2.8.0. Repo commit `368f3a9` (working tree). Date
    2026-07-16. `eps=1e-5`, `N_PERM=10000`, seed 20260716.
  - Models: the four accurate models (`add_d5_l2_h3_t15K_s372001`,
    `add_d6_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`,
    `add_d6_l2_h3_t20K_s572091`) + untrained control.
  - Artifacts (local; no HF): `results/study-ln-aware-embedding/results.json`.

- **Results**:
  - **Controls PASS (validate the LN-composed pipeline, C-2)**: a planted circle
    survives LN (freq-1 0.605→0.606, ordering p 0.0005 both); the untrained
    LN-effective embedding does **not** gain ordering (raw p 0.33 → LN p 0.56,
    both n.s.); the permutation-null false-positive rate is 0.01 at α=0.01
    (calibrated).
  - **Ordering-p (PRIMARY, A-11), raw → LN-full**: primary 0.0001→0.0001; repA
    (d6,s372001) 0.0001→0.0001; repC (d6,s572091) 0.0001→0.0001; repB (d6,s173289)
    0.048→0.089. Significant (p<0.01) in **3/4 raw AND 3/4 LN** — the same three
    models. `normalize-only` LN is identical to full LN (γ≈constant).
  - **freq-1 share** unchanged raw→LN (~0.24–0.28, all in the 25–40% "mid" band;
    no band crossing). γ spread: mean 0.914–0.939, std 0.005–0.006; β≈0.
  - **Option B (model-true `ln1.hook_normalized`)**: at the operand digit
    positions, ordering p is 0.0001–0.0002 for the three ordering-significant
    models and 0.06–0.17 for repB — reproducing the (A) pattern; position does
    not overturn the verdict.
  - **A-11 agreement**: `ordering_sig_cross=False` and `freq1_band_cross=False`
    in all four models; the only trigger is a `read_change` (R1↔partial/ambiguous)
    at *identical* freq-1 shares — a classifier-boundary label artifact at the
    pinned noise-floor share, not a geometry change.

- **Interpretation** (corrected post-Gate-2; goalposts unchanged):
  - **Outcome: ordering persists under LN → CE1 ordering de-provisionalized, with
    two qualifiers.** By the A-11 rule the *substantive* comparison (ordering-p,
    freq-1 band) **agrees** raw vs LN in all models. But (F2) LN is
    **near-isometric** here (γ std ~0.005; normalize-only ≡ full-LN), so surviving
    it is a *weak* robustness result — LN could barely reshape the geometry. And
    (F3) the ordering is significant in only **2 of 3 independent seeds**
    (fails s173289 raw and LN). Per A-15 this de-provisionalizes CE1's ordering
    half to **"Low; LN-robust (weakly) but seed-fragile"** — robustness rises
    modestly, magnitude and seed-generality do not.
  - **`disagree=True` reconciliation (F4)**: the committed rule flags disagree in
    3/4 models, but 100% via the R1↔partial/ambiguous classified-read flip (a
    `wraparound_ratio` crossing the R1 2.0 cutoff) at identical freq-1 (0.27) and
    ordering-p (0.0001). `ordering_sig_cross=False`, `freq1_band_cross=False`
    everywhere. Per A-11 (ordering-p + freq-1-band primary) the verdict is
    **agreement on the substance**; the classified-read is an **unstable harness
    readout** and is not leaned on.
  - The near-isotropic / no-dominant-geometry half of CE1 is confirmed under LN
    (freq-1 unchanged, still noise-floor).

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A1** (ordering sub-claim: "digits carry a weak circular ordering"):
    **weak ordering persists under LN, but robustness is modest** — LN is
    near-isometric here (γ std ~0.005) so it was structurally unlikely to change
    an angular statistic; and the signal holds in only **2 of 3 independent
    seeds** (fails s173289). Confirms *robustness-to-LN* (weakly) and
    position-robustness, NOT strength (still +~0.04 over baseline) or
    seed-generality. Not scored "confirmed" outright.
  - **C1 (human)** ("simple, near-linear"): **weakly / narrowly supported** — LN
    introduces no curvature, but that is near-guaranteed by γ≈const, so this is
    low-information evidence for C1's thesis, not a genuine test.
  - **A8**: **untouched** (near-isotropic embedding confirmed under LN, but the
    activation-level A8 claim is untested here).
  - Others untouched.

- **Skeptic review (post-result)**: Run 2026-07-16, separate thread (docs + JSON
  + script), independently verified. **Verdict: PASS WITH CONDITIONS** —
  evidence integrity clean (the raw arm reproduces CE1 bit-for-bit; every
  headline number is in `results.json`). Four wording/scoping conditions (all
  applied, no re-run):

  - **F2 (near-isometry)**: γ std ~0.005 (mean ~0.92, β≈0), so LN is
    **near-isometric** on the digit geometry — `normalize-only` and `full-LN`
    ordering-p are identical. "Survives LN" is therefore a **weak** form of
    robustness (LN was structurally near-incapable of reshaping an angular
    statistic here). Must lead the headline, not sit in Limitations.
  - **F3 (seed-fragility)**: the 3 significant models are 2 *independent seeds*
    (primary + repA both `s372001`; repC `s572091`); repB (`s173289`) fails raw
    AND LN. So it is **2 of 3 seeds**, not "3/4 independent replications" — the
    signal is seed-fragile, and "LN-robust" must not paper over that.
  - **F4 (rule reconciliation)**: the committed `agreement()` outputs
    `disagree=True` in 3/4 models — driven entirely by the R1↔partial/ambiguous
    classified-read flip, which is a **wraparound-ratio classifier-boundary
    artifact** (LN pushes wraparound_ratio across the 2.0 cutoff) at *identical*
    freq1 and *identical* ordering-p. Per the pre-registered A-11 (ordering-p +
    freq1-band are PRIMARY, classified-read secondary), the substantive
    comparison agrees; but I must state the "rule says disagree / I conclude
    agreement" reconciliation explicitly and record the classified-read
    instability as a harness caveat.
  - **F6 (scoring)**: soften A1 to "weak ordering persists under a near-isometric
    LN, 2/3 seeds — robustness modest, not confirmed-strong"; C1 to "weakly/
    narrowly supported (near-isometric LN adds no curvature; low info given
    γ≈const)".

  *Status: RESOLVED 2026-07-16 by the working thread — F2 near-isometry now leads
  the exec summary; F3 2/3-seed framing applied in interpretation + scoring + the
  CE1 revision; F4 reconciliation + read-instability caveat recorded; F6 A1/C1
  softened. Data and controls are sound (unchanged). Gate 2 PASSED.*

  **F4 reconciliation (recorded):** per the committed script,
  `raw_vs_lnfull_agreement.disagree = True` in primary/repA/repC — but this is
  100% the classified-read label flip (a `wraparound_ratio` crossing the 2.0 R1
  cutoff) at freq1 0.271→0.272 and ordering-p 0.0001→0.0001. `ordering_sig_cross
  = False` and `freq1_band_cross = False` in all four models. Per A-11 (ordering-p
  + freq1-band primary), the substantive verdict is **agreement**; the
  classified-read (R1/partial/ambiguous) is an **unstable harness readout** under
  a near-isometric transform and should not be leaned on (Option B shows the same
  instability: primary is R1 at one digit position, partial/ambiguous at
  another).

- **Limitations**:
  - Weights-only; the ordering is a *representational* tendency, not shown to be
    *used* (causal — backlog B1).
  - γ near-constant in these models, so LN barely reshaped the geometry — the
    LN-robustness result is partly "LN happens to be near-isometric here", which
    is itself informative but model-family-specific.
  - The signal remains weak (boundary significance; repB fails); "LN-robust" does
    not make it strong.
  - Single 10-token vocabulary, n=10 per matrix; permutation null is the
    significance gate.
  - The classified-read label (R1/partial/ambiguous) is unstable at the
    noise-floor freq-1 share and should not be over-read (the ordering-p is the
    load-bearing statistic).

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (revise
  **CE1**: ordering half de-provisionalized to **Low; LN-robust weakly
  [near-isometric] but seed-fragile [2/3 seeds]**; remove the "pending A-9"
  caveat but *replace* it with the near-isometry + seed-fragility caveats);
  synthesis + summary (drop "provisional", add the seed-fragility qualifier);
  conjectures (A1 ordering sub-claim: weak, LN-robust modestly, 2/3 seeds; C1
  weakly supported); agenda (complete entry 1 as `question answered`).

- **Next read**: CE1 is now settled (near-isotropic + weak, LN-robust-but-
  seed-fragile ordering). The embedding-geometry line is closed at the
  representational level; proceed to the next-ranked agenda item
  (attention-pattern invariance census). The only deeper embedding question left
  is *causal* (does the model use the ordering — backlog B1).
