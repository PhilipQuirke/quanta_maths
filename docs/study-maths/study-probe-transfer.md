# Study: Cross-Position and Cross-Subtask Probe Transfer (study-probe-transfer.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #12

The model does the "same" little sub-calculation at every digit (weights are
shared), so a natural guess is that it stores each digit's result the *same way*,
just in a different slot — a reusable template. We tested this by training a simple
readout on one digit and checking whether it still works on another digit.

It **doesn't** — for the tri-state carry, a readout trained on one digit position
fails on the others. So despite sharing the same weights, the model writes each
digit's carry in a *position-specific* way, not a portable template. We also found
the carry-related sub-tasks are geometrically tangled together rather than kept
neatly separate. Both findings push back on a human conjecture (C2) and an agent one
(A4). A nice side-discovery: the final sum digit isn't computed at the input — it
shows up at the *answer* position, which set up the next study.

## Pre-run (write before the experiment)

- **Question**: For the per-digit addition sub-tasks computed at question
  positions — `SA_n = (Dn+D'n) % 10` (10-way), `ST_n ∈ {0,1,U}` (tri-state carry
  class), `SV_n` (resolved binary carry) — (a) **template sharing**: does a linear
  readout for a sub-task trained at one digit position `i` transfer to another
  position `j` (better than chance, and better than a cross-*subtask* transfer
  floor)? (b) **cross-subtask geometry**: are the `SA` / `ST` / `SV` subspaces at a
  fixed position more nearly orthogonal to each other than same-subtask subspaces
  across positions? (c) does the raw-vs-position-centered comparison change the
  answer (A4 predicts a *position-derived offset* — transfer should improve after
  removing a per-position mean)?

- **Motivation**: This is agenda entry 1 and a compact, largely assay-shared
  deliverable (a transfer matrix + an angle table) that scores three conjectures at
  once and needs no patching harness — a cheap breadth study on existing HF
  artifacts. C2 and A4 agree on template sharing (transfer should work) but split
  on the orthogonality half: C2 says different sub-tasks are near-orthogonal
  *because* the algorithm keeps them from interfering; A4 says at question positions
  the separation that matters is *positional*, so cross-subtask angle matters less
  than C2 implies and the interesting binding question is at the answer phase. A8's
  interference half predicts unrelated sub-task readouts are near-orthogonal (low
  cross-talk). A disagreement between the two halves (templates shared but subtasks
  entangled, or vice versa) would be a genuinely new structural fact.

- **Ground-truth facts the design uses** (self-sufficiency):
  - **Sub-task labels** are deterministic functions of the operand digits:
    `SA_n = (Dn+D'n) % 10`; `ST_n = 0` if `Dn+D'n ≤ 8`, `1` if `≥ 10`, `U` if `= 9`;
    `SV_n` = the resolved carry *into* digit `n` after the full cascade (a binary
    function of all digits `≤ n`). All computable from `(a, b)` without the model.
  - **Read site**: `blocks.0.hook_resid_post` at the **`D'n` token position** (the
    second operand of digit `n`), where L0 attention has paired `Dn`+`D'n` and the
    L0 MLP has run — the question-position site Paper 2 locates the sub-task nodes.
    Position map (6-digit, `n_ctx=22`): `D'n` at `2·n_digits − n` = {digit0:12,
    1:11, 2:10, 3:9, 4:8, 5:7}. (5-digit analogous: `2·5 − n`.) The site is
    pre-registered; a secondary read at `resid_post(L1)` and at the `Dn` position
    is reported as a robustness check, not the primary.
  - **Probes**: multinomial logistic regression (SA: 10-way; ST: 3-way; SV: 2-way)
    on the `d_model` residual, trained on a large random-addition sample, evaluated
    by held-out balanced accuracy. Linear only (the linear-representation frame;
    C2/A4 are about linear directions).

- **Hypothesis / competing reads** (neutral):
  - **R-template-shared (C2 + A4 agree)**: a probe trained on `ST_i` transfers to
    `ST_j` well above chance and above the cross-subtask floor; same for `SA`, `SV`.
    Cross-position same-subtask transfer ≫ cross-subtask same-position transfer.
  - **R-position-specific (C2/A4 falsifier)**: same-subtask probes do *not* transfer
    across positions (each position idiosyncratic) — no shared template.
  - **R-orthogonal-subtasks (C2's orthogonality half)**: `SA`/`ST`/`SV` subspaces at
    a fixed position are near-orthogonal (principal angles ≈ 90°, low mutual
    decodability) — sub-tasks do not interfere.
  - **R-entangled-subtasks (C2 falsifier / A8 falsifier)**: `SA`/`ST`/`SV` subspaces
    overlap substantially (small angles, high cross-decodability) — shared or
    interfering directions.
  - **R-positional-offset (A4-specific)**: raw cross-position transfer is *imperfect*
    but improves markedly after removing a per-position mean (the position-derived
    offset A4 posits) — distinguishes A4's "shared template + positional binding"
    from a position-invariant template.
  - **R-split (new structural fact)**: the two halves disagree (e.g. templates
    shared AND subtasks entangled) — reported as-is.

- **Design**:
  - **Models**: `add_d6_l2_h3_t20K_s173289` (primary), `add_d5_l2_h3_t15K_s372001`
    (replication). Weights via `MathsConfig`/TransformerLens, CPU; accuracy-verified
    (invalid if < 0.99). Middle digits only for the transfer matrix (exclude digit 0
    — no carry-in — and the top digit — no carry-out — as edge cases; report them
    separately).
  - **Data**: a large random-addition sample (≥ 4000 questions) with **balanced**
    sub-task classes where feasible (oversample `ST=U` and rare `SA` values so the
    probe is not dominated by the majority class). Train/test split 70/30, fixed seed.
  - **Battery T — transfer matrix (C2/A4 template half)**: for each sub-task
    S ∈ {SA, ST, SV} and each ordered position pair `(i, j)`, train a probe on
    position `i`'s activations and evaluate its balanced accuracy on position `j`.
    Report the full `position × position` transfer matrix per sub-task, both **raw**
    and **position-mean-centered** (subtract each position's mean activation before
    train/test — the A4 positional-offset arm). Summary statistics: mean
    off-diagonal transfer vs diagonal (self) accuracy; transfer retention =
    off-diag / diag.
  - **Battery A — subspace angles (C2/A8 orthogonality half)**: at a fixed position,
    take each sub-task probe's weight directions (the class-logit directions span a
    subspace: SA 9-dim after centering, ST 2-dim, SV 1-dim) and compute **principal
    angles** between the SA, ST, SV subspaces. Also **cross-decodability**: can an
    `ST` probe be read from the subspace orthogonal to `SA`/`SV`? Report the
    principal-angle table + a cross-subtask transfer floor (probe trained for `SA`,
    tested as `ST` labels — should be near chance if orthogonal).
  - **Controls / baselines**:
    - **Chance floor**: shuffle-label probe accuracy per sub-task (SA ~0.10 balanced,
      ST ~0.33, SV ~0.50).
    - **Positive control (probe validity)**: the *self* (diagonal) probe must decode
      each sub-task well above chance at its own position — proves the sub-task IS
      linearly present at the read site. If a sub-task is not decodable at all
      (diagonal ≈ chance), transfer for it is `invalid`, not "no template".
    - **Cross-subtask floor (the key comparison)**: same-position `SA`-probe applied
      to `ST` labels (and all cross pairs) — the "no shared structure" baseline that
      the same-subtask cross-position transfer must beat for R-template-shared.
    - **Position-only null**: a probe trained to predict the *digit position* from
      the activation — quantifies how much positional information is in the read site
      (bears on A4's positional-binding claim and guards against transfer being
      driven by, or defeated by, position identity).
  - **Minimal effects / bars**: template sharing (R-template-shared) requires
    off-diagonal same-subtask transfer ≥ 0.6 × diagonal accuracy AND ≥ 2× the
    cross-subtask floor, for ST and SA, in both models (or explicitly model-scoped).
    Orthogonality (C2 half) requires mean principal angle ≥ 60° AND cross-subtask
    decodability ≤ 1.5× chance. A4's positional-offset arm requires
    centered-transfer − raw-transfer ≥ 0.15 to claim a position-derived offset.
    Sample sizes: ≥ 4000 questions gives ≥ ~1000 test points per position, so a
    balanced-accuracy difference of 0.05 is resolvable well beyond noise.
  - **Pre-registered decision table**:
    | Same-subtask transfer | Cross-subtask angle | Read |
    | --- | --- | --- |
    | high (≥ bar) | orthogonal (≥ 60°) | **C2 confirmed (both halves)** |
    | high | entangled (< 60°) | **R-split**: templates shared, subtasks entangled (new fact; A8 interference challenged) |
    | low | orthogonal | **R-split**: position-specific but orthogonal (C2 template half falsified, A4 falsified) |
    | low | entangled | **R-position-specific + entangled** (C2 both halves falsified) |
    | high raw, higher centered | — | **A4 positional-offset** supported |
  - **Scope**: 2-layer addition, question-position sub-tasks, linear probes; no
    causal patching; no subtraction/mixed models. Answer-position binding (A4's
    "just-in-time fetch" claim) is a **separate** question (backlog B4), not tested
    here — this study is the question-position template/orthogonality half.

- **Positive control**: the self (diagonal) probe accuracy per sub-task per position
  (must be ≥ 2× chance to validate the read site holds the sub-task linearly). Plus
  the shuffle-label chance floor. A flat transfer result whose diagonal probe also
  fails is `invalid` / read-site-wrong, never "no template".

- **Success condition** (defined now): the transfer matrix + principal-angle table
  are produced for both models with passing diagonal controls, and select one row of
  the decision table for each half (template: shared / position-specific;
  orthogonality: orthogonal / entangled), with the A4 positional-offset arm scored.

- **Failure condition** (defined now): with diagonal controls passing —
  same-subtask cross-position transfer at/near the cross-subtask floor (no shared
  template, C2/A4 template half falsified), or `SA`/`ST`/`SV` subspaces strongly
  overlapping (C2 orthogonality half + A8 interference falsified).

- **Ambiguous / invalid condition**:
  - Ambiguous: transfer between the floor and the bar; the two models disagree;
    principal angles in the 45–60° mid-band.
  - Invalid: a sub-task's diagonal probe is at chance (read site wrong for it); model
    accuracy < 0.99; severe class imbalance the balancing did not fix.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + glossary + claim-evidence + the reused harness; token/position map
  verified in code). **Verdict: PASS WITH CONDITIONS** — four blocking findings,
  all improving discriminating power:
  - **C1 (blocking) — SA read site.** CE2/CE3 place base-add/SA at *answer*
    positions, and the pre-check found SA near chance at `D'n`@L0 — so the plan
    pre-dooms the SA half to `invalid`. Validate each sub-task's read site
    empirically (a pre-flight over {`D'n`@L0, `D'n`@L1, `Dn`@L0, answer-pos@L0}) and
    read each sub-task at *its own* validated site.
  - **C2 (blocking) — C2-vs-A4 discrimination.** The cross-subtask *angle* does not
    separate C2 from A4 (both survive either outcome). The split must be carried by
    the **positional-offset (centered−raw) + position-only-null** axis, not the
    angle axis. Rebuild the decision table accordingly.
  - **C3 (blocking) — label-correlation confounds.** SA/ST/SV labels are correlated,
    so (a) same-subtask cross-position transfer needs a **cross-position
    cross-subtask floor** (train SA@i, test ST@j) to beat, not just the same-position
    floor; (b) the angle/cross-decodability side needs a **label-correlation null**
    (entangled only if angles are small *relative to* the correlation null).
  - **C4 (blocking) — angle geometry.** Principal angles between LR *weight*
    directions are whitening-distorted; apply the ≥ 60° bar to **class-mean
    activation-subspace** angles instead (LR-weight angles secondary).
  - C5/C6 (non-blocking): pre-register the ST-only degenerate outcome as acceptable
    + read SV at `=`/answer; balance classes on train AND test, and flag
    low-variance-projection templates.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments T-1…T-6 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede conflicting pre-run text where noted.

**2026-07-16 — T-1 (resolves C1): per-subtask validated read site via a pre-flight.**
Before the transfer matrix, run a **read-site pre-flight**: for each sub-task
S ∈ {SA, ST, SV} and each candidate site ∈ {`D'n`@resid_post(L0), `D'n`@resid_post(L1),
`Dn`@resid_post(L0), answer-position@resid_post(L0), `=`@resid_post(L1)}, fit the
diagonal probe at a middle digit and record balanced accuracy. Each sub-task's
**own read site** is the earliest candidate where its diagonal ≥ 2× chance; the
transfer matrix for that sub-task is computed at *its* site (sub-tasks need not
share a site). The full pre-flight table goes to `results.json`. A sub-task is
`invalid` only if it is < 2× chance at *every* candidate site.

**2026-07-16 — T-2 (resolves C2): C2-vs-A4 split decided by the positional axis,
not the angle.** The cross-subtask **principal-angle** battery tests **C2's
orthogonality half and A8's interference half only** — NOT the C2/A4 split. The
**C2/A4 split** is decided by: (i) the **positional-offset arm** (centered−raw
same-subtask transfer ≥ 0.15 ⇒ A4's position-derived offset) and (ii) the
**position-only null** (how much digit-position identity the read site carries).
Amended decision rows: *transfer high, raw ≈ centered, position-null low* ⇒
**position-invariant template (C2-style; A4 positional-binding arm unsupported)*;
*transfer high only after centering, position-null high* ⇒ **A4 positional-offset
supported**. The angle table feeds only the orthogonality/A8 verdict.

**2026-07-16 — T-3 (resolves C3): cross-position cross-subtask floor + angle
label-correlation null.** (a) Same-subtask cross-position transfer must beat the
**cross-position cross-subtask floor** (train SA@i, test ST@j) — removing both the
template and the position confounds — not merely the same-position cross-subtask
floor. (b) The orthogonality/cross-decodability side is scored against a
**label-correlation null**: recompute cross-subtask decodability on
correlation-matched permuted labels (preserve each sub-task's marginal correlation
with the others, break the shared-direction hypothesis); "entangled" is claimed
only if observed cross-decodability materially exceeds this null.

**2026-07-16 — T-4 (resolves C4): angles on class-mean activation subspaces.** The
≥ 60° orthogonality bar is applied to principal angles between the **class-mean
activation subspaces** (per sub-task: the subspace spanned by its class-conditional
mean vectors μ_class − μ_global, i.e. a PCA of the class-mean matrix), not the LR
weight directions. LR-weight angles are reported as a secondary comparison; a
disagreement between the two is flagged `ambiguous`.

**2026-07-16 — T-5 (resolves C5): ST-only degenerate outcome pre-accepted; SV read
late.** If only ST has a passing diagonal at any site, the template-sharing verdict
is **ST-scoped** and stated as such (not "no template" for SA/SV). SV is read at its
resolved site (`=`/answer position per the glossary), not pre-doomed at the early
question site.

**2026-07-16 — T-6 (resolves C6): balanced train+test; low-variance flag.** Classes
are balanced (oversampled) on **both** train and test splits at every position;
per-position class counts to `results.json`. A passing transfer that rides a
sub-task subspace occupying < 3% of activation variance is annotated
"low-variance, mechanistic weight uncertain", not a clean confirmation.

**2026-07-16 — T-7 (implementation fix, discovered on first run — restructures the
analysis, no goalpost change): all cross-subtask comparisons at ONE shared
question site; per-subtask best-site diagonal reported separately.** The first run
exposed that letting each sub-task pick its own best read site makes the
cross-subtask floor and principal-angle comparisons impossible (they require a
common site and, for the floor, comparable label cardinality) and conflates SA's
*answer-position readout* (diag ≈ 1.00 at `ans_L0`, a trivial per-position readout)
with a *question-position template*. Fix: the transfer-matrix and cross-subtask
batteries are computed at the **pre-registered primary question site `Dpn_L0`** for
all three sub-tasks (this is the question-position template/orthogonality question
C2/A4 actually pose); the read-site pre-flight is reported separately as a
descriptive *where-does-each-subtask-live* result (and confirms SA lives at the
answer position, ST/SV at the question site — itself an A4-relevant finding).
Because SA at `Dpn_L0` is weak (~0.17–0.44, above chance but modest), the SA
template verdict is explicitly *weak/low-confidence*, not "invalid". The
cross-subtask floor is made cardinality-robust by scoring **chance-relative
accuracy gain** (accuracy − chance) so SA(10-way)/ST(3-way)/SV(2-way) are
comparable; principal angles are computed at `Dpn_L0` for all pairs. The
`lowvar_frac` metric is corrected (the first-run value > 1 was a normalization
bug: fraction = variance in the class-mean subspace ÷ total activation variance).

- **Decision impact**:
  - **C2 confirmed (both halves)**: C2 → higher confidence; A4's template half
    supported; A8 interference half supported.
  - **R-split (templates shared, subtasks entangled)**: a genuinely new structural
    fact — the algorithm reuses templates but does *not* keep sub-tasks orthogonal;
    challenges A8's no-interference claim and C2's orthogonality half; interesting.
  - **A4 positional-offset supported**: A4's "shared template + positional binding"
    mechanism gets direct evidence (raw transfer imperfect, centered transfer good).
  - **Template half falsified**: both C2 and A4 lose their shared prediction — a big
    update (position-specific idiosyncratic codes).

- **Risks / confounds** (with mitigations):
  - **Position identity leaks into the probe**: the read site contains the
    positional embedding, so a probe could exploit position rather than the sub-task,
    or transfer could fail purely from a position offset. Mitigation: the raw vs
    position-mean-centered arms; the position-only null quantifies how much position
    info is present.
  - **Label correlation between sub-tasks**: `ST` and `SV` and `SA` are correlated
    (e.g. `ST=1 ⇒ SV contribution`), so a "cross-subtask transfer" could be real
    label correlation, not shared geometry. Mitigation: the cross-subtask floor is
    computed on the *same* data, so it already includes label correlation; the
    template claim must beat *that* floor, not chance.
  - **Class imbalance** (`ST=U` is rare — only sum=9; `SA` uniform-ish): balance by
    oversampling; report balanced accuracy, not raw.
  - **Probe overfitting / regularization**: L2-regularized logistic regression, fixed
    C, train/test split, report test accuracy only; the diagonal control uses the
    same regularization so transfer is comparable.
  - **Low-variance-projection trap** (the recurring lesson): a "shared template" that
    is a low-variance direction could transfer trivially; report the probe margin and
    the fraction of activation variance the sub-task subspace occupies.
  - **`SV` at question positions may not be resolved yet** (the cascade resolves
    later): `SV_n` decodability at the `D'n` question site may be low by construction
    — if the diagonal `SV` probe is at chance, `SV` transfer is `invalid` (read site
    too early), reported honestly, not "no SV template".

- **Expected artifacts**:
  - Standalone CPU script `scripts/probe_transfer.py` (reuses `load_model`,
    `make_q`; sklearn logistic regression).
  - `results/study-probe-transfer/`: `results.json` (every headline number —
    diagonal accuracies, full transfer matrices raw + centered, principal-angle
    tables, cross-subtask floors, chance floors, position-only null); plots
    (transfer heatmaps per sub-task, principal-angle bars). No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (ST-scoped per Gate 2): **A new structural fact for the
  tri-state carry `ST`: at question positions its representation is
  POSITION-SPECIFIC — an `ST` probe trained at digit `i` does NOT transfer to digit
  `j` — falsifying the C2/A4 shared-template prediction for `ST`; and `ST` is
  geometrically ENTANGLED with the resolved carry `SV` beyond what their labels
  force, challenging C2's orthogonality half and A8's low-interference claim.** `ST`
  is the one sub-task with a solid diagonal probe at the shared question site
  (`D'n`@L0: diag gain 0.31/0.39 in the two models), so it is the sub-task whose
  transfer is validly assessable there. Its transfer matrix is strong on the
  diagonal, near-chance off-diagonal (6-digit ST diagonal 0.48–0.92 vs off-diagonal
  ≈ chance 0.33), chance-relative **retention 0.10/0.12** (bar 0.6), both models —
  no shared cross-position template. Crucially, **mean-centering does not restore
  transfer** (positional-offset gain ≤ 0.09 < 0.15), so this is not a mere
  per-position mean offset (A4's specific positional-binding mechanism) — the
  representation is position-specific more deeply. (That digit position is perfectly
  decodable, acc 1.00, is corroborating but is a trivial consequence of positional
  embeddings, not the mechanism.)

  On orthogonality, no `SA`/`ST`/`SV` pair reaches 60° (all 21–49°), and against the
  **label-correlation null** the `ST`–`SV` overlap is genuine geometric
  **entanglement** (observed 21° ≪ null 64°/50°, both models) — and since `ST_n`
  and `SV_n` are label-*independent* at the same digit (MI≈0), this is a real shared
  direction, not a label near-identity. `SA`–`SV` overlap is explained by label
  correlation (observed ≈ null).

  **SA and SV are not assessable for transfer at this question site** — they are
  diag-weak there (SA near chance at `D'n`; SA decodes perfectly, 1.00, only at the
  **answer position**, confirming CE2/CE3). So the template-sharing falsification is
  **ST-scoped**; SA/SV template sharing at their own home sites (and the
  answer-phase "tape vs register" binding question) is untested here (backlog B4).
  Net (both models agree): for the tri-state carry at question positions, the
  picture is **position-specific + entangled with SV**, not C2's
  **shared-template + orthogonal** nor A4's **transfer-for-free**.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/probe_transfer.py preflight` then
    `... all`.
  - Script: [`scripts/probe_transfer.py`](../../scripts/probe_transfer.py)
    (standalone CPU; reuses `load_model`/`make_q`; sklearn logistic regression +
    scipy `subspace_angles`). Env: python 3.13.7, macOS-26.5.2-arm64, torch 2.8.0,
    sklearn. Repo commit `92edb5f` (working tree). Date 2026-07-16. Seed 20260716,
    n_q = 4000, C = 0.5, balanced train+test.
  - Models: `add_d6_l2_h3_t20K_s173289` (acc 1.000, digits 1–4),
    `add_d5_l2_h3_t15K_s372001` (acc 1.000, digits 1–3).
  - Artifacts (local; no HF): `results/study-probe-transfer/results.json`,
    `transfer_<model>.png`.

- **Results**:
  - **Read-site pre-flight (T-1)**: SA is near-chance at all question sites
    (`D'n`@L0 0.17–0.44) but **1.00 at the answer position** (confirms CE2/CE3:
    base-add lives at answer positions); ST decodable at the question site
    (`D'n`@L0 0.58–0.85) and 1.00 at the answer position; SV 0.60–0.67 at question
    sites, 0.94–0.95 at the answer position. Per T-7, the transfer/angle batteries
    use the shared question site `D'n`@L0.
    (`ST` is the sub-task with a passing diagonal here — diag gain 0.31/0.39; SA
    diag gain 0.10 and SV 0.08 are diag-weak at this site, so their transfer is
    underpowered/invalid here, not "no template".)
  - **Transfer (T-2/T-3), both models — ST is the valid case**: chance-relative
    **retention_gain** ST 0.10–0.12 (far below the 0.6 bar); off-diagonal gain ≈
    chance; transfer matrix strong diagonal, near-chance off-diagonal (6-digit ST
    diagonal 0.48–0.92, off-diagonal ≈ 0.33). (SA 0.02–0.13, SV −0.01–0.26 are on
    diag-weak bases — reported but not verdict-driving.) **Positional-offset gain**
    (centered−raw) ≤ 0.09 (< 0.15) — mean-centering does *not* rescue ST transfer,
    so A4's per-position "position-derived offset" is not the explanation either.
  - **Cross-subtask floor**: computed **same-position** (decode a sub-task from
    another sub-task's class-mean subspace at the mid digit) — a label-correlation
    control; ST off-diagonal transfer does not beat even this floor. (One exception:
    d5 SV offdiag 0.038 > its floor 0.034 — SV being diag-weak/not-verdict-driving.)
  - **Position-only null**: digit position decodes at **1.00** — corroborating (the
    site carries position), but this is a trivial positional-embedding consequence,
    NOT the mechanism (the mechanism evidence is centering-fails, above).
  - **Subspace angles (T-4) vs label-correlation null (T-3b)**: `ST`–`SV` observed
    21.2° / 21.7° vs null 64.2° / 50.7° → **genuinely entangled** (both models);
    `SA`–`SV` 36.9°/48.7° vs null 48.4°/46.9° → overlap ≈ label correlation
    (not extra entanglement); `SA`–`ST` 34.1°/42.3° vs null 29.7°/22.3° → mildly
    more aligned than the null. No pair reaches 60° (orthogonal).
  - **Low-variance flag (T-6)**: the class-mean subspaces occupy 11–41% of
    activation variance — not a low-variance-projection artifact.

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **Template half: R-position-specific selected for `ST` — C2/A4 shared-template
    FALSIFIED for the tri-state carry at question positions.** With ST's diagonal
    control passing, ST cross-position transfer is at/near the (same-position)
    cross-subtask floor in both models (the pre-registered failure condition). SA/SV
    are diag-weak at this site → their transfer is not-assessable here, not "no
    template".
  - **Orthogonality half: R-entangled selected (partly) — C2 orthogonality half
    challenged.** No pair is orthogonal (all < 60°); `ST`–`SV` is entangled *beyond*
    label correlation. A8's low-interference prediction is challenged for the
    carry-family sub-tasks.
  - **A4 positional-offset: not supported** — centering does not restore transfer
    (gain < 0.15); the representation is position-specific in a way a per-position
    mean offset does not capture. A4's *transfer-for-free* prediction fails; its
    *position-is-the-binding-tag* spirit survives only in the weak sense that
    position is perfectly decodable.
  - **Consistency**: both models agree on all three verdicts — a rare clean
    cross-model replication in this thread.
  - **Scope caveat**: this is the *question-position* representation. SA's real home
    is the answer position (diag 1.00 there); whether SA/SV transfer across *answer*
    positions (the A4 just-in-time-fetch / "tape vs register" question) is untested
    here (B4).

- **Prediction scoring** (records evidence; conjecture updates after Gate 2;
  ST-scoped per the post-result gate):
  - **C2 (human)**: **challenged for `ST` at question positions.** Template sharing
    **falsified for ST** (no cross-position transfer, centering doesn't help);
    cross-subtask orthogonality **challenged** (`ST`–`SV` entangled beyond label
    correlation, no pair ≥ 60°). SA/SV template sharing not-assessable at this site.
    Lower C2 confidence for the question-position `ST` case. (Human-owned; noted.)
  - **A4**: the **shared-template / transfer-for-free** prediction is **falsified
    for `ST`** at question positions (no transfer, centering doesn't restore it — so
    not even A4's per-position-offset form). A4's positional-binding *spirit* is not
    contradicted, but its concrete transfer claim fails for ST. Lower A4's
    template-sharing sub-claim (was medium-high).
  - **A8**: the **low-interference** prediction is **challenged** — `ST`–`SV` share
    overlapping directions beyond (label-independent) chance, not near-orthogonal
    dedicated subspaces. Lower A8 (interference half) for the carry-family at
    question positions.
  - **A1/A3**: untouched (not a geometry-of-a-single-subtask study).
  - **Scope note**: all updates are ST / ST–SV and question-position-specific; SA's
    home is the answer position (untested for transfer — B4).

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + script), with independent checks (ST class balance across
  digits ~45/45/10; ST/SV label independence at the same digit MI≈0.0002).
  **Verdict: PASS WITH CONDITIONS.** The **ST** falsification is verified earned and
  clean (strong diagonal in both models, retention 0.10/0.12 ≪ 0.6, off-diagonal ≈
  chance, passing positive control) and the positional-offset discriminator + the
  ST–SV entanglement (with a correctly-implemented label-correlation null) are
  sound. Conditions:
  - **F1-post (BLOCKING) — SA/SV are diag-weak at the shared question site.** At
    `Dpn_L0` (6-digit) SA `diag_gain` 0.10, SV 0.08 — near-absent, so their "no
    transfer" is **underpowered/invalid there**, not a clean falsification. The
    template-half falsification must be **ST-scoped**; SA/SV template sharing is
    not-assessable at the question site (SA's home is the answer position).
  - **F2-post (BLOCKING for CE11) — "position decodes at 1.00" is a trivial
    positional-embedding consequence**, not the cause of no-transfer. Demote it to a
    non-diagnostic corroborator; lead the mechanism with the **centering-fails**
    result (positional_offset_gain < 0.15), the correct T-2 discriminator.
  - **F3-post (BLOCKING) — the implemented cross-subtask floor is same-position**
    (project onto another sub-task's class-mean subspace at the mid digit), NOT the
    cross-position floor T-3a described. Fix the note wording to match the code (a
    valid same-position label-correlation floor); note the one d5-SV exception
    (offdiag 0.038 > floor 0.034).
  - **F4-post (non-blocking) — correct the ST/SV rationale**: at the same digit,
    ST_n and SV_n are label-*independent* (MI≈0), so the 21° ≪ 64° null overlap is
    *purely geometric* — which strengthens (not weakens) the entanglement finding.
  - **F5-post** — T-7 confirmed a legitimate amendment (uses the pre-registered
    primary site, makes the falsification harder if anything), not a goalpost move.
  - **F6-post (non-blocking) — scope the conjecture lowering to ST** (template half)
    and to ST–SV (orthogonality/A8).

  *Status: RESOLVED 2026-07-16 by the working thread. Amendments T-8/T-9 applied:
  the falsification is narrowed to **ST** (SA/SV not-assessable at the question
  site); the position-1.00 gloss is demoted; the floor wording corrected to
  same-position; the ST/SV independence rationale fixed; conjecture updates scoped
  to ST / ST–SV. The corrected Executive summary / scoring below supersede the
  first draft. Gate 2 PASSED on the corrected, ST-scoped read.*

**2026-07-16 — T-8 (resolves F1/F2/F3/F6-post): falsification narrowed to ST;
mechanism = centering-fails, not position-1.00; floor is same-position.** The
template-half result is **ST-scoped**: only ST has a passing diagonal at the shared
question site (`Dpn_L0`), so only ST's no-transfer is a valid falsification; SA and
SV are diag-weak there (their sub-task lives elsewhere — SA at the answer position)
and are **not-assessable** at this site, not "no template". The "position-specific"
mechanism is carried by the **positional-offset arm** (mean-centering does not
restore ST transfer: gain ≤ 0.09 < 0.15), NOT by "position decodes at 1.00" (a
trivial positional-embedding fact, demoted to corroborator). The cross-subtask
floor is the **same-position** class-mean-subspace decodability (a label-correlation
control), not a cross-position floor.

**2026-07-16 — T-9 (resolves F4-post): ST/SV are label-independent at the same
digit.** ST_n (local sum class) and SV_n (carry *into* n) are ~independent at a
fixed digit (MI≈0), so the ST–SV subspace overlap (21° ≪ 64° null, both models) is
**purely geometric entanglement**, not a label near-identity — strengthening the
A8-interference challenge for the carry-family sub-tasks.

- **Limitations**:
  - Question-position representation only; SA's true home (answer position) is not
    transfer-tested — the answer-phase "tape vs register" binding question (A4
    just-in-time fetch) is backlog B4.
  - Linear probes only (C2/A4 are linear claims); a shared template realized
    non-linearly would be missed — but the position-only 1.00 decode and the
    near-chance off-diagonal are strong linear evidence for position-specificity.
  - "Position-specific" here means *not linearly transferable*; the same computation
    is still applied by shared weights (weight sharing is architectural) — the
    finding is that the *representation* at the read site is position-tied, not that
    the model uses different algorithms per digit.
  - 2-layer addition, two models (both agree); middle digits only.

- **Doc updates** (after Gate 2 passes): ledger; claim-evidence (new **CE11**:
  the tri-state carry `ST` has a **position-specific** representation at question
  positions [an `ST` probe does not transfer across digits; mean-centering doesn't
  restore it] and is **entangled with `SV`** beyond their (independent) labels —
  C2's template + orthogonality halves, A4's transfer-for-free, and A8's
  interference all challenged **for `ST` at question positions**; SA lives at the
  answer position, not assessable here); synthesis + summary; conjectures (C2
  lowered for the ST question-position case — noted; A4 template-sharing sub-claim
  lowered; A8 interference half lowered for the carry-family); agenda (complete
  entry 1; the answer-position transfer / tape-vs-register question B4 gains
  priority as the natural follow-up).

- **Next read**: the surprise is the *position-specificity* — despite weight
  sharing, the question-position representation does not present a linearly-shared
  template across digits, and position is perfectly decodable. The natural
  follow-up is **B4** (answer-position coexistence / transfer: do SA/SV transfer
  across *answer* positions, and are several resolved per-digit states orthogonally
  co-stored at `=`/answer — the "tape vs register" A4 question), now the sharpest
  open binding question. Whether the entangled `ST`/`SV` overlap is functional
  (B2 neuron-level) is secondary.
