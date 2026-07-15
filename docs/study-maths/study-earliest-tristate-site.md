# Study: Earliest Tri-State Site — where does `{0,1,U}` exist before it collapses? (study-earliest-tristate-site.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #7

**Verdict: no dedicated `{0,1,U}` tri-state symbol at any answer-position
residual site — the carry is binary throughout, `U` resolved around
L1-attention.** Sweeping seven residual sites (embedding → combiner) with an
axis-decomposition discriminator, the resolution/committed-orthogonal "is-U" axis
is never significant vs a permutation null (perm p 0.10–1.0) in either model.
Per-site distances reveal the trajectory: `U→1` sits nearest committed-0 through
`L1.attn_in` and flips to committed-1 at `L1.resid_mid`, so the binary resolution
is applied around **L1-attention** — never as an off-axis third symbol.

Recorded as CE7 (the fourth consecutive A3-family refutation, CE6+CE7). So the
tri-state `U` of the paper's *algorithm* is a functional description; the model's
*representation* is a binary carry, resolving `U` implicitly rather than storing
a third symbol. Scope: refutes a symbol comparable in magnitude to the binary
carry at answer positions; a weak sub-detection symbol or a transient `U` at
question positions (backlog B12) remains untested.

## Pre-run (write before the experiment)

- **Question**: Along the residual stream (from the token embedding through the
  layers, at the answer position that computes `A_{n+1}`), is there a site where
  the tri-state `U` is a **distinct, not-yet-resolved** third state —
  i.e. `U→0` and `U→1` are *together* (not split by their eventual resolution)
  and *distinct* from committed-0 and committed-1 — before the carry collapses to
  the resolved binary code CE6 found at the combiner input? If so, where; if not,
  is `U` always either resolved-binary or merely an ingredient superposition
  (`carry_in` + sum==9 flag, never a dedicated `U` symbol)?

- **Motivation**: This settles the live remnant of A3 (does a genuine tri-state
  representation exist *anywhere*, or is `U` never a dedicated symbol?), bears on
  A6 (where along the stream the cascade `U` resolves), and complements A2 (the
  nonlinear discretization step's location). It is the direct follow-up CE6
  flagged as the sharpest open question.

- **The discriminating metric (site-resolved)**: at each probe site, using the
  4-class design (committed-0 `c0`, committed-1 `c1`, U→0 `u0`, U→1 `u1`), the
  carry state can be in one of three regimes:
  - **RESOLVED**: `u0`≈`c0` and `u1`≈`c1` (split by resolution). Signature:
    `d(u0,u1)` large, `d(u0,c0)` and `d(u1,c1)` small. (This is CE6 at the
    combiner input.)
  - **UNRESOLVED TRI-STATE (a genuine third symbol)**: `u0`≈`u1` (both U cases
    together, resolution not yet applied) and the U-cluster is **distinct** from
    both `c0` and `c1` (off the 0–1 line). Signature: `d(u0,u1)` small,
    U-mean off-axis (significant vs permutation null).
  - **INGREDIENT / no dedicated symbol**: `u0`,`u1` separated but the separation
    is the raw `carry_in` bit (also present on *committed* digits) rather than a
    resolved carry_out or a dedicated U direction. Discriminated by the
    ingredient probes below.
  - The primary site-resolved statistics are the **U-split ratio**
    `d(u0,u1)/d(c0,c1)` and the **U off-axis fraction vs permutation null**,
    tracked across sites so we can see *where* the transition RESOLVED↔UNRESOLVED
    happens.

- **Hypothesis / competing reads** (neutral):
  - R-tristate-exists: at some early site (e.g. L0 output / post-L0-MLP), `U→0`
    and `U→1` are together and off-axis (a genuine `{0,1,U}` third symbol),
    resolving to binary only later. A3's live remnant supported.
  - R-never-tristate: at every site `U` is either already resolved-binary or an
    ingredient superposition; there is no site where `U` is a dedicated,
    resolution-independent third symbol. A3's remnant refuted.
  - R-gradual: the split emerges gradually across sites (partial at early sites),
    with no clean "tri-state" site.

- **Design**:
  - **Models**: `add_d5_l2_h3_t15K_s372001` (answer pos for `A_{n+1}`=`A3` is
    combiner pos 14, digit `n`=2), `add_d6_l2_h3_t20K_s173289` (pos 16, digit 3).
    Accuracy-verified (invalid if < 0.99).
  - **Probe sites** (at the combiner's answer position, increasing depth):
    `blocks.0.hook_resid_pre` (embedding+pos), `blocks.0.ln2.hook_normalized`
    (L0-MLP input, post-L0-attn), `blocks.0.hook_resid_post` (L0 output, post
    L0-MLP), `blocks.1.hook_resid_pre` (=L0 output into L1),
    `blocks.1.ln1.hook_normalized` (L1-attn input), `blocks.1.hook_resid_mid`
    (post-L1-attn), `blocks.1.ln2.hook_normalized` (combiner input, the CE6
    site). This traces embedding → L0-attn → L0-MLP → L1-attn → combiner.
  - **Also probe the make-carry head region** (CE3) and, for a genuine
    "where is U before resolution" question, the earlier *question* positions
    (D'n) if the carry state is present there.
  - **4-class stimuli** (reuse `build_class_question`): `c0`,`c1`,`u0`,`u1`, with
    lower-carry control as in CE6 (committed classes get random lower carry;
    U classes get the lower carry that sets their resolution).
  - **Metrics per site**: (1) U-split ratio `d(u0,u1)/d(c0,c1)`; (2) per-class
    distances `d(u0,c0)`,`d(u1,c1)`,`d(u0,c1)`,`d(u1,c0)`; (3) U off-axis
    fraction + permutation-null p-value (from the tri-state harness); (4)
    resolution probe `u0`-vs-`u1` (CV); (5) **ingredient probes** — `carry_in`
    decodable on *committed* digits (raw carry bit present?), and sum==9 flag
    (`U` vs committed) decodable — to distinguish "dedicated U symbol" from
    "ingredient superposition".
  - **Controls / nulls**:
    - **Permutation null** for off-axis / separation at every site (the
      digit-embedding lesson: 3–4 high-D centroids look structured by default).
    - **Committed lower-carry control** (CE6's, d≈1.2–1.7): confirms the U-split
      at a site is resolved carry, not lower-operand identity.
    - **Positive control**: the combiner-input site (CE6) must reproduce the
      RESOLVED signature (U-split large, off-axis n.s.) — a within-study anchor
      that the pipeline reports "resolved" where we know it is resolved. A
      **planted unresolved-tri-state** (synthetic: u0≈u1 off-axis) and a
      **planted resolved** (u0≈c0,u1≈c1) confirm the site-classifier separates
      the two regimes.
  - **Minimal effect / bars**: a site is UNRESOLVED-TRI-STATE only if
    `d(u0,u1)/d(c0,c1) ≤ 0.3` (U cases together) AND the U-mean off-axis fraction
    is significant (permutation p < 0.01) AND `U`-vs-committed is separable — in
    **both** models. A site is RESOLVED if `d(u0,u1)/d(c0,c1) ≥ 0.7` with
    `u0`≈`c0`,`u1`≈`c1`. Between 0.3 and 0.7 = partial/gradual.

- **Success condition** (defined now): the RESOLVED↔UNRESOLVED transition is
  located across the probed sites in both models (or shown never to be
  unresolved), with per-site U-split ratio + off-axis + ingredient probes, and
  A3's remnant scored (tri-state exists at an earlier site: yes/no).

- **Failure condition** (defined now): with the positive control passing (CE6
  site reproduces RESOLVED; planted shapes classified correctly) — no probed
  site shows the UNRESOLVED-TRI-STATE signature (`U` always resolved-binary or
  ingredient-superposition). Verdict: **no dedicated `{0,1,U}` tri-state
  representation** — A3's remnant refuted; `U` is an ingredient/resolved-bit
  phenomenon, never a dedicated symbol.

- **Ambiguous / invalid condition**:
  - Ambiguous: only R-gradual (split emerges partially, no clean tri-state site);
    or models disagree on where the transition is; or the U-split ratio sits in
    the 0.3–0.7 band at the candidate sites.
  - Invalid: positive control fails (CE6 site does not reproduce RESOLVED, or
    planted shapes mis-classified); or accuracy check fails.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + six prior study notes + the tri-state harness source; no working-thread
  context).

  **Verdict: BLOCK** — the primary metric does not match the hypothesis (the
  recurring failure mode):

  - **F1/F3 (fatal, same root)**: `u0` and `u1` differ in their **lower-digit
    operands by construction** (u0 forces no lower carry, u1 forces a lower
    carry), so `d(u0,u1)` is driven by operands regardless of any carry
    representation — the "UNRESOLVED = `d(u0,u1)` small" signature is
    unachievable at post-attention sites and degenerate (0/0) at pre-attention
    sites (where the answer position holds no operand content). Centroid distance
    is the **wrong instrument**. "A dedicated unresolved `U` symbol" actually
    predicts a **shared is-U direction** (common to u0,u1; distinct from
    committed; *not* the resolution direction) — a **subspace/axis question**.
  - **F2**: the U-mean off-axis criterion re-imports the fatal `U ≡ SA_n=9`
    confound at early sites (all U share SA_n=9); the "u0,u1 together"
    corroboration is broken by F1.
  - **F4**: the ingredient probe (`carry_in` decodable) fires **everywhere**
    (carry_in is trivially present from lower operands) → vacuous; must be
    **contrastive** (partial out carry_in; U-vs-committed must *survive*).
  - **F5/F6 (must-fix)**: planted controls don't exercise the new discriminator;
    commit **all** headline numbers to the committed script → `results.json`
    (CE6's evidence-integrity lesson).

  *Status: RESOLVED 2026-07-16 by the working thread — replaced the
  centroid-distance discriminator with an **axis-decomposition** (is-U /
  resolution / committed-carry axes) that is `SA_n=9`-immune (A-1), added an
  information-absent floor (A-2), made the ingredient probe contrastive (A-3),
  specified planted controls for the new discriminator (A-4), and committed all
  numbers to the script (A-5). Goalposts unchanged (does a dedicated `{0,1,U}`
  tri-state exist at any site). Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed. These
supersede the conflicting pre-run text where noted.

**2026-07-16 — A-1 (resolves F1/F2/F3, the core fix): AXIS DECOMPOSITION, not
centroid distance.** At each site, define three directions from class means:
  - **is-U axis** `u_hat` = mean(u0,u1) − mean(c0,c1) (does the site mark "digit
    n is U/sum==9" at all?);
  - **resolution axis** `r_hat` = u1 − u0 (is the U outcome split by the lower
    carry?);
  - **committed-carry axis** `k_hat` = c1 − c0 (the binary carry code).
  Verdicts (per site, both models, vs permutation null):
  - **RESOLVED**: `r_hat` is large and aligned with `k_hat` (|cos| high), and u0
    projects to the c0 end / u1 to the c1 end — the is-U axis is *absorbed by*
    the committed carry axis (CE6's combiner-input finding).
  - **UNRESOLVED-TRI-STATE (dedicated U symbol)**: the **is-U axis is significant
    AND largely orthogonal to both `k_hat` and `r_hat`** (a "sum==9, not yet
    decided" direction), with its variance share above the permutation null.
    This is `SA_n=9`-immune: SA_n=9 is a single resolution-independent point, so
    it cannot manufacture an is-U direction orthogonal to the resolution axis.
  - **INGREDIENT**: the only U-vs-committed separability is carried by the
    `carry_in`-collinear direction (A-3), i.e. the is-U axis vanishes after
    partialling `carry_in`.
  - **INFORMATION-ABSENT**: the four classes are not separable at the site at all
    (A-2). The U-mean off-axis-of-centroid statistic is **demoted to descriptor**
    (it is SA_n=9-confounded); the load-bearing test is the is-U-axis
    orthogonality + significance.

**2026-07-16 — A-2 (resolves F1 degeneracy): information-absent floor + same-site
separability gate.** Before classifying a site, require the 4 classes be
separable there: `d(c0,c1)` materially above within-class spread AND between-
class variance share > permutation null. A site failing this is
**INFORMATION-ABSENT** (e.g. `resid_pre` at the answer position, which holds no
operand content) — never UNRESOLVED. Store `d(c0,c1)`, within-class spread, and
between-var share per site.

**2026-07-16 — A-3 (resolves F4): contrastive ingredient probe.** Report the
`carry_in`-decodability on committed digits (expected ~vacuously high at
post-attention sites — stated in advance so vacuity is not re-discovered), then
**partial `carry_in` out**: fit the `carry_in` direction (from committed digits)
and project it out of the activations; INGREDIENT is diagnosed only if the is-U
axis / U-vs-committed separability **vanishes** after partialling. A non-INGREDIENT
verdict requires the is-U axis to **survive** partialling `carry_in`.

**2026-07-16 — A-4 (resolves F5): planted controls for the new discriminator.**
Planted synthetic 4-class sets (d=64, graded variance share 0.3–0.9, isotropic +
spectrum-matched noise, tolerance stated): (i) **planted RESOLVED** (is-U axis
collinear with committed carry; u0→c0, u1→c1); (ii) **planted UNRESOLVED**
(is-U axis present + orthogonal to resolution & committed axes; u0≈u1 in the
is-U subspace); (iii) **planted INGREDIENT** (separability only along a carry_in
direction); (iv) **planted INFORMATION-ABSENT** (no class separation). The
classifier must recover each. The CE6 combiner-input site is a *pipeline anchor*
(must reproduce RESOLVED) but NOT the positive control for the UNRESOLVED regime.

**2026-07-16 — A-5 (resolves F6): evidence integrity.** Every headline number —
per-site is-U/resolution/committed axis angles + variance shares, the partialled
ingredient gap, `d(c0,c1)` floor, between-var share, and the site verdict — is
computed in the committed `scripts/earliest_tristate_site.py` and written to
`results.json`. No number in Results/Executive-summary originates off-script.

**2026-07-16 — A-6 (F7): scope/registration.** Bands re-registered against the
axis statistics: UNRESOLVED requires is-U-axis variance share significant
(perm p < 0.01) AND |cos(is-U, committed)| < 0.5 AND |cos(is-U, resolution)| <
0.5, in both models. The make-carry-region and question-position side-probes are
**dropped** from the pre-run (no metric/null specified; a question-position carry
probe needs its own construction) — deferred to a follow-up if the answer-site
sweep motivates it. Representational framing only (a direction exists), never
causal (this site decides). `build_class_question` note: u0/u1 differ in
lower-operand distribution by construction (the basis of F1).

- **Decision impact**:
  - **Tri-state site found (R-tristate-exists)**: A3's remnant supported — a
    genuine `{0,1,U}` symbol exists early and resolves to binary later; adds a
    claim naming the site; bears on A6 (resolution happens between that site and
    the combiner) and A2 (where the nonlinear step is).
  - **No tri-state site (R-never-tristate)**: A3 fully refuted — `U` is never a
    dedicated symbol; the model handles the tri-state as ingredient/resolved-bit
    superposition. Strong simplification of the mechanism; C1's near-linear
    reading strengthened.

- **Risks / confounds**:
  - **The `U ≡ SA_n=9` structural identity** (the fatal confound from CE6's
    pre-launch gate): handled by the resolution-variable design (U→0 vs U→1) as
    the discriminator, NOT the raw off-axis of the U-mean alone. At an early site
    where U is unresolved, `SA_n=9` could place the U-mean off-axis for reasons
    unrelated to a carry `U` symbol — so "off-axis" at an early site must be
    corroborated by the U cases being *together* (low `d(u0,u1)`) AND the
    ingredient probes, not off-axis alone.
  - **Position choice**: the answer position that computes `A_{n+1}` is the
    natural site; earlier *question* positions may carry the running carry
    differently — probe both but interpret separately.
  - **LN**: report both pre-LN (`resid`) and post-LN (`ln.normalized`) where
    relevant; the "input a component reads" is post-LN.
  - **Ingredient vs symbol** (CE6 F3 lesson): a site linearly separating `u0`
    from `u1` does not prove a dedicated U symbol — the split may be the raw
    `carry_in` bit. The ingredient probes (carry_in on committed) guard this.
  - 2-layer, addition-only, single-digit `U`.

- **Expected artifacts**:
  - Standalone CPU script `scripts/earliest_tristate_site.py`.
  - Per (model, site): U-split ratio, per-class distances, off-axis + perm p,
    resolution probe, ingredient probes (carry_in-on-committed, sum9-flag),
    committed lower-carry control; site classification RESOLVED / UNRESOLVED /
    INGREDIENT / partial; planted-shape controls.
  - Plots: U-split ratio + off-axis across sites (the "resolution trajectory").
  - A site-trajectory registry JSON.
  - Local results folder `results/study-earliest-tristate-site/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary**: **No dedicated `{0,1,U}` tri-state symbol exists at any
  probed site — `U` is never a resolution-independent third state.** Sweeping
  seven residual sites (embedding → L0-attn → L0-MLP → L1-attn → combiner) at the
  answer position in both models, the **is-U axis** (U-pooled vs committed-pooled,
  orthogonalized against the resolution and committed-carry axes) is **never
  significant** vs the permutation null (perm p 0.10–1.0, never < 0.01), so there
  is no site where `U` sits off-axis as a dedicated symbol. Instead, at every
  carry-informative site the `U` cases are **fully separated by their
  resolution** (`U`-vs-committed separability 1.00, and this **survives**
  partialling out `carry_in` — so it is resolution, not the raw ingredient),
  with `U→0` tracking committed-0 and `U→1` committed-1. The earliest site
  (`L0.resid_pre` at the answer position) is correctly **information-absent**
  (identical across classes — pure positional). Verdict: **A3's live remnant
  refuted (R-never-tristate)** — the model handles the tri-state without ever
  representing `U` as a dedicated symbol; from the moment carry information
  appears at the answer position it is already resolution-split toward the binary
   carry. The planted controls confirm the discriminator *would* have detected a
   genuine unresolved tri-state (planted-unresolved: perm p 0.003, is-U
   orthogonal) **whose off-axis component is ≥~0.5× the committed-carry
   separation**; a *modest* symbol (≤~0.3×) could be missed (Gate-2 F2).
   **Trajectory (in-artifact)**: the binary resolution is *not* present at the
   earliest carry-informative sites — through `L1.attn_in`, U→1 still sits
   nearest committed-0; it flips to committed-1 only at **`L1.resid_mid`
   (post-L1-attention)**. So U is resolved to binary around **L1-attention**,
   never as an off-axis third symbol.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/earliest_tristate_site.py control`
    then `... models`.
  - Script:
    [`scripts/earliest_tristate_site.py`](../../scripts/earliest_tristate_site.py)
    (standalone CPU; reuses the tri-state harness + confirm-ST-node helpers).
    Env: python 3.13.7, macOS-26.5.2-arm64, torch 2.8.0, scikit-learn 1.7.1.
    Repo commit `368f3a9` (working tree). Date 2026-07-16.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000, digit 2, combiner pos 14),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000, digit 3, combiner pos 16).
  - Artifacts (local; no HF): `results/study-earliest-tristate-site/`:
    `results.json`, `site_trajectory_registry.json`, `isU_trajectory.png`.

- **Results**:
  - **Planted controls PASS** (discriminator validated): planted-unresolved →
    is-U-perp significant (perm p 0.003), is-U orthogonal to both axes
    (cos 0.06/0.07); planted-resolved → is-U n.s. (perm p 0.90); planted-ingredient
    → is-U aligned with the carry_in axis (cos 0.96), n.s. (0.85);
    planted-info-absent → between-var 0.002, correctly floored.
  - **Every real site: is-U-perp NOT significant** — perm p per site (5-digit):
    L0.mlp_in 0.74, L0.resid_post 0.52, L1.resid_pre 0.54, L1.attn_in 0.42,
    L1.resid_mid 0.58, combiner 0.71; (6-digit) 1.0, 0.12, 0.11, 0.10, 0.57,
    0.56. Closest approach (6-digit L1.attn_in, p 0.10) still far above α=0.01.
  - **`U`-vs-committed separability = 1.00 at every post-attention site, and
    survives carry_in-partialling (1.00/1.00)** — the U separation is
    resolution-based, not the raw `carry_in` ingredient. (INGREDIENT verdict thus
    not triggered; the split is real resolution, but *along* the committed axis,
    not a third dimension.)
  - **`L0.resid_pre` = INFO-ABSENT** (d(c0,c1)=0.0, classes identical — the
    answer position pre-attention is pure positional, no carry content).
  - **Resolution trajectory (in-artifact per-class distances, Gate-2 F3 fix)**:
    through L0 and `L1.attn_in`, *both* U cases sit nearest committed-0 (U→1 not
    yet resolved: 5-digit d(c0,u1)=15.4 < d(c1,u1)=19.9); U→1 flips to nearest
    committed-1 at **`L1.resid_mid` (post-L1-attention)** (d(c0,u1)=35.0,
    d(c1,u1)=14.0). So the binary resolution is applied around **L1-attention** —
    still never a tri-state (is-U-perp n.s. throughout; early sites have
    u0≈u1≈committed-0, not an off-axis third state).
  - Sites are now labeled RESOLVED (U split toward the committed axis) at
    carry-informative sites and INFO-ABSENT at `resid_pre` (Gate-2 F5 label fix);
    no site is UNRESOLVED-TRISTATE.

- **Interpretation** (against pre-stated conditions; goalposts unchanged):
  - **Failure condition MET → A3's live remnant refuted (R-never-tristate)**:
    with the positive/planted controls passing, **no probed site shows the
    UNRESOLVED-TRI-STATE signature** (significant orthogonal is-U axis). `U` is
    never a dedicated, resolution-independent third symbol anywhere along the
    answer-position residual stream in these models.
  - **What `U` actually is**: from the first carry-informative site, the `U`
    cases are already *resolution-split* toward the binary carry (`U→0` with
    committed-0, `U→1` with committed-1), and this split is genuine resolution
    (survives partialling `carry_in`), not the raw ingredient bit. So the model
    resolves `U` extremely early / implicitly — there is no representational
    stage where "this digit is uncertain (`U`)" is held as its own state before
    being decided.
  - **Consistency with CE6/CE5**: CE6 found the combiner input resolved-binary;
    this extends that *all the way back* to the earliest carry-informative site.
    The tri-state `U` of the Paper-2 algorithm is a *functional* description; the
    model's *representation* is binary carry throughout, with `U` handled by
    resolving it immediately rather than storing a third symbol.
  - **A6**: consistent with a carried binary carry state (no dedicated U buffer);
    but this is a static geometry read, not a cascade/tie-break test.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2):
  - **A3** ("`ST` tri-state is a 2D categorical code with `U` off the 0–1 axis"):
    **refuted for a dedicated off-axis symbol comparable to the binary carry,
    across answer-position residual sites** — no significant orthogonal is-U axis
    at any probed site in either model; `U` is resolved to binary around
    L1-attention, never as a third symbol. **Untested**: a *weak* symbol
    (≤~0.3× the committed separation, below the detection floor) and a transient
    tri-state at *question* positions (descoped). Combined with CE6, A3's
    off-axis form is disfavored across the answer-position stream, but **not
    globally dead** — the live remnant narrows to weak/question-position regimes.
  - **C1 (human)** ("simple, near-linear"): **supported** — the carry is a clean
    binary linear code throughout; the model's solution is *simpler* than a
    tri-state representation (no third symbol needed).
  - **A6**: **untouched** (static read; carried-binary-carry consistent but the
    tie-break/cascade dynamics untested).
  - **A2**: **untouched** (no aggregation/discretization measurement here).
  - Others untouched.

- **Skeptic review (post-result)**: Run 2026-07-16, separate thread (docs + JSON
  + script), with independent reproduction + extra probes. **Verdict: PASS WITH
  CONDITIONS.** The skeptic *verified the two make-or-break concerns do NOT
  hold*: (Q2) the is-U test is **non-circular** — a shared off-axis symbol
  *on top of* a resolution-split is detectable (planted: offset 1× committed-sep
  → perm p 0.002), so is-U-perp≈0 is genuine absence, not forced by the split;
  (Q1) the null is not over-conservative — a real varshare-0.2 symbol reads
  p≈0.002, so the sites' 0.2 at p≈0.1 is null-consistent incidental variance, not
  a dismissed weak hit. Four hedging/integrity conditions (all applied):

  - **F2 (power)**: the detection floor is ~0.5× the committed-carry separation;
    a **modest** symbol (≤~0.3×) could be missed. The 6-digit L1.attn_in
    "near-miss" (perm p 0.10) is null-consistent, not a weak tri-state.
  - **F3 (evidence integrity)**: the *positive* "u0~c0/u1~c1 resolution-split"
    claim was imported from CE6, not computed here. **Fixed**: the script now
    computes per-site `d_c0_u0/d_c1_u1/d_u0_u1` (in `results.json`). This also
    revealed a **trajectory** (below).
  - **F5 (labels)**: my RESOLVED criterion was mis-specified (a clean split
    *nulls* is_u, so cos can't be high). **Fixed**: RESOLVED = is-U-perp n.s. +
    U-vs-committed separable; labels now reconcile with the prose.
  - **F7 (A3 scope)**: "A3 refuted" softened to "refuted for a dedicated
    off-axis symbol comparable to the binary carry, across answer-position
    residual sites; weak (≤~0.3×) and question-position regimes untested".

  **Trajectory revealed by the F3 fix (new, in-artifact)**: through L0 and into
  L1 (up to `L1.attn_in`), *both* U cases sit nearest committed-0 (U→1 not yet
  resolved: d(c0,u1)=15.4 < d(c1,u1)=19.9, 5-digit); the split completes at
  **`L1.resid_mid` (post-L1-attention)** where U→1 flips to nearest committed-1
  (d(c0,u1)=35.0, d(c1,u1)=14.0). So the binary resolution is applied around
  **L1-attention** — and it is still *not* a tri-state (early sites have
  u0≈u1≈committed-0, is-U-perp n.s. throughout), which strengthens/localizes the
  negative.

  *Status: PASS WITH CONDITIONS — all four conditions applied (F3 by adding
  per-site distances to the committed script; F5 by fixing the label criterion;
  F2/F7 by hedging power + A3 scope below). Core negative independently
  reproduced and sound. Gate 2 PASSED.*

- **Limitations**:
  - Probed at **answer positions** only (the site computing `A_{n+1}`); a
    transient `U` tri-state at a *question* position (D'n) was descoped in the
    Gate-1 amendment (needs a different construction). "No tri-state" is scoped
    to the answer-position residual stream.
  - Single-digit `U`; 2-layer; addition only. Multi-digit `...999` cascades
    (where `U` might need to persist longer) untested.
  - The is-U-perp permutation null is the significance gate; a *very weak*
    tri-state (below the planted-control detectability) could be missed — but the
    planted-unresolved control (perm p 0.003) shows a real one at this
    signal/noise would be caught.
  - Representational, not causal (a direction's presence/absence, not a
    computation).

- **Doc updates** (after Gate 2): ledger; claim-evidence (**CE7** — no dedicated
  `{0,1,U}` tri-state at any answer-position site; `U` resolved to binary around
  L1-attention; scoped/hedged, extends CE6); synthesis + summary; conjectures
  (A3 refuted-scoped, streak note; C1 narrowly supported; A6/A2 untouched);
  agenda (complete entry 1 scoped; promote LN-aware; backlog the question-position
  `U` probe — the only place A3's remnant can live).

- **Next read**: the answer-position `U`-representation line is answered
  (no dedicated symbol; resolved around L1-attn). Per the skeptic's streak note
  (CE6 + this = repeated A3-family refutations), the productive move is **not**
  more answer-position sweeps but (a) the **question-position (D'n) transient
  `U` probe** (purpose-built construction — the only regime A3 can still live),
  and/or (b) the deeper **multi-digit `...999` cascade** test (A6 tie-break).
  Otherwise proceed to the next-ranked agenda items (LN-aware close-out;
  attention-invariance census).
