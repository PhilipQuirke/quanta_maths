# Study: Compounding Locus — Decisive A11 (L0 relay) vs L1-read Assay (study-compounding-locus.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #20

The one open piece of the SV mechanism was *where* single-digit carries get
compounded into the full "999→carry-all-the-way-up" resolved carry. CE17 couldn't
tell — was it a **sequential relay across the question-tail positions at layer 0**
(A11, the human's sequential lean), or a computation **inside the final-layer
consumer head's read** (L1-read)? The blocker was that on normal carry chains a
site's local digit-sum and the resolved carry are perfectly correlated, so you
can't tell whether a site "knows" the resolved carry or just its own digit.

This study found a **clean natural decorrelation**: a tri-state site sitting *inside*
the 999-run has local sum fixed at 9 (uninformative) while the resolved carry
reaching it still flips 0/1. Split those into two kinds — cells where the site
**can see** the deciding digit (could compute the carry itself) vs cells where the
deciding digit is **below its visibility horizon** (the carry could *only* have been
relayed to it). The invisible cells are the decisive test for A11.

Result (both models, run 2026-07-16; post-result-corrected):
**A11 (positional L0 relay) is unsupported — refuted where testable (5d), and
underpowered where the deep-chain writes wash out (6d).**
- **5d = R-L1-read (properly controlled)**: at the invisible-decorrelated cell
  (P9H1 k=3) the write is **readable at its own depth** (a known-present local
  signal decodes) yet the **resolved carry decodes at chance (0.57)** — so no carry
  was relayed to a site that couldn't see the deciding digit. The multi-digit
  resolution happens in/after the **L1 consumer read**, not at the L0 write.
- **6d = inconclusive/underpowered [post-result F2]**: at the invisible-decorrelated
  cells the deep-chain ST write is **not readable even for a known-present local
  signal** (the write washes out at k=3/4), so the chance carry-decode there cannot
  distinguish "no relay" from "unreadable write". 6d does **not** support A11 but
  cannot positively refute it either.
- **No relay signal anywhere**: across both models, no invisible-decorrelated cell
  decodes a relayed carry; the ST write decodes carry only at the fully-correlated
  cell (site = deciding digit, 0.99–1.00, a linear code) — the CE13 local
  single-step picture, with **no evidence of cross-position relay**.
- **Knock-out (class-necessity anchor, DH is the load-bearing battery [F4])**:
  ablating the whole sufficient-ST class breaks the digit (differential 0.24–0.50
  over a 0.00 specificity-null and 0.00 untagged baseline — the class is genuinely
  carry-necessary), but KO does **not** discriminate relay from local resolution
  (leave-deepest-survives is consistent with both); only DH discriminates.

Bottom line: **no evidence the cascade is a sequential positional relay at L0**;
where the instrument has power (5d), the resolved carry is provably absent from the
L0 write and the **compounding is completed in the L1 consumer read** (then
discretized by the step combiner, CE17). This is a **linear-probe** result: a
non-linear/different-subspace relay is not excluded, and 6d is underpowered — so
**A11 → low (not supported), not formally rejected**. Does not revive A9; the
human's sequential-cascade lean is not supported at the L0-tail. This substantially
resolves the last open SV-mechanism question (with the stated caveats). Dual-gated
(pre-launch BLOCK→LOC-1…LOC-6; post-result PASS-WITH-CORRECTIONS F1–F4). Filed as
CE19.

Status: **pre-run written 2026-07-16; SPRINT study** (agenda entry 1, `maths`
thread; a separate thread owns entry 2, the mixed-model replication — no shared
files). Gate: **single combined skeptic pass** (pre-launch + post-result in one
thread). Evidence-integrity rules unchanged (all headline numbers from the
committed script into `results.json`).

## Pre-run (write before the experiment)

- **Framing (working axioms)**: the multi-digit cascade is computed *somewhere*
  between the operand tokens and the combiner input (established: CE16 shows a
  canonical resolved carry arrives on the consumer edge; CE13 shows individual ST
  writes hold only local class + single-step resolution). This study **locates**
  that computation. It is not an existence test — it estimates *where* (L0 tail
  relay, [A11](../maths-conjectures-agent.md#a11-multi-digit-compounding-is-a-positional-l0-relay-across-the-question-tail-st-sites),
  vs inside the L1 consumer read) with an instrument built to survive the two
  blockers that made [CE17](study-compounding-arithmetic.md) come back R-mixed.

- **The two CE17 blockers this redesign must clear** (the whole point):
  1. **Interchange nulls are causally undetermined** (CE17-F2): single-site
     twin-interchange of an ST write flips the leading digit 0.00 at every depth,
     but a valid ablation control tests a *different unit/target*, so 0.00 cannot
     be split into "redundant" vs "interchange-too-weak". → This study uses a
     **class-level cumulative-knockout causal arm** (below) that redundancy cannot
     mask, instead of single-site interchange.
  2. **The consumer edge is local-class-sufficient** (CE17 Battery L: the edge
     output reconstructs from per-site local class at r²≈0.95, ΔR²_horizon≈0), so
     "L1 reads local" was not separable from "L0 relayed a resolved carry" —
     because on chain stimuli each site's **local class and the resolved carry are
     correlated**. → This study exploits a **natural decorrelation** (below) so
     the two are statistically independent, on-distribution.

- **The decorrelation lever (verified 2026-07-16, the key idea)**: on a chain
  `C(n,k,·)`, a chain-ST site tagging digit `j` that sits **inside the 999-run**
  (i.e. `j > d`, the deciding digit is *below* it) has **local class fixed at U
  (pair-sum = 9)** while the **resolved carry reaching it varies 0↔1** with the
  deciding digit far below. Concretely for 6d P11H2 (digit 2, horizon m=1): at
  k=2 (deciding = digit 2) local class tracks the carry (the CE17-confounded
  case), but at **k=3 and k=4 (deciding below digit 2) local class ≡ U is
  CONSTANT while carry_out(top) flips 0↔1**. So decoding the resolved carry from
  a 9-run site's L0 write at k≥3 **cannot be local-class leakage** — local class
  carries zero information there. This is on-distribution (no synthetic stimuli),
  resolving the entry-1 design question in favour of natural decorrelation.

- **Ground-truth facts / reused assets** (self-sufficiency):
  - **Models**: `add_d6_l2_h3_t20K_s173289` (primary, richest chain depth),
    `add_d5_l2_h3_t15K_s372001` (replication). Both acc 1.000. Chain-top n_top=4
    (6d) / 3 (5d), depths k∈{2,3,4}/{2,3}.
  - **Instruments reused (CE17/CE13/CE16)**: `chain_st_sites` + `horizon_of_site`
    (visibility m per site); `chain_carry_out` (ground-truth resolved carry);
    `_site_ov_write` (OV-projected L0 write capture); `_mean_ablate_multi` /
    `_mean_ablate_acc` (CE13 ablation unit); `build_chain` / `twin_pair`
    (matched pairs); `pair_at_top` / `edge_patch_pred` (consumer edge);
    `carry_axis` (CE6). All n-parameterized.
  - **The horizon rule (CE17, re-derived correct)**: a site at horizon `m`
    resolves a chain with deciding digit `d` iff `d ≥ m` (it must see the deciding
    operand pair). A 9-run *decorrelation* site is one with `d < j` (deciding
    below the tagged digit) — for such a site local class ≡ U.

- **Hypotheses / competing reads**:
  - **A11-relay**: on the decorrelated (9-run, k≥3) sites, the L0 write **decodes
    the resolved carry** (above wrole/shuffled baselines) despite fixed local
    class, matching the site's visibility horizon; AND the class-level knockout
    arm shows the relay chain is causally load-bearing (removing the whole
    sufficient-relay set breaks the digit; the surviving deepest relay suffices).
  - **R-L1-read**: the decorrelated-site L0 write does **not** decode the resolved
    carry beyond local class (flat at chance when local class is held at U), i.e.
    the resolution is not present at L0 — it happens in the L1 read.
  - **R-mixed / irreducible**: partial (some sites/depths decode, the causal arm
    stays redundancy-masked) — reported as the measured split, and if the causal
    arm cannot separate even at class level, the A11 disposition is stated as
    **final-mixed with the reason it is irreducible at this 2-layer architecture**.

- **Design** (three batteries; the first two are the CE17-blocker-beaters):
  - **Battery DH — Decorrelated Horizon decode (beats blocker 2)**: restrict to
    **9-run decorrelation sites** (tagged digit `j` with a deciding digit `d < j`,
    so local class ≡ U). For each such (site, depth), decode `carry_out(top)` from
    the site's L0 OV write and score against (i) wrole co-located-head baseline,
    (ii) label-shuffled null, (iii) a **local-class control**: since local class ≡
    U is constant here, a local-class decoder is *at chance by construction* —
    verify it (must be ≈0.5), which is the positive proof that any carry signal is
    NOT local leakage. A11 predicts decode ≥ baselines+0.2 exactly when `d ≥ m`
    (visible); R-L1-read predicts chance everywhere on decorrelated sites.
    Requires ≥2 independent decorrelation sites and both models (hardening the
    CE17 single-cell trace).
  - **Battery KO — cumulative Knock-Out causal arm (beats blocker 1)**: instead of
    single-site interchange (undetermined), test the relay chain at **class
    level**. Order the sufficient relays by horizon (deepest-visible first). Arms:
    (a) **all-sufficient-relays mean-ablated** → expect the leading digit breaks
    if the relay path is load-bearing (class-necessity, redundancy cannot mask);
    (b) **leave-deepest-in**: ablate all sufficient relays EXCEPT the deepest
    visible one → if the deepest relay carries the resolved carry (A11), the digit
    survives; if it doesn't (L1 does the work), it breaks like (a); (c)
    **leave-shallowest-in** (control): ablate all except an insufficient/shallow
    relay → should break (the shallow one can't resolve). Baseline: the same
    knockout on an untagged co-located-head set (CE13 unit). Deciding-matched
    stimuli. This is a *necessity/sufficiency* test on the relay SET, which
    redundancy predicts will be informative even though single nodes are not
    (working axiom: class-level necessity).
  - **Battery RC — Reconstruction on decorrelated stimuli (confirmatory, cheap)**:
    the CE17 Battery-L reconstruction of the consumer edge output, but computed
    ONLY on the decorrelated 9-run subset where local class ≡ U. If φ_horizon
    still beats φ_local here (where φ_local is constant, so any predictive power
    is horizon), the relay content reaches the consumer; if not, the consumer edge
    carries no more than local class even when local class is uninformative →
    L1-read. Held-out R²; the CE17 ceiling is removed because φ_local is
    degenerate on this subset.
  - **Not re-run**: A10 iv (combiner is a step, CE17 DONE). Neuron decomposition
    (B2) out of scope.

- **Pre-registered decision table** (LOC-4, keyed on the INVISIBLE-decorrelated
  cell `d<m<j` — the only cell that separates the three mechanisms):
  | DH invisible-decorrelated | DH visible-decorrelated | KO (leave-deepest, invisible) | Read |
  | --- | --- | --- | --- |
  | decodes carry > baseline | decodes | survives differentially over null; all-ablate breaks | **A11-relay** (carry relayed across positions) |
  | chance | decodes | breaks at invisible, survives at visible | **Self-computation at L0** (each site sums what it sees; A11-relay refuted, NOT L1-read) |
  | chance | chance | leave-deepest breaks like all-ablate | **R-L1-read** (resolution inside the consumer read) |
  | underpowered (no invisible cell clears baseline) | — | — | **inconclusive / single-cell** (A11 stays low) |
  RC corroborates on the invisible subset (φ_horizon>φ_local there ⇒ relay reaches
  the consumer). CONFIRMED requires the discriminating (invisible) cell to clear
  its baseline with a non-overlapping CI — unreachable if only P11H2 qualifies
  (capped at "relay-consistent single-cell trace").

- **Positive controls** (fail → battery `invalid`, never a negative): (1) acc ≥
  0.99; (2) Battery-DH local-class control ≈ 0.5 on decorrelated sites (proves
  decorrelation — the load-bearing validity check); (3) Battery-KO instrument:
  the all-sufficient-relays ablation must break the digit above the untagged
  baseline (reproduces CE13/CE18 class-ablation impact) — else KO invalid; (4)
  a correlated-site positive control for DH (at k where the site IS the deciding
  digit, local class decodes carry ~1.0 — instrument can read the write); (5)
  behavioral gates per depth.

- **Success condition**: DH + KO + RC agree on relay / L1-read (per the table), or
  their disagreement is reported as the final A11 disposition with the reason
  (redundancy-irreducibility at 2 layers). The decorrelation control (2) and the
  KO instrument control (3) must pass for any causal claim.

- **Failure condition**: only instrument failure (a battery's control fails →
  that battery invalid) or irreducible instability (estimates vary beyond CI
  without pattern → "heterogeneous", reported with numbers). R-mixed is a result.

- **Skeptic review (combined, sprint) — PRE-LAUNCH half**: Run 2026-07-16 in a
  separate skeptic thread. **Verdict: BLOCK → resolved to PASS via amendments
  LOC-1…LOC-6.** The decorrelation lever was verified real (kills blocker 2), but
  two decisive design flaws were caught and fixed:
  - **C1 (blocking) — DH on VISIBLE-decorrelated cells does not discriminate
    A11-relay from "the site sums what it sees".** On a decorrelated cell with
    `d ≥ m` the site *can see* the deciding operands (that's what `d≥m` means), so
    decoding the resolved carry there is equally predicted by A11-relay AND by "this
    L0 site computes the carry itself from visible operands" — a local computation,
    not a positional relay. Fix (LOC-1): the discriminating cells are the
    **INVISIBLE-decorrelated** ones (`d < m < j`): the deciding digit is *below the
    site's horizon* (can't self-compute) yet still inside its 9-run (local class ≡
    U). A carry decode THERE can only be a **relayed** carry (A11); chance there but
    decode at visible cells = **self-computation (A11-relay refuted, not L1-read)** —
    a new third disposition. Verified invisible-decorrelated cells: 6d P11H2/k=4
    (d=0<m=1), P10H2/k=3,4 (weak writer); 5d P9H1/k=3, P6H2. Add the promised
    value-shuffled null. If no invisible-decorrelated cell clears its wrole
    baseline+0.2, DH is `underpowered`, not evidence either way.
  - **C2 (blocking) — Battery KO reproduces CE17-F2 at set level.** "Leave-deepest-in
    survives" is not attributable to the relayed carry without (a) excluding
    route-around (the consumer re-computing from the surviving site's visible
    operands) and (b) a specificity control (set mean-ablation perturbs the residual
    broadly). Fix (LOC-2): every KO arm gets a **deciding-matched specificity null**
    (run on carry-free/committed + `same_class_twin` stimuli — a carry-attributable
    break must be DIFFERENTIAL over the null, with CIs); AND a **leave-deepest-in at
    an INVISIBLE-decorrelated depth** (if the surviving deepest site couldn't
    self-compute yet the digit survives, a relay reached it = A11; if it breaks
    there but survives at visible depths = self-computation). KO is `undetermined`
    if the specificity null isn't cleared.
  - **C3 (power ceiling) — "≥2 sites both models" is not met for the discriminating
    test.** Only P11H2 (6d) has a genuine horizon crossing on decorrelated stimuli;
    m=0 sites are presence-only (trivial); P10H2 is a weak writer; 5d P9H1 was
    CE17-null. Fix (LOC-3): count only sites with BOTH a visible and an invisible
    decorrelated depth as "discriminating"; pre-register that 6d has exactly one
    (P11H2), 5d expected underpowered. **If DH rests on P11H2 alone the verdict is
    capped at "single-cell, not replicated" — cannot reach CONFIRMED**, only
    "relay-consistent trace" or "self-computation" or underpowered.
  - **C4 — decision table targets the wrong cells.** Fix (LOC-4): rebuild the table
    around the invisible-decorrelated cell as the A11 signal; add the
    **self-computation** row as a first-class third outcome.
  - **C5 — RC φ_horizon confounded the same way.** Fix (LOC-5): report RC on visible
    vs invisible decorrelated subsets separately; only the invisible subset
    discriminates relay from self-computation.
  - **C6 — CIs / underpowered scoring.** Fix (LOC-6): per-cell wrole baseline +
    bootstrap CI on every DH bacc; a cell not clearing baseline+0.2 with
    non-overlapping CI is `underpowered`, never L1-read evidence.

  *Status: RESOLVED 2026-07-16 via LOC-1…LOC-6. The study now targets the
  invisible-decorrelated cells as the decisive relay-vs-self-computation-vs-L1
  discriminator, with an honest single-cell power ceiling at 6d.*

## Amendments (post-skeptic pre-launch, sprint)

**2026-07-16 — LOC-1 (C1): the DISCRIMINATING test is decode on INVISIBLE-
decorrelated cells** (`d < m < j`: deciding digit below the site's horizon, still
inside its 9-run). Decode>baseline there ⇒ **relayed** carry (A11); chance there
while decoding at visible cells ⇒ **self-computation** (A11-relay refuted, not
L1-read); chance everywhere ⇒ L1-read. Visible-decorrelated cells (`m ≤ d < j`)
are demoted to a presence control. Add a value-shuffled null.

**2026-07-16 — LOC-2 (C2): every KO arm gets a deciding-matched specificity null**
(carry-free/committed + `same_class_twin`; break must be DIFFERENTIAL over the
null, CIs), plus a **leave-deepest-in at an invisible-decorrelated depth**. KO
`undetermined` if the null isn't cleared.

**2026-07-16 — LOC-3 (C3): only sites with both a visible AND an invisible
decorrelated depth count as discriminating** (6d: P11H2 only; 5d: expected
underpowered). Verdict capped at "single-cell / not replicated" if DH rests on
P11H2 alone — CONFIRMED unreachable, honest ceiling stated up front.

**2026-07-16 — LOC-4 (C4): decision table rebuilt** around the invisible-
decorrelated cell; **self-computation** added as a first-class third outcome.

**2026-07-16 — LOC-5 (C5): RC reported on visible vs invisible decorrelated
subsets separately** (only invisible discriminates relay from self-computation).

**2026-07-16 — LOC-6 (C6): per-cell wrole baseline + bootstrap CI on every DH
bacc**; cell not clearing baseline+0.2 with non-overlapping CI = `underpowered`.

- **Skeptic review (combined, sprint) — POST-RESULT half**: **PENDING** — one combined pass;
  rehydrate from this note, the conjecture files (working axioms + A11 + A9 + A6),
  document rules, agenda, CE13/CE16/CE17/CE18 notes + results.json. Suggested
  audit focus: (a) is the **decorrelation** real and complete — is local class
  truly uninformative at the 9-run sites (control 2), or does some residual
  operand structure leak the carry; (b) does **Battery KO** actually escape the
  CE17-F2 ambiguity — is "leave-deepest-in survives" separable from "the model
  routed around ablation entirely" (needs the all-ablate-breaks arm to bite as
  the necessity anchor); (c) is "≥2 decorrelation sites, both models" enough to
  call replication given CE17's single-cell fragility; (d) is the final-mixed
  disposition honestly falsifiable or an escape hatch.

- **Decision impact**: closes (or finally scopes) the last open SV-mechanism
  question for the paper hand-off (entry 3). A11-relay confirmed → the paper's
  cascade description becomes "sequential positional relay at L0 across the
  question tail + L1 fetch + step-combine", partially vindicating the human's
  sequential lean at L0; R-L1-read → the compounding is an L1-read computation,
  sequential-at-L0 refuted; final-mixed → stated as an architectural
  irreducibility. No paper edits from this thread; feeds the entry-3 hand-off.

- **Risks / confounds**:
  - **Residual operand leakage on decorrelation sites**: even with local class ≡
    U, the specific operand *values* at the site differ across carry-in — control
    (2) (local-class decoder at chance) + a value-shuffled null guard this.
  - **KO "survives" ambiguity**: mitigated by requiring all-ablate to break
    (necessity anchor) so "survives" means "deepest relay is sufficient", not
    "ablation had no route".
  - **Redundancy irreducibility**: if even the class-level KO cannot separate, the
    honest output is final-mixed — pre-registered, not an escape (the reason is
    stated: 2-layer redundant relays are not individually necessary).
  - **Time**: DH + KO are the core (never dropped); RC is cheap; 5d replication is
    the only droppable scope.

- **Expected artifacts**: standalone CPU script `scripts/compounding_locus.py`
  (reuses CE17/CE13/CE16 modules); `results/study-compounding-locus/`:
  `results.json` (per-site decorrelated-decode table + baselines + local-class
  control, KO arm flips + necessity anchor + baseline, RC R² on decorrelated
  subset, all controls). No HF uploads.

## Post-run (filled 2026-07-16)

- **Run record**: `PYTHONPATH=. python3 scripts/compounding_locus.py all` (CPU).
  Both models acc 1.000. n=250 decode Qs, 40 causal pairs, 200 instrument-control
  Qs; bootstrap CIs on all DH baccs. Artifact
  `results/study-compounding-locus/results.json`. Seed 20260716. Script
  `scripts/compounding_locus.py` (reuses CE13/CE16/CE17 modules). Nothing dropped.

- **Results** (headline; full numbers + CIs in results.json):
  - **Controls**: decorrelation control passes (local-class decode = 0.50 exactly
    on all decorrelated cells); **correlated-depth instrument control passes**
    (tagged ST write decodes carry 0.99–1.00 at k=n_top−j); **F2 per-depth
    readability control** (decode a known-present local operand signal from the
    same write AT the invisible depth): **PASSES at 5d P9H1 k3 but FAILS at all 6d
    invisible cells** (the deep-chain writes wash out); behavioral gates pass.
  - **DH (decorrelated decode)**: at every invisible-decorrelated cell the resolved
    carry decodes at **chance** (6d P10H2 k3/k4, P11H2 k4 ≈ 0.45–0.57; 5d P9H1 k3 =
    0.57) — none clears baseline+0.2. **But** only the **5d cell is readable at its
    own depth** (F2), so only 5d's chance = a real "no relay"; the 6d cells are
    **underpowered** (write unreadable). The ST write decodes carry **only at the
    fully-correlated cell** (site = deciding digit, 0.99–1.00, linear).
  - **KO (class-necessity anchor; DH is load-bearing [F4])**: all-sufficient-ablate
    breaks the digit **differentially** (6d 0.24–0.50, 5d 0.50) over a **0.00
    specificity null** and **0.00 untagged baseline** → ST class carry-necessary.
    KO does NOT discriminate relay vs local resolution (leave-deepest-survives is
    consistent with both; and at 6d k=3 leave-deepest ≈ all-ablate, so it does not
    even cleanly "survive"). KO corroborates necessity only.
  - **RC (corroborating-only per LOC-5)**: φ_horizon ≫ φ_local (ΔR² ~0.93–0.99) —
    the consumer edge encodes the resolved carry (known, CE16); does not
    discriminate relay vs L1; not verdict-driving.

- **Interpretation** (post-result-corrected): the decisive **invisible-decorrelated**
  decode — the only measurement that can reveal a *relayed* carry — is at chance in
  both models. Where the write is **readable at that depth (5d P9H1 k3)**, chance =
  a genuine **"no relayed carry"** → the resolution is not at the L0 write; it is
  completed in the **L1 consumer read** (R-L1-read). Where the write **washes out
  at deep chains (6d k3/k4)**, the cell is **underpowered** — 6d neither supports
  nor refutes A11. Across both models there is **no positive evidence of a
  cross-position relay anywhere**; the ST writes carry only locally-visible content
  (CE13 single-step). KO adds that the ST class is carry-necessary (the L1 read
  depends on it) but does not itself separate relay from local resolution. Net: the
  compounding is **not shown to be a positional L0 relay; it is an L1-read
  computation** (5d proven, 6d consistent-but-underpowered), then discretized by
  the step combiner (CE17). This escapes CE17's two blockers via the decorrelation
  + per-depth readability control, but is a **linear-probe** result.

- **Prediction scoring**:
  - **A11 (positional L0 relay)**: **NOT SUPPORTED → low** (not formally rejected).
    The relay-only signal is chance where testable (5d, with a valid per-depth
    reader) and underpowered at 6d; no relay evidence anywhere. Caveat: linear
    probe (a non-linear/different-subspace relay not excluded); 6d underpowered.
  - **Compounding locus**: **R-L1-read** (5d proven; 6d consistent, underpowered)
    — the multi-digit compounding is completed at/after the L1 consumer read, not
    by an L0 positional relay. The C5-step-3 fork is settled against the relay
    (with the caveats).
  - **A9 (single-node selection)**: **stays retired** — the L1 read is a
    class-level (redundant) computation, not a single-head selector.
  - **A6 (economy)**: sufficient-ST class carry-necessary (KO all-ablate breaks
    differentially over baseline) while single sites are not — class economy again.
  - **Human C3/sequential-cascade lean**: **not supported at the L0 tail** (no
    positional relay found; caveat: linear probe + 6d underpowered).

- **Skeptic review (post-result half of the combined pass)**: Run 2026-07-16
  (separate thread). **Verdict: PASS WITH CORRECTIONS.** Design confirmed to escape
  both CE17 blockers; finding real; four corrections, all applied:
  - **F2 (blocking, RESOLVED)**: the correlated-depth instrument control (k=1/2)
    did not license readability at the deep invisible depth (k=3/4). Added a
    **per-depth readability control** (decode a known-present local operand signal
    from the same write at the invisible depth). Result: **6d invisible cells FAIL
    it (underpowered, not "no relay")**; only 5d passes → 6d downgraded to
    inconclusive, headline scoped accordingly.
  - **F1 (RESOLVED)**: "A11 REFUTED/very-low" softened to "not supported → low"
    (linear-probe null; non-linear relay not excluded).
  - **F4 (RESOLVED)**: KO does not discriminate relay vs local resolution and 6d
    k=3 leave-deepest ≈ all-ablate — **DH is the load-bearing battery**; KO
    reframed as a class-necessity anchor only; survival threshold tightened.
  - **F3 (RESOLVED)**: the 5d "self-computation" label (one visible cell) dropped;
    5d reported as R-L1-read (invisible cell at chance, readable) — the single
    visible-decode cell is noted as a non-robust hint only.

- **Limitations**: DH is a **linear** decode (non-linear/different-subspace relay
  not excluded, though the correlated-cell carry code IS linear); the decisive
  refutation rests on **one readable invisible cell (5d P9H1 k3)** — 6d is
  underpowered (deep-chain writes wash out, per the F2 control); KO is a necessity
  anchor, not a relay discriminator; RC confounded (corroborating only). 2-layer/
  3-head addition, two models.

- **Doc updates** (feeds paper hand-off, entry 3): A11 **not supported → low**
  (linear-probe null; 5d proven, 6d underpowered); compounding is **L1-read, not
  an L0 positional relay** (with caveats); A9 **retired**; A6 class-economy; add
  CE19; append router docs. (Applied 2026-07-16.)

- **Next read**: router-doc updates, then the paper hand-off (entry 3) — the SV
  mechanism is now resolved end-to-end (message/source/path/necessity/combiner)
  with the compounding-locus settled against an L0 relay (L1-read; linear-probe,
  6d-underpowered caveats).
