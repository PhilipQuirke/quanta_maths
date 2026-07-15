# Study: Locate the Tri-State U-Resolution Path (study-u-resolution-path.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #4

**Verdict: positive-but-ambiguous — U-flip *transmitters* located, but
combiner-vs-conduit unresolved.** MLP-heavy layer-0/layer-1 nodes causally
transmit the tri-state `U`-resolution flip (both models: 5-digit `P10.L0.MLP`,
`P14.L1.MLP`; 6-digit `P11.L0.MLP`, `P16.L1.MLP`, `P11.L0.H2`), and these
are genuinely distinct from the CE3 binary make-carry heads (whose U-flip is
0.00). Recorded as CE4.

The pre-registered discriminator meant to tell a true *combiner* from a mere
carry *conduit* proved **vacuous** (a definite digit's `A_{n+1}` has no
lower-carry dependence, so it auto-passed even the trivial readout site), so the
node's *role* was left unconfirmed. The follow-up combiner-vs-conduit study
(CE5) resolved this with a node-level activation-invariance discriminator.

## Pre-run (write before the experiment)

- **Question**: Which node(s) causally transmit the tri-state `U`-resolution —
  i.e. when digit `n` has `Dn+D'n = 9` (`U`), which node(s), when patched from a
  source where the *lower* carry resolves `U`→carry-out=1 into a target where it
  resolves `U`→carry-out=0, flip the answer digit `A_{n+1}`? Is the resolution
  **localized** (one/few nodes), **staged** (a specific layer/position), or
  **irreducibly distributed**?

- **Motivation**: The `U` tri-state and its cascade are the *specific novelty* of
  the Paper-2 ST/SV algorithm. The make-carry heads (CE3) handle only the binary
  `Dn+D'n≥10` case; the `U` path is the missing mechanistic piece and the
  sharpest open question. It directly informs A6 (how/where the cascade `U` is
  resolved — carried state vs local recompute vs wide-fetch), A3 (whether a
  tri-state representation exists at the resolution locus), and gives entry 2 a
  genuine tri-state anchor.

- **Hypothesis / competing reads** (neutral; all live):
  - **R-late-MLP**: `U` is resolved late, at the answer position `A_{n+1}`, by a
    layer-1 node/MLP that combines digit-`n`'s "sum==9" flag with the
    lower-digit carry. Patching that node transmits the flip.
  - **R-lower-attend**: the `A_{n+1}` answer-position head attends to *lower*
    digit positions (not only digit `n`), so its output already encodes the
    resolved cascade; patching it transmits the flip.
  - **R-carried-state (A6)**: the resolved carry is written into the residual
    stream at an earlier position and carried forward; patching the residual at
    the carrying position(s) transmits the flip, and the effect is localized to
    a cascade "bus".
  - **R-distributed**: no single node/position transmits the flip above the
    null; the `U`-resolution is spread across many components (an
    `instrument`-relevant outcome, but a substantive finding about the
    mechanism).
  - These are not mutually exclusive across digits; the design reports the
    per-node transmission map, not a forced single winner.

- **Design**:
  - **Models**: primary `add_d5_l2_h3_t15K_s372001`, replication
    `add_d6_l2_h3_t20K_s173289` (independent seed). Loaded CPU via `MathsConfig`
    + TransformerLens; accuracy-verified (invalid if < 0.99).
  - **The U counterfactual** (reused, Gate-2-validated from the confirm-ST-node
    study): hold digit `n` at `Dn+D'n = 9` (`U`) in both source and target;
    **source** has a lower carry (so `U`→carry-out 1), **target** has no lower
    carry (`U`→carry-out 0). Clean predictions differ on `A_{n+1}` by
    construction (verified 40/40 previously). Patch a candidate node's
    activation source→target and measure the `A_{n+1}` flip rate.
  - **Candidate sweep**: all attention heads (`hook_z`) AND MLP outputs
    (`hook_mlp_out`) AND residual (`hook_resid_post`) at every token position,
    both layers. Rank by `A_{n+1}` flip rate on the U counterfactual. The MLP
    and residual arms are essential because the make-carry study already ruled
    out the layer-0 heads and the confirm study hinted the resolution is not in
    the value path of those heads.
  - **The critical confound and its control**: in the U counterfactual, source
    and target differ in their *lower digits* (source has a lower carry, target
    does not). So a node can "transmit the flip" merely by carrying lower-digit
    *operand* information that the target then recomputes — this is
    lower-digit-transport, NOT U-resolution. Controls to separate them:
    1. **Lower-operand-matched counterfactual**: construct source/target where
       the lower digits are *identical* in value but the lower carry differs...
       — impossible if lower digits are identical (carry is a function of them).
       Instead: **carry-matched-operands** — vary which lower digit pair
       produces the carry, holding the digit-`n` `U` fixed, and require the node
       to transmit the flip across *different* lower-operand realizations of the
       same carry state. A pure lower-operand-transport node will not give a
       consistent `A_{n+1}` flip tied to the *carry bit* across realizations;
       a U-resolution node will.
    2. **Make-carry-head baseline**: the CE3 make-carry heads are the known
       negative for U-resolution (tristate flip 0.00). Any confirmed
       U-resolution node must exceed them.
    3. **Digit-n-driver control**: also run the counterfactual holding the lower
       carry FIXED and instead moving digit `n` from `U`(sum9) to a definite
       class — a U-resolution node should be the one whose patch matters
       specifically in the `U` regime, distinguishable from the binary
       make-carry signature.
  - **Confirmation criteria** (mirroring CE3 rigor): a candidate is a
    U-resolution node if its patch flips `A_{n+1}` ≥ bar on the U
    counterfactual, ≤ null on a same-carry-state null, in **both** patch
    directions, and the effect survives the carry-matched-operands control
    (consistent flip tied to the carry bit, not the lower-operand identity).
  - **Positive control**: patch the *residual stream at the `A_{n+1}` answer
    position* between the two U-resolution outcomes — this must flip `A_{n+1}`
    at high rate (the resolved value is read out there), demonstrating the
    harness + metric can move the target on the U counterfactual. A same-state
    null (patch between two same-carry sources) must be ≤ 10%. If the positive
    control fails, the run is `invalid`/`underpowered`, never "no path found".
  - **Minimal effect / bar**: U-resolution node flips `A_{n+1}` on ≥ 50% of U
    counterfactuals (bar tied to the positive control's rate minus a margin,
    per the CE3 A-6 convention), ≤ 10% same-state null, both directions,
    replicated across ≥ 2 digits and both models.
  - **Sample sizes**: ≥ 60 U-counterfactual pairs per (node, digit, direction);
    per-cell null. Power from the flip-rate gap + null, not asymptotics.

- **Success condition** (defined now): in both models, ≥ 1 node/position (head,
  MLP, or a small localized residual site) transmits the U-resolution flip
  (≥ bar, ≤ null, both directions, survives the carry-matched control),
  identifying **where** `U` is resolved. Verdict: **U-resolution path located**;
  record node id, target digit, mechanism read (R-late-MLP / R-lower-attend /
  R-carried-state), for entry 2 and A6.

- **Failure condition** (defined now): with the positive control passing — no
  single node or small set transmits the flip above null in either model, and
  the per-node map is diffuse (many small contributors, none ≥ bar). Verdict:
  **U-resolution is irreducibly distributed** (R-distributed) — a substantive
  mechanism finding that reshapes A6 (no localizable cascade node) and A3
  (no single tri-state locus).

- **Ambiguous / invalid condition**:
  - Ambiguous: a node passes the flip bar but fails the carry-matched-operands
    control (i.e. it is transmitting lower-operand identity, not the carry) —
    report as lower-transport, not U-resolution; if all candidates are like
    this, the true resolver is elsewhere/distributed.
  - Invalid: positive control fails; model accuracy check fails.

- **Skeptic review (pre-launch)**: Run 2026-07-15 in a separate skeptic thread
  (docs + prior study notes + Paper-2 facts only; no working-thread context).

  **Verdict: PASS WITH CONDITIONS.** Right question, inherited Gate-2-validated U
  counterfactual, honest about its own confound — but four BLOCKING gaps:

  - **F1 (make-or-break)** — the carry-matched-operands control separates raw
    lower-operand transport from carry-bit transport, but NOT carry-bit
    transport from the true U-combiner: a pure **carry-bit conduit** (relays the
    resolved lower carry past digit `n` without combining it with digit-`n`'s
    sum==9 flag) *passes* the control yet is not the resolver. Need a **U-regime
    interaction gate**: the carry-tracking flip must be *specific to digit-`n`=U*
    (present at U, absent/degenerate when digit `n` is a definite 0/1 class,
    which ignores the lower carry).
  - **F2** — the positive control patches the `A_{n+1}` readout site (trivially
    causal); it does not calibrate whether a *single-node* patch can move
    `A_{n+1}` on the U counterfactual (repeats confirm-ST-node G1). Need a
    node-level control on a CE3 make-carry head; tie the bar to it.
  - **F3** — "U-resolution node" is under-defined; single-node patching lights
    up the whole path (combiner + conduits + readout). Need a decision table
    (combiner vs carry-conduit vs readout) and to report a diffuse map as a
    *path*, not a locus.
  - **F4** — cannot conclude "irreducibly distributed" from single-node
    patching (Paper-2 head+MLP joint-necessity); need joint/pair patching first.

  Should-fix: F5 (state grid size + FWER argument), F6 (2-layer depth limits
  R-late-MLP vs R-carried-state distinction; addition-only scope up front),
  F7 (specify exact lower-carry construction + code assertion that only the
  lower carry, not digit-`n`'s own class, differs — the trap that sank the prior
  study's tri-state claim).

  *Status: RESOLVED 2026-07-15 by the working thread — F1–F4 fixed and F5–F7
  adopted in the amendment below. Goalposts unchanged (still locating the
  U-resolution path); the amendment makes the assay discriminate a true combiner
  from a conduit/readout. Gate 1 PASSED — launch authorized.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed.

**2026-07-15 — A-1 (resolves F1 + F3): a node is a U-COMBINER only if its
carry-tracking flip is U-regime-specific.** Two-part gate, both required for
"U-resolution locus":
- **(a) carry-matched-operands consistency**: with digit `n` held at `U`
  (sum=9), across ≥ 4 distinct lower-operand realizations of carry-in=1 and of
  carry-in=0, the patch must flip `A_{n+1}` tracking the *carry bit* (not the
  operand identity), with a matched null (patch between two same-carry,
  different-lower-operand sources) ≤ 0.10. Rules out raw operand transport.
- **(b) U-regime interaction gate** (the combiner discriminator): run the same
  carry-toggle patch with digit `n` set to a **definite** class (sum ≤ 8 and
  sum ≥ 10). A true U-combiner's carry-tracking flip is **specific to the U
  regime**: `flip_U − flip_definite ≥ 0.4`. A node whose carry-tracking flip is
  regime-independent (`flip_U ≈ flip_definite`) is a **carry-bit conduit**, not
  the combiner — reported as conduit.
- **Decision table**: `combiner` = passes (a) and (b) and reads digit-`n`
  operands/flag; `carry-conduit` = passes (a) but fails (b); `readout` = the
  `A_{n+1}`-position resid site (positive control only); `none` = below null.
  A diffuse map (many nodes pass (a) but none passes (b)) is reported as a
  **path, not a localized resolver**.

**2026-07-15 — A-2 (resolves F2): node-level positive control on the U
counterfactual.** Load-bearing control: patch a **CE3 make-carry head's
`hook_z`** at the relevant answer position between two sources differing in the
*binary* carry into `A_{n+1}` — must flip `A_{n+1}` at high rate (make-carry
heads are known-causal for binary carry, CE3), proving a single-head z-patch
moves the target under the FLIP metric. The bar for a U-resolver is
`max(0.50, node_control_rate − 0.10)`. The resid-at-`A_{n+1}` patch is demoted
to a coarse sanity check. The prior 40/40 clean-prediction separation is a
*precondition* (target can differ), NOT this positive control.

**2026-07-15 — A-3 (resolves F4): joint patching before any distributed
verdict.** The sweep patches heads, MLPs, and resid individually AND
**joint (head + adjacent MLP) at each position/layer**. R-distributed is only
declared if no single node AND no tested head+MLP pair AND no small greedy set
of top sub-bar nodes transmits the flip above the node-level-calibrated bar;
otherwise the flat result is `underpowered`, not distributed.

**2026-07-15 — A-4 (adopts F5): grid + FWER.** Grid ≈ (heads n_heads + MLP 1 +
resid 1) × positions × 2 layers × ≥ 2 digits × 2 directions. A chance pass must
clear the bar in *both* directions AND replicate across 2 models AND pass the
(a)+(b) gates — making the family-wise false-positive rate negligible; no
separate correction needed beyond reporting the per-cell null.

**2026-07-15 — A-5 (adopts F6): depth + scope stated.** These are **2-layer
addition models**; with layer-0 make-carry heads (CE3), a localized U-combiner
has essentially only layer-1 (answer-position head or MLP) available, so
R-late-MLP and R-carried-state may be hard to distinguish here — noted as a
scope limit. Findings are about these 2-layer addition models; cross-size /
1-layer generalization is deferred (backlog B5/B10).

**2026-07-15 — A-6 (adopts F7): exact construction + assertion.** For digit `n`
(`n ≥ 1`): set `Dn+D'n = 9` (U); set digit `n−1` to sum `≥ 10` (lower carry) in
the source and sum `< 10` in the target; all digits `< n−1` set to no-carry
(sum < 10) and identical structure; all digits `> n` fixed to 0. The script
asserts per pair that (i) digit `n` sum == 9 in both, (ii) carry-out of digit
`n−1` differs (1 in source, 0 in target), (iii) no digit other than `n−1`
changes its carry-out — so the *only* difference in the carry reaching digit `n`
is the toggled lower carry. `d_model=510, d_head=170, n_ctx=19` (5-digit);
hooks `blocks.{L}.attn.hook_z`, `blocks.{L}.hook_mlp_out`,
`blocks.{L}.hook_resid_post`.

- **Decision impact**:
  - **Located (R-late-MLP / R-lower-attend / R-carried-state)**: gives entry 2
    its genuine tri-state anchor; scores A6's mechanism sub-reads (carried-state
    vs recompute vs wide-fetch) with real evidence; adds a claim-evidence entry
    naming the U-resolution locus and mechanism.
  - **Distributed (R-distributed)**: strong evidence against a single tri-state
    node — reshapes A3 (tri-state may not live at one place) and A6 (the cascade
    is not a localizable bus); redirects entry 2 toward a distributed-readout
    analysis.

- **Risks / confounds**:
  - **Lower-operand transport vs U-resolution** (the main confound) — handled by
    the carry-matched-operands control; a node passing only the naive flip is
    reported as lower-transport.
  - **Answer-position readout is not resolution**: patching resid at `A_{n+1}`
    (the positive control) flips the answer trivially because that is where the
    value is read; it validates the harness but is NOT itself "the resolution
    node". The resolution node is one whose *output feeds* that readout.
  - Autoregressive subtlety: `A_{n+1}` is predicted at its own position from the
    residual there; patch effects must be read at that position's logit.
  - hook_z is pre-`W_O`; effect read at logits includes `W_O`. MLP-out patched
    as its own unit.
  - Multiple-comparisons over the full node grid: the per-cell null + both-
    directions + carry-matched control + 2-model replication guard against
    chance passes.

- **Expected artifacts**:
  - Standalone CPU script `scripts/u_resolution_path.py`.
  - Per (model, node, digit): U-counterfactual flip, same-state null, both
    directions, carry-matched-control consistency, classification.
  - Transmission heatmap (node × digit) per model; positive-control results.
  - A U-resolution registry JSON.
  - Local results folder `results/study-u-resolution-path/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (reframed post-Gate-2): **A small set of nodes causally
  *transmit* the tri-state `U`-resolution flip — real and distinct from the CE3
  make-carry heads — but the assay cannot show they *combine* (vs merely relay)
  the carry, so "resolution locus" is not established.** In both models,
  MLP-heavy layer-0/layer-1 nodes flip `A_{n+1}` at 1.00 on the U counterfactual,
  both directions, same-carry null 0.00 (5-digit: `P10.L0.MLP` question-position,
  `P14.L1.MLP` answer-position; 6-digit: `P11.L0.MLP`, `P16.L1.MLP`, and head
  `P11.L0.H2`). These are genuinely distinct from the CE3 binary make-carry heads
  (whose U-flip is 0.00). **However**, the pre-registered combiner discriminator
  (A-1b: transmission must be U-regime-*specific*) proved **vacuous** — because a
  definite digit's `A_{n+1}` has no lower-carry dependence, `def_flip = 0` for
  *every* node including the trivial readout site (which scores gap 1.00). So
  these are **U-flip *transmitters*, role (combiner vs carry-conduit vs readout)
  unconfirmed** — the study's own Ambiguous condition. Positive controls passed,
  so this is a positive-but-ambiguous result, not invalid: the U-flip *is*
  transmitted by an MLP-heavy L0/L1 path distinct from the make-carry heads;
  *where it is combined* is still open.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/u_resolution_path.py control` then
    `... models`.
  - Script: [`scripts/u_resolution_path.py`](../../scripts/u_resolution_path.py)
    (standalone CPU; reuses the confirm-ST-node helpers + Gate-2-validated U
    counterfactual).
  - Env: python 3.13.7, macOS-26.5.2-arm64, torch 2.8.0, numpy 2.3.2. Repo
    commit `368f3a9` (working tree). Date 2026-07-15.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000).
  - Artifacts (local; no HF): `results/study-u-resolution-path/`:
    `control_*.json`, `results.json`, `u_resolution_registry.json`,
    `u_resolution_candidates.png`.

- **Results**:
  - **Positive controls PASSED**: node-level control (make-carry head z-patch on
    the *binary* carry counterfactual) flips `A_{n+1}` at 1.00 in both models →
    a single-node patch can move the target under the FLIP metric; bar set to
    `max(0.50, 1.00−0.10) = 0.90`. Coarse readout control (patch resid at the
    position that *predicts* `A_{n+1}`) flips at 1.00 on the U counterfactual.
    (An off-by-one — patching the A_{n+1} *token* position instead of the
    *predicting* position — was caught and fixed before the sweep; the fixed
    control transmits the U-flip 1.00.)
  - **Combiner gate results** (bar 0.90; `flip_U`, `both_dir`, null,
    `def_flip`, regime `gap`):
    - 5-digit `P10.L0.MLP`: 1.00 / 1.00 / 0.00 / 0.00 / **1.00** → combiner.
    - 5-digit `P14.L1.MLP`: 1.00 / 1.00 / 0.00 / 0.00 / **1.00** → combiner.
    - 5-digit heads `P10.L0.H2` (0.78) and `P14.L1.H2` (both-dir 0.77): below
      bar → `none` (the head alone is not the resolver).
    - 6-digit `P11.L0.MLP`, `P16.L1.MLP`: combiners (1.00 / gap 1.00);
      `P11.L0.H2` head also passes (1.00 / gap 1.00) — so the 6-digit L0 locus
      is head+MLP, not MLP-only.
  - **Joint (head+MLP) patches** all classify combiner (as expected — they
    include the MLP).
  - **The A-1(b) gate is vacuous (Gate-2 F1).** For definite digit-`n` classes,
    clean `A_{n+1}` does not change under a lower-carry toggle (0/40 for `lo` and
    `hi`; 40/40 for `U`) — so `def_flip = 0.00` is forced for **every** node,
    including the trivial readout site (verified: readout scores `u_flip_U 1.00`,
    `def 0.00`, gap 1.00). The gap therefore equals `u_flip` minus a constant and
    does **not** discriminate a combiner from a carry-conduit or the readout.
    The earlier claim that "gap genuinely measures U-specific combination" is
    **withdrawn**. Only gate (a) (transmission + carry-matched null) is earned.

- **Interpretation** (reframed post-Gate-2 to the Ambiguous condition; goalposts
  unchanged):
  - **Success condition NOT cleanly met; Ambiguous condition invoked.** Nodes
    pass gate (a) (transmit the U-flip ≥ bar, both directions, null 0.00), but
    the combiner discriminator (gate b) is vacuous (F1), so per the pre-stated
    Ambiguous condition — "passes the flip bar but is not distinguished from
    transport" — these are reported as **U-flip transmitters, not confirmed
    U-resolvers.**
  - **What is genuinely established** (positive): the U-resolution flip is
    causally carried by an **MLP-heavy L0/L1 path** (`P10.L0.MLP`+`P14.L1.MLP`
    5-digit; `P11.L0.MLP`+`P16.L1.MLP`+`P11.L0.H2` 6-digit), and this path is
    **distinct from the CE3 binary make-carry heads** (whose U-flip is 0.00) —
    so the confirm-ST-node "separate paths" finding is reconfirmed with named
    nodes. Whether any node *combines* digit-`n`'s sum==9 flag with the carry, or
    merely *relays* the resolved carry from the lower digits, is **unresolved**.
  - **Not "MLP-centered" model-generally**: the 6-digit L0 head `P11.L0.H2` also
    transmits (gap 1.00), and the 5-digit heads are only sub-bar (one is
    direction-asymmetric, 1.00 forward / 0.77 reverse) — a bar artifact, not
    proven non-involvement.
  - The "compute cascade low (L0 MLP), apply at answer (L1 MLP)" two-site story
    is an **inferred hypothesis** from position/layer, not evidence.

- **Prediction scoring** (rewritten post-Gate-2; all downgraded to untouched —
  the transmission evidence is confounded by conduit-vs-combiner ambiguity):
  - **A6** ("cascade is carried state … resolved by `=`; tie-break"):
    **untouched.** A question-position MLP transmitting the flip is exactly what
    lower-carry *transport* looks like; it does not establish "carried/combined
    state" over transport. A6's real predictions (selective harm to U-cascade
    questions; `U` gone by `=`) are untested.
  - **A2** ("aggregate-then-discretize; MLP does the nonlinear step"):
    **untouched.** The study shows MLP-heavy nodes *transmit* a flip, not that an
    MLP *computes the discretization*; and a 6-digit head also transmits.
  - **A5** ("attention static wiring; data-dependence in values/MLPs"):
    **untouched** here (adds nothing clean beyond CE3).
  - **A3** ("ST tri-state geometry"): **untouched.** Entry 2 gains a candidate
    anchor (the U-flip-transmitting MLPs) but their *locus role is unconfirmed*.
  - This study scores **no prediction confirmed/refuted**; it is a
    positive-but-ambiguous locator that (i) reconfirms the CE3 "separate paths"
    result with named MLP-heavy transmitters and (ii) exposed a vacuous
    discriminator to fix.

- **Skeptic review (post-result)**: Run 2026-07-15, separate thread, docs + JSON
  artifacts only. **Verdict: BLOCK** — upheld; the "combiner/located" verdict was
  overclaimed and is reframed below.

  Findings (verified by the working thread):
  - **F1 (critical)**: the A-1b "U-regime interaction gate" is **vacuous**. For a
    *definite* digit `n`, ground-truth `A_{n+1}` has no lower-carry dependence
    (clean flip 0/40 for `lo`/`hi`), so `def_flip = 0.00` is forced for **every**
    node — including the trivial readout site, which the working thread verified
    scores `u_flip_U = 1.00`, `def = 0.00`, **gap = 1.00**. So `gap ≥ 0.4`
    auto-passes anything that transmits the U-flip; it does **not** separate a
    true combiner from a carry-bit conduit or the readout. Gate-1's F1 concern
    was therefore *not* actually resolved by A-1b.
  - **F2**: consequently "combiner" is unearned. What is shown is only gate (a):
    these nodes *transmit* the U-flip. The Results sentence claiming the gap
    "genuinely measures U-specific combination" is struck.
  - **F3**: a question-position L0 MLP + answer-position L1 MLP both transmitting
    is a *path* lighting up, not a proven localized combiner; the
    "compute-low/apply-at-answer" two-site story is an inferred hypothesis, not a
    finding.
  - **F4**: heads are only *sub-bar* (5-digit `P14.L1.H2` is 1.00 forward /0.77
    reverse — a bar artifact, not "not involved"); the 6-digit `P11.L0.H2` head
    *passes* — so "MLP-centered" is 5-digit-specific, not model-general.
  - **F5**: A6/A2/A5 were scored "supported/partial" on confounded evidence —
    must be `untouched` (a question-position MLP transmitting is exactly what
    lower-carry transport looks like; it does not establish "carried state" or
    "MLP computes the discretization").
  - **F6**: positive controls genuinely pass, so this is a *positive-but-
    ambiguous* result, not invalid — invoke the study's own pre-stated Ambiguous
    condition ("passes flip bar but not distinguished from transport → report as
    transport, not U-resolution").

  *Status: BLOCK RESOLVED 2026-07-15 by the working thread — verified the
  vacuity (readout scores gap 1.00), reframed the verdict to "U-flip
  transmitters located; combiner-vs-conduit unresolved", renamed the registry
  key to `u_flip_transmitters`, rescored A6/A2/A5/A3 as untouched, and left
  agenda entry 1 open (re-scoped for a real interaction control). Gate 2 PASSED
  on the corrected read.*

- **Limitations**:
  - 2-layer models: R-late-MLP (answer-position L1 MLP) is confirmed, but with
    only 2 layers the "carried-state vs late-combine" distinction is coarse;
    generalization to deeper/other models deferred (B5).
  - Interchange localizes *where the U-flip is transmitted*; the L0-MLP + L1-MLP
    two-site pattern is inferred from position/layer, not proven to be
    "compute-cascade then apply". A finer path-patch (L0-MLP → L1-MLP) would
    confirm the hand-off (follow-up).
  - MLP patched as `hook_mlp_out` (whole layer at a position); does not decompose
    to neurons (that is backlog B2).
  - Addition only; single-digit-`n` `U` (not multi-digit `U` cascades like
    `...999`).

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (**CE4** —
  U-flip transmitters located, distinct from CE3 make-carry heads;
  combiner-vs-conduit unresolved, hedged Low–Medium); synthesis + summary;
  conjectures (A6/A2/A5/A3 **untouched**; incidental "separate paths reconfirmed
  with named MLP-heavy transmitters"); agenda (entry 1 **stays open**, re-scoped
  to distinguish combiner from conduit via a real interaction/corruption control
  + an L0-MLP→L1-MLP path-patch).

- **Next read**: The blocking methodological gap is a **real interaction
  control** that a conduit fails but a combiner passes — e.g. inject a
  lower-carry *inconsistent* with a definite digit `n` and check whether
  `A_{n+1}` is *corrupted away from correct* (a raw carry-conduit dumping a bit
  into a definite readout should error; a combiner reading digit-`n`'s flag
  should not). Plus the L0-MLP→L1-MLP path-patch for the two-site hypothesis.
  Only then can entry 2 point at a *confirmed* tri-state locus.
