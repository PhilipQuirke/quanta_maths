# Maths Claim-Evidence Map (maths-claim-evidence.md)

Read role and rules: [Claim-Evidence Map](thor-document-rules.md#claim-evidence-map).

Durable empirical claims for the `maths` thread and the evidence that supports,
weakens, or narrows them. Keep fewer claims than experiments. Do not organize by
chronology (that is [maths-results-by-time.md](maths-results-by-time.md)). Keep
conjecture out of this file.

Confidence labels:

- **High**: repeated across models or methods, with a direct artifact trail.
- **Medium**: supported but interpretation-, assay-, or replication-dependent.
- **Low**: suggestive, not yet stable.

## Claims

<a id="ce1"></a>

### CE1: Addition model — Trained addition-model digit embeddings are near-isotropic 9-D categorical codes with a weak, training-induced circular *ordering* — not a dominant low-rank circle/helix

- **Confidence**: **Medium** for the near-isotropic / no-dominant-geometry part
  (replicated across 4 accurate models + untrained baseline, direct artifact
  trail); **Low, LN-robust (weakly) but seed-fragile** for the weak
  circular-ordering part. *(De-provisionalized 2026-07-16 by the LN-aware
  close-out A-9, [study-ln-aware-embedding.md](study-maths/study-ln-aware-embedding.md).)*
  The ordering holds in the LN-effective geometry (position-free and model-true
  `ln1.hook_normalized`), but **LN is near-isometric on the digit code here
  (γ std ~0.005), so "survives LN" is a weak form of robustness**; and it holds
  in only **2 of 3 independent seeds** (`s372001`, `s572091`; fails in `s173289`
  raw and LN, ordering p 0.05→0.09), across 4 model instances (primary and repA
  share seed `s372001`). Magnitude unchanged (+~0.04 over the untrained baseline;
  freq-1 share ~0.24–0.28, still noise-floor).
- **Supporting evidence**: 2026-07-14 digit-embedding geometry bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-digit-embedding-geometry.md](study-maths/study-digit-embedding-geometry.md)).
  Embedding variance spectrum near-flat (participation ratio ≈ 8.7 of max 9);
  freq-1 variance share ≈ 0.27, essentially the untrained baseline (0.233) and,
  per the positive-control calibration, at the noise floor. Angular ordering in
  the top-2 PC plane is significant (permutation p = 0.0001) in 3 of 4 accurate
  models and absent in the untrained control (p = 0.34) — the one baseline-clean,
  replicated, training-induced circular signal.
- **Weakening / narrowing evidence**: The circular signal is present only as an
  *ordering* tendency, not as variance concentration; repB (independent seed
  s173289) does not show significant ordering (p = 0.045). Unembedding is
  unstructured (R4) in all trained models and misaligned with the embedding
  (principal angles 36–88°). No linear/helix magnitude component
  (unique-linear share never significant).
- **Caveats**: Weights-only — says nothing about whether the model *uses* any
  geometry (causal, backlog B1) or about activation-level feature geometry
  (agenda entries 2–3). The ordering is LN-robust but only *weakly* (LN is
  near-isometric here) and *seed-fragile* (2/3 seeds). Tiny 10-token vocabulary;
  n = 10 per matrix.

<a id="ce2"></a>

### CE2: Addition model — At layer-0 operand-fetch heads, the value-path output is indistinguishable from linear transport of the (weakly-circular) digit embeddings

- **Confidence**: **Medium** — replicated across 6 nodes in 2 accurate models
  (5-digit + independent-seed 6-digit); direct artifact trail; but from a single
  assay that was an `instrument failure` for its primary A2 question.
- **Supporting evidence**: 2026-07-14 pair-sum bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-pair-sum-sufficiency.md](study-maths/study-pair-sum-sufficiency.md)).
  At confirmed operand-fetch heads (near-equal attention to `Dn`,`D'n`, in the
  [0.4,0.6] band, ablation-relevant), the head output over operand pairs matches
  a **noise-free transport null** built from the model's own `W_E·W_V` on every
  metric: sum/pair ratio ≈ 0.19–0.24, `R²_pair` = 1.0 (deterministic given
  pair), a monotone ~1-D sum arc (`cos a + cos b` signature), and near-zero
  full-space tri-cluster silhouette. Extends CE1's "structure is a weak
  low-variance projection" motif from the embedding to the value path.
- **Weakening / narrowing evidence**: the assay could not separate this from a
  genuine aggregate computation (the metric is non-discriminating under the
  no-lower-carry stimulus); it therefore establishes *consistency with
  transport*, not that no aggregation occurs.
- **Caveats**: Nodes are answer-position operand-fetch heads, not confirmed `ST`
  carry-compute nodes; says nothing about where `ST` discretization happens.
  Correlational, not causal. Does **not** score A2 (which remains untested).

<a id="ce3"></a>

### CE3: Addition model — The carry is computed by binary make-carry heads at answer positions, dissociated from base-add heads; the tri-state U-resolution is a separate, unlocated path

- **Confidence**: **Medium-High** for the causal make-carry nodes + SA/SC head
  dissociation (clean, replicated across 2 accurate models, strong interchange
  effects with nulls and direction symmetry); **Medium** for "U-resolution is a
  separate path" (a clean 0.00 tri-state patch, but absence-of-effect under
  single-node patching).
- **Supporting evidence**: 2026-07-14 confirm-ST-node bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-confirm-st-node.md](study-maths/study-confirm-st-node.md),
  registry `results/study-confirm-st-node/confirmed_st_nodes.json`). Single-head
  activation patching: `P13/P14/P15.L0.H0` (5-digit) and `P14.L0.H1`,
  `P15–P19.L0.H2` (6-digit) flip the next-higher answer digit `A_{n+1}` at 1.00
  on carry↔no-carry patches, `A_n` flip 0.00, same-class null 0.00, both
  directions, operand attention ≥ 0.83. A co-located head computes base-add
  (`P14.L0.H1` 5-digit flips only `A_n`). Node-level positive control passed
  (SA head z-patch flips `A_n` 0.95, null 0.00).
- **Weakening / narrowing evidence**: the genuine tri-state test (fix
  `Dn+D'n = 9`, toggle lower carry) flips `A_{n+1}` at **0.00** at these nodes —
  so they are **binary make-carry (`SC`)**, not tri-state `ST`. The model does
  resolve `U` correctly (clean preds differ 40/40), so a separate U-resolution
  path exists but is not localized here.
- **Caveats**: Interchange shows causal-for-`A_{n+1}`, not the internal
  computation. Candidates discovery-selected on the same metric (mitigated by
  orthogonal `A_n`=0 specificity + attention gate + null + direction symmetry).
  "Answer-position locus" is under single-node patching; question-position
  involvement not excluded (recompute masking). Addition only.
- **C5-step-1 re-examination (2026-07-16, CE13 bundle)**: the "paper ST candidates
  not causal" reading is **refined to REDUNDANCY, baseline-controlled**. At the
  HF-map-named question-position `ST` heads, single-node interchange still flips
  nothing (0.00, reproduced), BUT mean-ablation impact **exceeds an untagged-head
  baseline** (baseline max 0.000–0.003) for the **low-digit** ST nodes (units/low
  carries, impact up to 0.04–0.047) and is at baseline for the high-digit ones
  (redundant). So the map's causal tags are **vindicated** (a C5 win): the nodes are
  load-bearing but redundantly (swap-insufficient, removal-harmful, concentrated
  where carries originate) — matching the paper's own "redundant ST node" caveat.
  The interchange-vs-ablation gap is the expected decodability/interchange/ablation
  dissociation (outside-view sweep: Huang & Chang 2025; causal scrubbing).

<a id="ce4"></a>

### CE4: Addition model — The tri-state U-resolution flip is transmitted by an MLP-heavy L0/L1 path distinct from the make-carry heads; whether it is combined or merely relayed is unresolved

- **Confidence**: **Low–Medium.** The *transmission* + *distinctness from CE3*
  are solid (replicated 2 models, both directions, null 0.00, positive controls
  pass); the *mechanistic role* (combiner vs conduit) is **unconfirmed** (the
  intended discriminator was vacuous).
- **Supporting evidence**: 2026-07-15 U-resolution bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-u-resolution-path.md](study-maths/study-u-resolution-path.md),
  registry `results/study-u-resolution-path/u_resolution_registry.json`).
  With the genuine U counterfactual (fix `Dn+D'n=9`, toggle lower carry),
  MLP-heavy nodes flip `A_{n+1}` at 1.00 both directions, same-carry null 0.00:
  5-digit `P10.L0.MLP` (question-position), `P14.L1.MLP` (answer-position);
  6-digit `P11.L0.MLP`, `P16.L1.MLP`, and head `P11.L0.H2`. Re-verified the CE3
  make-carry heads have U-flip 0.00, so the U-path and the binary-carry path are
  genuinely distinct components.
- **Weakening / narrowing evidence**: single-node interchange cannot tell a true
  U-*combiner* (reads digit-`n`'s sum==9 flag) from a carry-bit *conduit* or the
  readout — the pre-registered interaction gate is vacuous because a definite
  digit's `A_{n+1}` has no lower-carry dependence (the trivial readout site
  scores the same gap 1.00). "MLP-centered" is 5-digit-specific (6-digit L0 head
  also transmits).
- **Caveats**: 2-layer, addition-only, single-digit `U` (not multi-digit `...999`
  cascades). The "compute-low/apply-at-answer" two-site story is an inferred
  hypothesis, not established. Needs a real interaction/corruption control to
  confirm any node as the combiner. **Superseded by CE5**, which identifies the
  combiner.

<a id="ce5"></a>

### CE5: Addition model — The tri-state U-combiner is the answer-position layer-1 MLP; layer-0 nodes relay the running carry (conduit)

- **Confidence**: **Medium-High** for the combiner (L1 MLP; replicated in both
  models, survives a carry_out-centroid test); **Medium** for the L0-conduit /
  two-site companion (clean in 6-digit, borderline in 5-digit).
- **Supporting evidence**: 2026-07-15 combiner-vs-conduit bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-combiner-vs-conduit.md](study-maths/study-combiner-vs-conduit.md),
  registry `results/study-combiner-vs-conduit/combiner_registry.json`). A
  node-level activation-invariance discriminator (a combiner's output =
  `carry_out(n)`, invariant to `carry_in` for a definite digit but variant in
  `U`) identifies the answer-position **L1 MLP** as the combiner in both models:
  `P14.L1.MLP` (5-digit, definite-regime activation diff 0.004 / U-regime 1.18)
  and `P16.L1.MLP` (6-digit, 0.004 / 1.31). Beyond the ratio, the L1-MLP output
  fires equally for definite and U (norms ≈ 30.7 / 31.9 — not a "U-detector")
  and routes to the class-determined `carry_out` centroids (definite carry_out=0
  vs =1 centroids ~orthogonal, cos 1.27; U-resolved outputs land on the matching
  centroid, cos 0.058 / 0.045) — i.e. the output *is* the resolved `carry_out`.
  Positive controls separate (planted conduit relays `carry_in` even for a
  definite digit, cos 0.83; inert reference flat 0.00), so the discriminator is
  non-vacuous — the fix over the CE4 endpoint gate.
- **Weakening / narrowing evidence**: the layer-0 conduit companion is clean only
  in the 6-digit model (`P11.L0.MLP` 0.51/0.59, `P11.L0.H2` 0.95/0.90); the
  single 5-digit L0 candidate (`P10.L0.MLP`, 0.35/0.73, ratio 0.48) is
  **borderline/mixed**, so the full "two-site: L0 conducts, L1 combines"
  mechanism is fully evidenced only in the 6-digit model.
- **Caveats**: 2-layer, addition-only, single-digit `U` (not multi-digit
  `...999` cascades). Whole-MLP-output metric (not neuron-level, B2). Patching
  swaps the whole MLP output if it bundles other info. Distinct from the CE3
  binary make-carry heads (`carry_in`-inert here).

<a id="ce6"></a>

### CE6: Addition model — At the U-combiner's input the carry is a clean binary code — no distinct off-axis tri-state; the resolved carry is already linearly present there

- **Confidence**: **Medium** (replicated in both models; but locus-scoped and
  with the ingredient/decision caveat below).
- **Supporting evidence**: 2026-07-15 tri-state-geometry bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-st-tristate-geometry.md](study-maths/study-st-tristate-geometry.md),
  registry `results/study-st-tristate-geometry/shape_registry.json`). At the
  L1-MLP combiner input (`blocks.1.ln2.hook_normalized`), a 4-class design
  (committed-0, committed-1, U→0, U→1 — breaking the `U ≡ SA_n=9` confound via
  the resolution variable) shows: the `U`-mean is **not** significantly off the
  committed 0–1 axis (off-axis 0.19–0.23, permutation p 0.85–0.92), and the `U`
  cases split by resolution (U→0≈committed-0: d 10.1 vs 29.5; U→1≈committed-1:
  d 13.3 vs 30.5; U→0/U→1 probe 1.00). A control confirms the split is the
  resolved carry, not lower operands (committed-0 lower-carry move 1.21/1.65 vs
  ~28–29 U-split). So the `{0,1,U}` is a clean **binary** carry code here, not an
  off-axis simplex/square — **refuting A3's off-axis-third-symbol prediction at
  this locus**.
- **Weakening / narrowing evidence**: the raw `carry_in` bit is *also* ~98%/94%
  linearly decodable at this site (on committed digits), so the site holds the
  decision *ingredients*; this does **not** establish the U→{0,1} *decision* is
  completed upstream vs within the MLP. The 5-digit off-axis (0.23) is in the
  pre-registered ambiguous band. This **narrows CE5** (resolved carry linearly
  present at the MLP input, not only its output) without contradicting CE5's
  causal combiner result.
- **Caveats**: single locus (post-L1-attention — no L0 attribution); single-digit
  `U`; 2-layer; addition only. Does NOT refute a tri-state existing at an earlier
  site (untested). make-carry resolution probe 0.79/0.80 (not chance).

<a id="ce7"></a>

### CE7: Addition model — No dedicated `{0,1,U}` tri-state symbol at any answer-position residual site; `U` is resolved to binary around L1-attention

- **Confidence**: **Medium** (replicated in both models; independently
  reproduced at Gate 2; but scoped — see caveats).
- **Supporting evidence**: 2026-07-16 earliest-tri-state-site bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-earliest-tristate-site.md](study-maths/study-earliest-tristate-site.md),
  registry `results/study-earliest-tristate-site/site_trajectory_registry.json`).
  Across 7 residual sites at the answer position, the resolution/committed-
  orthogonal **is-U axis is never significant** vs a permutation null (perm p
  0.10–1.0, never < 0.01); no site shows a dedicated off-axis `U` symbol. The
  per-class distance trajectory (in-artifact) shows U→1 sits nearest committed-0
  through `L1.attn_in` and flips to committed-1 at `L1.resid_mid` — so the binary
  resolution is applied **around L1-attention**, never as a third symbol.
  U-vs-committed separability is 1.00 and survives carry_in-partialling
  (not the raw ingredient). Planted controls validate the discriminator
  (planted-unresolved perm p 0.003). Extends CE6 (combiner-input binary) back
  across the answer-position stream.
- **Weakening / narrowing evidence**: power floor ~0.5× the committed-carry
  separation — a *modest* dedicated symbol (≤~0.3×) could be missed (the 6-digit
  L1.attn_in perm p 0.10 is null-consistent, not a weak hit). Answer positions
  only.
- **Caveats**: scoped to a dedicated symbol *comparable in magnitude to the
  binary carry*, at *answer-position* residual sites. A weak symbol, or a
  transient `U` at *question* positions (D'n, descoped), is untested.
  Representational, not causal. Single-digit `U`, 2-layer, addition only.

<a id="ce8"></a>

### CE8: Addition model — Attention routing is hybrid — a few heads relocate their target with carry state (6-digit only); most cells are target-static

- **Confidence**: **Medium** — clean and Bonferroni-safe in the 6-digit model;
  not replicated (5-digit inconclusive); representational, not causal.
- **Supporting evidence**: 2026-07-16 attention-invariance bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-attention-invariance.md](study-maths/study-attention-invariance.md),
  registry `results/study-attention-invariance/attention_routing_registry.json`).
  Under a value-matched contrast (digit-`n` operands fixed at sum=9, only the
  lower carry toggled), the 6-digit model shows clean carry-state top-1
  target-moves at **`L1.H1` Q11 (operand-read, move 1.00 / same-state null ≤0.08),
  `L1.H1` Q14, `L0.H0` Q17** — surviving a strict null (digit-`n` fixed,
  lower operands re-randomized: strict-null 0.00–0.08) and Bonferroni ×108. A
  synthetic content-routed head is detected at 1.00 (the move-counter works).
  This is A5's own pre-registered falsifier ("target switching when `ST` is `U`").
- **Weakening / narrowing evidence**: the **5-digit model is inconclusive** — its
  candidate cells (L1H1 Q14 0.72/0.40, L1H2 Q15 1.00/0.53) ride a same-state null
  of 0.40–0.53 (top-1-argmax near-useless at near-tied attention; L1H1 Q14 fails
  MC). So this is clean in 6-digit only, not a cross-model replication.
- **Caveats**: representational (attention-pattern shift), **not causal** — does
  not show the routing is load-bearing (needs pattern-patching, the entry-3
  follow-up). At least one clean cell (Q11) is an operand-read position, so the
  effect is *not* merely "answer digit depends on carry"; but co-location with
  the CE5 answer-position L1-MLP combiner is *suggestive, not established*.
  top-1-argmax is tie-sensitive (top-k-mass follow-up). 2-layer, addition only.
- **Causal follow-up result (2026-07-16, deep-cascade bundle / CE9)**: the
  causal pattern-patch this claim flagged has now run. The `L1.H1` routing cell is
  **causally inert** under a single-head pattern redirect (answer-move 0.00 at
  every chain depth) — so CE8's routing signal **stays representational**; it is
  not shown load-bearing for deep-cascade resolution at node/pattern granularity
  (a finer edge path-patch is needed). Confidence unchanged (still Medium,
  representational).

<a id="ce9"></a>

### CE9: Addition model — At node/attention-pattern granularity the deep `...999` cascade mechanism is not localizable — no single-cell selection, no sequential per-digit state, real graded tail state

- **Confidence**: **Medium** as an *instrument-limit / ambiguous* result (both
  models agree it is not localizable here; two independent skeptic rounds
  enforced the scope). Low confidence on any specific positive mechanism.
- **What it establishes**: (1) **A9's predicted convergence is absent** — a
  causally deciding-selective consumer head (6-digit `L1.H0` Q14: redirect-to-
  deciding moves the answer 0.93/0.90 at k=2,3 vs redirect-to-wrong 0.42/0.55 and
  redirect-to-irrelevant 0.38/0.50) and a deciding-digit-*tracking* head
  (`L1.H2` Q15, top-2 key set follows the deciding position at ≥2 non-degenerate
  depths) are **different cells**, and the CE8 cell `L1.H1` is causally inert.
  A9 needs one head that both tracks and delivers; that is not found at this
  granularity. (2) **No sequential per-digit state** where testable: at the one
  transmitting locus with a passing deciding-digit control (5-digit `=`, k=3, ctrl
  0.75), the cascade state does not carry intermediate-digit identity. (3) **Real
  graded computed state** near the question tail (pure-state `resid_post(L0)` cells,
  null 0.00) but operand-adjacent and joint-flip 0.00 — not a single stored
  resolved bit.
- **Supporting evidence**: 2026-07-16 deep-cascade bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-deep-cascade-mechanism.md](study-maths/study-deep-cascade-mechanism.md),
  `results/study-deep-cascade-mechanism/results.json`). All 7 positive controls
  pass (5-digit Battery C2 pre-registered `invalid` — no causal L1 pattern).
- **Caveats**: the result is an **instrument limit**, not a mechanism verdict —
  single-site interchange and single-head uniform pattern-redirect are too blunt
  for a mechanism split across heads or graded across positions. A9 (one-hop
  selection) is *weakly consistent with* the pieces (selective head + tracking head
  + no sequential identity) but **not confirmed**; the human's sequential-cascade
  lean is *disfavored where testable* but **not refuted** (untestable in 6-digit).
  The confirming follow-up is an **edge path-patch** (candidate L1-head→combiner
  edge, freezing the combiner's other inputs). 5-digit inconclusive (matches CE8).
  2-layer, addition only.
- **Edge-patch follow-up (2026-07-16, CE10)**: the edge path-patch ran. It found
  `L1.H1` (this CE8/CE9 cell) **causally drives the combiner at one depth** (6-digit
  k=3) — resolving CE9's "inert" null on `L1.H1` — but the single-position edge
  instrument is underpowered at most cells, so this is a one-depth crumb, not a
  broad confirmation (see CE10). CE9's instrument-limit framing stands.

<a id="ce10"></a>

### CE10: Addition model — At the combiner edge, a single L1 head carries the top cascade digit's computed carry at one depth — a causal crumb, but the single-position edge instrument is underpowered

- **Confidence**: **Low-Medium** — one clean isolated cell per model; underpowered
  at most cells; one-depth (fails the ≥2-depth bar); two Gate-2 rounds (first BLOCK
  for an inverted power control over-claiming anti-A6).
- **What it establishes**: (1) the answer-position combiner residual is causal
  (full `resid_mid` edge patch flips 1.00); (2) at 6-digit k=3 a **single head
  `L1.H1`** — the CE8 carry-routing cell CE9 found causally inert under pattern-
  redirect — carries the top cascade digit's carry through the combiner **MLP
  input**, and does so with a **computed, deciding-selective** signature
  (deciding-class flip 1.00, same-class-operand null 0.00, wrong-digit 0.00);
  similarly 5-digit `L1.H2` at k=2,3. The finer edge path-patch thus resolves what
  CE9's blunt redirect could not.
- **What it does NOT establish**: the single-position edge battery is **underpowered**
  at most cells (6/9 6-digit, 3/9 5-digit: a single-head-magnitude direct edge
  cannot flip them), **no head carries at ≥2 depths** (fails the pre-registered
  R-edge-selective bar), and the **5-digit direct residual path is causally live**
  (0.70–0.93). So A9's attention-delivery is **circumstantial, not confirmed**, and
  A6's residual-carry is **not refuted**. The edge-carrier (`L1.H1`/`L1.H2`) is not
  the CE9 selective/tracking head A9 predicts convergence on — a *partial,
  reconfigured* A9.
- **Supporting evidence**: 2026-07-16 edge-patch bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-cascade-handoff-edge-patch.md](study-maths/study-cascade-handoff-edge-patch.md),
  `results/study-cascade-handoff-edge-patch/results.json`). Controls: additivity
  3.3e-6; full-`resid_mid` flip 1.00.
- **Caveats**: a less LN-damped / multi-position edge instrument and a second depth
  (6-digit k=4) are needed to raise power; neuron-level (B2) on `L1.H1`+combiner and
  the 5-digit/6-digit divergence (B5) are the follow-ups. 2-layer, addition only.

<a id="ce11"></a>

### CE11: Addition model — The tri-state carry `ST` is position-specific at question positions (no cross-position probe transfer) and geometrically entangled with `SV` — C2/A4 template-sharing and A8 interference challenged for `ST`

- **Confidence**: **Medium** — clean and cross-model-replicated for `ST` (the one
  sub-task with a strong diagonal at the question site); linear-probe / question-
  position scoped.
- **What it establishes** (for `ST` at the question site `D'n`@resid_post(L0)):
  (1) an `ST` probe trained at digit `i` does **not** transfer to digit `j`
  (chance-relative retention 0.10/0.12 ≪ 0.6; transfer matrix strong diagonal,
  off-diagonal ≈ chance), and **mean-centering does not restore it** (positional-
  offset gain ≤ 0.09) — so **C2's shared-template and A4's transfer-for-free
  predictions are falsified for `ST`**, and not even A4's per-position-offset form
  holds. (2) `ST` and `SV` occupy **geometrically entangled** subspaces (principal
  angle 21° ≪ label-correlation null 64°/50°, both models) despite being
  label-*independent* at the same digit (MI≈0) — challenging C2's cross-subtask
  orthogonality and A8's low-interference for the carry family.
- **What it does NOT establish**: SA and SV template sharing — both are **diag-weak
  at the question site** (SA near chance at `D'n`; it decodes 1.00 only at the
  **answer position**, confirming CE2/CE3's base-add locus), so their transfer is
  not-assessable here. The answer-phase "tape vs register" binding question (do
  SA/SV transfer across *answer* positions; A4 just-in-time fetch) is untested (B4).
  "Position decodes at 1.00" is corroborating but a trivial positional-embedding
  consequence, not the mechanism.
- **Supporting evidence**: 2026-07-16 probe-transfer bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-probe-transfer.md](study-maths/study-probe-transfer.md),
  `results/study-probe-transfer/results.json`). Positive control: the diagonal
  (self-position) `ST` probe decodes well above chance (gain 0.31/0.39); the
  class-mean subspaces occupy 11–41% of variance (not a low-variance artifact).
- **Caveats**: linear probes; question-position + `ST`-scoped; 2-layer addition,
  middle digits, two models (agree). "Position-specific" = not linearly transferable
  at this site; weight sharing is still architectural (the model applies shared
  weights per digit — the *representation* at the read site is position-tied).

<a id="ce12"></a>

### CE12: Addition model — Answer-phase layout is split — `SA` is a just-in-time register (absent at `=`); `SV` is resolved/present at `=` but its per-digit slots are not orthogonal (tape refuted); both share an answer-side template

- **Confidence**: **Medium** — the SA/SV split and the answer-side-template
  contrast replicate across both models; linear-probe / answer-phase scoped.
- **What it establishes**: (1) **`SA` (answer digit) is a register**: absent at `=`
  (raw balanced acc ≈ chance 0.10), present at 1.00 only at its own answer position
  → just-in-time fetch (A4). (2) **`SV` (resolved carry) is present at `=`** for all
  middle digits (decodes ~0.60–1.00), consistent with CE7's resolution locus. (3)
  **The `SV` per-digit slots at `=` are NOT orthogonal** (6-digit 12–27°; 5-digit
  2/3 pairs < 60°) → the orthogonal-"tape" layout (A4's own alternative) is
  **refuted**; the 6-digit slots are additionally more aligned than a
  label-correlation null (entangled), with a 1-D binary-subspace caveat. (4) Both
  `SA` and `SV` **transfer across answer positions** (a shared answer-side template
  — SV beats the operand floor + centering; SA transfer is caveated as operand
  re-derivation), in contrast to the position-specific question-side `ST` (CE11) —
  so template-sharing is **position-of-computation-dependent**.
- **What it does NOT establish**: whether `SV` presence at `=` is *beyond a full
  operand recompute* (the isolated-operand `Dn`/`D'n` baseline omits the
  carry-determining lower digits, so it is too weak; a digits-`0..n` baseline is the
  future test); whether the `=` carries are *used* downstream (no causal test) — so
  A6/C3 are **untouched** and no "storage/bus" claim is made.
- **Supporting evidence**: 2026-07-16 answer-binding bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-answer-binding.md](study-maths/study-answer-binding.md),
  `results/study-answer-binding/results.json`). Positive control: diagonal SA/SV
  decode ≈ 1.00 at own answer positions.
- **Caveats**: linear decodability, not causal use; SV isolated-operand baseline is
  weak (F1); entanglement 6-digit-only + 1-D caveat; one 5-digit `SA_1` exception
  (+0.19 at `=`); middle digits, SV_0 excluded; 2-layer addition, two models.

<a id="ce13"></a>

### CE13: Addition model — Map-named ST nodes encode their class and co-carry single-step U-resolution (not multi-digit compounding); map-named SA L0 heads do not write the answer digit

- **Confidence**: **Medium** — cross-model directional agreement; encoding + ablation
  robust for the strong nodes; the U-resolution locality is 5-digit-scoped (6-digit
  Battery-C positive control structurally void).
- **What it establishes** (C5 step 1, both studied models): (1) every HF-map-named
  `ST` node linearly **encodes** its 3-way class in its head-output write (~1.00),
  and `SC` nodes their binary make-carry — confirming the paper's output-only tags;
  a same-position wrong-role baseline flags 3 nodes as position-decodable-not-
  uniquely-head-written. (2) The ST write **co-carries single-step local
  U-resolution**: on a `U` (sum-9) pair the OV write depends on the single-step
  incoming carry `cin` (cin/null 0.75–19.4, low/middle-digit-concentrated) — so the
  write is not a *pure* local class code, but this is single-step U-resolution
  (carry-out = cin by definition on `U`), **not** shown to be multi-digit SV
  compounding. (3) Map-named `SA` L0 heads do **not** encode the answer digit (weakly
  above chance) except the leading digit — the sum digit is an answer-position
  computation (CE11/CE12), so the `SA` L0 tag marks operand-fetch, not answer-write.
- **Relation to conjectures**: **A10 premise REFINED** (ST write co-carries local
  U-resolution → not purely "output-only local", but multi-digit compounding is
  untested — entry 2); **CE3 refined to redundancy** (see the CE3 follow-up note);
  **A2 untouched** (post-attention write ≠ pre-MLP sum-sufficiency); **A3**
  weakly-against (clean 3-way class, no off-axis symbol).
- **Supporting evidence**: 2026-07-16 node-output-encoding bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-node-output-encoding.md](study-maths/study-node-output-encoding.md),
  `results/study-node-output-encoding/results.json`). Node lists from HF
  `features.json`. Two Gate-2 rounds corrected over-reach in both directions.
- **Caveats**: linear-probe + interchange + baseline-controlled ablation; Battery-C
  U-locality 5-digit-scoped (6-digit control void); some low-digit ablation nodes
  marginal (~1 SE at N=300); single-step cin toggle (not multi-digit cascade — the
  compounding question is deferred to entry 2); 2-layer addition, two models.

<a id="ce14"></a>

### CE14: Addition model — Carry-specific attention-edge delivery to the answer-position combiner (A10 core partially confirmed); single-head selection not shown; redundant, sufficiency-not-necessity

- **Confidence**: **Medium** — carry-specific + ≥2-depth causal + consumer-specific,
  cross-model directional agreement; but sufficiency-not-necessity, selection
  underdetermined, direct path not excluded.
- **What it establishes** (C5 steps 2-4, both studied models): patching a map-named
  answer-position L1 consumer head's output edge into the high-Fail% L1-MLP combiner
  (less-damped MLP-only instrument) flips the top cascade digit at **≥ 2 depths**
  (via the H1+H2 joint pair), and the flip is **carry-specific** (deciding-matched
  null = 0.00 — it responds to the deciding-carry toggle, not to same-class filler
  changes) and **consumer-head-specific** (a non-consumer co-located head is inert).
  This partially confirms **A10's core** "attention fetches the deep carry to the
  answer-position L1-MLP combiner" — strengthening CE10's one-depth crumb to a
  carry-specific ≥2-depth (joint) result.
- **What it does NOT establish**: (1) **single-head selection (A9 / A10-b)** — the
  value-matched deciding-digit tracking head (`L1.H2`) is **not** the single-depth
  edge-causal head (`L1.H1`); no one head is both ≥2-depth-edge-causal AND tracking
  (the CE10/CE11 same-cell requirement). (2) **Necessity** — the edge is *sufficient*
  (patching flips) but the sufficient head (H1) is not *necessary* (ablation inert;
  H2 carries necessity+tracking); the mechanism is a **redundant H1/H2 pair**.
  (3) **Direct-path exclusion** — the direct-path arm is underpowered
  (`direct_scaled`=0). (4) **Head-specific value content** — `SV_n` decodes from the
  head's value but a co-located head decodes it equally (position-level, not
  head-attributable). "Distributed/redundant" vs "H2-selects/H1-delivers" is
  underdetermined.
- **Relation to conjectures**: **A10 core partially confirmed** (carry-specific
  distributed delivery), **selection sub-claim not supported**; **A9 not supported**;
  **A6 selective economy supported at H2** (cascade harmed, carry-free spared, gap ≫
  untagged baseline 0.00); **A5** hybrid routing gains partial causal support.
- **Supporting evidence**: 2026-07-16 SV-compounding bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-sv-compounding.md](study-maths/study-sv-compounding.md),
  `results/study-sv-compounding/results.json`). Consumer heads from HF
  `behaviors.json`. Three Gate-2 rounds (positive over-claim → over-correction on a
  broken null → calibrated).
- **Caveats**: sufficiency-not-necessity; selection underdetermined; direct path not
  excluded; value content not head-specific; 2-layer addition, two models.

<a id="ce15"></a>

### CE15: Addition model — The leading answer digit is produced by carry-specific L1-head-edge delivery to the sign-position combiner (mirrors CE14) — C5 step 5

- **Confidence**: **Medium** — consolidates CE14 at the leading-digit locus;
  5-digit across a genuine depth spread, 6-digit deep-chains only; inherits CE14's
  partial-confirmation caveats. Does not raise A10 (a consolidation, per LW-5).
- **What it establishes** (C5 step 5, both studied models): in the hard
  graded-cascade case (`99..9 + 00..01`, chain of k nines reaching the leading
  digit), the sign-position L1 consumer head edge patched into the sign-position
  L1-MLP combiner flips the leading digit `A_top` **carry-specifically** (real flip
  1.00, deciding-matched null 0.00) — Link 3 **verified** (5-digit at all depths
  k=1–4; 6-digit at deep chains k=4,5). So the leading digit uses the **same
  carry-specific attention-edge delivery mechanism as the middle digits (CE14)**,
  at the sign position (which is `A_top`'s readout / consuming position). This
  completes the C5 five-step program for the addition model (CE13 output encodings →
  CE14 SV compounding → CE15 leading-digit hard case), all landing on the same
  picture.
- **What it does NOT establish**: (1) **uniform 6-digit delivery** — the 6-digit
  head edge carries only for deep chains (k=4,5); shallow leading cascades (k=1–3)
  flip via the readout (Link 4) but not these heads' edge, carried by an
  **unadjudicated** path (direct arm underpowered). (2) **Selective economy (A6)** —
  not testable at the sign position (untagged baseline ≈ tagged, ~0.49: a general
  bottleneck / ablation too destructive). (3) Inherits all CE14 open questions:
  sufficiency-not-necessity, direct-path-not-excluded, single-cell selection unshown.
  Link 4 (whole-`resid_mid` patch) is readout-only, never counted toward A10.
- **Relation to conjectures**: **A10 consolidated at the leading locus, NOT raised**
  (held at medium; extends CE14's partial confirmation to all answer digits with
  more caveats); **A6 not testable here**; **A9 untouched**.
- **Supporting evidence**: 2026-07-16 leading-digit-walkthrough bundle
  ([results-by-time](maths-results-by-time.md), study
  [study-leading-digit-walkthrough.md](study-maths/study-leading-digit-walkthrough.md),
  `results/study-leading-digit-walkthrough/results.json`). Wiring from HF maps.
- **Caveats**: 5-digit genuine depth spread, 6-digit deep-chains only; economy
  uninformative at the sign locus; direct path not excluded; 2-layer addition, two
  models; a documented worked example, not a new mechanism beyond CE14.

<a id="ce16"></a>

### CE16: Addition model — SV implementation — canonical resolved-carry message, distributed ST-cluster source (`=` is a depot, not a value source), head-pair (SV) path effective with negligible skip carry, class-necessary pair; A10 items i–iii resolved, iv open

- **Confidence**: **Medium-high** for the three resolved implementation items
  (dual-gated, both models, controls pass); item iv (combiner form) unestimated.
  A parameter-estimation study on the confirmed SV wiring (working-axioms mode),
  not an existence test.
- **What it establishes** (both studied models, acc 1.000; controls: PC1
  reproduces CE14 joint-pair flip 1.00 / deciding-matched null 0.00; PC4
  carry-axis anchor sep 28.6/32.3):
  - **(i) Message** — the head→combiner edge carries a **canonical
    (format-invariant) resolved carry**: within-chain cross-deciding-position
    carry-probe transfer 1.00 = within-acc. The deciding *position* is
    additionally *decodable* from the edge after carry-axis removal
    (norm-matched, existence-only / non-causal co-rider).
  - **(ii) Source** — the carry source is the **question-tail ST cluster and is
    NEVER `=`**: the `=` value-patch arm flips 0.00 with carry-axis OV-projection
    ≈ 0 (**`=` is a depot, not a value source**, consistent with CE13). Among
    named ST sources, the deciding-ST cluster is carry-specific (matched null
    0.00) at all depths and dominant at 6d k3 (0.60); the chain-ST `rest` sites
    carry the mass at k2/k4 (source **distributed** across ST sites).
  - **(iii) Path + necessity** — the **head-pair (SV) path is the effective
    carrier** (real patch flip 1.00); the **skip/direct residual carries
    negligible carry** (0.14 6d / 0.077 5d, ≈200× below the head-pair signal;
    power injection at that magnitude, 1× and 2×, flips 0.00 → the model does not
    route carry through the skip). The pair is **class-necessary**: joint H1+H2
    mean-ablation collapses cascade accuracy (0.00 6d / 0.15 5d), spares
    carry-free (1.00), over a ~0 untagged-pair baseline
    (necessity-over-baseline 1.07 6d / 0.85 5d).
- **What it does NOT establish**: **(iv) combiner functional form** — Battery F
  instrument invalid (α-sweep at the combiner input produced no flips; spread
  0.00); unestimated, not a negative. The skip is **not formally excluded** —
  the power control was matched to the skip's own tiny magnitude, so it shows the
  skip is not USED, not that it COULDN'T carry (F1 correction). "Source is the
  ST cluster distributed", not localized to deciding-ST (dominant at only 1
  depth; F2 correction). The position co-rider is decode-existence, not causal
  (F3). 5d k3 R arm is leaky (residual 0.50) → ordinal only.
- **Relation to conjectures**: **A10 core held at medium-high; items i–iii
  RESOLVED, iv OPEN.** Source fork adjudicated toward **distributed ST / depot**
  (against "`=` carry depot read as value"). **A9 selection stays RETIRED** at
  the mechanism level (deciding-ST dominant at only 1 independent depth < the ≥2
  needed; CE14 same-cell tracking+edge bar unmet). **A6 economy RAISED to class
  level.** Closes CE14's direct-arm underpower gap in the "is it used" sense and
  CE10's one-depth crumb.
- **Supporting evidence**: 2026-07-16 SV-implementation sprint
  ([results-by-time](maths-results-by-time.md), study
  [study-sv-implementation.md](study-maths/study-sv-implementation.md),
  `results/study-sv-implementation/results.json`). Dual-gated (combined sprint
  pass): pre-launch PASS-WITH-CONDITIONS (SI-1..SI-10) + post-result
  PASS-WITH-CORRECTIONS (F1/F2/F3). Wiring from HF `behaviors.json` maps;
  CE6 carry axis from `st_tristate_geometry`.
- **Caveats**: combiner form unestimated (F failed); skip not formally excluded
  (power matched to skip's own magnitude); source distributed not localized; 2×
  power arm does not lift "excluded"; 2-layer addition, two models.

<a id="ce17"></a>

### CE17: Addition model — SV compounding arithmetic — combiner is a STEP function (A10 iv resolved); the L0 tail-relay (A11) is a single-site, unreplicated representational trace, causally undetermined; L1 edge output is local-class-sufficient — R-mixed

- **Confidence**: **Medium-high** for the combiner-step result (A10 iv; dual-gated,
  both models, endpoint-gated instrument); **low** for A11 (single-cell trace,
  unreplicated, causally undetermined). Parameter-estimation/attribution study on
  the confirmed SV wiring (working-axioms mode), not an existence test.
- **What it establishes** (both models, acc 1.000; controls: Y-ablation
  instrument reproduces CE13 low-digit impact 0.067 6d / 0.05 5d over ~0 untagged
  baseline; carry-axis anchor + behavioral gates pass):
  - **(A10 iv) Combiner transfer = STEP.** On-manifold α-sweep — interpolate the
    real captured consumer head-pair edge contribution `edge(α)=(1−α)z_c0+α·z_c1`
    substituted PRE-LN at the consuming position, endpoints gated to CE16's
    0.00/1.00 (the CE16 Battery-F dead-zero does NOT recur) — flips the leading
    digit sharply: 6d {α .25: 0.00, .5: 0.42, .75: 1.00}, 5d {.5: 0.17, .75:
    0.83, 1: 1.00}, threshold **α*≈0.75** both models, while the carry-axis
    projection rises linearly (α parameterizes the carry axis; readout is
    thresholded). A hard discretization, not a graded pass-through.
  - **(A11, representational) A single horizon-consistent trace.** In the 6d
    model, one question-tail ST site (**P11H2, horizon m=1**) decodes the resolved
    chain carry at **bacc 1.00 exactly at the depth where it can see the deciding
    digit** (k=2, d=2≥1) — the A11 horizon prediction — over its wrole/shuffled
    baselines by ≥0.2.
- **What it does NOT establish**:
  - **A11 is NOT replicated or causally shown.** The P11H2 trace fails its own
    within-site prediction at k=3 (d=1≥m=1 should resolve, decodes 0.59=NO), and
    5d has no strong-writing m>0 boundary site (null). Twin-**interchange** of the
    tail-ST OV writes (deepest-sufficient, insufficient, joint-all) flips the
    leading digit **0.00 at every depth, both models** — but because the valid
    control tests a different unit/target (mean-**ablation**, all-digits, random
    Qs) than the interchange, this null is **causally UNDETERMINED** (redundancy
    per CE13's interchange=0.00, vs interchange-too-weak-for-the-leading-digit —
    the CE16-F1 trap; F2 correction). The consumer edge output reconstructs from
    per-site **local class alone** (r²≈0.95–0.99; horizon features add no
    held-out gain over a permutation null — 6d ΔR² −0.05, 5d +0.02 no-winner at
    ceiling). So nothing forces the L0-relay account.
- **Relation to conjectures**: **A10 iv RESOLVED — step, α*≈0.75** (A10 items
  i–iv now all resolved). **A11 LOWERED to low** — representational hint at one
  site, unreplicated, causally undetermined; the balance **leans L1-local-
  sufficient**. **A9 stays RETIRED** (no causal selection at either L0-relay or
  L1-read locus). **A6/C3 (human sequential-cascade lean) NOT adjudicated** (the
  only positional-sequential signal is the single unreplicated cell). **R-mixed.**
- **Supporting evidence**: 2026-07-16 compounding-arithmetic sprint
  ([results-by-time](maths-results-by-time.md), study
  [study-compounding-arithmetic.md](study-maths/study-compounding-arithmetic.md),
  `results/study-compounding-arithmetic/results.json`). Dual-gated (combined
  sprint pass): pre-launch PASS-WITH-CONDITIONS (CA-1..CA-6, incl. re-pin to
  n_top=4 for genuine horizon crossings + the CE13-ablation Y control) +
  post-result PASS-WITH-CORRECTIONS (F1 H-not-replicated, F2 Y-causally-
  undetermined, F3/F4/F5 calibration). Reuses CE13 `_mean_ablate_acc`/ST_NODES +
  CE16 patch/axis machinery.
- **Caveats**: A11 horizon trace rests on ONE strong-writing site per model
  (deeper sites write ~0); Y null causally undetermined (interchange vs
  ablation-target mismatch); L local-sufficiency may reflect local↔resolved-carry
  correlation on chain stimuli + a reconstruction ceiling; T z-interpolation is
  un-regressed (CE16 co-rider rides in the blend — "step" rests on the endpoint
  gate + linear carry-proj); 2-layer addition, two models.

<a id="ce18"></a>

### CE18: Addition model — Cross-size SV — the role skeleton + step combiner generalize d5→d13, but redundancy does NOT thin with size (C6 not supported); large-n causal source signatures are probe-limited

- **Confidence**: **Medium-high** for role transfer + step-combiner generalization
  (dual-gated; d5/d6/d10/d13 all acc 1.000; step combiner endpoint-gated at every
  size). **Low** for A12-tightening / C6. The causal source-fork is **untested at
  n≥10** (probe-limited). Working-axioms mode (attribution/estimation across sizes).
- **What it establishes**:
  - **Role skeleton is size-general.** Question-tail/sign **ST writers** and
    high-`Fail%` **combiner MLPs** are present in the published d10/d13 maps
    (`Algo:A{k}.ST` + `Fail%`), and a causal answer-position **consumer head** is
    empirically identifiable at every size (d10/d13 have NO L1 consumer-head map
    tags — itself a role-transfer datapoint; the head is found by carry-specific
    causal flip). ST count {d5:7, d6:6, d10:10, d13:11}; combiner MLPs {6,7,10,14}.
  - **The combiner is a STEP function d5→d13** (the one robust cross-size CAUSAL
    result): on-manifold α-sweep endpoint-gated to each model's real 0/1
    (`endpoint_ok` all four), α*≈0.75 (d5/d6/d10) / 0.5 (d13). CE17 generalizes.
  - **Redundancy PERSISTS (does not thin) d5→d10** — single-node ST ablation gap
    ~0 at every size (0.076/0.044/0.004/0.000) and the class-minus-single gap does
    NOT shrink ({0.056, 0.116, 0.324} d5→d6→d10, increasing). Single-node
    interventions stay null at all n. **C6 (redundancy = small-model slack) NOT
    SUPPORTED** — redundancy reads intrinsic to the algorithm.
- **What it does NOT establish**:
  - **Causal source-fork at large n.** At d10/d13 the carry axis is weak (sep ~6
    vs ~30) and both the `=` arm and deciding-ST arm flip 0.00, so `=`-not-a-source
    is a null on a null background and the CE16 source signatures are **not
    reproduced** (probe-limited), not confirmed, at n≥10 (F1).
  - **d13 tightness** — whole-class ST ablation is only 0.040 (≈ CE13 single-node
    magnitude; PC2b clears the untagged baseline by only 0.02), so "very redundant"
    vs "ST-ablation ineffective at n_ctx 43" is unseparated — **d13 inconclusive**
    on tightness (F2). C6 is therefore "not supported" (d5/d6/d10), not "refuted".
  - **Monotone tightening** — 4 size points, no d7/d8/d9 gradient (XS-F); no
    monotone claim. **A11** was not rescued by scale (Battery L untriggered).
- **Relation to conjectures**: **A12 role-transfer + step-combiner → medium-high**;
  A12-tightening / **C6 → not supported (low)**; **A10 iv step combiner generalizes**;
  **A11 unchanged (low)**; A6 class necessity holds where the identified head is the
  class (d5/d10) but is not cross-size comparable (single-head vs pair selector, F3).
- **Supporting evidence**: 2026-07-16 cross-size-sv sprint
  ([results-by-time](maths-results-by-time.md), study
  [study-cross-size-sv.md](study-maths/study-cross-size-sv.md),
  `results/study-cross-size-sv/results.json`). Dual-gated: pre-launch
  PASS-WITH-CONDITIONS (XS-A…XS-F: per-digit index normalization, registries
  repopulated from maps + PC2b anchor, independent non-interchange tightness leg,
  carry-specific consumer-ID) + post-result PASS-WITH-CORRECTIONS (F1 source
  probe-limited, F2 C6 not-supported/d13-inconclusive, F3/F5). Reuses CE13
  `_mean_ablate_acc`/maps, CE16 patch/axis, CE17 combiner. Reads published maps
  (no HF uploads).
- **Caveats**: large-n carry axis weak (sep ~6) → source arm-probes probe-limited;
  d13 ST ablation instrument-weak (0.04); ablation gaps un-intervalled point
  estimates; consumer-ID single-head at large n (not map-tagged, not pair); 4 sizes
  no gradient; interchange leg F2-ambiguous throughout. 2-layer/3-head addition
  zoo, one large-n seed family.

<a id="ce19"></a>

### CE19: Addition model — Compounding locus — the multi-digit carry is compounded in the L1 consumer READ, not by an L0 positional relay (A11 not supported); decisive via a decorrelation + invisible-cell discriminator

- **Confidence**: **Medium** — settles the CE17-open fork against an L0 relay where
  the instrument has power (5d), with a linear-probe caveat and 6d underpowered. A
  clean refutation-as-finding (working axioms), not an existence test.
- **What it establishes** (both models acc 1.000; the design escapes CE17's two
  blockers):
  - **Decorrelation lever**: a chain-ST site inside the 999-run has local class
    fixed at U (decode 0.50 exactly — verified uninformative) while the resolved
    carry varies. **INVISIBLE-decorrelated** cells (deciding digit below the site's
    horizon) are the sole discriminator: a carry there could ONLY be relayed.
  - **No relayed carry**: at invisible-decorrelated cells the resolved carry decodes
    at **chance** in both models. Where the write is **readable at that depth**
    (5d P9H1 k3, per-depth readability control passes), chance = a genuine "no
    relay" → **R-L1-read**: the compounding is completed in the L1 consumer read.
    The ST write decodes carry **only at the fully-correlated cell** (site =
    deciding digit, 0.99–1.00) — the CE13 local single-step picture, no relay.
  - **Class-necessity (KO)**: ablating the sufficient-ST class breaks the digit
    differentially (0.24–0.50) over a 0.00 specificity-null and 0.00 untagged
    baseline → the L1 read depends on the ST cluster; single sites are not
    necessary (redundant). KO is a necessity anchor (does not itself discriminate
    relay vs local resolution — DH is load-bearing).
- **What it does NOT establish**: **6d is underpowered** — the deep-chain (k=3/4)
  ST writes wash out (fail the per-depth readability control), so 6d neither
  supports nor refutes A11. DH is a **linear** probe (a non-linear/different-
  subspace relay is not excluded). The refutation rests on **one readable invisible
  cell (5d)**. So A11 is **not supported → low, not formally rejected**.
- **Relation to conjectures**: **A11 (positional L0 relay) NOT SUPPORTED → low**;
  **compounding locus = L1-read** (5d proven, 6d consistent-underpowered); **A9
  stays retired** (L1 read is class-level/redundant, not a single-head selector);
  **A6** class-necessity of the ST cluster confirmed again; **human C3/sequential-
  cascade lean not supported at the L0 tail** (caveats). Settles the CE17 R-mixed
  fork against a relay.
- **Supporting evidence**: 2026-07-16 compounding-locus sprint
  ([results-by-time](maths-results-by-time.md), study
  [study-compounding-locus.md](study-maths/study-compounding-locus.md),
  `results/study-compounding-locus/results.json`). Dual-gated: pre-launch
  BLOCK→PASS (LOC-1…LOC-6: invisible-decorrelated discriminator, KO specificity
  null, honest power ceiling) + post-result PASS-WITH-CORRECTIONS (F2 per-depth
  readability → 6d underpowered, F1 "refuted"→"not supported", F3/F4). Reuses
  CE13/CE16/CE17 modules.
- **Caveats**: linear probe (non-linear relay not excluded); one readable invisible
  cell (5d); 6d underpowered (deep-chain writes wash out); KO necessity-only.
  2-layer/3-head addition, two models.

<a id="ce20"></a>

### CE20: Mixed model — the addition SV representation replicates across ADD/SUB/NEG (writers encode the tri-state; resolved carry/borrow is a binary code at the last-layer combiner), and delivery is class-dependent (ADD residual-only, SUB/NEG residual + last-layer attention)

- **Confidence**: **Medium-high** for the representation replication (writer
  encoding + binary resolved cascade, all three classes, clean vs untrained
  control, on the 3-layer mixed model); **Medium** for the class-dependent
  delivery pathway (single model/seed, single-step U delivery at k=2, deciding-
  matched null). First mixed-model result; addition→mixed generalization.
- **What it establishes** (`ins1_mix_d6_l3_h4_t40K_s372001`, 3 layers/4 heads,
  per-class accuracy 1.000):
  - **Writer encoding (CE13 replicates)** — the question-tail tri-state writers
    (`ST` add / `MT` sub / `NT` neg) linearly encode their 3-way class at L0:
    balanced tri-acc ~1.00 at digits 1–2 for all three classes vs an untrained
    control at ~chance (0.45–0.60). (d3 weaker ~0.45 — a writer-locus caveat.)
  - **Binary resolved cascade at the combiner (CE6/CE7 replicates)** — the
    resolved carry/borrow-in (`SV`/`MV`/`NV`, binary) decodes ~1.00 at the
    **last-layer (L2)** answer-position MLP input for all three classes (control
    ~0.6). The tri-state is resolved to a clean binary at the combiner — the
    latent representation of intermediate results, replicated across classes.
  - **Cascade-specific delivery (CE14 analog)** — with a single-step U pair
    (carry/borrow-in toggled) the resolved cascade reaches the combiner
    **carry/borrow-specifically** (deciding-matched null 0.00 on every arm): it
    rides the **pre-last-layer residual** for all classes (`resid_pre` flip 1.00),
    and last-layer attention **additionally** delivers it for **SUB/NEG (1.00)**
    but **not ADD (0.00)** — the inserted addition circuit resolves earlier and
    rides the residual; the freshly-learned subtraction cascades also use
    last-layer-attention delivery (the 2-layer CE14 head-delivery picture).
- **What it does NOT establish**: multi-depth delivery (only single-step U at
  k=2; the CE14 ≥2-depth bar not attempted); the whole-MLP zero-ablation combiner
  test is redundancy-limited (clean only at scattered digits, esp. ADD [] —
  matching addition CE5's redundancy), so combiner *causality* rests on the
  `full_resid` edge (flip 1.00, null 0.00) not zero-ablation; d3 writer locus
  unpinned; linear probes; single model/seed.
- **Relation to conjectures**: **C5 confirmed** on a new architecture (map roles
  borne out); **A10 confirmed at the representation level, refined on delivery**
  (attention-delivery holds for SUB/NEG, ADD is residual-delivered — the extra
  layer relocates delivery earlier); **A12 confirmed** (interface generalises to
  3 layers and to borrow/neg-borrow tasks).
- **Supporting evidence**: 2026-07-16 mixed-SV-replication study
  ([study-mixed-sv-replication.md](study-maths/study-mixed-sv-replication.md),
  `results/study-mixed-sv/results.json`, `results/study-mixed-map/results.json`;
  `scripts/mixed_map.py`, `scripts/mixed_sv.py`). Library build:
  `neg_labels`/`neg_ntc_functions`/NTC-NT tags/class-aware `_combiner_is_causal`,
  tested in `tests/test_scaling_and_sub.py` (21 passed incl. HF mixed NEG).
- **Caveats**: single mixed model, one seed; single-step U delivery (k=2),
  no depth sweep; zero-ablation combiner underpowered; d3 writer locus caveat;
  representational probes linear; 3-layer/4-head mixed add/sub, initialised from
  a 6-digit addition model.

<a id="ce21"></a>

### CE21: Mixed model — SGN is the top-of-cascade `D≥D'` comparison delivered to the sign-position combiner (CE15 analog); OPR is broadcast-decodable but not an additive rank-1 control at the combiner; shared SA/MD/ND heads carry a hybrid (shared-head, operation-rotated) readout

- **Confidence**: **Medium-high** for SGN (three converging clean tests);
  **Medium** for the OPR broadcast + not-steerable-at-combiner split and the
  hybrid A7/C2 overlap verdict (single model/seed; rank-1 additive steer only;
  subspace angles on 10-way readouts).
- **What it establishes**:
  - **SGN = comparison → sign (CE15 mechanism replicates for the sign)**: crossing
    `D≥D' ↔ D<D'` at the deciding digit flips SGN 1.00; the sign is a perfectly
    decodable binary at `=` (1.000, no dedicated `U`); a full-resid edge patch at
    the sign-producing position (`=`) flips SGN **comparison-specifically**
    (flip 1.00, deciding-matched null 0.00). The sign is the top-of-cascade
    product delivered to its combiner, exactly as the leading carry is (CE15).
  - **OPR**: the operator (+/−) is linearly decodable **1.000 broadcast across
    every layer/position** (P6 → L2 combiner input) — A7's low-D operator signal
    available broadly — but a **rank-1 additive steer** (add-mean − sub-mean,
    norm 20.6 > site 17) at the combiner input changes the answer digit **0.00**
    of the time: the operator is consumed **upstream at the SLT selector**, so
    A7's "move along the direction to flip the family" form fails at the combine
    site.
  - **Shared engine (A7 vs C2) is a hybrid**: on the map's shared L0 `SA`/`MD`/`ND`
    heads the per-operation digit-readout subspaces are **more separated than a
    random-init control for ADD-vs-subtraction** (68°/71° trained vs 46°/50°
    control) but **overlapping for SUB-vs-NEG** (48° vs 46° control). Shared heads
    (structural A7) with operation-specific readout rotations; strong C2
    (near-orthogonal 90°) and strong A7 (heavy overlap / rank-1 steer) both
    refuted.
- **What it does NOT establish**: a learned low-rank (rank>1) operator control or
  an SLT-sited steer (post-deadline); causal use of the readout overlap; the
  angle verdict is relative to the random-init baseline only.
- **Relation to conjectures**: **A7 partially confirmed / partially refuted**
  (broadcast operator + shared heads + SUB/NEG overlap FOR; rank-1-steer null +
  ADD readout more-separated-than-chance AGAINST); **C2 partially confirmed /
  partially refuted** (ADD-vs-sub separation FOR; not orthogonal, heads shared
  AGAINST); **C5 confirmed** (`OPR`/`SGN`/`SLT` roles borne out); **A10 extended
  to the sign** (fetch-to-combiner produces SGN, CE15 mirror).
- **Supporting evidence**: 2026-07-16 mixed OPR/SGN/shared-engine study
  ([study-mixed-opr-sgn.md](study-maths/study-mixed-opr-sgn.md),
  `results/study-mixed-opr-sgn/results.json`; `scripts/mixed_opr_sgn.py`).
- **Caveats**: single mixed model/seed; M4 rank-1 additive steer only (not
  SLT-sited); M6 subspace angles on 10-dim readouts in d_model; linear probes.

<a id="ce22"></a>

### CE22: Mixed model — the SV-implementation signatures replicate across ADD/SUB/NEG: the combiner is a STEP (A10 iv), the resolved carry/borrow is a canonical format-invariant code, and `=` is not the middle-digit source; class-necessity is redundancy-blurred (not scored)

- **Confidence**: **Medium-high** for the STEP combiner + canonical-message +
  `=`-not-a-source replication (three converging batteries, all three classes,
  clean endpoints/controls); **not scored** for class-necessity (redundancy-blurred
  + stimulus-confounded). Single mixed model/seed; middle digit k=2. Completes the
  SV-*mechanism* (not just representation, CE20) account on the mixed model.
- **What it establishes** (`ins1_mix_d6_l3_h4_t40K_s372001`, per class):
  - **STEP combiner (CE17 / A10 iv)** — an on-manifold α-sweep of the L2
    combiner input (endpoints gated to the real A_k(0)/A_k(1)) flips the answer
    digit as a **two-level step at α*≈0.5** for ADD, SUB and NEG, while the
    carry-axis projection rises linearly. The combiner is a hard discretizer, not
    a graded pass-through — CE17 generalizes to the 3-layer model and to the
    borrow (MV) and neg-borrow (NV) cascades.
  - **Canonical message (CE16 i)** — a resolved carry/borrow probe trained at the
    combiner input for one deciding digit transfers to another at **1.00** (all
    classes): the delivered carry is a canonical, format-invariant code.
  - **`=`-not-a-source (CE16 ii)** — patching the `=` residual into a
    **middle-digit** target flips A_k **0** (all classes), while the
    combiner-input control flips **1**. `=` is a depot, not the middle-digit value
    source (the sign IS computed at `=`, scored separately in CE21).
- **What it does NOT establish**: **class-necessity (CE16 iii / A6)** — ablating
  the whole map-named writer class (ADD=ST, SUB=MT, NEG=MT+GT) is harmless for
  ADD/SUB cascades (== a truly-untagged baseline, both 1.00: redundancy-blurred,
  matching CE13/CE18), and the NEG signal (difference-writers load-bearing for
  carry-free NEG digits) is confounded by cascade-stimulus triviality — so A6 is
  **not scored** on mixed. B1 interpolation is on-manifold (rides non-carry content;
  mitigated by endpoint-gating + two-level readout). Single seed, k=2, 5-point α.
- **Relation to conjectures**: **A10 iv/i/ii confirmed on the mixed model**
  (STEP combiner, canonical message, `=`-not-source, all three classes);
  **A12 strengthened** (the SV *implementation* — not just representation — ports
  to a new architecture + the subtraction task families); **A6 untouched**
  (redundancy-blurred; the redundancy axiom holds).
- **Supporting evidence**: 2026-07-16 mixed SV-implementation study
  ([study-mixed-sv-implementation.md](study-maths/study-mixed-sv-implementation.md),
  `results/study-mixed-sv-impl/results.json`; `scripts/mixed_sv_impl.py`; reads the
  M0 map registry). Combined sprint gate (self-adversarial); B2 refixed after a
  first-run invalid baseline.
- **Caveats**: single mixed model/seed; middle digit k=2; B1 5-point α grid;
  necessity (B2) unresolved (redundancy + cascade-stimulus confound); linear
  probe (B4); 3-layer/4-head mixed add/sub, initialised from a 6-digit addition
  model.

<a id="ce23"></a>

### CE23: Mixed model — A7's low-rank operator-control form is refuted at the selector too (rank-1 steer + single SLT head both fail); the engine is shared at the combiner but the add/sub selection is a distributed, high-dimensional L1 transformation (leans C2 on selection)

- **Confidence**: **Medium-high** for the refutation (positive control passes at
  0.96; rank-1 and single-head both flat 0.00 to the correct digit; converges
  with CE21's combiner-site null). Single mixed model/seed, k=2, rank-1 (not
  rank-r) steer. The decisive A7-vs-C2 verdict for the mixed shared engine.
- **What it establishes** (`ins1_mix_d6_l3_h4_t40K_s372001`, matched operands,
  middle digit k=2, selector layer L1):
  - **Selector-stage state is causal / engine shared at the combiner** — patching
    the whole `resid_post(L1)` at the producing position from an ADD run onto a
    SUB run flips the answer digit to the **correct ADD digit 96%** of the time:
    the same downstream L2 combiner produces the right digit for either operation
    given the L1 state (a shared combiner).
  - **Not a low-rank control** — a rank-1 operator direction
    (`mean_add − mean_sub` of `resid_post(L1)`, norm 20) added to a SUB run flips
    the digit **0%** (changed 0%); and replacing the **single SLT head** (`L1H1`)
    z with its ADD value changes the digit only 22% and **never** to the correct
    ADD digit (0%). So the operator is not a compact additive control at the
    selector — mirroring CE21's null at the combiner input.
- **What it does NOT establish**: that *no* low-rank subspace steers (only rank-1
  additive was tested; a learned rank-r subspace is untested); the SGN-side
  analog; anything beyond k=2 / one model.
- **Relation to conjectures**: **A7's function-vector / low-rank-control mechanism
  is REFUTED** (rank-1 steer null at both the selector and the combiner; single
  SLT head does not select) — its "shared machinery exists" half survives (shared
  L2 combiner; shared SA/MD/ND head locations from the map). **C2 partially
  supported on the selection**: the add/sub choice needs the full distributed L1
  state (not a compact switch), and (with CE21's M6: ADD readout more separated
  than a random-init control) the *selection* leans C2, while the *combiner* is
  shared. Net A7→**low** on the control mechanism (hybrid overall); C2→partially
  up on selection.
- **Supporting evidence**: 2026-07-16 mixed SLT-sited shared-engine study
  ([study-mixed-shared-engine.md](study-maths/study-mixed-shared-engine.md),
  `results/study-mixed-shared-engine/results.json`; `scripts/mixed_shared_engine.py`).
  Builds on CE21 (combiner-site operator null) + CE21 M6 (readout overlap).
- **Caveats**: single model/seed; k=2; rank-1 additive steer only (rank-r
  subspace untested); one selector head for the head-patch arm; matched-operand
  design guards the operand confound.

<a id="ce24"></a>

### CE24: Addition model — Compounding locus v2 — the canonical resolved carry is a LAYER-1 property (emerges in the L1 read, not at L0's output); conclusive layer-localization via cross-depth transfer, superseding CE19

- **Confidence**: **Medium-high** for the layer-localization (both models,
  representational + CE16-causal; conclusive where CE19 was inconclusive), with a
  linear-probe caveat. The decisive successor to CE19.
- **What it establishes** (both addition models, acc 1.000; 9-free stimuli per the
  human's `66666+33334`/`66666+33433` insight, + a 9-containing equivalence arm):
  - **The canonical carry is an L1 property.** Cross-depth transfer (train a carry
    decoder at chain-depth k1, test at k2 — a canonical, input-pattern-invariant
    code transfers; raw input-reading cannot) of the resolved carry at the L1
    combiner input, projected on CE16's **answer-agnostic carry axis** (anchored 2
    digits from A_top), transfers **1.00** in both models; the **answer-top digit
    does NOT ride that axis** (0.12/0.13 vs 0.78 in the full residual) → carry-
    specific, answer-decorrelated.
  - **L0's output has no carry-specific canonical code** (at the `=` gather
    position): **extreme-pair** cross-depth transfer ≈ chance (6d 0.51/0.53; 5d
    0.13/0.50, with bootstrap CIs) despite a high within-depth ceiling (0.81/0.91,
    so signal is present — no washout); the weak all-pairs transfer (0.65/0.46) is
    no larger than a pure **magnitude nuisance** (0.65/0.53) = leakage, not carry.
  - **9-free ≡ 9-containing** (L0 0.65↔0.67; L1 1.00) → generalizes to the
    literal-`9` edge case.
  - Method: this fixes CE19's two flaws — the OV-write washout (reads the full
    residual) and the ill-posed invisible-cell discriminator (MSD-first layout +
    causal mask make an L0 carry-direction relay impossible; reframed to
    layer-localization via cross-depth transfer at the see-everything gather).
- **What it does NOT establish**: a fresh CAUSAL locus — this study's own causal
  battery (LC, OV-write patch at a sign-ST site) is **invalid** (mis-sited, ~0
  consumer attention; washout), so the causal locus leans on **CE16** (head-pair
  edge patch flips 1.00). Linear probe only (a non-linear/rotated-frame L0 carry is
  not excluded — the Procrustes backstop overfit, unreliable). L0 tested at the `=`
  gather position; ST/sign L0 sites are covered cross-study by CE17/CE19.
- **Relation to conjectures**: **A11 (positional L0 relay) rejected/low** (no
  canonical carry at L0), and the **compounding locus is conclusively the L1 read**
  (representational + CE16-causal, both models) — superseding CE19's
  5d-only/6d-underpowered verdict. **A9** stays retired; **A6** unaffected.
- **Supporting evidence**: 2026-07-16 compounding-locus-v2 sprint
  ([study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md),
  `results/study-compounding-locus-v2/results.json`;
  `scripts/compounding_locus_v2.py`). Dual-gated: pre-launch PASS-WITH-CONDITIONS
  (CLV2-1…CLV2-6: nuisance-transfer controls, CE16 carry-axis L1 anchor,
  answer-digit decorrelation, Procrustes backstop, consumer-read LC site, 9-free +
  equivalence arm) + post-result PASS-WITH-CORRECTIONS (F1 causal→representational
  scope, F2 extreme-pair CIs lead the L0-null, F3 drop Procrustes claim, F5/F7
  TF/LP not-run + `=`-site scope). Reuses CE16 carry axis + edge.
- **Caveats**: linear probe (rotated-frame L0 carry not excluded — Procrustes
  overfit); L0 tested at `=` only (ST/sign via CE17/CE19); own causal battery
  invalid (causal locus per CE16); nuisance transfers are point estimates.
  2-layer/3-head addition, two models.
- **TF addendum (token-time, run 2026-07-16, dual-gated)**: Battery TF (deferred
  from the main run) closes the *token-time* question (CE24 settled the *layer*).
  Using 9-free graded chains with a run-break (so `carry_out(top) = make-carry AND
  run-intact`, decorrelated from the local make-carry) and **cross-deciding-position
  transfer** at each (token position × layer): **LAZY (representational) propagation
  + eager local**, both models. The propagated canonical carry appears **only at the
  sign token, layer L1** (transfer→1.0; ~chance at every operand token and at `=`),
  **never canonical at L0** — a genuine **~2-token deferral past full input
  availability (D'_0)** and past the `=` gather (chance; consistent with
  `=`-is-a-depot). The **local single-step make-carry is eager, in-place at layer 0
  at its own token** (0.76–0.94; CE13). Post-result gate correction: the
  make-carry-token→D'_0 span is information-availability (MSD-first layout, CE19),
  **not** laziness. Reusable cross-model code:
  `quanta_maths/maths_temporal_finalization.py` (unit-tested); artifact
  `results/study-compounding-locus-tf/results.json`. Net picture: **single-step
  resolution computed eagerly in-place (L0); multi-digit propagation represented
  lazily at the answer read (L1)** — the token-time complement to CE24. Caveats:
  linear-probe; point estimates; coarse tail; may differ by model (run the zoo).

<a id="ce25"></a>

### CE25: Mixed model — the class-dependent carry/borrow delivery pathway holds across cascade depths 2–4 (ADD residual-only; SUB/NEG residual + last-layer attention), carry/borrow-specific; reusable cross-model sweep promoted to the library

- **Confidence**: **Medium-high** — clean and consistent across depths 2/3/4, all
  three classes, with deciding-matched nulls at 0.00, a passing full-resid positive
  control, and an untrained control that does not deliver; single mixed model/seed.
  Extends CE20's single-step (k=1) delivery to the CE14 ≥2-depth bar.
- **What it establishes** (`ins1_mix_d6_l3_h4_t40K_s372001`; deciding digit at
  units; read digit = depth k∈{2,3,4}; combiner = last layer L2):
  - The resolved carry/borrow reaches the combiner **carry/borrow-specifically**
    (deciding-matched null 0.00) at **every tested depth** for ADD, SUB and NEG.
  - The **delivery route is class-dependent and depth-stable**: for **ADD**,
    last-layer attention does **not** deliver (`lastlayer_attn` flip 0.00 at all
    depths) — the carry rides the residual (`resid_pre`/`full_resid` flip 1.00);
    for **SUB** and **NEG**, last-layer attention **does** deliver (`lastlayer_attn`
    flip 1.00 at all depths) in addition to the residual. So CE20's
    single-step class-dependent pathway is a genuine multi-digit-cascade property
    (the inserted addition circuit resolves the carry early and rides the residual;
    the freshly-learned subtraction cascades also use last-layer-attention delivery,
    the 2-layer CE14 picture).
  - **Negative control**: on an untrained model the discriminating arm
    (`lastlayer_attn`) delivers nothing (flip 0.00, all classes).
- **What it does NOT establish**: per-head attribution (the attention arm is a
  whole-last-layer-attention patch); depths ≥5 (no room for the class-control top
  digit at n=6); anything beyond one model/seed. `full_resid`/`resid_pre` are
  near-tautological whole-vector patches (their specificity comes from the null).
- **Relation to conjectures**: **A10 delivery confirmed at depth on the mixed
  model** (carry/borrow-specific across ≥2 depths per class); the delivery *route*
  is a layer-general class-dependent refinement of CE14's "attention delivers"
  (holds for SUB/NEG; ADD is residual-delivered). **A12** further supported
  (depth-general on the new architecture/tasks).
- **Supporting evidence**: 2026-07-16 mixed delivery-depth study
  ([study-mixed-delivery-depth.md](study-maths/study-mixed-delivery-depth.md),
  `results/study-mixed-delivery-depth/results.json`;
  `scripts/mixed_delivery_depth.py`). Reusable core promoted to
  `quanta_maths/maths_cascade.py` (`make_cascade_operands`,
  `combiner_delivery_flip`, `combiner_delivery_sweep`) + `tests/test_cascade.py`
  (5 offline + 3 HF; full suite 73 passed) — runnable across the model zoo.
- **Caveats**: single mixed model/seed; depths 2–4; deciding digit fixed at units;
  whole-last-layer-attention patch (not per-head); 3-layer/4-head mixed add/sub,
  initialised from a 6-digit addition model.

### CE26: Mixed model (8-digit) — the SV mechanism generalizes d6→d8 (binary resolved cascade, STEP combiner, canonical message, `=`-not-source, shared combiner) for ADD/SUB/NEG; the delivery route is depth/size-dependent (d8 SUB/NEG switch to attention at deep cascades)

- **Confidence**: **Medium-high** for the core-mechanism generalization (clean,
  all three classes, deciding-matched nulls 0.00, endpoint-gated STEP, untrained
  control at chance); **Medium** for the depth-dependent delivery-route difference
  (single d8 model/seed, whole-last-layer-attention patch). First 8-digit mixed
  datapoint; extends the d6-only CE20/CE22/CE25. Map-free (no published d8 map).
- **What it establishes** (`ins1_mix_d8_l3_h4_t70K_s572091`, 3 layers/4 heads,
  per-class accuracy ADD 1.00 / SUB 1.00 / NEG 0.997):
  - **Core SV mechanism replicates d6→d8, all classes**: the resolved
    carry/borrow is a clean **binary** code at the L2 combiner input (~1.00 vs
    untrained ~0.61); the combiner is a **STEP** function (on-manifold α-sweep,
    endpoints gated, α*≈0.5); the delivered carry is a **canonical**
    format-invariant code (cross-digit transfer 1.00); **`=` is not the
    middle-digit source** (`=`-arm flip 0, combiner-input control 1); the combiner
    is **shared** (A1–A6 each tag STC+MTC+NTC on the same L2 answer-MLP node).
  - **Delivery route is depth/size-dependent (the d6→d8 difference)**: ADD
    delivers via the **residual only** at all depths (attention 0.00). SUB and NEG
    deliver via the **residual at shallow depths (2, 3)** but **switch to
    last-layer attention at depth 4** (resid_pre 0.00 / attn 1.00) — whereas d6
    used both routes at every depth. Carry/borrow-specific throughout
    (deciding-matched null 0.00; stimuli valid at every depth).
- **What it does NOT establish** (map-blocked — no published d8 verified map):
  writer-encoding at the question tail (the probed site is not the d8 writer
  locus — untrained ≥ trained, invalid, not scored); class-necessity;
  SLT-sited shared-engine (A7); the paper Fail%/Impact behavior map + mechanism
  diagram. Single d8 model/seed. `Probe:DELIVERY` node tag records only the
  shallow (depth-2) route.
- **Relation to conjectures**: **A12 confirmed on a new size (d8)** for the core
  SV mechanism (representation / STEP / canonical / `=`-not-source / shared
  combiner). **A10 delivery refined**: the delivery *route* is not universal — it
  is model/size/depth-specific (d8 SUB/NEG use attention only at deep cascades),
  as CE25 flagged. **A6/A7 untouched** (map-blocked).
- **Supporting evidence**: 2026-07-16 d8 mixed SV study
  ([study-mixed-d8-sv.md](study-maths/study-mixed-d8-sv.md),
  `results/study-mixed-d8/results.json`; `scripts/mixed_d8_sv.py`). Reusable d8 SV
  node-tag dataset: `results/study-mixed-d8/features.json` (18 combiner `Algo`
  tags) + `behaviors.json` (21 `Probe:DELIVERY` tags). Reuses the model-general
  `maths_cascade` + the CE20/CE22 batteries; combiners tagged via the
  redundancy-proof combiner-input criterion (zero-ablation is redundancy-limited
  at d8).
- **Caveats**: single d8 model/seed; map-blocked pieces above; delivery tag
  depth-2 only (full depth sweep in results.json); whole-last-layer-attention
  patch; writer-encoding invalid on d8.
