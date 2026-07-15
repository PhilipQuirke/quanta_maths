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

### CE1: Trained addition-model digit embeddings are near-isotropic 9-D categorical codes with a weak, training-induced circular *ordering* — not a dominant low-rank circle/helix

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

### CE2: At layer-0 operand-fetch heads, the value-path output is indistinguishable from linear transport of the (weakly-circular) digit embeddings

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

### CE3: The carry is computed by binary make-carry heads at answer positions, dissociated from base-add heads; the tri-state U-resolution is a separate, unlocated path

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

### CE4: The tri-state U-resolution flip is transmitted by an MLP-heavy L0/L1 path distinct from the make-carry heads; whether it is combined or merely relayed is unresolved

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

### CE5: The tri-state U-combiner is the answer-position layer-1 MLP; layer-0 nodes relay the running carry (conduit)

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

### CE6: At the U-combiner's input the carry is a clean binary code — no distinct off-axis tri-state; the resolved carry is already linearly present there

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

### CE7: No dedicated `{0,1,U}` tri-state symbol at any answer-position residual site; `U` is resolved to binary around L1-attention

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

### CE8: Attention routing is hybrid — a few heads relocate their target with carry state (6-digit only); most cells are target-static

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

### CE9: At node/attention-pattern granularity the deep `...999` cascade mechanism is not localizable — no single-cell selection, no sequential per-digit state, real graded tail state

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

### CE10: At the combiner edge, a single L1 head carries the top cascade digit's computed carry at one depth — a causal crumb, but the single-position edge instrument is underpowered

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

### CE11: The tri-state carry `ST` is position-specific at question positions (no cross-position probe transfer) and geometrically entangled with `SV` — C2/A4 template-sharing and A8 interference challenged for `ST`

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

### CE12: Answer-phase layout is split — `SA` is a just-in-time register (absent at `=`); `SV` is resolved/present at `=` but its per-digit slots are not orthogonal (tape refuted); both share an answer-side template

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
