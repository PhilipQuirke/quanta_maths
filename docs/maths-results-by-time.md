# Maths Results By Time (maths-results-by-time.md)

Read role and rules: [Results Ledger](thor-document-rules.md#results-ledger).

Append-only chronological record of completed result bundles for the `maths`
thread. Preserve order. Record what each bundle covered and where its artifacts
live — not the current best story (that goes in
[maths-results-summary.md](maths-results-summary.md)).

## Bundles

### 2026-07-14 — Digit-embedding geometry audit

- **Covered**: Weights-only geometry of the 10 digit-token rows of `W_E` and
  `W_U` for 4 accurate addition models (`add_d5_l2_h3_t15K_s372001`,
  `add_d6_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`,
  `add_d6_l2_h3_t20K_s572091`), 1 inaccurate 1-layer model
  (`add_d5_l1_h3_t30K_s372001`, descriptive), and an untrained control.
  Metrics: per-component variance shares, real-DFT spectrum, freq-1-plane and
  unique-linear shares with 10k-draw label-permutation p-values, circular
  ordering / wrap-around, embed↔unembed principal angles. Positive control:
  planted circle/helix/noise sweep with detection + permutation-null
  calibration (isotropic + anisotropic).
- **Artifacts** (local, no HF upload):
  `results/study-digit-embedding-geometry/positive_control.json`,
  `model_geometry.json`, `embed_variance_spectra.png`, `embed_pc_planes.png`;
  script `scripts/digit_embedding_geometry.py`.
- **Caveats / coverage gaps**: Weights-only (no causal / computational use;
  no activation-level geometry). Raw pre-LayerNorm weights only — the
  pre-registered LN-aware secondary analysis (A-9) was **not** run this pass.
  n = 10 vectors per matrix; power from permutation null + replication.
  Replication is mostly across size + 2 independent seeds. Positive control's
  freq1 metric is noise-inflated (+0.1–0.2 at low structure).
- **Linked study**:
  [study-maths/study-digit-embedding-geometry.md](study-maths/study-digit-embedding-geometry.md)

### 2026-07-14 — Pair-sum sufficiency at operand-fetch heads (A2 assay; instrument failure)

- **Covered**: Head-output (pre-MLP value path), LN(MLP-in), MLP-post, and
  resid geometry over 100 `(Dn,D'n)` cells (no lower carry) at layer-0
  operand-fetch heads in `add_d5_l2_h3_t15K_s372001` (node `P14.L0.H1`) and
  `add_d6_l2_h3_t20K_s173289` (nodes `P15–P20.L0.H1`). Metrics: CV
  `R²_sum`/`R²_pair` + ratio, equal-sum collapse vs control, sum-arc PCA,
  tri-cluster silhouette at 4 stages, operand-attention gating, ablation-impact
  node confirmation. Positive control: sum-sufficient / operand-identity /
  categorical / **circular-transport (from real `W_E·W_V`)** references.
- **Artifacts** (local, no HF upload):
  `results/study-pair-sum-sufficiency/positive_control.json`,
  `model_results.json`, `head_output_pca.png`; script
  `scripts/pair_sum_sufficiency.py`.
- **Caveats / coverage gaps**: **Instrument failure** — confirmed nodes were
  answer-position operand-fetch heads, not question-position `ST` compute nodes
  (Paper-2 ST candidates P8/P9/P11 failed single-node ablation confirmation);
  and the no-lower-carry stimulus pins `R²_pair=1`, so the ratio metric cannot
  discriminate A2 from transport. A2 remains **untested**. Positive control
  initially corrupted by a noise term in the transport arm (deflated its
  `R²_pair`); corrected to noise-free, after which the transport null reproduces
  the real signature (ratio 0.19 vs 0.20). Weights-only forward passes; no
  causal path-patching.
- **Linked study**:
  [study-maths/study-pair-sum-sufficiency.md](study-maths/study-pair-sum-sufficiency.md)

### 2026-07-14 — Confirm an ST/carry compute node (path-patching)

- **Covered**: Causal interchange-intervention (single-head `hook_z`, joint
  `z+mlp_out`, `resid` patches) to locate the carry-computing node, in
  `add_d5_l2_h3_t15K_s372001` and `add_d6_l2_h3_t20K_s173289`. Node-level
  positive control (SA head), local (0↔1) + genuine tri-state (`U`, toggle lower
  carry) batteries, per-cell null, both-direction symmetry, operand-attention
  gate. Full position×layer×head `A_{n+1}`-flip discovery sweep to find
  candidates.
- **Artifacts** (local, no HF): `results/study-confirm-st-node/`:
  `control_*.json`, `results.json`, `confirmed_st_nodes.json` (registry),
  `signature_heatmap_5digit.png`; script `scripts/confirm_st_node.py`.
- **Caveats / coverage gaps**: Confirms **binary make-carry (`SC`)** nodes, not
  tri-state `ST` (genuine `U` test flips 0.00). Tri-state U-resolution locus
  unfound. `=`-resid patch layer-0 only + autoregressively confounded — no
  architectural claim. Candidates discovery-selected on the same `A_{n+1}`
  metric (mitigated by orthogonal specificity + attention + null + direction).
  Addition only. Single-node patching can't exclude question-position nodes
  masked by downstream recompute.
- **Linked study**:
  [study-maths/study-confirm-st-node.md](study-maths/study-confirm-st-node.md)

### 2026-07-15 — Locate the tri-state U-resolution path (U-flip transmitters; combiner unresolved)

- **Covered**: Causal interchange sweep (single-node `hook_z` / `hook_mlp_out` /
  `resid`, plus joint head+MLP) with the genuine U counterfactual (fix
  `Dn+D'n=9`, toggle lower carry) across all positions/layers in
  `add_d5_l2_h3_t15K_s372001` and `add_d6_l2_h3_t20K_s173289`. Node-level
  positive control (make-carry head, binary carry) + readout control;
  same-carry null; both-direction symmetry; intended U-regime interaction gate.
- **Artifacts** (local, no HF): `results/study-u-resolution-path/`:
  `control_*.json`, `results.json`, `u_resolution_registry.json`
  (key `u_flip_transmitters`), `u_resolution_candidates.png`; script
  `scripts/u_resolution_path.py`.
- **Caveats / coverage gaps**: **Ambiguous (positive-but-unresolved)**. Located
  MLP-heavy L0/L1 nodes that causally *transmit* the U-flip (5-digit
  `P10.L0.MLP`, `P14.L1.MLP`; 6-digit `P11.L0.MLP`, `P16.L1.MLP`, head
  `P11.L0.H2`), distinct from the CE3 make-carry heads (U-flip 0.00). But the
  combiner-vs-conduit discriminator (A-1b) was **vacuous** (definite-digit
  `A_{n+1}` has no lower-carry dependence, so `def_flip=0` for every node incl.
  the trivial readout, gap 1.00) — so locus role is unconfirmed. Off-by-one in
  the readout control (patched token vs predicting position) caught & fixed
  pre-sweep. 2-layer, addition-only, single-digit U.
- **Linked study**:
  [study-maths/study-u-resolution-path.md](study-maths/study-u-resolution-path.md)

### 2026-07-15 — Combiner vs conduit at the U-flip transmitters (U-combiner = answer-position L1 MLP)

- **Covered**: Node-level activation-invariance discriminator (a combiner's
  output is invariant to a `carry_in` toggle for a *definite* digit but varies
  in the `U` regime) applied to the CE4 U-flip transmitters + CE3 make-carry
  heads, in `add_d5_l2_h3_t15K_s372001` and `add_d6_l2_h3_t20K_s173289`. Planted
  conduit + inert positive controls; carry_out-centroid verification.
- **Artifacts** (local, no HF): `results/study-combiner-vs-conduit/`:
  `control_*.json`, `results.json`, `combiner_registry.json`,
  `combiner_vs_conduit.png`; script `scripts/combiner_vs_conduit.py`.
- **Caveats / coverage gaps**: `def_diff≈0.004` alone is not self-sufficient —
  the norm (fires equally for definite/U) + carry_out-centroid evidence is what
  earns "combiner". Combiner (L1 MLP) replicates both models; L0-conduit half
  clean only in 6-digit (5-digit L0 borderline). 2-layer, addition-only,
  single-digit U. Whole-MLP-output metric (not neuron-level). A-3/A-4 secondaries
  not run.
- **Linked study**:
  [study-maths/study-combiner-vs-conduit.md](study-maths/study-combiner-vs-conduit.md)

### 2026-07-15 — ST tri-state geometry at the L1-MLP combiner input (A3 refuted at locus)

- **Covered**: Full-space geometry of the carry classes at the confirmed
  combiner *input* (`blocks.1.ln2.hook_normalized`), using a 4-class design
  (committed-0, committed-1, U→0, U→1) that breaks the fatal `U ≡ SA_n=9`
  confound via the resolution variable. Centroid geometry + per-class distances,
  off-axis fraction vs permutation null, U→0/U→1 resolution probe,
  committed-digit lower-carry control, `carry_in`-on-committed decodability
  (precursor-vs-decision), make-carry discriminator, planted scalar/simplex/
  square controls. Models: `add_d5_l2_h3_t15K_s372001` (combiner `P14.L1.MLP`,
  digit 2), `add_d6_l2_h3_t20K_s173289` (`P16.L1.MLP`, digit 3).
- **Artifacts** (local, no HF): `results/study-st-tristate-geometry/`:
  `results.json`, `shape_registry.json`, `combiner_input_pca.png`; script
  `scripts/st_tristate_geometry.py`.
- **Caveats / coverage gaps**: locus-scoped (one probe site, post-L1-attention —
  no L0 attribution). `carry_in` also ~98% decodable on committed digits, so the
  site holds *ingredients* — cannot show the U→{0,1} decision is upstream vs in
  the MLP. 5-digit off-axis (0.23) is in the pre-registered ambiguous band.
  make-carry resolution probe 0.79/0.80 (not chance); planted-simplex control
  only reached perm p 0.076. Single-digit U, 2-layer, addition only. (Gate-2
  fixed an initial evidence-integrity gap: headline distances are now computed
  by the script and in `results.json`.)
- **Linked study**:
  [study-maths/study-st-tristate-geometry.md](study-maths/study-st-tristate-geometry.md)

### 2026-07-16 — Earliest tri-state site sweep (no dedicated `{0,1,U}` symbol)

- **Covered**: 7 residual sites (embedding → L0-attn → L0-MLP → L1-attn →
  combiner) at the answer position, both models, using an **axis-decomposition**
  discriminator (is-U vs resolution vs committed-carry axes; Gate-1 mandated,
  replacing a confounded centroid-distance metric). Per-site is-U-perp variance
  share + permutation null, per-class distances, U-vs-committed separability +
  carry_in-partialling, info-absent floor; planted resolved/unresolved/
  ingredient/absent controls.
- **Artifacts** (local, no HF): `results/study-earliest-tristate-site/`:
  `results.json`, `site_trajectory_registry.json`, `isU_trajectory.png`; script
  `scripts/earliest_tristate_site.py`.
- **Caveats / coverage gaps**: NEGATIVE result, scoped — no dedicated off-axis
  `{0,1,U}` symbol at any answer-position site; U resolved to binary around
  L1-attention. Power floor ~0.5× committed sep (a ≤0.3× weak symbol could be
  missed). Answer positions only (question-position D'n transient U descoped).
  Single-digit U, 2-layer, addition only. Representational not causal.
- **Linked study**:
  [study-maths/study-earliest-tristate-site.md](study-maths/study-earliest-tristate-site.md)

### 2026-07-16 — LN-aware digit-embedding close-out (A-9)

- **Covered**: Re-measured the CE1 digit-embedding geometry (freq-1 share,
  angular-ordering permutation p, participation ratio) on the **LN-effective**
  embedding — position-free LN of `W_E` rows (normalize-only + full γ/β) and the
  model-true `blocks.0.ln1.hook_normalized` at digit positions — for the 4
  accurate models + untrained control. Raw arm recomputed in-script (reproduces
  CE1). Planted-circle-through-LN + untrained-LN + permutation-null-FPR controls.
- **Artifacts** (local, no HF): `results/study-ln-aware-embedding/results.json`;
  script `scripts/ln_aware_embedding.py`.
- **Caveats / coverage gaps**: γ near-constant (std ~0.005) → LN near-isometric,
  so "survives LN" is weak robustness. Ordering holds in 2 of 3 *independent
  seeds* (fails s173289). `disagree=True` flags are a classified-read
  (wraparound-ratio) boundary artifact, not geometry change. Weights-only,
  representational not causal, n=10.
- **Linked study**:
  [study-maths/study-ln-aware-embedding.md](study-maths/study-ln-aware-embedding.md)

### 2026-07-16 — Attention-pattern invariance census (A5 falsified / hybrid, 6-digit)

- **Covered**: Per-(head,layer,query-position) attention-target stability across
  400 random additions + a **value-matched target-move** falsifier (fix digit-`n`
  operands sum=9, toggle only lower carry; any argmax-key change = carry-state
  routing) with a same-state null, on `add_d5_l2_h3_t15K_s372001` and
  `add_d6_l2_h3_t20K_s173289`. Synthetic content-routed positive control;
  informativeness gate.
- **Artifacts** (local, no HF): `results/study-attention-invariance/results.json`,
  `attention_routing_registry.json`, `carry_routing_heatmap.png`; script
  `scripts/attention_invariance.py`.
- **Caveats / coverage gaps**: clean carry-state target-routing in the **6-digit
  model only** (L1H1 Q11 operand-read, L1H1 Q14 / L0H0 Q17 answer; strict-null
  ≤0.08, Bonferroni-safe); the 5-digit model is **inconclusive** (same-state null
  0.40–0.53, top-1-argmax near-useless). Representational (pattern-shift), not
  causal. Synthetic control validates the move-counter but not the
  null-discrimination. top-1-argmax is tie-sensitive (top-k-mass follow-up).
  2-layer, addition only.
- **Linked study**:
  [study-maths/study-attention-invariance.md](study-maths/study-attention-invariance.md)

### 2026-07-16 — Deep-cascade mechanism (deciding-digit patching; R-hybrid/ambiguous, instrument-limited)

- **Covered**: Causal test of how multi-digit `...999` carry chains resolve, on
  `add_d5_l2_h3_t15K_s372001` (chain top digit 3) and `add_d6_l2_h3_t20K_s173289`
  (top digit 4), over graded chain depths `C(n,k,class)` with matched hi/lo pairs.
  Batteries: A (spatial causal map — pure-state vs operand-content
  `resid_post(L0)` cells + consumer-side L1 z/MLP), D-2/D-7 (controlled
  intermediate all-9s digit patch at the transmitting `=` locus, with a
  deciding-digit positive control), D-1 (tail-joint local decomposition +
  position-invariance), C1/D-8 (genuine value-matched top-2 key-set deciding-digit
  tracking with a units-end control, ≥2-non-degenerate-depth bar), C2/D-9 (causal
  pattern-patch selection with deciding/wrong/irrelevant redirects + same-cell
  requirement). 7 positive controls (harness-liveness conduit, computed-state CE5
  combiner, SA-head bar+null, readout, SA pattern-redirect, consumer-L1 instrument,
  per-depth behavioral gate).
- **Artifacts** (local, no HF): `results/study-deep-cascade-mechanism/`:
  `results.json`, `control_*.json`, `flip_heatmap_*.png`; scripts
  `scripts/deep_cascade_mechanism.py`, `scripts/deep_cascade_batteries.py`.
- **Caveats / coverage gaps**: **R-hybrid/ambiguous — node/pattern granularity is
  underpowered to localize a single mechanism** (the finding *is* an instrument
  limit). A9's predicted single-cell convergence is **absent**: a causally
  deciding-selective consumer head (6-digit `L1.H0` Q14, dec 0.93/0.90 vs both
  baselines at k=2,3) and a deciding-digit-tracking head (`L1.H2` Q15, 2/3 depths)
  are **different** cells, and the CE8 routing cell `L1.H1` is causally **inert**
  (0.00). Sequential per-digit state disfavored where the `=` control passes
  (5-digit k=3) but untestable in 6-digit. Real graded operand-adjacent tail state
  (joint-flip 0). 5-digit fully inconclusive (C2 inert). Two Gate-2 BLOCK rounds
  corrected over-reach in both directions (unearned negative; then a non-implemented
  value-matched metric + boundary-artifact tracking + mismatched-cell A9 claim).
  Follow-up = edge path-patching (B11). 2-layer, addition, single seed per size.
- **Linked study**:
  [study-maths/study-deep-cascade-mechanism.md](study-maths/study-deep-cascade-mechanism.md)

### 2026-07-16 — Deep-cascade hand-off: L1-head→combiner edge path-patch (one-depth causal crumb)

- **Covered**: Edge path-patch of the `L1.head_h → L1-MLP combiner` edge at the
  answer-position combiner (the CE5 site), on `add_d6_l2_h3_t20K_s173289` (primary)
  and `add_d5_l2_h3_t15K_s372001`. Per-digit matched-pair chain patches at each
  affected digit's own consuming position; three arms (raw `resid_mid`, LN-fair
  [freeze `ln2` std], MLP-only [`ln2.hook_normalized`]); direct-path (`resid_post(L0)`)
  edge; **per-cell power control** (direct edge scaled *down* to the cell's own
  single-head-edge norm); selectivity (deciding vs same-class-diff-operand null vs
  wrong-digit). Controls: additivity (exact decomposition), full-`resid_mid`
  validity, behavioral gate.
- **Artifacts** (local, no HF): `results/study-cascade-handoff-edge-patch/`:
  `results.json`, `control_*.json`; script `scripts/cascade_handoff_edge_patch.py`.
- **Caveats / coverage gaps**: **Scoped partial positive.** Edge decomposition
  exact (additivity 3.3e-6); full-`resid_mid` patch flips 1.00. At **one depth**
  (6-digit k=3) a **single head `L1.H1`** (the CE8 routing cell CE9 found inert)
  carries the top cascade digit's **computed, deciding-selective** carry through the
  combiner MLP input (dec 1.00 / same-class null 0.00 / wrong 0.00). But the
  single-position edge battery is **underpowered at most cells** (6/9 6-digit, 3/9
  5-digit — a single-head-magnitude direct edge can't flip them), no head meets the
  ≥2-depth bar, and 5-digit has **live direct-path** cells (0.70–0.93). So A9's
  attention-delivery is **circumstantial (one-depth), not confirmed**; A6's
  residual-carry is **not refuted**. Two Gate-2 rounds: round 1 BLOCK (inverted
  power control scaled the direct path UP ~20×, over-claiming an anti-A6 positive);
  corrected → round 2 PASS WITH CONDITIONS. 2-layer, addition, single seed per size.
- **Linked study**:
  [study-maths/study-cascade-handoff-edge-patch.md](study-maths/study-cascade-handoff-edge-patch.md)

### 2026-07-16 — Cross-position & cross-subtask probe transfer (ST is position-specific + entangled with SV)

- **Covered**: Linear probes (logistic regression, balanced train+test) for the
  per-digit question-position sub-tasks SA/ST/SV on `add_d6_l2_h3_t20K_s173289` and
  `add_d5_l2_h3_t15K_s372001`. Read-site pre-flight over 5 candidate sites;
  cross-position transfer matrix (raw + mean-centered) at the shared question site
  `D'n`@resid_post(L0); same-position cross-subtask floor (class-mean-subspace
  decodability, a label-correlation control); principal angles between class-mean
  activation subspaces vs a label-correlation null; position-only decode null.
- **Artifacts** (local, no HF): `results/study-probe-transfer/results.json`,
  `transfer_*.png`; script `scripts/probe_transfer.py`.
- **Caveats / coverage gaps**: **ST-scoped result.** `ST` (the only sub-task with a
  strong diagonal at the question site) does **not** transfer across digit
  positions (retention 0.10/0.12 ≪ 0.6 bar; off-diagonal ≈ chance; mean-centering
  doesn't restore it) → **C2/A4 shared-template falsified for ST at question
  positions**; and `ST`–`SV` are geometrically **entangled** beyond their
  (independent) labels (angle 21° ≪ 64°/50° null, both models) → C2 orthogonality +
  A8 interference challenged. **SA and SV are diag-weak at the question site**
  (SA lives at the *answer* position, decodes 1.00 there — confirms CE2/CE3), so
  their transfer is not-assessable here (not "no template"). Position decodes at
  1.00 (a trivial positional-embedding fact, not the mechanism). Both models agree.
  Linear probes; 2-layer addition; middle digits; answer-phase transfer
  (tape-vs-register) untested → B4. Gate 2 PASS WITH CONDITIONS (narrowed the
  falsification from all-subtasks to ST; demoted the position-1.00 gloss).
- **Linked study**:
  [study-maths/study-probe-transfer.md](study-maths/study-probe-transfer.md)

### 2026-07-16 — Answer-position binding: tape vs register (SA register, SV present-at-= non-orthogonal)

- **Covered**: Linear probes for the resolved answer-phase states `SA_n` (answer
  digit) and `SV_n` (resolved carry into n) at `=` and each answer consuming
  position, on `add_d6_l2_h3_t20K_s173289` and `add_d5_l2_h3_t15K_s372001`.
  Coexistence census (decode at position vs an isolated-operand `Dn`/`D'n` baseline),
  answer-side cross-position transfer (raw + centered, vs operand floor), per-digit
  `SV` slot principal angles at `=` vs a label-correlation null. Reuses the
  probe-transfer harness.
- **Artifacts** (local, no HF): `results/study-answer-binding/results.json`,
  `coexistence_*.png`; script `scripts/answer_binding.py`.
- **Caveats / coverage gaps**: **Split layout, both models.** `SA` is a
  just-in-time **register** — absent at `=` (raw acc ≈ chance), present 1.00 only at
  its own answer position (A4 just-in-time supported). `SV` (carries) is
  **resolved/present at `=`** for all digits (CE7-consistent) but its per-digit
  slots are **not orthogonal** (6-digit 12–27°; 5-digit 2/3 pairs < 60°) → the
  orthogonal-**tape** alternative is **refuted**. Both `SA`/`SV` **transfer across
  answer positions** (a shared answer-side template — SV beats the operand floor;
  SA transfer is caveated re-derivation), unlike the position-specific question-side
  `ST` (CE11). Gate-2 corrections: the SV isolated-operand baseline is too weak to
  claim "beyond re-derivation" (SV depends on all lower digits) → reframed
  CE7-consistent, no "carry bus" claim; entanglement-below-null is 6-digit-only with
  a 1-D binary-subspace caveat; AB-2 CIs / AB-5 lowvar not implemented (point
  estimates). A6/C3 untouched (no storage/causal claim). Linear probes; middle
  digits; SV_0 excluded; 2-layer addition.
- **Linked study**:
  [study-maths/study-answer-binding.md](study-maths/study-answer-binding.md)
