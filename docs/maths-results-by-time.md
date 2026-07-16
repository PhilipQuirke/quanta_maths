# Maths Results By Time (maths-results-by-time.md)

Read role and rules: [Results Ledger](thor-document-rules.md#results-ledger).

Append-only chronological record of completed result bundles for the `maths`
thread. Preserve order. Record what each bundle covered and where its artifacts
live — not the current best story (that goes in
[maths-results-summary.md](maths-results-summary.md)).

## Bundles

### 2026-07-14 — Addition model: Digit-embedding geometry audit

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

### 2026-07-14 — Addition model: Pair-sum sufficiency at operand-fetch heads (A2 assay; instrument failure)

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

### 2026-07-14 — Addition model: Confirm an ST/carry compute node (path-patching)

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

### 2026-07-15 — Addition model: Locate the tri-state U-resolution path (U-flip transmitters; combiner unresolved)

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

### 2026-07-15 — Addition model: Combiner vs conduit at the U-flip transmitters (U-combiner = answer-position L1 MLP)

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

### 2026-07-15 — Addition model: ST tri-state geometry at the L1-MLP combiner input (A3 refuted at locus)

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

### 2026-07-16 — Addition model: Earliest tri-state site sweep (no dedicated `{0,1,U}` symbol)

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

### 2026-07-16 — Addition model: LN-aware digit-embedding close-out (A-9)

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

### 2026-07-16 — Addition model: Attention-pattern invariance census (A5 falsified / hybrid, 6-digit)

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

### 2026-07-16 — Addition model: Deep-cascade mechanism (deciding-digit patching; R-hybrid/ambiguous, instrument-limited)

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

### 2026-07-16 — Addition model: Deep-cascade hand-off: L1-head→combiner edge path-patch (one-depth causal crumb)

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

### 2026-07-16 — Addition model: Cross-position & cross-subtask probe transfer (ST is position-specific + entangled with SV)

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

### 2026-07-16 — Addition model: Answer-position binding: tape vs register (SA register, SV present-at-= non-orthogonal)

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

### 2026-07-16 — Addition model: Node output encoding (C5 step 1): map-named ST/SA/SC write characterization

- **Covered**: For the HF-map-named `ST`/`SA`/`SC` attention heads (both studied
  models), the node's **write** (head output / OV-projected residual) as a function
  of its sub-task value. Battery E (full-space linear-probe encoding + permutation
  null + cross-digit baseline + **same-position wrong-role baseline** [N-4]);
  Battery C (cascade locality — fixed `(Dn,D'n)`, toggle `cin`, on the OV write,
  U-pair + definite-pair, cin/null ratio [N-9] + positive control); Battery P
  (matched-pair `tristate_test` interchange + CE3 SA-head control); Battery Ab
  (mean-ablation impact vs an **untagged-head baseline** [N-8]). Node lists read from
  HF `features.json`. Outside-view sweep done (decodability≠causality; causal
  scrubbing).
- **Artifacts** (local, no HF): `results/study-node-output-encoding/results.json`;
  script `scripts/node_output_encoding.py`.
- **Caveats / coverage gaps**: **Both models agree directionally.** (1) ST nodes
  **encode** their 3-way class (~1.00); N-4 baseline flags 3 nodes as
  position-decodable-not-uniquely-head-written. (2) The ST write **co-carries
  single-step local U-resolution** (cin-dependent on `U`, cin/null 0.75–19.4,
  low-digit-concentrated, 5-digit-scoped — 6-digit Battery-C control structurally
  void) — **not** shown to be multi-digit compounding; A10 premise **refined**, not
  supported/compound-confirmed. (3) **CE3 refined to redundancy, baseline-controlled**:
  single-node interchange = 0.00 (reproduced; SA control 0.88–0.93), but mean-ablation
  impact exceeds the untagged-head baseline (max 0.000–0.003) for **low-digit** ST
  nodes (map causally right) and is at baseline for high-digit (redundant). (4)
  Map-named `SA` L0 heads do **not** encode the answer digit except the leading digit
  (sum computed at answer position, CE11/CE12). Two Gate-2 rounds corrected over-reach
  in both directions (first "local write", then "compounding begins here" → settled
  "single-step U-resolution co-located"). A2 not tested. Linear probes; 2-layer
  addition.
- **Linked study**:
  [study-maths/study-node-output-encoding.md](study-maths/study-node-output-encoding.md)

### 2026-07-16 — Addition model: SV compounding at the map-named wires (C5 steps 2-4; A10 partial: carry-specific distributed delivery)

- **Covered**: Causal test of A10 at the map-named answer-position L1 consumer heads
  (feeding the high-Fail% L1-MLP combiners), both studied models. Battery V (value
  content via `W_V` + same-position wrong-role baseline); Battery D (deciding-digit
  edge hand-off through the head→MLP edge via a less-damped `ln2.hook_normalized`
  MLP-only instrument, single/joint-pair/multi-position arms, ≥2 depths, a
  **deciding-matched specificity null**, a non-consumer-head specificity arm, and a
  scaled direct-path arm); Battery S (value-matched deciding-digit tracking with
  units-end control); Battery E (mean-ablation economy cascade-vs-carry-free vs an
  untagged-head baseline). Consumer heads + combiners from HF `behaviors.json`.
- **Artifacts** (local, no HF): `results/study-sv-compounding/results.json`; script
  `scripts/sv_compounding.py`.
- **Caveats / coverage gaps**: **R-A10-distributed-delivery, both models — A10 core
  PARTIALLY confirmed.** Carry-specific (deciding-matched null = 0.00) head-edge
  delivery to the combiner is causally **sufficient at ≥ 2 depths** via a
  **redundant H1/H2 pair**, and consumer-head-specific (non-consumer head inert).
  BUT single-cell **selection is NOT shown** (value-matched tracking is on `L1.H2`,
  single-depth edge causality on `L1.H1` — different heads); the effect is
  **sufficiency, not necessity** (ablating the sufficient head H1 does nothing; H2
  carries necessity+tracking); the **direct path is not excluded** (its arm is
  underpowered, `direct_scaled`=0); and value content is **not head-specific**
  (Battery V wrong-role baseline equal). A9 (single-head selection) **not
  supported**; A6 selective economy **supported at H2**. THREE Gate-2 rounds:
  positive over-claim → negative over-correction on a broken null (toggled the
  deciding carry) → calibrated middle after fixing the null to deciding-matched.
  2-layer addition, two models; no neuron decomposition.
- **Linked study**:
  [study-maths/study-sv-compounding.md](study-maths/study-sv-compounding.md)

### 2026-07-16 — Addition model: Leading-digit hard-case walkthrough (C5 step 5; mirrors CE14, A10 consolidated not raised)

- **Covered**: Per-link causal trace of how the LEADING answer digit `A_top` is
  produced at the sign-token position in a hard graded-cascade case (`99..9+00..01`,
  chain of k nines reaching the leading digit, deciding `+1` just below), both
  studied models. Link 4 (whole-`resid_mid` readout patch, tagged readout-only, NOT
  counted toward A10); Link 3 (sign-position L1 head-edge delivery via the CE14
  less-damped MLP-only instrument + deciding-matched null + scaled direct-path arm,
  graded depths); Link 1 (sign-position L0 ST encoding/ablation vs baseline); economy
  (A6) + untagged baseline; static-output control; per-depth behavioral gate.
- **Artifacts** (local, no HF): `results/study-leading-digit-walkthrough/results.json`;
  script `scripts/leading_digit_walkthrough.py`.
- **Caveats / coverage gaps**: **C5 step 5 done — mirrors_CE14, both models.** The
  leading digit is produced by **carry-specific L1-head-edge delivery** to the
  sign-position combiner (Link 3: real flip 1.00, deciding-matched null 0.00) —
  5-digit across a **genuine depth spread (k=1–4)**, 6-digit at **deep chains only
  (k=4,5)** with shallow leading cascades (k=1–3) carried by an **unadjudicated**
  path (direct arm underpowered). Link 4 readout-only (quarantined). Economy (A6)
  **not testable** at the sign position (untagged baseline ≈ tagged, 0.49 — general
  bottleneck / ablation too destructive). Link 1 ST ablation verified 5-digit
  (`P11L0H2` 0.045), redundant 6-digit (matching CE13). Direct path not excluded.
  **A10 CONSOLIDATED across all answer digits incl. the hardest, NOT raised** (held
  at medium; inherits all CE14 caveats + the 6-digit-deep-only rider). Gate 2 nearly
  clean (Link 4 quarantine + consolidate-not-raise confirmed sound). 2-layer
  addition, two models.
- **Linked study**:
  [study-maths/study-leading-digit-walkthrough.md](study-maths/study-leading-digit-walkthrough.md)

### 2026-07-16 — Addition model: SV implementation sprint (CE16; A10 items i–iii resolved, iv open; dual-gated)

- **Covered**: Parameter-estimation of the four A10 implementation details on the
  confirmed SV wiring (working-axioms mode), both studied models. Battery **M**
  (edge message identity via within-chain cross-deciding-position carry-probe
  transfer + scale-normalized residual-family decode + committed-family descriptive
  arm); Battery **R** (source attribution via SI-8 per-key CONTRIBUTION
  decomposition — source pattern AND v per key group — `=` vs deciding-ST vs rest,
  with deciding-matched null per arm, leave-one-out complements + arm-sum residual);
  Battery **P** (SI-1 powered skip arm: inject the MEASURED real-skip carry
  direction at 1×/2× into `resid_post(L0)`; head-pair real arm; class necessity via
  joint H1+H2 mean-ablation vs carry-free with untagged baseline); Battery **F**
  (stretch, combiner α-sweep — INVALID). Controls PC1 (CE14 regression) + PC4
  (carry-axis anchor).
- **Result**: **(i) canonical format-invariant resolved carry** (transfer 1.00 =
  within-acc) + non-causal position co-rider; **(ii) source = distributed
  question-tail ST cluster, NEVER `=`** (`=` value arm 0.00, OV-proj ≈ 0 → depot;
  deciding-ST carry-specific, dominant at 6d k3; chain-ST elsewhere); **(iii)
  head-pair (SV) path effective (flip 1.00), skip carries negligible carry**
  (0.14/0.077, power 1×/2× = 0.00 → not used, not formally excluded), **pair
  class-necessary** (necessity-over-baseline 1.07 6d / 0.85 5d, selective); **(iv)
  combiner form OPEN** (F instrument invalid, spread 0.00). A9 stays retired
  (selection level); A6 raised to class level.
- **Artifacts** (local, no HF): `results/study-sv-implementation/results.json`;
  script `scripts/sv_implementation.py` (reuses the CE14 harness).
- **Caveats / coverage gaps**: skip **not formally excluded** (power matched to the
  skip's own tiny magnitude — F1); source **distributed** not localized to
  deciding-ST (dominant at 1 depth — F2); position co-rider is **decode-existence**,
  non-causal (F3); 5d k3 R leaky (ordinal only); combiner form unestimated (F
  failed). Dual-gated: pre-launch PASS-WITH-CONDITIONS (SI-1..SI-10) + post-result
  PASS-WITH-CORRECTIONS (F1/F2/F3). 2-layer addition, two models.
- **Linked study**:
  [study-maths/study-sv-implementation.md](study-maths/study-sv-implementation.md)

### 2026-07-16 — Addition model: SV compounding arithmetic (CE17; A10 iv resolved = STEP; A11 low; R-mixed; dual-gated)

- **Covered**: The compounding locus (A11 positional-L0-relay vs L1-read) + the
  last A10 detail (item iv, combiner transfer), both models. Battery **H**
  (horizon decode: does each chain-ST L0 write decode the resolved carry up to its
  position-visibility horizon m? boundary cells at n_top=4: P10 m=2, P11 m=1);
  Battery **Y** (relay causality: twin-interchange of tail-ST OV writes,
  deepest-sufficient/insufficient/joint arms, L1 re-attends; CE13-ablation instrument
  control); Battery **L** (L1-read reconstruction: nested φ_local ⊂ φ_horizon,
  ΔR² over permutation null); Battery **T** (combiner transfer on-manifold, PRE-LN
  interpolation of real c0/c1 edge captures, endpoint-gated to CE16 0/1).
- **Result**: **A10 iv RESOLVED — combiner is a STEP** (α*≈0.75 both models; flip
  0→0.42→1 across α .25/.5/.75; carry-proj linear). **A11 = single-site
  representational trace** (6d P11H2 bacc 1.00 at the horizon-visible depth) that
  **fails at k=3 and does not replicate in 5d**; relay **causally undetermined**
  (Y interchange 0.00 everywhere, valid ablation instrument but different
  unit/target — redundancy vs weak-interchange unresolved); L1 edge output
  **local-class-sufficient** (r²≈0.95–0.99, horizon adds no held-out gain). **A11
  → low; A9 stays retired; R-mixed, leaning L1-local.**
- **Artifacts** (local, no HF): `results/study-compounding-arithmetic/results.json`;
  script `scripts/compounding_arithmetic.py` (reuses CE13 `_mean_ablate_acc`/ST_NODES
  + CE16 patch/axis machinery).
- **Caveats / coverage gaps**: A11 trace one strong-writing site per model (deeper
  sites write ~0); Y null causally undetermined (interchange vs ablation-target
  mismatch — CE16-F1 trap, flagged); L local-sufficiency may reflect
  local↔resolved-carry correlation + a reconstruction ceiling; T interpolation
  un-regressed (co-rider caveat; step rests on the endpoint gate + linear
  carry-proj). Dual-gated: pre-launch PASS-WITH-CONDITIONS (CA-1..CA-6, incl.
  n_top=4 re-pin for real horizon crossings) + post-result PASS-WITH-CORRECTIONS
  (F1 H-not-replicated, F2 Y-causally-undetermined). 2-layer addition, two models.
- **Linked study**:
  [study-maths/study-compounding-arithmetic.md](study-maths/study-compounding-arithmetic.md)

### 2026-07-16 — Addition model: Cross-size SV validation & tightness census (CE18; d5/d6/d10/d13; A12 role+combiner transfer, C6 not supported; dual-gated)

- **Covered**: Does the SV interface generalize to larger models (A12) and does its
  redundancy thin with n (C6)? Ran d5, d6, **d10, d13** (all acc 1.000), registries
  built from the published maths.json/behavior.json (XS-B). Battery **C** (redundancy
  census / tightness index: map duplicate multiplicity (i), interchange decisiveness
  (ii, corroborating-only/F2), class-vs-single ablation gap (iii, PRIMARY, F2-free));
  Battery **I** (interface transfer: =-not-a-source per-key contribution patch, class
  necessity, step combiner endpoint-gated); Battery **L** (large-n locus, stretch —
  UNTRIGGERED). Empirical consumer-head ID at d10/d13 (map-untagged there).
- **Result**: **role skeleton + STEP combiner GENERALIZE** (ST writers + combiner
  MLPs present at d10/d13; consumer head causal at every size; combiner step,
  endpoint-gated, α*≈0.5–0.75 all four). **Causal source signatures NOT reproduced
  at large n** (carry axis sep ~6 vs ~30; =-arm and deciding-ST both 0.00 —
  probe-limited, source-fork untested at n≥10). **C6 NOT SUPPORTED**: single-node
  ST ablation ~0 at every size; class-minus-single gap {d5:0.056, d6:0.116,
  d10:0.324} does NOT shrink (d13 inconclusive, class ablation 0.040 ≈ CE13
  single-node magnitude) → redundancy intrinsic, not small-model slack. **A11 not
  rescued by scale** (Battery L untriggered). **A12 role+combiner → medium-high;
  C6/tightening → low; A10 iv step generalizes; A11 unchanged.**
- **Artifacts** (local; reads published maps, no HF uploads):
  `results/study-cross-size-sv/results.json`; script `scripts/cross_size_sv.py`
  (reuses CE13/CE16/CE17 modules).
- **Caveats / coverage gaps**: large-n carry axis weak (sep ~6) → source arm-probes
  probe-limited; d13 ST ablation instrument-weak (0.04, "redundant" vs "unmeasured"
  unseparated); ablation gaps un-intervalled; consumer-ID single-head (not pair) at
  large n; 4 sizes, no d7/d8/d9 gradient (no monotone claim, XS-F); interchange leg
  F2-ambiguous. Dual-gated: pre-launch PASS-WITH-CONDITIONS (XS-A…XS-F) + post-result
  PASS-WITH-CORRECTIONS (F1 source-probe-limited, F2 C6-not-supported/d13-inconclusive,
  F3/F5). 2-layer/3-head addition zoo, one large-n seed family.
- **Linked study**:
  [study-maths/study-cross-size-sv.md](study-maths/study-cross-size-sv.md)

### 2026-07-16 — Addition model: Compounding locus (CE19; A11 not supported = L1-read not L0-relay; dual-gated)

- **Covered**: the last open SV question — is multi-digit carry compounding an L0
  positional relay across the question tail (A11) or an L1-read computation? CE17
  was R-mixed. This study used a **decorrelation lever** (a chain-ST site inside the
  999-run has local class fixed at U while the resolved carry varies) and split
  cells into VISIBLE-decorrelated (site can see the deciding digit → self-computable)
  vs **INVISIBLE-decorrelated** (deciding digit below the site's horizon → a carry
  there could ONLY be relayed — the discriminator). Battery **DH** (decode resolved
  carry from the L0 write, with decorrelation + correlated-depth + per-depth
  readability controls), **KO** (class-level cumulative knock-out + specificity
  null), **RC** (reconstruction, corroborating).
- **Result**: **A11 NOT SUPPORTED — the compounding is an L1-read, not an L0
  relay.** At invisible-decorrelated cells the resolved carry decodes at **chance**
  in both models; where the write is readable at that depth (5d P9H1 k3) that chance
  = a genuine "no relayed carry" → R-L1-read; at 6d the deep-chain writes wash out
  (underpowered). The ST write decodes carry only at the fully-correlated cell
  (local single-step, CE13). KO shows the ST class is carry-necessary (all-ablate
  breaks differentially over 0.00 null/baseline) but is a necessity anchor only
  (DH load-bearing). **A11 → low (not rejected); A9 retired; C3 sequential-lean not
  supported at L0.** Caveats: linear probe; one readable invisible cell; 6d
  underpowered.
- **Artifacts** (local, no HF): `results/study-compounding-locus/results.json`;
  script `scripts/compounding_locus.py` (reuses CE13/CE16/CE17 modules).
- **Caveats / coverage gaps**: DH is a linear decode (non-linear relay not
  excluded); the refutation rests on one readable invisible cell (5d); 6d
  underpowered (deep-chain ST writes wash out — F2 per-depth readability control);
  KO does not discriminate relay vs local resolution. Dual-gated: pre-launch
  BLOCK→PASS (LOC-1…LOC-6: invisible-decorrelated discriminator + KO specificity
  null) + post-result PASS-WITH-CORRECTIONS (F1 "refuted"→"not supported", F2
  per-depth readability→6d underpowered, F3/F4). 2-layer/3-head addition, two models.
- **Linked study**:
  [study-maths/study-compounding-locus.md](study-maths/study-compounding-locus.md)

### 2026-07-16 — Mixed-model SV replication (entry 2, parallel thread): CE20 + CE21

- **Bundle**: the addition SV interface replicated on the accurate mixed add/sub
  model `ins1_mix_d6_l3_h4_t40K_s372001` (3 layers, 4 heads) across all three
  question classes ADD / SUB / NEG, plus the mixed-only OPR / SGN / shared-engine
  experiments. Per-class accuracy 1.000 (positive control).
- **Result (CE20)**: the SV **representation** replicates — writers encode the
  tri-state (ST/MT/NT ~1.00 vs untrained ~chance); the resolved carry/borrow is a
  clean **binary** code at the last-layer (L2) combiner input (SV/MV/NV ~1.00).
  **Delivery is class-dependent**: carry/borrow-specific (deciding-matched null
  0.00) and residual-borne for all classes, with **last-layer attention
  additionally delivering it for SUB/NEG but not ADD** (the inserted addition
  circuit resolves earlier and rides the residual). Scores C5 (confirmed on a new
  architecture), A10 (confirmed at representation, delivery refined), A12
  (confirmed onto 3 layers + borrow/neg tasks).
- **Result (CE21)**: **SGN** is the top-of-cascade `D≥D'` comparison delivered to
  the `=` combiner (CE15 analog — boundary flip 1.00; binary-decodable 1.00 at
  `=`; sign edge flip 1.00 / deciding null 0.00). **OPR** is broadcast-decodable
  everywhere (1.00) but not an additive rank-1 control at the combiner (consumed
  upstream at the SLT selector). **Shared SA/MD/ND heads** show a hybrid A7/C2
  geometry: shared heads, but ADD-vs-subtraction readouts more separated than a
  random-init control (68–71° vs 46–50°) while SUB-vs-NEG overlap (48°≈46°).
- **Library**: added `neg_labels`, `neg_ntc_functions` (NTC), `NT`/`NTC` tags,
  class-aware `_combiner_is_causal`; `tests/test_scaling_and_sub.py` 21 passed
  (incl. HF mixed-model NEG).
- **Artifacts** (local, no HF): `results/study-mixed-map/results.json`,
  `results/study-mixed-sv/results.json`, `results/study-mixed-opr-sgn/results.json`;
  scripts `scripts/mixed_map.py`, `scripts/mixed_sv.py`, `scripts/mixed_opr_sgn.py`.
- **Caveats**: single mixed model/seed; single-step U delivery (k=2, no depth
  sweep); zero-ablation combiner redundancy-limited; M4 rank-1 additive steer only
  (SLT-sited steer post-deadline); linear probes. Combined sprint gate; two errors
  caught & corrected before scoring (M4 decode label-alignment bug; full_resid/
  zero-ablation not read as trained-structure evidence).
- **Linked studies**:
  [study-maths/study-mixed-sv-replication.md](study-maths/study-mixed-sv-replication.md),
  [study-maths/study-mixed-opr-sgn.md](study-maths/study-mixed-opr-sgn.md),
  [study-maths/study-mixed-plan.md](study-maths/study-mixed-plan.md)

### 2026-07-16 — Mixed model: SV-implementation batteries + SLT-sited A7 test (entry 2a/2b): CE22 + CE23

- **Bundle**: completes entry 2's "Done when" — the SV *mechanism* batteries
  (CE16/CE17) per class and the decisive A7-vs-C2 shared-engine test, on
  `ins1_mix_d6_l3_h4_t40K_s372001`.
- **Result (CE22, 2a)**: across ADD/SUB/NEG the combiner is a **STEP** (α-sweep
  α*≈0.5, endpoints gated — CE17/A10 iv generalizes), the resolved carry/borrow is
  a **canonical format-invariant** code (cross-digit probe transfer 1.00 — CE16 i),
  and **`=` is not the middle-digit source** (`=`-patch flip 0 / combiner control
  1 — CE16 ii). Class-necessity (A6) **not scored** — redundancy-blurred
  (writer-class ablation == truly-untagged baseline for ADD/SUB; NEG confounded by
  cascade-stimulus triviality).
- **Result (CE23, 2b)**: the L1 selector-stage residual is decisive (full patch
  flips to the correct ADD digit **0.96**) so the **L2 combiner is shared**, but a
  **rank-1 operator steer flips 0.00** at the selector (as at the combiner, CE21)
  and the **single SLT head never selects** (0.00). **A7's low-rank/function-vector
  control form is refuted**; the add/sub selection is a **distributed,
  high-dimensional L1 transformation** — leans **C2** on selection, shared combiner.
- **Scoring**: A10 iv/i/ii confirmed on mixed; A12 strengthened (implementation,
  not just representation, transfers); A6 untouched (redundancy); A7 → low on the
  control mechanism (hybrid); C2 → partially up on selection.
- **Artifacts** (local, no HF): `results/study-mixed-sv-impl/results.json`,
  `results/study-mixed-shared-engine/results.json`; scripts
  `scripts/mixed_sv_impl.py`, `scripts/mixed_shared_engine.py`.
- **Caveats**: single mixed model/seed; k=2; B1 5-point α grid; necessity
  unresolved (redundancy + stimulus confound); rank-1 additive steer only (a
  learned rank-r operator subspace untested — the residual A7 escape hatch).
  Combined sprint gate; a first-run necessity baseline was invalid → refixed.
- **Linked studies**:
  [study-maths/study-mixed-sv-implementation.md](study-maths/study-mixed-sv-implementation.md),
  [study-maths/study-mixed-shared-engine.md](study-maths/study-mixed-shared-engine.md)

### 2026-07-16 — Compounding locus v2 (CE24; addition; A11 conclusively = L1-read; dual-gated)

- **Covered**: the conclusive successor to CE19 (which was inconclusive). Reframes
  the compounding-locus question to LAYER-localization — "at which layer does the
  resolved carry become a canonical (position/depth-invariant) abstract bit?" —
  fixing CE19's OV-write washout (reads the full residual) and its ill-posed
  invisible-cell discriminator (MSD-first layout + causal mask forbid an L0
  carry-direction relay). Battery **TR** (cross-depth transfer of the carry decoder:
  L0 output @ `=` gather vs the L1 combiner input on CE16's answer-agnostic carry
  axis; nuisance-transfer controls; Procrustes backstop), **9-free** stimuli (human's
  `66666+33334`/`33433` insight) + a 9-containing equivalence arm; LC causal
  (corroborating; came out invalid).
- **Result**: **canonical carry is an L1 property** — cross-depth transfer **1.00**
  at the L1 combiner input (answer-top does NOT ride the carry axis: 0.12/0.13 vs
  0.78 full-residual) while **L0's output has no carry-specific canonical code**
  (extreme-pair transfer ~chance with CIs; high within-depth ceiling; weak transfer
  = magnitude nuisance). 9-free ≡ 9-containing. Combined with CE16's causal
  head-pair delivery → **the L1 read is where the canonical carry emerges**,
  conclusive both models. **A11 rejected/low; compounding locus = L1 read.**
- **Artifacts**: `results/study-compounding-locus-v2/results.json`;
  `scripts/compounding_locus_v2.py` (reuses CE16 carry axis + edge).
- **Caveats**: linear probe (rotated-frame L0 carry not excluded — Procrustes
  overfit); L0 tested at `=` (ST/sign via CE17/CE19); own LC causal battery invalid
  (causal locus per CE16); nuisances point estimates; TF/LP not run. Dual-gated
  (CLV2-1..6 pre-launch; F1–F7 post-result). 2-layer/3-head addition, two models.
- **Linked study**:
  [study-maths/study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md)

### 2026-07-16 — Compounding locus TF: token-time finalization (CE24 addendum; addition; LAZY propagation + eager local; dual-gated)

- **Covered**: closes the token-time question left open by CE24 (which settled the
  LAYER = L1 read). "At which TOKEN POSITION does each carry SVn finalize — eager
  in-place at its make-carry token, or lazy at the answer region?" Battery TF:
  9-free graded chains (66666+33334/33433 insight) with a **run-break** (so
  `carry_out(top)=make-carry AND run-intact`, decorrelated from local make-carry);
  **cross-deciding-position transfer** at each (token position × layer) for the
  propagated carry; within-position decode for the local make-carry (CE13 ref).
- **Result** (both models): **LAZY (representational) propagation + eager local.**
  Propagated canonical carry appears **only at the sign token, L1** (transfer→1.0),
  ~chance at every operand token and at `=`, never at L0 — a **~2-token deferral
  past full input availability (D'_0)** and past the `=` gather (chance;
  `=`-is-a-depot, CE16). Local single-step make-carry **eager in-place at L0 at its
  own token** (0.76–0.94, CE13). Correction (post-result gate): the
  make-carry-token→D'_0 span is information-availability (MSD-first layout, CE19),
  not laziness. Net: single-step eager in-place (L0); multi-digit propagation lazy
  at the answer read (L1).
- **Artifacts / reusable cross-model code**: **`quanta_maths/maths_temporal_finalization.py`**
  (`run_temporal_finalization(model_name)`, layer-general, package-native;
  unit-tested `tests/test_temporal_finalization.py`); thin CLI
  `scripts/compounding_locus_tf.py`; `results/study-compounding-locus-tf/results.json`.
  Built to run across the model zoo (answers may differ by model).
- **Caveats**: linear-probe; point estimates (CIs via CE24); coarse token tail;
  onset = sign token (leading answer digit is constant 0 there, so the onset is a
  genuine prospective carry). Dual-gated (CLV2-5 pre-launch; post-result F1–F6).
  2-layer/3-head addition, two models.
- **Linked study**:
  [study-maths/study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md#addendum--battery-tf-temporal-finalization--token-time--run-2026-07-16)

### 2026-07-16 — Mixed model: ≥2-depth carry/borrow delivery sweep (entry 2(i)): CE25

- **Bundle**: extends CE20's single-step delivery to cascade depths 2/3/4 on
  `ins1_mix_d6_l3_h4_t40K_s372001`, and promotes the reusable sweep to the library.
- **Result (CE25)**: the class-dependent delivery pathway **holds at every depth**
  (all deciding-matched nulls 0.00): **ADD** delivers via the residual only
  (`lastlayer_attn` flip 0.00, `resid_pre`/`full_resid` 1.00); **SUB/NEG** deliver
  via residual + **last-layer attention** (both arms 1.00). Untrained control:
  `lastlayer_attn` flip 0.00 (no learned delivery). Clears the CE14 ≥2-depth bar
  on the mixed model. Scores A10 (delivery at depth), A12.
- **Reusable code**: promoted `quanta_maths/maths_cascade.py`
  (`make_cascade_operands`, `combiner_delivery_flip`, `combiner_delivery_sweep`) so
  the sweep runs across the zoo (delivery may differ by model); `tests/test_cascade.py`
  (5 offline + 3 HF); full suite 73 passed. Exported from `quanta_maths`.
- **Artifacts** (local, no HF): `results/study-mixed-delivery-depth/results.json`;
  `scripts/mixed_delivery_depth.py`.
- **Caveats**: single mixed model/seed; depths 2–4 (n=6 leaves no class-control top
  above depth 5); deciding digit fixed at units; whole-last-layer-attention patch
  (not per-head); `full_resid`/`resid_pre` near-tautological (specificity from the
  null). Combined sprint gate; a tautological negative control was replaced.
- **Linked study**:
  [study-maths/study-mixed-delivery-depth.md](study-maths/study-mixed-delivery-depth.md)
