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
