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
  trail); **Low / provisional** for the weak circular-ordering part (raw-weights
  only; pending the LN-aware secondary analysis A-9).
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
  (agenda entries 2–3). Raw pre-LayerNorm weights only; the LN-aware secondary
  analysis (A-9) is outstanding and could strengthen or remove the ordering
  signal. Tiny 10-token vocabulary; n = 10 per matrix.
