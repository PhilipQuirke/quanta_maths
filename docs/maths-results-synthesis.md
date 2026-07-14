# Maths Results Synthesis (maths-results-synthesis.md)

Read role and rules: [Results Synthesis](thor-document-rules.md#results-synthesis).

Mid-level empirical synthesis for the `maths` thread's primary results themes.
Organize by empirical question, not by experiment chronology. Keep only figures
that change interpretation or priority. This is the normal home for detailed
mechanism interpretation that would overload
[maths-results-summary.md](maths-results-summary.md).

## Q: How are digit values represented at the token level (embedding / unembedding)?

Status after the 2026-07-14 digit-embedding geometry audit (weights-only; see
[claim CE1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix)
and the [study note](study-maths/study-digit-embedding-geometry.md)).

- The digit-token embedding is **near-isotropic in 9 dimensions** (variance
  spread ≈ 0.15→0.08 across components, participation ratio ≈ 8.7/9) for both
  trained and untrained models. There is **no dominant low-rank circle or
  helix**: the frequency-1 variance share (~0.27) barely exceeds the untrained
  baseline (0.23) and, by the positive control's own calibration (planted 0.40 →
  recovered 0.52), sits at the noise floor.
- The one **training-induced** effect is a weak **circular ordering** of digit
  values in the top-2 PC plane: permutation p = 0.0001 in 3 of 4 accurate models,
  absent in the untrained control (p = 0.34). This is an ordering tendency, not
  variance concentration, and is **provisional** pending the LN-aware secondary
  analysis (A-9).
- The **unembedding** carries no comparable structure (R4 everywhere) and does
  not align with the embedding geometry (principal angles 36–88°).
- **Decision-relevant read**: a clean geometric digit code was only ever one
  route to numeric computation, and it is largely absent here at the token level.
  This makes the **activation- and MLP-level** story (aggregate-then-discretize;
  tri-state `ST` geometry) and a **causal** test of whether the model uses digit
  magnitude/circularity the more central questions — not less. See
  [maths-next-steps.md](maths-next-steps.md).

Figure: `results/study-digit-embedding-geometry/embed_pc_planes.png`
(top-2 PC plane per model, digit-labeled) is the decision-relevant plot; the
near-flat `embed_variance_spectra.png` is the evidence for isotropy.

## Q: Where is the ST carry computed, and does attention aggregate or transport?

Status after the 2026-07-14 pair-sum study (an `instrument failure` for its
primary question; see [CE2](maths-claim-evidence.md#ce2-at-layer-0-operand-fetch-heads-the-value-path-output-is-indistinguishable-from-linear-transport-of-the-weakly-circular-digit-embeddings)
and the [study note](study-maths/study-pair-sum-sufficiency.md)).

- The A2 "aggregate-then-discretize" question is **still open**. The first assay
  mis-targeted (it confirmed answer-position operand-fetch heads, not
  question-position `ST` compute nodes) and used a metric that the no-lower-carry
  stimulus rendered non-discriminating.
- Durable by-product: at those operand-fetch heads, the value-path output is
  **indistinguishable from linear transport** of the weakly-circular digit
  embeddings — the same "weak low-variance projection" pattern seen for the
  embedding itself (CE1). The monotone sum arc there is the arithmetic identity
  `cos a + cos b = 2cos((a−b)/2)cos((a+b)/2)`, not evidence of an aggregation
  computation.
- **Method lesson (decision-relevant)**: full-space geometry keeps showing that
  the tidy low-D structures (digit circle, ST tri-cluster, sum arc) are
  *low-variance projections*; verdicts must compare against explicit nulls
  (permutation, transport) and confirm the node's role independently, not read
  a projection at face value.

## Open empirical questions

- Where is `ST`/carry actually computed, and does attention aggregate operands
  or merely transport them? (Needs a **confirmed question-position ST node** via
  path-patching to the `SV` cascade; the answer-position assay could not tell.)
- Does the model *use* digit magnitude / circular ordering in its computation,
  regardless of a weak embedding geometry? (Causal; backlog B1.)
- Is the tri-state `ST` code well-separated at a *confirmed* ST node in the full
  space, or is it (like the digit circle) a low-variance projection? (Entry 2,
  re-scoped to a confirmed ST node.)
- Does the weak embedding circular ordering survive LayerNorm folding? (A-9.)
