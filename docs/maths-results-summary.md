# Maths Results Summary (maths-results-summary.md)

Read role and rules: [Results Summary](thor-document-rules.md#results-summary).

Compact current-state synthesis for the `maths` thread. Orients a human or agent
quickly. Update only when a result changes the current story. Link downward for
detail; do not turn this into a ledger.

## Executive summary

One study complete. Trained addition models do **not** store digits as a clean
low-dimensional circle or helix at the token level: the digit embedding is
near-isotropic in 9 dimensions, with a circular signal at the noise floor and no
structure in the unembedding. The only training-induced effect is a weak, still
provisional circular *ordering* of digit values. This refutes the strong
"dominant digit geometry" reading and points the thread toward
activation/MLP-level and causal questions rather than token-embedding geometry.

## Top current claims

- **CE1** — digit embeddings are near-isotropic 9-D categorical codes with a
  weak, training-induced circular *ordering* (not a dominant circle/helix);
  confidence Medium (no-dominant-geometry) / Low-provisional (ordering). See
  [maths-claim-evidence.md#ce1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix).
- **CE2** — at layer-0 operand-fetch heads the value-path output is
  indistinguishable from linear transport of the (weakly-circular) embeddings;
  confidence Medium. Extends the "weak low-variance projection" motif from the
  embedding to the value path. See
  [maths-claim-evidence.md#ce2](maths-claim-evidence.md#ce2-at-layer-0-operand-fetch-heads-the-value-path-output-is-indistinguishable-from-linear-transport-of-the-weakly-circular-digit-embeddings).

## Strongest live caveats

- Weights-only / correlational: nothing yet on whether the model *uses* any of
  this geometry (needs causal path-patching).
- **A2 (aggregate-then-discretize) is untested**: the first assay was an
  `instrument failure` (answer-position nodes, not confirmed `ST` nodes; a
  ratio metric that could not separate aggregation from transport).
- A recurring method trap: tidy low-D structures (digit circle, ST tri-cluster,
  sum arc) keep turning out to be **low-variance projections**; verdicts require
  explicit nulls and independent node-role confirmation.

## Open questions shaping near-term work

- Where is `ST`/carry computed, and does attention aggregate or transport?
  (Needs a confirmed question-position ST node.) Does the model causally use
  digit geometry? See [maths-next-steps.md](maths-next-steps.md).
