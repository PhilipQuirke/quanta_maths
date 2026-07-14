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

## Strongest live caveats

- Weights-only: says nothing about whether the model *uses* digit geometry
  (causal) or about activation-level feature geometry.
- The circular-ordering signal is raw-weights only; the LN-aware secondary
  analysis (A-9) is outstanding and could change it.
- Findings so far concern the token embedding, not the tri-state `ST`/`SV`
  cascade features that motivate the thread.

## Open questions shaping near-term work

- Does the model causally use digit magnitude/circularity despite a weak
  embedding geometry? Where do the `ST`/pair-sum features live if not in the
  embedding? See [maths-next-steps.md](maths-next-steps.md).
