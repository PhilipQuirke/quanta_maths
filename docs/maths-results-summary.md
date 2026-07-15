# Maths Results Summary (maths-results-summary.md)

Read role and rules: [Results Summary](thor-document-rules.md#results-summary).

Compact current-state synthesis for the `maths` thread. Orients a human or agent
quickly. Update only when a result changes the current story. Link downward for
detail; do not turn this into a ledger.

## Executive summary

Five studies complete. The thread has moved from representation to a causal
mechanism for the carry. (1) Digit embeddings are **not** a clean circle/helix —
near-isotropic in 9-D, only a weak circular *ordering* (CE1). (2) At operand-fetch
heads the value path looks like linear *transport* of those weak embeddings
(CE2). (3) The per-digit **binary carry** is computed by specific layer-0
attention heads at answer positions, dissociated from base-add heads (CE3). (4–5)
The **tri-state `U`-resolution runs on a separate path**: layer-0 nodes conduct
the running carry and the **answer-position layer-1 MLP combines** it with the
digit's sum-class to resolve `U` (CE5, replicated; supersedes the ambiguous CE4).
The emerging chain: embeddings → L0 heads (base-add + binary carry) → L0 conduits
→ **L1 MLP U-combiner** → readout.

## Top current claims

- **CE1** — digit embeddings are near-isotropic 9-D categorical codes with a
  weak, training-induced circular *ordering* (not a dominant circle/helix);
  confidence Medium (no-dominant-geometry) / Low, LN-robust-weakly & seed-fragile
  (ordering — settled by the LN-aware close-out; no longer provisional). See
  [maths-claim-evidence.md#ce1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix).
- **CE2** — at layer-0 operand-fetch heads the value-path output is
  indistinguishable from linear transport of the (weakly-circular) embeddings;
  confidence Medium. Extends the "weak low-variance projection" motif from the
  embedding to the value path. See
  [maths-claim-evidence.md#ce2](maths-claim-evidence.md#ce2-at-layer-0-operand-fetch-heads-the-value-path-output-is-indistinguishable-from-linear-transport-of-the-weakly-circular-digit-embeddings).
- **CE3** — the carry is computed by **binary make-carry heads at answer
  positions**, cleanly dissociated from base-add heads (one head per role per
  position); the **tri-state `U`-resolution is a separate, not-yet-located
  path**. Confidence Medium-High (make-carry + dissociation) / Medium
  (separate path). See
  [maths-claim-evidence.md#ce3](maths-claim-evidence.md#ce3-the-carry-is-computed-by-binary-make-carry-heads-at-answer-positions-dissociated-from-base-add-heads-the-tri-state-u-resolution-is-a-separate-unlocated-path).
- **CE4** — the tri-state `U`-resolution flip is *transmitted* by an **MLP-heavy
  L0/L1 path distinct from the make-carry heads**; combiner-vs-conduit role
  unresolved (the interaction discriminator was vacuous). Confidence Low–Medium.
  *Superseded by CE5.* See
  [maths-claim-evidence.md#ce4](maths-claim-evidence.md#ce4-the-tri-state-u-resolution-flip-is-transmitted-by-an-mlp-heavy-l0l1-path-distinct-from-the-make-carry-heads-whether-it-is-combined-or-merely-relayed-is-unresolved).
- **CE5** — the tri-state `U`-**combiner is the answer-position layer-1 MLP**
  (`P14/P16.L1.MLP`; its output is the resolved `carry_out`, replicated in both
  models); **layer-0 nodes relay the running carry (conduit)**, clean in
  6-digit. Confidence Medium-High (combiner) / Medium (L0-conduit). See
  [maths-claim-evidence.md#ce5](maths-claim-evidence.md#ce5-the-tri-state-u-combiner-is-the-answer-position-layer-1-mlp-layer-0-nodes-relay-the-running-carry-conduit).
- **CE6** — at the **combiner input** the carry is a clean **binary** code (no
  off-axis tri-state; `U` split by resolution, `U→0`≈committed-0, `U→1`≈
  committed-1) — **A3's off-axis third-symbol refuted at this locus** (a
  tri-state could still exist upstream). Confidence Medium. See
  [maths-claim-evidence.md#ce6](maths-claim-evidence.md#ce6-at-the-u-combiners-input-the-carry-is-a-clean-binary-code--no-distinct-off-axis-tri-state-the-resolved-carry-is-already-linearly-present-there).
- **CE7** — **no dedicated `{0,1,U}` tri-state symbol at any answer-position
  residual site**; the carry is binary throughout and `U` is resolved to binary
  **around L1-attention** — A3's off-axis form refuted across the answer-position
  stream (scoped: comparable-magnitude symbol; weak/question-position untested).
  Confidence Medium. See
  [maths-claim-evidence.md#ce7](maths-claim-evidence.md#ce7-no-dedicated-01u-tri-state-symbol-at-any-answer-position-residual-site-u-is-resolved-to-binary-around-l1-attention).
- **CE8** — attention routing is **hybrid**: a few heads (`L1.H1` operand-read
  Q11 & answer Q14; `L0.H0` answer Q17) **relocate their target with the carry
  state** (A5's strong static-wiring form falsified) — but most cells are
  target-static. 6-digit only (5-digit inconclusive); representational not
  causal. Confidence Medium. See
  [maths-claim-evidence.md#ce8](maths-claim-evidence.md#ce8-attention-routing-is-hybrid--a-few-heads-relocate-their-target-with-carry-state-6-digit-only-most-cells-are-target-static).

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

- Attention routing is **hybrid** (CE8): are the carry-routing L1 heads
  *causally* load-bearing (pattern-patching, cascade-tracing entry)? The tri-state
  `U` is **never a dedicated symbol** at answer positions (CE7); the remaining A3
  remnant is only a *transient* `U` at question positions or a weak symbol.
  Deeper: the multi-digit `...999` cascade (A6 tie-break), whether the model
  causally uses digit geometry (B1), and A2 aggregate-vs-transport. See
  [maths-next-steps.md](maths-next-steps.md).
