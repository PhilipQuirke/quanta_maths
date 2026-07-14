# Maths Human Conjectures (maths-conjectures-human.md)

Read role and rules: [Conjectures](thor-document-rules.md#conjectures).

Human-owned. The agent may read but not edit this file. State beliefs plainly,
including ones the agent may be unaware of. Expect these to be wrong and to
change over time — that is the point.

Phrase entries as beliefs, predictions, falsifiers, and alternatives — not as
"we observed". Observations belong in the results docs.

## Research Direction

**Goal: a worked example of "how a model calculates".** The logical features a
trained transformer uses to perform addition and subtraction are already well
documented (see [Context](#context-what-the-paper-already-established) below).
The next step is to explain how those known logical features are *physically
represented and moved* inside the model:

1. **Representation** — How are these features represented in high-dimensional
   activation space?
2. **Geometry / relationships** — Are the different features "near" each other,
   orthogonal, or otherwise physically related?
3. **Storage** — How and where is a feature's output information stored?
4. **Propagation** — How is stored information propagated token-to-token,
   especially the [SV](thor-glossary.md#sv) and
   [ST](thor-glossary.md#st) carry/borrow cascade?

This deliberately goes beyond the project's earlier work, which (a) used only 2D
/ 3D PCA *projections* of activations and never engaged the full
high-dimensional space the activations live in, and (b) never characterized how
activations interact token-by-token. Newer interpretability work reports model
features living on sparse, curved manifolds; these are *toy* models with
*understood* features, so they are an ideal, tractable testbed for asking whether
that complexity is present or whether the representation is simple.

### Relation to the published paper

This direction is exactly the paper's own declared Limitations / Future Work:
the paper identifies the functional *role* of each node but explicitly does not
analyze the *data representations* and *transformations* those nodes use in the
residual stream, nor the detailed MLP mechanisms. We are executing that gap.

**Guardrail:** Make **no changes to the published paper** (`study-paper/paper.tex`,
`paper.bib`) until this research lands and we understand the findings. The
`paper` thread stays hands-off on this work until then.

## Context: what the paper already established

A future agent should treat the following as settled background (source:
`study-paper/paper.tex`, arXiv:2402.02619; long-form terms in
[terminology.md](terminology.md) and the [glossary](thor-glossary.md#project-terms)).

### Task and notation

- Small transformers (2–3 layers, 2–4 heads, 5–15 digits) trained from scratch
  perform n-digit integer addition and subtraction at >99.999% accuracy.
- Tokens: first number digits `D_n..D0`, second number `D'_n..D'0`, answer
  `A_n..A0` (with `A_max` the `+`/`-` sign). `PnLnHn` names an attention head at
  a token position and layer; `PnLnMn` an MLP neuron. A **node** is the logical
  locus of a subtask — physically an attention head, an MLP layer, or
  occasionally two heads working together.

### The logical features (subtasks) — the "what", already known

The subtasks are defined canonically in the
[glossary Project Terms](thor-glossary.md#project-terms); this direction relies
on the addition features [SA_n](thor-glossary.md#s-addition-sub-tasks-sa-sc-ss-st-sv)
(Base Add), [ST_n](thor-glossary.md#st) (TriCase, tri-state `{0, 1, U}`),
and [SV_n](thor-glossary.md#sv) (cascaded carry), their subtraction parallels
[MD_n / MB_n / MV_n](thor-glossary.md#m-positive-answer-subtraction-sub-tasks-md-mb-mz-mt)
and [ND_n](thor-glossary.md#n-negative-answer-subtraction-sub-tasks-nd-nb-nz-nt),
and the selection features [SGN](thor-glossary.md#sgn) (answer sign) and
[OPR](thor-glossary.md#opr) (operator).

The two research-critical points behind this direction:

- **The tri-state is the novelty.** `ST_n` (and its borrow parallel `MB_n`) has an
  **uncertain `U`** value (the digit pair sums to exactly 9), which is what makes
  the carry cascade non-trivial. This tri-state shows up as 3 PCA clusters.
- **The cascade resolves at `=`.** `SV_n` combines `ST` values high→low digit via
  `TriAdd`, resolving all `U` uncertainty by the `=` token so every `SV_n` ∈
  {0, 1}; the final answer combines `SV_n` with `SA_n`. In mixed models the
  resolved `MV_n` at `=` selects the answer sign and whether to emit `MD_n` or
  `ND_n` digits, using `SGN`/`OPR`.

### Where these features live and how they behave — established constraints

- **Attention vs MLP role split (paper's read):** attention heads route/move
  per-digit information across token positions; MLP layers perform the tri-state
  transformation. Example (5-digit addition, Hypothesis 3): `ST_n` is computed at
  heads like `P8.L0.H1`, `P9.L0.H1`, `P11.L0.H2`, `P14.L0.H1`; some `ST`
  calculations are attributed to *both* an attention head and its MLP
  (`P9.L0.H1` **and** `P9.L0.MLP`), and ablation shows both are necessary.
- **Ordering constraints:** because `SV1 = TriAdd(ST1, ST0)`, an `SV1` node must
  sit after the `ST1`/`ST0` nodes; a 6-digit model has 30+ such constraints. The
  cascade genuinely propagates left-to-right across token positions until the
  `=` token.
- **Cross-position reuse is real but partial:** e.g. `P14` recomputes `ST1`
  directly and only falls back to the earlier `P10.ST2` value when `ST2 != ST1`.
  So information is both recomputed and carried.
- **Redundancy and variability:** models vary in which heads compute which
  subtask; some redundantly compute a subtask twice; some retain legacy `SC`
  (single-digit carry) nodes even though `ST`/`SV` supersede them; `SA_n` is
  sometimes split across two heads.
- **Tri-state PCA evidence:** `ST` nodes show 3 visually distinct PCA clusters
  aligned to `{0, 1, U}`, verified with purpose-built test question sets. This is
  the prior representational evidence — but only as a low-dim *projection*.
- **Polysemanticity (mixed models):** parameter-transfer-initialized mixed models
  reuse addition circuits rather than growing new ones; a node that did only
  `SA` often becomes polysemantic, handling `SA`, `MD`, and `ND`.

## Model scope for this direction

- **Addition first, then mixed.** Establish representation geometry and cascade
  dynamics on a simple, well-mapped addition model (the 5-digit addition model
  with the Hypothesis-3 node map is a strong starting point), then extend to the
  mixed model (`ins1_mix_d6_l3_h4_t40K_s372001`, already featured in
  [mixed_model.md](mixed_model.md), [pca.md](pca.md), and `assets/`) to study
  polysemantic `SA`/`MD`/`ND` overlap and the `OPR`/`SGN` selection geometry.

## Priors the agent should know

- These are toy models with fully-understood *logical* features. Do not assume
  the elaborate curved/sparse-manifold structure reported for large LMs is
  present here; test whether it is, but the live hypothesis is that the
  representation is *simple*.
- The earlier PCA work is a *projection*, not the geometry. Three visible ST
  clusters in 2D do not tell us the true dimensionality, curvature, or
  inter-feature angles. Prefer methods that engage the full high-dimensional
  residual-stream geometry over more 2D/3D projections.
- The `=` token is the resolution point of the cascade; token position matters
  as much as layer. Any representation/propagation account must respect the
  paper's ordering constraints.

## Current conjectures

### C1: Representations are simple, near-linear, and low effective-dimension

- **Belief**: Because the models are toy and the features are well understood,
  each subtask feature is encoded in a simple, near-linear, low-dimensional form
  (approximately one direction / small subspace per feature), *not* on a complex
  curved or heavily superposed manifold.
- **Why**: Small models, exact algorithm, abundant capacity relative to task; the
  tri-state `ST` clusters already look like clean, separable structure in low-dim
  PCA.
- **Prediction**: The effective dimensionality of a subtask's activation
  variation is small; linear probes recover `SA_n`/`ST_n`/`SV_n` with near-perfect
  accuracy; a `U`-vs-{0,1} `ST` split is close to linearly separable in the full
  space, not only in a curved projection.
- **Falsifier**: Linear probes fail where a curved/manifold decoder succeeds, or
  the feature needs many components to be read out — indicating genuine curvature
  or superposition.
- **Alternatives**: (a) structured but curved manifolds; (b) superposed features
  sharing directions that only separate non-linearly.
- **Confidence**: medium.

### C2: Same-subtask features share a geometric template; different subtasks are more orthogonal

- **Belief**: The same subtask across digit positions (all `ST_n`, or all
  `SA_n`) shares a common geometric template — the same directions, offset or
  rotated per position — while *different* subtasks (`SA` vs `ST` vs `SV`) occupy
  more nearly orthogonal subspaces so they do not interfere during the cascade.
- **Why**: The algorithm applies the same operation at each digit; orthogonality
  across subtasks is the natural way to avoid cross-talk in a shared residual
  stream.
- **Prediction**: Cosine similarity / subspace angle between `ST_i` and `ST_j`
  directions is high (shared template); between `SA`, `ST`, `SV` subspaces is
  low (near-orthogonal). A probe trained on `ST2` transfers to `ST3` better than
  a probe trained on `SA2` transfers to `ST2`.
- **Falsifier**: `ST_n` across positions are mutually orthogonal (no shared
  template), or `SA`/`ST`/`SV` are entangled/non-orthogonal.
- **Alternatives**: position-specific idiosyncratic encodings; a single mixed
  subspace carrying everything.
- **Confidence**: medium.

### C3: Attention moves, MLP transforms; the cascade rides the residual stream

- **Belief**: Feature output information is written to the residual stream at
  specific token positions; attention heads *move/route* per-digit state across
  positions while MLP layers perform the tri-state *transformation* and write the
  resolved value. The `ST`→`SV` carry/borrow cascade is carried forward token-by-
  token in the residual stream and resolved at the `=` token.
- **Why**: Matches the paper's role split and the ordering constraints
  (`SV` nodes must follow their `ST` inputs; `P14` reuses `P10.ST2`).
- **Prediction**: The `SV_n` "resolved carry" direction becomes decodable at
  progressively later token positions and is fully resolved (all `U` gone) by
  `=`; zeroing/patching the residual stream between an `ST` node and its
  downstream `SV` node breaks the cascade; ablating an MLP removes the tri-state
  transform while ablating the routing head removes cross-position transfer.
- **Falsifier**: The cascade state is not present in the residual stream between
  positions (e.g. recomputed locally each step with no carried state), or MLPs
  merely pass through while attention does the transformation.
- **Alternatives**: attention alone both routes and transforms; information is
  stored in attention-pattern/KV structure rather than the residual stream.
- **Confidence**: medium.

## Notes

- It is healthy for human and agent conjectures to disagree; that tension should
  drive experiment ranking in [maths-next-steps.md](maths-next-steps.md).
- These conjectures are the *human's* starting priors and are expected to be
  partly wrong. Bias early experiments toward discriminating C1 (simple/linear)
  vs the curved/superposed alternatives, and toward the C2 geometry question,
  since those most cheaply invalidate large areas of the search space.
- Empirical results, prediction scoring, and any curved-vs-linear verdicts belong
  in the study notes and results docs — not here.
