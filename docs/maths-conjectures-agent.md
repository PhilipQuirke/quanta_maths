# Maths Agent Conjectures (maths-conjectures-agent.md)

Read role and rules: [Conjectures](thor-document-rules.md#conjectures).

Agent-owned. The agent writes and maintains this file, updating it after new
empirical results land and after reflecting on them (and only after the
post-result skeptic gate passes). It may disagree with
[maths-conjectures-human.md](maths-conjectures-human.md).

Phrase entries as beliefs, predictions, falsifiers, and alternatives. Link to
results docs as backlinks rather than restating results here.

## The agent's overall picture (speculative)

My expected end-to-end story for how a trained addition model computes
`D5..D0 + D'5..D'0 = A6..A0`, phrased as belief, not observation:

1. **Numeric geometry at the embedding.** Digit tokens `0-9` are embedded with
   functional structure — a few circular (Fourier-like) components realizing
   mod-10 geometry, possibly plus an ordered magnitude component — rather than
   as 10 arbitrary near-orthogonal symbols (A1). *[2026-07-14: this stage took
   damage — the embedding is near-isotropic with only a weak circular ordering;
   see A1 confidence update and [CE1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix).
   The "how it computes" weight now leans on stages 2–5 (activation/MLP), not
   the embedding.]*
2. **Aggregate, then discretize.** Fixed position-based attention (the double
   staircase) pairs `Dn` with `D'n`; because the attention value path is
   linear, the head output contains the superposition
   `embed(Dn) + embed(D'n)` — the raw pair sum, one of 19 ordered states. The
   adjacent MLP applies the nonlinearity that snaps this continuum into
   categorical sub-task outputs: the
   [SA_n](thor-glossary.md#s-addition-sub-tasks-sa-sc-ss-st-sv) digit and the
   [ST_n](thor-glossary.md#st) tri-state (A2, A3). This division of labor is
   why ablating either the head or its MLP breaks the node.
3. **Position-addressed storage.** A sub-task's output is written into a small
   subspace of the residual stream *at a specific token position* — a
   position-addressed register. The feature template is shared across digit
   positions (the same weights compute it everywhere); *which* digit a value
   belongs to is encoded by where it sits, not by a per-digit direction (A4).
4. **Static routing; tie-break cascade.** Downstream nodes fetch stored values
   with attention patterns that are essentially fixed wiring (A5). The
   [SV](thor-glossary.md#sv) carry cascade rides the residual stream token by
   token, but mostly as a tie-breaker: positions recompute `ST` locally where
   they can and consult the carried state only to resolve `U` (A6). All `U` is
   resolved by the `=` token.
5. **Answer emission.** Each answer position fetches its `SA_n` and resolved
   carry just in time; an MLP computes `(SA_n + carry) % 10`; the logits are
   read off the same digit geometry the embeddings started with (A1).
6. **Mixed models reuse the engine.** Subtraction reuses the addition
   machinery under low-rank control: [OPR](thor-glossary.md#opr) and
   [SGN](thor-glossary.md#sgn) act as switch directions selecting the
   add / subtract / negated readout of shared per-digit computation. On a
   mod-10 circle, `a - b` is `a + b` with one operand reflected, so reuse is
   geometrically cheap (A7).

Against the human file's four framing questions, this picture says: features
are **represented** as small linear subspaces whose content may be curved
(circular); **related** by shared templates across positions and weak
interference across unrelated sub-tasks; **stored** as position-addressed
registers in the residual stream; **propagated** by static positional
attention, with the cascade carried forward as compact tie-breaking state.

## Relation to human conjectures

- **C1 (simple, near-linear, low-dimensional)** — agree on simple and
  low-dimensional; refine "near-linear": I expect *linear subspaces containing
  curved (circular) content*. A strictly linear-probe frame can under-detect a
  circle (A1, A8).
- **C2 (shared template within sub-task; orthogonal across sub-tasks)** —
  agree on template sharing, and supply a mechanism: weight sharing across
  token positions forces it (A4). But I relocate the orthogonality half: at
  question positions, separation is *positional*, so cross-sub-task angle
  matters less than C2 implies; the interesting binding question is where
  values must coexist (`=` and answer positions). For mixed models I predict
  the opposite of orthogonality between `SA` and `MD`/`ND` on shared nodes
  (A7).
- **C3 (attention moves, MLP transforms, cascade rides the stream)** — agree,
  with two sharpenings: attention's linear mixing *is itself half the
  arithmetic* (it performs the sum in embedding space, not mere transport)
  (A2), and the routing is static wiring (A5). A6 adds a tie-break economy and
  a "wide fetch" alternative C3's framing could miss.

## Current conjectures

### A1: Digit embeddings carry functional numeric geometry (mod-10 circle, maybe helix)

- **Belief**: The embeddings (and unembeddings) of digit tokens `0-9` encode
  functional numeric structure: a few circular Fourier-like components
  realizing mod-10 geometry, possibly plus an ordered magnitude direction
  (together, a helix). Sub-task computations operate *in* this geometry —
  `SA` as rotation, `MD`/`ND` as reflection plus rotation, carry/borrow as
  thresholds — rather than treating digits as 10 unrelated symbols.
- **Why**: From-scratch models trained on modular arithmetic reliably converge
  to Fourier-feature algorithms, and per-digit base-10 addition *is* mod-10
  addition plus a carry bit. A circle makes addition and subtraction the same
  cheap operation family. The vocabulary is tiny, though, so this is genuinely
  contestable — a 10×10 lookup is also cheap (see Alternatives).
- **Support**: Nanda et al. 2023 (from-scratch mod-p models learn Fourier
  features); Zhong et al. 2023 (families of circular algorithms); Kantamneni &
  Tegmark 2025 (numbers as a generalized helix; "Clock" addition); Levy & Geva
  2024 (per-digit base-10 circular representations); Engels et al. 2024
  (circular features are genuine multi-dimensional features). Pulling the
  other way: Zhou et al. 2024 found *from-scratch* models leaned on
  low-frequency features only — though in a multi-digit-number-token regime,
  unlike this repo's 10-digit vocabulary.
- **Prediction**: Digit embedding geometry is dominated by a few components
  with circular order (wrap-around adjacency: `9` next to `0`); the same
  structure appears in the unembedding; sub-task outputs such as `SA_n` align
  with this geometry rather than with an unstructured 10-way code.
- **Falsifier**: Digit embeddings are mutually near-orthogonal with no
  consistent circular/ordered structure, and node outputs show lookup-style
  idiosyncrasy per digit pair.
- **Alternatives**: (a) pure lookup — MLP key-value memories over the 100
  digit-pair patterns (Geva et al. 2021), viable because the table is tiny;
  (b) a bag of pattern-specific heuristics rather than one clean geometry
  (Nikankin et al. 2024); (c) a hybrid: geometric common path plus memorized
  patches, as in the parallel-paths addition story found in a production LLM
  (Anthropic 2025).
- **Tension with human**: Refines C1 — "simple" survives but "approximately
  one direction per feature" may not; the natural unit is a small subspace
  with circular content.
- **Confidence**: LOWERED 2026-07-14 after the digit-embedding study
  ([CE1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix)):
  the strong "dominant circular/helical geometry at the token level" form is
  **refuted** for these models' embeddings (near-isotropic 9-D; circular
  *variance* at noise floor; unembedding unstructured and misaligned). A weak
  circular *ordering* survives (3/4 accurate models, absent untrained). **Update
  2026-07-16 (LN-aware close-out, A-9):** the weak ordering **persists in the
  LN-effective geometry** the computation reads (position-free and model-true) —
  but LN is *near-isometric* here (γ std ~0.005), so this is only *weak*
  robustness, and the ordering holds in **2 of 3 independent seeds** (fails
  s173289). So: **low** that a dominant circle/helix organizes the digit
  *embedding*; **low** that a weak circular ordering exists (real, LN-robust
  weakly, seed-fragile — no longer provisional); the belief's real test moves to
  whether the *computation* uses circular/magnitude structure (causal, B1)
  regardless of a clean embedding — was medium-high/medium.

### A2: The core computation is aggregate-then-discretize

- **Belief**: At an arithmetic node, attention and MLP split the work. The
  head, with an essentially fixed pattern, attends roughly equally to `Dn` and
  `D'n`; by linearity of the value path its output is approximately
  `(embed(Dn) + embed(D'n)) / 2`, a representation whose sufficient statistic
  is the pair sum `Dn + D'n` (19 ordered states). The adjacent MLP applies the
  nonlinearity that discretizes this continuum into the categorical sub-task
  output (`SA` digit, `ST` tri-state). Neither half does arithmetic alone.
- **Why**: This is the cheapest circuit a transformer can express — summation
  is free in a linear value path; only discretization needs the MLP. It also
  *explains* the joint-necessity ablation result at `ST` nodes (head and MLP
  both required) reported in Paper 2.
- **Support**: Elhage et al. 2021 (linear OV / residual framework); Nanda et
  al. 2023 (attention mixes operands, MLPs do the nonlinear step); Hanna et
  al. 2023 (MLPs compute the comparison in greater-than); Stolfo et al. 2023
  (attention routes operands, MLPs compute results); Paper 2 ablations
  (backlink).
- **Prediction**: Between head and MLP, activations depend on the operands
  almost solely through `Dn + D'n` — equal-sum digit pairs are near-identical
  pre-MLP — and form an ordered low-dimensional arc; the 3-way `ST` clustering
  exists only post-MLP, with `U` sitting at the `Dn + D'n = 9` point of the
  pre-MLP arc.
- **Falsifier**: Head outputs are already categorical (attention itself
  transforms), or no pair-sum-sufficient representation exists pre-MLP
  (operands routed separately and combined elsewhere).
- **Alternatives**: Separate routing of the two operands with late
  combination at answer positions; comparisons computed inside the attention
  pattern (QK) rather than the value path.
- **Tension with human**: Sharpens C3 — "attention moves" undersells it;
  attention's linear mixing is the addition itself, performed in embedding
  space.
- **Confidence**: HELD at medium-high (2026-07-14). The first pre-MLP assay
  ([study-pair-sum-sufficiency.md](study-maths/study-pair-sum-sufficiency.md))
  was an **`instrument failure`** — it confirmed answer-position operand-fetch
  heads rather than question-position `ST` compute nodes, and its no-lower-carry
  stimulus pinned `R²_pair=1` so the sum/pair ratio could not distinguish
  aggregation from transport (a noise-free transport null reproduced the whole
  signature). A2's actual prediction (pre-MLP sum-sufficiency *at a confirmed ST
  node*, with MLP-created tri-state) is therefore **untested**, not refuted.
  A real test needs a path-patching-confirmed ST node and a discriminating
  metric (see agenda). One incidental datapoint: at operand-fetch heads the
  value path looks like pure transport of weakly-circular embeddings (CE2),
  which weakly cautions against over-reading attention as adding "extra"
  structure, but does not bear on A2's ST-node claim.

### A3: The `ST` tri-state is a 2D categorical code, with `U` off the 0–1 axis

- **Belief**: Post-MLP, `ST ∈ {0, 1, U}` is stored as a categorical code:
  three well-separated centroids spanning roughly two dimensions, with the `U`
  centroid carrying a substantial component off the 0–1 axis. `U` is a third
  symbol, not a midpoint. (Per A2, the MLP's job is exactly to bend the
  ordered pre-MLP arc into this categorical shape.)
- **Why**: Downstream `TriAdd` logic consumes `U` as "defer to the next-lower
  digit" — a branch, not an intermediate magnitude. Categorical readout by
  linear consumers is easiest with separated directions, and categorical
  concepts in LMs tend to form simplex-like polytopes.
- **Support**: Park et al. 2024 (categorical concepts as simplices); Engels et
  al. 2024 (multi-dimensional feature geometry); the 3-cluster PCA structure
  from Paper 2 as prior — though only a projection, which is the human file's
  own caveat.
- **Prediction**: In the full space the three `ST` centroids are pairwise well
  separated, with `U` substantially off the 0–1 line; a single direction with
  two thresholds does not capture what downstream consumers read.
- **Falsifier**: `U` lies on the segment between the 0 and 1 centroids — an
  ordered scalar code; one direction suffices.
- **Alternatives**: (a) the ordered 1D scalar code above; (b) two
  near-independent binary directions — a "definitely-carry" bit and a
  "sum-is-9" bit, echoing the legacy `SC`/`SS` sub-tasks — giving three used
  corners of a square, distinguishable from a simplex by angle geometry.
- **Tension with human**: None direct; turns C1's cluster observation into a
  geometry question with three discrete candidate shapes.
- **Confidence**: LOWERED to **low** 2026-07-16. Two studies now refute the
  off-axis-third-symbol form: [CE6](maths-claim-evidence.md) at the combiner
  input, and [CE7](maths-claim-evidence.md) across **all 7 answer-position
  residual sites** (the is-U axis is never significant vs the null; `U` is
  resolved to binary around L1-attention, never a dedicated symbol). Scoped: this
  refutes a symbol *comparable in magnitude to the binary carry* at
  *answer-position* sites; **untested** are a *weak* symbol (≤~0.3× the committed
  separation, below the detection floor) and a *transient* `U` at **question
  positions** (D'n, descoped) — the only regimes where A3's remnant can still
  live. A3 is not globally dead, but the "U as a dedicated symbol" frame is
  strongly disfavored (see streak note in the reflection log).

### A4: Features are position-addressed: shared templates plus positional binding

- **Belief**: Because model weights are shared across token positions, a node
  computing the same sub-task at several positions must read and write one
  shared representational template; the digit index `n` is encoded by *where*
  the value sits, with positional-embedding components as the binding tag —
  not by per-digit feature directions. Consumers fetch by position. At `=` and
  answer positions, coexistence of several resolved per-digit states is mostly
  avoided by just-in-time fetching rather than solved with orthogonal
  per-digit slots.
- **Why**: Weight sharing makes template reuse the only cheap option; the
  rigid input format makes positional addressing perfectly reliable; storing
  everything at one position would recreate a binding problem the architecture
  lets the model avoid.
- **Support**: Elhage et al. 2021 (weight sharing, residual stream); Feng &
  Steinhardt 2023 (binding via tag vectors); Paper 2 (single heads serving
  multiple `ST_n`; answer positions attending to specific question positions)
  (backlink).
- **Prediction**: Same-sub-task representations at different positions are
  strongly aligned — an `ST` readout transfers across digit positions nearly
  for free (agreeing with C2's transfer prediction); at any single position
  only a few sub-task values are simultaneously present; where two
  same-template values must coexist at one position they differ by a
  position-derived offset, not by unrelated directions.
- **Falsifier**: Same-sub-task representations at different positions are
  unrelated (no transfer), or all resolved `SV_n` are simultaneously and
  orthogonally stored at the `=` position.
- **Alternatives**: Fully position-specific idiosyncratic codes; or a "tape"
  layout where every digit's state has a dedicated orthogonal slot at `=`.
- **Tension with human**: Agrees with C2's template half and supplies its
  mechanism; demotes C2's orthogonality half at question positions (separation
  is positional there) and relocates the interesting version of the question
  to the answer phase.
- **Confidence**: medium-high on template sharing; medium on just-in-time
  fetch.

### A5: Attention routing is static wiring; data-dependence lives in values and MLPs

- **Belief**: Attention patterns are near-fixed functions of token position
  (question-independent staircases); QK structure is dominated by position;
  digit values flow through the value path. Algorithmic branches (use the
  carried `ST` only when needed; add vs subtract) are implemented by
  value-space selection and MLP nonlinearity, not by moving attention targets.
- **Why**: With a rigid input format, positional wiring is available, learned
  early, and robust; content-based routing is only worth its complexity when
  the task's structure varies across examples, which this task's does not.
- **Support**: The stereotyped double-staircase attention of Papers 1–2
  (backlink); contrast with Wang et al. 2022 (IOI), where the task itself
  forces content-dependent routing.
- **Prediction**: Attention maps are near-invariant across questions of a
  given format; changing digit values changes outputs while leaving patterns
  near-unchanged; every data-dependent branch in the algorithm is located
  after the attention pattern, in values and MLPs.
- **Falsifier**: Systematic value-dependent attention shifts at cascade or
  selection nodes — e.g. a head's target switching when an `ST` input is `U`.
- **Alternatives**: Hybrid routing in which a few selection nodes genuinely
  move attention by content.
- **Tension with human**: Consistent with C3; sharpens "attention moves" into
  "attention is static wiring".
- **Confidence**: LOWERED to **medium 2026-07-16** — the strong "static
  positional wiring" form is **falsified / narrowed to hybrid** by the
  attention-invariance census ([CE8](maths-claim-evidence.md)). Under a
  value-matched contrast (digit-`n` fixed, only lower carry toggled), a few heads
  (`L1.H1` operand-read Q11 & answer Q14; `L0.H0` answer Q17) **relocate their
  attention target with the carry state** — A5's own pre-registered falsifier —
  clean and Bonferroni-safe in the 6-digit model (5-digit inconclusive;
  representational, not causal). A5's *majority* claim (most cells target-static)
  survives, and the earlier CE3 partial support (confirmed carry heads have
  question-independent operand attention) stands, but "attention is static
  wiring" as an unqualified statement is refuted. **Reword the belief to: mostly
  static positional wiring, with genuine carry-state target-routing at a few L1
  (and one L0) heads.** The routing cells are candidate nodes for a causal
  pattern-patching test (cascade-tracing entry).

### A6: The carry cascade is carried state, but load-bearing only as a tie-breaker

- **Belief**: The `SV`/`MV` cascade state is genuinely carried forward in the
  residual stream across positions (agreeing with C3 against
  local-recompute-only), but it is load-bearing only as a tie-breaker:
  positions recompute `ST` locally where operands remain attendable and
  consult the carried state only to resolve `U`. Written features persist —
  the stream behaves as an accumulating bus (layers add, rarely erase) — until
  consumed.
- **Why**: Recomputation is cheap here because operand digits stay visible
  from every later position; carried state is only *necessary* where per-digit
  locality fails, which is exactly the nines cascade. Training pressure plus
  abundant capacity tends to produce this kind of redundancy, and it matches
  Paper 2's recompute-with-fallback reading of `P14` (backlink).
- **Support**: Elhage et al. 2021 (residual bus); Paper 2 ordering constraints
  and the fallback observation.
- **Prediction**: Disrupting the inter-position cascade state selectively
  harms the `U`-cascade question family (long `...999 + 1`-style chains) while
  sparing carry-free questions; cascade information becomes decodable at
  progressively later positions, with its `U` component gone by `=`; a written
  sub-task output stays decodable from its write point forward at that
  position.
- **Falsifier**: Carried state is equally load-bearing for all questions (a
  pure pipeline with no local recompute), or there is no inter-position
  cascade state at all (pure recompute — C3's own falsifier).
- **Alternatives**: "Wide fetch": one late node attends to all `ST_n`
  positions at once and computes the whole cascade in a single nonlinear step
  at `=`. Shallow models permit this within the ordering constraints, so it
  stays live.
- **Tension with human**: Agrees with C3's storage/propagation core; adds the
  tie-break economy and the wide-fetch alternative.
- **Confidence**: medium; **carried+combined sub-claim SUPPORTED (refined)
  2026-07-15** by the combiner-vs-conduit study ([CE5](maths-claim-evidence.md)):
  layer-0 nodes conduct a running carry (clean in 6-digit) which the
  **answer-position L1 MLP combines** with the digit's sum-class to resolve `U`
  — genuine carried state consulted for the `U` tie-break, resolved at the
  answer position (not `=`; refines the wording). The **tie-break-economy**
  sub-claim (selective harm to multi-digit `...999` cascades) remains
  **untested** (single-digit `U` only). Raise confidence on the
  carried+combined core to medium-high; hold the tie-break economy at medium.

### A7: Mixed models are one engine under low-rank `OPR`/`SGN` control, not parallel circuits

- **Belief**: Mixed models run one shared per-digit arithmetic engine steered
  by low-rank control. On a node serving `SA`/`MD`/`ND`, operand aggregation
  and much of the representation are shared; `OPR` and `SGN` behave like
  function vectors — compact directions selecting which readout applies —
  rather than gates between disjoint circuits. If A1 holds this is
  geometrically natural: on a mod-10 circle, subtraction is addition with one
  operand reflected, so the switch is small.
- **Why**: The mixed models were initialized from addition models and became
  polysemantic rather than growing separate circuits — which is what a
  shared-engine-plus-switch solution looks like from outside. A task-selector
  direction is the cheapest way to multiplex shared machinery.
- **Support**: Paper 2 (parameter-transfer initialization; polysemantic
  `SA`/`MD`/`ND` nodes) (backlink); Todd et al. 2023 (function vectors);
  Merullo et al. 2023 (compact additive task signals steering shared
  machinery).
- **Prediction**: On shared nodes, `SA_n` and `MD_n`/`ND_n` readouts overlap
  heavily (far from orthogonal); operator information is compressed into a
  low-dimensional direction available early and broadly; moving activations
  along that direction flips which answer family is produced without
  destroying digit information; `SGN` at the sign position reads the resolved
  borrow cascade plus a [GT](thor-glossary.md#gt)-like comparison.
- **Falsifier**: Shared nodes hold near-orthogonal, disjoint add and subtract
  subspaces — parallel circuits multiplexed only at the output.
- **Tension with human**: Direct tension with C2's "different sub-tasks are
  more orthogonal" for the `SA`-vs-`MD`/`ND` pair on shared mixed-model nodes
  — I predict the opposite there.
- **Confidence**: medium.

### A8: No superposition pressure: complexity is compositional, not superposed

- **Belief**: The feature inventory is small relative to model width, so
  features occupy dedicated, weakly interfering subspaces and the total
  activation variation across the task has low effective dimension. Where the
  geometry looks rich, it will be *compositional* (template ⊗ position;
  circle plus magnitude), not superposed or irreducibly curved.
- **Why**: Superposition is an adaptation to feature counts exceeding capacity
  under sparsity; neither condition holds here. Fully grokked algorithmic
  models tend to be crisp and enumerable. The polysemanticity seen in mixed
  models arises from reuse (A7), not capacity pressure.
- **Support**: Elhage et al. 2022 (superposition emerges under capacity
  pressure); Gurnee et al. 2023 (monosemanticity when capacity allows); Nanda
  et al. 2023 (crisp circuits in grokked models).
- **Prediction**: Activation spectra across the task distribution show sharp
  low-rank structure; unrelated sub-task readouts are near-orthogonal;
  dictionary-learning-style decompositions roughly recover the known feature
  inventory (per-digit `SA`/`ST`/`SV`, operator, sign) without heavy feature
  splitting; curvature beyond the digit circle is minimal.
- **Falsifier**: Substantial interference between unrelated features, or high
  effective dimensionality / curvature not accounted for by
  circle-plus-position structure.
- **Tension with human**: Agrees with C1's spirit and supplies the mechanism
  (no capacity pressure); bounds it via A1/A3 (simple ≠ strictly linear).
- **Confidence**: PARTIALLY CHALLENGED 2026-07-14. The digit-embedding
  low-rank prediction is **refuted at the embedding level** (participation ratio
  ≈ 8.7/9 — near-isotropic, not sharp low-rank) for both trained *and* untrained
  models, so this is partly a property of 10 points in high-dim, not of the
  trained features. The core A8 claim is about *activation* effective dimension
  across the task (agenda entry 7), which remains **untouched**. Net: was
  medium-high; hold medium-high for the activation claim, but note the embedding
  matrix is not itself low-rank.

## Sharpest forks

Where discriminating evidence would most cheaply reshape this file (ranking
itself belongs in [maths-next-steps.md](maths-next-steps.md)):

1. **Geometry vs lookup (A1)** — decides the vocabulary the whole "how" story
   is written in; every other conjecture reads differently under each branch.
2. **Where discretization happens (A2)** — pair-sum sufficiency before the MLP
   is a crisp yes/no that also explains the known head-plus-MLP joint
   necessity.
3. **`U`'s shape (A3)** — scalar vs two-bit square vs simplex: three discrete
   geometries with different downstream readings.
4. **Carried vs wide-fetch cascade (A6)** — distinguishes rival answers to the
   human's propagation question.

## Supporting literature

Internal anchors:

- Paper 1 — Quirke & Barez, *Understanding Addition in Transformers*,
  [arXiv:2310.13121](https://arxiv.org/abs/2310.13121).
- Paper 2 — *Understanding Addition and Subtraction in Transformers*,
  [arXiv:2402.02619](https://arxiv.org/abs/2402.02619).

External:

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits*,
  [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html).
- Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value Memories*,
  [arXiv:2012.14913](https://arxiv.org/abs/2012.14913).
- Elhage et al. 2022, *Toy Models of Superposition*,
  [arXiv:2209.10652](https://arxiv.org/abs/2209.10652).
- Wang et al. 2022, *Interpretability in the Wild: a Circuit for Indirect
  Object Identification in GPT-2 small*,
  [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).
- Nanda et al. 2023, *Progress measures for grokking via mechanistic
  interpretability*, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217).
- Hanna et al. 2023, *How does GPT-2 compute greater-than?*,
  [arXiv:2305.00586](https://arxiv.org/abs/2305.00586).
- Gurnee et al. 2023, *Finding Neurons in a Haystack: Case Studies with Sparse
  Probing*, [arXiv:2305.01610](https://arxiv.org/abs/2305.01610).
- Stolfo et al. 2023, *A Mechanistic Interpretation of Arithmetic Reasoning in
  Language Models using Causal Mediation Analysis*,
  [arXiv:2305.15054](https://arxiv.org/abs/2305.15054).
- Merullo et al. 2023, *Language Models Implement Simple Word2Vec-style Vector
  Arithmetic*, [arXiv:2305.16130](https://arxiv.org/abs/2305.16130).
- Zhong et al. 2023, *The Clock and the Pizza: Two Stories in Mechanistic
  Explanation of Neural Networks*,
  [arXiv:2306.17844](https://arxiv.org/abs/2306.17844).
- Todd et al. 2023, *Function Vectors in Large Language Models*,
  [arXiv:2310.15213](https://arxiv.org/abs/2310.15213).
- Feng & Steinhardt 2023, *How do Language Models Bind Entities in Context?*,
  [arXiv:2310.17191](https://arxiv.org/abs/2310.17191).
- Park et al. 2023, *The Linear Representation Hypothesis and the Geometry of
  Large Language Models*, [arXiv:2311.03658](https://arxiv.org/abs/2311.03658).
- Engels et al. 2024, *Not All Language Model Features Are Linear*,
  [arXiv:2405.14860](https://arxiv.org/abs/2405.14860).
- Park et al. 2024, *The Geometry of Categorical and Hierarchical Concepts in
  Large Language Models*, [arXiv:2406.01506](https://arxiv.org/abs/2406.01506).
- Zhou et al. 2024, *Pre-trained Large Language Models Use Fourier Features to
  Compute Addition*, [arXiv:2406.03445](https://arxiv.org/abs/2406.03445).
- Levy & Geva 2024, *Language Models Encode Numbers Using Digit
  Representations in Base 10*,
  [arXiv:2410.11781](https://arxiv.org/abs/2410.11781).
- Nikankin et al. 2024, *Arithmetic Without Algorithms: Language Models Solve
  Math With a Bag of Heuristics*,
  [arXiv:2410.21272](https://arxiv.org/abs/2410.21272).
- Kantamneni & Tegmark 2025, *Language Models Use Trigonometry to Do
  Addition*, [arXiv:2502.00873](https://arxiv.org/abs/2502.00873).
- Anthropic 2025, *On the Biology of a Large Language Model* (addition case
  study: parallel lookup and magnitude paths),
  [transformer-circuits.pub](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

## Reflection log (optional)

- **2026-07-14** — Initial population, at the human's request, from priors and
  external literature before any experiments in this thread. Contains no
  empirical content. The "update only after the post-result skeptic gate"
  rule applies from the first study onward.
- **2026-07-14** — After the digit-embedding geometry study (Gate 2 passed;
  [study note](study-maths/study-digit-embedding-geometry.md),
  [CE1](maths-claim-evidence.md)): lowered **A1** (strong dominant-geometry form
  refuted at the token embedding; weak circular *ordering* survives,
  provisional pending LN-aware A-9) and annotated **A8** (embedding matrix is
  near-isotropic, not low-rank — but that is partly true of the untrained model
  too; the activation-level A8 claim is untouched). Stage 1 of the overall
  picture annotated. Prediction scoring lives in the study note; these are the
  post-gate belief updates. No change to A2–A7 (untouched by a weights-only
  study). Reframe: the "how it computes" story now hinges on the
  activation/MLP stages and a causal test, not on a clean embedding geometry.
- **2026-07-14** — After the pair-sum sufficiency study (Gate 2 returned BLOCK,
  resolved; `instrument failure`;
  [study note](study-maths/study-pair-sum-sufficiency.md),
  [CE2](maths-claim-evidence.md)): **A2 confidence HELD** (not lowered) — the
  assay mis-targeted (answer-position heads, not confirmed `ST` nodes) and its
  metric could not separate aggregation from transport, so A2 is untested. Added
  the value-path transport by-product as CE2. No change to A3 (deferred to the
  re-scoped entry 2) or C3 (MLP-transform half untested). Method lesson recorded
  in the results synthesis: tidy low-D structures keep being low-variance
  projections; require nulls + independent node-role confirmation.
- **2026-07-16** — After the attention-invariance census (Gate 2 PASS WITH
  CONDITIONS; [study note](study-maths/study-attention-invariance.md),
  [CE8](maths-claim-evidence.md)): **A5's strong static-wiring form falsified /
  narrowed to hybrid** (lowered to medium) — a few L1 heads (+one L0) relocate
  their attention target with the carry state (value-matched contrast, clean in
  6-digit, Bonferroni-safe; A5's own falsifier). This **breaks the recent
  A3-family refuted/untouched streak** with a genuine new structural fact — but
  a narrow one (6-digit only; representational, not causal). **C3** routing half
  supported (the content-dependence *refines A5*, not new C3 — human-owned,
  noted). **A6 untouched** (no cascade patching; the routing cells are candidate
  nodes for the entry-3 causal test, not an A6 test). Gate-2 correctly stopped
  two over-reaches: scoring the noisy 5-digit as a "replication" (it is
  inconclusive), and the "answer-position/CE5-combiner-locus" gloss (the cleanest
  cell is an operand-read position, which actually strengthens the finding by
  rebutting the trivial "answer-digit-depends-on-carry" reading). Method: only a
  value-matched target-move falsifies A5 — pattern-variance is A5-consistent
  because a static head still has value-dependent softmax weights.
- **2026-07-16** — After the LN-aware digit-embedding close-out (Gate 2 PASS
  WITH CONDITIONS; [study note](study-maths/study-ln-aware-embedding.md),
  revises [CE1](maths-claim-evidence.md)): the CE1 circular-**ordering** signal
  is **de-provisionalized** — it persists in the LN-effective geometry, but only
  *weakly* (LN is near-isometric here, γ std ~0.005) and it is **seed-fragile**
  (2 of 3 independent seeds; fails s173289). A1's ordering sub-claim scored
  "weak ordering, LN-robust modestly, 2/3 seeds — not confirmed-strong"; C1
  weakly/narrowly supported (near-isometric LN adds no curvature — low-info).
  Evidence integrity clean (raw arm reproduced CE1 bit-for-bit in the committed
  script). The embedding-geometry line is now closed at the representational
  level; the only deeper embedding question is *causal* (B1). Gate-2 caught a
  headline over-read (near-isometry makes "LN-robust" weak) and a replication
  over-count (model-count vs seed-count) — both corrected.
- **2026-07-16** — After the earliest-tri-state-site study (Gate 2 PASS WITH
  CONDITIONS; [study note](study-maths/study-earliest-tristate-site.md),
  [CE7](maths-claim-evidence.md)): **A3 off-axis-third-symbol form refuted across
  all answer-position residual sites** (dropped to low); the binary resolution is
  applied around L1-attention, never a dedicated `U` symbol. C1 narrowly
  supported (binary linear carry throughout — human-owned, noted). A6/A2
  untouched. **STREAK SIGNAL (Evidence Rules)**: this is the *fourth consecutive*
  A3-family refuted/untouched result (pair-sum instrument-failure → U-resolution
  ambiguous → CE6 combiner refuted → CE7 all-sites refuted). The "U as a distinct
  tri-state symbol" conceptual frame is **stale** at answer positions; the
  productive next moves are the *question-position* transient-U probe and the
  *multi-digit `...999` cascade* (A6), NOT more answer-position geometry sweeps.
  The Gate-1 lesson recurred usefully: the `U ≡ SA_n=9` identity + a
  metric-vs-hypothesis mismatch (centroid distance vs the is-U subspace) had to
  be fixed before the assay could discriminate — the axis-decomposition
  discriminator was the fix.
- **2026-07-15** — After the tri-state-geometry study (Gate 2 BLOCK, resolved;
  [study note](study-maths/study-st-tristate-geometry.md),
  [CE6](maths-claim-evidence.md)): **A3 off-axis-third-symbol form refuted at the
  combiner input** (locus-scoped; `U` on the 0–1 axis, split by resolution),
  lowered to low-medium; A3's live remnant is a possible tri-state at an earlier
  (untested) site. **C1** narrowly supported at this locus (clean binary linear
  carry) — human-owned, noted not edited. **A2/A6 untouched** (the assay can't
  show the U-decision is upstream vs in the MLP — `carry_in` ingredients are
  linearly present at the input; consistent with CE5). Gate-2 fixed an
  evidence-integrity gap (headline numbers now reproducible in the script) and
  the "resolution is upstream / no-L0-attribution" over-reach. Method lesson: a
  site linearly separating an outcome ≠ that site computing it (ingredients vs
  decision), and `U ≡ SA_n=9` is a fatal confound the resolution variable breaks.
- **2026-07-15** — After the combiner-vs-conduit study (Gate 2 PASS WITH
  CONDITIONS — first pass after three BLOCKs;
  [study note](study-maths/study-combiner-vs-conduit.md),
  [CE5](maths-claim-evidence.md)): **A6 carried+combined sub-claim supported**
  (L0 conduits + answer-position **L1 MLP combiner** resolves `U`;
  tie-break-economy still untested); **A2 supported at the U-combine step** (the
  nonlinear carry-class combination is in the L1 MLP output; pre-MLP
  sum-sufficiency still untested); **A5 partial** (U branch in the MLP); **A3
  anchored** (geometry not measured; combiner locus `P14/P16.L1.MLP` now
  confirmed for entry 2). The working discriminator was node-level activation
  invariance (definite-invariant / U-variant), the endpoint-independent fix for
  the definite-digit carry-independence trap that vacuated the prior gate.
- **2026-07-15** — After the U-resolution study (Gate 2 returned BLOCK,
  resolved; positive-but-ambiguous;
  [study note](study-maths/study-u-resolution-path.md),
  [CE4](maths-claim-evidence.md)): **no conjecture updated** — A6/A2/A5/A3 all
  scored **untouched**. The study located an MLP-heavy L0/L1 path that
  *transmits* the tri-state U-flip and is distinct from the CE3 make-carry heads
  (reconfirming "separate paths" with named nodes), but its pre-registered
  combiner discriminator was **vacuous** (a definite digit's `A_{n+1}` has no
  lower-carry dependence, so the trivial readout also passed), so it cannot tell
  a U-combiner from a carry-conduit. Method lesson: an interaction gate needs a
  control arm with a signal a non-target node could transmit; use a *corruption*
  control next. This tempers the temptation to score A6 "carried state
  supported" — a question-position MLP transmitting is exactly what transport
  looks like.
- **2026-07-14** — After the confirm-ST-node study (Gate 2 returned BLOCK,
  resolved; [study note](study-maths/study-confirm-st-node.md),
  [CE3](maths-claim-evidence.md)): **A5 partial support** (confirmed carry heads
  are static-position, question-independent-attention, value-path-causal).
  **A6 untouched** (the `=`-resolution sub-claim could not be scored — the `=`
  patch was layer-0-only and autoregressively confounded). **A2** now has a
  better anchor (a causally-confirmed carry node) but it is `SC`-grade, not the
  tri-state `ST` node A2 ultimately concerns. **New incidental finding** (candidate
  for a future conjecture): the *binary make-carry* and the *tri-state
  `U`-resolution* are computed by **different paths** — the make-carry head does
  not transmit U-resolution, yet the model resolves `U` correctly. This splits
  A2/A3's implicit "one ST node does it all" picture and makes *locating the
  U-resolution path* the top mechanistic question.
