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
   why ablating either the head or its MLP breaks the node. *[2026-07-15:
   partially revised — the discretization target is not a stored tri-state: no
   dedicated `{0,1,U}` symbol exists anywhere along the answer-position stream
   (CE6/CE7); the nonlinear carry-class combination is real but lives in the
   answer-position L1 MLP (CE5); pre-MLP sum-sufficiency remains untested (CE2
   instrument failure).]*
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
   resolved by the `=` token. *[2026-07-15: revised — routing is hybrid, not
   fully static (CE8: a few L1/L0 heads relocate targets with carry state);
   single-digit `U` is resolved at the answer position around L1-attention
   (CE5/CE7), not at `=`; and the human C4 leading-digit argument means
   deep-cascade resolution must be complete by the answer-sign position (by
   `=` for the mixed model's sign). How deep chains resolve is now the top
   fork: sequential carried state (A6, human lean) vs one-hop selection
   (A9).]*
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
*[2026-07-15: the propagation clause is the part now most in doubt — routing
is mostly-static with carry-routed exceptions (CE8), and whether any cascade
state is carried at all (vs per-digit bits fetched on demand, A9) is the top
open fork.]*

*[2026-07-15 (C5): the paper's per-model verified maps name the physical
instantiation of stages 2–5 for both studied models: `ST` at question-tail
and sign-token L0 heads, `SA`/`SC` at answer-position L0 heads, and the
compounding at SP-tagged answer-position L1 heads (attending `=` + the ST
sites) feeding the answer-position L1 MLPs. The open questions are the
*content and combination rule on those named wires* — C5 steps 1–5 — not the
location of the nodes, which this thread wastefully re-derived. See A10.]*

*[2026-07-16 (post CE13–CE15, working-axioms mode): current best account of
stages 2–5, now medium-high at the wiring level — question-tail/sign L0 ST
nodes write their tri-state class plus single-step local U-resolution (CE13;
the map-named `SA` L0 heads do NOT write the answer digit — the digit is
computed at its answer position, CE12/CE13); a redundant SP-tagged L1 head
pair at each answer position, including the sign position for the leading
digit, delivers the carry carry-specifically into the answer-position L1-MLP
combiner (CE14/CE15), which emits the resolved carry_out (CE5). Open:
the edge message's content, its source (`=` depot vs deciding ST site),
path shares/class necessity, and the combiner's functional form — see A10's
consolidation and the sprint agenda.]*

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
- **C4 (reflections on studies #1–#7)** — two points accepted, one lean
  contested. Accepted: (1) every `U` result so far (CE4–CE7) is single-digit
  `U` at a *middle* answer digit, where `carry_in` is one make-carry bit — so
  CE5's combiner evidence says nothing about deep `...999` chains, and reading
  it as "the whole `U` mechanism" would indeed be wrong; (2) the leading
  answer digit is predicted from the answer-sign position (and the mixed
  model's sign from `=`), so the *full* cascade must be resolved at or before
  that position — a locus none of the nine studies probed. Contested: the
  human lean (recorded 2026-07-15) is that the missing piece is the paper's
  *sequential* TriAdd cascade riding the residual stream. I think a 2-layer
  model cannot iterate a chain sequentially, and the deep carry is available
  in one hop as a *selection* — fetch the deciding digit, the highest lower
  digit with pair-sum ≠ 9; CE7 (resolution applied around L1-attention) and
  CE8 (carry-state target-relocation at a few heads) are circumstantially
  consistent with selection. Formalized as A9; the deciding-digit patch
  discriminates the two leans cheaply (agenda entry 1). I also endorse C4's
  "combining 8 cascading `ST` values in one MLP is implausible" — that
  implausibility cuts against *both* the naive-combiner reading and the
  wide-fetch alternative, which is part of why I lean selection.
- **C5 (the paper's empirical results are reliable; build on them)** —
  **accepted, with a self-correction.** The per-model verified node maps on
  Hugging Face (`<model>/behaviors.json` + `features.json`: ablation `Fail%`,
  per-answer-digit `Impact`, attention targets, `Algo:` role assignments, `SP`
  tri-state-PCA tags) are the anchor this thread should have started from and
  mostly did not — studies #2–#13 re-located nodes from scratch with narrower
  stimuli, effectively re-deriving (parts of) the paper's map at high cost.
  Worse: the confirm-ST-node study dismissed the paper's question-position ST
  candidates using a **no-lower-carry stimulus — the one regime where ST's
  cascade role cannot matter** — while the maps (now read for both studied
  models) show those nodes failing 14–23% of *random* questions with
  multi-digit `Impact` (5-digit `P11L0H2` = A0.ST, Impact A5..A1; 6-digit
  `P12L0H1` = A1.ST, Impact A6..A2). CE3's "P8/P9/P11 not causal" reading is
  therefore suspect and must be formally re-examined (next gated study) before
  it is leaned on again. The maps also *already name the candidate SV wiring*
  the last four studies groped toward: at every answer position an SP-tagged
  L1 head attends `=` plus the question-tail ST sites and feeds the
  high-`Fail%` L1 MLP — CE5/CE8/CE10's protagonists, rediscovered. Where I
  still push back (mildly, and in agreement with C5's own "incomplete"
  caveat): the map is an **anchor, not a ceiling** — ablation-based `Impact`
  misses redundant nodes (the 5-digit map lists no A2.SC node; CE3's
  interchange found `P14L0H0` doing exactly that job at flip 1.00), so
  interchange/path methods complement the map rather than merely re-verify
  it. A10 operationalizes C5's five-step program; the agenda is reworked
  around it.

## Working axioms

Adopted 2026-07-16 after the human's over-caution feedback ("experiments seem
framed to disprove things I already consider proved"). These are standing
priors that experiments must *assume*, not re-test; a study framed as an
existence test of any of them is mis-framed.

1. **Existence**: the accurate models compute deep `...999` cascades correctly
   (paper: >99.999% on curated sets including cascades; this thread: clean
   hi/lo behavioral separation 40/40 at every depth tested, both models). A
   working SV implementation therefore **exists** in each accurate model. The
   empirical question is always *which* candidate implementation, on *which
   wires*, with *what shares* — attribution and estimation — never "whether".
   Verdicts state the best current account with per-link confidence; "not
   shown" is a statement about an open parameter, not a null result.
2. **Redundancy is the norm**: Paper 2 and CE13/CE14 document duplicated
   nodes/heads as standard in these models. A single-node necessity failure is
   the *expected* outcome and is not evidence against a mechanism. The
   evidential standard is sufficiency + specificity (matched nulls) +
   replication, with necessity tested at the **class** level (paired/grouped
   ablation), not per node.
3. **The map is trusted wiring** (C5): the paper's per-model node maps are
   verified anchors; interchange complements them (fills redundancy gaps) but
   does not overturn them without map-consistent stimuli.

Corollary for verdict style: single studies stay calibrated to their own
scope, but this file may **aggregate across studies** — convergent
sufficiency evidence from independent assays is itself evidence, and holding
every claim at the weakest single-study verdict is the over-caution failure
mode.

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
  structure, but does not bear on A2's ST-node claim. **Update 2026-07-15
  (CE5, CE6/CE7):** the *discretize* half is now supported at the `U`-combine
  step — the nonlinear carry-class combination is localized to the
  answer-position L1 MLP ([CE5](maths-claim-evidence.md)). The *aggregate*
  half (pre-MLP sum-sufficiency at a question-position compute node) remains
  untested. One prediction revision: the discretization target at answer
  positions is the **binary carry** (plus the answer digit), not a stored
  `{0,1,U}` tri-state — no dedicated `U` symbol appears anywhere along the
  answer-position stream (CE6/CE7), so "the MLP snaps the arc into the `ST`
  tri-state" should read "into the resolved binary classes".

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
  strongly disfavored (see streak note in the reflection log). **2026-07-15
  note:** the human C4 leading-digit argument independently motivates the
  question-position remnant (B12): the top answer digit is predicted from the
  answer-sign position, so whatever carry information feeds it must already
  exist at or before that position — a deep `...999` chain at question
  positions is the one regime where a transient `U`-like state would earn its
  keep (though A9 predicts even there it is skipped by selection).

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
- **Confidence**: **template-sharing sub-claim LOWERED 2026-07-16** by the
  probe-transfer study ([CE11](maths-claim-evidence.md)): for the tri-state carry
  `ST` at question positions, a probe trained at digit `i` **does not transfer** to
  digit `j` (retention 0.10/0.12 ≪ 0.6, both models), and mean-centering does not
  restore it — so **A4's "transfer-for-free" prediction is falsified for `ST`**, and
  not even A4's per-position-offset form holds. The representation is
  position-specific at the question site despite architectural weight sharing. A4's
  positional-binding *spirit* (position is decodable/where-based) is not
  contradicted, but its concrete transfer claim fails. Was medium-high on template
  sharing → **low-medium** (question-position `ST`); the just-in-time-fetch half is
  now the sharp open test at the **answer** position (B4). SA/SV template sharing at
  their own home sites remains untested (SA lives at the answer position, CE2/CE3).
  **Update 2026-07-16 (CE12, answer-binding):** the answer-phase binding was tested.
  **Just-in-time fetch is SUPPORTED for `SA`** (absent at `=`, present only at its
  own answer position) and the **orthogonal-tape alternative is REFUTED for `SV`**
  (present at `=` but per-digit slots not orthogonal). And — unlike the
  question-side `ST` — `SA`/`SV` **transfer across answer positions** (a shared
  answer-side template), so A4's template claim is **position-of-computation
  dependent** (fails question-side, holds answer-side). Net: **raise the
  just-in-time-fetch sub-claim to medium-high**; the "orthogonal per-digit slots at
  `=`" tape alternative A4 itself named is refuted; no *stored*-cascade claim (A6/C3
  untouched — SV presence at `=` is CE7-consistent resolution, not shown stored).

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
  pattern-patching test (cascade-tracing entry). **2026-07-15 note:** under A9
  those routing cells stop being a curiosity and become the *mechanism of
  deep-cascade resolution* (the target relocation = picking the deciding
  digit); the causal pattern-patching test doubles as an A9 test. **Update
  2026-07-16 (CE9): the causal pattern-patch ran, and the CE8 `L1.H1` routing
  cell is causally INERT under a single-head redirect (0.00 answer-move) — so
  CE8's routing stays *representational*, not shown load-bearing.** Hold A5 at
  medium (the representational hybrid-routing finding stands; its causal role is
  not established at this granularity — an edge path-patch is the open test).
  **Update 2026-07-16 (CE10):** the edge path-patch shows `L1.H1` **is causally
  load-bearing at the combiner edge at one depth** (6-digit k=3) — so the CE8
  routing cell has a *causal* role after all, but only a one-depth crumb (the edge
  instrument is underpowered at most cells). Note (do not raise) A5: hybrid routing
  is causal at `L1.H1`/k=3, pending a less-damped multi-depth confirmation.

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
- **Alternatives**: (a) "wide fetch": one late MLP reads all lower digit
  pairs at once and computes the whole cascade in a single nonlinear step;
  (b) **selection (A9)**: an attention head fetches the single *deciding*
  digit's carry bit — no sequential state and no big nonlinear combine.
  Shallow models permit both within the ordering constraints, so both stay
  live.
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
  **Scoping update 2026-07-15 (human C4):** the CE5 support covers only
  single-digit `U` at a middle answer digit, where `carry_in` is one
  make-carry bit — evidence that cannot distinguish "the cascade rides the
  stream" from "there is no cascade to ride". The load-bearing untested case
  is the deep `...999` chain, plus the leading-digit constraint (full
  resolution must be available at the answer-sign position; at `=` for the
  mixed model's sign). Hold medium-high on the single-digit carried+combined
  core, but the multi-digit mechanism — sequential carried state (human lean)
  vs selection (A9) vs wide fetch — is now the thread's top fork and wholly
  untested. **Update 2026-07-16 (CE9, deep-cascade study):** the deep-chain fork
  was tested at node/pattern granularity and is **not resolvable there** — the
  **sequential-accumulated** form is *disfavored where testable* (no
  intermediate-digit identity carried in the cascade state at the one controlled
  locus, 5-digit `=` k=3) but not refuted (untestable in 6-digit); the tie-break
  economy sub-claim remains untested. So A6's deep-chain extrapolation is
  **weakened but open**; the fork awaits an edge path-patch (B11).

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
  matrix is not itself low-rank. **Interference half LOWERED 2026-07-16**
  ([CE11](maths-claim-evidence.md)): the probe-transfer study found `ST` and `SV`
  are **geometrically entangled** at question positions (principal angle 21° ≪
  64°/50° label-correlation null, both models) despite being label-independent at
  the same digit — i.e. the carry-family sub-tasks share overlapping directions
  rather than occupying near-orthogonal dedicated subspaces. This challenges A8's
  "unrelated sub-task readouts are near-orthogonal / low interference" prediction
  for the carry family at question positions. The task-wide effective-dimension
  claim (agenda entry 2) is still the untouched core test.

### A9: Deep `U`-cascades are resolved by one-hop attention selection, not sequential propagation

- **Belief**: For a carry chain of any depth, the resolved carry into digit
  `n+1` equals the make-carry bit of the **deciding digit** — the highest
  digit `m ≤ n` with `Dm + D'm ≠ 9` (0 if there is none). The model
  implements this as a *selection*: at the position that needs the carry
  (each answer position; the answer-sign position for the leading digit), an
  L1 attention head relocates its target to the deciding digit's stored
  information (make-carry / L0-conduit outputs) and delivers that one bit;
  the L1-MLP combiner (CE5) combines it with the local sum-class. No
  `{0,1,U}` symbol is ever stored and no carry state propagates
  token-to-token — the paper's "cascade" is a *functional* description of
  this selection, not a physical process.
- **Why**: (1) Depth: a 2-layer model cannot iterate `TriAdd` over even a
  5-deep chain sequentially across layers, and asking one MLP to combine
  many cascading `ST` values (the human C4 objection) is equally
  implausible — but "first non-9 below me" is exactly the kind of
  content-dependent selection attention does in one hop. (2) Parsimony with
  the negative streak: selection explains at once why no dedicated `U`
  symbol exists at any answer-position site (CE6/CE7 — nothing needs
  deferring if the deciding bit is fetched directly), why the binary
  resolution appears exactly at L1-attention (CE7's trajectory), and why a
  few L1 heads relocate targets with carry state while most routing stays
  static (CE8 / hybrid A5).
- **Support** (circumstantial, none causal): CE7 — `U→1` flips to the
  committed-1 side only at `L1.resid_mid`, i.e. the resolution is *applied*
  by L1-attention; CE8 — value-matched carry-state target-relocation at
  `L1.H1` (operand-read Q11, answer Q14) and `L0.H0` (Q17), clean in the
  6-digit model; CE5 — the combiner needs only `carry_in` + local class;
  CE3 — per-digit binary make-carry bits exist as selectable sources. Paper
  2's own fallback observation (`P14` recomputes `ST1` and consults
  `P10.ST2` only when they differ) already reads like conditional fetching.
- **Prediction**: (1) **Deciding-digit patch**: in a `...999`-chain question,
  patching the deciding digit's make-carry information flips *all* cascade
  answer digits above it at once, while patching an intermediate all-9s
  digit's nodes does ~nothing; a sequential cascade predicts the opposite
  (patching an intermediate link breaks everything downstream of it).
  (2) **Target tracking**: the CE8 routing heads' attention targets track
  the deciding digit's *position* as chain depth is varied. (3) **Depth
  invariance**: accuracy and mechanism are ~flat in chain depth up to
  attention precision, and errors look like selection errors (fetching the
  wrong digit), not accumulation errors. (4) The same selection signature
  appears at the answer-sign position for the leading digit
  (`99999+00001`-type questions).
- **Falsifier**: patching an intermediate `9`-digit's question-position
  nodes in a long chain breaks higher answer digits (stepwise dependence =
  genuine sequential carried state); or routing-head targets do not track
  the deciding digit; or the combiner input on deep chains carries
  multi-digit `ST` information beyond the one deciding bit.
- **Alternatives**: (a) the paper's sequential `TriAdd` cascade across token
  positions (the human's recorded C4 lean) — would require each question
  position's L0 to wide-fetch all lower pairs, or state to hop layer-by-layer;
  (b) nonlinear wide fetch: one MLP reads all lower pairs at once (A6's
  original alternative); (c) hybrid by depth: selection for short chains,
  memorized patches for rare deep ones (deep cascades are exponentially rare
  in random training data); (d) per-model idiosyncrasy (Paper 2 documents
  node-level variability).
- **Tension with human**: Direct — C3/C4 read the cascade as sequential
  carried state riding the residual stream and resolved by `=`; A9 says
  there is no ridden state at all, only stored per-digit bits plus a
  data-dependent fetch. C4's own leading-digit argument is *accepted* and is
  part of what forces a mechanism like this.
- **Confidence**: **held at low-medium 2026-07-16** after the deep-cascade
  mechanism study ([CE9](maths-claim-evidence.md)). The deciding-digit patch test
  ran, but at node/attention-pattern granularity it is **underpowered to confirm
  A9**: the two signatures A9 predicts to co-occur on one head land on *different*
  cells (6-digit: a causally deciding-selective consumer head `L1.H0` Q14, and a
  separate deciding-digit-tracking head `L1.H2` Q15), and the CE8 routing cell
  `L1.H1` is causally **inert** (single-head pattern-redirect moves 0.00). So the
  *ingredients* A9 needs exist (selective causal head; tracking head; no sequential
  per-digit state carried where testable) but A9's specific single-cell selection
  mechanism is **not demonstrated** — neither confirmed nor refuted. The
  discriminating test is now an **edge path-patch** (candidate L1-head→L1-MLP-
  combiner edge), not a single-head redirect (too blunt). Two skeptic rounds were
  needed here: the first over-claimed A9 *refuted*, the second over-claimed A9
  *confirmed via `L1.H1`* on a value-matched metric that had not actually been
  implemented; the honest read is *pieces-present-convergence-unshown*. Still the
  thread's top fork; the edge path-patch (B11) is promoted to the confirming test.
  **Update 2026-07-16 (CE10, edge path-patch ran):** a **one-depth causal crumb,
  not a confirmation** — at 6-digit k=3, `L1.H1` (the CE8 routing cell CE9 found
  inert) **causally drives the combiner** with a computed, deciding-selective carry
  through the combiner MLP input (dec 1.00 / same-class null 0.00 / wrong 0.00).
  But the single-position edge instrument is **underpowered at most cells** (6/9
  6-digit; LN renormalizes a single head's edge against the whole residual), no head
  clears the ≥2-depth bar, and 5-digit has a **live direct residual path** — so A9's
  attention-delivery is causally supported *at one cell* but **held at low-medium**
  (not confirmed), and A6's residual-carry is **not refuted**. (Gate 2 round 1
  BLOCKed an inverted-power-control over-claim of "A9 confirmed / A6 refuted"; this
  crumb is the corrected read.) Follow-up: a less LN-damped / multi-position edge
  instrument + a second depth (6-digit k=4). **2026-07-15 map note (C5):** the
  paper's verified maps show the consumer L1 heads attending a *set* of ST
  sites plus `=` with moderate mass (e.g. 6-digit `P16L1H1`: P16=32, P13=24,
  P12=17, P11=13), so A9's "relocate to one deciding target" sharpens to
  **selection *within* the fetched ST-site cluster** (A10 variant b); the rival
  is a static weighted read over the cluster with the MLP arbitrating (A10
  variant c). Same fork, now scoped to named nodes. **2026-07-16
  (deprioritized):** after two attempts (CE9, CE14) selection remains unshown —
  the tracking head and the edge-causal head keep landing on different cells.
  Under the working axioms the SV *account* does not need this variant settled
  (delivery is established either way); A9 is **parked** unless the sprint's
  edge-message/source decode resolves it as a by-product (a message read off
  the deciding ST site would revive it; a message read off `=` would retire it
  toward the human/paper depot reading).

### A10: The SV compounding mechanism is the map-named answer-position L1 fetch-and-combine over the question-tail ST cluster

- **Belief**: For each answer digit `A_{n+1}`: the tri-state `ST` values are
  computed and written by the map-named L0 `ST` nodes clustered at the
  question-tail `D'` positions and the sign token (5-digit P6/P9/P10/P11/P12;
  6-digit P10–P12/P14, per the HF `behaviors.json`/`features.json` for the
  two studied models). The **compounding (`SV`) step** — C5 steps 2–3 — is
  performed at the answer position by the **SP-tagged L1 head(s)**, which
  attend `=` plus several ST sites and deliver carry information into the
  high-`Fail%` answer-position **L1 MLP**, which combines it with the local
  `SA`/`SC` information to emit the digit (C5 step 4). The **leading digit**
  (C5 step 5) is the same mechanism executed at the sign position (5-digit
  `P12L1H0`/`P12L1H2`; 6-digit `P14L1H0`/`P14L1H1`), whose ST inputs include
  the sign-token L0 ST nodes (A4/A5.ST).
- **Why**: This is what the verified map says when read as wiring: ST nodes
  carry multi-digit `Impact` (their outputs feed all higher answers — the
  compounding structure C4 found missing from our studies); the only useful
  L1 heads at answer positions carry `SP` (tri-state-PCA) feature tags and
  attend exactly {`=`, ST-site cluster}; the L1 MLPs are the highest-`Fail%`
  answer-position nodes; and CE5/CE7/CE10 independently landed on the same
  L1-attention→MLP locus from the activation side without knowing the map.
- **Support**: the HF per-model maps (backlink:
  [hugging_models.md](hugging_models.md)); CE5 (combiner), CE7 (resolution
  applied around L1-attention), CE10 (one-depth causal crumb on the
  `L1.H1`→MLP edge at the map-named head), CE8 (carry-state target moves at
  these heads); Paper 2's ordering constraints.
- **Prediction**: (1) **Value content**: the SP-tagged L1 heads' value inputs
  at the attended ST sites carry the tri-state/carry information (decodable
  there, and transported by the head's OV path into the combiner input).
  (2) **Deciding-digit propagation**: in a deep `...999` chain, patching the
  map-named ST node of the *deciding* digit moves exactly its `Impact`-tagged
  answer digits, and the change reaches the combiner through the SP-head edge
  (extends CE10 beyond one depth). (3) **Selective economy**: ablating an
  SP-tagged L1 head harms cascade questions and spares carry-free ones (A6's
  economy, tested at the named node). (4) The heads' large `=`-token mass
  either carries necessary content or is shown to be a sink — either way
  resolving what `=` contributes (the maps list no useful nodes at `=`, yet
  every consumer attends it).
- **Falsifier**: the SP heads' value path carries no tri-state/carry content
  (their ST-site attention is incidental); or deep-chain ST-node patches
  bypass the SP-head edge entirely (carry arrives only via the direct
  residual path — then the L1 "fetch" is a red herring and the compounding
  is elsewhere, e.g. already accumulated at the ST nodes' own outputs).
- **Alternatives**: (a) **direct-path compounding**: the answer-position
  residual carries the compound carry without the L1 heads (5-digit CE10
  hints a live direct path); (b) **selection-within-cluster** (A9's sharpened
  form): the SP head's mass relocates *within* the ST-site cluster to the
  deciding digit (CE8's moves); (c) **static weighted read** over the whole
  cluster with the MLP arbitrating. (b) vs (c) is the surviving
  A9-vs-wide-fetch question; (a) vs (b/c) is the surviving A6-vs-A9 question
  — all now scoped to named nodes.
- **Tension with human**: none on direction — this is C5's program stated as
  a mechanism conjecture. The human's sequential-cascade lean maps onto
  variants (a)/(c) (accumulated state, read statically); my selection lean
  onto (b).
- **Confidence**: medium — the wiring is the paper's verified map plus four
  convergent activation studies; the value-content, multi-depth-causal, and
  economy predictions are untested. **Update 2026-07-16 (CE14, SV-compounding, C5
  steps 2–4):** the **core "fetch-to-combiner" claim is now PARTIALLY confirmed
  causally** — patching the map-named answer-position L1 consumer heads' output edge
  into the L1-MLP combiner flips the top cascade digit at ≥ 2 depths,
  **carry-specifically** (deciding-matched null = 0.00) and consumer-head-specific,
  in both models (attention delivers, strengthening CE10's one-depth crumb). BUT the
  **selection variant (b)/A9 is NOT shown** (the value-matched tracking head `L1.H2`
  ≠ the single-depth-edge-causal head `L1.H1`); the effect is **sufficiency not
  necessity** (ablating H1 is harmless; H2 carries necessity+tracking → a redundant
  H1/H2 pair); the **direct path is not excluded** (underpowered arm); and the
  value-content prediction (1) is **not head-specific**. **Prediction (3) economy:
  supported at H2.** Net: **hold at medium** — core wiring partially confirmed
  (net positive within medium), selection sub-claim unsupported, delivery
  sufficient-not-necessary. Took THREE Gate-2 rounds (over-claim → over-correction
  on a broken specificity null → calibrated). The value-content, necessity
  (paired-ablation), selection (same-cell ≥2-depth on H2), and direct-path
  (powered arm) questions remain open. **Update 2026-07-16 (CE13, node-output-encoding,
  C5 step 1):** the premise that the map-named `ST` nodes are "output-only" writers
  of a *local* class is **REFINED, not confirmed**: the ST write cleanly encodes the
  class but on the `U` case also co-carries the **single-step** incoming carry
  (local U-resolution) — so some resolution is already co-located at the ST write,
  though this is *not* multi-digit compounding (single-step cin toggle; carry-out =
  cin by definition on `U`). The map's *causal* tags are **vindicated** under a
  baseline-controlled ablation (low-digit ST nodes load-bearing above an
  untagged-head baseline; high-digit redundant — a C5 win, matching the paper's
  "redundant ST node" note). Two Gate-2 rounds corrected over-reach in both
  directions (first "local write", then "compounding begins here"). The **L1
  fetch/combine** steps and the **multi-digit** compounding question remain
  **untested** — that is entry 2. Confidence held at medium; the "ST-local /
  L1-compounds" division is sharpened (some U-resolution is early), not settled.
  **Consolidation 2026-07-16 (CE13+CE14+CE15 aggregated; working-axioms mode):**
  across three studies, two models, ≥ 2 chain depths, and **every answer digit
  including the sign-position leading digit** (CE15), one account holds:
  map-named ST writes (local class + single-step U-resolution, CE13) → a
  **redundant SP-tagged L1 head pair** delivers the carry **carry-specifically**
  (deciding-matched null 0.00) through the head→combiner edge (CE14) → the
  combiner MLP emits resolved `carry_out` on the CE5 centroids → the answer
  digit is computed just-in-time at its answer position (CE12/CE13). Under
  axiom 1 (attribution, not existence) and axiom 2 (redundancy expected;
  sufficiency+specificity+replication is the standard), this cross-study
  convergence justifies **raising the core wiring claim to MEDIUM-HIGH** — the
  per-study "hold at medium" verdicts were each scoped to a single study, and
  aggregation is exactly the evidence they could not individually use. What
  remains open is **implementation detail, not wiring**: (i) the *message* on
  the causal edge (compound carry vs deciding-digit class vs U-flag); (ii) the
  *source* the heads read it from (`=` vs the deciding ST site vs distributed —
  CE12 found all-digit `SV` decodable at `=`, and every consumer head attends
  `=` heavily, so "`=` as carry depot with pre-L1 compounding" is live and is
  the human/paper lean); (iii) **path shares** (head-pair vs direct residual)
  and **class-level necessity** (paired H1+H2 ablation); (iv) the combiner's
  functional form (B2). These four are the 40-hour sprint targets (agenda
  entry 1).
- **Update 2026-07-16 (CE16, SV-implementation sprint; combined dual-gated;
  working-axioms mode):** the four implementation parameters were estimated on
  both models (controls PASS: PC1 reproduces CE14 joint-pair flip 1.00 / null
  0.00; PC4 carry-axis anchor separates committed classes). **Item (i) message
  — RESOLVED:** the head→combiner edge carries a **canonical (format-invariant)
  resolved carry** — within-chain cross-deciding-position carry-probe transfer
  1.00 = within-acc — with the deciding *position* additionally *decodable* from
  the edge (existence only, non-causal co-rider). **Item (ii) source — RESOLVED
  toward distributed-ST/depot:** the source is the **question-tail ST cluster
  and is NEVER `=`** — the `=` value arm flips 0.00 with carry-axis OV-projection
  ≈ 0 (**`=` is a depot, not a value source**, consistent with CE13 — this
  ADJUDICATES the source fork AGAINST the "`=` carry depot with pre-L1
  compounding read as a value" reading and toward distributed ST); among named
  ST sources the deciding-ST cluster is carry-specific (null 0.00) at all depths
  and dominant at 6d k3, chain-ST elsewhere. **Item (iii) path shares/necessity
  — RESOLVED:** the **head-pair (SV) path is the effective carrier** (real patch
  flip 1.00); the **skip/direct residual carries negligible carry** (0.14/0.077,
  ≈200× below the head-pair signal; power injection at that magnitude, 1× and 2×,
  flips 0.00) — the model does not route carry through the skip, closing CE14's
  direct-arm underpower gap in the *"is it used"* sense (NOT a formal exclusion —
  a sufficient-magnitude injection was not tested); and the pair is **class-
  necessary** (joint H1+H2 ablation collapses cascade acc, spares carry-free,
  over a ~0 untagged baseline; necessity-over-baseline 1.07 6d / 0.85 5d).
   **Item (iv) combiner form — OPEN (RESOLVED later by CE17: STEP function,
   α*≈0.75 — see the A11/CE17 block).** Battery F instrument invalid (combiner
   α-sweep produced no flips); unestimated → B2 stands. **A9 (selection): stays
  RETIRED at the selection-mechanism level** — deciding-ST is dominant at only 1
  independent depth (6d k3) vs the ≥2 needed, and the CE14 same-cell
  tracking+edge bar remains unmet. **A6 economy: RAISED to class level.** Net:
  **A10 core held at MEDIUM-HIGH; items i–iii resolved (implementation now
  described), iv open.** Dual-gated: pre-launch PASS-WITH-CONDITIONS (SI-1..SI-10,
  incl. skip-power-space fix and M scale-normalization) + post-result
  PASS-WITH-CORRECTIONS (F1 "skip carries ≈0" not "excluded"; F2 "ST cluster
  distributed" not "deciding-ST"; F3 co-rider = decode-existence). Backlink:
  [study-sv-implementation.md](study-maths/study-sv-implementation.md),
  `results/study-sv-implementation/results.json`, CE16.

### A11: Multi-digit compounding is a positional L0 relay across the question-tail ST sites

- **Belief**: The compounding that turns per-digit tri-states into a resolved
  carry happens **at layer 0, sequentially in position space across the
  question-tail ST sites**. Under the causal mask each later tail position
  sees one more digit pair (the ST node for digit `n` sits at/after `D'n`,
  where digit `n−1`'s pair is already visible; the sign-token ST nodes see
  the whole question), so each site's L0 write carries the carry **resolved
  up to its visibility horizon** — CE13's "single-step co-resolution" is the
  horizon-1 case of this, not an anomaly. The consumer L1 pair then merely
  **fetches the most-resolved relay(s)** available for its digit; it does not
  itself compute the cascade.
- **Why**: (1) The causal mask forces the horizon structure: lower-digit
  operands appear later in token order, and the map's ST nodes are strung
  across the tail (6-digit P10→P11→P12→P14), each position one step deeper.
  (2) One picture reconciles all current evidence: CE13 single-step
  co-resolution (horizon-1); CE16-M's canonical format-invariant carry on the
  wire (a most-resolved relay *is* the canonical carry); CE16-R's
  depth-dependent distributed source mass (which relay suffices depends on
  depth); CE12's SV-decodable-at-`=` presence (deep-horizon writes sit in the
  tail region) alongside CE16's `=`-not-a-value-source (consumers read the ST
  sites directly). (3) It gives the L1 pair the cheap job (fetch), matching
  the combiner's carry-specific, format-invariant input.
- **Prediction**: (1) **Horizon decode**: on depth-`k` chains, each chain-ST
  site's L0 write decodes the resolved carry up to exactly its
  position-determined visibility horizon — not merely local class ± one step.
  (2) **Relay causality**: patching the *most-resolved visible* chain-ST
  write flips the consumer's digit; shallower relays matter only when deeper
  ones are ablated (redundant-relay structure, tested jointly per the
  axioms). (3) CE16-R's per-depth source weighting tracks which site is the
  shallowest sufficient relay at that depth. (4) Sign-token ST writes (full
  horizon) alone suffice for the leading digit.
- **Falsifier**: chain-ST writes on deep chains carry only local class +
  single-step resolution regardless of position (horizon decode flat at 1) —
  then multi-digit compounding must happen **inside the L1 read**
  (attention-weighted value combination), reviving an A9-flavored L1
  computation.
- **Alternatives**: (a) L1 value-path combination (the head's weighted sum
  over local writes implements the cascade select); (b) mixed — partial relay
  plus L1 finishing; (c) per-model idiosyncrasy (the paper's "ST nodes in
  semi-random order" means horizons vary per model — the prediction is
  map-relative, not position-absolute).
- **Tension with human**: partially **vindicates** the human's
  sequential-cascade lean (C3/C4) — sequential compounding would be real, but
  in *position space at L0 across the tail*, not as state riding
  token-to-token and read off `=` (that value-source reading is causally
  refuted by CE16). Retires my one-hop-selection A9 as an L1 computation if
  confirmed; revives its spirit at L0 if refuted.
- **Confidence**: medium — the visibility logic is architecturally forced and
  four independent observations fit it, but the horizon decode and relay
  patches are untested (agenda entry 1).
- **Update 2026-07-16 (CE17, compounding-arithmetic; dual-gated): LOWERED to
  low.** The horizon decode + relay patches were run (chain-top n_top=4,
  k∈{2,3,4}). A horizon-consistent decode appears in **exactly one m>0 boundary
  site (6d P11H2 at k=2/d=2, carry-bacc 1.00 over baseline 0.70)** — resolved only
  where it can see the deciding digit — but its own within-site prediction **fails
  at k=3** and **5d does not replicate** (its only m>0 site is null). The relay is
  **not causally isolable**: twin-interchange of the tail-ST OV writes (deepest-
  sufficient, insufficient, joint-all arms) flips the leading digit **0.00 at every
  depth, both models**, with a valid CE13-ablation instrument (low-digit impact
  0.067/0.05 over ~0 baseline) — but because that control tests a different
  unit/target than the interchange, the null is **causally undetermined**
  (redundancy vs interchange-too-weak; the CE16-F1 trap, flagged). And the L1 edge
  output reconstructs from **local class alone** (r²≈0.95–0.99; horizon features
  add no held-out gain over a permutation null). Net: **A11's horizon mechanism is
  a single-site representational trace, neither replicated nor causally shown; the
  balance leans L1-local-sufficient. A11 → low.** A9 stays retired (no causal
  selection at either locus). The human's sequential-cascade lean (A6/C3) is
  **not adjudicated** by this study (the only positional-sequential signal is the
  single unreplicated cell). Post-result gate downgraded two over-claims (H
  "replication"; Y "redundancy-masked" → "causally undetermined"). Backlink:
  [study-compounding-arithmetic.md](study-maths/study-compounding-arithmetic.md),
  `results/study-compounding-arithmetic/results.json`, CE17.

## Sharpest forks

Where discriminating evidence would most cheaply reshape this file (ranking
itself belongs in [maths-next-steps.md](maths-next-steps.md)). Reranked
2026-07-16 after CE16 (A10 items i–iii resolved — canonical carry message;
distributed ST-cluster source, never `=`; head-pair effective and
class-necessary carrier; trails in the A10 confidence block):

1. ~~The compounding locus (A11 vs L1-read)~~ — **ATTEMPTED (CE17), inconclusive
   (R-mixed).** Horizon decode + relay patches ran: A11's horizon structure is a
   single-site, unreplicated representational trace (6d P11H2 only); the relay is
   not causally isolable (Y interchange null, cause undetermined); the L1 edge
   output is local-class-sufficient. Leans L1-local but not cleanly separable.
   A11 → low; A9 stays retired. A deeper causal relay test (a stronger
   interchange unit / more strong-writing boundary sites) would be needed to
   settle it — post-deadline.
2. ~~The combiner transfer function (A10 item iv; B2)~~ — **RESOLVED (CE17)**: the
   combiner is a **STEP** (threshold α*≈0.75, both models), via the on-manifold
   redesign (real-capture interpolation, endpoint-gated to CE16's 0.00/1.00 — the
   Battery-F dead-zero fixed). A10 items i–iv now all resolved. Neuron-level
   sparsity (B2) remains for the mixed-model work.
3. **Paper hand-off timing** — deferred by the human 2026-07-16; must trigger
   no later than ~12 h before the deadline. Post-deadline: the mixed-model
   shared engine (A7 vs C2).

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
- **2026-07-16** — After the leading-digit-walkthrough study (C5 step 5; Gate 2 PASS
  WITH CONDITIONS, nearly clean; [study note](study-maths/study-leading-digit-walkthrough.md),
  [CE15](maths-claim-evidence.md)): the C5 capstone. The leading digit `A_top` in the
  hard case is produced by the **same carry-specific L1-head-edge delivery** to the
  sign-position combiner as the middle digits (CE14) — Link 3 verified (5-digit
  genuine depth spread; 6-digit deep-chains only, shallow unadjudicated). **A10
  consolidated across all answer digits incl. the hardest, but NOT raised** (held at
  medium; Link-4 readout quarantined; economy uninformative at the sign bottleneck;
  direct path not excluded; inherits all CE14 caveats). **This completes C5's 5-step
  program for the addition model** (CE13 → CE14 → CE15). Notably the *first nearly-
  clean Gate 2 of the session* — the discipline forced by CE14's three-round ordeal
  (quarantine the readout patch; consolidate-not-raise as a machine tag; deciding-
  matched null; verified only at ≥2 genuine depths) carried over and pre-empted the
  over-reach. One fix: two adjacent maximal chains ≠ an independent 2-depth spread
  (6-digit re-labeled deep-chains-only). Next: the mixed model (C5 "then repeat"),
  and a consolidation/referee checkpoint on the addition-model story is warranted
  before it (A10 is *partial*, hard-won).
- **2026-07-16** — After the SV-compounding study (C5 steps 2–4; Gate 2 PASS after
  THREE rounds; [study note](study-maths/study-sv-compounding.md),
  [CE14](maths-claim-evidence.md)): the core test of the agent's flagship A10.
  **A10's fetch-to-combiner core is PARTIALLY confirmed** — carry-specific
  (deciding-matched null = 0.00), ≥2-depth causal, consumer-head-specific
  attention-edge delivery to the L1-MLP combiner, both models (attention delivers,
  not the direct path). **But selection (A9/A10-b) is NOT shown** (tracking head H2
  ≠ single-depth-edge-causal H1), it is **sufficiency not necessity** (H1 ablation
  inert; H2 necessary+tracking → redundant pair), the direct path is **not excluded**
  (underpowered), and value content is not head-specific. **A10 held at medium**
  (core partially confirmed, selection unsupported); **A9 not supported**; **A6
  economy supported at H2**; **A5** hybrid routing partial causal support. This took
  **three** Gate-2 rounds on the flagship conjecture: round 1 over-claimed
  "R-A10-selection at L1.H1" (a three-attribution smear — single-depth edge on H1,
  tracking on H2, necessity on H2); round 2 over-corrected to "not carry-specific /
  A10 not confirmed" on a **broken specificity null** (a "same-class null" that
  actually re-toggled the deciding carry, so null=real=1.0 was the *expected*
  signature of a carry-specific head); round 3 fixed the null to deciding-matched
  (null = 0.00) → the calibrated middle. Durable method lessons: (1) a specificity
  null MUST hold the tested variable fixed — re-toggling it proves nothing (reuse
  the CE10/CE13 `same_class_diff_operand` convention); (2) edge SUFFICIENCY
  (patching flips) ≠ NECESSITY (ablation) — report both, they can land on different
  heads; (3) on one's own flagship conjecture, expect and pre-empt the smear of
  distinct attributions (tracking/sufficiency/necessity) into one "selection cell".
  This is the sixth consecutive study needing a Gate-2 correction — the gates are
  doing essential work against a persistent first-draft optimism.
- **2026-07-16** — After the node-output-encoding study (C5 step 1; Gate 2 PASS
  after TWO rounds; [study note](study-maths/study-node-output-encoding.md),
  [CE13](maths-claim-evidence.md); revises [CE3](maths-claim-evidence.md)): the
  first study under the human C5 refocus (start from the HF verified node maps).
  Map-named `ST`/`SC` nodes **encode** their class (paper tags confirmed); the `ST`
  write **co-carries single-step local U-resolution** (cin-dependent on `U`) —
  **A10 premise refined**, not supported, and NOT shown to be multi-digit
  compounding. **CE3 refined to redundancy, baseline-controlled** (interchange
  flips nothing, but low-digit ST-node ablation exceeds an untagged-head baseline —
  the map's causal tags **vindicated**, a C5 win; high-digit redundant). Map-named
  `SA` L0 heads do **not** write the answer digit except the leading one. A2
  untouched; A3 weakly-against. **Two Gate-2 rounds** corrected over-reach in
  **both** directions (first draft "ST write is local"; the fix over-swung to
  "compounding begins at the ST write"; settled on "single-step U-resolution
  co-located, not multi-digit compounding"). Method lessons: (1) a locality test
  must use the case where the toggled variable *can* matter (U pair, not definite)
  and report a ratio, not a slack-dominated threshold; (2) an ablation-impact claim
  needs an untagged-head baseline; (3) distinguish a *single-step* cin dependence
  (local resolution, definitionally expected on U) from *multi-digit* compounding
  before scoring A10. This C5-anchored study is the counterweight to the earlier
  paper-disconnected breadth studies — starting from the verified map made the
  findings sharper and the conjecture scoring more mechanical.
- **2026-07-16** — After the answer-position binding study (Gate 2 PASS WITH
  CONDITIONS; [study note](study-maths/study-answer-binding.md),
  [CE12](maths-claim-evidence.md)): the CE11 follow-up. **Answer phase is a split
  layout**: `SA` (sum) is a just-in-time **register** (absent at `=`) — **A4
  just-in-time fetch supported, raised to medium-high**; `SV` (carry) is
  resolved/present at `=` (CE7-consistent) but its per-digit slots are **not
  orthogonal** — **A4's orthogonal-tape alternative refuted**; both `SA`/`SV`
  **transfer across answer positions** (shared answer-side template), unlike the
  question-side `ST` (CE11), so A4's template claim is
  **position-of-computation-dependent**. **C2/A8 non-orthogonality reinforced** (SV
  slots not orthogonal, 6-digit entangled-below-null with a 1-D caveat). **A6/C3
  untouched** (AB-3 gate: SV presence at `=` is resolution, not shown stored; no
  causal test). Gate 2 caught an over-reach (a "coexistent carry bus beyond
  re-derivation" claim resting on a too-weak isolated-operand baseline — SV depends
  on all lower digits) and I reframed it CE7-consistent. Method lesson: a
  binding/coexistence baseline must include *all* the label's causal determinants,
  not just the same-index operands. Both models agree on the split. Next: the
  effective-dimensionality breadth study (entry 1 now); a causal "is the `=` carry
  used?" test is backlog.
- **2026-07-16** — After the cross-position/cross-subtask probe-transfer study
  (Gate 2 PASS WITH CONDITIONS; [study note](study-maths/study-probe-transfer.md),
  [CE11](maths-claim-evidence.md)): a breadth study (cascade line paused for
  breadth). **New structural fact for `ST`**: at question positions the tri-state
  carry is **position-specific** (an `ST` probe does not transfer across digits;
  mean-centering doesn't restore it) — **A4's transfer-for-free and C2's
  shared-template halves falsified for `ST`** — and `ST`/`SV` are **geometrically
  entangled** beyond their (independent) labels — **C2 orthogonality + A8
  interference challenged**. **A4 template-sharing lowered to low-medium**; **A8
  interference half lowered**; **C2 lowered for the ST question-position case**
  (human-owned, noted). Scope: `ST`-only (SA/SV diag-weak at the question site; SA
  lives at the answer position, confirming CE2/CE3); linear probes; both models
  agree — a rare clean cross-model replication. Gate 2 narrowed an initial
  over-reach (first draft claimed *all* sub-tasks position-specific + leaned on the
  trivial "position decodes at 1.00"; corrected to ST-scoped, mechanism = the
  centering-fails discriminator). The sharp follow-up is the **answer-phase** binding
  (B4, tape-vs-register) — promoted. Method lesson: a "no transfer" verdict is only
  valid where the diagonal probe passes; and a positional-embedding truism
  (position decodes perfectly) must not be dressed as a mechanism.
- **2026-07-16** — After the deep-cascade hand-off edge path-patch (Gate 2 PASS
  WITH CONDITIONS after one BLOCK; [study note](study-maths/study-cascade-handoff-edge-patch.md),
  [CE10](maths-claim-evidence.md)): the edge path-patch gives a **one-depth causal
  crumb** — `L1.H1` (the CE8 cell CE9 found inert) causally drives the combiner at
  6-digit k=3 with a computed, deciding-selective carry through the MLP. But the
  single-position edge instrument is underpowered at most cells (LN renormalization
  damps single-head edges), no head clears the ≥2-depth bar, and 5-digit has a live
  direct residual path. **A9 held at low-medium** (first causal crumb, not
  confirmed); **A6 held (not refuted)**; **A5 noted** (`L1.H1` causal at k=3, not
  raised). Gate 2 round 1 BLOCKed an **inverted power control** (scaled the direct
  path UP ~20×, over-claiming "A9 confirmed / A6 refuted / direct-path inert 0/9");
  corrected (scale DOWN to per-cell head-edge norm) → 6/9 cells underpowered. This
  is the **third** skeptic catch in the cascade line (CE9 ×2 + this), twice
  positive-direction. Durable method lessons: (1) a power/underpower control must be
  built at the *effect's own magnitude* and its scaling *direction verified*; (2) LN
  renormalization makes single-head edge patches systematically underpowered — a
  hand-off question needs a less-damped / multi-position edge instrument.
- **2026-07-16** — After the deep-cascade mechanism study (Gate 2 PASS after two
  BLOCK rounds; [study note](study-maths/study-deep-cascade-mechanism.md),
  [CE9](maths-claim-evidence.md)): the deep `...999` fork — the thread's #1 — was
  tested and found **not resolvable at node/attention-pattern granularity**.
  **A9 held at low-medium** (its predicted single-cell tracking+causation
  convergence is absent: causal-selective `L1.H0` and tracking `L1.H2` are
  different heads; CE8 `L1.H1` causally inert); **A6's sequential-accumulated deep
  form weakly disfavored** (no intermediate-digit identity where testable) but not
  refuted; **A5's CE8 routing cell stays representational** (causally inert under
  single-head redirect — the round-2 "load-bearing" over-claim withdrawn); **C3's
  physical-cascade clause weakly disfavored where testable** (human-owned, noted).
  Two skeptic rounds were decisive and instructive: round 1 caught a
  **negative over-reach** (unearned refutations resting on an uncontrolled
  intermediate-locus, an unimplemented tracking metric, and no selectivity
  baseline); round 2 caught the symmetric **positive over-reach** (A9 "confirmed"
  assembled from a value-matched metric asserted-but-not-coded, a units-end
  boundary-artifact tracking gap, and three signatures pulled from three different
  cells). Method lessons: (1) a null result needs a positive control *on the same
  locus/unit* or it is `invalid`, not `negative`; (2) a metric named in a docstring
  must be the metric in the code (integrity trap); (3) require a mechanism's
  predicted signatures to converge on the *same* cell before scoring the
  mechanism; (4) single-head uniform pattern-redirect is too blunt — the
  selection→combiner hand-off needs an **edge path-patch** (B11, now the promoted
  confirming test). No conjecture confidence raised on this study — the honest
  outcome is an instrument limit, and the fork stays open.
- **2026-07-15** — After the human's C4 reflections (all nine studies gated;
  no new empirical result — a conjecture-level update triggered by human
  review, with the human's answers to three clarifying questions recorded).
  C4 makes two points this file accepts: the `U` evidence base (CE4–CE7) is
  single-digit-`U`, middle-answer-digit only, so the CE5 combiner is not shown
  to be "the whole `U` mechanism"; and the leading answer digit forces full
  cascade resolution at or before the answer-sign position (mixed-model sign:
  at or before `=`) — a locus no study probed. Changes: **A6** support
  explicitly scoped to single-digit `U`, deep-chain mechanism reopened as the
  top fork; **A9 added** — deep cascades resolved by one-hop attention
  selection of the deciding digit (agent lean), against the human's recorded
  sequential-cascade lean; **A2** confidence updated in place (CE5 U-combine
  support; discretization target revised to binary carry per CE6/CE7);
  **A3/A5** annotated (leading-digit motivation for B12; CE8 routing cells as
  candidate A9 machinery); overall-picture stages 2/4 annotated; sharpest
  forks reranked (deep-cascade mechanism now #1). Note: C4 was written before
  studies #8–#9 landed; CE8's carry-state routing in fact supplies candidate
  machinery for the very cascade C4 found missing — convergent pressure
  toward the same fork from human reflection and agent data.
- **2026-07-15** — After the human's C5 reflections (studies #10–#13 gated; a
  conjecture-level update triggered by human review, plus a direct read of the
  paper's per-model verified maps — `behaviors.json`/`features.json` from
  `PhilipQuirke/VerifiedArithmetic` — for both studied models). **C5 accepted
  with a self-correction**: the thread spent much of studies #2–#13
  re-locating nodes the maps already name, and the last four studies
  (CE9–CE12) rediscovered the SP-tagged answer-position L1 wiring the maps
  record outright; worse, CE3's dismissal of the paper's question-position ST
  candidates used a no-lower-carry stimulus that removes exactly the regime
  where ST matters, while the maps show those nodes failing 14–23% of random
  questions with multi-digit Impact — that dismissal is flagged suspect and
  must be re-examined in a gated study before it is relied on again (CE docs
  untouched pending that study; conjecture-level flag only). Changes: **A10
  added** (map-anchored SV mechanism: question-tail/sign L0 ST nodes →
  SP-tagged answer-position L1 heads attending `=`+ST sites → L1-MLP combine;
  A6/A9 recast as its variants a/b/c); **A9 sharpened** to
  selection-within-the-fetched-ST-cluster; **C5 relation bullet added**
  (anchor-not-ceiling caveat: interchange fills map gaps, e.g. the unlisted
  5-digit A2.SC role at `P14L0H0`); **overall picture annotated** (the maps
  name stages 2–5's physical instantiation; the open work is the content and
  combination rule on named wires); **sharpest forks reworked** to the C5
  five-step program; **agenda reworked** accordingly (effective-dim/SAE
  breadth entry demoted to backlog — not on the C5 critical path). Method
  lesson recorded: consult the project's own verified artifacts before
  designing de-novo assays; re-verification is a by-product, not a goal.
- **2026-07-16** — After the human's over-caution feedback and the ~40-hour
  paper-revision deadline (also back-filling this log for CE13–CE15, whose
  updates had landed only in A10's confidence trail). **Mode shift adopted as
  the new [Working axioms](#working-axioms)**: (1) a working SV implementation
  exists (behaviorally entailed) — experiments are attribution/estimation,
  never existence tests; (2) redundancy is the norm — single-node necessity
  failures are expected, the standard is sufficiency + specificity +
  replication with class-level necessity; (3) the map is trusted wiring; plus
  the aggregation corollary. Under these axioms the CE13+CE14+CE15
  convergence (two models, ≥ 2 depths, every answer digit including the sign
  position, carry-specific with matched nulls) justifies **raising A10's core
  wiring to medium-high** — the per-study "hold at medium" verdicts were each
  single-study-scoped; cross-study aggregation is evidence the individual
  gates could not use. **A9 parked** (variant detail; two attempts failed to
  show selection; the sprint's source decode may settle it as a by-product).
  Sharpest forks rewritten to A10's four implementation gaps; agenda rebuilt
  as the 40-hour sprint (entry 1 = edge message / source / path shares /
  necessity; entry 2 = paper hand-off consolidation merged with the referee
  checkpoint; mixed-model deferred past the deadline). One point of retained
  pushback, recorded for honesty: the skeptic gates caught real errors in
  *both* directions (CE14's three rounds included a broken null that would
  have shipped a false negative), so the sprint *compresses* gating (one
  combined pass per study, proposed to the human) rather than dropping it —
  the over-caution was in the framing (existence tests; refusing cross-study
  aggregation), not in having adversarial review.
- **2026-07-16** — After the SV-implementation sprint (CE16; combined gate
  PASS-WITH-CORRECTIONS; the working thread applied the A10 annotations —
  this entry back-fills the log and adds the synthesis). CE16 settles A10
  items i–iii: **canonical format-invariant carry** on the head→combiner
  edge (deciding position decodable as a non-causal co-rider); source = the
  **question-tail ST cluster, distributed, never `=`**; the **head pair is
  the effective and class-necessary carrier** (skip carries ≈200×-smaller
  carry; power-matched injection at that magnitude moves nothing). Human-lean
  scoring, honest in both directions: the "`=` as carry depot *read as a
  value source*" reading is **causally refuted** (CE12's SV-at-`=` was
  presence-not-use), which damages the paper's "resolved at `=`" delivery
  story — but the new **A11** (added now) partially revives the sequential
  lean: the causal mask plus the map's tail ST placements force each later
  site to see one more digit pair, so compounding plausibly proceeds
  **sequentially across L0 tail positions** (CE13's single-step
  co-resolution = horizon-1), with L1 fetching the most-resolved relay. A9
  stays retired at the L1-selection level; A11's falsifier would partially
  revive its spirit at L1. Forks rewritten (compounding locus #1; combiner
  transfer #2, redesigned on-manifold so the Battery-F dead-instrument
  failure cannot recur). Agenda: completed sprint entry deleted; new entry 1
  = compounding-arithmetic + combiner-transfer study (both batteries reuse
  CE16/CE13 artifacts); paper hand-off held at entry 2 per the human's defer,
  trigger no later than ~T-12h.
