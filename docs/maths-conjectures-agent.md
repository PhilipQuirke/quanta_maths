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
   see A1 confidence update and [CE1](maths-claim-evidence.md#ce1).
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

*[2026-07-16 (terminology, per the human): the L0/L1 indices in this picture
are the 2-layer instantiation. Phrase the account layer-generally — **writer
layer(s)** (early layers computing ST/SA/SC at question-tail/sign positions),
**consumer layer** (the final layer's answer-position heads fetching the
carry), **combiner** (the final-layer answer-position MLP, now known to be a
step-function discretizer, CE17) — so it transfers to the deeper mixed models
(l3/l4) and the larger zoo. Size scope per C6/A12: the redundancy woven
through this picture is a 5/6-digit observation; tightness at n=10+ is
untested.]*

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
- **C6 (verbal prior, 2026-07-16: redundancy is small-model slack; larger n
  forces a tighter algorithm)** — accepted as the working read of the
  recurring redundancy findings, and it makes a testable prediction this file
  adopts as A12. The human's reasoning: a 6-digit model has capacity slack, so
  duplicate carry circuits are cheap; at n=10+ the number of sequential steps
  that must *all* work for accuracy squeezes that slack, so the implementation
  should be tighter. This retro-explains why our sharpest nulls (CE13/CE17
  interchange 0.00 despite ablation impact; CE14 single-head harmlessness)
  keep landing on *small* models — redundancy blurs single-node causality —
  and it predicts the same assays give sharper verdicts at n=10/13. The human
  also directed (same date) that the account be phrased **layer-generally**
  (writer-layer / consumer-layer / combiner roles, not "L0/L1"), since larger
  and mixed models have more layers; adopted throughout new text.

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
   ablation), not per node. **Size scope (C6, 2026-07-16)**: this axiom is an
   observation about the *small* (5/6-digit) models studied so far; the
   human's C6 prior predicts redundancy thins as n grows, making single-node
   interventions sharper at n=10+ — claims are size-scoped and re-tested per
   A12 before generalizing across the zoo.
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
  ([CE1](maths-claim-evidence.md#ce1)):
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
  strongly disfavored (see streak note in the reflection log). **Update
  2026-07-16 ([CE27](maths-claim-evidence.md#ce27), F1 write-site manifold
  shape): the descoped question-position remnant is now settled DESCRIPTIVELY
  against a simplex.** At the strong question-tail ST writers the tri-state write
  is **collinear-ordered** — U's off-rail fraction is not above a permutation
  null (p 0.99–1.0) at any strong writer, the U centroid split by incoming carry
  *on the rail* (CE13 single-step co-resolution; site-concentrated). So even at
  the question write position there is no off-axis U symbol; the linear/ordered
  read closes A3's last hiding place (a *weak* sub-floor or *nonlinear* U symbol
  is still not excluded — linear probes only). **2026-07-15
  note:** the human C4 leading-digit argument independently motivates the
  question-position remnant (B12): the top answer digit is predicted from the
  answer-sign position, so whatever carry information feeds it must already
  exist at or before that position — a deep `...999` chain at question
  positions is the one regime where   a transient `U`-like state would earn its
  keep (though A9 predicts even there it is skipped by selection).
  **Update 2026-07-16 (CE28):** at the L1-read carry rail the `U` write is
  **cin-dependent on the resolution axis and never a static third symbol** — it sits
  between the committed values at moderate-weight sites and *overshoots* them (a full
  cin-gate) at the highest-weight deciding site. Reinforces A3-low: `U` is a
  transparency/relay state parameterized by the incoming carry, not an off-axis symbol.

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
  **Update 2026-07-16 (CE28, indirect support):** on the carry-rail reconstruction,
  a **static-attention** additive read of the per-site class values reconstructs the
  combiner input and predicts carry-out (agreement 0.92–1.00), whereas rescaling by
  **per-prompt** attention makes it *worse* (negative R²) — i.e. the carry read is
  well-modelled by *static* attention weights, consistent with A5's majority
  (mostly-static) claim at the consumer read.
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
- **Confidence**: medium → **hybrid / low on the control mechanism 2026-07-16**
  after the mixed model ([CE21](maths-claim-evidence.md), [CE23](maths-claim-evidence.md)).
  The "shared engine" half is **supported** (SA/MD/ND share the same head nodes;
  the L2 combiner is shared — patching an ADD L1-state into a SUB run emits the
  correct ADD digit 0.96); but A7's **function-vector / low-rank-control** form is
  **refuted** — a rank-1 operator steer flips the readout 0% at both the combiner
  and the SLT selector, and no single SLT head selects; the add/sub selection is a
  **distributed, high-dimensional** L1 transformation (leans C2). Residual A7
  escape hatch: a learned rank-r operator subspace (untested). See the mixed
  reflection-log entries and CE21/CE23.

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
  claim (agenda entry 2) is still the untouched core test. **Update 2026-07-16
  ([CE27](maths-claim-evidence.md#ce27)):** the ST–SV entanglement reproduces
  (principal angle 26°/21°), but it is **not cleanly "the carry rail"** — the
  consumer-OV carry direction is only moderately aligned with the SV axis
  (|cos| 0.35–0.47) and oblique to the ST plane (52–57°), so removing it leaves
  a residual angle that only modestly exceeds a recomputed null. The entanglement
  is real but its geometric source is not a single shared 1-D carry direction.

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
  [CE16](maths-claim-evidence.md#ce16).
- **Update 2026-07-16 (Mixed model, CE20/CE22/CE25):** A10's core wiring +
  implementation **replicate on the 3-layer mixed model across ADD/SUB/NEG** — the
  combiner is a STEP (CE22 iv), the delivered carry/borrow is canonical and
  `=`-is-not-the-source (CE22 i/ii), and delivery is carry/borrow-specific across
  depths 2–4 (CE25). One layer-general refinement: the delivery *route* is
   class-dependent — ADD rides the residual, SUB/NEG also use last-layer attention
   (CE20/CE25). Held at medium-high on the wiring; delivery-route now a documented
   per-model property (tag `Probe:DELIVERY.*`).
- **Update 2026-07-16 (CE28, geometric implementation of the combiner STEP):** the
  CE17 STEP combiner now has a geometric account — its input is an **ordered place-
  value carry rail** whose per-site writes have super-increasing weighted gaps in
  significance within the cascade region, so the STEP is a **threshold on a dominance
  code** that (at a middle digit) provably discretizes the delivered carry into the
  digit. Scoped: the certificate is met at the d6 middle consumer but not the leading
  digit (compressed top-gap) — the geometry *does* the compounding-to-carry step,
  provably at a middle digit, empirically elsewhere. Does not change A10's wiring
  confidence.

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
  [CE17](maths-claim-evidence.md#ce17).
- **Update 2026-07-16 (CE19, compounding-locus; dual-gated): A11 NOT SUPPORTED,
  held at low (not formally rejected).** A redesigned assay used a natural
  **decorrelation** (a chain-ST site inside the 999-run has local class fixed at U
  while the resolved carry varies) and split cells into VISIBLE-decorrelated (site
  can see the deciding digit — self-computable) vs **INVISIBLE-decorrelated**
  (deciding digit below the site's horizon — a carry there could only be RELAYED,
  the sole discriminator). Decorrelation control passes exactly (local-class decode
  0.50). At the invisible-decorrelated cells the resolved carry decodes at **chance
  in both models**; where the write is **readable at that depth** (5d P9H1 k3, a
  per-depth readability control passes) chance = a genuine **"no relayed carry"**
  → **R-L1-read** (the multi-digit compounding is completed in the L1 consumer
  read, not by an L0 positional relay); at 6d the deep-chain writes **wash out**
  (per-depth control fails) → underpowered, neither supports nor refutes. A
  class-level knock-out shows the ST class is carry-**necessary** (all-ablate breaks
  differentially over 0.00 specificity-null + 0.00 untagged baseline) but does not
  itself discriminate relay from local resolution (DH is load-bearing). Net: **no
  positive evidence of a cross-position L0 relay anywhere; A11's positional relay
  is unsupported** — the compounding is an L1-read computation (5d proven, 6d
  consistent-but-underpowered), then discretized by the step combiner (CE17).
  Caveats: **linear probe** (a non-linear/different-subspace relay not excluded);
  6d underpowered; rests on one readable invisible cell. **A11 → low (not
  supported, not rejected); A9 stays retired; the human's C3/sequential-cascade
  lean is not supported at the L0 tail** (with caveats). This escapes CE17's two
  blockers and substantially settles the compounding-locus fork against a relay.
  Post-result gate corrections F1 (softened "refuted"→"not supported"), F2 (added
  per-depth readability control → 6d underpowered), F3 (dropped 5d
  "self-computation" over-label), F4 (KO is necessity-only, DH load-bearing).
  Backlink: [study-compounding-locus.md](study-maths/study-compounding-locus.md),
  [CE19](maths-claim-evidence.md#ce19).
- **Update 2026-07-16 (CE24, compounding-locus-v2; dual-gated): CONCLUSIVE — the
  compounding is an L1-read; A11 (positional L0 relay) rejected/low.** A redesign
  (fixing CE19's OV-write washout + the ill-posed invisible-cell discriminator via
  full-residual reads + **cross-depth transfer** at the see-everything gather, on
  **9-free** stimuli per the human's `66666+33334`/`33433` insight) **layer-localizes**
  the resolved carry: the canonical, answer-agnostic carry **transfers cross-depth
  1.00 at the L1 combiner input** (answer-top does NOT ride the carry axis:
  0.12/0.13 vs 0.78 full-residual) while **L0's output has no carry-specific
  canonical code** (extreme-pair transfer ~chance with CIs, high within-depth
  ceiling; weak transfer = magnitude nuisance). 9-free ≡ 9-containing. Combined with
  CE16's causal head-pair delivery, **the L1 read is where the canonical carry
  emerges** — conclusive both models, superseding CE19's 5d-only/6d-underpowered.
  Representational + cited-causal (this study's own LC instrument invalid); linear
  probe; L0 tested at `=` (ST/sign via CE17/CE19). **A11 stays rejected/low; A9
  retired.** Backlink:
  [study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md),
  [CE24](maths-claim-evidence.md#ce24).
- **Update 2026-07-16 (CE24 TF addendum, token-time; dual-gated): the propagated
  carry is finalized LAZILY at the answer read (L1), single-step eagerly in-place
  (L0).** Closing the token-time question (CE24 settled the layer): the canonical
  propagated carry (cross-deciding-position transfer, run-break-decorrelated,
  9-free) appears **only at the sign token, L1** — ~chance at every operand token
  and at `=`, never at L0 — a **~2-token deferral past full input availability
  (D'_0)** and past the `=` gather (chance; `=`-is-a-depot, CE16). The **local
  single-step make-carry is eager in-place at L0 at its own token** (0.76–0.94,
  CE13). The make-carry-token→D'_0 span is information-availability (MSD-first
  layout, CE19), not laziness. A11 unchanged (rejected/low; TF reinforces the
  L1-read locus). Reusable cross-model tool
  `quanta_maths/maths_temporal_finalization.py` (may differ by model — run the zoo).
  Backlink: [study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md#addendum--battery-tf-temporal-finalization--token-time--run-2026-07-16),
  [CE24](maths-claim-evidence.md#ce24) (TF addendum).

### A12: The SV interface generalizes across model sizes, and the implementation tightens as n grows

- **Belief**: Phrased layer-generally, the SV account holds across the
  addition-model zoo: **early-layer ST writers** at question-tail/sign
  positions (local class + visibility-bounded resolution), **final-layer
  consumer heads** at each answer position fetching a canonical resolved
  carry from the ST cluster (never from `=` as a value source), and a
  **step-function combiner MLP** at each answer position discretizing the
  delivered carry into the digit. And, per the human's C6 prior, the
  *implementation tightens with n*: at n=10+ the sequential-accuracy pressure
  strips the duplicate circuits that small models afford, so the redundancy
  census shrinks and single-node interventions become decisive — including
  exactly the interventions (site-write interchange, single-head edges) that
  nulled at 5/6-digit.
- **Why**: C6's capacity argument; Paper 2's zoo-wide algorithm claim (the
  same sub-task structure is found across ~40 models — C5); and the pattern
  of our own nulls, which concentrate where redundancy is cheapest.
- **Prediction**: on `add_d10_l2_h3_t40K_s572091` (and d13): (1) the HF map
  shows the same roles (question-tail/sign ST writers with multi-digit
  Impact; answer-position consumer heads attending the ST cluster;
  high-`Fail%` answer-position combiner MLPs); (2) the per-role duplicate
  count (map `Fail%`/`Impact` census + interchange decisiveness) is **lower**
  than at 5/6-digit; (3) the CE16 batteries transfer (carry-specific
  head-class delivery, class necessity, `=`-not-a-source); (4) where
  tightness permits, the CE17 compounding-locus assays (horizon decode,
  relay interchange) give the sharp verdict the 6-digit redundancy blurred —
  the best remaining lever on A11-vs-L1-read.
- **Falsifier**: d10+ shows equal or greater redundancy than 6-digit
  (C6 refuted; redundancy is not capacity slack), or the interface roles do
  not appear in the larger maps (the account is size-specific — a major
  scope restriction for the paper).
- **Alternatives**: (a) tightness varies by role (e.g. ST writers tighten,
  consumers stay duplicated); (b) larger models change *mechanism*, not just
  tightness (e.g. genuinely sequential multi-layer relay in deeper mixed
  models — where the extra layers exist).
- **Tension with human**: none — this operationalizes C6; disagreements would
  only emerge from the data.
- **Confidence**: medium on role transfer (the zoo maps + Paper 2 support
  it); low-medium on monotone tightening (plausible capacity argument,
  untested).
- **Update 2026-07-16 (CE18, cross-size-sv; dual-gated): role-transfer +
  step-combiner RAISED to medium-high; tightening/C6 NOT SUPPORTED (→ low);
  large-n source-fork untested.** Ran d5/d6/d10/d13 (all acc 1.000), registries
  built from the published maps (d10/d13 have ST-writer + combiner-MLP tags but
  NO L1 consumer-head tags — consumer role identified empirically, itself a
  role-transfer datapoint). **Confirmed (robust):** all three roles present at
  d10/d13; the **combiner is a STEP at every size** (endpoint-gated to each
  model's real 0/1, α*≈0.5–0.75 — CE17 generalizes); PC2b (ST class ablation >
  untagged baseline) passes at d10/d13. **NOT reproduced at large n
  (probe-limited):** the CE16 causal source signatures — the carry axis collapses
  to sep ~6 (vs ~30 at d5/d6) and both the `=` arm and the deciding-ST arm flip
  0.00, so `=`-not-a-source sits on a null background and the source-fork is
  UNTESTED at n≥10, not confirmed. **C6 (redundancy = small-model slack) NOT
  SUPPORTED:** single-node ST ablation ~0 at every size and the class-vs-single
  redundancy gap does NOT shrink across d5→d6→d10 ({0.056, 0.116, 0.324} —
  increasing); **d13 inconclusive** (whole-class ST ablation 0.040 ≈ CE13
  single-node magnitude — "very redundant" vs "ST-ablation ineffective at n_ctx
  43" not separated). Redundancy reads **intrinsic, not capacity slack** —
  A12's tightening sub-claim → **low** (not refuted; d13 can't separate). A11 was
  **not** rescued by scale (single-node null everywhere; Battery L untriggered).
  Post-result gate downgraded two over-claims (F1 transfer "confirmed" → "role +
  combiner confirmed, source probe-limited"; F2 C6 "refuted" → "not supported, d13
  inconclusive"). Backlink:
  [study-cross-size-sv.md](study-maths/study-cross-size-sv.md),
  [CE18](maths-claim-evidence.md#ce18).
- **Update 2026-07-16 (8-digit fill-in, confirmatory):** ran the SV suite on the
  previously-uncovered accurate **d8** (`add_d8_l2_h3_t45K_s173289`). All CE18
  signatures reproduce: role skeleton present (9 ST / 9 combiners + empirical
  consumer), **STEP combiner** (α*=0.5), **`=`-not-a-source 0.00**, **class-necessity
  1.00**, redundancy class-minus-single **0.152** (fits the d6 0.116 → d10 0.324
  trend), carry axis weak (sep 5.3 — deciding-ST probe-limited, as d10/d13). CE24
  **L1-property** reproduces (L0 cross-depth 0.57 ~chance, L1 carry-axis 1.00), and
  TF is **LAZY at L1** with **deferral 3 tokens (vs 2 at d5/d6)** — the eager/lazy
  split holds but the lazy gap grows modestly with size. SV interface now confirmed
  on **five addition sizes (d5,d6,d8,d10,d13)**. Confirmatory (not gated as a new
  CE). Artifacts `results/study-d8-batteries/d8_sv_summary.json`.
- **Update 2026-07-16 (Mixed model, CE20/CE22/CE25):** A12 generalization holds on
  a **new axis** — not just size but a **different architecture (3 layers/4 heads)
  and the borrow/neg-borrow task families**: the SV representation + STEP combiner +
   canonical message + delivery all replicate across ADD/SUB/NEG on the mixed model.
   Raised confidence that the SV interface is architecture/task-general (the delivery
   *route* can differ by class/model — run `combiner_delivery_sweep` across the zoo).
- **Update 2026-07-16 (Mixed model, 8-digit, [CE26](maths-claim-evidence.md)):**
   confirmed on a **new size** — the core SV mechanism (binary resolved cascade,
   STEP combiner, canonical message, `=`-not-source, shared combiner) replicates on
   the accurate d8 mixed model `ins1_mix_d8_l3_h4_t70K_s572091` for ADD/SUB/NEG. The
   delivery route is now shown to be **depth/size-dependent**: d8 SUB/NEG ride the
   residual for shallow cascades but switch to last-layer attention at depth 4 (d6
   used both routes at all depths) — so A10's delivery *route* is model-specific,
   not universal (the SV interface still generalizes). Map-blocked pieces
   (writer-necessity, SLT shared-engine, mechanism diagram) await a published d8
   map. Dataset: `results/study-mixed-d8/`.

## New stream 2026-07-16: latent-geometry conjectures (G-series)

Opened by human directive on the evening of 2026-07-16: the mechanistic
account (which nodes compute what, where, and by which token) is confirmed to
working-axiom standard; what is missing is a **geometric, latent-space account
of the same arithmetic process** — the shape of the storage manifolds, how
different features' manifolds (ST, SV, …) relate, and whether the geometry
itself does computational work. All G-entries are **speculative, pre-evidence**
(confidence: proposed). Two overnight studies are designed to score them:
[study-geometry-certificate.md](study-maths/study-geometry-certificate.md)
(G2, G3) and
[study-geometry-factorization.md](study-maths/study-geometry-factorization.md)
(G1, G4).

### G1: Rail-and-address factorization — storage = one shared carry rail ⊕ private per-site address subspaces

- **Belief**: Each question-side `ST` write factors into (a) a coordinate on
  a **single shared 1-D carry rail** — the pre-image, under the consumer
  heads' OV, of the CE16/CE17 canonical carry axis at the combiner input —
  and (b) a **private, position-specific address component** that dominates
  the write's raw variance. The rail carries the arithmetic content; the
  address carries binding/position and whatever co-rider information CE16
  decoded (deciding position).
- **Why**: This single geometry explains three otherwise-awkward accepted
  results at once: CE11's cross-digit probe NO-transfer (raw probes latch
  onto the dominant address components, which don't transfer), CE11's ST–SV
  entanglement at 21° (the shared component *is* the rail — the carry-bearing
  part of the ST write is literally the quantity SV accumulates; entanglement
  is the computation, not interference), and CE16/CE24's canonical
  format-invariant carry emerging in the L1 read (the OV projection strips
  the address, keeps the rail).
- **Prediction**: (1) cross-site tri-state probe transfer, dead raw
  (0.10–0.12, CE11), is **restored** (≥ 0.6) on the 1-D rail-projection
  coordinate after per-site offset centering — gaps transfer, offsets don't;
  (2) removing the rail direction from the ST and SV subspaces raises their
  principal angle from ~21° toward the 50–64° label-null.
- **Falsifier**: no 1-D shared component rescues transfer at the strong
  map-named writers (retention stays ≤ 0.2 with harness parity to CE11
  proven) — then canonicalization is a **read-time rotation** (per-source OV
  aligning genuinely different site codes), and the common currency exists
  only on the wire, not in storage.
- **Alternatives**: read-time rotation (above); partial sharing (rail exists
  at strong writers only); a shared but *nonlinear* code invisible to 1-D
  linear projection.
- **Tension with human**: refines C1/C2 — "shared template" fails at the raw
  activation level (CE11) but may hold on exactly one dimension; C2's
  orthogonality question becomes "address subspaces are private/orthogonalish,
  the rail is deliberately shared".
- **Confidence**: **REFUTED (OV-preimage form) 2026-07-16** by the
  geometry-factorization study ([CE27](maths-claim-evidence.md#ce27), both
  addition models, dual-gated). The proposed rail — the consumer heads' OV
  pre-image of the CE16 carry axis — does **not** rescue CE11's cross-digit
  no-transfer: its cross-site binary-carry off-diagonal gain (0.099/0.116) does
  not beat the pre-registered OV-pre-image-of-random wrong-axis null
  (0.078/0.106) by the 0.05 margin, and it is a weak within-site instrument
  (binary diag 0.60). **No shared *stored* carry rail is demonstrated at all**:
  a full-activation *binary* carry probe (CE11 tested only tri-state) transfers
  at off-diagonal gain 0.088/0.096 — at the same null band, no nuisance control
  — so the tidy factorization fails and the earlier smoke-test "rescue" was a
  small-n retention-ratio artifact. The prediction (2) half (removing the rail
  raises the ST–SV angle) is only weakly met: |cos|(rail, SV) 0.35–0.47, the
  rail is oblique to the ST plane (52–57°), and the residual angle overshoots a
  recomputed null (gap ~9–10°). **Net: G1's storage-factorization is refuted;
  canonicalization is a read-time transformation (consistent with CE24 — the
  canonical carry emerges in the L1 read), the common currency living on the
  wire, not in storage.** Residual life: a *nonlinear* shared code is not
  excluded (linear probes only). Was proposed → **low** (refuted for the linear
  OV-preimage rail; the read-time-rotation alternative is now the standing
  account).

### G2: The cascade is computed by a place-value dominance code on the rail, and the STEP combiner is what makes it possible

- **Belief**: Per-site writes project onto the rail with class values ordered
  `p_i < u_i < q_i` (local 0 < `U` < local 1), the `U` value split by the
  incoming single-step carry (CE13); the attention-weighted class gaps
  `g_i = w_i (q_i − p_i)` form a **dominance hierarchy in digit significance**
  (each gap exceeding the sum of all lower gaps, like place value); the
  `=`-token arm contributes a class-independent **bias anchor** (the real
  content of "`=` is a depot", CE16); and the combiner's STEP threshold
  (α* ≈ 0.75, CE17) sits inside the feasible interval those inequalities
  define. Consequence: **an exact linear read of local-class writes provably
  cannot equal the resolved carry at depth ≥ 2** (two-site contradiction),
  but a *thresholded* linear read can — the STEP is not an implementation
  detail; it is the thing that makes local-class storage + a linear attention
  read sufficient to compute TriAdd.
- **Why**: CE24 (canonical carry emerges in the L1 read) + CE13 (writes are
  local-class + single-step) + A11 rejected (no L0 relay) jointly *force* the
  read itself to perform the compounding; a weighted average can only compute
  a priority function ("highest non-9 wins") if the geometry encodes
  significance as dominance and `U` as transparency. The asymmetric threshold
  (0.75, not 0.5) and the depth-dependent source mass (CE16) fall out of the
  same inequalities.
- **Prediction**: per-site rail ordering with U-between (+ cin lean);
  super-increasing weighted gaps at each consumer (sign-position consumer =
  the full-depth case); per-prompt deciding-site contribution is the largest
  class-dependent term on depth-≥2 chains; additive reconstruction of α
  (R² ≥ 0.7) and of carry-out via threshold at α*; the measured feasible
  threshold interval is non-empty and contains α*.
- **Falsifier**: ordering violated (U outside (p,q)) at strong writers, or
  deciding-site dominance absent per-prompt, with positive controls passing —
  then the resolved carry comes from prompt-dependent attention reallocation
  or non-additive interactions, not a static dominance geometry.
- **Alternatives**: interaction/attention-selected read (CE8-style routing
  doing the work); ordered-but-uncertified geometry (relies on rarity of
  adversarial configs rather than worst-case-correct margins).
- **Tension with human**: none on direction; gives C3's "MLP transforms" a
  sharp form (the MLP's step *is* the cascade's nonlinearity) and gives the
  paper's `=`-attention puzzle a concrete answer (bias anchoring).
- **Confidence**: **LOW-MEDIUM (partial) 2026-07-16** after the certificate study
  ([CE28](maths-claim-evidence.md#ce28), dual-gated). **Supported**: on the
  read-time carry rail (canonical at the L1 read, CE24/CE27 — not a stored rail),
  an **exact** per-prompt LN-fair OV decomposition (recon_err ~5e-8) shows the
  per-site writes form an **ordered place-value code** — class-1 rail > class-0 at
  every site, **super-increasing weighted gaps within the cascade region** (both
  models, both the shared and each consumer's own refit rail), **cin-dependent U**
  (CE13), and a **static-attention additive read** that a single rail threshold
  turns into the resolved carry (agreement 0.92–1.00; the per-prompt-attention read
  is worse → supports A5 static routing). The **worst-case TriAdd certificate is
  MET at the d6 middle consumer** (interval non-empty + α\* inside + config-accuracy
  1.00) — a real geometry-computes-TriAdd datapoint. **Not established / refuted**:
  the **worst-case-certificate form at the leading digit** (empty interval on both
  the shared and the consumer's own local rail — a genuine *compressed top-gap*, not
  rail-misalignment, confirmed by the local-rail control) and at the **primary d6**
  (static R² 0.24–0.42, non-additive residual); the **U-as-neutral-midpoint** form
  (at the single highest-weight deciding site U *overshoots* the committed range — a
  full cin-gate); α\* is a different coordinate from CE17's ≈0.75 (not "explained").
  Controls all pass (untrained twin sep 28→2.7; wrong-axis gap collapse ~100×;
  PC1 1.00/0.00). Net: the **constructive ordered-dominance geometry is real and does
  computational work**, but the strong "provably computes TriAdd everywhere via one
  STEP" form holds only at a middle digit. Competing read 3 (ordered-but-uncertified)
  + read 2 (non-additive residual) survive at the leading digit. Scored by
  [study-geometry-certificate.md](study-maths/study-geometry-certificate.md).

### G3: Dynamic-range law — dominance coding under bounded norms compresses gaps exponentially with operand count

- **Belief**: Dominance requires `g_i ≳ Σ_{j<i} g_j`, i.e. roughly geometric
  gap growth with significance; LN and finite residual budget bound the total,
  so the *smallest* gaps must shrink ~exponentially as digit count n grows.
- **Why**: pure arithmetic of G2 plus bounded norms. It **re-explains CE18**:
  the carry-axis separation collapse at d10/d13 (sep ~30 → ~6) and the
  "probe-limited" source signatures are not instrument failures but the
  predicted dynamic-range compression of a place-value code.
- **Prediction**: measured gap profiles `g_i ~ ρ^i` with ρ roughly constant
  per model; the smallest gaps at d10/d13 approach the noise floor exactly
  where CE18's probes degraded; behavioral depth limits (if any exist at very
  deep chains on large-n models) coincide with gaps crossing the floor.
- **Falsifier**: gap profiles flat or non-monotone at models whose cascade
  behavior is accurate (would also refute G2's dominance form), or large-n
  separations that do NOT shrink with the digit count.
- **Alternatives**: per-consumer renormalization (each answer position's
  consumer re-scales so only a few sites below it matter — dominance locally,
  no global compression); mixed strategies at large n.
- **Tension with human**: touches C6's spirit from a new angle — what
  changes with n is not redundancy (CE18 refuted that) but **signal
  allocation**.
- **Confidence**: **LOW (consistent, not discriminated) 2026-07-16** after the T5
  battery ([CE28](maths-claim-evidence.md#ce28)). The **carry-axis dynamic-range
  collapse is observed** — committed-carry separation 28.6/32.2 (d5/d6) → 5.57/6.04
  (d10/d13), **reproducing CE18's 30→6 as physics (bounded-norm + dominance gap
  compression), not instrument failure** — and gap profiles are super-increasing
  (d10 ρ≈6.9, r²=0.78). BUT the strict `g_i ~ ρ^i` law is **weak at d13** (ρ≈1.9,
  r²=0.41, non-contiguous digit set), and the near-zero low-significance gaps are
  **not discriminated from plain irrelevance / per-consumer renormalization**
  (G3's own listed alternative — those digits sit far below the consumer). So the
  **CE18 re-explanation is the durable part**; the geometric-decay-law form is
  suggestive at d10, unproven. Falsifier (sep NOT shrinking with n) not triggered.

### G4: One rail, many meanings — ADD/SUB/NEG share a single unit-adjust rail; data is low-D even though control is high-D

- **Belief**: On the mixed model, the resolved carry (SV), borrow (MV), and
  negative-borrow (NV) binary codes at the shared combiner input live on
  **one shared 1-D rail** (up to sign/rotation conventions), with OPR/SGN
  determining how the combiner *interprets* the rail rather than where the
  value is stored. Sharpens the shared-engine result: CE23 showed the
  operator **control** is distributed/high-dimensional; G4 says the
  **carried datum** ("adjust the digit by one: yes/no") is one-dimensional
  and shared.
- **Why**: the combiner is shared (CE23 full-state patch 0.96) and its input
  code is binary for all three classes (CE20); maintaining three parallel
  rails into one step-function combiner would need three thresholds where one
  suffices; parameter-transfer initialization from the addition model makes
  reusing the existing carry rail the cheapest solution.
- **Prediction**: pairwise |cos| between the fitted SV/MV/NV class axes at
  the L2 combiner input ≥ 0.7, one direction explaining ≥ 70% of the three
  axes' class separation; the SGN axis (CE21) relates to the same rail
  (descriptive).
- **Falsifier**: the three axes are mutually ~orthogonal (|cos| ≤ 0.3) — the
  shared combiner reads class-specific rails through different input
  directions.
- **Alternatives**: two rails (carry vs borrow-family) with NEG sharing the
  borrow rail; route-specific axes (ADD residual-borne vs SUB/NEG
  attention-borne, CE20/CE25, could carry geometrically distinct codes).
- **Tension with human**: gives C2-vs-A7 a finer resolution: C2-like
  separation for *control*, A7-like reuse for *data*.
- **Confidence**: **single-rail REFUTED → TWO-RAIL (descriptive) 2026-07-16**
  ([CE27](maths-claim-evidence.md#ce27), mixed model, dual-gated). The proposed
  single shared unit-adjust rail is not what the geometry shows: at the shared
  L2 combiner input **borrow (SUB) and neg-borrow (NEG) share one rail**
  (|cos| 0.90 combiner / 0.96 resid_pre) but **add-carry is ~orthogonal** to
  them (|cos| 0.20/0.22, 0.01/0.05), and one shared direction explains only
  0.66 of the three axes' separation (< the 0.70 bar). The pre-registered scheme
  returns "intermediate → report spectrum"; the descriptive reading is the
  alternative already named — **two rails: add-carry ⊥ subtract-borrow, with NEG
  reusing the borrow rail** (SGN orthogonal to all, |cos| ≤ 0.09). So the
  *carried datum* is low-D but **not one universal rail** — it splits by
  operation family. Caveat: ADD's separateness carries an untrained-decodable
  nuisance (mixed untrained decodes ADD's bit 0.68 vs SUB 0.51), so the ADD
  ⊥ borrow claim needs a nuisance-controlled re-fit before it firms. Was
  proposed → **low-medium** (two-rail descriptive; the C2-for-control /
  reuse-for-data framing survives only for the *subtraction family*).

## Sharpest forks

Where discriminating evidence would most cheaply reshape this file (ranking
itself belongs in [maths-next-steps.md](maths-next-steps.md)). Reranked
2026-07-16 after CE16 (A10 items i–iii resolved — canonical carry message;
distributed ST-cluster source, never `=`; head-pair effective and
class-necessary carrier; trails in the A10 confidence block):

1. ~~The compounding locus (A11 vs L1-read)~~ — **CONCLUSIVELY SETTLED (CE24):
   the canonical carry emerges in the L1 read** (cross-depth transfer 1.00 at the
   L1 combiner input, ~chance at L0's output; both models, 9-free). A11 positional
   L0 relay rejected/low. Supersedes the CE19 note below:
1a. ~~The compounding locus (A11 vs L1-read)~~ — **SETTLED (CE19): L1-read, not
   an L0 relay** (5d proven via the invisible-decorrelated discriminator with a
   per-depth readability control; 6d underpowered; linear-probe caveat). A11 not
   supported → low; A9 stays retired. Superseding the CE17 note below:
1b. ~~The compounding locus (A11 vs L1-read)~~ — **ATTEMPTED (CE17), inconclusive
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
3. **Cross-size tightness and role transfer (A12 / C6) — the new #1.** Does
   the layer-general SV interface hold at n=10/13, and does the redundancy
   census shrink as C6 predicts? This is also the best remaining lever on the
   parked compounding-locus fork: if larger models are tighter, the CE17
   assays that nulled under 6-digit redundancy (relay interchange, horizon
   decode) may finally discriminate A11-relay from L1-read there. Serves the
   paper's generalization claim directly.
4. **Paper hand-off timing** — deferred by the human 2026-07-16; must trigger
   no later than ~12 h before the deadline. Post-deadline: the mixed-model
   shared engine (A7 vs C2) and the neuron-level combiner decomposition (B2).
5. ~~The latent-geometry stream (G1–G4)~~ — **BOTH overnight studies LANDED
   2026-07-16.** G1 (storage factorization): **refuted** as a shared *stored* rail
   ([CE27](maths-claim-evidence.md#ce27)) — the common carry currency is a
   **read-time** transformation (consistent with CE24), not a stored 1-D coordinate;
   G4's single-rail form refuted, two-rail (borrow/neg-borrow share one, add-carry
   orthogonal) descriptive. G2 (dominance code): **partial**
   ([CE28](maths-claim-evidence.md#ce28)) — on the read-time rail the per-site writes
   *are* an ordered place-value dominance code (super-increasing gaps, cin-U) and a
   single STEP threshold provably computes TriAdd **at a middle digit**, but the
   worst-case certificate is empty at the **leading digit** (compressed top-gap,
   genuine) and additivity is model-dependent (clean d5, non-additive residual d6).
   G3 (dynamic-range law): **low/consistent** — the carry-axis 30→6 collapse *is*
   dynamic-range compression (re-explains CE18) but the strict decay law is not
   discriminated from irrelevance. Net for the paper's representation section: it can
   now make a **constructive but scoped** geometric statement (ordered dominance rail,
   read-time canonicalization, STEP-as-threshold) rather than a list of CE11/CE18
   negatives. **Remaining latent-geometry fork (post-deadline):** a genuine
   multi-depth certificate at n=10/13 (where G3 predicts the sharpest gap structure)
   to test whether leading-digit compression is the same physics as CE18's collapse.

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

## Reflection log

Per-study belief-update history is empirical/process content and is not kept here
(it duplicated the ledger and drifted this conjecture doc into a "we observed"
record). The chronological trail lives in the
[results ledger](maths-results-by-time.md) and the per-study notes under
`study-maths/`; durable claims and their evidence live in the
[claim-evidence map](maths-claim-evidence.md). Each conjecture above carries its
current confidence with CE backlinks; update those (post-gate) rather than
re-adding a chronological log here.
