# Maths Next Steps (maths-next-steps.md)

Read role and rules: [Experiment Agenda](thor-document-rules.md#experiment-agenda).

Ranked, current-facing agenda for the `maths` thread. This records what to do
next, not the history of what was done. Re-rank after each meaningful result
instead of appending notes. Delete completed entries (their trail lives in the
study note and [results ledger](maths-results-by-time.md)) and stay within the
agenda entry cap in [maths-agent.md](maths-agent.md).

Only the [Ranked queue](#ranked-queue) counts toward the entry cap. The
[Candidate backlog](#candidate-backlog-not-active) is a non-active idea pool;
promoting a backlog item into the queue requires deleting or demoting a queue
entry.

Each entry states what to learn and why, not how. Detailed design (data,
controls, positive control, power, success conditions) happens in the study
note under `study-maths/` per the
[study-note rules](thor-document-rules.md#study-notes), written by the working
agent at pick-up time. Entries are self-contained: an agent should be able to
design the study from the entry, the linked conjecture files, and the linked
reference docs, without this thread's conversation context.

## Ranking logic

- Until a conjecture is strongly evidenced, prefer breadth and cheap
  discriminators over depth (contract default).
- Rank by information-per-cost against the sharpest live human/agent forks
  listed in
  [maths-conjectures-agent.md](maths-conjectures-agent.md#sharpest-forks):
  geometry-vs-lookup, where discretization happens, the `U` shape, and
  carried-vs-wide-fetch cascade.
- Front-load quick wins that double as instrument calibration
  (weights-only or small forward-pass studies on existing Hugging Face
  artifacts) before building patching harnesses or training anything.
- Addition model first, mixed model second, per the scope in
  [maths-conjectures-human.md](maths-conjectures-human.md#model-scope-for-this-direction).
- Every entry names the conjecture predictions it bears on
  ([C1–C3](maths-conjectures-human.md#current-conjectures),
  [A1–A8](maths-conjectures-agent.md#current-conjectures)) so post-run
  prediction scoring is mechanical.

## Ranked queue

| Priority | Status | Experiment | Updates |
| --- | --- | --- | --- |
| 1 | ready | [Pair-sum sufficiency at an ST node](#1-pair-sum-sufficiency-at-an-st-node) | A2, C3 |
| 2 | ready | [Full-space ST tri-state geometry](#2-full-space-st-tri-state-geometry) | A3, C1 |
| 3 | ready | [LN-aware digit-embedding close-out (A-9)](#3-ln-aware-digit-embedding-close-out-a-9) | A1, C1 |
| 4 | ready | [Attention-pattern invariance census](#4-attention-pattern-invariance-census) | A5, C3 |
| 5 | ready | [Cross-position and cross-subtask probe transfer](#5-cross-position-and-cross-subtask-probe-transfer) | C2, A4, A8 |
| 6 | ready | [Cascade tracing and inter-position patching](#6-cascade-tracing-and-inter-position-patching) | A6, C3, A4 |
| 7 | ready | [Task-wide effective dimensionality and dictionary recovery](#7-task-wide-effective-dimensionality-and-dictionary-recovery) | A8, C1 |
| 8 | sequenced | [Mixed-model shared-engine geometry](#8-mixed-model-shared-engine-geometry) | A7, C2 |

Entry-order note (2026-07-14 rerank, after the digit-embedding study): the
former entry 1 (digit-embedding geometry audit) is **complete** —
verdict AMBIGUOUS, strong-form A1 refuted, weak circular *ordering*
provisionally confirmed; trail in the
[results ledger](maths-results-by-time.md) and
[study note](study-maths/study-digit-embedding-geometry.md). Because the token
embedding turned out to carry little dominant geometry, the mechanism studies
(pair-sum sufficiency, `ST` geometry) are now the sharpest cheap discriminators
and move to the top. The LN-aware close-out (new entry 3) is the required
finish of the digit-embedding line before its CE1 claim is de-provisionalized;
it is cheap (weights-only) but ranks below the two mechanism studies because it
only refines an already-ambiguous verdict.

### 1. Pair-sum sufficiency at an ST node

At a known [ST](thor-glossary.md#st) node where Paper 2's ablations say the
attention head and its MLP are *both* necessary, examine the representation
between head output and MLP input. A2 predicts the head output is the raw
aggregate — digit pairs with equal `Dn + D'n` should be near-identical there,
forming an ordered low-dimensional arc with the `U` case at the
`Dn + D'n = 9` point — while the familiar 3-cluster tri-state structure should
exist only after the MLP.

This is the crux of A2 (aggregate-then-discretize) and the second-sharpest
fork: it decides whether attention *performs* the addition in embedding space
or merely transports operands, and it would turn the paper's unexplained
"head and MLP jointly necessary" observation into a mechanism. It also feeds
entry 3, since it locates where the categorical shape is created.

Done when: equal-sum collapse is quantified against a matched unequal-sum
control, the pre-MLP versus post-MLP shape verdict is recorded, and the A2 and
C3 predictions are scored.

### 2. Full-space ST tri-state geometry

Measure the geometry of the `ST` output classes `{0, 1, U}` at one or two ST
nodes in the full residual-stream dimensionality, executing the human file's
own caveat that the existing 3-cluster [PCA](pca.md) evidence is only a
projection. The discrete candidate shapes, from
[A3](maths-conjectures-agent.md#a3-the-st-tri-state-is-a-2d-categorical-code-with-u-off-the-01-axis):
(a) a 1D ordered scalar with `U` between `0` and `1`; (b) two near-independent
binary directions (a carry bit and a sum-is-9 bit, echoing the legacy
`SC`/`SS` sub-tasks); (c) a 2D simplex-like categorical code with `U` off the
0–1 axis.

Each shape implies a different downstream `TriAdd` reading and a different
verdict on C1's "near-linear" wording, so this entry converts the thread's
best-known observation into its first real representation claim.

Done when: centroid separations and angles classify the shape into one of the
three candidates (or an explicit fourth), with full-space statistics, scored
against A3 and C1.

### 3. LN-aware digit-embedding close-out (A-9)

Finish the pre-registered digit-embedding design by repeating the geometry
statistics on the **LayerNorm-effective** embedding (fold the first LN into
`W_E`), the secondary analysis (amendment A-9) not run in the first pass. The
weak circular *ordering* signal that survived — significant in 3 of 4 accurate
models, absent in the untrained control — is exactly the metric most sensitive
to LN, so the CE1 claim's ordering half stays **provisional** until this lands.
Cheap (weights-only, reuses
[`scripts/digit_embedding_geometry.py`](../scripts/digit_embedding_geometry.py)).

Done when: raw-vs-LN agreement is reported per the A-9 rule (classified-read or
band-boundary change = disagreement → ambiguous), the CE1 ordering claim is
de-provisionalized or revised, and A1's ordering sub-claim is re-scored.

### 4. Attention-pattern invariance census

Measure how much attention patterns vary across a large, stratified question
set: per head, per token position, on an addition model. A5 predicts
near-invariance (static positional wiring) with data-dependence confined to
the value path and MLPs; the informative exceptions, if any, should sit at
cascade or selection nodes (for example when an `ST` input is `U` versus not).

This is a quick win: cheap forward passes, no interventions, and it produces
two reusable byproducts — the stratified question classes (carry-free,
single-carry, `U`-cascade chains) and a per-node attention baseline — that
entry 6 needs anyway. A clear invariance verdict also simplifies every later
patching design, because static routing means interventions can target the
value path with less confounding.

Done when: pattern variance is quantified per head and position with an
explicit exception list, scored against A5 and the routing half of C3.

### 5. Cross-position and cross-subtask probe transfer

Test C2's two halves separately. First, template sharing: does a linear
readout for `ST` trained at one digit position transfer to other positions
(and across nodes computing the same sub-task)? A4 predicts near-free
transfer, forced by weight sharing. Second, cross-subtask geometry: measure
subspace angles between
[SA](thor-glossary.md#s-addition-sub-tasks-sa-sc-ss-st-sv), `ST`, and
[SV](thor-glossary.md#sv) representations — C2 predicts near-orthogonality,
while A4 predicts the question matters less at question positions because
separation there is positional.

The transfer matrix plus angle table is a compact, largely assay-shared
deliverable that scores C2, A4, and the interference half of A8 in one study.
Disagreement between its two halves (templates shared but sub-tasks
entangled, or vice versa) would be the first genuinely new structural fact
this thread produces.

Done when: the transfer and angle results give explicit verdicts on both
halves of C2 and on A4's template claim, scored accordingly.

### 6. Cascade tracing and inter-position patching

Trace where the carry-cascade state lives and when it matters. First, trace:
at which token positions and layers the running `ST`/`SV` cascade state is
decodable, and whether its `U` component vanishes by `=` as both C3 and A6
predict. Second, intervene: patch the residual stream between `ST` producers
and downstream `SV` consumers, separately on `U`-cascade questions (long
`...999 + 1`-style chains) and carry-free questions.

The per-question-class effect pattern discriminates three live stories: A6's
tie-break-only economy (carried state matters only on `U`-cascade questions),
a pure pipeline (carried state matters everywhere), and the wide-fetch
alternative (no meaningful inter-position state; one late node computes the
whole cascade at `=`) — the carried-vs-wide-fetch fork. It is also the direct
test of C3's central claim and its falsifier. This is the heaviest
addition-model entry (patching harness plus question-class construction), so
it sits after the cheap studies that supply its node maps and question sets.

Done when: decodability-by-position tracing plus per-class patching effects
jointly select one of the three cascade stories (or explicitly none), scored
against A6, C3, and A4's just-in-time-fetch prediction.

### 7. Task-wide effective dimensionality and dictionary recovery

Measure the effective dimensionality of residual-stream activity across the
whole task distribution, per layer and position, and compare against the size
of the known feature inventory (per-digit `SA`/`ST`/`SV`, operator, sign).
Then check whether a dictionary-learning decomposition (the existing `QMSAE`
notebook infrastructure) recovers roughly that known inventory without heavy
feature splitting.

This is the direct test of A8 (no superposition pressure; complexity is
compositional) and the low-dimensionality half of C1, and it bounds how much
undiscovered structure the earlier targeted studies might have missed. It
ranks below the targeted geometry studies because its interpretation depends
on their verdicts — for example, any circular features from the digit-embedding
study would legitimately occupy two dimensions each.

Done when: an effective-dimension table and an inventory-match verdict exist
and are scored against A8 and C1.

### 8. Mixed-model shared-engine geometry

On the mixed model `ins1_mix_d6_l3_h4_t40K_s372001` (see
[mixed_model.md](mixed_model.md)), measure the geometry of polysemantic nodes
that Paper 2 says serve `SA`, [MD](thor-glossary.md#m-positive-answer-subtraction-sub-tasks-md-mb-mz-mt),
and [ND](thor-glossary.md#n-negative-answer-subtraction-sub-tasks-nd-nb-nz-nt):
how much do the per-operation readouts overlap, and does a compact
[OPR](thor-glossary.md#opr)/[SGN](thor-glossary.md#sgn) control direction
exist that selects among them?

This is where C2 and A7 make opposite predictions on the same measurement —
C2 expects different sub-tasks to be near-orthogonal; A7 expects heavy
`SA`/`MD`/`ND` overlap steered by low-rank control — so whichever way it
lands, a conjecture takes real damage. The digit-embedding study found no
dominant circular geometry, so A1's corollary that subtraction is addition with
a reflected operand is now tested at the *node/activation* level here rather
than assumed from the embedding. Status `sequenced`: it should inherit
instruments and geometry vocabulary from entries 1–2 (and the digit-embedding
study) rather than develop its own, per the addition-first scope.

Done when: overlap and control-direction verdicts exist for the shared nodes
and are scored against A7 and C2.

## Candidate backlog (not active)

One-paragraph candidates, not ranked and not counted against the entry cap.
Promote by swapping into the queue.

- **B1 — Causal embedding-geometry test.** The digit-embedding study found only
  a weak, correlational circular *ordering* (not a dominant geometry), so the
  causal question is now sharper: does the model *use* digit magnitude/circular
  ordering at all? Interventions that move a digit embedding along the
  hypothesized circle should shift outputs the way the geometry predicts if the
  ordering is load-bearing, versus breakage under a lookup story. **Sequence
  after entry 3 (LN-aware A-9)**, since whether the ordering survives LN
  determines whether a causal test is worth building. Updates A1 and its lookup
  alternative.
- **B2 — MLP discretization mechanism.** Zoom into how MLP neurons implement
  the `Dn + D'n → SA/ST` map at one node: neuron activation profiles as a
  function of pair sum, key-value-memory signatures versus Fourier-product
  signatures. Updates the alternatives inside A1 and A2.
- **B3 — Positional-embedding contribution audit.** Quantify how much of the
  QK attention computation is driven by positional components versus content,
  and whether stored features carry position-derived tags. Directly supports
  or undermines A4's binding-tag mechanism and A5's static wiring.
- **B4 — Coexistence census at `=` and answer positions.** Count how many
  resolved per-digit states are simultaneously decodable at a single position
  ("register versus tape"). Discriminates A4's just-in-time-fetch belief from
  the orthogonal-slots layout C2 implies at answer time.
- **B5 — Cross-seed and cross-size universality sweep.** Re-run the headline
  geometry verdicts (the digit-embedding study and entries 1–2) across the
  model zoo (seeds, sizes, layer
  counts), since Paper 2 documents node-level variability between models.
  Determines whether claims are about *these* models or this task; updates
  the confidence of A1, A3, and C1 claims before anything reaches the paper
  thread.
- **B6 — Subtraction tri-state mirror.** Measure the borrow-side tri-state
  ([MT](thor-glossary.md#m-positive-answer-subtraction-sub-tasks-md-mb-mz-mt))
  geometry the way entry 3 measures `ST`, and compare shapes. A1 predicts a
  reflection-related geometry; a mismatch would break the shared-engine story
  before entry 8 relies on it. Updates A1, A3, A7.
- **B7 — Legacy `SC`/`SS` remnant nodes.** Paper 2 notes some models retain
  legacy single-bit carry nodes. Check whether their outputs align with the
  two-binary-directions alternative in A3 — evidence that the tri-state code
  evolved from, or coexists with, a two-bit square code. Updates A3's
  alternatives.
- **B8 — Answer-readout mechanism.** At answer positions, examine how
  `(SA_n + carry) % 10` becomes logits: whether the readout uses the same
  digit geometry as the embeddings (A1's stage-5 prediction) and where the
  final mod-10 wrap happens. Updates A1 and A2 on the output side.
- **B9 — Training-dynamics emergence order.** Retrain one addition model with
  dense checkpointing and track when the digit geometry, tri-state clusters,
  and cascade behavior each emerge (grokking-style progress measures). High
  cost (training run, currently paused below); high payoff as an origin story
  for whichever geometry the queue confirms. Updates all of A1–A6.
- **B10 — One-layer model contrast.** The `add_d5_l1_h3_t30K`-style one-layer
  model cannot pipeline the cascade across layers the same way deeper models
  can. Analyze how it resolves the nines cascade — a forced wide-fetch would
  be an existence proof for A6's main alternative. Updates A6.

## Frozen lines

Each freeze carries a classification (`question answered` / `question failure`
/ `instrument failure`) and an explicit reopen condition.

- *None yet.*

## Deliberately paused

- **New training runs** (including the checkpoint-instrumented retrain B9
  needs): paused until the existing-artifact studies (entries 1–5 and the
  digit-embedding study) have
  calibrated instruments and confirmed the question is worth the compute.
  Reopen when a queue entry's design genuinely requires a new model.
- **Paper-facing work**: per the guardrail in
  [maths-conjectures-human.md](maths-conjectures-human.md#relation-to-the-published-paper),
  no changes to `study-paper/paper.tex` or paper framing until this research
  direction lands. Owned by the `paper` thread regardless.
