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
| 1 | ready | [Cross-position and cross-subtask probe transfer](#1-cross-position-and-cross-subtask-probe-transfer) | C2, A4, A8 |
| 2 | ready | [Cascade tracing and inter-position patching](#2-cascade-tracing-and-inter-position-patching) | A6, C3, A4 |
| 3 | ready | [Task-wide effective dimensionality and dictionary recovery](#3-task-wide-effective-dimensionality-and-dictionary-recovery) | A8, C1 |
| 4 | sequenced | [Mixed-model shared-engine geometry](#4-mixed-model-shared-engine-geometry) | A7, C2 |

Entry-order note (2026-07-16 rerank #9, after the attention-invariance census):
the census is **complete** — **A5's strong static-wiring form is falsified /
narrowed to hybrid**: a few heads (`L1.H1` operand-read Q11 & answer Q14;
`L0.H0` answer Q17) relocate their attention target with the carry state
(value-matched contrast, clean & Bonferroni-safe in the 6-digit model; 5-digit
inconclusive; representational not causal) ([CE8](maths-claim-evidence.md)). This
is a **genuine new structural fact** (breaking the A3-family refutation streak).
The carry-routing cells become candidate content-routed nodes for the
**cascade-tracing** entry, which must confirm them *causally* (pattern-patching).
Cross-position/cross-subtask probe transfer (C2/A4/A8) promotes to #1 — a compact
assay-shared study scoring three conjectures. Trail in the
[results ledger](maths-results-by-time.md) and
[study note](study-maths/study-attention-invariance.md).

### 1. Cross-position and cross-subtask probe transfer

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

### 2. Cascade tracing and inter-position patching

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
**New input from CE8**: the attention-invariance census found **carry-state
target-routing** heads (6-digit `L1.H1` Q11/Q14, `L0.H0` Q17;
[registry](../results/study-attention-invariance/attention_routing_registry.json)) —
these are candidate content-routed cascade nodes and should be **confirmed
causally** here by patching the *attention pattern* (not just the value path).
Do not inherit the "CE5-combiner-locus" gloss unchallenged (CE8's cleanest cell
is an operand-read position, not an answer position).

Done when: decodability-by-position tracing plus per-class patching effects
jointly select one of the three cascade stories (or explicitly none), scored
against A6, C3, and A4's just-in-time-fetch prediction.

### 3. Task-wide effective dimensionality and dictionary recovery

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

### 4. Mixed-model shared-engine geometry

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
- **B11 — L0→L1 U-resolution hand-off path-patch.** CE5 shows layer-0 nodes
  conduct the running carry and the answer-position L1 MLP combines it. Directly
  test the hand-off with an edge path-patch (freeze the L1 MLP's other inputs,
  vary only the L0-conduit→L1-MLP edge) to confirm the two-site "compute-low,
  apply-at-answer" mechanism — clean in 6-digit, borderline in 5-digit, so worth
  a direct edge test. Also test A6's **tie-break economy** on multi-digit
  `...999` cascades (single-digit `U` only so far). Updates A6, A2.
- **B12 — Question-position transient `U` probe.** CE6/CE7 refuted a dedicated
  `{0,1,U}` symbol across *answer-position* residual sites (carry is binary,
  resolved around L1-attention). The only live A3 remnant is a *transient* `U`
  at **question positions** (D'n), where the carry is being built before it is
  resolution-split — needs a purpose-built construction (the answer-position
  4-class assay does not place the pre-resolution carry state at question
  positions). Low expected yield given the four-study streak, but it is the one
  regime where A3 can still live. Updates A3 (final disposition).

## Frozen lines

Each freeze carries a classification (`question answered` / `question failure`
/ `instrument failure`) and an explicit reopen condition.

- **Pair-sum sufficiency via answer-position attention+ablation selection**
  (2026-07-14) — **`instrument failure`**. The assay selected answer-position
  operand-fetch heads (not confirmed question-position `ST` compute nodes), and
  its no-lower-carry stimulus pinned `R²_pair=1` so the sum/pair ratio could not
  separate aggregation from transport (a noise-free transport null reproduced
  the whole signature). A2 untested. Trail:
  [study note](study-maths/study-pair-sum-sufficiency.md),
  [CE2](maths-claim-evidence.md). **Reopen** only via a confirmed compute node
  (now available — the make-carry heads CE3 and the L1-MLP `U`-combiner CE5)
  *and* a discriminating metric that does not over-determine the ratio (compare
  head output directly against the transport null; include cascade-varying
  context). Per the contract, an `instrument
  failure` freeze triggers an **outside-view sweep** before the reopened
  experiment: check how the arithmetic-interpretability literature
  (e.g. Nanda 2023 Fourier features, Zhong 2023 Clock/Pizza, Kantamneni &
  Tegmark 2025) *distinguishes* "attention transports circular codes" from
  "attention computes" — since that is exactly the confound that sank this assay.

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
