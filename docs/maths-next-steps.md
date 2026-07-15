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
  the deep-cascade mechanism (sequential vs selection vs wide-fetch), where
  discretization happens, causal use of digit geometry, and the mixed-model
  shared engine.
- Front-load quick wins that double as instrument calibration
  (weights-only or small forward-pass studies on existing Hugging Face
  artifacts) before building patching harnesses or training anything.
- Addition model first, mixed model second, per the scope in
  [maths-conjectures-human.md](maths-conjectures-human.md#model-scope-for-this-direction).
- Every entry names the conjecture predictions it bears on
  ([C1–C3](maths-conjectures-human.md#current-conjectures),
  [A1–A9](maths-conjectures-agent.md#current-conjectures)) so post-run
  prediction scoring is mechanical.

## Ranked queue

| Priority | Status | Experiment | Updates |
| --- | --- | --- | --- |
| 1 | ready | [Task-wide effective dimensionality and dictionary recovery](#1-task-wide-effective-dimensionality-and-dictionary-recovery) | A8, C1 |
| 2 | ready | [Deep-cascade hand-off, higher-power re-test](#2-deep-cascade-hand-off-higher-power-re-test) | A9, A6, A5 |
| 3 | sequenced | [Mixed-model shared-engine geometry](#3-mixed-model-shared-engine-geometry) | A7, C2 |

Entry-order note (2026-07-16 rerank, after the answer-binding study,
[CE12](maths-claim-evidence.md)): the answer-binding entry (old entry 1) **ran** and
delivered a clean cross-model split — `SA` is a just-in-time register, `SV` is
resolved/present at `=` in a non-orthogonal (not-tape) layout, and both share an
answer-side template (A4 just-in-time supported, tape refuted; C2/A8
non-orthogonality reinforced). Old entry 1 deleted (trail in the study note +
ledger + CE12). The two remaining breadth/depth entries shift up:
effective-dimensionality is now #1 (the CE11/CE12 non-orthogonality findings make
its A8 low-rank test especially pointed), the cascade hand-off higher-power re-test
is #2, mixed-model is #3. Trail: [CE12](maths-claim-evidence.md), the 2026-07-16
answer-binding entry in the agent
[reflection log](maths-conjectures-agent.md#reflection-log-optional).

### 1. Task-wide effective dimensionality and dictionary recovery

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

### 2. Deep-cascade hand-off, higher-power re-test

The edge path-patch ([CE10](maths-claim-evidence.md),
[study note](study-maths/study-cascade-handoff-edge-patch.md)) gave a **one-depth
causal crumb**: `L1.H1` (the CE8 routing cell CE9 found inert) causally drives the
answer-position combiner at 6-digit k=3, carrying a computed, deciding-selective
carry through the combiner MLP. But the single-position edge instrument is
**underpowered at most cells** (LN renormalizes a single head's additive edge
against the whole residual), no head cleared the ≥2-depth bar, and the 5-digit
direct residual path is causally live — so A9's attention-delivery is
circumstantial, not confirmed, and A6 is not refuted.

Re-test the hand-off with a **higher-power / less-damped instrument** — patch the
candidate head's contribution across the affected digit's *whole* consuming path
(or several positions jointly), or normalize-out the LN damping — to test whether
`L1.H1`→combiner delivery holds at **≥ 2 depths** (add 6-digit k=4) and whether the
5-digit direct-path contribution is real. Pair with neuron-level analysis (B2) of
how `L1.H1` + the combiner MLP compute `carry_out`. Sequence *after* the breadth
entries 1–2 (depth-budget: the cascade line has had three consecutive studies).

Done when: a powered result shows whether the `L1.H1`→combiner hand-off replicates
across depths (confirming or bounding A9's attention-delivery), scored against A9,
A6, A5.

### 3. Mixed-model shared-engine geometry

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
- **B4 — Coexistence census at `=` and answer positions.** *(Promoted into the
  queue as entry 1, 2026-07-16, after CE11.)*
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
  can. Analyze how it resolves the nines cascade — a forced wide-fetch or
  one-hop selection would be an existence proof for A6's alternatives
  (including A9). Updates A6, A9.
- **B11 — L0→L1 U-resolution hand-off path-patch (single-digit `U`).** CE5 shows
  layer-0 nodes conduct the running carry and the answer-position L1 MLP combines
  it. The *deep-chain* edge path-patch (L1-head→combiner) is now **queue entry 1**;
  this backlog item is the narrower **single-digit** hand-off (L0-conduit→L1-MLP
  edge) to confirm the two-site "compute-low, apply-at-answer" mechanism directly —
  clean in 6-digit, borderline in 5-digit. Fold into entry 1 if the harness covers
  both. Updates A6, A2.
- **B12 — Question-position transient `U` probe.** CE6/CE7 refuted a dedicated
  `{0,1,U}` symbol across *answer-position* residual sites (carry is binary,
  resolved around L1-attention). The only live A3 remnant is a *transient* `U`
  at **question positions** (D'n), where the carry is being built before it is
  resolution-split — needs a purpose-built construction (the answer-position
  4-class assay does not place the pre-resolution carry state at question
  positions). Low expected yield given the four-study streak, but it is the one
  regime where A3 can still live. The deep-chain constructions built for
  queue entry 1 are the natural stimulus base for this probe. Updates A3
  (final disposition).

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
