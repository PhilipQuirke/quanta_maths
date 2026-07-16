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

- **SPRINT MODE until the paper-revision deadline (~2026-07-18, set
  2026-07-16)**: the revision must include (a) details of the SV mechanism and
  (b) insights into the latent-space representation of intermediate results.
  Rank by contribution to those two deliverables; depth on the SV account
  beats breadth for this window. Framing follows the
  [working axioms](maths-conjectures-agent.md#working-axioms) (attribution,
  not existence; redundancy is the norm; class-level necessity). Proposed
  sprint process, pending human sign-off: **one combined skeptic pass per
  study** (pre+post in a single Opus thread) instead of two separate gates;
  evidence-integrity rules (committed script → `results.json`) unchanged.
- Until a conjecture is strongly evidenced, prefer breadth and cheap
  discriminators over depth (contract default; suspended during sprint mode).
- **Start from the paper's verified per-model node maps** (Hugging Face
  `<model>/behaviors.json` + `features.json`: `Algo:` roles, `Fail%`,
  per-answer-digit `Impact`, attention targets, `SP` tri-state-PCA tags; see
  [hugging_models.md](hugging_models.md) and
  [useful_tags.md](useful_tags.md)). Design assays *at map-named nodes*;
  re-locating or re-verifying the paper's nodes is a by-product, not a goal
  ([C5](maths-conjectures-human.md#c5-the-paper-empirical-results-are-reliable)).
- Rank by information-per-cost against the C5 five-step program and the
  sharpest live human/agent forks in
  [maths-conjectures-agent.md](maths-conjectures-agent.md#sharpest-forks):
  what the map-named nodes write (step 1), the SV compounding rule at the
  map-named wires (steps 2–4, A10 with A6/A9 as variants), the leading-digit
  hard case (step 5), and the mixed-model shared engine.
- Front-load quick wins that double as instrument calibration
  (weights-only or small forward-pass studies on existing Hugging Face
  artifacts) before building patching harnesses or training anything.
- Addition model first, mixed model second, per the scope in
  [maths-conjectures-human.md](maths-conjectures-human.md#model-scope-for-this-direction).
- Every entry names the conjecture predictions it bears on
  ([C1–C3](maths-conjectures-human.md#current-conjectures),
  [A1–A10](maths-conjectures-agent.md#current-conjectures)) so post-run
  prediction scoring is mechanical.

## Ranked queue

| Priority | Status | Experiment | Updates |
| --- | --- | --- | --- |
| 1 | ready | [SV implementation sprint: edge message, source, path shares, necessity](#1-sv-implementation-sprint-edge-message-source-path-shares-necessity) | A10, A9, A6 |
| 2 | ready | [Paper hand-off consolidation and referee checkpoint](#2-paper-hand-off-consolidation-and-referee-checkpoint) | (synthesis) |
| 3 | sequenced | [Mixed-model shared-engine geometry](#3-mixed-model-shared-engine-geometry) | A7, C2 |

Entry-order note (2026-07-16 sprint rerank — deadline-driven, after the human's
over-caution feedback): a paper revision is due in ~40 hours (~2026-07-18) and
must include **details of the SV mechanism** and **insights into the
latent-space representation of intermediate results**. The C5 addition-model
program is complete (CE13 output encodings → CE14 SV compounding → CE15
leading-digit walkthrough; trails in the ledger and study notes), and the
[working axioms](maths-conjectures-agent.md#working-axioms) now govern framing
(attribution not existence; redundancy is the norm) — under them A10's core
wiring was raised to medium-high by cross-study aggregation (see the
2026-07-16 reflection-log entry). CE14's three open follow-ups become sprint
**entry 1** (they are exactly what turns the wiring diagram into an
implementation description); the contract-mandated referee checkpoint merges
into the paper hand-off consolidation (**entry 2**); mixed-model defers past
the deadline (**entry 3**).

### 1. SV implementation sprint: edge message, source, path shares, necessity

Turn the confirmed SV wiring (CE13–CE15, A10 at medium-high) into an
**implementation description** for the paper — A10's four open items, all
reusing the CE14 harness (hours each, parallelizable; frame per the working
axioms — these estimate parameters of a mechanism known to exist):

- **Edge message decode (A10 item i)**: the causal head→combiner edge flips
  answers carry-specifically (CE14); decode what the edge *contribution*
  carries — compound carry vs deciding-digit class vs U-flag — against the
  CE5/CE6 carry axes at the combiner input.
- **Source attribution (A10 item ii)**: where do the heads read it?
  Key-source ablation / per-source decomposition over `=` vs the deciding ST
  site vs distributed. CE12 found all-digit `SV` decodable at `=` and every
  consumer head attends `=` heavily — if the message is read off `=`, the
  compounding largely happens pre-L1 and `=` is the carry depot (the
  human/paper lean); if off the deciding ST site, A9 selection revives.
- **Path shares + class-level necessity (A10 item iii)**: the powered
  carry-specific direct-path arm and the paired H1+H2 ablation (CE14's
  mandated follow-ups) — replace "not excluded"/"not necessary" with numbers
  (head-pair share vs direct share; joint necessity of the head class).
- **Stretch — combiner functional form (A10 item iv, B2-lite)**: fit
  `carry_out = f(delivered carry, local sum class)` at the combiner MLP;
  report neuron sparsity if time allows.

Done when: the SV account states, with numbers, what message flows on which
edges from which sources with what path shares and class-level necessity —
scored against A10 (items i–iv), A9 (revived or retired), A6.

### 2. Paper hand-off consolidation and referee checkpoint

Produce the maths-thread deliverable for the paper revision, ready **≥ 12
hours before the deadline**: update
[maths-results-summary.md](maths-results-summary.md) /
[maths-results-synthesis.md](maths-results-synthesis.md) (and claim-evidence
as entry 1 lands) into two paper-ready sections with per-claim confidence and
artifact links: (a) **the SV mechanism** — ST writes (local class +
single-step U-resolution, CE13) → redundant SP-tagged L1 head pair delivering
the carry carry-specifically into the answer-position combiner at every
answer digit including the sign position (CE14/CE15) → resolved `carry_out`
on the CE5 centroids → just-in-time answer-digit computation (CE12/CE13) —
plus entry 1's implementation numbers; (b) **latent representation of
intermediate results** — near-isotropic categorical digit embeddings with a
weak seed-fragile circular ordering (CE1); binary linear carry code with no
dedicated `U` symbol at answer positions, resolution applied around
L1-attention (CE6/CE7); position-specific question-side ST writes vs a shared
answer-side template (CE11/CE12/CE13); ST/SV geometric entanglement and
non-orthogonal `=` carry slots (CE11/CE12). Fold the contract-mandated
**adversarial referee report** (confirmed / partial / open tag per claim,
plus the cheapest picture-changing follow-ups) into the same document as its
caveats section — one artifact, two uses. Paper edits themselves remain the
`paper` thread's job; this entry is the evidence hand-off.

Done when: the consolidated two-section account exists in the results docs
with referee tagging, handed to the paper thread in time.

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
than assumed from the embedding. It should inherit instruments and geometry
vocabulary from the completed addition-model studies (CE1, CE13/CE14/CE15 — the
edge-patch, tricase/PCA, deciding-matched null, ablation-vs-baseline) rather than
develop its own, per the addition-first scope. Best done after the entry-2
consolidation checkpoint so the mixed model builds on a reviewed addition story.

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
  signatures. Updates the alternatives inside A1 and A2. *(The combiner-MLP
  functional-form half is the stretch battery in sprint entry 1.)*
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
  it. The *deep-chain* edge path-patch (L1-head→combiner) is now **queue entry 2**;
  this backlog item is the narrower **single-digit** hand-off (L0-conduit→L1-MLP
  edge) to confirm the two-site "compute-low, apply-at-answer" mechanism directly —
  clean in 6-digit, borderline in 5-digit. Fold into entry 2 if the harness covers
  both. Updates A6, A2.
- **B12 — Question-position transient `U` probe.** *(Folded into queue entry 1,
  2026-07-15, after C5 — the map-named `ST` nodes are the question-position
  sites, so their output-encoding study is this probe.)*
- **B13 — Task-wide effective dimensionality and dictionary recovery.**
  *(Demoted from the queue 2026-07-15 after C5 — not on the C5 critical path.)*
  Measure residual-stream effective dimensionality across the task per
  layer/position against the known feature inventory, and check whether a
  dictionary-learning decomposition (`QMSAE` infrastructure) recovers that
  inventory without heavy splitting. Direct test of A8 and C1's
  low-dimensionality half; revisit once the C5 steps land (its interpretation
  will then have the named-node encodings to compare against). Updates A8, C1.

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
