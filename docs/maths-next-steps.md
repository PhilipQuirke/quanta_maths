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
  [A1–A9](maths-conjectures-agent.md#current-conjectures)) so post-run
  prediction scoring is mechanical.

## Ranked queue

| Priority | Status | Experiment | Updates |
| --- | --- | --- | --- |
| 1 | ready | [Output encoding of the map-named ST/SA/SC nodes](#1-output-encoding-of-the-map-named-stsasc-nodes) | A2, A3, A10 |
| 2 | ready | [SV compounding rule at the map-named wires](#2-sv-compounding-rule-at-the-map-named-wires) | A10, A9, A6, A5 |
| 3 | sequenced | [Leading-digit hard-case walkthrough](#3-leading-digit-hard-case-walkthrough) | A10, A6 |
| 4 | sequenced | [Mixed-model shared-engine geometry](#4-mixed-model-shared-engine-geometry) | A7, C2 |

Entry-order note (2026-07-15 rerank, after the human C5 reflections — no new
empirical result): C5 redirects the thread to **start from the paper's verified
per-model node maps** and execute its five-step program (node output encodings →
SV mechanism → answer generation → the leading-digit hard case → mixed models),
rather than re-locating nodes or running paper-disconnected breadth studies. The
maps for both studied models were read and already name the SV-candidate wiring
(SP-tagged answer-position L1 heads attending `=` + the question-tail ST sites,
feeding the high-`Fail%` L1 MLPs) that CE9/CE10 groped toward. Queue rebuilt
around C5 (new entries 1–3, absorbing the old hand-off re-test into entry 2);
the effective-dimensionality/SAE breadth entry is demoted to backlog **B13**
(not on the C5 critical path). Trail:
[C5](maths-conjectures-human.md#c5-the-paper-empirical-results-are-reliable),
[A10](maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster),
and the 2026-07-15 C5 entry in the agent
[reflection log](maths-conjectures-agent.md#reflection-log-optional).

### 1. Output encoding of the map-named ST/SA/SC nodes

C5 step 1: how each *verified* "output only" node stores/outputs its sub-task
value. For the map-named nodes in both studied models (`ST` at
question-tail/sign L0 heads — 5-digit `P9L0H1`=A2.ST, `P10L0H1/H2`=A1.ST,
`P11L0H2`=A0.ST, `P12L0H1`=A3.ST, `P6L0H2`/`P12L1H2`=A4.ST; 6-digit
`P10L0H2`..`P12L0H2`, `P14L0H1/H2`; `SC`/`SA` at answer-position L0 heads),
characterize the node's *write* — its head-output/OV-projected residual
contribution — as a function of the sub-task value, using
**cascade-exercising stimuli** (not the no-lower-carry family that made ST
nodes look inert). This entry also **re-examines CE3's "paper ST candidates
not causal" reading**, which the maps contradict on random questions
(`P11L0H2` Fail 23%, Impact A5..A1), and it is the correct locus for two
long-parked items: A2's pre-MLP sum-sufficiency (satisfying the pair-sum
freeze's reopen condition at the named ST heads, including the mandated
outside-view sweep) and A3's question-position transient-`U` remnant (B12
folds in — the ST nodes *are* the question-position sites).

Reuse the library's own instruments rather than rebuilding (pointer from the
human, 2026-07-15): the `SP` tags were generated by **QMAnalyse Part 19A** —
PCA of a head's output over the pre-built **ST8/ST9/ST10 tricase question
groups** (`cfg.tricase_questions_dict`, from
`quanta_maths/MathsTestQuestions/tricase_test_questions_generator.py`), via
`calc_pca_for_an` / `manual_nodes_pca` in
[`quanta_maths/maths_pca.py`](../quanta_maths/maths_pca.py) (background:
[pca.md](pca.md)). The ST search/confirm filters and intervention test live in
`add_st_functions` in
[`quanta_maths/maths_search_add.py`](../quanta_maths/maths_search_add.py) —
note its clean question is the all-nines `333...+666...=999...`, i.e. the
paper's own ST test is already **cascade-exercising**, unlike our
no-lower-carry stimulus. This entry extends that machinery from 2-D PCA
clusters to full-space output encoding with nulls, per this thread's
instrument standards.

Done when: per-node output-encoding verdicts exist for the named `ST`/`SA`/`SC`
nodes in both models; A2 and the A3 remnant are scored at the right locus; and
the CE3 dismissal is confirmed or corrected under map-consistent stimuli.

### 2. SV compounding rule at the map-named wires

C5 steps 2–4, testing
[A10](maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster):
(i) **value content** — what the SP-tagged answer-position L1 heads actually
read from the ST sites and from `=` (the maps show every consumer attending
`=`, where no useful nodes are listed — content or sink?); (ii)
**deciding-digit propagation** — does an ST-node patch in a deep chain reach
the combiner through the head→MLP edge, at **≥ 2 depths**, using the
less-damped / multi-position instrument the CE10 skeptic mandated (the old
"hand-off higher-power re-test" is absorbed here, as is B11's single-digit
hand-off if the harness covers both); (iii) the **A10 variant split** —
direct-path (a) vs selection-within-cluster (b, A9) vs static weighted read
(c) — and (iv) the **selective economy** test (A6: harm cascade questions,
spare carry-free) at the named heads. Pair with neuron-level analysis (B2) of
how the head + combiner MLP compute `carry_out` where the budget allows.
For (i), seed the value-content probes with the same tricase machinery that
generated the `SP` tags (QMAnalyse Part 19A / `maths_pca.py` — see entry 1's
reuse note): the `SP` evidence is 2-D PCA clustering of these heads' outputs
over ST8/ST9/ST10 groups; this entry upgrades it to full-space content plus
OV-path transport into the combiner.

Done when: a powered multi-depth result attributes the carry delivery among
A10's variants (or explicitly none), scored against A10, A9, A6, A5.

### 3. Leading-digit hard-case walkthrough

C5 step 5: document, per model, how the **first non-static answer token** is
generated in a hard edge case (`99999+00001=`, `999999+000001=`): which
map-named nodes carry it (sign-token L0 ST nodes A4/A5.ST, the SP-tagged L1
heads at the sign position, the sign-position L1 MLP), with each link causally
verified by the entry-1/2 instruments or explicitly marked inferred. The
deliverable is an end-to-end, node-by-node documented trace — the C5 step-5
artifact and the eventual paper-thread worked example. Status `sequenced`:
consumes entries 1–2's instruments and findings.

Done when: a per-model account of the leading digit in the hard edge case
exists with verified/inferred status per link, scored against A10 and A6.

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
