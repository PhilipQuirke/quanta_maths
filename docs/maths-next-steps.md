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

| Priority | Status | Owner | Experiment | Updates |
| --- | --- | --- | --- | --- |
| 1 | designed, awaiting gate + overnight launch | **`maths` thread (geometry stream)** | [Latent-geometry stream: dominance certificate + rail factorization](#1-latent-geometry-stream-overnight-pair) | G1–G4, A3, A4, A8, C1/C2 |
| 2 | mostly done | **`maths` mixed thread** | [Mixed model: remaining follow-ups (SV mechanism + A7 landed)](#2-mixed-model-remaining-follow-ups-sv-mechanism--a7-landed) | A10, A6, depth |

Rerank note (2026-07-16 evening): the former entry 1 (compounding locus,
decisive A11 vs L1-read) **completed** — CE24/CE24-TF settled it conclusively
(L1-read; lazy propagation, eager local) — and is deleted per the agenda rules
(trail: [study-compounding-locus-v2.md](study-maths/study-compounding-locus-v2.md),
[CE24](maths-claim-evidence.md#ce24)). By human directive the `maths` thread's
new focus is the **latent-geometry stream**: a geometric, latent-space account
of the same arithmetic process — manifold shapes, inter-feature relations
(ST/SV), and whether the geometry itself does computational work. Conjectures:
[G1–G4](maths-conjectures-agent.md#new-stream-2026-07-16-latent-geometry-conjectures-g-series).

Entry-order note (2026-07-16 rerank after CE18; parallel-thread split by human
directive): the cross-size study **ran** ([CE18](maths-claim-evidence.md),
[study note](study-maths/study-cross-size-sv.md)) — the SV **role skeleton +
step combiner generalize d5→d13**, but **C6 is not supported** (redundancy is
intrinsic, does not thin with n), so **scale did NOT rescue the compounding
locus** (A11 stays low; R-mixed from CE17 stands). Two open threads remain and
the human has split them across **two parallel agents**:
- **Entry 1 (this `maths` thread)**: attack the **compounding locus** directly
  with a redesigned assay that overcomes CE17's two blockers (single-node
  interchange undetermined vs redundancy; consumer edge local-class-sufficient),
  rather than hoping scale would sharpen it.
- **Entry 2 (a separate thread)**: **replicate the SV findings on the mixed
  add/sub model** — begins the addition→mixed generalization the paper needs,
  and folds in the old A7/C2 shared-engine geometry question as a replication.

**Parallel-work protocol (both threads active at once)**: each thread owns its
own study note, `scripts/<name>.py`, and `results/<study>/` — no shared files.
Router docs (claim-evidence, results-by-time/summary/synthesis, conjectures) are
**append-only per thread**; add your CE entry / confidence-update block, do not
rewrite the other thread's. On any dirty-file collision in `git status`, assume
it belongs to the other thread and leave it. The paper hand-off (entry 3)
consolidates BOTH threads' outputs and triggers **no later than ~T-12h**.

### 1. Latent-geometry stream (overnight pair)

*(Owner: `maths` thread, geometry stream. Opened 2026-07-16 evening by human
directive; the compounding-locus predecessor completed via CE24.)*

**What to learn**: a geometric, latent-space account of the arithmetic the
mechanistic account already localizes — the **shape** of the storage manifolds
(per-digit ST tri-states; the carry code), the **relations** between feature
manifolds (ST vs SV; carry vs borrow vs neg-borrow), and whether the geometry
**does computational work** (does the manifold arrangement itself implement the
cascade?). Conjectures
[G1–G4](maths-conjectures-agent.md#new-stream-2026-07-16-latent-geometry-conjectures-g-series).

**Why now**: the paper hand-off's section (b) is currently a list of scoped
negatives (no dominant embedding circle CE1; no ST transfer CE11; non-orthogonal
slots CE12; probe-limited large-n CE18). G1/G2 would flip those into one
constructive geometric story — rail-and-address storage + a place-value
dominance code that the STEP combiner (CE17) reads — and G3 re-explains the
CE18 collapse as a dynamic-range law. High payoff, cheap (forward passes +
linear algebra on existing HF artifacts, tested library helpers), overnight-
runnable before the hand-off.

**The two designed studies** (pre-run notes written; shared stimulus grids and
caches):

- **[study-geometry-certificate.md](study-maths/study-geometry-certificate.md)**
  (scores G2, G3; touches A3/A5/A10): per-prompt OV/LN decomposition of the
  combiner input along the CE16/CE17 carry rail into per-source-site
  contributions; test class-value ordering (`p_i < u_i < q_i`, U cin-split),
  weighted-gap dominance, additive α reconstruction, and the feasible-threshold
  **certificate** (is measured α* inside the interval that makes TriAdd
  provably correct to depth k?). Stretch: d10/d13 gap-compression law (G3).
- **[study-geometry-factorization.md](study-maths/study-geometry-factorization.md)**
  (scores G1, G4; touches A3/A4/A8/C1/C2): does OV-rail projection rescue the
  CE11 cross-digit no-transfer (storage factorization vs read-time rotation)?
  Does removing the rail explain the ST–SV 21° entanglement? Write-site
  manifold shape (collinear vs simplex). Mixed model: do SV/MV/NV share one
  unit-adjust rail (F4)?

**Launch checklist (per sprint protocol)**: (1) combined skeptic gate on both
pre-run notes in a separate thread; (2) implement
`scripts/geometry_certificate.py` + `scripts/geometry_factorization.py` from
the notes (reuse `maths_probe` / `maths_edge_patch` / `maths_cascade` /
`maths_temporal_finalization` helpers; smoke-test on d6 before the full run);
(3) run overnight (d5/d6 + mixed; d10/d13 stretch), results to
`results/study-geometry-*/`; (4) morning: score G1–G4, post-result gate, CE
entries, fold into entry 3's section (b) **before the ~T-12h hand-off
trigger**.

Done when: G1 and G2 are each scored confirmed / refuted / scoped-partial
against the pre-stated bars (with G3/G4 scored where the stretch batteries
ran), and the verdicts are folded into the entry-3 hand-off — or the window
closes, in which case hand off section (b) unchanged and park the stream
post-deadline.

### 2. Mixed model: remaining follow-ups (SV mechanism + A7 landed)

*(Owner: the `maths` mixed thread. All sub-steps here are **mixed model**
`ins1_mix_d6_l3_h4_t40K_s372001`; do not read them as addition steps.)*

**LANDED 2026-07-16 — CE20–CE23** (studies
[study-mixed-sv-replication.md](study-maths/study-mixed-sv-replication.md),
[study-mixed-opr-sgn.md](study-maths/study-mixed-opr-sgn.md),
[study-mixed-sv-implementation.md](study-maths/study-mixed-sv-implementation.md),
[study-mixed-shared-engine.md](study-maths/study-mixed-shared-engine.md); plan
[study-mixed-plan.md](study-maths/study-mixed-plan.md)):
- **SV representation + mechanism replicate across ADD/SUB/NEG** — writers encode
  the tri-state (ST/MT/NT); the resolved carry/borrow is a clean binary at the L2
  combiner input; the combiner is a **STEP** (α*≈0.5, endpoints gated); the
  delivered carry is a **canonical format-invariant** code; **`=` is not the
  middle-digit source** (CE20/CE22). Delivery is class-dependent (ADD residual-only;
  SUB/NEG residual + last-layer attention).
- **SGN** = the top-of-cascade `D≥D'` comparison delivered to the `=` combiner
  (CE15 analog, clean) (CE21).
- **A7-vs-C2 decided:** the engine is **shared at the combiner** (full-L1-state
  patch flips to the correct ADD digit 0.96) but **A7's low-rank/function-vector
  control is refuted** — a rank-1 operator steer flips 0% at both the combiner and
  the SLT selector, and no single head selects; the add/sub selection is a
  **distributed, high-dimensional L1 transformation** → leans **C2** on selection
  (CE21/CE23). C5 confirmed on a new architecture; A10 iv/i/ii + A12
  confirmed/strengthened; A6 not scored (redundancy); A7 → low (control),
  C2 → partially up.

**LANDED (i) 2026-07-16 — ≥2-depth delivery sweep (CE25;
[study-mixed-delivery-depth.md](study-maths/study-mixed-delivery-depth.md)):** the
class-dependent delivery pathway holds across depths 2/3/4 (ADD residual-only;
SUB/NEG residual + last-layer attention; deciding-matched null 0.00; untrained
control delivers nothing) — clears the CE14 ≥2-depth bar. The reusable sweep is now
in `quanta_maths/maths_cascade.py` (`make_cascade_operands`,
`combiner_delivery_sweep`) + tests.

**Remaining follow-ups (all `mixed model`; lower priority — the paper-critical
core has landed):**

- **Mixed model — (a) cross-model delivery sweep (zoo).** Run the promoted
  `combiner_delivery_sweep` across the mixed-model zoo (other seeds/sizes/layer
  counts) — the human's stated future use; the delivery *route* (ADD residual vs
  SUB/NEG attention) may differ by model. Cheap (library + forward passes). Scores
  **A10, A12** universality.
- **Mixed model — (b) clean class-necessity instrument.** CE22 B2 was
  redundancy-blurred + cascade-stimulus-confounded (A6 not scored). Redesign with
  a genuine cascade-depth contrast (non-trivial NEG answers) and digit-only
  accuracy; test class-level (grouped) necessity of the ST/MT/NT writers. Scores **A6**.
- **Mixed model — (c) rank-r operator subspace steer.** CE23 refuted *rank-1*
  additive control; the residual A7 escape hatch is a learned **rank-r** operator
  subspace. Fit it and test whether a low-rank (r≪d) steer selects the readout —
  bounds how compact the control can be. Scores **A7, C2**.
- **Mixed model — (d, optional) representation binding + d3 writer pin.**
  SV/MV/NV slots at `=` (tape-vs-register, CE11/CE12 analog) and pin the weak d3
  tri-state writer locus (CE20 caveat). Scores A4/A8-family on mixed.

Done when: (a) the cross-model sweep is scored, or the mixed line is frozen.
(b)–(d) are stretch. The landed CE20–CE23, CE25 are the mixed deliverables. Owns
`study-maths/study-mixed-*.md`, `scripts/mixed_*.py`, `results/study-mixed-*/`,
`quanta_maths/maths_cascade.py`.

### 3. Paper hand-off consolidation and referee checkpoint

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
with referee tagging, handed to the paper thread in time. Consolidates BOTH
parallel threads (the completed compounding-locus line — CE24 — plus entry 2
mixed-model replication) once each lands; trigger no later than ~T-12h even if
one thread is still in flight (hand off what exists, tagged by confidence).
If the geometry stream (entry 1) lands in time, fold the G1–G4 verdicts into
section (b) — confirmed G-claims upgrade it from scoped negatives to a
constructive geometric account; refuted ones ship as the (equally concrete)
read-time-rotation / interaction-read alternative.

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
- **B14 — Temporal finalization (TF) across the model zoo.** The token-time
  eager-vs-lazy result (CE24 TF addendum: single-step eager in-place at L0,
  multi-digit propagation lazy at the answer read L1) is on two small addition
  models; the human flagged the answer may differ by model/size. The reusable
  cross-model tool ships in the package —
  `quanta_maths.maths_temporal_finalization.run_temporal_finalization(model_name)`
  (layer-general, unit-tested) — so run it across d7/d8/d9/d10/d13 (and the mixed
  model) and compare the deferral/onset-layer. Cheap (forward passes + linear
  probes, no training). Updates A11/A12 (does the eager/lazy split tighten or
  shift with size?).
- **B15 — Maximal per-model map JSON completeness (visualization pipeline).**
  The auto-generated mechanism docs (`results/maps/<model>_mechanism.md`, built
  from `results/maps/<model>.json` via `scripts/gen_model_maps.py` →
  `scripts/gen_mechanism_docs.py`; see
  [mixed_model_mechanism.md](mixed_model_mechanism.md)) are structural-only and
  currently miss fields the hand exemplar carried. Enrich the maximal map JSON
  (and, upstream, the HF verified maps / `maths_hf_update.py` techniques) with:
  (a) the map-absent cascade-resolver `SV`/`MV`/`NV` role tags (studies
  study-mixed-sv / study-sv-implementation located these — write them into the
  map so the logical diagram stops drawing a generic resolver box);
  (b) the combiner `STC`/`MTC`/`NTC` `Algo` tags (currently the combiner is only
  inferred from high-`Fail%` last-layer answer-position MLPs);
  (c) model provenance `init_from` (e.g. `ins1_*` initialised from a d6 addition
  model), derivable from the name/HF metadata;
  (d) per-class positive-control accuracy for non-mixed models (mixed already has
  it via `scripts/mixed_map.py`).
  Pure tooling/evidence-hygiene, no new experiment — do it when the auto-doc
  needs to reach hand-exemplar parity for the paper hand-off (entry 3). Bundles
  with a cross-zoo `gen_model_maps.py` run once HF auth is confirmed.

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
