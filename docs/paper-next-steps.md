# Paper Next Steps (paper-next-steps.md)

Read role and rules: [Experiment Agenda](thor-document-rules.md#experiment-agenda).

Ranked, current-facing publication agenda for the `paper` thread. This records
what to publish next, not the history of what was published. Re-rank as pieces
ship; delete shipped entries.

## Active piece: reframe Paper 2 for BlackboxNLP 2026

**Venue**: BlackboxNLP 2026 (co-located with EMNLP 2026, Budapest), **archival
track = 8 pages + references + appendix**. ACL style files (`acl.sty`,
`acl_natbib.bst`); `\usepackage[review]{acl}` for submission (anonymized).

**New angle** ("what we can say with confidence"): a deep worked-example study of
the learned arithmetic algorithm — **logical** (sound exact left-to-right add/sub
algorithms) and **mechanical** (located + causally-confirmed circuits) — plus the
benefits of data enrichment and the similarity/variation across the model family.

**Calibration guardrail**: publish only claims at or below the confidence in
[maths-claim-evidence.md](maths-claim-evidence.md). Headline SV claim caps at the
CE16/CE17/CE24 (Medium-high) level: *step-function combiner integrates a
canonically-coded carry delivered by a redundant attention head-pair*; explicit
limit that **single-head carry selection is not established** and much is 2-layer /
linear-probe.

### Reporting structure ("~40 then 4")

- **Tier 1 — general (~40 accurate models)**: role skeleton found across the family
  via node-location + ablation maps (SA/SC/ST subtasks, redundancy, shared-across-
  heads). Source: accuracy tables + per-model analysis summaries.
- **Tier 2 — deep worked examples (4 models)**: swap to detailed causal reporting on
  **6d & 8d addition** and **6d & 8d mixed**. Detail → appendix; main body names
  the techniques. (Chosen over 6d/10d because both 8d models are clean 99.999% and
  exist today — no training run needed, unlike 10d mixed.)

### Model / evidence coverage (as of 2026-07-16)

| Target | 99.999%? | Deep SV battery | Map JSON | Gap |
| --- | --- | --- | --- | --- |
| 6d addition (`add_d6_l2_h3_t20K_s173289`) | yes (0) | full (CE13–24) | HF | none |
| 8d addition (`add_d8_l2_h3_t45K_s173289`) | yes (0) | **full — done** (confirmatory: STC combiner, CE2 transport, =-depot, class-necessity, step combiner, intrinsic redundancy, L1-locus, lazy-TF; source-fork probe-limited) | pending upload | upload maps |
| 6d mixed (`ins1_mix_d6_l3_h4_t40K_s372001`) | yes (0/0) | CE20–25 | HF + local | none |
| 8d mixed (`mix_d8_l3_h4_t60K_s173289`, from-scratch) | yes (7) | in progress (HO-2) | not on HF | run full battery; generate/upload map |

Notes:
- 8d mixed is a **from-scratch** model; the 6d mixed capstone (`ins1`) carries the
  insertion/polysemantic story. The 8d `ins1` initialized model is NOT clean
  (50/1196 fails), so mixed types cannot be matched at 8d — 8d mixed is the
  cross-size generalization point only.
- No clean 99.999% 10d mixed exists (from-scratch 295 sub fails; init 13 fails).
  Deferred; 8d chosen instead.
- HF holds analysis JSON for only ~14 models; the Tier-1 general claim rests on the
  accuracy-table analysis summaries, not full maps.

### Decisions still needed from human

1. **Spine confirmation**: 6d addition (core) + 6d mixed (capstone), with 8d as the
   cross-size generalization for both operations — confirmed direction, pending HO-2.

## Handoff tasks to the `maths` thread (this thread is read-only on experiments)

- **HO-2 — full SV battery on the two 8d worked-example models.**
  - **8d addition — DONE (2026-07-16).** Every SV finding replicates: STC combiner
    at all digits, CE2 transport, =-depot, class-necessity, step combiner, intrinsic
    redundancy, L1-locus, lazy-TF. SV interface now confirmed on **five addition
    sizes (d5, d6, d8, d10, d13)**. Cross-model variation: TF/CARRY deferral is
    **3 tokens on d8 vs 2 on d5/d6** (eager-local / lazy-propagation split holds; lazy
    gap grows modestly with size — the CARRY tag captures this). Caveat: carry axis
    weak (sep 5.3) so the deciding-ST source arm is probe-limited (0.00), as at
    d10/d13 — robust d8 evidence is step combiner + class-necessity + =-depot +
    L1-locus, NOT the source-fork. Artifacts: `results/study-d8-batteries/`
    (`d8_sv_summary.json`, `results.json`, `ce24_d8.json`),
    `results/study-d8-wired-techniques/results.json`. Documented in
    maths-results-by-time.md + A12 block (confirmatory, not a new CE).
  - **8d mixed — in progress** (`mix_d8_l3_h4_t60K_s173289`, from-scratch). Report per
    model: role skeleton? step combiner endpoint-gated? carry-specific delivery
    (deciding-matched null 0.00)? polysemantic node reuse across add/sub?
- **HO-3 — Tier-1 breadth.** For the ~40 accurate models, confirm the role-skeleton /
  subtask-coverage summary is derived from actual per-model analysis (not asserted),
  and upload the per-model maps (behaviors.json / maths.json / train.json) to HF so
  the family claim is reproducible — including maps for the 8d worked-example models.

## Ranking logic

- Unblock the honest "in all models" reporting first (HO-2/HO-3), then prose.
- Prefer changes that reduce reviewer doubt that the claimed algorithm was actually
  found (the prior rejection reason): calibrated SV confirmation + cross-model breadth.

## Ranked queue

| Priority | Status | Piece | Evidence source | Done when |
| --- | --- | --- | --- | --- |
| 1 | in progress | Abstract reframe (confidence / worked-example / enrichment / family) | claim-evidence CE13–24, CE20–25 | draft approved by human |
| 2 | blocked on HO-2/HO-3 | Rework Experimental Results (Tier-1 general → Tier-2 4-model deep dive; name techniques, detail to appendix) | maths-results-summary + claim-evidence | body reads at right level, "in all models" honest |
| 3 | pending | Appendix de-dump + techniques table (technique → tests → establishes → models/scale → confidence) | study-maths/* | table added, dead sections cut |
| 4 | pending | Cut LLM survey (Fig 2 / ModelScores + App: Surveying LLM Addition Capability) + related-work trim | n/a | removed, intro re-hooked |
| 5 | pending | Subtraction-only models → appendix (keep subtraction **algorithm** in main body — novel) | n/a | main body slimmed |
| 6 | pending | Strip multi-venue cruft (ICLR/arXiv toggles, `\textcolor{blue}` notes, commented Hyp1/2, TriAdd) | n/a | paper.tex clean, compiles under acl.sty |

## Deliberately paused

- Non-archival 2-page track — not chosen (archival 8pg confirmed).
- Clean 99.999% **10d mixed** model (would need a training run) — deferred; 8d mixed
  used instead. Revisit only if a larger mixed worked example is wanted later.
