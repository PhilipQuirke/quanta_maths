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

### Front-matter status (2026-07-17) — human considers these FROZEN

Abstract, Introduction, Related Work, Methodology, and Training Models / Training
mixed models are drafted and **frozen** by the human. Also completed this session:
Experimental Results reworked to the Tier-1/Tier-2 structure; LLM-survey + Fig 2 +
survey appendix removed; dead code stripped (ICLR/arXiv toggles, `\textcolor{blue}`,
commented Hyp1/2, TriAdd); confirmation-bias reframe applied (inductive
observe→hypothesise→confirm narrative + refuted-hypotheses note with a TODO); model
count corrected (49→46, seed-sensitivity 48→45); addition-table d11–d14 seeds fixed
(173289→572091) to match HF; US spelling standardized throughout; `Analysis
Techniques` appendix stub + table added. Remaining work is in the ranked queue below.

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
| 8d addition (`add_d8_l2_h3_t45K_s173289`) | yes (0) | **full — done** (confirmatory: STC combiner, CE2 transport, =-depot, class-necessity, step combiner, intrinsic redundancy, L1-locus, lazy-TF; source-fork probe-limited) | **HF** (per-model repo) | none |
| 6d mixed (`ins1_mix_d6_l3_h4_t40K_s372001`) | yes (0/0) | CE20–25 | HF + local | none |
| 8d mixed (`mix_d8_l3_h4_t60K_s173289`, from-scratch) | yes (1.0000/class) | **full — done (CE32)** | **HF (deep + combiner-complete STC/MTC/NTC)** | none |

Notes:
- 8d mixed is a **from-scratch** model; the 6d mixed capstone (`ins1`) carries the
  insertion/polysemantic story. The 8d `ins1` initialized model is NOT clean
  (50/1196 fails), so mixed types cannot be matched at 8d — 8d mixed is the
  cross-size generalization point only.
- No clean 99.999% 10d mixed exists (from-scratch 295 sub fails; init 13 fails).
  Deferred; 8d chosen instead.
- HF coverage (per [hugging_models.md](hugging_models.md), verified 2026-07-16):
  per-model analysis repos (`behaviors.json` + `features.json`) exist for **all 53
  models**; the legacy flat repo carries 33 — so the Tier-1 "in all models" breadth is
  well-covered. Map *depth* still varies (5/6-digit fullest; large-n lack L1 consumer-head tags).
- Model count: the paper studies **46 models** (per Tabs.); HF hosts a superset (48
  flat weight-sets / 53 per-model repos, incl. d20, d15 and `gf` training-variant
  twins not in the paper). Paper prose corrected 49→46 and seed-sensitivity 48→45.
- **Map completeness (2026-07-17, `maths` thread verified)**: both 8d mixed maps were
  made combiner-complete on HF via `maths_hf_update` (dry-run → verified → upload,
  round-trip checked): `ins1_mix_d8_l3_h4_t70K_s572091` had **zero** combiner tags (a
  CE26 upload gap) → now **STC6/MTC6/NTC6**; `mix_d8_l3_h4_t60K_s173289` was missing
  MTC → now **STC4/MTC6/NTC5**. Both carry the full role skeleton + Probe
  `LINXFER`/`CARRY*` + CE25 `DELIVERY.{ADD,SUB,NEG}` route tags. The rest of the
  `mix_*` from-scratch zoo is similarly tagged (d5/d9 spot-checked).
- **Cross-size variation (paper asset — the "family" story)**: two clean, honest
  divergences across sizes, same skeleton otherwise. (a) Carry-propagation *deferral*
  grows **2→3 tokens** d6→d8 (addition, CARRY tag). (b) Borrow/carry *delivery*
  concentrates in a **single head** at d8 mixed (H2, reading `=`/SGN) vs **distributed**
  across heads at d6. Good "similarity and variation across the model family" material.

### Decisions still needed from human

1. **Spine confirmed** (2026-07-17): 6d addition (core) + 6d mixed (capstone), with 8d
   addition + 8d mixed as the cross-size generalization. No open spine decisions.
2. **Done (2026-07-17, human-directed)**: the Experimental-models appendix now maps the
   46 studied models (16 add / 7 sub / 23 mixed) to the HF superset (48 flat / 53
   per-model repos), and the README carries the matching 46-studied / 29-accurate
   summary with contract-correct artifact file names.

## Handoff tasks to the `maths` thread (this thread is read-only on experiments)

- **HO-2 — full SV battery on the two 8d worked-example models. DONE (2026-07-17).**
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
  - **8d mixed — DONE (2026-07-17, CE32)** (`mix_d8_l3_h4_t60K_s173289`, from-scratch,
    accurate 1.0000/class; [study-mixed-d8-crosssize.md](study-maths/study-mixed-d8-crosssize.md)).
    All four HO-2 questions answered: **role skeleton** — yes (writer tri-state 1.00,
    combiner causal digits 1-7, roles mapped); **step combiner endpoint-gated** — yes
    (α*≈0.5-0.75, endpoints gated, all classes); **carry-specific delivery** — yes
    (deciding-matched null 0.00; route ADD res@d2→attn@d3/4, SUB/NEG both routes all
    depths — model-specific); **polysemantic reuse** — yes (STC/MTC/NTC on the same
    L2 answer-MLP nodes = shared combiner). Also: resolved-cascade binary 0.97-1.00,
    `=`-not-source, canonical 1.00, untrained control fails. Plus the CE29/30/31
    story replicates at d8 (map incomplete for subtraction — SUB 0.19/NEG 0.66 keep-
    useful; missing = last-layer borrow-DELIVERY heads at the failing digits).
    Both d8 maps were also made **combiner-complete** on HF (ins1 had 0 combiner
    tags; from-scratch was missing MTC).
- **HO-3 — Tier-1 breadth. Largely DONE (2026-07-16).** Per-model analysis repos
  (`behaviors.json` + `features.json`) now exist for **all 53 HF models** (verified;
  see [hugging_models.md](hugging_models.md)), so the "in all models" family claim is
  reproducible. Residual: map *depth* varies (5/6-digit fullest; large-n lack L1
  consumer-head tags) and the paper-table numeric columns (fails/M, heads/MLPs-used)
  are not yet re-verified against the maps — see queue item on numeric-column audit.

## Ranking logic

- Front matter is frozen; remaining work is back matter (Conclusion/Limitations),
  the appendix, and submission mechanics (anonymization, build).
- Prefer changes that reduce reviewer doubt that the claimed algorithm was actually
  found (the prior rejection reason): calibrated SV confirmation + cross-model breadth.
- Shipped entries are deleted from this queue (per the agenda contract), not marked
  done in place; the session summary lives in Front-matter status above.

## Ranked queue

| Priority | Status | Piece | Evidence source | Done when |
| --- | --- | --- | --- | --- |
| 1 | pending | **Conclusion + Limitations & Future Work** — align to new framing; keep honest limits (single-head selection not shown, 2-layer, linear-probe); drop any stale/overclaiming lines | claim-evidence caveats | reads consistently with frozen front matter |
| 2 | in progress | **Appendix de-dump + flesh out `Analysis Techniques` appendix** (stub table exists) and add an **SV-detail appendix** (CE13–24/CE20–25 detail referenced from the reworked Results); cut remaining dumping-ground appendices | study-maths/*, claim-evidence | techniques + SV detail complete; dead appendices cut |
| 3 | pending | **Subtraction-only models → appendix**; verify main body keeps only the (novel) subtraction *algorithm* | n/a | main body slimmed, no stray sub-only prose |
| 4 | pending | **Restore brief "Refuted hypotheses" content** (intro carries a `% TODO`); name 1–2 ruled-out mechanisms (e.g. no single carry-selecting head; old compact-representation Hyp) | claim-evidence (A3/A9/A11); old Hyp1/2 | note/short appendix added, TODO cleared |
| 5 | pending | **Anonymization** — regenerate the two anon repo URLs for BlackboxNLP (2 `% TODO` comments in `paper.tex`); keep third-person self-cites | n/a | valid anonymized links for review build |
| 6 | blocked (maths) | **Numeric-column audit** of both model tables (fails/M, heads/MLPs-used) against HF maps (HO-2 8d-mixed causal write-up now DONE, CE32) | maths thread | table numbers verified |
| 7 | pending | **Build** — add `Figures/` PNGs/PDFs + `acl.sty`/`acl_natbib.bst`; compile under `acl.sty`; fix floats/overfull; confirm ≤ 8pp main text | assets/, ACL style repo | compiles to a within-limit PDF |

## Deliberately paused

- Non-archival 2-page track — not chosen (archival 8pg confirmed).
- Clean 99.999% **10d mixed** model (would need a training run) — deferred; 8d mixed
  used instead. Revisit only if a larger mixed worked example is wanted later.
