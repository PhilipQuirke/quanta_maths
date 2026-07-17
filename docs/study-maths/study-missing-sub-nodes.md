# Study: Identify the subtraction-important nodes the map misses — study-missing-sub-nodes.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).
Agenda: [maths-next-steps.md](../maths-next-steps.md) entry 1 follow-up (a).
Follows **CE29** (skeptic-passed HOLD-WITH-CAVEATS): on `ins1_mix_d6_l3_h4_t40K_s372001`
keeping only the map-useful nodes retains only **~0.5 (SUB) / ~0.37 (NEG)** under
mean-ablation, with the **sign/selection retained at 1.00** — the gap is the
**subtraction digit/borrow engine** (digits 10^5, hundreds fail). This study
**identifies which complement (map-omitted) nodes carry that missing computation**.
Understanding *what they do* is a later study.

## Executive Summary (CE30)

The map-omitted subtraction-important nodes are the **last-layer (L2) attention
heads at the answer-producing positions** — dominated by **P15L2H{0-3}** and
**P18L2H{0-3}**, i.e. the produce-positions of the two **failing digits A5 (10^5)
and A2 (hundreds)** from CE29. Restoring only the complement's layer-2 / answer-region
/ heads recovers SUB 0.48→~1.00 and NEG 0.36→~0.99; **a compact ~5-10-node set
suffices** (SUB +top20→1.00, NEG +top10→1.00) and **beats a same-size random-augment**
(≈0.5-0.6 SUB / ≈0.36 NEG) — the nodes are specific, not "more nodes help".
Restoration fixes exactly the failing digits (A5, A2 → 1.00) with the sign token
already at 1.00. **Disambiguation**: these are heads **tagged useful *elsewhere* but
under-tagged at these answer positions** (restoring tagged-elsewhere heads recovers
1.00; restoring entirely-untagged heads does not) — so the map's gap is **under-tagged
POSITIONS of the last-layer SV borrow-delivery heads (CE20/CE25 route)**, not a missing
subsystem. Caveat: under resample, top40 restores **SUB to 0.83** but **NEG only 0.46**
(NEG recovery is mean-ablation-dependent / more distributed). Single seed; d8 not run.

## Pre-run

- **Question**: which nodes **outside** the verified map are load-bearing for the
  mixed model's SUBTRACTION digit computation (the ones whose loss drives CE29's
  SUB 0.5 / NEG 0.37)?
- **Design** (mixed d6; classes SUB, NEG; keep-set = published `behavior.json`
  useful nodes; complement = all other `(pos,layer,head)`/`(pos,layer,MLP)`;
  destroy = mean-ablation from same-class inputs unless noted). **Restoration**
  (add complement nodes back to the keep-set and measure recovery) is the
  group-aware complement of ablation:
  - **Phase 1 — localize (group restoration).** For each structural group G
    (per layer; heads-vs-MLPs; question-tail vs answer-position regions; and
    "heads tagged useful *somewhere* restored at their untagged positions" vs
    "entirely-untagged heads/MLPs"), measure SUB/NEG accuracy with
    keep = map ∪ (complement ∩ G). Localizes where the missing mass is + the
    tagged-elsewhere-vs-untagged disambiguation the skeptic required.
  - **Phase 2 — rank + minimal set.** Single-node restoration gain
    Δ(c)=acc(map∪{c})−acc(map); rank the complement; **cumulative restoration** of
    the top-k (k=5,10,20,40,80,160) with recovery vs k; report the minimal k that
    recovers SUB to ≥0.9. Because per-node gains may be ~0 under redundancy, the
    cumulative curve (not single-node Δ) is the identification.
  - **Phase 3 — characterize + controls.** (a) **Specificity**: each restored
    top-k must beat a **same-size RANDOM-augment** of the map (random-augment must
    recover much less). (b) **Per-digit**: report retention at the failing digits
    (10^5, hundreds) to confirm restoration targets the digit engine, not the
    (already-1.00) sign. (c) **Method bracket**: score the final set under both
    mean and resample. (d) **Power**: ≥3 seeds; d8 mixed if time.
- **Positive controls**: keep-map baseline (~0.5 SUB) and keep-all (~1.0) bracket
  the restoration; sign-token retention 1.00 (unchanged) as a sanity anchor.
- **Success**: a compact, localized, map-omitted node set restores SUB/NEG to
  ≈ baseline, **beating random-augment of the same size** (specificity), with the
  recovery concentrated at the failing digits. Output = the ranked/localized
  missing-node list (hand-off to the later "what do they do" study).
- **Failure/ambiguous**: no compact set recovers (missing mass is truly diffuse
  across most of the complement — still informative); or random-augment recovers
  as well (then "more nodes help" generically, not specific nodes).
- **Skeptic (pre-launch)**: group-aware by construction (Phase-1 groups + Phase-2
  cumulative, not per-node ablation); random-augment specificity control;
  per-digit stratification; mean+resample bracket; multi-seed. Caveat: restoration
  under mean can inflate via modal-digit defaults — the random-augment control and
  per-digit check guard this.
- **Artifacts**: `scripts/find_missing_sub_nodes.py`,
  `results/study-missing-sub-nodes/results.json`.

## Post-run

Run 2026-07-17, `scripts/find_missing_sub_nodes.py`, mixed d6 (keep 98 / complement
232), classes SUB + NEG, mean-ablation, `make_batch` seed 0. Data:
`results/study-missing-sub-nodes/results_ins1_mix_d6_l3_h4_t40K_s372001.json`.

**Baselines**: keep-all 1.00 both; keep-map SUB 0.483 / NEG 0.357.

**Phase 1 — group restoration** (keep = map ∪ complement∩G):

| group | n | SUB restore | NEG restore |
|---|---|---|---|
| layer0 | 56 | 0.507 | 0.353 |
| layer1 | 78 | 0.503 | 0.370 |
| **layer2** | 98 | **0.997** | **0.993** |
| **heads** | 192 | **1.000** | **1.000** |
| mlps | 40 | 0.487 | 0.363 |
| question_region | 184 | 0.523 | 0.353 |
| **answer_region** | 48 | **1.000** | **1.000** |
| **tagged_elsewhere_heads** | 170 | **1.000** | **0.993** |
| untagged_heads | 22 | 0.657 | 0.363 |

→ Missing mass = **last-layer (L2) attention at answer positions**; it lives in
**heads the map already knows elsewhere but under-tags at these positions**
(tagged-elsewhere recovers; entirely-untagged does not; MLPs / earlier layers /
question-tail do not).

**Phase 2 — cumulative top-k vs same-size random-augment** (recovery):

| k | SUB top-k | SUB rand-aug | NEG top-k | NEG rand-aug |
|---|---|---|---|---|
| 5 | 0.940 | 0.493 | 0.990 | 0.360 |
| 10 | 0.963 | 0.627 | 1.000 | 0.363 |
| 20 | 1.000 | 0.583 | 1.000 | 0.370 |
| 40 | 1.000 | 0.600 | 1.000 | 0.360 |
| 80 | 1.000 | 0.870 | 0.987 | 0.987 |
| 160 | 1.000 | 0.980 | 0.993 | 1.000 |

→ **~5-20 nodes recover fully**, random-augment lags badly until k≥80 (**specificity**).

**Top-ranked missing nodes** (both classes dominated by L2 answer-position heads):
- SUB: `P15L2H1 P15L2H0 P18L2H1 P18L2H0 P15L2H3 P15L2H2 P18L2H2 P18L2H3 P17L2H3
  P17L2H1 P16L2H3 P16L2H1 P13L1M0 P6L0H3 P6L0M0 P19L1H2 P13L0H2 P11L1M0 P10L1M0 P10L0H2`
- NEG: `P15L2H0 P15L2H1 P18L2H1 P18L2H0 P15L2H2 P18L2H2 P15L2H3 P15L1H0 P13L1M0
  P13L0H2 P6L0H3 P5L0H1 P19L2H3 P19L1H3 P17L2H3 P17L2H1 P13L1H1 P9L0H3 P7L0H2 P5L0H3`
- **P15 = produce-pos(A5), P18 = produce-pos(A2)** — the CE29 failing digits.

**Phase 3 — per-digit** (keep-map → +top40):
- SUB keep-map `{SGN 1.0, A6 1.0, A5 0.71, A4 0.97, A3 0.98, A2 0.73, A1 0.99, A0 1.0}`
  → +top40 all **1.00**.
- NEG keep-map `{SGN 1.0, A6 1.0, A5 0.53, A4 1.0, A3 0.997, A2 0.66, A1 0.99, A0 1.0}`
  → +top40 all **1.00**.
- **Method bracket (top40 under resample)**: SUB **0.827** (robust), NEG **0.457**
  (mean-inflated / more distributed).

**Verdict**: success — a compact, localized, map-omitted set restores SUB/NEG,
beats random-augment, targets exactly the failing digits. Locus = under-tagged
positions of the last-layer SV borrow-delivery heads (CE20/CE25). Landed as **CE30**.

**Open (hand-off to next study)**: what do these heads *compute* (the borrow/digit
algorithm)? multi-seed (≥3) + d8 to firm the ranking and the weak NEG-resample.
