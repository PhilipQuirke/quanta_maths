# Study: d8-mixed cross-size — full subtraction story on the from-scratch worked example + ins1 map-anchored pieces — study-mixed-d8-crosssize.md

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).
Agenda: paper [HO-2](../paper-next-steps.md) (handed to this thread) + the CE29/30/31
d8-robustness follow-ups.

## Executive Summary (CE32)

Runs the whole mixed-subtraction account on the **accurate from-scratch 8-digit
worked example `mix_d8_l3_h4_t60K_s173289`** (the paper's Tier-2 8d-mixed model;
ADD/SUB/NEG = 1.0000/class), and completes the two d8 maps + the CE26 map-blocked
`ins1` pieces:

- **HO-2 causal SV battery replicates d6→d8** (all classes): writer tri-state
  **1.00** (VALID here, unlike `ins1` d8), resolved-cascade binary 0.97-1.00
  (untrained 0.61), combiner causal at digits 1-7, **STEP** endpoint-gated
  (α*≈0.5-0.75), `=`-not-source (flip 0/control 1), canonical transfer 1.00,
  **shared combiner** (STC/MTC/NTC co-located). Delivery route is **model-specific**:
  here ADD rides the residual at d2 then switches to attention at d3/d4, SUB/NEG use
  both routes at all depths (nulls 0.00) — a *different* pattern from `ins1` d8 (CE26).
- **CE29/CE30/CE31 replicate at d8**: keep-useful is **sufficient for ADD (0.92)** but
  **not for subtraction** (SUB 0.19 / NEG 0.66; keep-random 0.00); the missing nodes
  are the **last-layer attention heads at the producing positions of the failing
  digits** (A5→P21, A1→P25), tagged-elsewhere, a compact top-k recovering (SUB
  +top40→1.00) beating random-augment; and those heads are **borrow-in DELIVERY
  heads** (OV-decode `SV`=1.00 » base-diff `SA`; group causal flip 1.00/null 0.00;
  untrained fails). Cross-size nuance: on d8 the delivery is **concentrated in one
  head (H2, reading `=`/`SGN`)** rather than distributed across H0-H2 as on d6.
- **Map data-integrity fixes (both d8 maps, uploaded)**: `ins1_mix_d8` features.json
  had **no** combiner tags (a CE26 upload gap) → now STC6/MTC6/NTC6; from-scratch d8
  was **missing MTC** → now MTC6. Both via the standard `maths_hf_update` pipeline
  (library `_combiner_is_causal` fallback).
- **CE26 map-blocked `ins1` pieces, now unblocked**: writer-encoding is **VALID at
  the MAPPED writer locus** (ADD/ST 1.00@P14L0, SUB/MT 1.00@P12L0 vs untrained
  ~0.5) — CE26's invalidity was a wrong-locus (Dpn) artifact; writer-necessity shows
  the **ST writers are ADD-specific** (ablate→ADD 0.89 but SUB stays 1.00);
  shared-combiner = STC/MTC/NTC co-located at {P20-P25 L2 M0}.

## Method / artifacts

- HO-2: `scripts/mixed_d8_sv.py mix_d8_l3_h4_t60K_s173289` (parametrized d8 runner;
  batteries A/C/B/STEP/=-source/canonical + `combiner_delivery_sweep`; untrained
  control). Data `results/study-mix_d8_l3_h4_t60K_s173289/`.
- CE29: `scripts/circuit_sufficiency.py mix_d8_l3_h4_t60K_s173289`
  (`results/study-circuit-sufficiency/`).
- CE30: `scripts/find_missing_sub_nodes.py mix_d8_l3_h4_t60K_s173289`
  (`results/study-missing-sub-nodes/`).
- CE31: `scripts/missing_sub_mechanism.py mix_d8_l3_h4_t60K_s173289 5,1`
  (`results/study-missing-sub-mechanism/`).
- Map fixes: `quanta_maths.maths_hf_update.update_model(name, dry_run=False)` for
  both d8 models (features.json only; round-trip verified; uploaded).
- ins1 pieces: `scripts/mixed_d8_writer.py`
  (`results/study-ins1_mix_d8_l3_h4_t70K_s572091-writer/`).

## Results

### HO-2 causal battery — `mix_d8_l3_h4_t60K_s173289` (accurate 1.0/class)
| class | A tri-acc | C carry-acc | B causal digits | STEP | =-src flip | canonical |
|---|---|---|---|---|---|---|
| ADD | 1.00 | 0.97-1.00 | 1-7 | 9→0, α*0.75, gated | 0 (ctrl 1) | 1.00 |
| SUB | 1.00 | 0.99-1.00 | 1-6 | 0→9, α*0.5, gated | 0 (ctrl 1) | 1.00 |
| NEG | 1.00 | 0.99-1.00 | 1-7 | 0→9, α*0.75, gated | 0 (ctrl 1) | 1.00 |

Untrained control: tri-acc 0.59 / carry-acc 0.61 (fails). DELIVERY (flip/null):
ADD d2 res 1.00/attn 1.00, d3/d4 res 0.00/attn 1.00; SUB & NEG d2/d3/d4 res 1.00 &
attn 1.00; all nulls 0.00. Node tags emitted: STC6/MTC6/NTC6/DELIVERY21.

### CE29 sufficiency — `mix_d8_l3_h4_t60K_s173289`
keep 123/336 heads, 45/84 MLPs. ADD mean **0.923** (resample 0.230); SUB mean
**0.190** (0.033); NEG mean **0.663** (0.127); **keep-random 0.000** all. (ADD
sufficient; subtraction incomplete — SUB worse than d6, NEG milder.)

### CE30 missing nodes — `mix_d8_l3_h4_t60K_s173289`
SUB keep-map 0.190 → restore layer2 0.897 / answer_region 0.987 / heads 0.997 /
tagged_elsewhere_heads 0.997 (mlps/question/untagged: no); +top5 0.843, +top40
**1.00** vs random-augment 0.19. Per-digit worst **A5 0.37, A1 0.69**; +top40 all
1.00. Top nodes: P21/P24/P25 L2 heads (+P20 L1, P0 L0). NEG keep-map 0.663; +top20
0.997 vs random 0.68; worst A1 0.86, A5 0.90.

### CE31 borrow-delivery mechanism — `mix_d8_l3_h4_t60K_s173289` (fail digits A5,A1)
WRITE group `SV`=**1.00** all cells (» `SA` 0.58-0.67); per-head SV≈1.00, SA
0.13-0.39. CARRY group flip **1.00**/null 0.00; **per-head H2=1.00** (H0/H1/H3 0.00)
— delivery concentrated in H2, which READS `=`/`SGN` (0.50-0.99), own-operands ≈0.
Untrained: WRITE SV 0.61, CARRY flip 0.00 (fails). Map: H2 tagged OPR/SGN at those
positions but the borrow-delivery role is unrecognized (same "under-tagged role" as d6).

### `ins1_mix_d8_l3_h4_t70K_s572091` map-anchored pieces
- **shared-combiner**: STC=MTC=NTC = {P20,P21,P22,P23,P24,P25}L2M0 (one shared engine).
- **writer-encoding @ mapped locus**: ADD(ST) 1.00@P14L0 / untrained 0.52; SUB(MT)
  1.00@P12L0 / untrained 0.59; NEG(NT) n/a (no NT writer tagged — map gap).
- **writer-necessity** (mean-ablate mapped writers): ADD base 1.00 → ablate-ST 0.888
  / ablate-MT 0.892; SUB 1.00 → ablate-ST **1.000** / ablate-MT 0.928; NEG 0.996 →
  ablate-ST 0.984 / ablate-MT 0.940. → **ST writers ADD-specific** (SUB unaffected);
  MT writers broader (redundant, partial drops).

## Interpretation / scope

The mixed subtraction mechanism (SV representation → shared STEP combiner, `=`-not-
source, canonical code) is **size-general** (now d6 + two independent d8 models),
and the CE29/30/31 "map misses the redundant last-layer borrow-DELIVERY heads" story
is **size-general** too. The **delivery route is the model-specific dimension** (d6
both routes; `ins1` d8 SUB/NEG switch to attention at depth 4; from-scratch d8 ADD
switches to attention while SUB/NEG keep both) — within A10/CE25's "route may differ
by model". Cross-size, the borrow delivery **concentrates into fewer heads** as size
grows (d6 distributed H0-H2 reading lower operands; d8 one head H2 reading the
`=`/`SGN` staging). **A12** (size-general SV) + **A6** (redundancy hides the delivery
from per-node ablation) corroborated.

## Caveats
Single seed per model; CARRY uses one deterministic cascade stimulus per cell; the
d8 CE30 SUB deficit is severe (0.19) while NEG is mild (0.66) — subtraction map
incompleteness varies by model/class; `ins1` d8 is not 99.999%-clean (the from-scratch
d8 is the clean worked example); NEG writer-encoding unscored on `ins1` (no NT writer
tag). Delivery `Probe` tag is per-model coarse (records the shallow route).
