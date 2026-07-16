# Study: SV Compounding Rule at the Map-Named Wires (study-sv-compounding.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #15

When you add numbers with a long run of carries, the model has to get each carry to
the right answer digit. The paper's map says specific "layer-1" attention heads at
each answer position read from the `=` sign and from the digit sites, then hand the
carry to the little network that writes the digit. This study tests, causally,
**whether those heads actually deliver the deciding carry — and how** (do they pick
out the one digit that decides the carry, or read everything and let the network
sort it out?). It's the core test of the agent's main mechanism guess (A10).

**Outcome (partial confirmation):** yes — those heads *do* carry-specifically
deliver the deciding carry to the writing network (cutting the head's wire and
swapping in a different carry flips the answer, while swapping in a *same*-carry
signal does not), and it holds across two chain depths in both models. So A10's core
"attention fetches the carry and hands it to the combiner" is **partially confirmed**.
BUT three things stop it being a clean win: the delivery is **redundant** (two heads
share the job, so removing either one alone does no harm — sufficient but not
necessary); we could **not** show it's a single head *selecting* the one deciding
digit (the head that best tracks the deciding digit isn't the same one whose wire
carries the effect); and we could **not rule out** that some of the carry also
arrives by a different route (that test was underpowered). It took **three review
rounds** to land here — the first draft over-claimed "selection confirmed", the
second over-corrected to "not confirmed" on a broken control, and the third (this
one) is the calibrated middle. Net: A10's mechanism is real but redundant and only
partly pinned down; held at "medium" confidence, not raised.

Tests
[A10](../maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster),
[A9](../maths-conjectures-agent.md), [A6](../maths-conjectures-agent.md),
[A5](../maths-conjectures-agent.md). Builds on the C5-step-1 finding
([CE13](../maths-claim-evidence.md)) that the map-named `ST` nodes write a local
class (+ single-step U-resolution) — those are the *inputs* the L1 heads fetch.

## Pre-run (write before the experiment)

- **Question**: For the map-named answer-position **L1 heads** that feed the
  high-`Fail%` answer-position **L1 MLP** combiners: (V, C5 step 2) what do they
  read — do their value inputs at the attended `ST` sites and at `=` carry the
  tri-state/carry information (content), or is the large `=` mass a sink? (D, C5
  step 3) does patching the *deciding* digit's `ST` information reach the combiner
  **through the L1-head→MLP edge**, at **≥ 2 chain depths**, using a less-damped /
  multi-position edge instrument (the CE10-mandated upgrade)? (S) which A10 variant
  delivers the carry — direct residual path (a), selection-within-cluster (b, A9),
  or static weighted read (c)? (E, A6) is the delivery **selective economy** —
  ablating an L1 consumer head harms cascade questions but spares carry-free ones?

- **Motivation**: This is the crux of A10 and the C5 program's steps 2–4. CE5/CE7
  put the U-resolution at the answer-position L1-attention→MLP; CE10 got a one-depth
  causal crumb on an `L1.H1`→MLP edge but was underpowered (single-head edges
  LN-damped); CE13 confirmed the ST nodes write a *local* class (so compounding is
  downstream, at these L1 wires). The `behaviors.json` maps show the answer-position
  L1 heads attend exactly {`=`, the ST question-tail cluster} and feed the
  highest-`Fail%` L1 MLPs — A10's wiring is literally in the map. This study makes
  the *delivery* causal and multi-depth, and adjudicates *how* (selection vs
  direct-path vs static).

- **Ground-truth facts the design uses** (self-sufficiency):
  - **The consumer L1 heads** (from HF `behaviors.json`, read 2026-07-16; answer-
    position L1 heads with `Impact:A_k`, `Fail%`, and `Attn` to {`=`, ST-cluster}):
    - 6-digit `add_d6_l2_h3_t20K_s173289`: `P15L1H1`(→A5, attn P13=`=` 26, ST-cluster
      P10/P12), `P16L1H1`(→A4), `P17L1H1`(→A3), `P19L1H1`(→A1); also H2 variants
      `P15L1H2`,`P16L1H2`. Sign/leading: `P14L1H0/H1`(→A6). `=` is P13.
    - 5-digit `add_d5_l2_h3_t15K_s372001`: `P13L1H2`(→A4), `P14L1H2`(→A3), and the
      sign-position `P12L1H0/H2`(→A5). `=` is P11.
    - "SP-tagged" in the agenda = these ST-PCA-clustering answer-position L1 heads;
      **no literal `SP` tag exists in the JSON** — the property is derived (a
      pre-flight confirms each candidate's output clusters by ST class).
  - **The combiners** (high-`Fail%` answer-position L1 MLPs, CE5): 5-digit
    `P13–P17 L1MLP` (Fail 35–58%); 6-digit `P15–P20 L1MLP` (Fail 24–59%).
  - **The ST inputs** (CE13, confirmed local-class writes): 5-digit `P9/P10/P11/P12`
    L0 ST heads; 6-digit `P10/P11/P12/P14` L0 ST heads.
  - **Chain stimulus** `C(n,k,class)` (deep `...999`) + matched pairs: reuse
    `build_chain`/`consuming_pos`/`affected_digits` from the deep-cascade study.
  - **Edge decomposition** (CE10): `resid_mid(L1,pos) = resid_post(L0,pos) +
    Σ_h z[h]@W_O[h] + b_O`; the combiner reads `ln2(resid_mid)`. The CE10 skeptic
    mandated a **less-damped / multi-position** edge instrument (single-head edges
    are LN-renormalized to near-invisibility).

- **Hypothesis / competing reads** (neutral):
  - **R-A10-selection (b, A9-sharpened)**: the consumer L1 head's value input carries
    tri-state/carry content from the ST sites; patching the *deciding* digit's ST
    info reaches the combiner through the head→MLP edge at ≥ 2 depths; the head's
    attention relocates within the ST-cluster to the deciding digit; ablation harms
    cascade, spares carry-free.
  - **R-A10-static (c)**: the head reads the whole ST-cluster with a static weighted
    pattern and the MLP arbitrates — content present, edge causal, but **no**
    deciding-digit selection (attention doesn't relocate).
  - **R-direct-path (a)**: the deep carry reaches the combiner via the **direct
    `resid_post(L0)` path**, not the L1 heads (the L1 "fetch" is a red herring) —
    edge-patching the head does little; direct-path patch does the work.
  - **R-sink**: the `=` mass carries no content (a positional sink); the ST-cluster
    attention is incidental — the L1 heads' value path has no tri-state content.
  - **R-none / distributed / underpowered**: no single instrument localizes the
    delivery even after escalation.

- **Design**:
  - **Models**: 6-digit `add_d6_l2_h3_t20K_s173289` (primary — cleaner in CE8/CE10),
    5-digit `add_d5_l2_h3_t15K_s372001` (replication). CPU; acc-verified (< 0.99
    invalid).
  - **Pre-flight (SP-property confirmation)**: for each candidate consumer L1 head,
    confirm its output clusters by the served digit's ST class (the SP property);
    keep the heads that do. Report the list.
  - **Battery V — value content (C5 step 2)**: decompose the consumer head's value
    input by attended source. (i) At the attended **ST sites**, is the carry/ST info
    linearly present in what the head reads (probe the head's per-source value
    `v = x_source @ W_V` restricted to the ST-site keys)? (ii) At **`=`**, content
    vs sink: does the `=` value contribution carry decodable carry info, or is it
    constant across carry classes (sink)? Report per-source content.
  - **Battery D — deciding-digit propagation through the edge (C5 step 3;
    CE10 upgrade)**: in a deep `...999` chain, patch the **deciding** digit's ST
    information and measure whether it reaches the combiner **through the consumer
    L1-head→MLP edge**, at **≥ 2 depths**. Use the **less-damped instrument**: patch
    the head's OV contribution into `resid_mid` **and** (the multi-position variant)
    patch the head's `hook_z` across the *set* of consuming positions jointly, and
    route through the MLP-only arm (freeze the direct skip) — the CE10-mandated
    fixes. Read the affected-digit flip at each consuming position. Compare to the
    **direct-path** edge (patch `resid_post(L0)` into the combiner, freeze heads).
  - **Battery S — variant split (a/b/c)**: (a) direct-path flip vs (b/c) head-edge
    flip decides direct-vs-attention; (b) vs (c) by **deciding-digit target
    tracking** — does the consumer head's top-k attention key set relocate to the
    deciding digit as chain depth varies (value-matched, CE8/CE11 protocol)?
    Selection (b) = tracking + edge-causal; static (c) = edge-causal, no tracking.
  - **Battery E — selective economy (A6)**: mean-ablate each consumer L1 head; does
    accuracy drop on **cascade** questions (`...999` chains) more than on
    **carry-free** questions? vs an untagged-head baseline (the CE13 N-8 lesson).
  - **Controls / baselines**: SP-property pre-flight (positive control the read site
    holds ST); a known-causal readout control (patching the combiner-position
    residual flips the digit, CE10 control 2); the untagged-head ablation baseline
    (Battery E); the value-matched null for tracking (Battery S); per-cell nulls;
    the **≥ 2-depth** and **same-cell** requirements (CE10/CE11 lessons — tracking
    and edge-causality must be on the *same* head to score R-A10-selection).
  - **Minimal effects / bars**: content decodable ≥ chance + 0.2 above a
    shuffled-source null; edge flip ≥ 0.5 at ≥ 2 depths with the less-damped
    instrument, beating the direct-path arm to score attention-delivery; tracking
    gap ≥ 0.40 (value-matched) for selection; economy = cascade-impact − carry-free-
    impact ≥ 2× the untagged baseline. Both models or explicitly model-scoped.
  - **Pre-registered decision table**:
    | Head-edge flip (≥2 depths) | Direct-path flip | Tracking | Economy | Read |
    | --- | --- | --- | --- | --- |
    | yes (same cell) | — | yes (same cell) | cascade≫free | **R-A10-selection (A9)** |
    | yes | — | no | — | **R-A10-static** |
    | no | yes | — | — | **R-direct-path** |
    | no | no (after escalation) | — | — | **R-none/distributed** |
    | content absent at ST sites / `=` only | — | — | — | **R-sink** |
  - **Scope**: 2-layer addition, map-named L1 wires, edge-patch + tracking +
    ablation; no full neuron decomposition (B2 partial where cheap); no
    subtraction/mixed.

- **Positive controls** (any failure → `invalid`, never "negative"): (1) SP-property
  pre-flight — each kept consumer head's output decodes the served ST class ≫ chance;
  (2) readout control — patching the combiner-position residual flips the digit ~1.0;
  (3) less-damped-instrument validity — the multi-position/OV edge patch on a
  *known-causal* head moves the answer (closes CE10's underpower gap); (4)
  economy baseline — an untagged head's cascade-vs-free impact gap ≈ 0.

- **Success condition** (defined now): controls pass; the decision table selects one
  row (or model-scoped), with the deciding-digit edge causality shown at ≥ 2 depths
  and the selection-vs-static question resolved by same-cell tracking. Scored
  against A10 (and its variants), A9, A6, A5.

- **Failure condition** (defined now): with controls passing — no head-edge and no
  direct-path flip clears bar after escalation (**R-none/distributed**), or the
  value path carries no ST content (**R-sink**) — both substantive negatives for A10.

- **Ambiguous / invalid condition**: edge flip at one depth only (CE10 redux) →
  ambiguous; tracking and edge-causality on *different* heads (CE11 lesson) →
  R-A10-static not -selection; controls fail → invalid; models disagree → scoped.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + HF behaviors.json/features.json + the reused CE10/CE13 code, all
  verified). **Verdict: PASS WITH CONDITIONS** — four blocking findings, all
  confirmed against the map/code; the first reframes the study:
  - **C1 (blocking, linchpin) — redundancy can pre-ordain an uninterpretable
    null.** The consumer L1 heads are **Fail%:1** in the map (near-redundant; the
    load is on the L1 MLPs, Fail% 28–59%), and CE13 showed the ST inputs are
    redundant too. So a *single-head* edge flip may be structurally unreachable
    whether or not A10 is true — the study could null-out into R-direct-path/R-none
    for redundancy reasons. Fix: make the **joint (H1+H2 / multi-position) edge
    patch a first-class arm** (not escalation-only); add an explicit
    **R-A10-distributed-delivery** verdict row (joint-flips but no single head =
    A10 wiring supported, single-cell selection not); a single-head null is scored
    **underpowered/uninformative**, and R-direct-path only if the joint head-edge
    arm is sub-bar *while* the scaled direct-path control is supra-bar.
  - **C2 (blocking) — the less-damped/multi-position instrument is new code and
    under-specified.** Precisely define it (same head index patched at ≥ 2 consuming
    positions, per CE10 E-1's joint-flip idea), route through `ln2.hook_normalized`/
    MLP-only (inherit CE10 E-3's skip-clean property), report MLP-out-delta vs
    skip-delta, and **hard-gate Battery D on control 3** (less-damped instrument
    must move a known-causal head, else Battery D is `invalid` not `negative`).
  - **C3 (blocking) — Battery V needs the CE13 N-4 same-position wrong-role
    baseline** (a shuffled-source null doesn't control for "decodable everywhere at
    the position", CE11). Content = beats both the wrong-role co-located head AND
    the shuffled null by ≥ 0.2.
  - **C4 (blocking) — same-cell selection must be a machine-checkable identity**:
    `selection[pos] = tracking_gap[pos,head]≥0.40 AND edge_flip[pos,head,≥2 depths]
    ≥0.5` for the **identical** (pos,head); tracking-on-X + flip-on-Y = R-A10-static.
  - **C5 (non-blocking) — use `direct_scaled_percell`** (CE10 E-7) for the
    direct-path arm, not `direct_full`; R-direct-path requires a single-head-magnitude
    direct edge to be detectable.
  - **C6 (non-blocking) — `=`-sink test needs matched-pair stratification** (hold
    aggregate lower content, toggle only the deciding class); economy (Battery E)
    needs an **absolute floor** above the untagged baseline before the 2× ratio, or
    it is noise-dominated (consumer heads Fail%:1) → score untested/underpowered.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments SV-1…SV-6 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

**2026-07-16 — SV-1 (resolves C1): joint edge arm first-class + R-A10-distributed
row + null discipline.** Battery D's primary causal arm is the **joint** patch (the
consumer head *pair* H1+H2 at the consuming position, AND the same head index across
≥ 2 consuming positions), not single-head-only. New pre-registered verdict row
**R-A10-distributed-delivery**: joint head-edge flips ≥ bar at ≥ 2 depths but no
single head does → *A10's attention-fetch wiring supported, single-cell selection
not shown* (the redundancy analogue of CE13). A **single-head** edge null is scored
**underpowered/uninformative** (never "attention doesn't deliver"). **R-direct-path**
requires the joint head-edge arm sub-bar AND the scaled direct-path control supra-bar
(power demonstrably present).

**2026-07-16 — SV-2 (resolves C2): multi-position instrument specified + MLP-only +
hard gate.** The less-damped instrument = patch the consumer head's OV contribution
via `ln2.hook_normalized` (freeze `ln2` std to clean — CE10 lnfair) at the consuming
position, AND the **same head index** patched at ≥ 2 consuming positions jointly
(CE10 E-1 "joint = same head at ≥2 positions"). Route through the MLP input
(skip-clean, CE10 E-3); report MLP-out-delta vs skip-delta. **Control 3 is a hard
invalid-gate**: the instrument must move the answer on a known-causal head (largest
clean attention to the combiner's operands); if not, Battery D is `invalid`.

**2026-07-16 — SV-3 (resolves C3): Battery V same-position wrong-role baseline.**
Value content counts only if the consumer head's `x_source @ W_V` decodes the
ST/carry content ≥ 0.2 above BOTH (a) a co-located non-consumer head at the same
position (CE13 N-4) AND (b) the shuffled-source null. R-sink is scored from V only;
R-A10 requires Battery D regardless of V.

**2026-07-16 — SV-4 (resolves C4): same-cell machine check.** `results.json`
records `selection[pos,head] = (tracking_gap ≥ 0.40) AND (joint_or_single edge_flip
≥ 0.5 at ≥ 2 depths)` for the **identical** (pos,head). R-A10-selection requires
same-cell True; tracking and edge-causality on different heads → **R-A10-static**.

**2026-07-16 — SV-5 (resolves C5): scaled direct-path control.** The direct-path
arm uses `direct_scaled_percell` (CE10 E-7, `sc=min(1,head/direct)≤1`); R-direct-path
requires both `direct_full` and the scaled control to flip (else underpowered).

**2026-07-16 — SV-6 (resolves C6): `=`-sink stratified; economy floor.** The
`=`-content test uses matched pairs holding aggregate lower content and toggling
only the deciding class (reuse `build_chain`); reported as necessary-not-sufficient
content, never delivery. Battery E requires both cascade and carry-free ablation
impacts to clear the untagged-head baseline max by a stated margin before the 2×
ratio; if both are at the noise floor (expected, consumer heads Fail%:1) → economy
**untested/underpowered**, not "A6 refuted".

- **Decision impact**:
  - **R-A10-selection**: A10 confirmed in full (fetch-and-combine by selection);
    A9's selection lean vindicated at named nodes; A6 economy supported; C5 step 3–4
    done. The thread's central mechanism lands.
  - **R-A10-static**: A10's wiring confirmed but the *selection* half refuted — the
    head reads the whole cluster statically and the MLP arbitrates (closer to
    wide-fetch); A9 refuted, A6 economy still testable.
  - **R-direct-path**: A10's L1-fetch is a red herring; the carry rides the direct
    residual path — reshapes A10 toward a residual-carried account (partial A6).
  - **R-sink / R-none**: A10's value-content premise fails or nothing localizes —
    the compounding is elsewhere / distributed; escalate to neuron-level (B2).

- **Risks / confounds** (with mitigations):
  - **LN-damping underpower** (CE10): the mandated less-damped / multi-position edge
    instrument + a validity positive control (3); no single-head-only conclusion.
  - **Decodable ≠ delivered** (CE13/outside-view): content (V) is necessary not
    sufficient; the causal edge patch (D) is the load-bearing test; same-cell
    tracking+causality required for selection.
  - **Tracking tie-noise** (CE8): value-matched top-k target-set metric, not raw
    argmax; ≥ 2 non-degenerate depths.
  - **`=`-sink ambiguity**: test `=` content by whether its value contribution
    varies with carry class (content) or is constant (sink).
  - **ST/SA co-variance & matched pairs** (CE13 lesson): deep-chain matched pairs
    differ only at the deciding digit; stratify.
  - **Ablation baseline** (CE13 N-8): economy vs an untagged-head baseline, not raw.
  - **Redundancy** (CE13): single-node interchange may flip nothing though ablation
    hurts — use ablation + edge-patch, not interchange alone.

- **Expected artifacts**:
  - Standalone CPU script `scripts/sv_compounding.py` (reuses `build_chain`,
    `consuming_pos`, the CE10 edge-patch helpers, the tricase machinery, sklearn).
  - `results/study-sv-compounding/`: `results.json` (SP pre-flight; value-content
    per source; per-depth edge-flip + direct-path arm; tracking gaps; economy
    cascade-vs-free vs baseline; all controls); plots (edge-flip×depth, tracking,
    economy). No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (calibrated after TWO Gate-2 rounds corrected over-reach in
  BOTH directions): **A10's fetch-and-deliver wiring is PARTIALLY confirmed —
  carry-specific, causal, redundant — but its clean single-head SELECTION form is
  NOT shown. R-A10-distributed-delivery, both models.** Patching the map-named
  answer-position L1 heads' output edge into the combiner flips the top cascade
  digit at ≥ 2 depths (via the H1+H2 joint pair), and this is **carry-specific**
  (the deciding-matched null = 0.00 — it flips on the deciding-carry toggle, NOT on
  same-class filler changes) and **consumer-head-specific** (a non-consumer head is
  inert). BUT: (i) **single-cell selection is not shown** — the head that *tracks*
  the deciding digit (`L1.H2`, value-matched) is **not** the head with single-depth
  edge causality (`L1.H1`); no one head is both ≥2-depth-edge-causal AND tracking
  (the CE10/CE11 same-cell trap, correctly avoided as a claim here); (ii) the
  mechanism is **redundant** (edge-sufficiency on H1, necessity/economy on H2); (iii)
  the **direct path is not excluded** (its arm is underpowered, `direct_scaled`=0).
  *(Process: round 1 over-claimed "R-A10-selection at L1.H1" — a three-attribution
  smear; round 2 over-corrected to "not carry-specific / generic disruption" on a
  broken null that itself toggled the deciding carry; the deciding-matched null =
  0.00 gives this calibrated middle.)* With controls passing (instrument valid 1.00;
  gate ≥ 0.88):
  - **Battery D + deciding-matched null (calibrated)**: the edge patch flips the top
    digit (6-digit `single_H1` 1.00 at k=3, `joint_pair` 1.00 at k=2,3; 5-digit
    `single_H2` 0.95/1.00), and the **deciding-matched null = 0.00** (both same
    class, only non-deciding content re-drawn) → the flip **IS carry-specific** (it
    responds to the deciding-carry toggle, not to filler). The non-consumer head is
    inert (`specificity_H0`=0.00) → consumer-head-specific. `direct_full`/
    `direct_scaled` = 0.00 → the direct arm is **underpowered** (cannot exclude the
    direct path). *(The first-draft "same-class null flips equally → generic" was a
    broken null that toggled the deciding carry; corrected in SV-9.)*
  - **Battery S (value-matched tracking, corrected)**: only **`L1.H2` tracks in
    6-digit** (2 non-degenerate depths beating the units-end control); `L1.H1` (the
    single-depth edge-causal head) does **not** track; **no head tracks in 5-digit**.
    So tracking and single-head edge-causality are on **different heads** (6-digit) —
    selection is **not** same-cell.
  - **Battery E (economy, A6)**: ablation harms cascade / spares carry-free at
    **`L1.H2`** (gap 0.33–0.84 6-digit; 0.37–0.97 5-digit; untagged baseline 0.00),
    but **`L1.H1` ablation does nothing** (gap 0.0). So the *necessity* is on H2, the
    *single-depth edge sufficiency* on H1, and *tracking* on H2 — three attributions,
    not one selector.
  - **Battery V (value content)**: `SV_n` decodes ~0.93–1.00 from the heads' value
    projection but the co-located non-consumer head decodes it **equally**
    (wrong-role baseline ≈ equal) → content present at the position, **not
    head-specific** — A10's value-content sub-prediction **fails its head-specificity
    test**.

  Net (calibrated): **A10's fetch-and-deliver wiring is PARTIALLY confirmed —
  R-A10-distributed-delivery, both models.** Carry-specific head-edge delivery is
  causally sufficient at ≥ 2 depths through a redundant H1/H2 pair, and it is
  consumer-head-specific. But A10's clean *single-head selection* (variant b / A9)
  is **not shown** (tracking head ≠ single-depth-edge-causal head), the mechanism is
  **redundant** (H1 edge-sufficient, H2 necessary + tracking), the value-content
  sub-prediction is **not head-specific**, and the direct path is **not excluded**
  (underpowered arm). A real partial advance for A10's core wiring; not the full
  select-and-deliver mechanism. Scoped: 2-layer addition, map-named L1 wires.

- **Run record**:
  - Command: `PYTHONPATH=. python3 scripts/sv_compounding.py all`.
  - Script: [`scripts/sv_compounding.py`](../../scripts/sv_compounding.py)
    (standalone CPU; reuses `build_chain`/`consuming_pos`/`affected_digits`, the CE10
    edge-patch idea via `ln2.hook_normalized` MLP-only, sklearn). Env: python 3.13.7,
    macOS-26.5.2-arm64, torch 2.8.0. Repo commit `d285797` (working tree). Date
    2026-07-16. Seed 20260716. Consumer heads from HF `behaviors.json`.
  - Models: `add_d6_l2_h3_t20K_s173289`, `add_d5_l2_h3_t15K_s372001` (both acc 1.000).
  - Artifacts (local; no HF): `results/study-sv-compounding/results.json`.

- **Results**:
  - **Control 3** (less-damped instrument validity): consumer head edge patch moves
    the answer 1.00 (both models) — instrument valid, CE10 underpower gap closed.
  - **Battery D**: 6-digit `single_H1` k=3 1.00 (k=2 0.00), `joint_pair` k=2,3 1.00,
    `multipos_H1` k=3 1.00; **deciding-matched `NULL_*` = 0.00 all arms/depths**
    (carry-specific); `direct_full`/`direct_scaled` 0.00 (direct arm underpowered).
    5-digit `single_H2` k=2,3 0.88/1.00, `NULL_single_H2` 0.00; direct 0.00.
  - **Battery S** (value-matched): **only `L1.H2` tracks** in 6-digit (2
    non-degenerate depths beating the units-end control); `L1.H1` does not; **no head
    tracks in 5-digit**.
  - **Battery E**: 6-digit economy gap `P15L1H2` 0.83, `P16L1H2` 0.37, all H1 = 0.00;
    5-digit `P13L1H2` 0.91, `P14L1H2` 0.53; untagged baseline max 0.00 both models.
  - **Battery V**: SV content acc 0.93–1.00 but wrong-role baseline equal → content
    present at position, not head-specific (all flagged `content=False`).

- **Interpretation** (against pre-stated conditions; goalposts unchanged;
  calibrated post-2-rounds):
  - **R-A10-distributed-delivery (both models)**: the head-edge patch is
    **carry-specific** (deciding-matched null = 0.00) and causally *sufficient* to
    flip the top digit at ≥ 2 depths (joint H1/H2), and consumer-head-specific
    (non-consumer inert). A10's core "attention fetches the deep carry to the
    answer-position combiner" is **partially confirmed** — a genuine partial advance.
  - **Single-cell selection NOT shown**: tracking (H2, value-matched) and
    single-depth edge causality (H1) are on **different heads**; no same-cell
    ≥2-depth edge + tracking. A10's selection variant (b)/A9 is not demonstrated
    (the CE10/CE11 trap correctly not-claimed).
  - **Direct path NOT excluded**: `direct_scaled`=0 = underpowered arm; the study
    shows head-edge *sufficiency*, not direct-path inertness.
  - **Redundancy**: edge-sufficiency on H1, necessity/economy + tracking on H2 — a
    redundant pair; single-head ablation misses it (CE13 lesson).
  - **Both models agree** on the calibrated read (carry-specific, distributed,
    selection-unshown).

- **Prediction scoring** (records evidence; conjecture updates after Gate 2;
  calibrated):
  - **A10**: **core wiring PARTIALLY confirmed (distributed-delivery), selection
    NOT shown.** Carry-specific, causal, consumer-head-specific head-edge delivery
    at ≥ 2 depths (both models) supports A10's fetch-to-combiner premise; but the
    single-head selection variant (b) is unshown (tracking ≠ edge-causal head), the
    mechanism is redundant, and the direct path is not excluded. **Hold A10 at
    medium** — the distributed-delivery core moves from a one-depth CE10 crumb to
    carry-specific ≥2-depth (joint) sufficiency + consumer-specificity (a net
    positive *within* medium); the selection sub-claim is **not supported**.
  - **A9**: **not supported** — deciding-digit tracking is on `L1.H2`, not the
    single-depth-edge-causal head (`L1.H1`); no same-cell selection.
  - **A6**: **selective economy supported (at H2)** — ablating `L1.H2` harms cascade
    and spares carry-free (gap ≫ untagged baseline 0.00); real, but one head of a
    redundant pair.
  - **A5**: routing content-dependence is present AND the edge is carry-specific and
    causal — so CE8's hybrid routing gains **partial causal support** here (the
    distributed delivery is real), though not localized to a single tracking head.

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + script), independent analysis. **Verdict: PASS WITH
  CONDITIONS**, and the conditions — plus a specificity control the fixes added —
  **substantially deflated the first-draft "R-A10-selection" headline**. Findings:
  - **F1-post (blocking) — "selection at L1.H1" was a three-attribution smear**:
    single-head edge causality was **one depth** (H1 at k=3 only; the *pair* carried
    ≥2 depths), tracking was on *both* H1 and H2, and necessity/economy was on a
    *different* head (H2; ablating H1 does nothing). The CE10/CE11 "tracking and
    causation on different heads" trap. Rescored to **R-A10-distributed-delivery**
    (the pre-registered SV-1 row).
  - **F2-post (blocking) — "attention-delivered, not direct-path" inverted SV-5**:
    `direct_scaled = 0.00` means the direct arm is **underpowered** (SV-5's own rule),
    so the direct path is **not excluded**; and the direct arm runs through *live*
    LN while the head arm freezes it (power-mismatched). Claim reworded to head-edge
    **sufficiency**, not direct-path exclusion.
  - **F3-post (blocking) — tracking metric regressed** to a raw top-2-hit measure a
    prior round rejected. Fixed: re-ran with the value-matched metric + units-end
    control (`deep_cascade_batteries.deciding_target_tracking`). Result: only
    **`L1.H2` tracks in 6-digit** (H1 does not clear it); **no head tracks in
    5-digit**. So selection is *not* on the edge-causal head.
  - **F4-post (blocking, decisive) — Battery D had no specificity control**; the
    added **same-class null** (source a different *hi* question, deciding class held)
    **flips as much as the real deciding contrast** (`NULL_single_H1`=1.00=`single_H1`;
    `NULL_joint_pair`=1.00; 5-digit `NULL_single_H2`=1.00). So the edge patch is
    **generic breakage**, NOT carry-specific delivery — `flip=1.00` = "answer
    changed", not "the deciding carry was delivered". (The non-consumer-head
    `specificity_H0` is clean at 0.00, so it is *a* consumer head, but not
    carry-specific.)
  - **F5/F6/F7 (non-blocking)** — control-3 was circular (patched the test head);
    Battery V's A10 value-content sub-prediction **failed** its head-specificity
    test (under-scored); A10 had few losing branches.

  *Status: RESOLVED 2026-07-16 by the working thread over THREE Gate-2 rounds. Round
  1 caught a positive over-claim ("R-A10-selection at L1.H1" — a three-attribution
  smear). Round 2's deflation ("not carry-specific / A10 not confirmed") rested on a
  **broken null** (SV-8: it toggled the deciding carry, so null=real=1.0 was the
  EXPECTED signature of a carry-specific head, not evidence against). Round 3
  corrected the null to deciding-matched (SV-9: null = 0.00) → the effect **IS
  carry-specific**, landing the calibrated **R-A10-distributed-delivery** (partial
  positive). The three genuinely-sound round-2 corrections are retained
  (selection-not-same-cell; direct-path-not-excluded; value-not-head-specific); only
  the incorrect broken-null deflation was undone. Gate 2 PASSED on the calibrated
  read (see the calibrated Executive summary / scoring).*

**2026-07-16 — SV-7 (resolves F1/F2/F3-post): same-cell ≥2-depth requirement +
value-matched tracking + head-edge-sufficiency framing.** Same-cell selection
requires a single head edge-causal at **≥ 2 depths** AND tracking (value-matched
metric with units-end control) — met by **no** head in either model. "Attention
delivers" is scored as head-edge **sufficiency** (joint/single flip), NOT
direct-path exclusion (the direct arm is underpowered, `direct_scaled`=0).

**2026-07-16 — SV-8 (resolves F4-post): specificity null — FIRST VERSION BROKEN,
CORRECTED in SV-9.** The first same-class null was mis-constructed (source a
different *hi* question patched into the *lo* target — which itself toggles the
deciding carry), so NULL≈real≈1.00 and I wrongly read the edge as "generic
disruption". That was an over-correction on a broken control (Gate-2 round-2 BLOCK).

**2026-07-16 — SV-9 (resolves the SV-8 error; the calibrated fix): deciding-MATCHED
null.** The correct null holds the deciding class constant (source AND target both
`lo`), re-drawing only non-deciding content — the CE10/CE13 convention. Result:
**deciding-matched null = 0.00** in both models while the real deciding-toggle flips
1.00 → the head-edge effect **IS carry-specific** after all. This reinstates the
calibrated verdict **R-A10-distributed-delivery** (partial positive): carry-specific
head-edge delivery sufficient at ≥ 2 depths via the redundant H1/H2 pair,
consumer-head-specific (non-consumer H0 inert); **single-cell selection still not
shown** (tracking on H2 ≠ single-depth-edge-causal H1); **direct path not excluded**
(underpowered arm). Neither the round-1 "selection confirmed" over-claim nor the
round-2 "not confirmed / generic" over-correction — the honest middle.

- **Limitations**:
  - **Sufficiency, not necessity**: the carry-specific head-edge effect is
    *sufficient* (patching flips the digit) but the sufficient single head (H1) is
    not *necessary* (ablating it does nothing; H2 covers) — so "delivery" means
    joint-sufficiency, not single-head necessity. A paired/joint necessity proof is
    the follow-up.
  - **Selection underdetermined**: tracking + necessity are on H2, single-depth edge
    sufficiency on H1. This is consistent with BOTH "distributed/redundant, no single
    selector" AND "H2 selects, H1 redundantly delivers" — the study cannot resolve
    which (H2's own ≥2-depth single-edge causality was not established). A same-cell
    ≥2-depth edge test on H2 is the deciding follow-up.
  - The direct-path arm is underpowered (LN-mismatched, `direct_scaled`=0), so the
    direct path is **not excluded**.
  - Battery V is position-level, not head-attributable (value content not shown
    uniquely read by the consumer head).
  - 2-layer addition, two models; no neuron decomposition (B2).

- **Doc updates** (after Gate 2 passes; APPEND to the committed C5 docs): ledger;
  claim-evidence (new **CE14**: **carry-specific** head-edge delivery from the
  map-named answer-position L1 heads to the combiner is causally sufficient at ≥ 2
  depths [deciding-matched null = 0.00], consumer-head-specific, via a **redundant
  H1/H2 pair** — A10's fetch-to-combiner core partially confirmed; but **single-cell
  selection NOT shown** [tracking on H2 ≠ single-depth-edge-causal H1], the direct
  path is not excluded [underpowered], and value content is not head-specific);
  synthesis + summary; conjectures (**A10 → medium, distributed-delivery core
  partially confirmed, selection sub-claim not supported**; A9 not supported; A6
  economy supported at H2; A5 partial causal support for hybrid routing); agenda
  (complete C5 steps 2–4 entry 1; entry 1 becomes the leading-digit hard-case
  walkthrough, C5 step 5).

- **Next read**: C5 steps 2–4 are partially answered — carry-specific attention-edge
  delivery to the combiner is confirmed (A10 core), via a redundant H1/H2 pair, but
  the single-head *selection* variant is not shown and the direct path is not
  excluded. Productive follow-ups: (a) a **carry-specific path-patch** that varies
  only the resolved carry into the combiner (to strengthen delivery beyond
  sufficiency and adjudicate the direct path with a powered arm); (b) a **same-cell
  test on `L1.H2`** (the tracking+necessity head) — is it ≥2-depth edge-causal, i.e.
  the real selector, with H1 a redundant deliverer? The next agenda item is **C5
  step 5 (leading-digit hard-case walkthrough)**, which inherits these instruments
  and the redundant-pair picture.
