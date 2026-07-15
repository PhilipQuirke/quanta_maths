# Study: Confirm an ST Compute Node via Path-Patching (study-confirm-st-node.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #2

**Verdict: binary make-carry (`SC`) compute nodes confirmed — but NOT tri-state
`ST` nodes.** Causal interchange-intervention identified specific answer-position
layer-0 heads that compute the per-digit carry (flip the next-higher answer digit
`A_{n+1}` at 1.00, `A_n` at 0.00, both models: 5-digit `P13/P14/P15.L0.H0`;
6-digit `P14–P19.L0.H2`), with a clean **head-role dissociation** — at each
answer position one head does the carry and a *different* head does base-add
(`P14.L0.H1` = SA).

A genuine tri-state test (fix `Dn+D'n=9`, toggle the lower carry) flips nothing
through these heads, so they compute only the **binary** `Dn+D'n≥10` carry; the
tri-state `U`-resolution runs on a **separate path** (the model resolves `U`
correctly — clean predictions differ 40/40). Recorded as CE3. This confirmed a
causal carry locus (which the frozen pair-sum line lacked) and set up the hunt
for the separate U-resolution path.

## Pre-run (write before the experiment)

- **Question**: Which node(s) in an accurate addition model *causally compute*
  the [ST](../thor-glossary.md#st) tri-state carry for a given digit `n`, in the
  sense that setting the node's activation to its value under a counterfactual
  question with a *different* `ST_n` value causally changes the answer digits
  the carry cascade should control — and can we distinguish such an `ST` node
  from an `SA` (base-add) node by the *signature* of which answer digits move?

- **Motivation**: A confirmed `ST` compute node is the prerequisite anchor for
  the re-scoped entry 2 (`ST` tri-state geometry) and for any real test of A2
  (aggregate-then-discretize). The pair-sum study showed selection by attention
  + single-node ablation cannot find it. A causal interchange-intervention that
  keys on the *carry cascade signature* is the correct locator.

- **Hypothesis / competing reads** (neutral):
  - **The discriminating signature.** In the algorithm (Paper 2), `SA_n`
    (base-add) determines answer digit `A_n` only; `ST_n` (tri-state carry)
    feeds the `SV` cascade, which determines the carry *into the next-higher
    answer digit* `A_{n+1}` (and can cascade to `A_{n+2}...` through `U`). So:
    - **R-ST**: patching a node that computes `ST_n` (swapping `ST_n` between a
      carry and a no-carry source) flips `A_{n+1}` (and possibly higher digits
      via cascade) while leaving `A_n` unchanged.
    - **R-SA**: patching a node that carries `SA_n` (base-add) changes `A_n`
      itself, not `A_{n+1}`.
    - **R-none**: patching changes no answer digit (node not causal for this
      digit) — or changes digits inconsistent with either signature (mixed /
      polysemantic node).
  - Multiple nodes may share an `ST_n` role (Paper 2 redundancy); the design
    reports all nodes passing the R-ST criterion, not a unique node.

- **Design**:
  - **Models**: primary `add_d5_l2_h3_t15K_s372001` (5-digit, accurate;
    Paper-2 ST candidates P8.L0.H1, P9.L0.H1, P11.L0.H2, P14.L0.H1),
    replication `add_d6_l2_h3_t20K_s173289` (6-digit, independent seed). Loaded
    via `MathsConfig` + TransformerLens `HookedTransformer`, `device=cpu`,
    accuracy-verified before use (invalid if < 0.99 on random additions).
  - **Interchange intervention (activation patching)**: for a candidate node
    (attention head `z` at a token position, or the position's MLP) and a target
    digit `n`, build matched question pairs that are identical except that the
    digit-`n` pair sum falls in a *different* ST class:
    - **source** question: `ST_n` class A (e.g. no-carry, sum ≤ 8),
    - **target** question: `ST_n` class B (e.g. carry, sum ≥ 10),
    - other digits held so that (i) no lower carry propagates into `n`
      (isolates the local `ST_n`), and (ii) the two questions are matched on all
      digits except the manipulated `ST_n` driver where possible.
    Patch the node's activation at its token position from source→target and
    read the change in each answer digit `A_0..A_top`.
  - **Candidate set**: for the primary, the four Paper-2 named layer-0 nodes,
    each tested against every digit `n` (the node's target digit is *not*
    assumed; it is read off from which digit's patch produces the R-ST
    signature). For the replication (positions not pre-known), sweep all
    layer-0 heads and MLPs at question positions (indices `0..2n_digits`) ×
    digits. Also test layer-1 nodes, since the cascade resolves over layers.
  - **Metric per (node, digit)**: over many matched pairs, the fraction of
    trials where patching flips `A_{n+1}` (ST signature) and, separately, the
    fraction where it flips `A_n` (SA signature) and higher digits `A_{n+2}+`
    (cascade). Report a **signature vector** (per-answer-digit flip rate) so R-ST
    vs R-SA vs mixed is read off directly, not forced.
  - **Controls / nulls**:
    - **Same-class null**: patch between two source questions of the *same* ST
      class (should cause ~no flip) — the false-positive floor.
    - **Random-node null**: run the identical patch on a layer-0 head Paper 2
      marks unused / non-arithmetic at that position — should show no R-ST
      signature.
    - **Direction symmetry**: patch both A→B and B→A; a genuine `ST` node should
      be causal in both directions.
    - **No-lower-carry verification** per question (as in the pair-sum study).
  - **Minimal effect of interest**: an `ST_n` node must flip `A_{n+1}` on
    **≥ 50%** of counterfactual (different-class) patches while flipping it on
    **≤ 10%** of same-class patches (null), with the effect present in **both**
    patch directions and reproduced in ≥ 2 questions-sets. A node that flips
    `A_n` (not `A_{n+1}`) at ≥ 50% is classified `SA`, not `ST`.
  - **Sample sizes**: ≥ 60 matched pairs per (node, digit, direction); enough to
    separate a ≥ 50% effect from a ≤ 10% null by a wide margin. Power comes from
    the large flip-rate gap and the same-class null, not asymptotics.

- **Positive control**: a **known-causal reference patch**. Patch the *residual
  stream at the `=` token* (which by the algorithm must carry the resolved carry
  `SV`) between a carry and no-carry source for digit `n`; this must flip
  `A_{n+1}` at high rate — demonstrating the interchange harness can move the
  target metric. Conversely, patching an *input-embedding position for an
  unrelated high digit* must not flip `A_{n+1}`. If the known-causal reference
  fails to move `A_{n+1}`, the harness is broken and the run is
  `invalid`/`underpowered`, never a negative "no ST node found".

- **Success condition** (defined now): in the primary model, **≥ 1 node**
  exhibits the R-ST signature for **≥ 1 digit** (flips `A_{n+1}` ≥ 50% on
  counterfactual patches, ≤ 10% same-class null, both directions), and the
  same holds in the replication model. Verdict: **ST compute node(s) confirmed**;
  record node id, target digit, and signature vector for reuse by entry 2 and
  the A2 re-test.

- **Failure condition** (defined now): with the positive control passing — no
  candidate node in a model produces the R-ST signature for any digit (all are
  R-SA, R-none, or mixed). Verdict: **no cleanly separable ST compute node under
  this assay** — a substantive finding (carry may be computed diffusely /
  distributed), which would itself reshape entry 2 and A2/A6.

- **Ambiguous / invalid condition**:
  - Ambiguous: nodes show mixed signatures (flip both `A_n` and `A_{n+1}`) —
    consistent with polysemantic SA+ST nodes; report as mixed, do not force ST.
  - Invalid: positive-control reference patch fails; or model accuracy check
    fails.

- **Skeptic review (pre-launch)**: Run 2026-07-14 in a separate skeptic thread
  that rehydrated only from the version-controlled docs + Paper-2 facts and did
  not inherit the working thread's context.

  **Verdict: PASS WITH CONDITIONS.** The causal-interchange + signature-vector
  design is a genuine improvement (it discriminates ST from SA where the
  pair-sum ratio could not). Three BLOCKING findings:

  - **G1** — The positive control (full-resid `=` patch) validates that logits
    depend on the residual stream, not that a *single-head `hook_z` patch* — the
    assay's actual unit — can move `A_{n+1}`. Could pass trivially while a real
    "no ST node" run is an instrument failure in disguise.
  - **G2 (sharpest)** — The `U` case (sum = 9) is **incompatible** with the
    no-lower-carry construction: `U` only changes `A_{n+1}` *via* a lower carry,
    which the construction forbids. So the stated stimulus tests only a binary
    0↔1 make-carry distinction, never the tri-state `U` — i.e. it would confirm
    a *binary carry* (`SC`) node, not the tri-state `ST` that is the novelty.
  - **G3** — Two of the four Paper-2 candidates violate the note's own position
    heuristic (`P11` is `=`, `P14` is an answer position — the pair-sum study
    found `P14` is an operand-fetch head). And no pre-registered rule maps the
    signature vector to R-ST/R-SV/R-SA/mixed; operand-attention-read is a
    footnote, not a gate.

  Should-fix: G4 (SA-stratified counterfactuals), G5 (sweep triage + joint
  head+MLP patch given Paper-2 joint necessity + multiple-comparisons guard),
  G6 (per-cell null; tie ≥50% bar to the node-level control), G7 (state the
  token/position maps in the note).

  *Status: RESOLVED 2026-07-14 by the working thread. G1–G3 fixed and
  G4–G7 adopted in the amendment below. Goalposts unchanged (still confirming a
  causal ST node); the amendment makes the assay discriminating and clears the
  frozen pair-sum line's reopen conditions. Gate 1 PASSED — launch authorized.*

## Amendments (post-skeptic, pre-launch)

Dated amendments by the working thread, before any data was analyzed.

**2026-07-14 — A-1 (resolves G1): node-level positive control.** The
load-bearing positive control is now a **single-head `hook_z` patch on a
Paper-2/pair-sum-confirmed causal head**: patch `P20.L0.H1` (6-digit) /
`P18..P13` answer-position operand-fetch heads' `z` between two sources with
different `SA_n`, which must flip `A_n` at high rate — proving a single-head
z-patch moves a digit under the FLIP metric. The full-resid `=` patch is demoted
to a coarse "stream carries the resolved carry" sanity check. The ≥50% ST bar
must sit above the flip rate this node-level control achieves (A-6 below).

**2026-07-14 — A-2 (resolves G2): two stimulus batteries; tri-state claim needs
the cascade battery.**
- **Local battery (no lower carry)**: varies `Dn+D'n` across 0↔1 classes
  (sum ≤ 8 vs sum ≥ 10) only. Establishes a *binary make-carry* signature. `U`
  is NOT causally exercised here and no tri-state claim is made from it.
- **Cascade battery (lower carry present)**: constructs `...9` chains at digit
  `n` with a real lower carry so that `ST_n = U` resolves to a carry and *does*
  change `A_{n+1}` — the only regime where the `U` tri-state is causal. A node
  is confirmed a **tri-state `ST` node** only if its patch is causal in the
  cascade battery (distinguishing it from a binary `SC`/make-carry node). If a
  node is causal only in the local battery, it is reported as a **binary
  make-carry node**, not `ST`. This is the explicit anti-`SC` gate.

**2026-07-14 — A-3 (resolves G3): signature decision table + attention gate +
candidate reconciliation.**
- **Decision table** (per node×digit, over answer-digit flip-rate vector, both
  directions, vs per-cell null):
  - **R-SA(n)**: flips `A_n` ≥ bar, `A_{n+1}` ≤ null → base-add node.
  - **R-ST(n)**: flips `A_{n+1}` ≥ bar, `A_n` ≤ null, **and** attends the
    digit-`n` operand positions ≥ 0.30 mass, **and** causal in the cascade
    battery → tri-state carry producer.
  - **R-SV/cascade**: flips `A_{n+1}` ≥ bar but does **not** attend digit-`n`
    operands (consumes ST from the stream) or sits after all ST inputs →
    cascade/consumer node, not the producer.
  - **R-mixed**: flips both `A_n` and `A_{n+1}` ≥ bar → polysemantic; reported
    as mixed, not forced to ST.
  - **R-none**: no answer digit flips above null.
- **Attention-read is a gate**, not a footnote (measured operand-attention mass
  reported per node, mirroring pair-sum A-4).
- **Candidate reconciliation**: the "producer sits at/after the digit-`n`
  operands and before `=`" heuristic is *tested, not assumed*. `P11` (`=`) and
  `P14` (answer position) are retained in the sweep but flagged: if they pass
  R-ST they refute the heuristic (interesting); the primary question-position
  candidates are `P8`, `P9` and a full question-position sweep.

**2026-07-14 — A-4 (adopts G4): SA-stratified counterfactuals.** Every
source→target pair records its `(ΔST_n, ΔSA_n)`. Flip rates are reported
stratified by whether `SA_n` also changed; the `A_n`-flip null is read on pairs
where `SA_n` did change (so a null `A_n` flip is meaningful). Where feasible,
prefer pairs with colliding `SA_n` but different ST class.

**2026-07-14 — A-5 (adopts G5): sweep triage, joint patch, MC guard.** Order:
(1) Paper-2 named candidates, (2) targeted question-position expansion only if
those fail. At each candidate, patch `hook_z` alone **and** joint
`hook_z + mlp_out` at that position (Paper-2 joint-necessity), **and**
`resid_post` as a superset; z-alone null but joint/resid positive localizes to
the MLP. Multiple-comparisons: the per-cell null (A-6) is the guard; a node is
"confirmed" only if it clears the bar in both directions AND replicates across
≥ 2 question-sets, making chance passes over the grid negligible.

**2026-07-14 — A-6 (adopts G6): per-cell null, control-tied bar.** The
same-class null is computed per (node, digit, direction). The ST/SA "≥ 50%" bar
is set to `max(0.50, node_level_control_flip_rate − 0.10)` after the control
reports, so a redundant/distributed true node is not rejected by an arbitrary
absolute bar.

**2026-07-14 — A-7 (adopts G7): token/position maps.**
- 5-digit (`n_ctx = 19`): pos 0–4 = `D4..D0`, 5 = `+`, 6–10 = `D'4..D'0`,
  11 = `=`, 12 = answer sign, 13–18 = `A5..A0`. Operand of digit `n`:
  `Dn` at pos `4−n`, `D'n` at pos `10−n`. Answer digit `A_k` at pos `18−k`.
- 6-digit (`n_ctx = 22`): pos 0–5 = `D5..D0`, 6 = `+`, 7–12 = `D'5..D'0`,
  13 = `=`, 14 = sign, 15–21 = `A6..A0`. `Dn` at `5−n`, `D'n` at `12−n`,
  `A_k` at `21−k`.

**2026-07-14 — A-8 (scope): the verdict is causal-interchange ("this node is
causal for the carry into `A_{n+1}`"), and does NOT license A2's "attention
performs the addition" — that stays a separate test (pair-sum A-7).**

- **Decision impact**:
  - **ST node(s) confirmed**: unblocks entry 2 (measure that node's tri-state
    geometry) and the A2 re-test (pre-MLP sum-sufficiency at *this* node);
    bears on A5 (is the carry computed at a question position with static
    wiring?). Adds a claim-evidence entry naming the confirmed node(s).
  - **No separable ST node**: strong evidence the carry is computed
    diffusely/distributed — reshapes A2 (no single aggregate-then-discretize
    node), A3 (tri-state may not live at one node), and A6 (cascade mechanism);
    promotes a distributed-computation line.

- **Risks / confounds**:
  - Confounding the manipulated `ST_n` with `SA_n`: matched pairs must change
    the carry class while controlling the base-add digit where possible; report
    the `A_n` flip rate explicitly to catch SA leakage.
  - Lower-carry contamination (mitigated by no-lower-carry construction).
  - Patching `z` (pre-`W_O`) vs the residual contribution: patch `z` (the node's
    output) as the causal unit; the effect is read at the answer logits, so
    `W_O` is included in the causal path automatically.
  - Layer-1 nodes may be where the cascade resolves; do not restrict to layer 0.
  - A node causal for `A_{n+1}` might be an `SV`/cascade node rather than the
    `ST` producer; distinguish by whether it reads the digit-`n` operands
    (attention) and by token position (an `ST_n` producer sits at/after the
    digit-`n` operand positions and before `=`).

- **Expected artifacts**:
  - Standalone CPU script `scripts/confirm_st_node.py`.
  - Per (model, node, digit, direction): counterfactual flip-rate signature
    vector over answer digits, same-class null, positive-control reference
    results, node classification (ST / SA / mixed / none).
  - A confirmed-node registry JSON (node id, target digit, signature) for reuse.
  - Plots: signature heatmap (node × answer-digit flip rate) per model.
  - Local results folder `results/study-confirm-st-node/`; no HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (rewritten post-Gate-2): **Binary make-carry (`SC`)
  compute nodes confirmed in both models** by causal interchange-intervention,
  with a clean carry-vs-base-add head-role dissociation — but **NOT tri-state
  `ST` nodes**. In the 5-digit model `P13/P14/P15.L0.H0` (and, sub-bar,
  `P16.L0.H0`) each flip the next-higher answer digit `A_{n+1}` at **1.00** on
  counterfactual carry↔no-carry patches, flip `A_n` at **0.00**, null **0.00**,
  attend the digit-`n` operands (~0.99), both directions. The 6-digit
  replication matches at `P14–P19.L0.H2` (different head index, as Paper 2
  predicts). At each answer position **one head computes the carry (flips
  `A_{n+1}`) and a different head computes base-add (flips `A_n`)** — e.g.
  `P14.L0.H1` = SA (flips only `A_3`). **Key correction (Gate 2):** a proper
  tri-state test — fix `Dn+D'n = 9` (`U`) and toggle the *lower* carry — flips
  `A_{n+1}` at **0.00** at every node, so these heads compute the *binary*
  `Dn+D'n ≥ 10` carry; the **tri-state `U`-resolution (consulting the lower
  carry) is computed by a separate path, not localized to these nodes**, even
  though the model resolves `U` correctly (clean U+carry vs U+no-carry
  predictions differ 40/40). This clears the frozen pair-sum line's reopen
  condition (a causally-confirmed carry node exists) but the *tri-state* anchor
  entry 2 wanted is only partial: we have the make-carry locus, not yet the
  U-resolution locus.

- **Run record**:
  - Commands: `PYTHONPATH=. python3 scripts/confirm_st_node.py control`
    then `... models`; plus a full position×layer×head A_{n+1}-flip discovery
    sweep (inline) that located the head-0/head-2 carry nodes.
  - Script: [`scripts/confirm_st_node.py`](../../scripts/confirm_st_node.py)
    (standalone, CPU).
  - Env: python 3.13.7, macOS-26.5.2-arm64, numpy 2.3.2, torch 2.8.0. Repo
    commit `368f3a9` (working tree). Date 2026-07-14.
  - Models: `add_d5_l2_h3_t15K_s372001` (acc 1.000),
    `add_d6_l2_h3_t20K_s173289` (acc 1.000), weights via `huggingface_hub`.
  - Artifacts (local; no HF): `results/study-confirm-st-node/`:
    `control_*.json`, `results.json`, `confirmed_st_nodes.json` (registry for
    reuse), `signature_heatmap_5digit.png`.

- **Results**:
  - **Positive control (A-1) PASSED**: node-level single-head z-patch on the
    confirmed SA head (`P14.L0.H1` 5-digit / `P20.L0.H1` 6-digit) flips `A_n` at
    0.95 / 0.93 with same-class null 0.00 — a single-head z-patch demonstrably
    moves a digit under the FLIP metric. Bar set to
    `max(0.50, 0.95−0.10) = 0.85`.
  - **Coarse `=`-resid control returned 0.00** — not a harness failure but a
    real finding: patching `resid_post` at the `=` token (either layer) changes
    no answer digit, whereas patching residual at the answer positions does.
    The model computes each answer digit *at its own answer position*, attending
    back to the operand tokens — it does not stage the resolved answer at `=`.
  - **Confirmed ST nodes (5-digit)**: `P13.L0.H0`→A4 (digit 3), `P14.L0.H0`→A3
    (digit 2), `P15.L0.H0`→A2 (digit 1): counterfactual `A_{n+1}` flip 1.00
    (both directions), cascade-battery `A_{n+1}` flip 1.00, `A_n` flip 0.00,
    same-class null 0.00, operand attention 0.98–1.00, joint z+MLP agrees.
    `P16.L0.H0`→A1 (digit 0) flips at 0.80 — same mechanism but below the 0.85
    bar (borderline; the units digit has no lower carry to cascade so its ST is
    effectively binary).
  - **Confirmed ST nodes (6-digit, independent seed)**: `P14.L0.H1`→A6,
    `P15–P19.L0.H2`→A5..A1, all R-ST(tri) at flip 1.00 / cascade 1.00 / null
    0.00 / attention 0.83–0.97. (Different head index than the 5-digit model,
    consistent with Paper-2's documented cross-model head variability.)
  - **Head-role dissociation**: at the answer positions, one head index carries
    the base-add (`SA`, flips `A_n`) and a *different* head index carries the
    carry (`ST`, flips `A_{n+1}`). The four Paper-2-named candidates: `P14.L0.H1`
    = SA; `P8/P9/P11.L0` = no interchange effect on any answer digit (they are
    not the causal carry producers under this assay).

- **Interpretation** (rewritten post-Gate-2; goalposts unchanged, claim
  corrected):
  - **Success condition MET at the make-carry level**: ≥ 1 node with the carry
    signature (flips `A_{n+1}` ≥ bar, `A_n` ≤ null, both directions, operand
    attention ≥ 0.30) in the primary model, replicated in the independent-seed
    model. Verdict: **binary make-carry (`SC`) compute node(s) confirmed** — a
    causally-confirmed carry locus, which the pair-sum assay never had.
  - **NOT tri-state ST**: the genuine `U` test (fix sum = 9, toggle lower carry)
    flips `A_{n+1}` at 0.00, so these heads compute the binary `Dn+D'n ≥ 10`
    carry only. The tri-state resolution is real in the model (clean preds
    differ 40/40) but is produced by a path these single-head patches don't
    carry — an open locus.
  - **Head-role dissociation (solid)**: at each answer position, distinct heads
    carry `SC` (flips `A_{n+1}`) and `SA` (flips `A_n`); `P14.L0.H1` is the
    confirmed `SA` node. This is a clean, replicated functional dissociation.
  - **`=`-token observation (hedged, F2)**: a layer-0 `=`-resid patch moved no
    answer digit, whereas answer-position patches did. This is *consistent with*
    per-answer-position carry computation but is confounded (pos-11 residual
    feeds only the first answer token directly; layer-1 not tested). **Not**
    recorded as an architectural fact and **not** used to score A6.
  - **Locus caveat (F4)**: P8/P9/P11 showed no interchange effect under
    single-node answer-position patching; this is consistent with either
    non-involvement or downstream-recompute masking, so "carry computed at
    answer positions" is a statement about the causal locus *under this assay*,
    not a proof that no question-position node participates.

- **Prediction scoring** (rewritten post-Gate-2; records evidence only):
  - **A5** ("attention is static positional wiring; data-dependence in values/
    MLPs"): **supported (partial)**. Carry nodes sit at fixed positions with
    high, question-independent operand attention and the interchange flows
    through the value path (z patch suffices). Not a full test (no
    attention-variance census — entry 4). Kept.
  - **A2** ("aggregate-then-discretize at an ST node"): **untouched / now
    partially testable** — the study delivers a confirmed *make-carry* node
    locus (`P13/P14/P15.L0.H0`), better than the pair-sum SA heads, but it is an
    `SC`-grade anchor, not the tri-state `ST` node A2 ultimately concerns.
  - **A6** ("SV cascade rides the residual stream, resolved by `=`"):
    **untouched** (reverted from "leaning refuted" per F2 — the `=` patch is too
    weak/confounded to score).
  - **C3 (human)** ("attention moves, MLP transforms"): the "attention moves the
    per-digit carry" half is **supported** (carry nodes are attention heads
    reading operands); MLP-transform half untested. Partial support.
  - **A3** ("ST tri-state geometry"): **untouched**; note entry 2's anchor is a
    make-carry node, and the tri-state U-resolution locus is not yet found.
  - New incidental finding (for the record, not a standing prediction): the
    binary carry and the tri-state U-resolution are computed by **different
    paths** — the make-carry head does not transmit U-resolution.

- **Skeptic review (post-result)**: Run 2026-07-14, separate thread, docs +
  JSON artifacts only. **Verdict: BLOCK** — upheld; the original write-up
  over-claimed "tri-state ST" and the `=` architecture. Corrected below.

  Findings (verified by the working thread):
  - **F1 (critical, mandatory)**: the "cascade battery" as coded still toggled
    digit `n`'s *own* class 0↔1 (with an added lower carry that doesn't change
    digit `n`'s carry-out: class 0 → ≤ 8+1 = 9 → carry-out 0; class 1 →
    ≥ 10+1 → carry-out 1). So the `U` (sum = 9) tri-state was **never the
    manipulated variable** — a binary make-carry (`SC`) node scores 1.00 there
    too. Gate-1 blocker G2 was marked resolved in prose but not in code.
    **Fix applied**: added a genuine tri-state test (`tristate_test`) that fixes
    `Dn+D'n = 9` and toggles the *lower* carry 0↔1. **Result: tri-state flip =
    0.00 at every confirmed node** → these are **binary make-carry (`SC`)
    nodes, not tri-state `ST` nodes.** (Verified the test is live: clean
    predictions for U+carry vs U+no-carry differ 40/40, so the model *does*
    resolve U correctly — just not via these nodes' patched output.)
  - **F2 (mandatory hedge)**: the `=`-resid patch was layer-0 only and is
    autoregressively confounded (pos-11 residual feeds only the first answer
    token directly). The "carry computed at answer positions, not staged at `=`"
    architectural claim is **not earned**; demoted to a hedged note, and the
    A6 `=`-resolution score reverted to **untouched**.
  - **F4 (hedge)**: "P8/P9/P11 not causal" cannot be distinguished from
    downstream-recompute masking; softened to "no interchange effect under
    single-node answer-position patching."
  - **F3/F7**: node selection was discovery-swept on the same `A_{n+1}` metric
    then re-tested — soft circularity, mitigated by the orthogonal `A_n`=0
    specificity + operand-attention gate + null + direction symmetry; the
    discovery sweep is now noted as un-saved (reproducible via the inline sweep).

  *Status: BLOCK RESOLVED 2026-07-14 — tri-state test added and run (result:
  binary, not tri-state); exec summary, interpretation, and scoring rewritten;
  claim downgraded to "make-carry (SC) node confirmed". Gate 2 PASSED on the
  corrected read.*

- **Limitations**:
  - Interchange confirms these nodes are *causal for* `A_{n+1}`; it localizes
    the carry computation but does not by itself prove the *internal*
    computation is "compute ST then feed SV" vs "directly compute the carry into
    A_{n+1}" — the SV-cascade decomposition is entry 6.
  - The `A_{n+1}`-flip discovery sweep used the flip metric; a node that
    contributes sub-threshold (e.g. `P16` at 0.80, or redundant partial nodes)
    may be under-counted. Confirmed nodes are a sufficient set, not proven
    exhaustive.
  - hook_z patch includes `W_O`; the head+MLP joint patch agreed, but the
    head-alone-vs-MLP contribution split was not separately quantified.
  - Addition only; subtraction/mixed borrow nodes untested.

- **Doc updates** (after corrected Gate 2): ledger; claim-evidence (**CE3** —
  confirmed binary make-carry nodes + SA/SC head dissociation, hedged; and the
  incidental "binary carry and U-resolution are separate paths"); synthesis +
  summary; conjectures (A5 partial support; A6 untouched; A2 partially testable;
  incidental separate-paths note); agenda (mark entry 1 done as *make-carry node
  confirmed*; add a follow-up to locate the **tri-state U-resolution path** so
  entry 2 can point at a true tri-state locus; do **not** re-scope entry 6's
  `=` assumption on this evidence).

- **Next read**: The confirmed-node registry (`confirmed_st_nodes.json`) gives
  entry 2 a make-carry anchor and the A2 re-test a better (question-reading,
  carry-signature) locus than the SA heads. But the **tri-state `U`-resolution
  locus is unfound** — the highest-value follow-up is to path-patch the
  U-resolution (fix sum=9, vary lower carry) to find which node(s) carry it,
  since that is the actual novelty of the ST/SV algorithm. This also sharpens
  A6 (where/how the cascade `U` is resolved) far better than the `=` patch did.
