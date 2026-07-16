# Maths Results Summary (maths-results-summary.md)

Read role and rules: [Results Summary](thor-document-rules.md#results-summary).

Compact current-state synthesis for the `maths` thread. Orients a human or agent
quickly. Update only when a result changes the current story. Link downward for
detail; do not turn this into a ledger.

## Executive summary

**Scope note (multi-thread): the Executive summary, Top current claims, Strongest
live caveats, and Open questions sections below all concern the ADDITION models
(CE1–CE19). The mixed add/sub model is covered separately in
[Mixed-model generalization](#mixed-model-generalization-entry-2-parallel-thread--ce20ce21)
(CE20/CE21). Do not read addition-model claims as mixed-model claims or vice
versa.**

Twenty studies complete (CE1–CE19). The addition-model SV mechanism is now
resolved end-to-end; studies #10–#20 were dual-gated.

**Representation** (Section B of the hand-off): (1) digit embeddings are **not** a
clean circle/helix — near-isotropic 9-D, weak seed-fragile ordering (CE1); (2) L0
operand-fetch looks like linear *transport* (CE2); (3) the per-digit **binary
carry** is computed by specific L0 heads, dissociated from base-add (CE3); (4–5)
the **answer-position L1 MLP is the carry combiner** (CE5); (6–7) **no dedicated
`{0,1,U}` symbol** on the answer stream — carry is a binary linear code, `U`
resolved around L1-attention (CE6/CE7); (8) hybrid attention routing, 6-digit only
(CE8); (11–12) question-side ST is **position-specific** & entangled with SV, while
the answer side shares a template with a just-in-time `SA` register and
**non-orthogonal** `=` carry slots ("tape" refuted) (CE11/CE12).

**Mechanism** (Section A of the hand-off): (13) map-named ST nodes write local
class + **single-step** U-resolution, causally vindicated (CE13); (14–15) a
**redundant SP-tagged L1 consumer head-pair** delivers the carry
**carry-specifically** into the answer-position combiner at **every** answer digit
incl. the sign-position leading digit (CE14/CE15); (16) the message is a
**canonical resolved carry**, the source is the **distributed ST cluster and never
`=`** (a depot), the head-pair path is effective (skip negligible), and the pair is
**class-necessary** (CE16); (17) the **combiner is a STEP function** (α*≈0.75,
CE17); (18) the role skeleton + step combiner **generalize d5→d13**, and redundancy
is **intrinsic, not small-model slack** (C6 not supported, CE18); (19) the
multi-digit compounding is completed **in the L1 consumer read, not by an L0
positional relay** (A11 not supported; 5d proven, 6d underpowered, linear-probe
caveat — CE19). Open/most-caveated: A11's non-linear 6d cell (CE19), the mixed
model (entry 2).

## Top current claims

- **CE1** — digit embeddings are near-isotropic 9-D categorical codes with a
  weak, training-induced circular *ordering* (not a dominant circle/helix);
  confidence Medium (no-dominant-geometry) / Low, LN-robust-weakly & seed-fragile
  (ordering — settled by the LN-aware close-out; no longer provisional). See
  [maths-claim-evidence.md#ce1](maths-claim-evidence.md#ce1-trained-addition-model-digit-embeddings-are-near-isotropic-9-d-categorical-codes-with-a-weak-training-induced-circular-ordering--not-a-dominant-low-rank-circlehelix).
- **CE2** — at layer-0 operand-fetch heads the value-path output is
  indistinguishable from linear transport of the (weakly-circular) embeddings;
  confidence Medium. Extends the "weak low-variance projection" motif from the
  embedding to the value path. See
  [maths-claim-evidence.md#ce2](maths-claim-evidence.md#ce2-at-layer-0-operand-fetch-heads-the-value-path-output-is-indistinguishable-from-linear-transport-of-the-weakly-circular-digit-embeddings).
- **CE3** — the carry is computed by **binary make-carry heads at answer
  positions**, cleanly dissociated from base-add heads (one head per role per
  position); the **tri-state `U`-resolution is a separate, not-yet-located
  path**. Confidence Medium-High (make-carry + dissociation) / Medium
  (separate path). See
  [maths-claim-evidence.md#ce3](maths-claim-evidence.md#ce3-the-carry-is-computed-by-binary-make-carry-heads-at-answer-positions-dissociated-from-base-add-heads-the-tri-state-u-resolution-is-a-separate-unlocated-path).
- **CE4** — the tri-state `U`-resolution flip is *transmitted* by an **MLP-heavy
  L0/L1 path distinct from the make-carry heads**; combiner-vs-conduit role
  unresolved (the interaction discriminator was vacuous). Confidence Low–Medium.
  *Superseded by CE5.* See
  [maths-claim-evidence.md#ce4](maths-claim-evidence.md#ce4-the-tri-state-u-resolution-flip-is-transmitted-by-an-mlp-heavy-l0l1-path-distinct-from-the-make-carry-heads-whether-it-is-combined-or-merely-relayed-is-unresolved).
- **CE5** — the tri-state `U`-**combiner is the answer-position layer-1 MLP**
  (`P14/P16.L1.MLP`; its output is the resolved `carry_out`, replicated in both
  models); **layer-0 nodes relay the running carry (conduit)**, clean in
  6-digit. Confidence Medium-High (combiner) / Medium (L0-conduit). See
  [maths-claim-evidence.md#ce5](maths-claim-evidence.md#ce5-the-tri-state-u-combiner-is-the-answer-position-layer-1-mlp-layer-0-nodes-relay-the-running-carry-conduit).
- **CE6** — at the **combiner input** the carry is a clean **binary** code (no
  off-axis tri-state; `U` split by resolution, `U→0`≈committed-0, `U→1`≈
  committed-1) — **A3's off-axis third-symbol refuted at this locus** (a
  tri-state could still exist upstream). Confidence Medium. See
  [maths-claim-evidence.md#ce6](maths-claim-evidence.md#ce6-at-the-u-combiners-input-the-carry-is-a-clean-binary-code--no-distinct-off-axis-tri-state-the-resolved-carry-is-already-linearly-present-there).
- **CE7** — **no dedicated `{0,1,U}` tri-state symbol at any answer-position
  residual site**; the carry is binary throughout and `U` is resolved to binary
  **around L1-attention** — A3's off-axis form refuted across the answer-position
  stream (scoped: comparable-magnitude symbol; weak/question-position untested).
  Confidence Medium. See
  [maths-claim-evidence.md#ce7](maths-claim-evidence.md#ce7-no-dedicated-01u-tri-state-symbol-at-any-answer-position-residual-site-u-is-resolved-to-binary-around-l1-attention).
- **CE8** — attention routing is **hybrid**: a few heads (`L1.H1` operand-read
  Q11 & answer Q14; `L0.H0` answer Q17) **relocate their target with the carry
  state** (A5's strong static-wiring form falsified) — but most cells are
  target-static. 6-digit only (5-digit inconclusive); representational not
  causal. Confidence Medium. See
  [maths-claim-evidence.md#ce8](maths-claim-evidence.md#ce8-attention-routing-is-hybrid--a-few-heads-relocate-their-target-with-carry-state-6-digit-only-most-cells-are-target-static).
- **CE9** — the **deep `...999` cascade mechanism is not localizable** at
  node/attention-pattern granularity: A9's selection signatures (a causally
  deciding-selective consumer head, a deciding-digit-tracking head) land on
  *different* cells, the CE8 routing cell is causally **inert**, sequential
  per-digit state is disfavored where testable, and real tail state is graded (not
  a stored bit). An **instrument limit** — the confirming test is an edge
  path-patch. Confidence Medium (as an ambiguous/underpowered result). See
  [maths-claim-evidence.md#ce9](maths-claim-evidence.md#ce9-at-nodeattention-pattern-granularity-the-deep-999-cascade-mechanism-is-not-localizable--no-single-cell-selection-no-sequential-per-digit-state-real-graded-tail-state).
- **CE10** — the edge path-patch resolves CE9's null on `L1.H1`: at **one depth**
  (6-digit k=3) `L1.H1` **causally drives the combiner** with a computed,
  deciding-selective carry through the combiner MLP — but the single-position edge
  instrument is **underpowered at most cells**, no head clears the ≥2-depth bar, and
  the 5-digit direct residual path is live, so A9 stays **circumstantial (not
  confirmed)** and A6 is **not refuted**. Confidence Low-Medium. See
  [maths-claim-evidence.md#ce10](maths-claim-evidence.md#ce10-at-the-combiner-edge-a-single-l1-head-carries-the-top-cascade-digits-computed-carry-at-one-depth--a-causal-crumb-but-the-single-position-edge-instrument-is-underpowered).
- **CE11** — the tri-state carry `ST` is **position-specific** at question positions
  (an `ST` probe does not transfer across digits; mean-centering doesn't restore it)
  and **entangled with `SV`** beyond their independent labels — **C2's
  template-sharing + orthogonality halves, A4's transfer-for-free, and A8's
  interference are all challenged for `ST`**. SA lives at the answer position (not
  assessable at the question site). Confidence Medium (ST-scoped, cross-model). See
  [maths-claim-evidence.md#ce11](maths-claim-evidence.md#ce11-the-tri-state-carry-st-is-position-specific-at-question-positions-no-cross-position-probe-transfer-and-geometrically-entangled-with-sv--c2a4-template-sharing-and-a8-interference-challenged-for-st).
- **CE12** — the **answer phase is a split layout**: `SA` is a just-in-time
  **register** (absent at `=`, present at its own answer position — A4 supported);
  `SV` is **resolved/present at `=`** (CE7-consistent) but its per-digit slots are
  **not orthogonal** (orthogonal-**tape** refuted); both share an **answer-side
  template** that transfers across answer positions, unlike question-side `ST`
  (CE11) — so template-sharing is position-of-computation-dependent. Confidence
  Medium (cross-model split). See
  [maths-claim-evidence.md#ce12](maths-claim-evidence.md#ce12-answer-phase-layout-is-split--sa-is-a-just-in-time-register-absent-at---sv-is-resolvedpresent-at--but-its-per-digit-slots-are-not-orthogonal-tape-refuted-both-share-an-answer-side-template).
- **CE13** — **C5 step 1**: the HF-map-named `ST`/`SC` nodes **encode** their
  sub-task class in their write (paper tags confirmed); the `ST` write also
  co-carries **single-step local U-resolution** (cin-dependent on `U`) — not shown
  to be multi-digit compounding (A10 premise **refined**). **CE3 refined to
  redundancy, baseline-controlled**: single-node interchange flips nothing but
  low-digit ST-node ablation exceeds an untagged-head baseline (map causally
  vindicated — a C5 win; high-digit nodes redundant). Map-named `SA` L0 heads do
  **not** write the answer digit (except leading). Confidence Medium (directional
  cross-model). See
  [maths-claim-evidence.md#ce13](maths-claim-evidence.md#ce13-map-named-st-nodes-encode-their-class-and-co-carry-single-step-u-resolution-not-multi-digit-compounding-map-named-sa-l0-heads-do-not-write-the-answer-digit).
- **CE14** — **C5 steps 2–4 (A10 core)**: **carry-specific** attention-edge delivery
  from the map-named answer-position L1 heads to the L1-MLP combiner is causally
  **sufficient at ≥ 2 depths** (deciding-matched null = 0.00), via a **redundant
  H1/H2 pair**, consumer-head-specific — **A10's fetch-to-combiner core partially
  confirmed**. But single-head **selection is NOT shown** (tracking head H2 ≠
  single-depth-edge-causal H1), it is **sufficiency not necessity** (H1 ablation
  inert), the **direct path is not excluded** (underpowered), and value content is
  not head-specific. A9 not supported; A6 economy supported at H2. Confidence Medium.
  Took 3 Gate-2 rounds (over-claim → over-correction-on-broken-null → calibrated).
  See
  [maths-claim-evidence.md#ce14](maths-claim-evidence.md#ce14-carry-specific-attention-edge-delivery-to-the-answer-position-combiner-a10-core-partially-confirmed-single-head-selection-not-shown-redundant-sufficiency-not-necessity).
- **CE15** — **C5 step 5 (leading-digit hard case)**: the leading digit `A_top` is
  produced by the **same carry-specific L1-head-edge delivery** to the sign-position
  combiner (mirrors CE14; 5-digit genuine depth spread, 6-digit deep-chains only,
  shallow unadjudicated). **A10 consolidated across all answer digits, NOT raised**
  (held at medium; Link-4 readout quarantined; economy uninformative at the sign
   bottleneck; direct path not excluded). **This completes C5's 5-step program for the
   addition model.** Confidence Medium. See
   [maths-claim-evidence.md#ce15](maths-claim-evidence.md#ce15-the-leading-answer-digit-is-produced-by-carry-specific-l1-head-edge-delivery-to-the-sign-position-combiner-mirrors-ce14--c5-step-5).
- **CE16** — **SV implementation (A10 items i–iii resolved, iv open)**: on the
  confirmed wiring, the head→combiner edge carries a **canonical (format-invariant)
  resolved carry** (i); the **source is the distributed question-tail ST cluster,
  never `=`** — the `=` value arm flips 0.00 with OV-projection ≈ 0, so **`=` is a
  depot not a value source** (ii, adjudicates the source fork toward distributed
  ST); the **head-pair (SV) path is effective (flip 1.00) while the skip carries
  negligible carry** (0.14/0.077; power 1×/2× = 0.00 → not used, not formally
  excluded) and the **pair is class-necessary** (necessity-over-baseline 1.07 6d /
  0.85 5d) (iii); the **combiner form is OPEN** (Battery F instrument invalid) (iv).
  **A9 stays retired** (selection level); **A6 raised to class level.** Dual-gated
  (pre-launch SI-1..SI-10 + post-result F1/F2/F3). Confidence Medium-high for the
  three resolved items. See
  [maths-claim-evidence.md#ce16](maths-claim-evidence.md).
- **CE17** — **SV compounding arithmetic (A10 iv resolved; A11 low; R-mixed)**:
  the **combiner is a STEP function** (threshold α*≈0.75, both models; on-manifold
  α-sweep endpoint-gated to CE16's 0/1) — the last A10 item resolved. The **L0
  tail-relay (A11) is a single-site, unreplicated representational trace** (6d
  P11H2 decodes the resolved carry only where it sees the deciding digit, but
  fails at an adjacent depth and does not replicate in 5d); the relay is
  **causally undetermined** (twin-interchange of tail-ST writes flips the leading
  digit 0.00 everywhere, valid ablation instrument but a different unit/target —
  redundancy vs weak-interchange unresolved); the L1 edge output is
  **local-class-sufficient**. **A11 → low; A9 stays retired; A6/C3 sequential lean
   not adjudicated.** Dual-gated (CA-1..CA-6 + F1/F2). Confidence Medium-high
   (combiner), low (A11). See [maths-claim-evidence.md#ce17](maths-claim-evidence.md).
- **CE18** — **Cross-size SV (A12 role+combiner transfer; C6 not supported)**: on
  d5/d6/d10/d13 (all accurate), the **SV role skeleton generalizes** (ST writers +
  combiner MLPs from the maps; an empirically-identified consumer head at every
  size) and the **combiner is a STEP at all sizes** (endpoint-gated; CE17
  generalizes). BUT the causal **source signatures do not reproduce at large n**
  (carry axis weak, sep ~6; =-arm & deciding-ST both 0.00 — probe-limited, source
  fork untested at n≥10). **C6 (redundancy = small-model slack) NOT SUPPORTED**:
  single-node ST ablation ~0 at every size and the class-vs-single redundancy gap
  does not shrink d5→d6→d10 ({0.056,0.116,0.324}; d13 inconclusive) → **redundancy
  is intrinsic, not slack**. A11 not rescued by scale. **A12 role+combiner →
  medium-high; tightening/C6 → low; A10 iv step generalizes; A11 unchanged.**
   Dual-gated (XS-A..XS-F + F1/F2). See
   [maths-claim-evidence.md#ce18](maths-claim-evidence.md).
- **CE19** — **Compounding locus: L1-read, not an L0 relay (A11 not supported)**:
  the last open SV question. Using a **decorrelation lever** (a chain-ST site inside
  the 999-run has local class fixed at U while the resolved carry varies) and the
  decisive **invisible-decorrelated** cells (deciding digit below the site's
  horizon — a carry there could only be relayed), the resolved carry decodes at
  **chance** in both models. Where the write is readable at that depth (5d) that =
  a genuine "no relayed carry" → **R-L1-read** (compounding completed in the L1
  consumer read); 6d underpowered (deep-chain writes wash out). A class-level
  knock-out confirms the ST cluster is carry-**necessary** (necessity anchor; DH
  load-bearing). **A11 → low (not supported, not rejected); A9 retired; human
  sequential-cascade lean not supported at L0.** Caveats: linear probe; one
   readable invisible cell; 6d underpowered. Dual-gated (LOC-1..LOC-6 + F1–F4). See
   [maths-claim-evidence.md#ce19](maths-claim-evidence.md).
- **CE24** — **Compounding locus v2: CONCLUSIVE — the canonical carry emerges in the
  L1 read** (supersedes CE19). Reframed to LAYER-localization via **cross-depth
  transfer** on the full residual at the see-everything gather (fixing CE19's
  washout + ill-posed invisible cell), **9-free** stimuli (human's `66666+33334`/
  `33433` insight). The answer-agnostic carry transfers cross-depth **1.00 at the L1
  combiner input** (answer-top NOT on the axis: 0.12 vs 0.78 full-residual) while
  **L0's output has no carry-specific canonical code** (extreme-pair transfer
  ~chance with CIs, high ceiling; weak transfer = magnitude nuisance). 9-free ≡
  9-containing. With CE16's causal delivery → **L1 read is where the canonical carry
  emerges**, both models. **A11 rejected/low; compounding locus = L1 read.** Caveats:
  linear probe (rotated-frame L0 not excluded); L0 at `=` (ST/sign via CE17/CE19);
  own causal battery invalid (locus per CE16). Dual-gated (CLV2-1..6 + F1–F7). See
  [maths-claim-evidence.md#ce24](maths-claim-evidence.md).
- **CE24 TF addendum (token-time)** — **LAZY propagation + eager local**: closing
  the token-time question (CE24 settled the layer). The propagated canonical carry
  (cross-deciding-position transfer, run-break-decorrelated, 9-free) appears **only
  at the sign token, L1** — ~chance at every operand token and at `=`, never at L0 —
  a **~2-token deferral past full input availability (D'_0)** and past the `=`
  gather (chance; `=`-is-a-depot). The **local single-step make-carry is eager
  in-place at L0** (CE13). So: single-step computed eagerly in place (L0);
  multi-digit propagation represented lazily at the answer read (L1). Reusable
  cross-model tool `quanta_maths/maths_temporal_finalization.py` (may differ by
  model — run the zoo). Linear-probe; coarse tail; both models. See
  [maths-claim-evidence.md#ce24](maths-claim-evidence.md).

## Strongest live caveats

- Weights-only / correlational: nothing yet on whether the model *uses* any of
  this geometry (needs causal path-patching).
- **A2 (aggregate-then-discretize) is untested**: the first assay was an
  `instrument failure` (answer-position nodes, not confirmed `ST` nodes; a
  ratio metric that could not separate aggregation from transport).
- A recurring method trap: tidy low-D structures (digit circle, ST tri-cluster,
  sum arc) keep turning out to be **low-variance projections**; verdicts require
  explicit nulls and independent node-role confirmation.

## Open questions shaping near-term work

- The **deep `...999` cascade fork** (A6 sequential vs A9 selection) is the top
  question but was shown (CE9) to be **unresolvable at node/attention-pattern
  granularity** — the next instrument is an **edge path-patch** of the candidate
  L1-head→L1-MLP-combiner edge (queue entry 1). Are the CE8 carry-routing L1 heads
  *causally* load-bearing? — CE9 says the single-head pattern-redirect finds them
  inert, so this needs the finer edge test.
- **CE11/CE12 together**: the sub-task representation is **position-of-computation
  dependent** — question-side `ST` is position-specific (no transfer) while
  answer-side `SA`/`SV` share a transferable template; the answer phase is a
  **just-in-time sum register + a non-orthogonal (not-tape) resolved-carry layout at
  `=`**. Open: is the `=` carry actually *used* downstream (causal), and is SV
  present beyond a full operand recompute (fair baseline)?
- Still open: the tri-state `U` is **never a dedicated symbol** at answer positions
  (CE7); the A3 remnant is only a *transient* `U` at question positions or a weak
  symbol. Whether the model causally uses digit geometry (B1); A2
  aggregate-vs-transport. See [maths-next-steps.md](maths-next-steps.md).

## Mixed-model generalization (entry 2, parallel thread — CE20–CE23, CE25)

- **CE20** — the addition SV **representation replicates on the mixed add/sub
  model** `ins1_mix_d6_l3_h4_t40K_s372001` (3 layers/4 heads) across **ADD, SUB
  and NEG**: writers encode the tri-state (ST/MT/NT), the resolved carry/borrow is
  a clean **binary** code at the last-layer combiner input (SV/MV/NV), all ~1.00
  vs an untrained control at chance. Delivery is carry/borrow-specific
  (deciding-matched null 0.00) but **class-dependent**: residual-borne for all
  classes; last-layer attention additionally delivers for SUB/NEG (learned fresh)
  but not ADD (inserted addition circuit, resolves earlier). Confidence
  Medium-high (representation) / Medium (delivery). Scores C5, A10, A12.
- **CE21** — **SGN = the top-of-cascade `D≥D'` comparison delivered to the `=`
  combiner** (the CE15 leading-digit analog; boundary flip 1.00, sign
  binary-decodable 1.00, edge flip 1.00 / deciding null 0.00). **OPR** broadcast-
  decodable everywhere but consumed upstream at the SLT selector (not an additive
  rank-1 control at the combiner). Shared `SA`/`MD`/`ND` heads give a **hybrid
  A7/C2** verdict (shared heads; ADD-vs-sub readouts separated > random, SUB-vs-NEG
  overlap). Confidence Medium-high (SGN) / Medium (OPR, shared-engine).
- **CE22** — the SV **mechanism** (not just representation) replicates across
  ADD/SUB/NEG: the combiner is a **STEP** (α-sweep α*≈0.5, endpoints gated —
  CE17/A10 iv), the delivered carry/borrow is a **canonical format-invariant**
  code (cross-digit transfer 1.00 — CE16 i), and **`=` is not the middle-digit
  source** (CE16 ii). Class-necessity (A6) **not scored** (redundancy-blurred).
  Confidence Medium-high. Scores A10 iv/i/ii, A12.
- **CE23** — the decisive **A7-vs-C2** test: the L1 selector state is causal
  (full patch flips to the correct ADD digit 0.96 → **shared L2 combiner**) but a
  **rank-1 operator steer flips 0%** at the selector (as at the combiner, CE21)
  and the single SLT head never selects → **A7's low-rank / function-vector
  control form is REFUTED**; the add/sub selection is a **distributed,
  high-dimensional L1 transformation** (leans C2 on selection, shared combiner).
  Confidence Medium-high. A7 → low (control mechanism); C2 → partially up.
- **CE25** — the class-dependent **delivery pathway holds across cascade depths
  2–4** (carry/borrow-specific, deciding-matched null 0.00): ADD residual-only
  (last-layer attention flip 0.00 at every depth), SUB/NEG residual + last-layer
  attention (1.00). Clears the CE14 ≥2-depth bar on the mixed model; untrained
  control delivers nothing. Confidence Medium-high. Scores A10 (delivery at depth),
  A12. Reusable cross-model sweep promoted to `quanta_maths/maths_cascade.py`.
- Library gained full three-class support (`neg_labels`, `neg_ntc_functions`/NTC,
  class-aware combiner check; deep-cascade delivery sweep `maths_cascade.py`; all
  tested). Detail:
  [study-mixed-sv-replication.md](study-maths/study-mixed-sv-replication.md),
  [study-mixed-opr-sgn.md](study-maths/study-mixed-opr-sgn.md),
  [study-mixed-sv-implementation.md](study-maths/study-mixed-sv-implementation.md),
  [study-mixed-shared-engine.md](study-maths/study-mixed-shared-engine.md),
  [study-mixed-delivery-depth.md](study-maths/study-mixed-delivery-depth.md).
