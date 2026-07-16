# Maths Results Summary (maths-results-summary.md)

Read role and rules: [Results Summary](thor-document-rules.md#results-summary).

Compact current-state synthesis for the `maths` thread. Orients a human or agent
quickly. Update only when a result changes the current story. Link downward for
detail; do not turn this into a ledger.

## Executive summary

Sixteen studies complete. The thread has moved from representation to a causal
mechanism for the single-digit carry, and has now probed the deep-chain regime and
the question- and answer-position representation geometry.
(1) Digit embeddings are **not** a clean circle/helix — near-isotropic in 9-D,
only a weak circular *ordering* (CE1). (2) At operand-fetch heads the value path
looks like linear *transport* of those weak embeddings (CE2). (3) The per-digit
**binary carry** is computed by specific layer-0 attention heads at answer
positions, dissociated from base-add heads (CE3). (4–5) The **tri-state
`U`-resolution runs on a separate path**: layer-0 nodes conduct the running carry
and the **answer-position layer-1 MLP combines** it with the digit's sum-class to
resolve `U` (CE5, replicated; supersedes the ambiguous CE4). (6–7) There is **no
dedicated `{0,1,U}` symbol** anywhere on the answer-position stream — the carry is
binary throughout, `U` resolved around L1-attention (CE6/CE7). (8) Attention
routing is **hybrid** — a few heads relocate their target with carry state
(6-digit; representational) (CE8). (9) The **deep `...999` cascade mechanism is not
localizable at node/attention-pattern granularity** (CE9): A9's one-hop-selection
signatures do not converge on one cell, sequential per-digit state is disfavored
where testable, and the CE8 routing cell is causally inert — an instrument limit
pointing to an edge path-patch. (10) The edge path-patch (CE10) then found `L1.H1`
(the CE8 cell) **does** causally drive the combiner at one depth (6-digit k=3) — a
computed, deciding-selective carry through the combiner MLP — but the
single-position edge instrument is underpowered at most cells, so A9 stays
circumstantial and A6 is not refuted. The emerging single-digit chain: embeddings →
L0 heads (base-add + binary carry) → L0 conduits → **L1 MLP U-combiner** → readout;
the *deep-chain* resolver is partly localized (`L1.H1`→combiner at the top digit)
but not fully.

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
