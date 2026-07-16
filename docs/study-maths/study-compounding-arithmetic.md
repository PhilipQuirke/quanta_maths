# Study: SV Compounding Arithmetic — Tail-Relay vs L1-Read, and Combiner Transfer (study-compounding-arithmetic.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #18

CE16 left one sharp puzzle: the wire into the digit-combiner carries a *fully
resolved* carry, yet each individual tri-state (ST) site (CE13) only writes a
*single-step* resolution. Where does the "999...9 carry all the way up" get
computed? Two candidates: **A11 tail-relay** — it happens at layer 0, walking
along the question-tail positions, each seeing one more digit than the last (so
each can resolve the carry only "up to its visibility horizon"); or **L1-read** —
the writes stay local and the consumer attention head does the combining. Plus we
re-attacked the one combiner detail CE16 couldn't measure (its transfer function).

What we found (both models, run 2026-07-16) — a genuine **mixed** result:
- **The combiner is a STEP.** Redesigned on-manifold (the CE16 version dead-zeroed;
  this one is gated so its endpoints reproduce CE16's known 0/1), the combiner
  flips the digit sharply once the delivered carry passes ~75% of the way from
  "no carry" to "carry" — a hard threshold, not a gradual dial. This resolves the
  last open A10 detail.
- **The tail-relay (A11) shows only a faint, single-site fingerprint.** Exactly
  one question-tail site (in the 6-digit model) reads its carry in the
  horizon-bounded way A11 predicts — resolved only when it can "see" the deciding
  digit. But it's one cell, it doesn't hold up at an adjacent depth, and the
  5-digit model doesn't replicate it. So A11 is *glimpsed, not confirmed*.
- **We could not causally pin the relay.** Swapping the tail sites' writes between
  matched problems changes the leading answer digit **not at all** — even though a
  validity check (a different kind of knockout) confirms those sites do matter for
  other digits. This is consistent with the carry being delivered **redundantly**
  (many sites, no single one decisive — the recurring theme), but honestly we
  *cannot distinguish* "redundant" from "our swap was too weak for this target."
- **The consumer's output is explained just as well by purely local information**
  as by the horizon story — so nothing forces the relay account.

Bottom line: **R-mixed.** The combiner is a step (A10 iv done); the cascade
compounding is not cleanly localizable to L0-relay vs L1-read at this granularity;
A11 stays **low** (a representational hint, not replicated, not causally shown);
A9 (single-node selection) **stays retired**. This is the last mechanism study
before the paper hand-off. Dual-gated (pre-launch conditions CA-1..CA-6;
post-result corrections F1/F2 downgrading two over-claims). Filed as CE17.

Status: **pre-run written 2026-07-16; SPRINT study** (~37 h to the paper
deadline; paper work deferred by the human — this is the last mechanism study
before the hand-off). Gate: **single combined skeptic pass** (pre-launch +
post-result in one Opus thread), per the sprint process. Evidence-integrity
rules unchanged (all headline numbers from the committed script into
`results.json`).

## Pre-run (write before the experiment)

- **Framing (working axioms)**: per the
  [working axioms](../maths-conjectures-agent.md#working-axioms), the cascade
  is provably computed *somewhere* between the operand tokens and the combiner
  input; CE16 bounded where (a canonical resolved carry arrives on the
  consumer head-pair edge, sourced from the question-tail ST cluster, never
  `=`). This study **locates and characterizes the computation** — every
  battery attributes or estimates; none tests existence.

- **The puzzle this study resolves (motivation)**: CE16 showed the wire
  carries a **canonical, format-invariant resolved carry**; CE13 showed the
  individual ST writes hold only **local class + single-step resolution**.
  Something turns single-step writes into a fully-resolved carry. Two
  candidate loci:
  - **A11 tail-relay**
    ([A11](../maths-conjectures-agent.md#a11-multi-digit-compounding-is-a-positional-l0-relay-across-the-question-tail-st-sites)):
    compounding happens **at L0, sequentially in position space** — each tail
    ST site sees one more digit pair under the causal mask, so each write is
    resolved up to its *visibility horizon*; CE13's single-step finding is the
    horizon-1 case. The L1 pair fetches the most-resolved relay. (Partially
    vindicates the human's sequential-cascade lean, relocated to L0 tail
    positions.)
  - **L1-read combination**: the writes stay local; the consumer head's
    attention-weighted **value-path sum over many local writes** implements
    the cascade select (an A9-flavored computation inside the L1 read).
  Plus the one remaining A10 parameter: **item iv, the combiner transfer
  function**, whose CE16 Battery-F instrument dead-zeroed and is redesigned
  here on-manifold.

- **Ground-truth facts / reused assets** (self-sufficiency):
  - **Visibility horizons** (from the token layout; map-relative per model).
    6-digit (`n_ctx=22`; `D_j` at pos `5−j`, `D'_j` at pos `12−j`, `=` 13,
    sign 14): a site at `D'_m` (pos `12−m`) can see the operand pairs of
    digits `j ≥ m` only (lower digits' `D'` tokens come later). Map-named
    chain-ST sites and their horizons for the `A4` consumer (chain top `n=3`):
    `P10L0H2` (A3.ST, at D'2 → sees digits ≥ 2), `P11L0H2` (A2.ST, D'1 →
    ≥ 1), `P12L0H1`/`P12L0H2` (A1/A0.ST, D'0 → all), `P14L0H1/H2`
    (A5/A4.ST, sign → all). **Prediction structure**: a site at `D'_m` can
    fully resolve a chain with deciding digit `d` iff `d ≥ m`. 5-digit
    analogous (`D'_m` at `10−m`; sign pos 12).
  - **Stimuli**: chain family `C(n,k,class)` matched pairs (reused); 6-digit
    n=3, k ∈ {2,3,4} → deciding d ∈ {1,0}; 5-digit n=3, k ∈ {2,3}. Behavioral
    gates inherited.
  - **Captures/instruments reused**: CE13's per-node write capture
    (OV-projected residual contribution at the site); CE16's per-key v-patch
    and edge-patch machinery, its Battery-R captures (for the reconstruction
    battery), its real head-pair edge contributions per carry class (for the
    combiner sweep), and the in-script carry axes (CE5/CE6).
  - **CE16 anchors**: deciding-ST v-patch arm causal at 6d k3 (0.60);
    head-pair edge real patch flip 1.00 / 0.00 by carry class — these are the
    α=1/α=0 endpoints of the combiner sweep.

- **Hypothesis / competing reads** (attribution among candidates):
  - **R-relay (A11)**: horizon decode matches visibility (site at `D'_m`
    decodes the resolved chain carry iff `d ≥ m`; below that only the
    "all-9s-down-to-m" relay state); the deepest *sufficient* relay's write
    carries the causal flip; edge reconstruction requires multi-step site
    content.
  - **R-L1-read**: horizon decode flat at single-step everywhere; no single
    site's write is decisive (only joint/all-site arms flip); edge output
    reconstructs from attention × local-class-only content.
  - **R-mixed**: partial relay (horizon decode holds at some sites/depths)
    plus L1 finishing — reported as the measured split, per the axioms a
    legitimate estimate, not an ambiguity.
  - **Combiner (independent)**: F-step (threshold α*, sharp transition) vs
    F-linear (graded); neuron top-k share high (sparse) vs low (distributed).

- **Design**:
  - **Models**: 6-digit primary, 5-digit replication (CPU; acc ≥ 0.99 else
    invalid). n ≥ 40 pairs per causal cell; n ≥ 200 questions per decode;
    Wilson/normal CIs; all numbers → `results.json`.
  - **Battery H — horizon decode (representational)**: for each chain-ST site
    and depth, decode from the site's captured L0 write: (i) the site's local
    class (positive anchor — must reproduce CE13); (ii) the resolved chain
    carry `carry_out(n)`; (iii) the intermediate relay state ("all-9s down to
    m"). Score per (site, k): does (ii) succeed exactly when `d ≥ m`
    (horizon-matched) and fail when `d < m`? Baselines per the CE13-N-4/SV-3
    conventions: wrong-role co-located head + shuffled null; decode counts at
    ≥ 0.2 above both.
  - **Battery Y — relay causality**: patch chain-ST *writes* (site-level OV
    contribution, CE13 machinery) between matched pairs, per site and jointly:
    (i) the deepest-visible sufficient relay (`m ≤ d`, largest `m`); (ii) an
    insufficient site (`m > d`); (iii) joint all-sufficient-relays (axiom-2
    redundancy arm); (iv) deciding-matched nulls per arm (expected ≤ 0.1).
    A11 predicts (i) and (iii) flip ≥ bar, (ii) ≈ null; R-L1-read predicts
    only (iii)-or-wider flips. Bar = `max(0.5, control − 0.1)`.
  - **Battery L — L1-read reconstruction (the alternative's own test)**:
    regress the consumer head's edge output (CE16 Battery-R captures + fresh
    captures) on `Σ_key attn[key] · φ(key)` with two feature sets:
    φ_local = per-site local class only; φ_horizon = per-site
    horizon-resolved state. Compare held-out R² on the carry-axis projection;
    ΔR² ≥ 0.2 calls a winner, else reported as the split.
  - **Battery T — combiner transfer, on-manifold (A10 iv redesign)**: patch
    the combiner input's head-pair contribution to
    `edge(α) = (1−α)·edge_c0 + α·edge_c1` (real captured contributions,
    lnfair conventions), α ∈ {−0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5}; local
    class fixed committed-lo. Read per α: answer-flip probability, combiner
    `carry_out`-axis output projection, and per-neuron activation profiles
    (top-k share of the output effect). **Instrument validity is built in**:
    α=0/α=1 are the real patches CE16 measured at 0.00/1.00 — if they do not
    reproduce, the battery is `invalid` before any transfer claim (the
    Battery-F dead-zero cannot recur silently).
  - **Drop order under time pressure** (the only sanctioned scope reduction):
    5-digit replication → Battery L (if CE16 captures suffice, keep) →
    never drop H, Y-core (arms i/ii/iv), or T.
  - **Pre-registered decision table**:
    | H (horizon match) | Y (deepest relay) | L (winner) | Read |
    | --- | --- | --- | --- |
    | matches visibility | flips; insufficient ≈ null | φ_horizon | **R-relay (A11)** |
    | flat single-step | no single site; joint-only | φ_local | **R-L1-read** |
    | partial | mixed | mixed / no winner | **R-mixed** (report the split) |
    Combiner: step vs linear by model comparison (logistic vs linear fit) +
    threshold α* with CI; top-k neuron share reported either way.

- **Positive controls** (failure → `invalid` for the affected battery):
  (1) Battery H's local-class decode reproduces CE13 per site; (2) Battery Y
  instrument — the site-write patch moves the answer on a CE13-known-causal
  cell (low-digit ST ablation was load-bearing); (3) Battery T endpoints
  reproduce CE16's 0.00/1.00; (4) decode baselines behave (wrong-role ≈
  chance on carry-resolved decode at horizon-0 sites); (5) behavioral gates.

- **Success condition** (defined now): controls pass; the compounding locus is
  attributed (R-relay / R-L1-read / R-mixed with the split quantified) with H,
  Y, and L agreeing or their disagreement stated as the finding; the combiner
  transfer class and threshold are estimated with neuron top-k share. Scored
  against A11 (confirmed / refuted / split), A10 iv (estimated), A9 (final
  disposition: retired if R-relay, spirit-revived-at-L1 if R-L1-read), A6
  (sequential-lean disposition), C3 (human-owned, noted).

- **Failure condition** (defined now): under the axioms only instrument
  failure (a battery's own control fails → that battery `invalid`) or
  irreducible instability (estimates vary beyond CI across depths/pairs →
  "heterogeneous at this granularity", reported with the numbers).

- **Ambiguous / invalid condition**: R-mixed is a *result* (the split is the
  estimate), not an ambiguity; models disagreeing → model-scoped; controls
  failing → invalid per battery.

- **Skeptic review (combined, sprint) — PRE-LAUNCH half**: Run 2026-07-16 in a
  separate skeptic thread (docs + working axioms + A11 + CE13/CE16 + the reused
  harness + HF maps + a verified token-layout table). **Verdict: PASS WITH
  CONDITIONS** — 2 blocking, both resolved via amendments CA-1…CA-6 below. The
  horizon *resolution rule* (`d ≥ m`) was independently re-derived as CORRECT
  (not off-by-one: the binding constraint is `D'_d ≤ D'_m ⇔ d ≥ m`; `D_d` is
  always earlier and never binds), and the 6d ST_NODES map + per-site visibility
  sets (P10 m=2, P11 m=1, P12/P14 m=0) match the layout exactly — the design's
  single feared failure is sound.
  - **C1 (blocking) — invalid depth + thin horizon discrimination.** At the
    guide's chain-top `n=3`, `k=4 → d=n−k=−1` is invalid (trips `build_chain`'s
    `assert d>=0`); and with only k∈{2,3} valid the *horizon-boundary* test (a
    site that decodes YES then NO as `d` drops below `m`) rests on a single
    within-site cell (P11 at k2→k3). Fix (CA-1): re-pin the chain-top to
    **`n_top=4`** (as CE16 did — consumer cell A5/combiner, GEO digit), recover
    k∈{2,3,4} → d∈{2,1,0}, giving genuine horizon crossings at P10 (m=2: YES at
    k=2/d=2, NO at k=3,4) and P11 (m=1: YES at k≤3, NO at k=4/d=0) — multiple
    boundary cells, not one. Report each boundary cell's N-4 wrole baseline; if a
    boundary cell fails its own baseline that arm is `underpowered`, not evidence
    for R-L1-read. P12/P14 (m=0) discriminate A11 vs R-L1-read but do NOT test the
    boundary (labelled "multi-step-presence" cells).
  - **C2 (blocking) — Battery Y positive control names the wrong CE13 unit.** The
    guide's control (3) cites CE13's low-digit *ablation* load-bearing result, but
    Y's unit is twin *interchange* — and CE13's interchange (Battery P) returned
    flip 0.00 on EVERY ST node (redundancy-blind); so the cited control is
    known-null for Y's actual intervention (the CE16-F1 trap in disguise: a
    control matched to the wrong unit makes "not used" look like "cannot"). Fix
    (CA-2): (a) make the **joint relay arm first-class** (single-site interchange
    is pre-expected null under redundancy — score it `underpowered-uninformative`,
    tied to the CE13 interchange=0.00 precedent); (b) add a **class-level ablation
    positive control** on the low-digit ST sites reproducing CE13's Battery-Ab
    above-untagged-baseline result under THIS script (that is the "instrument can
    move the answer" demo); (c) pre-register the joint arm's own deciding-matched
    null (CE16 SI-6 `same_class_twin`) so a joint flip is carry-specific. NOTE the
    skeptic confirmed Y does NOT inherit CE16's value-only SI-8 under-read, because
    Y patches at the L0 source and the L1 consumer RE-ATTENDS on the patched
    residual (pattern-borne carry captured for free).
  - **C3 (non-blocking) — "deepest visible sufficient relay" needs a set/tie-break
    + SI-4 brackets.** Fix (CA-3): define it as the SET of all sites at the largest
    `m ≤ d` (patched jointly); adopt CE16 SI-4 bracket [single, full−complement],
    report ordinal dominance if |arm-sum residual| > 0.15.
  - **C4 (non-blocking) — Battery T LN handling + non-carry blending.** Fix
    (CA-4): substitute `edge(α)` into the **pre-LN residual** at the consuming
    position (heads' z replaced), let LN act naturally (NOT CE16-F's post-LN
    inject); regress non-carry variance out of `edge_c1−edge_c0` so α parameterizes
    the carry axis; report the positional-co-rider residual per α; decide the class
    on α∈[0,1]; report α∈{−0.5,1.5} descriptively AND flag their post-LN distance
    from the c0..c1 segment as the off-manifold indicator.
  - **C5 (non-blocking) — Battery L feature nesting.** Fix (CA-5): make φ_local a
    strict sub-vector of φ_horizon (nested); report ΔR² as the held-out gain of the
    extra columns via a permutation/shuffle null on those columns (capacity alone
    cannot pass); L remains droppable.
  - **C6 (non-blocking) — lock decode-vs-use + R-mixed.** Fix (CA-6): a
    horizon-matched H decode with a FAILING Y patch at the same site scores
    "present, not used" and does NOT support A11; R-mixed must report a
    per-(site,depth) share vector with CIs (an estimate, not an escape).

  *Status: RESOLVED 2026-07-16 by the working thread via CA-1…CA-6. Pre-launch
  half of the combined sprint pass; post-result half is in Post-run.*

## Amendments (post-skeptic pre-launch, sprint)

**2026-07-16 — CA-1 (C1): chain-top re-pinned to `n_top=4`, depths k∈{2,3,4}**
(d∈{2,1,0}); horizon-boundary cells P10 (m=2) and P11 (m=1) now both cross in
range; P12/P14 (m=0) are presence-only. Consumer cell + combiner + carry axis
taken at the CE16 `n_top=4` configuration. Each boundary cell reports its N-4
wrole baseline; a boundary cell failing its baseline → `underpowered`.

**2026-07-16 — CA-2 (C2): Battery Y — joint arm first-class; class-level ablation
positive control** (reproduce CE13 Battery-Ab low-digit above untagged baseline
under this script) replaces the mis-matched interchange control; single-site
interchange nulls scored `underpowered-uninformative` (CE13 interchange=0.00
precedent); joint arm carries a deciding-matched null (`same_class_twin`).

**2026-07-16 — CA-3 (C3): "deepest sufficient relay" = SET at largest m≤d, joint;
SI-4 brackets** ([single, full−complement], ordinal if |residual|>0.15).

**2026-07-16 — CA-4 (C4): Battery T substitutes edge(α) PRE-LN** (heads' z
replaced at the consuming position), LN acts naturally; non-carry variance
regressed out of the c0→c1 direction; class decided on α∈[0,1]; extrapolation
labelled by post-LN distance from the c0..c1 segment.

**2026-07-16 — CA-5 (C5): Battery L φ_local ⊂ φ_horizon nested; ΔR² via
held-out permutation null on the extra columns** (capacity alone cannot pass).

**2026-07-16 — CA-6 (C6): decode-vs-use locked** (H-decode + failing-Y = "present
not used", not A11 support); R-mixed reports a per-(site,depth) share vector + CIs.

- **Skeptic review (combined, sprint) — POST-RESULT half**: **PENDING** — one combined pass
  (Opus thread; rehydrate from this note, the conjecture files incl. working
  axioms + A11, document rules, agenda, CE13/CE16 study notes + results.json,
  the HF maps). Suggested audit focus: (a) the horizon table — are the
  per-site visibility sets computed correctly from the token layout (an
  off-by-one here corrupts every H/Y prediction); (b) Battery Y's site-write
  patch unit (OV contribution at the site) vs CE13's ablation unit —
  instrument equivalence; (c) Battery T's α-extrapolation points (−0.5, 1.5)
  — LN may renormalize extrapolations differently than interpolations; (d)
  whether Battery L's two feature sets are genuinely nested/comparable.

- **Decision impact**: completes the SV implementation story for the paper
  hand-off (agenda entry 2): R-relay → A11 confirmed, the human's sequential
  lean partially vindicated at L0-tail, A9 fully retired, and the paper's
  cascade description becomes "sequential positional relay at L0 + L1 fetch +
  MLP combine"; R-L1-read → A11 refuted, A9's spirit revives inside the L1
  value read ("attention-weighted cascade select"), the sequential lean
  retired; R-mixed → both, with shares. Battery T fills A10 iv either way
  (and seeds B2). No paper edits from this thread.

- **Risks / confounds**:
  - **Horizon-table errors** (the design's single point of failure): the
    visibility sets are derived in-script from the token maps and asserted
    against attention masks; the skeptic is pointed at it.
  - **Site redundancy** (axiom 2): multiple sufficient relays may share the
    load — single-site nulls at sufficient sites are scored
    underpowered-uninformative; the joint arm (iii) is first-class.
  - **Decode-presence vs use** (CE12 lesson): Battery H is representational;
    only Y confers causal status; the decision table requires both.
  - **Capture-space mismatch** (Battery F's killer): T interpolates *real*
    contributions in the exact space the patch writes to (same hook, lnfair),
    with endpoint reproduction as a hard gate.
  - **Extrapolation α outside [0,1]**: reported descriptively; the transfer
    class is decided on the interpolation range.
  - **Time**: pre-registered drop order; H/Y-core/T are protected.

- **Expected artifacts**: standalone CPU script
  `scripts/compounding_arithmetic.py` (reuses CE13 capture + CE16 patch
  modules); `results/study-compounding-arithmetic/`: `results.json` (per-site
  horizon-decode table with baselines, relay-patch flips + nulls + joint arm,
  reconstruction R² table, transfer curve + neuron top-k, all controls),
  `horizon_decode.png`, `relay_patches.png`, `combiner_transfer.png`. No HF
  uploads.

## Post-run (filled 2026-07-16)

- **Executive summary**: **R-mixed, and A10 iv resolved.** Controls pass (Battery
  Y ablation instrument reproduces CE13: low-digit ST impact 0.067 6d / 0.05 5d
  over a ~0 untagged baseline; carry-axis anchor separates classes; behavioral
  gates hold). Three findings pull in different directions and the honest result
  is a **quantified split**:
  1. **Horizon decode (H) shows a SINGLE-CELL representational trace of A11, NOT
     replicated** [post-result F1 correction]. The resolved-carry decode clears
     its ≥0.2-over-both-baselines bar in exactly **one m>0 boundary cell across
     everything: 6d P11H2 (m=1) at k=2/d=2 (bacc 1.00 vs base 0.70)** — where the
     site can see the deciding digit. Its OWN within-site prediction then **fails
     at k=3** (d=1≥m=1 predicts resolve, decodes 0.59<base 0.87 = NO — a
     mismatch), so this is not a clean 1.00→fall-below-horizon curve. **5d does
     NOT replicate**: its only m>0 boundary site (P9H1) is null at both depths;
     the 5d sites that decode at 1.00 are all m=0 presence sites (test no
     horizon). The 0.67 (6d) / 0.75 (5d) "boundary match" rates are **dominated
     by trivial predicts-NO/decodes-NO cells** (weak/insufficient writers like
     P10H2, ‖write‖≈0.03); only P11H2 k2 is a non-trivial YES/YES match.
  2. **Relay causality (Y) is NULL — cause UNDETERMINED (redundancy vs
     interchange-too-weak)** [post-result F2 correction, blocking]. Patching the
     chain-ST *writes* (deepest sufficient set, insufficient, joint-all) flips the
     leading digit **0.00 at every depth, both models**. The Y-ablation control is
     valid but validates a **different unit/target** (CE13 mean-ablation of
     hook_z, full-answer accuracy on random Qs; impact 0.067/0.05 over ~0
     baseline) than Battery Y's intervention (twin-interchange of the OV write,
     leading-digit flip on chains). CE13 already showed these dissociate
     (interchange 0.00, ablation >0). So the control does NOT license "the Y
     interchange COULD move the leading digit if the relay were used" — the null
     is **consistent with redundancy (CE13 interchange 0.00) but the design cannot
     distinguish redundancy from "interchange too weak for this target"** (the
     CE16-F1 trap, which CA-2 flagged: this control validates ablation-
     load-bearing, not interchange-can-move-leading-digit).
  3. **L1-read reconstruction (L): local class is SUFFICIENT (6d), no winner
     (5d)** [F4 correction]. The consumer head-pair edge output reconstructs from
     per-site **local class alone** at r²≈0.95–0.99. 6d: ΔR²-over-null −0.05 →
     phi_local sufficient. 5d: ΔR²-over-null +0.02 (a genuine but sub-bar horizon
     gain that survived the permutation null) → **no winner**; and with r²≈0.99
     there is a **ceiling** (~1–5% headroom) so ΔR²≈0 weakly discriminates either
     way. Caveat (Limitations): local class may correlate with the resolved carry
     on chain stimuli, so local-sufficiency ≠ "the L1 reads only local".
  4. **Combiner transfer (T, A10 iv) = STEP function.** On-manifold α-sweep
     (PRE-LN substitution, endpoints gated to CE16's 0.00/1.00 — the Battery-F
     dead-zero does NOT recur): flip probability jumps 0.00→0.42→1.00 across
     α∈{0.25,0.5,0.75} with threshold **α*≈0.75**, both models, while the
     carry-axis projection rises smoothly/linearly (signature of α parameterizing
     the carry axis with a thresholded readout). A sharp discretization, not a
     graded pass-through. Caveat [F5]: CA-4's non-carry regression was applied to
     the L regressor but NOT to T's z-interpolation (raw class-mean z's), so the
     CE16 positional co-rider rides in the blend; "step" rests on the endpoint
     gate + the linear carry-proj, with the co-rider as a caveat.
  Net [F3 correction]: **R-mixed — the L1 read is local-class-sufficient (L), the
  combiner is a step (T), there is a single-cell representational horizon trace
  (H, 6d P11H2 only, not replicated), and the L0 relay is causally undetermined
  (Y null, redundancy vs weak-interchange unresolved).** A11's horizon mechanism
  is *seen* representationally at one strong-writing site but is neither replicated
  nor causally isolated; the compounding locus is not cleanly separable at this
  granularity. A10 iv resolved: the combiner is a **step**. Filed as CE17.

- **Run record**: `PYTHONPATH=. python3 scripts/compounding_arithmetic.py all`
  (CPU). Both models acc 1.000. Chain-top n_top=4 (6d) / 3 (5d) per CA-1,
  depths k∈{2,3,4}/{2,3}; n=200 decode / 300 ablation / 40 causal pairs / 60
  transfer. Artifact `results/study-compounding-arithmetic/results.json`. Seed
  20260716. Script `scripts/compounding_arithmetic.py` (reuses CE13
  `_mean_ablate_acc`/ST_NODES + CE16 patch/axis machinery).

- **Results** (headline; full numbers + CIs in results.json):
  - **Controls**: Y-ablation low-digit impact 0.067 (6d P11H2) / 0.05 (5d P11H2)
    vs untagged baseline 0.00 (6d) / 0.02 (5d) → `instrument_ok`. Per-site 6d:
    P11H2/d2 0.067, P12H1/d1 0.050, P12H2/d0 0.037, P10H2/d3 0.027, P14 d4/d5
    0.000 (reproduces CE13). Axis anchor + behavioral gates pass.
  - **H (6d boundary)**: P11H2(m=1) carry-bacc {k2/d2: **1.00**, k3/d1: 0.59,
    k4/d0: 0.55}, decodes-YES only at k2 (horizon-matched); P10H2(m=2) 0.50–0.54
    throughout (weak writer, never decodes). 5d: P9H1(m=1) {k2/d1: 0.56, k3/d0:
    0.49} — weak. Boundary match 0.67 / 0.75.
  - **Y**: deepest-sufficient / insufficient / joint all **0.00** at every depth,
    both models; instrument valid.
  - **L**: r²_local 0.95 (6d)/0.97 (5d), r²_horizon 0.94/0.99, ΔR²-over-null
    −0.048/+0.023 → winner **phi_local** both.
  - **T**: class **step**, α*=0.75, curve 6d {−0.5:0, 0:0, .25:0, .5:0.42, .75:1,
    1:1, 1.5:1}, 5d {.5:0.17, .75:0.83, 1:1}; endpoints 0.00/1.00 reproduce CE16.

- **Interpretation** (drop order untouched — nothing dropped; all four batteries +
  L ran): the compounding cannot be cleanly localized to an L0 positional relay
  *or* an L1 value-read at this granularity, and the balance of evidence leans
  **L1-local-sufficient** rather than relay. The L1 edge output is reconstructed
  **just as well by local class alone** (L, 6d), the L0 relay writes are **not
  causally isolable** (Y null; cause undetermined — redundancy vs weak
  interchange), and A11's horizon mechanism shows only a **single-cell
  representational trace** (6d P11H2, not replicated in 5d). The one solid
  positive is the **combiner**: a hard **step at α*≈0.75** (A10 iv resolved). The
  coherent reading: whatever visibility-bounded resolution exists at the L0 tail
  is at most one redundant, causally-undetermined contributor to a carry that the
  L1 pair fetches and a step-combiner discretizes — not a demonstrated L0 relay,
  and not cleanly separable from an L1-local read.

- **Prediction scoring**:
  - **A11 (positional L0 relay)**: **NOT REPLICATED / causally undetermined.** A
    horizon-consistent decode appears in exactly one m>0 boundary cell (6d P11H2
    k2, bacc 1.00), whose own within-site prediction fails at k3, and does not
    replicate in 5d; the relay is not causally isolable (Y null, cause
    undetermined) and adds no reconstruction power over local class (L). Confidence:
    hold A11 at **low** — a representational trace at one site, neither replicated
    nor causally shown.
  - **A10 iv (combiner transfer)**: **RESOLVED — step function**, α*≈0.75 both
    models (the CE16 open item closed; Battery-F dead-zero fixed by on-manifold
    PRE-LN substitution with an endpoint gate; carry-proj rises linearly while
    flip steps). Caveat: T's z-interpolation is un-regressed (co-rider rides
    along) — the step class rests on the endpoint gate + linear carry-proj.
  - **A9 (selection)**: **stays retired.** Y shows no single/deepest-site causal
    selection (all 0.00); no L1 attention-weighted select is isolable (L
    local-sufficient). Neither locus's selection is causally demonstrated.
  - **A6 / C3 (human sequential-cascade lean)**: **not vindicated by this study** —
    the only positional-sequential structure (A11's horizon trace) is single-cell,
    unreplicated, and causally undetermined; the sequential lean is neither
    confirmed nor refuted here. C3 human-owned; noted.

- **Skeptic review (post-result half of the combined pass)**: Run 2026-07-16
  (separate thread; rehydrated from this note, results.json, working axioms + A11,
  CE13/CE16, HF maps, the code). **Verdict: PASS WITH CORRECTIONS.** All headline
  numbers traceable to results.json; controls valid; the four batteries ran. One
  blocking + three wording downgrades, all applied above:
  - **F2 (blocking, RESOLVED)**: "redundancy-masked, not absent" over-read the Y
    null — the ablation control validates a different unit/target (mean-ablation,
    all-digits, random Qs) than Y's interchange (leading-digit, chains), so the
    null is causally UNDETERMINED (redundancy vs interchange-too-weak), not a
    demonstration of redundant use (CE16-F1 trap, CA-2-flagged). Reworded
    throughout (Exec point 2, Interpretation, A11 scoring, verdict).
  - **F1 (RESOLVED)**: H replication over-stated — downgraded to a single 6d cell
    (P11H2 k2), k3 mismatch noted, 5d non-replication stated, trivial-match
    inflation of the 0.67/0.75 rates flagged; A11 → low.
  - **F3 (RESOLVED)**: dropped "relay-leaning"; verdict reframed L1-local-
    sufficient + single-cell horizon trace + undetermined relay.
  - **F4 (RESOLVED)**: 5d L reported as "no winner (ceiling)", not "leans local".
  - **F5 (noted)**: CA-4 non-carry regression applied to L but not T's
    z-interpolation → co-rider caveat added to the A10-iv verdict.
  What genuinely stands (per the auditor): the **step combiner** (A10 iv), the
  **local-class sufficiency** of the L1 edge output (6d), a **single-site
  representational horizon trace**, and a **causally-undetermined (Y-null) relay**.
  Substance unchanged; only "replicated"/"redundancy-masked" over-claims removed.

- **Limitations**: the A11 horizon signal rests on ONE strong-writing boundary
  site per model (P11H2); deeper sites write too weakly to test their horizon.
  Y's null is redundancy-consistent but cannot *prove* relay use (absence of
  causal isolation). L's local-class sufficiency may reflect local↔resolved-carry
  correlation on chain stimuli. T interpolates real c0/c1 contributions that
  carry non-carry structure too (CE16 co-rider); the step class is decided on
  α∈[0,1]. 2-layer addition, two models.

- **Doc updates** (feeds paper hand-off, agenda entry 2): A11 → **low** (single-cell
  representational trace, not replicated, causally undetermined); A10 iv **RESOLVED
  — step, α*≈0.75**; A9 **stays retired**; A6/C3 sequential lean **not
  adjudicated** here. CE17 added to claim-evidence; router docs appended.
  (Applied 2026-07-16.)

- **Next read**: post-result skeptic audit, then router-doc updates, then the
  paper hand-off (agenda entry 2) — the SV implementation story is now complete
  (message, source, path, necessity, combiner-form) modulo the A11 causal-relay
  redundancy caveat.
