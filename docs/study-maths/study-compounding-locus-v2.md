# Study: Compounding Locus v2 — Linear-Canonical-Carry Layer-Localization via Cross-Depth Transfer (study-compounding-locus-v2.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

## Executive Summary #21

CE19 left the compounding-locus question largely inconclusive (5d only; 6d
underpowered) because it read a weak OV-write at an ill-posed "invisible" site.
This redesign asks a well-posed question — **at which layer does the multi-digit
carry become a *canonical* (position/depth-invariant) abstract bit?** — using the
full residual at the see-everything gather position and, decisively, **cross-depth
transfer** (train a carry decoder at one chain-depth, test at another): a decoder
reading the visible deciding digit *cannot* transfer, so transfer isolates a
*computed* canonical carry from raw input-reading (decode ≠ computation). Stimuli
are **9-free** (per the human's `66666+33334`/`66666+33433` insight — the U-state
comes from 6+3, so no literal `9` token can leak into the carry decode).

Result (both models, run 2026-07-16, dual-gated):
- **The canonical carry is an L1 property.** At the L1 combiner input, the
  answer-agnostic carry axis (CE16, anchored 2 digits away from the top) transfers
  cross-depth at **1.00**, while the answer-top digit does **not** ride that axis
  (0.12/0.13 vs 0.78 in the full residual) — carry-specific, answer-decorrelated.
- **L0's output (at the `=` gather position) has no carry-specific canonical code.**
  Extreme-pair cross-depth transfer is at/near chance (6d 0.51/0.53; 5d 0.13/0.50,
  with CIs) despite a high within-depth ceiling (0.81/0.91 — the signal is present,
  no washout), and the weak all-pairs transfer (0.65/0.46) is no larger than a pure
  magnitude nuisance (0.65/0.53) → leakage, not carry.
- **9-free ≡ 9-containing** (L0 0.65↔0.67; L1 1.00 both) → generalizes to the
  literal-`9` edge case.
- Combined with **CE16's causal head-pair delivery**, the **L1 read is the causal
  locus** where the resolved carry becomes canonical.

Net: this **conclusively localizes** (where CE19 could not) the emergence of the
canonical resolved carry to the **L1 read**, both models — a representational +
cited-causal result. Caveats (post-result gate): **linear-probe**; L0 tested at the
`=` gather position (ST/sign L0 sites covered by CE17/CE19); this study's own causal
battery (LC) is invalid (OV-write washout + mis-sited); a rotated-frame L0 carry is
not excluded by a reliable test (Procrustes overfit). Filed as CE24. This is the
sharper, conclusive successor to CE19; A11's positional-L0-relay stays **rejected/
low**, now with a clean layer-localization of where compounding *does* happen.

Status: **pre-run written 2026-07-16; redesign of CE19** (which was largely
inconclusive — 5d only, 6d underpowered). Gate: single combined skeptic pass.
Evidence-integrity rules unchanged. `maths` thread (addition); the mixed-model
work is a separate thread.

## Why CE19 ([study-compounding-locus.md](study-compounding-locus.md)) was inconclusive — diagnosis

CE19 asked whether the multi-digit carry is compounded by an L0 positional relay
(A11) or in the L1 consumer read, by decoding the resolved carry from a chain-ST
site's **OV write** at **invisible-decorrelated** cells. It failed on two counts:

1. **OV-write washout.** The read was `z @ W_O` — a tiny slice of the residual.
   At deep chains it is swamped, so the per-depth readability control failed at 6d
   (the reader couldn't decode *anything* from the write) → 6d underpowered.
2. **The discriminator was ill-posed under the architecture (the deeper flaw).**
   Operands are laid out **MSD-first** (`D'ₙ` early, `D'₀` late) but carries flow
   **LSD-first**. Under the causal mask, a site that *cannot see* the deciding
   digit (an "invisible" cell, high horizon `m`) is *earlier* in token order than
   any site that *can* see it. A relay would have to pass the carry **backwards in
   causal order** to reach it — impossible. So the invisible cell is at chance
   under **both** hypotheses; chance there never discriminated relay from L1-read.

**The reframe this forces (and the key architectural insight):** a naive positional
L0 relay in the carry-natural direction is **architecturally impossible**; the model
must **gather** all digit pairs at a see-everything late position (units `D'₀`, `=`,
or the sign token) and compute the carry there — either by end of **L0** or inside
the **L1 read**. So the well-posed question is not "which tail site relays" but
**"at which layer does the resolved carry become a canonical, abstract bit at the
gathering position?"**

## Pre-run

- **Framing (working axioms)**: the cascade is computed *somewhere*; CE16 showed a
  canonical resolved carry is present at the combiner **input** (transfers across
  deciding positions). This study **localizes the layer** at which that abstraction
  forms — L0 output vs the L1 read — with a strong-signal, well-posed, non-confounded
  instrument.

- **The discriminator: cross-depth transfer of the carry decode.** At a fixed
  depth, `carry_out(top) = [deciding digit is hi]`, and the gathering position
  *sees* the deciding digit, so a within-depth decode is confounded (it may just be
  reading the visible deciding digit's local class). **Cross-depth transfer breaks
  this**: train the carry decoder at depth k₁, test at depth k₂ (the deciding digit
  is at a different position). A decoder reading a position-specific local digit
  **cannot** transfer; only a **canonical** (depth-invariant) resolved-carry code
  transfers. This is CE16's canonical-carry idea turned into a **layer probe**.

- **Ground-truth / reused assets** (verified 2026-07-16 in exploratory probes):
  - Models `add_d6_l2_h3_t20K_s173289` (n_top 4), `add_d5_l2_h3_t15K_s372001`
    (n_top 3), acc 1.000. Gathering positions (see all digit pairs): units `D'₀`
    (`dpn_pos(0)`), `=` (`2·n_digits+1`), sign (`+1`). Consuming pos of `A_top` =
    `consuming_pos(n_top+1)`; combiner input = `blocks.1.ln2.hook_normalized` there.
  - `build_chain` (matched depth-k chains), `chain_carry_out` (ground-truth resolved
    carry), full `resid_post` capture. No OV-write reads (that was the CE19 washout).
  - **9-free stimuli (adopted 2026-07-16, human's `66666+33334` / `66666+33433`
    insight)**: build the chains from **non-9 digit pairs** — chain (U) digits from
    sum-9 pairs with both digits in 1–8, make-carry digits from sum≥10 pairs with
    both in 2–8, no-carry digits sum≤8. This preserves the exact carry structure of
    the `99999+000…` edge cases while ensuring **no literal `9` token flows through
    the model**, removing the residual confound where a decoder reads a `9` input
    token (present in `_rand_sum_ge10`'s `9+x` make-carries) instead of the computed
    carry. Randomized among non-9 pairs (not fixed at `6/3`) to keep diversity so
    the transfer test stays strong. **Verified 2026-07-16**: the 9-free stimuli
    reproduce the clean layer contrast (L0 gathering cross-depth transfer
    0.51/0.55/0.12/0.50 = chance, within-depth ceiling 0.83–1.0; combiner input
    transfer 1.00 — both models).
  - **Preliminary signal (motivating, to be confirmed with CIs)**: cross-depth
    transfer (train k=1 ↔ test k=n_top) of `carry_out`:
    L0 `resid_post` @ `=` ≈ **0.50/0.55 (6d), 0.11/0.50 (5d) = chance-or-below**;
    combiner input @ cons = **1.00 both models**; within-depth ceiling 0.90–1.0.

- **Hypotheses**:
  - **L0-computes**: the canonical resolved carry is present at L0's output at the
    gathering position → L0 cross-depth transfer ≈ within-depth ceiling.
  - **L1-read** (expected from the probe): L0 output carries only depth-specific raw
    content (cross-depth transfer ≈ chance) while the combiner input carries the
    canonical carry (transfer ≈ 1.0) → the abstraction forms **in the L1 read**.
  - **Partial**: intermediate L0 transfer — report the split.
  - **Eager-local + lazy-propagation** (the refined synthesis, from the edge-case
    framing + TR/TF): the **local single-step** make-carry is finalized **eagerly,
    in-place at each digit's own token, at L0** (CE13); the **multi-digit
    propagation** (the actual compounding) is finalized **lazily, at the gather
    position's L1 read** (v2). The two differ in both token position and layer —
    Battery TF localizes each.

- **Design**:
  - **Battery TR — cross-depth transfer (representational core)**: for each model,
    at each gathering position (units, `=`, sign) at **L0 output**, and at the
    **combiner input** at the consuming position, decode `carry_out(top)` and
    measure **all-pairs cross-depth transfer** (train kᵢ → test kⱼ, i≠j) plus the
    **within-depth ceiling** and a **label-shuffled null**. Bootstrap CIs on every
    bacc. Verdict per site: transfer ≈ ceiling ⇒ canonical carry present at that
    layer/position; transfer ≈ chance (CI includes 0.5, or below) ⇒ not canonical
    there. The **layer contrast** (L0-output gathering vs L1 combiner input) is the
    headline: L0-chance + L1-high ⇒ **abstraction forms in the L1 read**.
  - **Battery LC — layer-of-abstraction causal check**: the canonical carry at the
    combiner input is *produced by the L1 consumer read of the tail* (CE16 head-pair
    edge), not by the L0 content at the gathering position directly. Confirm the
    novel direction here: **cross-depth activation-patch** `resid_post(L0)` at the
    gathering position from a hi-twin at depth k₁ into a lo-target at depth k₂ (a
    **carry-matched** cross-depth patch). If L0 held the canonical carry this would
    flip the answer; predict it does **not** (L0 code is depth-specific), while the
    CE16 within-depth head-pair edge patch **does** flip (reproduced as the positive
    control). This dissociates "L0 gathered content" (depth-specific, not
    sufficient across depths) from "L1-read canonical carry" (what actually drives
    the digit).
  - **Battery LP — local-only ceiling (control for the L0 partial signal)**: the
    within-depth L0 decode is ~0.8; is that genuine partial resolution or just the
    visible deciding digit's local class? Decode `carry_out` from the **ground-truth
    visible digit values** the gathering position sees (a features-from-input
    regressor). If that local-only ceiling ≈ the L0 within-depth decode, L0 carries
    no more than locally-available raw content; the resolution is not (even
    partially) abstracted at L0.
  - **Battery TF — temporal finalization (eager vs lazy), the initialization-vs-
     finalization test** *(added 2026-07-16 from the human's `99999+00001` vs
     `99999+00100` framing)*. Build two matched edge-case families that dissociate
     **local single-step resolution** from **multi-digit propagation**:
     - **Family A** (`99…9 + 0…01`): the only make-carry is at the **units**, so
       every SVₙ is bottlenecked on the last operand token — nothing is finalizable
       early.
     - **Family B** (`99…9 + 0…0 d 0…0`): a make-carry at an **interior** digit
       (e.g. hundreds), 2+ tokens before the units, with a 9-run above it. The
       **local** make-carry SVᵢ is finalizable at its own (interior) token; the
       **propagated** top SV_top requires the ripple through the 9-run.
     Map, per **token position p × layer L**, the appearance of the **canonical**
     resolved carry — using **cross-context transfer (à la TR)**, NOT raw decode:
     train the carry decoder on Family-B-interior-make-carry stimuli, test on a
     DIFFERENT-position make-carry family, at each (p, L). **This defeats the
     input-reading confound** (raw decode of a resolved carry only tracks input
     visibility — decode ≠ computation — because the carry is a deterministic
     function of the visible digits; only a canonical, transfer-able code reflects a
     *computed* abstraction). Read out:
     - (i) **initialization vs finalization**: the earliest p where the LOCAL SVᵢ
       transfers (eager, expected at the interior make-carry token, L0 — CE13
       single-step) vs the earliest p where the PROPAGATED SV_top transfers
       (expected only at the gather, L1 — the v2 finding). Eager-local +
       lazy-propagation ⇒ the two loci differ in both position and layer.
     - (ii) **U-ratchet monotonicity** (human's claim): scanning p left→right, once
       SVₙ's canonical resolved value becomes transfer-decodable it must **stay**
       decodable (never returns to chance). A non-monotone reversal would falsify
       the ratchet — but note CE7 found **no dedicated `U` symbol** at answer
       positions (carry is binary, `U` resolved around L1-attention), so the ratchet
       is tested on the resolved-carry decodability trajectory, not on a literal `U`
       state. Report the monotonicity per SVₙ.
     Preliminary probe (2026-07-16, raw-decode, **confound-flagged**): the LOCAL
     make-carry resolves in-place at its interior token at L0 (~0.95, CE13); at the
     gather the propagated carry is L0 0.66 vs L1 0.94 (computation in the L1 read,
     not input visibility, since inputs are equally visible to both layers there).
     The raw propagation decode was input-reading-confounded at early tokens — hence
     TF uses the transfer trick.
  - **Drop order**: 5d replication is the only droppable scope; TR (both layers) +
     LC + TF are core.

- **Pre-registered decision table** (headline = the layer contrast in TR):
  | L0-output gathering transfer | Combiner-input transfer | LC cross-depth patch | Read |
  | --- | --- | --- | --- |
  | ≈ chance | ≈ ceiling (~1.0) | no flip (L0 depth-specific) | **L1-read** (abstraction forms in the L1 read) |
  | ≈ ceiling | ≈ ceiling | L0 patch flips cross-depth | **L0-computes** (gathered & abstracted by end of L0) |
  | intermediate | ~1.0 | partial | **partial** (report the split) |

- **Positive controls** (fail → battery `invalid`): (1) acc ≥ 0.99; (2) within-depth
  decode ceiling high at both layers (the signal is present — no washout, unlike
  CE19); (3) shuffled null ≈ 0.5; (4) the gathering positions verified to see all
  digit pairs (causal-mask check); (5) CE16 within-depth head-pair edge patch flips
  (LC positive control); (6) behavioral gates per depth.

- **Success condition**: the layer contrast is resolved (L1-read / L0-computes /
  partial) with CIs, **both models**, TR and LC agreeing or the split stated. This
  is expected to be conclusive where CE19 was not, because the signal is strong
  (full residual), the discriminator is well-posed (cross-depth transfer at the
  gathering position, not the ill-posed invisible tail site), and it is not
  confounded by reading the visible deciding digit.

- **Failure condition**: only instrument failure (a control fails) or intermediate
  transfer that is stable → reported as "partial abstraction at L0".

- **Skeptic review (combined, sprint)**: **PENDING**. Suggested audit focus: (a) is
  cross-depth transfer truly confound-free, or could a decoder exploit some
  depth-invariant-but-non-carry feature (e.g. total magnitude) that correlates with
  carry across depths — needs a carry-vs-magnitude control; (b) is the combiner-input
  ~1.0 transfer genuinely the *carry* and not the answer-digit identity leaking
  (CE16 handled this via the carry axis — reuse it); (c) does LC's cross-depth patch
  have a fair carry-matched null; (d) is "L1-read" over-claimed if L0 shows a real
  intermediate (~0.8 within-depth) partial signal — the LP control must adjudicate
  "partial L0" vs "local-only".

- **Decision impact**: upgrades A11's disposition from CE19's "not supported (5d
  only; 6d underpowered)" to a **both-model layer-localization**: if L1-read
  confirms, the paper states the compounding is completed in the L1 read (with the
  architectural reason — MSD-first layout forbids an L0 carry-direction relay), a
  cleaner and stronger claim than CE19. Battery TF adds the **temporal** dimension:
  a paper-grade statement of the form "**single-step carry resolution is computed
  eagerly in-place (L0, at each digit's token); multi-digit propagation is deferred
  to the gather position's L1 read**" — which unifies CE13 + v2 and directly
  answers the human's eager-vs-lazy question. Feeds the paper hand-off (updates
  claim A11 and the referee tag from [P] toward [C] on the L1-read side; adds the
  eager-local/lazy-propagation decomposition).

- **Risks / confounds**:
  - **Decode ≠ computation (the central methodological trap, surfaced by the
    edge-case probes)**: the resolved carry is a deterministic function of the
    visible input digits, so raw decodability of a resolved carry at position p only
    tracks *input visibility*, not *computation/finalization*. Every battery here
    must use **cross-context transfer** (canonical, input-pattern-invariant code) or
    a **causal** test — never raw decode alone — to claim a carry is *computed* at a
    (position, layer). This is why the CE19-style and naive spatiotemporal decodes
    were confounded.
  - cross-depth-invariant non-carry features (control (a)); answer-digit leakage at
    the combiner input (reuse CE16 carry axis); still a linear probe (add an
    MLP-probe robustness arm — a non-linear canonical carry at L0 would raise
    transfer; predict it stays chance). 2-layer/3-head addition, two models.

- **Expected artifacts**: `scripts/compounding_locus_v2.py` (reuses build_chain,
  chain_carry_out, CE16 carry axis); `results/study-compounding-locus-v2/results.json`
  (per-site/per-layer transfer matrices + ceiling + null + CIs, LC patch flips +
  null, LP local-only ceiling, controls). No HF uploads.

## Skeptic review (combined, sprint) — PRE-LAUNCH half

Run 2026-07-16 (separate thread). **Verdict: PASS WITH CONDITIONS** — 4 blocking
(control-tightening, no redesign) + 2 non-blocking, resolved via CLV2-1…CLV2-6.
The cross-depth-transfer discriminator was judged sound and the decode≠computation
rule correctly central.

- **CLV2-1 (C1, blocking) — nuisance-transfer controls.** "Transfer ⇒ canonical
  carry" needs proof the transfer is carry-specific. Add cross-depth transfer arms
  at BOTH layers for nuisances {deciding-digit magnitude bucket, total-sum bucket,
  answer-top-digit identity, #make-carries}. Verdict requires: carry transfers ≈
  ceiling at L1 while each nuisance ≤ chance+0.1 (or only as far as mechanically
  carry-correlated). Any nuisance transferring ≈1.0 at L1 ⇒ canonical-carry reading
  `invalid`. Bootstrap CIs → results.json.
- **CLV2-2 (C2, blocking) — L1 anchor pinned to the CE16 carry axis + edge, not a
  full-residual fresh probe.** Project onto CE16's answer-agnostic committed carry
  axis (`c1−c0` at a DIFFERENT digit than `A_top`) AND report the edge-contribution
  transfer (CE16 unit) beside the full-residual number. Add an answer-digit
  decorrelation control (hold `sum(n+1)` fixed while carry varies, and vice-versa);
  "canonical carry at L1" requires axis+edge to agree — full-residual-only ⇒
  "resolved-state present", not "carry".
- **CLV2-3 (C3, blocking) — don't over-claim the L0 null; add an aligned-probe
  backstop + scope the claim.** L0 within-depth ceiling is high (0.83–1.0) so L0
  HAS carry-relevant content; chance transfer only means "no shared linear frame".
  Add a held-out **Procrustes-aligned** per-depth probe (with CLV2-1 nuisance
  decorrelation): if carry transfers only after alignment, L0 has a rotated-frame
  carry (report as such); if still chance, the null is robust. Scope every claim to
  **"no position-invariant *linear* canonical carry at L0"**; delete
  "propagation is computed/deferred at L1" causal phrasings (a linear null does not
  establish where it is *computed*).
- **CLV2-4 (C4, blocking) — fix LC and make it corroborating-only.** `=` is never
  read by the consumer (CE16), so an LC patch at `=` is inert under both hypotheses.
  Patch a **consumer-read full-horizon site** (sign-token ST site; verify by
  attention mass). Add a **within-depth same-unit positive control** (resid_post(L0)
  @ that site hi→lo within depth MUST move the digit) and a **carry-matched
  cross-depth null** (hi→hi must not). If the within-depth same-unit patch doesn't
  move the answer, LC is `invalid (instrument)`, never L1-read evidence. **TR drives
  the verdict; LC corroborates.**
- **CLV2-5 (C5, non-blocking) — demote TF to secondary/exploratory (drop-first).**
  Add a within-family present/absent local-content null proving the TF transfer
  isolates abstract-carry presence (not local content that moves with the make-carry
  position). Reframe the ratchet as **"carry-decodability onset per (position,
  layer)"**; drop "U-ratchet"/"U-symbol" language (CE6/CE7: no stored U symbol).
  Not a headline claim unless clean.
- **CLV2-6 (C6, non-blocking) — 9-equivalence arm + falsifier + retitle.** Keep a
  small **9-containing arm through Battery TR** (predict equivalence to 9-free; if
  it diverges, scope the finding to 9-free). Pre-register the eager/lazy
  **falsifier**: refuted if the local single-step make-carry FAILS to transfer at L0
  at its own token, OR the propagated carry DOES transfer at L0 at the gather.
  Title/claim retitled to **"linear-canonical-carry layer-localization"** (done).

*Status: RESOLVED 2026-07-16 via CLV2-1…CLV2-6. TR (with CLV2-1/2/3 controls) is the
decisive battery; LC corroborates; LP + TF secondary.*

## Post-run (filled 2026-07-16)

- **Run record**: `PYTHONPATH=. python3 scripts/compounding_locus_v2.py all` (CPU),
  both models acc 1.000. n=350 per depth, all depths k=1..n_top, both directions +
  all-pairs cross-depth transfer, bootstrap CIs on the extreme-pair headline.
  Artifact `results/study-compounding-locus-v2/results.json`. Script
  `scripts/compounding_locus_v2.py` (reuses CE16 carry axis + edge; 9-free +
  9-containing arms). TF and LP batteries **not run** (secondary/drop-first,
  CLV2-5) — no temporal/eager-vs-lazy claim is made here.

- **Results** (headline; numbers in results.json):
  - **TR — L1 (canonical)**: carry-axis cross-depth transfer **1.00** (ceiling 1.00),
    both models; **answer-top on the axis 0.12/0.13** while answer-top on the FULL
    residual = 0.78 → the axis is carry-specific, not answer-digit (the decisive
    internal control). 9-containing arm identical (1.00).
  - **TR — L0 (`=` gather)**: **extreme-pair** cross-depth transfer ≈ chance —
    6d 0.51 [0.50,0.52] / 0.53 [0.52,0.55]; 5d 0.13 [0.07,0.18] (below-chance =
    sign-flipped, position-specific) / 0.50 — vs within-depth ceiling 0.81/0.91
    (signal present, not washout). All-pairs transfer 0.65/0.46 ≤ magnitude nuisance
    0.65/0.53 (weak transfer = magnitude leakage, not carry). 9-containing ≡ 9-free.
  - **LC (causal, corroborating)**: **INVALID** — the sign-ST site chosen has ~0
    consumer attention and the OV-write patch flips 0.00 within-depth (CE19-style
    washout + mis-sited). No causal weight taken from LC.
  - **Procrustes backstop**: unreliable (6d 0.51, 5d rose to 0.67 from 0.46 =
    overfit); reported, NOT gated on.

- **Interpretation**: the resolved multi-digit carry becomes a **canonical,
  position-invariant, answer-decorrelated bit at the L1 combiner input** (transfers
  1.00) but is **absent as a carry-specific canonical code at L0's `=` output**
  (extreme-pair transfer ~chance with high ceiling; weak transfer = magnitude
  nuisance). This is a **representational** localization; the **causal** locus is
  supplied by CE16 (head-pair edge patch flips the digit 1.00) — together they place
  the emergence of the canonical carry in the **L1 read**. This is the conclusive
  version of CE19's inconclusive fork. L0 was tested at the `=` gather position; the
  ST/sign L0 sites are covered by CE17 (local-class-sufficient) and CE19
  (invisible-decorrelated → L1-read), so the aggregate "L0 lacks a carry-specific
  canonical code" is a **cross-study** claim.

- **Prediction scoring**:
  - **A11 (positional L0 relay)**: **rejected / low** (unchanged from CE19,
    strengthened): no canonical carry at L0; the abstraction is an L1 property.
  - **Compounding locus**: **L1 read** — conclusively localized (both models,
    representational + CE16-causal), superseding CE19's 5d-only/6d-underpowered.
  - **A9**: stays retired (the L1 read is class-level/redundant, not single-node).
  - **A6**: unaffected.

- **Skeptic review (post-result half of the combined pass)**: Run 2026-07-16.
  **Verdict: PASS WITH CORRECTIONS** (1 blocking + scoping), all applied:
  - **F1 (blocking, RESOLVED)**: "compounding completes in the L1 read" was a fresh
    causal claim this study's dead LC cannot make → rescoped to "canonical carry is
    representationally an L1 property; L1 read is the causal locus **per CE16**".
  - **F2 (RESOLVED)**: L0-null now led by the **extreme-pair bootstrap CIs**
    (~chance) — the position-invariance-relevant number — with carry≈magnitude as
    corroboration (6d carry 0.652 vs magnitude 0.647 = within noise, reworded).
  - **F3 (RESOLVED)**: dropped "even after Procrustes" (5d rose = overfit); softened
    to "no carry in the raw linear frame; rotated-frame not excluded by a reliable
    test".
  - **F4 (PASS)**: L1 carry-specificity confirmed (answer-top 0.78 full → 0.12 axis).
  - **F5/F7 (RESOLVED)**: TF/LP marked not-run; L0 null scoped to `=` with CE17/CE19
    cited for ST/sign sites.
  - **F6 (noted)**: headline carry transfers have extreme-pair bootstrap CIs; the
    nuisance transfers are all-pairs point estimates (corroborating) — flagged.

- **Limitations**: linear probe (a non-linear/rotated-frame L0 carry not excluded —
  Procrustes backstop overfit/unreliable); L0 tested only at the `=` gather position
  (ST/sign covered cross-study by CE17/CE19); this study's own causal battery (LC)
  invalid (washout + mis-sited) so the causal locus leans on CE16; nuisance
  transfers are point estimates. 2-layer/3-head addition, two models.

- **Doc updates**: A11 rejected/low + **compounding locus conclusively = L1 read**
  (layer-localized); add **CE24**; update the paper hand-off (upgrades the A11 line
  from CE19's hedge to a clean both-model layer-localization). Append router docs.

- **Next read**: router-doc updates + paper hand-off refresh; then (optional,
  post-deadline) the TF temporal battery (eager-local vs lazy-propagation) and a
  non-linear/rotated-frame L0 backstop.
