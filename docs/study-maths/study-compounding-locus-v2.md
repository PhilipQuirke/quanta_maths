# Study: Compounding Locus v2 — Layer-Localization of the Canonical Carry via Cross-Depth Transfer (study-compounding-locus-v2.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes).

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

## Post-run (fill in after the experiment)

- **Executive summary**: *(pending)*
- **Run record / Results / Interpretation**: *(pending)*
- **Prediction scoring**: *(pending — A11 layer-localization, A9, A6)*
- **Skeptic review (post-result half)**: *(pending)*
- **Limitations / Doc updates / Next read**: *(pending)*
