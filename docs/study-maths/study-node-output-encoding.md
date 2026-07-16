# Study: Output Encoding of the Map-Named ST/SA/SC Nodes (study-node-output-encoding.md)

Read role and rules: [Study Notes](../thor-document-rules.md#study-notes)

## Executive Summary #14

The paper ships a "map" of which little parts of the model compute each piece of an
addition (e.g. "this attention head decides the carry for digit 3"). Trusting that
map, we asked *how each of those parts writes down its answer* — and whether the map
is right.

Findings: the carry-classifier ("ST") parts **do** cleanly encode their piece
(the map is right), and removing the low-digit ones genuinely hurts the model
(they're causally load-bearing) — but they're **redundant** (swapping one out
doesn't change the answer because another covers for it). A subtlety: on the tricky
"sum-is-9" case the ST part's output already reflects the incoming carry, so it's
doing a bit of local resolution, not purely a clean local label. And the parts the
map tags as "base-add" (the sum digit) turn out **not** to write the answer digit at
the input side — the sum digit is actually computed later, at the output position.
Two review passes corrected us in both directions before this settled.

Starts from the paper's **verified per-model node maps** on Hugging Face (per the human
[C5](../maths-conjectures-human.md#c5-the-paper-empirical-results-are-reliable))
and tests
[A10](../maths-conjectures-agent.md#a10-the-sv-compounding-mechanism-is-the-map-named-answer-position-l1-fetch-and-combine-over-the-question-tail-st-cluster),
A2, and the A3 question-position remnant.

## Pre-run (write before the experiment)

- **Question**: For each map-named "output-only" node — the `ST`, `SA`, and `SC`
  attention heads named in the HF `features.json` for the two studied models — how
  does the node's **write** (its head output / OV-projected residual contribution)
  encode its sub-task value? Specifically: (a) does the node's output linearly
  separate its sub-task's classes in **full space** (ST: 3-way `{0,1,U}`; SA:
  10-way sum digit; SC: binary make-carry), above a permutation null and a
  wrong-node baseline; (b) for `ST`, is the code a **3-way categorical** (three
  separated centroids) or an ordered/binary shape (bearing on A3); (c) under
  **cascade-exercising** stimuli (incoming carry into digit `n` varies), does the
  `ST` node's output reflect the *local* sum-class only (a pure `ST` write) or does
  it already carry resolved/`SV` information (bearing on where compounding starts,
  A10); (d) does patching the map-named `ST` node causally move the tagged answer
  digit(s) on **random** questions — re-examining CE3's "paper ST candidates not
  causal" dismissal, which the map contradicts (`P11L0H2` Fail 23%, multi-digit
  Impact)?

- **Motivation**: C5 redirects the thread to build on the paper's verified maps
  rather than re-locating nodes. Step 1 is to characterize how each output-only
  node stores its value — the foundation for steps 2–5 (the `SV` mechanism, answer
  generation, the leading-digit case). It is also the correct locus for three
  long-parked items: (i) **CE3's dismissal** of the Paper-2 ST candidates as
  "not causal" was measured with a **no-lower-carry** stimulus that makes ST nodes
  look inert; the map says these nodes have real multi-digit Impact on random
  questions, so the dismissal is likely a stimulus artifact. (ii) **A2's pre-MLP
  sum-sufficiency** — the pair-sum freeze's reopen condition (a confirmed ST node +
  a discriminating metric) is now satisfiable at the *named* ST heads; the freeze
  also mandates an **outside-view literature sweep** before reopening. (iii)
  **A3's question-position transient-`U`** (B12) — the ST nodes *are* the
  question-position sites, so this is where a transient `U` would live if anywhere.

- **Ground-truth facts the design uses** (self-sufficiency):
  - **Map-named nodes** (from HF `features.json`, read 2026-07-16; the authoritative
    list, superseding any prose transcription):
    - 5-digit `add_d5_l2_h3_t15K_s372001` — **ST**: `P6L0H2`(A4), `P9L0H1`(A2),
      `P10L0H0`(A2), `P10L0H1`(A1), `P10L0H2`(A1), `P11L0H2`(A0), `P12L0H1`(A3),
      `P12L1H2`(A4); **SC**: `P13L0H0`(A3), `P15L0H0`(A1), `P16L0H0`(A0); **SA**
      (shared across two heads): `P13L0H1/H2`(A4), `P14L0H1/H2`(A3), `P15L0H1/H2`(A2),
      `P16L0H1/H2`(A1), `P17L0H1/H2`(A0).
    - 6-digit `add_d6_l2_h3_t20K_s173289` — **ST**: `P10L0H2`(A3), `P11L0H2`(A2),
      `P12L0H1`(A1), `P12L0H2`(A0), `P14L0H1`(A5), `P14L0H2`(A4); **SC**:
      `P15L0H2`(A4), `P16L0H2`(A3), `P17L0H2`(A2), `P19L0H2`(A0); **SA**:
      `P15L0H1`(A5), `P16L0H1`(A4), `P17L0H1`(A3), `P18L0H1`(A2), `P19L0H1`(A1),
      `P20L0H1`(A0).
  - **Sub-task labels** (deterministic from `(a,b)`): `SA_n=(Dn+D'n+cin)%10`
    (answer digit); `ST_n = 0/1/U` on the *local* pair sum `Dn+D'n` (≤8 / ≥10 / =9);
    `SC_n = [Dn+D'n ≥ 10]` (binary local make-carry). `cin` = carry into digit n.
  - **Read site**: the node's **head output** `blocks.{L}.attn.hook_z[:,pos,head,:]`
    (the write, pre-OV-into-residual) — the quantity the PCA `SP` tags were built
    from. Also report the **OV-projected residual write** `z @ W_O[head]` (what
    actually enters the residual) for the transport question (c).
  - **Library instruments to reuse** (C5 pointer): `make_maths_tricase_questions`
    (`tricase_test_questions_generator.py`) builds ST8/ST9/ST10 groups;
    `calc_pca_for_an`/`manual_node_pca` (`maths_pca.py`) do the 2-D PCA the `SP`
    tags used. This study **extends** that 2-D PCA to a full-space linear-probe
    encoding verdict with nulls. **Caveat found in pre-check**: the tricase
    generator randomizes lower digits with *no make-carry* into the tested digit —
    so it is *not* cascade-exercising for the incoming carry. Question (c) therefore
    uses a **separate cascade-exercising family** (below), not the raw tricase.

- **Hypothesis / competing reads** (neutral):
  - **R-clean-encoding (paper + A10 wiring)**: each map-named node's output linearly
    encodes its tagged sub-task well above null and above wrong-node baselines; ST is
    3-way categorical; the encoding is *local* (reflects `Dn+D'n` class, not the
    incoming carry) — a clean "output-only" write.
  - **R-already-compounded**: the `ST` node's output already carries resolved/`SV`
    information under cascade-exercising stimuli (the incoming carry leaks into the
    write) — compounding starts earlier than the answer-position L1 (against A10's
    "ST nodes write local class, L1 compounds").
  - **R-not-categorical (A3-relevant)**: `ST` is an ordered/binary code, not 3
    separated centroids (the `U` on the 0–1 axis, echoing CE6/CE7) — A3's off-axis
    third symbol still absent at the write.
  - **R-map-wrong / weak**: a map-named node's output does *not* encode its tag above
    baseline (the map tag is wrong for it, or it is a weak/redundant node) — reported
    per node (the paper itself notes redundant ST/SC nodes).
  - **R-CE3-corrected vs R-CE3-upheld**: patching the map-named `ST` node on random
    questions moves the tagged answer digit(s) (CE3 dismissal was a stimulus
    artifact) vs does not (CE3 upheld).

- **Design**:
  - **Models**: both studied models (5-digit primary for the fuller map; 6-digit
    replication). CPU; accuracy-verified (invalid if < 0.99).
  - **Battery E — full-space output encoding (primary)**: for each map-named node,
    train a linear probe (multinomial LR) on its **head output** to predict its
    tagged sub-task value, over a balanced stimulus set; report held-out balanced
    accuracy vs (i) a **permutation null** (shuffled labels), (ii) a **wrong-node
    baseline** (same probe target read from a *different* digit's same-role node —
    should be near chance if the write is digit-local), (iii) the **2-D PCA
    reference** (the `SP`-tag statistic, for continuity with the paper). Verdict per
    node: encodes / weak / absent.
  - **Battery S — ST shape (A3)**: at the confirmed `ST` nodes, the full-space
    geometry of the 3 classes — pairwise centroid separation, the `U`-off-axis
    fraction vs a permutation null (the CE6/CE7 axis-decomposition discriminator),
    to score 3-way-categorical vs ordered/binary.
  - **Battery C — locality / cascade (A10, R-already-compounded)**: cascade-exercising
    stimuli — fix digit `n`'s local pair (hold `ST_n`/`SA_n` local class) and **vary
    the incoming carry `cin`** by toggling the lower digits' make-carry. Does the
    `ST` node's *output* change with `cin` at fixed local class? R-clean: output
    invariant to `cin` (local write); R-already-compounded: output varies with `cin`.
    This reuses the CE5 activation-invariance idea at the *write* of the ST node.
  - **Battery P — causal re-examination of CE3 (random questions)**: interchange-patch
    the map-named `ST`/`SA`/`SC` node between random questions differing in the tagged
    sub-task value; measure the flip of the tagged answer digit(s) and the node's
    `Impact` set (which answer digits move). Compare to CE3's no-lower-carry result.
    Positive control: a known-causal SA head (CE3) must flip `A_n`.
  - **Controls / baselines**: permutation null (per node, per battery); wrong-node
    baseline; the CE3 SA-head positive control for patching; balanced classes (ST
    `U` and rare SA values oversampled); the 2-D PCA cross-check against the `SP`
    tag. A node whose **own tagged probe is at null** is `R-map-wrong/weak` for that
    node, not evidence about the algorithm.
  - **Minimal effects / bars**: "encodes" = probe balanced-acc ≥ chance + 0.20 AND
    ≥ 2× the wrong-node baseline, above the permutation null (p < 0.01, ≥ 1000
    draws). ST categorical requires 3 pairwise-separated centroids with `U`-off-axis
    beyond null. Locality: cascade `cin`-variation < 0.10 flip at fixed local class
    ⇒ local write. Causal (Battery P): tagged-digit flip ≥ 0.5 with the SA-head
    control passing. ≥ 4000 stimuli; both models or explicitly model-scoped.
  - **Outside-view sweep (freeze-mandated)**: because this reopens the pair-sum
    `instrument failure` freeze (A2), before Battery C/P a short external-literature
    read on how arithmetic-interpretability work distinguishes "a node writes a
    local sub-task value" from "a node already carries downstream/compounded state"
    (the transports-vs-computes confound) is recorded in the study note.
  - **Scope**: 2-layer addition, map-named output-only nodes, linear-probe encoding +
    interchange patching; no neuron decomposition (B2); no subtraction/mixed.

- **Positive control**: (1) a map-named `ST` node must separate its 3 ST classes
  well above chance (pre-check: `P9L0H1` = 0.93) — validates the read site; (2) the
  CE3 SA head must flip `A_n` under Battery P — validates patching. A flat encoding
  result whose positive control also fails is `invalid`.

- **Success condition** (defined now): per-node output-encoding verdicts
  (encodes/weak/absent, categorical/ordered, local/compounded) exist for the named
  `ST`/`SA`/`SC` nodes in both models; A2 (pre-MLP sum-sufficiency at a confirmed ST
  node) and the A3 remnant are scored at this locus; and the CE3 dismissal is
  confirmed or corrected under map-consistent (random / cascade) stimuli.

- **Failure / ambiguous / invalid**:
  - Substantive outcomes = the decision reads above (clean-local / already-compounded
    / not-categorical / map-wrong; CE3-corrected / upheld).
  - Invalid: positive controls fail; accuracy < 0.99.
  - Ambiguous: encoding between null and bar; models disagree; ST shape in the
    mid-band.

- **Skeptic review (pre-launch)**: Run 2026-07-16 in a separate skeptic thread
  (docs + HF `features.json`/`behaviors.json` for both models + the reused library
  code, all independently verified). **Verdict: PASS WITH CONDITIONS** — four
  blocking findings plus two should-fixes, all confirmed against the artifacts:
  - **C1 (blocking) — Battery C must hold the exact local *pair* `(Dn,D'n)` fixed**
    (not just the ST class) when varying `cin`; otherwise SA co-variance within a
    class contaminates the locality test. Add a fixed-pair + fixed-`cin`,
    re-randomize-unrelated-digits null.
  - **C2 (blocking) — verify the toggled lower digits are OUTSIDE the ST node's
    attention window** (from its pattern) and report the **OV-projected write**
    `z @ W_O` — else a `cin`-dependent write could be an in-window operand read, not
    compounding.
  - **C3 (blocking) — Battery P must use MATCHED-PAIR interchange** (reuse
    `tristate_test`/`make_pair`), not random questions, and stratify the flip by
    whether `SA_n` co-changes — else the flip is confounded by co-varying digits and
    the co-located `SA_n` (the causal-scrubbing "perfectly-correlated features"
    trap, confirmed by the outside-view sweep).
  - **C4 (blocking) — add a SAME-POSITION wrong-role baseline** (probe the tag off a
    co-located non-tagged head, and/or a remove-this-head ablation-probe); the
    cross-*digit* wrong-node baseline changes position, so it doesn't control for
    CE11's "position decodes at ~1.00 everywhere". "Encodes its tag" means "writes
    it here" only if the tagged head beats a same-position non-writer.
  - **C5 (non-blocking) — scope down the A2 claim**: the head's post-attention write
    (`hook_z`) is not A2's *pre-MLP sum-sufficiency* site; this study satisfies only
    the "confirmed ST node" half of the pair-sum reopen condition unless a dedicated
    pre-MLP equal-sum-collapse arm is added. A2 is **not** scored as tested here.
  - **C6 (should-fix) — outside-view sweep scoped** (done below).

  **Outside-view sweep (freeze-mandated, done 2026-07-16)**: literature on
  distinguishing *decodable* from *written/used* at a site. Huang & Chang 2025
  (*Causality ≠ Decodability*, counting ViTs) and Sharma et al. 2026 (Dyck
  bracket transformers) both show decodability and causal use **systematically
  dissociate** (decodable-but-inert final-layer tokens; causal-but-weakly-decodable
  mid-layer tokens). Causal scrubbing (Chan et al.) prescribes **equivalent-input
  resampling** (matched pairs, not noise/random) and explicitly **cannot separate
  perfectly-correlated features** — exactly the ST/SA co-variance risk. Read
  confirms: (i) Battery E (decoding) alone cannot claim "written here" → the
  same-position baseline (C4) and causal Battery P are load-bearing; (ii) Battery P
  must use matched-pair resampling (C3); (iii) the ST/SA correlation must be broken
  by stratification (C3), since scrubbing warns it will otherwise misattribute.
  This sweep also satisfies the pair-sum-freeze reopen requirement.

  *Status: RESOLVED 2026-07-16 by the working thread via amendments N-1…N-6 below.
  Gate 1 PASSED on the amended design.*

## Amendments (post-skeptic, pre-launch)

**2026-07-16 — N-1 (resolves C1): Battery C fixes the exact local pair.** The
cascade-locality test holds `(Dn, D'n)` **identical** across the `cin` toggle (so
`ST_n` and the local `SA` contribution are constant), varying only lower digits to
flip `cin`. Locality null: with `(Dn,D'n)` AND `cin` both fixed, re-randomize
unrelated lower/higher no-carry digits → the write change is the false-positive
floor (must be < the 0.10 locality bar). A `cin`-dependent write beyond this floor
= compounding at the ST write.

**2026-07-16 — N-2 (resolves C2): attention-window check + OV write.** Before
Battery C, confirm from each ST node's attention pattern that the lower digits used
to toggle `cin` are **not attended** by that node at its position; report the
attended positions. The transport-relevant quantity is the **OV-projected residual
write** `z @ W_O[head]` (reported alongside raw `hook_z`); the locality verdict uses
the OV write.

**2026-07-16 — N-3 (resolves C3): Battery P is matched-pair interchange,
SA-stratified.** The CE3 re-examination patches the map-named node between
**matched pairs** that differ only in the intended variable (reuse
`make_pair`/`tristate_test` from confirm-st-node). For `ST`: a genuine tri-state
contrast (fix `Dn+D'n=9`, toggle the lower carry → the *resolved* `ST` outcome
changes while the local pair is held) plus a class-change arm reported **stratified
by whether `SA_n` co-changes**, so the tagged-digit flip is attributable to the
carry/`ST` role vs the co-located `SA`. Random-question patching is withdrawn.

**2026-07-16 — N-4 (resolves C4): same-position wrong-role baseline.** A node
"encodes its tag" only if its output-probe beats BOTH (i) the cross-digit wrong-node
baseline AND (ii) a **same-position wrong-role baseline** — the same target decoded
from a *co-located head the map does not tag for that sub-task* (and, where cheap, a
remove-this-head residual-ablation probe). This closes the CE11 "position decodes
everywhere" loophole so "encodes" means "written by this head", not "decodable at
this position".

**2026-07-16 — N-5 (resolves C5): A2 scoped out.** This study does **not** test
A2's pre-MLP sum-sufficiency (a different, pre-MLP site); it only confirms/【】
characterizes the ST head's post-attention write and thereby satisfies the
"confirmed ST node" half of the pair-sum-freeze reopen condition. A2 remains
**untested**; the dedicated pre-MLP equal-sum-collapse arm (with a transport null)
is left to a future entry. The A3 question-position remnant *is* scored here (the ST
shape battery at the question-position write).

**2026-07-16 — N-6 (resolves C6): outside-view sweep recorded** (see the Skeptic
Review section above) — decodability≠causality (Huang & Chang 2025; Sharma 2026);
causal scrubbing's matched-pair resampling + correlated-feature caveat (Chan et al.).

**2026-07-16 — N-7 (added mid-run, resolves an interpretation gap): mean-ablation
companion to Battery P.** Battery P (matched-pair interchange) returned flip 0.00
for every ST node, while the map tags them causal (Fail%/Impact). To reconcile,
**Battery Ab** measures the map's *own* causal measure — mean-ablation accuracy
impact per ST node on random questions (removing the node's write vs swapping it).
This distinguishes "swap-insufficient" (interchange 0, redundancy) from "not
load-bearing" (ablation 0 too). It exercises the same unit (the head's `hook_z`
write) and is pre-registered here before interpretation. Directly consistent with
the outside-view sweep: decodability/interchange/ablation are complementary and can
dissociate (Huang & Chang 2025).

- **Decision impact**:
  - **R-clean-encoding + CE3-corrected**: confirms the map's output-only nodes and
    corrects CE3 (stimulus artifact); A2 gets its first real test at a named ST node;
    sets up A10's step-2 (the L1 fetch reads these clean local ST writes).
  - **R-already-compounded**: compounding starts at the ST write, not the answer L1 —
    reshapes A10 (the "output-only" framing is wrong); important.
  - **R-not-categorical**: A3's off-axis third symbol stays absent at the write
    (consistent with CE6/CE7 at a new, question-position locus).
  - **R-map-wrong for some node**: flags a specific map tag as not causally/repr.
    borne out — feeds back to the paper thread as a node-level correction.

- **Risks / confounds** (with mitigations):
  - **Transports-vs-computes / already-compounded confound** (the recurring trap):
    Battery C's fixed-local-class, vary-`cin` design + the wrong-node baseline
    separate a local write from carried state; the outside-view sweep is recorded.
  - **`U ≡ SA_n=9` confound**: ST `U` is the sum-9 class; the ST-shape battery uses
    the CE6/CE7 axis-decomposition (is-U vs resolution axes), not raw centroid
    distance.
  - **Shared-head SA** (two heads per SA_n): probe each head separately AND the
    summed pair; report whether the code is split across the two heads.
  - **Discovery-on-the-same-metric**: nodes are map-given (not selected on our
    metric), so Battery E/P are confirmatory, not circular; the permutation null +
    wrong-node baseline guard against reading position/label structure as encoding.
  - **Low-variance-projection trap**: report the encoding subspace's variance
    fraction; flag < 3%.

- **Expected artifacts**:
  - Standalone CPU script `scripts/node_output_encoding.py` (reuses `load_model`,
    the tricase generator, `maths_pca` helpers, and confirm-st-node patching).
  - `results/study-node-output-encoding/`: `results.json` (per-node encoding acc +
    nulls + wrong-node baselines + PCA cross-check; ST-shape table; cascade-locality
    table; Battery-P flip vectors + control); plots (per-node encoding bar,
    ST-class PCA, cascade-locality). No HF uploads.

## Post-run (fill in after the experiment)

- **Executive summary** (twice-corrected post-Gate-2; two rounds caught over-reach
  in *both* directions): **The map-named `ST` nodes cleanly *encode* their tri-state
  class in their write, and on the `U` (sum-9) case the write also reflects the
  *single-step* incoming carry `cin` — i.e. the node co-carries **local
  U-resolution**, so the write is not a *pure* local class code, but this is NOT
  shown to be multi-digit SV *compounding* (the toggle is single-step, and on `U`
  carry-out = cin by definition). A10's "ST writes a local class" premise is
  therefore **refined, not supported and not compound-confirmed**. The CE3 "not
  causal" reading is refined to REDUNDANCY and survives a proper baseline:
  single-node interchange never flips the answer (CE3 reproduced), but mean-ablation
  impact exceeds an untagged-head baseline for the **low-digit** ST nodes (map
  causal tags vindicated) and is at baseline for the high-digit ones (redundant).
  The map-named `SA` L0 heads do NOT encode the answer digit (except the leading
  digit) — the sum is computed at the answer position (CE11/CE12).** Both models
  agree *directionally* (Battery C is 5-digit-scoped — see below). Details:
  - **ST output encoding (Battery E)**: every map-named `ST` node linearly encodes
    its 3-way class from its head output at ≈ 1.00 (chance 0.33). The **N-4
    same-position wrong-role baseline** flags two 5-digit nodes (`P9L0H1`,
    `P12L0H1`) and one 6-digit (`P12L0H1`) as position-decodable-but-not-uniquely-
    head-written (wrole ≈ 0.96–1.00); the rest beat their baseline.
  - **ST write co-carries single-step U-resolution, NOT shown to compound (Battery
    C, 5-digit-scoped)**: on a **`U` (sum-9) pair** the ST node's OV write depends on
    the *single-step* `cin` (cin/null **0.75–19.4**, low/middle-digit-concentrated;
    the sign-token `P6L0H2` A4 is local at 0.75). On a `U` pair carry-out = cin *by
    definition*, and the toggle is a single lower digit — so this shows the node
    performing **local U-resolution**, NOT multi-digit compounding (which needs a
    `...999`-depth test, entry 2). The first-draft "local write" was a definite-pair
    artifact; the opposite "compounding begins here" over-reads single-step
    U-resolution. **5-digit-scoped**: the 6-digit positive control is structurally
    void (returns 0.0 — its control digit has no lower digit for `cin`), so 6-digit
    ratios are uncertified.
  - **CE3 re-examination (Batteries P + Ab, baseline-controlled)**: single-node
    **interchange flip = 0.00** for every ST node (CE3 reproduced; SA-head control
    flips 0.88–0.93). **Mean-ablation impact** exceeds the **untagged-head baseline**
    (baseline max 0.000–0.003) for the **low-digit** ST nodes (5-digit `P11L0H2`
    d0 = 0.04, `P10L0H0`/`P10L0H2` = 0.01; 6-digit all of A0–A3 = 0.02–0.037) and is
    **at baseline for the high/leading-digit** nodes (d4/d5 = 0.000). So the nodes
    ARE causally load-bearing (map correct) but **redundantly** — swap-insufficient,
    removal-harmful, concentrated where carries originate. CE3 → **redundancy
    account**, matching the paper's own "redundant ST node" caveat, and it survives
    the baseline control.
  - **SA L0 heads (Battery E)**: do **not** linearly encode the answer digit (acc
    weakly above chance, 0.11–0.22 vs chance 0.10, perm null 0.09–0.12) — **except
    the leading digit** (5-digit SA4 ≈ 0.61; 6-digit SA5 ≈ 0.60). So the `SA` L0 tag
    does **not** imply "writes `A_n`"; the sum digit is an answer-position
    computation (CE11/CE12). **SC nodes** encode their binary make-carry cleanly.

  Net (directional agreement across models): the paper's ST/SC tags are **confirmed
  as class encoders**, and the map's *causal* tags survive a baseline control
  (low-digit ST nodes matter, high-digit redundant) — a **vindication of C5's
  "trust the map"** with a redundancy caveat. **A10's "ST writes a *local* class"
  premise is refined** — the ST write also co-carries *single-step* U-resolution on
  the `U` case (so not a pure local class code), but this is **not** shown to be
  multi-digit compounding (that is entry-2's test). The `SA` L0 tag ≠ "writes the
  answer digit". Scoped: 2-layer addition, map-named nodes, encoding + interchange +
  baseline-controlled ablation + (5-digit) U-pair locality; A2 explicitly **not**
  tested (N-5).

- **Run record**:
  - Command: `PYTHONPATH=. python3 scripts/node_output_encoding.py all`.
  - Script: [`scripts/node_output_encoding.py`](../../scripts/node_output_encoding.py)
    (standalone CPU; reuses `load_model`, `make_q`, `tristate_test`,
    `flip_signature`; sklearn; node lists from HF `features.json`). Env: python
    3.13.7, macOS-26.5.2-arm64, torch 2.8.0. Repo commit `e2f5456` (working tree).
    Date 2026-07-16. Seed 20260716, n_q = 4000.
  - Models: `add_d5_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289` (both acc 1.000).
  - Artifacts (local; no HF): `results/study-node-output-encoding/results.json`.

- **Results** (corrected controls, both models):
  - **Battery E**: ST acc ≈ 0.97–1.00 (chance 0.33); SC ≈ 1.00 (chance 0.50); SA
    weakly-above-chance 0.11–0.22 (chance 0.10, perm null 0.09–0.12) except
    leading-digit SA ≈ 0.60. N-4 same-position wrole flags `P9L0H1`/`P12L0H1`
    (5-digit) + `P12L0H1` (6-digit) as position-decodable (wrole 0.96–1.00).
  - **Battery C (U pair, cin/null ratio, 5-digit-scoped)**: U_cin/null = **0.75–19.4**
    (5-digit P9L0H1 19.4, P10L0H1 15.4, but P6L0H2 sign-token A4 = 0.75 local;
    low/middle-digit-concentrated). Positive control: 5-digit SA-head 6.4 (✓);
    **6-digit control = 0.0 (structurally void — control digit has no lower digit
    for cin), so 6-digit ratios uncertified**. `attn_on_lower_carry` ≈ 0.00–0.03.
  - **Battery P**: tristate interchange flip = 0.00 for all ST nodes; SA-head control
    A_n flip 0.93 (5-digit) / 0.88 (6-digit).
  - **Battery Ab (baseline-controlled)**: untagged-head baseline max impact
    **0.000–0.003**. Convincingly above baseline: 5-digit `P11L0H2`(A0) 0.04 and the
    four 6-digit A0–A3 (0.020–0.037). **Marginal** (≤ ~1 SE at N=300): 5-digit
    `P10L0H0` 0.010, `P10L0H2` 0.007. At baseline (redundant): high-digit ST nodes
    (5-digit `P6`/`P9`/`P12`; 6-digit `P14` A4/A5) 0.000. 5-digit pattern non-monotone
    but low-digit-leaning; 6-digit cleanly low-digit-concentrated.

- **Interpretation** (against pre-stated conditions; goalposts unchanged; corrected):
  - **ST write co-carries single-step U-resolution (between R-clean and
    R-already-compounded)**: on the `U` case the write depends on the single-step
    `cin` (cin/null 0.75–19.4, low-digit-concentrated, 5-digit-scoped) — so not a
    pure local class code, but this is local U-resolution, **not** multi-digit
    compounding (single-step toggle; carry-out=cin by definition on `U`). Neither
    the first-draft "local" nor the round-2 "compounding" verdict; the honest read
    is "local class + single-step U-resolution co-located at the ST write".
  - **CE3 refined to redundancy, baseline-controlled**: interchange flips nothing
    (CE3 reproduced), but baseline-controlled ablation shows the **low-digit** ST
    nodes are genuinely load-bearing (above the 0.000–0.003 untagged baseline) while
    high-digit ones are redundant — matching the paper's "redundant ST node" note.
    The map's causal tags are **vindicated** (a C5 win) with a redundancy caveat.
  - **A3**: ST write is a clean 3-way class code; no off-axis symbol (CE6/CE7).
  - **`SA` tag ≠ writes-the-answer-digit**: only the leading-digit SA head encodes
    `A_n` (≈ 0.60, plausibly aided by the top digit's no-carry-in distribution); the
    rest are weakly-above-chance → the sum is computed at the answer position
    (CE11/CE12). The `SA` L0 tag marks operand-fetch/aggregation, not answer-write.
  - **Both models agree** on all reads.

- **Prediction scoring** (records evidence; conjecture updates after Gate 2;
  corrected):
  - **A10**: **premise REFINED, not supported and not compound-confirmed** — the ST
    write encodes the local class AND co-carries *single-step* U-resolution on `U`
    (cin-dependent), so it is not a *pure* local class code; but this is single-step
    local resolution, **not** the multi-digit SV compounding A10 places at L1 (that
    needs the entry-2 `...999`-depth test). A10's "ST local / L1 compounds" division
    is neither confirmed nor refuted here — it is sharpened: some U-resolution is
    already co-located at the ST write. (L1 fetch/combine remains entry-2's test.)
  - **A2**: **untouched** (N-5: post-attention write, not the pre-MLP sum-sufficiency
    site).
  - **A3**: question-position remnant **weakly-against** — clean 3-way class, no
    off-axis symbol.
  - **CE3 (claim-evidence)**: **refined to redundancy, baseline-controlled** —
    interchange-null reproduced; low-digit ST nodes ablation-load-bearing above an
    untagged baseline; high-digit redundant. Map tags vindicated (C5) with a
    redundancy + partial-compounding caveat.

- **Skeptic review (post-result)**: Run 2026-07-16 in a separate skeptic thread
  (docs + results.json + script), with independent analysis. **Verdict: PASS WITH
  CONDITIONS**, and the conditions **overturned a first-draft claim** — a good
  catch. Findings:
  - **F1-post (blocking for "map causally right") — Battery Ab lacked a random/
    untagged-head ablation baseline.** A 0.02–0.08 impact means nothing without a
    baseline. Fixed (N-8): the untagged-head baseline max impact is **0.000–0.003**,
    so the low-digit ST nodes (impact 0.007–0.047) **are** genuinely above baseline
    → the map's causal tags are vindicated for those nodes; high-digit ST nodes
    (impact 0.000) are at baseline → redundant. The redundancy + low-digit reading
    **survives** the control.
  - **F2-post (blocking) — the "ST write is LOCAL" claim was an artifact.** The
    locality test used a *definite* (sum≤8) pair where cin cannot change carry-out,
    plus a slack-dominated threshold, and reported a 4–27×-null effect as
    "≈ null". Fixed (N-9): re-run on a **U (sum-9) pair** where cin decides
    carry-out, reporting the **cin/null ratio**, with a positive control. Result:
    **U_cin/null = 2.4–19.4** — the ST write **does** depend on cin on the U case,
    so the write is **NOT purely local**; compounding *partly starts at the ST
    write for U inputs*. This **overturns the first-draft "local write / A10 premise
    supported" conclusion.**
  - **F3–F5 (non-blocking)** — document the `encodes` chance-correction deviation;
    SA is weakly-above-chance not flat; 0.599→"0.60"; 5-digit ablation non-monotone.

  A **Gate-2 round-2 re-review** then caught that the correction over-swung the
  other way. Findings (all applied via N-10):
  - **F1-r2 (linchpin) — single-step ≠ compounding.** `build_fixed_pair_cin`
    toggles `cin` via ONE lower digit (n−1), and on a `U` pair carry-out = `cin`
    *by definition*, so a cin-dependent write is **local U-resolution**
    (single-step), NOT the multi-digit SV **compounding** A10 concerns. Score A10
    "premise **refined** (ST write co-carries local U-resolution), **not shown to
    compound**" — reserve the compounding verdict for entry-2's multi-digit test.
  - **F2-r2 — factual: not "all ST nodes".** `P6L0H2` (5-digit sign-token A4) has
    U_cin/null = 0.75 (< 1, local). Range is **0.75–19.4** and the cin-dependence is
    **low/middle-digit-concentrated** (aligning with the ablation redundancy
    pattern), not universal.
  - **F3-r2 — 6-digit Battery-C control is structurally void.** Its positive
    control returned exactly 0.0 (the control uses a leading/units digit with no
    lower digit to source `cin`, so the effect is forced to 0). The 6-digit U-pair
    ratios are therefore **uncertified** → Battery C is **5-digit-scoped**.
  - **F4-r2 — marginal ablation nodes.** Only `P11L0H2` (5-digit) and the four
    6-digit A0–A3 clear baseline convincingly; `P10L0H0`/`P10L0H2` (impact
    0.007–0.010, baseline max 0.003, N=300) are within ~1 SE — mark "marginal".
  - **F5-r2 — "both models agree" is directional**, not cell-by-cell (6-digit C
    control void; 5-digit ablation non-monotone).

  *Status: RESOLVED 2026-07-16 by the working thread via N-10 (below). Corrected
  reads: A10 premise **refined, not shown to compound** (single-step U-resolution,
  not multi-digit compounding); cin-dependence **low-digit-concentrated (0.75–19.4),
  5-digit-scoped** (6-digit C control void); CE3 redundancy **survives** the ablation
  baseline for the strong low-digit nodes; SA-tag finding stands. Gate 2 PASSED on
  the twice-corrected read.*

**2026-07-16 — N-10 (resolves Gate-2 round-2 F1–F5): single-step vs compounding,
scope, hedging.** (F1) The Battery-C `cin` toggle is **single-step** (one lower
digit), so on `U` it shows **local U-resolution**, not multi-digit compounding; A10
is scored **premise refined, not shown to compound**; the compounding question is
entry-2's multi-digit-cascade test. (F2) The U_cin/null range is **0.75–19.4** and
**low/middle-digit-concentrated** (`P6L0H2` sign-token A4 is local, 0.75). (F3)
Battery C is **5-digit-scoped** — the 6-digit positive control is structurally void
(leading/units control digit has no lower digit to source `cin`); a valid 6-digit
control needs a non-leading answer-position SA head (future). (F4) Convincingly
above-baseline ablation nodes = `P11L0H2` (5-digit) + the four 6-digit A0–A3;
`P10L0H0`/`P10L0H2` are **marginal** (≤ ~1 SE at N=300). (F5) "Both models agree" is
**directional** only.

**2026-07-16 — N-8 (resolves F1-post): untagged-head ablation baseline.** Battery
Ab now measures the mean-ablation impact of **untagged L0 heads at the ST nodes'
positions** as the baseline; an ST node counts as causally load-bearing only if its
impact exceeds the untagged-head baseline max. Baseline max = 0.000–0.003 (both
models); the low-digit ST nodes clear it, high-digit ones do not.

**2026-07-16 — N-9 (resolves F2-post): locality on the U pair + positive control +
ratio.** Battery C now measures cin-dependence on a **U (sum-9) pair** (cin decides
carry-out — the case where cin can matter) and reports the **cin/null ratio**
(replacing the slack-dominated threshold), plus a positive control (answer-position
SA head whose write must depend on cin). Result overturns the first-draft "local
write": U_cin/null = 2.4–19.4, so the ST write reflects cin on U inputs. The
definite-pair result (where cin can't matter) is reported as a lower bound only.

- **Limitations**:
  - Linear-probe encoding + interchange/ablation; no neuron decomposition (B2).
  - Battery C's positive control was mixed (5-digit SA-head cin/null 6.4 registers,
    6-digit 0.0 does not) — so the U-pair cin-dependence is the robust finding but
    the metric's sensitivity is only partly demonstrated; the *pattern* (all ST
    nodes cin-dependent on U) is consistent, not a single-cell result.
  - Mean-ablation is one causal measure (vs an untagged-head baseline); path-ablation
    / the paper's exact Fail% may differ in magnitude — the *pattern*
    (low-digit-above-baseline, high-digit-at-baseline) is the robust finding.
  - The 5-digit ablation pattern is low-digit-*leaning* but non-monotone; 6-digit is
    cleanly low-digit-concentrated.
  - A2 not tested (pre-MLP site); the answer-digit SA locus is inferred from
    CE11/CE12 + the weak L0 SA encoding + the leading-digit exception's plausible
    distribution artifact, not directly patched.
  - The `encodes` boolean uses a chance-corrected 2×-baseline (not the literal
    2×-raw-baseline in the pre-run text); did not change any verdict.
  - 2-layer addition, two models (agree), middle+edge digits.

- **Doc updates** (after Gate 2 passes; APPEND to the human C5 working-tree edits,
  do not overwrite): ledger; claim-evidence (revise **CE3** to the
  baseline-controlled redundancy account: ST nodes encode their class,
  interchange-insufficient, low-digit ST nodes ablation-load-bearing above an
  untagged baseline, high-digit redundant; **new CE13**: the ST write is
  cin-dependent on `U` (compounding partly starts at the ST write) + `SA` L0 tag ≠
  answer-digit-write); synthesis + summary; conjectures (**A10 premise weakened** —
  ST write not purely local; A2 untouched; A3 weakly-against); agenda (complete
  entry 1; entry 2 — SV compounding at the L1 wires — inherits the corrected picture:
  the ST write is *partly* compounded already, which entry 2 must account for).

- **Next read**: the correction matters for entry 2 — the ST write is **not** a
  clean local class code (it is cin-dependent on `U`), so A10's clean "ST-local →
  L1-compounds" division is wrong; entry 2 (SV compounding at the map-named L1
  wires) must test how much compounding is already in the ST write vs added at L1.
  Entry 2 inherits: the baseline-controlled low-digit ST nodes as the load-bearing
  inputs, the redundancy caveat (single-node interchange won't flip — use
  ablation/path methods), and the answer-position SA-computation finding (the L1
  combiner produces `A_n`, consistent with CE5).
