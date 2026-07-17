# Glossary (thor-glossary.md)

Read role and rules: [Glossary](thor-document-rules.md#glossary).

Project-wide. There is exactly one glossary per project, not one per topic, so
different topics (`maths`, `paper`) cannot invent rival terms for the same idea.

Canonical project vocabulary. Agents tend to coin compact terms for token
efficiency; without a shared glossary those meanings drift. Define recurring
terms here, and use glossary terms consistently in code, study notes, plots, and
artifacts across every topic.

Rules:

- Define terms that recur, affect interpretation, or appear in artifacts.
- Define process terms that affect verdicts, phase order, or document updates.
- Prefer stable definitions over chronology or planning.
- Do not create near-synonyms; if a synonym is unavoidable, alias it to the canonical term.
- In reader-facing docs, link glossary terms on first important use.
- Multi-topic projects share this one file.

## Process Terms

### `agenda`

The ranked, current-facing list of planned next experiments for a topic. The agenda is not a history log; completed entries are deleted during post-run reranking and recorded in the study note and results ledger.

### `assay`

The concrete measurement harness used by a study: data, intervention or manipulation, controls, metrics, and verdict criteria. A failed assay can make a result `invalid` or cause an `instrument failure` freeze even when the scientific question remains live.

### `claim`

A durable empirical statement in the claim-evidence map, with linked evidence, confidence, and caveats. Claims are fewer than studies and should not be speculative.

### `conjecture doc`

A document containing speculative beliefs, priors, predictions, falsifiers, and alternatives. Conjecture docs are allowed to disagree with each other and must not be used as empirical evidence.

### `empirical doc`

A document whose job is to record observations, results, evidence, claims, artifacts, or current empirical synthesis. Study notes, results summaries, results synthesis, results ledgers, and claim-evidence maps are empirical docs.

### `freeze decision`

A documented decision to stop or pause an experiment line. Each freeze decision is classified as `question answered`, `question failure`, or `instrument failure`, and includes reopen conditions.

### `instrument failure`

A freeze classification where the assay class, unit of intervention, metric, data, or harness was the limiting factor. This triggers the outside-view sweep required by the agent contract.

### `question answered`

A freeze classification where the study line has answered the intended question well enough under its stated scope.

### `question failure`

A freeze classification where the question was ill-posed, unanswerable as stated, or no longer maps cleanly to a useful experiment.

### `skeptic gate`

An independent adversarial review in a separate thread. The pre-launch gate audits the study plan before execution. The post-result gate audits interpretation, verdict, and prediction scoring before conjecture updates or agenda reranking.

### `streak`

A repeated pattern in prediction scoring, such as several consecutive `refuted` or `untouched` predictions. A stale streak is a signal to revisit the conjecture frame after the post-result skeptic gate.

### `study`

One planned experiment or audit with a study note. A study contains pre-run sections, amendments, run record, results, interpretation, prediction scoring, skeptic reviews, limitations, doc updates, and next read.

### `thread`

A separate agent conversation or execution context. Thor uses separate threads for the working phase and skeptic phases so in-memory context cannot bleed across planning, review, interpretation, and document updates. This project's topic threads are `maths` and `paper`.

## Project Terms

These are the arithmetic-model interpretability terms used across this repo's
Colabs, Python code, plots, and artifacts. The long-form reference is
[terminology.md](terminology.md); this section is the canonical, anti-drift
subset. Definitions follow Paper 2
([Understanding Addition and Subtraction in Transformers](https://arxiv.org/abs/2402.02619)).

### `Pn`

Model (input or output) token position, zero-based. Example: `P18`, `P18L1H0`.

### `Ln`

Model layer `n`, zero-based. Example: `add_d6_l2_h3_t15K`, `P18L1H2`.

### `Hn`

Attention head `n`, zero-based. Example: `add_d6_l2_h3_t15K`, `P18L1H2`.

### `Mn`

MLP neuron `n`, zero-based.

### `PnLnHn`

Location / name of a single attention head at a specified layer and token position.

### `PnLnMn`

Location / name of a single MLP neuron at a specified layer and token position.

### `model name` (e.g. `add_d5_l2_h3_t30K_s372001`)

Encoded model configuration. Read as: performs addition, `n_digits=5`,
`n_layers=2`, `n_heads=3`, `training_epochs=30K`, `training_seed=372001`.

### `D`, `Dn`, `D'`, `D'n`

`D` is the first question number; `Dn` its `n`th numeric token (zero-based, `D0`
is units). `D'` is the second question number; `D'n` its `n`th token.

### `A`, `An`, `Amax`

`A` is the answer including sign. `An` is the `n`th answer token (zero-based,
`A0` is units). `Amax` is the highest token, always the `+` or `-` sign.

### `S` (Addition sub-tasks: `SA`, `SC`, `SS`, `ST`, `SV`)

`S` is the addition prefix (think Sum; aka ADD). Sub-tasks:
`SA` Basic Add `(Dn + D'n) % 10`; `SC` Make Carry `Dn + D'n >= 10`;
`SS` Make Sum 9 `Dn + D'n == 9`; `ST` TriCase; `SV` cascaded carry. See the
[ST](#st) and [SV](#sv) entries below.

### `ST`

TriCase. An addition sub-task that classifies a digit-pair sum as tri-state:
`1` (definitely carries, `Dn + D'n >= 10`), `0` (definitely no carry,
`Dn + D'n <= 8`, and `ST0` is always `0`/`1`), or `U` (uncertain — the digits
sum to exactly `9`, so a carry from the next-lower digit would cascade). The
tri-state `{0, 1, U}` output is the key novelty and shows up as 3 distinct PCA
clusters. `SV` resolves the `U` values across tokens. See [SV](#sv).

### `SV`

Cascaded carry. An addition sub-task that handles multi-digit carry cascades by
combining `ST` values from higher- to lower-value digits with the `TriAdd`
function (`SV1 = TriAdd(ST1, ST0)`, `SV2 = TriAdd(TriAdd(ST2, ST1), ST0)`, ...).
Because the last term is always `ST0` (which is `0`/`1`), every `SVn` resolves to
`0` or `1` — all `U` uncertainty is gone by the `=` token. The final answer
combines `SVn` with [SA](#s-addition-sub-tasks-sa-sc-ss-st-sv). The subtraction
parallels are `MV` (positive-answer borrow-in) and `NV` (negative-answer
neg-borrow-in).

### `STC` / `MTC` / `NTC` (combiners)

Combiner. The answer-position **last-layer MLP** that combines the resolved
cascade into the emitted answer digit `An`: `STC` for addition
(`(SA + carry) % 10`), `MTC` for positive-answer subtraction
(`(MD - borrow) % 10`), and `NTC` for negative-answer subtraction
(`(ND - neg-borrow) % 10`). Empirically a **step function** of the delivered
carry/borrow (CE17 addition; CE22 mixed). Tags: `Algo:A{d}.{STC,MTC,NTC}`.

### `M` (positive-answer Subtraction sub-tasks: `MD`, `MB`, `MZ`, `MT`, `MV`)

`M` is the positive-answer subtraction prefix (think Minus; aka SUB). Sub-tasks:
`MD` Basic Difference `(Dn - D'n) % 10`; `MB` Borrow One `Dn - D'n < 0`;
`MZ` Make Zero `Dn - D'n == 0`; `MT` TriCase (outputs `MT1/MT0/MT-1`); `MV`
cascaded borrow (the `SV` parallel: borrow INTO digit `n`).

### `N` (negative-answer Subtraction sub-tasks: `ND`, `NB`, `NZ`, `NT`, `NV`)

`N` is the negative-answer subtraction prefix (think Negative; aka NEG). The
answer magnitude is `D' - D`, so the base difference and cascade run on the
swapped operands. Sub-tasks: `ND` Basic Difference `(D'n - Dn) % 10`; `NB` Borrow
One `D'n - Dn < 0`; `NZ` Make Zero; `NT` TriCase (parallel of `ST`/`MT`); `NV`
cascaded neg-borrow (the `SV` parallel: neg-borrow INTO digit `n`).

### `GT`

Greater Than. A subtraction sub-task; `Dn.GT` is `Dn > D'n`.

### `OPR`

Operator. A sub-task attending to the `+`/`-` token in the question (determines
whether the question is addition or subtraction).

### `SGN`

Sign. A sub-task attending to the first answer token (`+` or `-`).

### `SLT`

Select. A mixed-model sub-task that selects the addition / positive-subtraction /
negative-subtraction readout (`S` / `M` / `N`) based on the `OPR` and `SGN`
values. On the studied mixed model this selection is distributed/high-dimensional,
not a low-rank switch (CE23).

### `PCA`

Principal Component Analysis. See [pca.md](pca.md).

### `EVR`

Explained Variance Ratio: the percentage of variance explained by each selected
PCA component.

### `useful tag`

A JSON-stored fact about a useful token position or node (attention head / MLP
neuron) that the model uses in predictions. See [useful_tags.md](useful_tags.md).
