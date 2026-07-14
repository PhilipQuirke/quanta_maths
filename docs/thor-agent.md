# quanta_maths Agent Auto-research Rehydration and Contract

You are a valued collaborator in this research project.
Push back if you disagree with my instructions or direction.
Make suggestions where you think they would add value.
Tell me if I am under-using your capabilities.

Read role and rules: [Agent Contract](thor-document-rules.md#agent-contract).

This is the repo-wide agent rehydration and operating guide. Read this first.

## Topics / Threads

This repo runs as a multi-topic project. Each topic has its own agent contract
and its own owned docs; the project-wide docs below are shared across topics.

| Thread | Kind | Contract |
| --- | --- | --- |
| `maths` | experiment | [maths-agent.md](maths-agent.md) |
| `paper` | publication (no experiments) | [paper-agent.md](paper-agent.md) |

You are started as a named thread, e.g. "You are the `maths` thread. Read
agents.md". After reading this file, read your thread-specific contract next.

## Project-Wide Docs

| Doc | Purpose |
| --- | --- |
| [thor-document-rules.md](thor-document-rules.md) | Portable document roles, update order, terminology discipline, skeptic gates, anti-patterns |
| [thor-glossary.md](thor-glossary.md) | Single canonical vocabulary across all topics |

## Compute And Artifacts

This project runs **locally and in Google Colab**. There is no GCP/VM fleet.

- Interactive and exploratory work happens in the Colab notebooks under
  [`notebooks/`](../notebooks/) (e.g. `QMTrain.ipynb`, `QMAnalyse.ipynb`,
  `QMAlgorithm.ipynb`, `QMSAE.ipynb`).
- Reusable, tested code lives in the `quanta_maths` and `maths_catgen` Python
  packages and is imported into the notebooks.
- Unit tests live in `tests/` and run with `pytest tests`.
- Long-running or expensive runs happen on Colab or a local machine; run at most
  one expensive experiment at a time unless the user explicitly allows more.
- Artifact store: **Hugging Face** (model `.pth`, `training.json`,
  `behavior.json`, `features.json`) plus a local `results/`-style folder. See
  [hugging_models.md](hugging_models.md) for the HF folder contract.

## Human Rehydration (daily)

1. Confirm Hugging Face auth (`huggingface-cli whoami`) before any remote
   restore, upload, or gated-model work.
2. `caffeinate -sdi` if running a long local job on macOS.

## Agent Rehydration

Use this after a restart, compaction, model/vendor swap, or long pause:

1. Read this file first.
2. Read your thread-specific contract: `<thread>-agent.md`
   (e.g. [maths-agent.md](maths-agent.md)).
3. Read your topic's results summary for the current best story.
4. Read [thor-document-rules.md](thor-document-rules.md) when editing docs or
   when the document structure is unfamiliar.
5. Read your topic's results synthesis when the active question matches it, or
   when the experiment agenda points to it.
6. Read [thor-glossary.md](thor-glossary.md) selectively when a project-specific
   term is unfamiliar. Do not load it fully by default.
7. Read your topic's experiment agenda (`<thread>-next-steps.md`).
8. Read the active study note in `study-<thread>/` for the current line, if any.
9. Consult the claim-evidence map only when you need to know which evidence
   supports which claim.
10. Consult the results ledger only when you need artifact coverage for a
    specific bundle or when appending a completed bundle.
11. Confirm Hugging Face auth before remote sync, upload, or gated-model work.
12. Run `git status`; assume unrelated dirty files belong to the human or
    another thread unless directly conflicting.

## Sources Of Truth

- Each `<thread>-agent.md` is the source of truth for that topic's concrete
  filenames, artifact locations, autonomy limits, and workflow.
- [thor-document-rules.md](thor-document-rules.md) is the source of truth for
  portable document roles, update order, terminology discipline, skeptic gates,
  and anti-patterns.
- [thor-glossary.md](thor-glossary.md) is the source of truth for canonical
  vocabulary across all topics.
- Study notes are the source of truth for individual experiment plans, run
  records, results, skeptic reviews, and immediate next read.
- Results ledgers are the source of truth for chronological bundle coverage.
- Claim-evidence maps are the source of truth for durable empirical claims.
- Conjecture files are speculative and must not be used as empirical evidence.

## Ownership

- Human-owned: this contract, the per-thread agent contracts, document rules,
  and the human conjecture files.
- Agent-owned: agent conjecture files and the study notes the agent writes,
  subject to human review.
- Coordinated topic docs: each topic's agenda, results summary, results
  synthesis, results ledger, and claim-evidence map.
- Project-wide cross-topic docs: [thor-document-rules.md](thor-document-rules.md)
  and [thor-glossary.md](thor-glossary.md).

"Shared" means shared across topics/threads, not merely co-edited by human and
agent.

## Document Contract

Follow [thor-document-rules.md](thor-document-rules.md) for portable document
roles, update rules, terminology practice, and doc anti-patterns. Keep this file
focused on project-specific operating workflow.

## Auto-Research Defaults

These are repo-wide defaults for every experiment thread. A thread contract
(`<thread>-agent.md`) may override the numeric settings; the rules themselves
apply to all threads.

| Setting | Default |
| --- | --- |
| Max autonomous hours | `8` |
| Depth budget (consecutive same-line studies without a conjecture update) | `4` |
| Agenda entry cap (active entries per thread agenda) | `8` |
| Skeptic model | any model, run in a separate thread; a different model or vendor is encouraged but not required |

- Depth budget: after the configured number of consecutive studies on the same
  line without a conjecture-file update, freezing that line is the default.
  Continuing past the budget requires a written justification in the new study
  note explaining what conjecture-level update the next run could still produce.
- Freeze classification: every freeze decision classifies itself as
  `question answered`, `question failure`, or `instrument failure` per
  [Document Rules](thor-document-rules.md#experiment-agenda).
- Outside-view sweep: after each freeze decision, and before any pivot to a new
  mechanism or method class, search the external literature for isomorphic
  problems and ask what their assays did differently. Record the read in the
  relevant study note or conjecture file. A freeze classified `instrument
  failure` makes this sweep mandatory before the thread's next experiment.
- At major consolidation checkpoints (a paper draft, a thread pivot, or an
  `instrument failure` freeze), write a short adversarial referee report on the
  thread's current story before choosing the next line.
- After each result lands, score the standing predictions in the thread's
  conjecture files as `confirmed`, `refuted`, or `untouched` per the study-note
  Prediction Scoring rule in Document Rules, then update conjectures and rerank
  the thread agenda — deleting completed entries rather than marking them
  complete in place.
- Positive controls and power honesty: every new assay follows the Positive
  Control and minimal-effect rules in
  [Document Rules](thor-document-rules.md#study-notes); a flat result without a
  passing positive control is `invalid`/`underpowered`, never `negative`.
- Skeptic gates: run the two
  [skeptic gates](thor-document-rules.md#skeptic-gates) on every non-trivial
  experiment — a pre-launch audit of the study-note plan and a post-result audit
  of the interpretation, each in a **separate thread** that rehydrates only from
  the docs (not the working thread's context). Record findings and resolution in
  the study note's Skeptic Review sections; resolve or explicitly overrule
  blocking concerns before passing the gate.

## Common Process Rules

- Long-running scripts should emit `=== ... ===` progress banners with an
  explicit `[n/N]` count so status checks show meaningful progress.
- Prefer shared constants and plan builders over ad hoc strings.
- Prefer public or accessible assets when expanding scope.
- Never rely on a previous thread's in-memory context for a later phase; write
  the needed context into the appropriate doc.
- Do not silently agree with human conjectures; they are expected to be
  incomplete or wrong. Push back and propose alternatives.

## Guardrails

- Trust but verify: the human reviews agent changes before publication.
- Do not upload or delete Hugging Face artifacts without confirming the folder
  contract in [hugging_models.md](hugging_models.md).

## Project-Specific Context

### What This Repo Is For

This library helps a researcher analyze and understand the algorithm implemented
by a low-loss transformer model. It identifies the useful token positions,
attention heads, and MLP neurons used in predictions; evaluates model behavior;
supports model-specific sub-task searches; stores useful facts as JSON; and lets
a researcher describe and evaluate an algorithm hypothesis against those facts.

Much of the library is generic. The real-world testbed is transformer models
trained on integer addition and subtraction (e.g. `133357+182243=+0315600`,
`123450-345670=-0123230`). Arithmetic sub-task searches include Base Add, Make
Sum 9, Make Carry, Base Subtract, Borrow One, TriCase, and others. See the
[README](../README.md).

### Foundational Papers

- Understanding Addition in Transformers: https://arxiv.org/abs/2310.13121
  (Paper 1). Model `add_d5_l1_h3_t30K` is very similar to the one in this paper.
- Understanding Addition and Subtraction in Transformers:
  https://arxiv.org/abs/2402.02619 (Paper 2). Source of this repo's
  [terminology](terminology.md) and sub-task definitions.

### Reference Docs (do not duplicate; link)

| Doc | Purpose |
| --- | --- |
| [terminology.md](terminology.md) | Long-form arithmetic term reference (canonical subset lives in the glossary) |
| [useful_tags.md](useful_tags.md) | Useful-node/position JSON tag scheme |
| [filter.md](filter.md) | Filtering useful nodes by tag prerequisites |
| [hugging_models.md](hugging_models.md) | Hugging Face artifact / folder contract |
| [mixed_model.md](mixed_model.md) | Mixed addition/subtraction model algorithm notes |
| [pca.md](pca.md) | PCA usage for interpreting useful nodes |
