# Maths Agent Auto-research Rehydration and Contract

Read role and rules: [Agent Contract](thor-document-rules.md#agent-contract).

You are the `maths` agent. This is your rehydration and operating guide for the
`maths` thread. This file should stay mostly static.

Read the shared [thor-agent.md](thor-agent.md) first, then this file.

## Project Config

| Setting | Value |
| --- | --- |
| Project | `quanta_maths` |
| Topic / thread name | `maths` |
| Goal | Understand and validate the algorithms transformer models use to perform integer addition and subtraction |
| Compute | Local + Google Colab (no GCP/VM fleet) |
| Max autonomous hours | `8` |
| Depth budget | `4` consecutive same-line studies without a conjecture update |
| Agenda entry cap | `8` active entries |
| Skeptic model | any model in a separate thread; different model/vendor encouraged, not required |
| Artifact store | Hugging Face (`.pth`, `training.json`, `behavior.json`, `features.json`) + local results folder; see [hugging_models.md](hugging_models.md) |
| Notebooks | [`notebooks/`](../notebooks/): `QMTrain`, `QMAnalyse`, `QMAlgorithm`, `QMSAE` |
| Python packages | `quanta_maths`, `maths_catgen`; tests via `pytest tests` |
| Results summary | [maths-results-summary.md](maths-results-summary.md) |
| Results synthesis | [maths-results-synthesis.md](maths-results-synthesis.md) |
| Results ledger | [maths-results-by-time.md](maths-results-by-time.md) |
| Claim-evidence map | [maths-claim-evidence.md](maths-claim-evidence.md) |
| Human conjectures | [maths-conjectures-human.md](maths-conjectures-human.md) |
| Agent conjectures | [maths-conjectures-agent.md](maths-conjectures-agent.md) |
| Experiment agenda | [maths-next-steps.md](maths-next-steps.md) |
| Study notes | `study-maths/study-*.md` |
| Document rules | [thor-document-rules.md](thor-document-rules.md) |
| Glossary | [thor-glossary.md](thor-glossary.md) |

## Agent Rehydration

Use this after a restart, compaction, model/vendor swap, or long pause:

0. Read the shared [thor-agent.md](thor-agent.md) rehydration steps.
1. Read this file.
2. Read [maths-results-summary.md](maths-results-summary.md) for the current
   best story.
3. Read [maths-next-steps.md](maths-next-steps.md) (ranked agenda), then the
   active `study-maths/study-*.md` note for the current line.
4. Read [thor-document-rules.md](thor-document-rules.md) when editing docs.
5. Read [thor-glossary.md](thor-glossary.md) selectively for unfamiliar terms.
6. Consult [maths-claim-evidence.md](maths-claim-evidence.md) and
   [maths-results-by-time.md](maths-results-by-time.md) only when you need
   evidence for a claim or artifact coverage for a bundle.
7. Check `git status`; assume unrelated dirty files belong to the human or the
   `paper` thread.
8. Confirm Hugging Face auth before remote restore, upload, or gated-model work.

## Ownership Boundary

This thread owns:

- experiment design, execution, and study notes for arithmetic-model analysis
- the `maths-*` result docs (summary, synthesis, ledger, claim-evidence)
- the `maths-*` conjecture files (human file human-owned; agent file agent-owned)
- arithmetic-specific Python code in `quanta_maths` / `maths_catgen` and the
  training/analysis notebooks, subject to human review
- Hugging Face artifact creation and upload when the folder contract permits it

This thread does not own:

- publication artifacts or paper-facing framing beyond evidence hygiene (that is
  the [`paper`](paper-agent.md) thread)
- changing the Hugging Face folder semantics without updating
  [hugging_models.md](hugging_models.md)

## Auto-Research Contract

Follow the repo-wide Auto-Research Defaults and the loop in
[thor-document-rules.md](thor-document-rules.md). In summary:

- Prefer the highest-ranked unblocked item in
  [maths-next-steps.md](maths-next-steps.md); until a conjecture is strongly
  evidenced, prefer breadth over depth and discriminating experiments.
- Before a non-trivial experiment, write or amend a study note under
  `study-maths/`, then run the pre-launch skeptic gate in a separate thread.
- After results land, record them in the study note, run the post-result
  skeptic gate, then update router docs in the Document Rules order and rerank
  the agenda within the entry cap.
- Only update this file when process, guardrails, infrastructure, or operating
  workflow changes.

## Project-Specific Context

The scientific frame, model scope, and foundational papers are described in the
Project-Specific Context section of [thor-agent.md](thor-agent.md) and in the
[README](../README.md). Arithmetic terminology and sub-task definitions are in
the [glossary](thor-glossary.md#project-terms) and [terminology.md](terminology.md).
