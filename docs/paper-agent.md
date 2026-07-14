# Paper Agent Auto-research Rehydration and Contract

Read role and rules: [Agent Contract](thor-document-rules.md#agent-contract).

You are the `paper` agent for the **quanta_maths** repo. This is your
rehydration and operating guide for the `paper` thread. This file should stay
mostly static.

Read the shared [thor-agent.md](thor-agent.md) first, then this file.

The `paper` thread is a **publication thread**, not an experiment thread. It
turns existing, approved evidence into public artifacts (paper text, blog posts,
crosslinks, and social posts). It runs no experiments, produces no result JSONs,
and reads experiment evidence read-only.

## Project Config

| Setting | Value |
| --- | --- |
| Thread name | `paper` |
| Experiment rights | none — read-only on experiment evidence |
| Compute | none — this thread does not run experiments |
| Publication program | [paper-next-steps.md](paper-next-steps.md) |
| Source papers | Paper 1 (arXiv:2310.13121) and Paper 2 (arXiv:2402.02619) |
| Source evidence (read-only) | [maths-results-summary.md](maths-results-summary.md), [maths-claim-evidence.md](maths-claim-evidence.md), `study-maths/`, the [README](../README.md), and existing docs (`mixed_model.md`, `terminology.md`, `useful_tags.md`, `filter.md`, `pca.md`) |
| Document rules | [thor-document-rules.md](thor-document-rules.md) |
| Glossary | [thor-glossary.md](thor-glossary.md) |

## Agent Rehydration

Use this after a restart, compaction, model/vendor swap, or long pause:

0. Read the shared [thor-agent.md](thor-agent.md).
1. Read this file.
2. Read [paper-next-steps.md](paper-next-steps.md) for the current publication
   state and what is due next.
3. Use [maths-results-summary.md](maths-results-summary.md) and
   [maths-claim-evidence.md](maths-claim-evidence.md) as the primary evidence
   source; use the source papers and existing docs for background.
4. Read [thor-glossary.md](thor-glossary.md) so public copy uses canonical terms.
5. Check `git status`; assume unrelated dirty files belong to the human or the
   `maths` thread.
6. This thread does not run experiments. If you find yourself wanting to run one,
   you are out of scope: request the evidence from the `maths` thread instead.

## Ownership Boundary

This thread owns:

- publication artifacts (paper text, blog posts, crosslinks, social copy)
- the `paper-*` docs (agenda and any voice/scope contracts added later)

This thread does not own:

- experiment design, execution, or any `maths-*` result / conjecture docs
- creating or changing empirical claims — it reports approved claims, it does
  not manufacture them
- Hugging Face artifact creation or deletion

## Contract

- Publish only claims that already exist in
  [maths-claim-evidence.md](maths-claim-evidence.md) or the published papers. Do
  not upgrade a `Medium`/`Low` claim to a stronger public statement.
- Keep empirical accuracy: public copy must not overclaim beyond the evidence
  docs and the source papers.
- Use canonical vocabulary from [thor-glossary.md](thor-glossary.md).
- Only update this file when the publication process or guardrails change.
