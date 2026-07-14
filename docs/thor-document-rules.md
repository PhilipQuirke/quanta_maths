# Document Rules (thor-document-rules.md)

Read role and rules: [Document Rules](#document-rules).

Project-wide. One per repo, not one per topic. This is the portable document
contract for research repos that use the Thor doc structure. It defines each
document's job, the update workflow, and the anti-drift rules.

Project-specific or topic-specific docs should link to the relevant role here
instead of repeating these rules.

## Role Catalog

These are document roles. A repo may have multiple topics, multiple summaries,
or prefixed filenames. Put the concrete file mapping, including the default
study-note location, in the topic-specific agent contract.

| Role | Purpose | Scope |
| --- | --- | --- |
| Agent Contract | Topic-specific operating contract and rehydration entry point. | per-topic |
| Document Rules | This file: document roles, update workflow, terminology discipline, and anti-patterns. | project-wide |
| Glossary | Canonical project vocabulary. | project-wide |
| Results Summary | Compact current-state synthesis. | per-topic |
| Results Synthesis | Mid-level empirical synthesis for a primary results theme. | per-topic or topic-index |
| Results Ledger | Append-only chronological record of completed result bundles. | per-topic |
| Claim-Evidence Map | Durable empirical claims and the evidence/caveats attached to them. | per-topic |
| Experiment Agenda | Ranked, current-facing planned next experiments. | per-topic |
| Conjectures | Human-owned or agent-owned speculative beliefs and predictions. | per-topic, owner-specific |
| Study Notes | Individual study plans, methods, artifacts, results, interpretation, skeptic gates, and immediate next read. | per-study |

## Terminology Discipline

- Use stable project vocabulary from `thor-glossary.md`.
- Define terms that recur, affect interpretation, appear in artifacts, or could drift across threads before spreading them through docs, code, plots, or artifact names.
- In reader-facing docs, link project-specific glossary terms on first important use and use plain English when a term would make prose opaque.
- Do not create near-synonyms for existing terms. If a synonym is unavoidable, add it as an alias to the canonical glossary term.
- Terms that shape process or verdicts must be defined. At minimum, keep definitions for: `assay`, `study`, `skeptic gate`, `freeze decision`, `agenda`, `streak`, `empirical doc`, `conjecture doc`, `claim`, and `thread`.
- Do not let agent-defined jargon drift without a glossary entry.

## General Rules

- Keep empirical docs separate from conjecture docs.
- Prefer current-state language in synthesis docs and chronological language only in ledgers or study notes.
- Keep claims fewer than experiments.
- Link out instead of duplicating long evidence, artifact trails, or update rules.
- Start each concrete doc with a one-line role pointer to this file:
  `Read role and rules: [Role](thor-document-rules.md#role-anchor).`
- If a document's purpose is unclear, add a one-line role pointer to this file rather than copying the full rule block.
- The docs must be sufficient for a fresh agent or skeptic to rehydrate without relying on a previous thread's in-memory context.

## Thread Isolation And Phase Boundaries

Thor uses separate work phases to prevent conjectures, private reasoning, or stale in-memory context from bleeding into empirical records.

Required phase order for non-trivial experiments:

1. **Planning phase**: the working thread rehydrates from docs, chooses the highest-ranked unblocked agenda item, and writes or amends the study note pre-run sections.
2. **Pre-launch skeptic phase**: a separate skeptic thread rehydrates only from version-controlled docs and audits the study plan. It must not inherit the working thread's live or compacted conversation context.
3. **Execution phase**: the working thread resolves or explicitly overrules blocking skeptic concerns in the study note, then runs the experiment.
4. **Result-recording phase**: the working thread syncs artifacts and completes the study note's run record, results, interpretation, limitations, and prediction scoring.
5. **Post-result skeptic phase**: a separate skeptic thread rehydrates only from version-controlled docs and audits the interpretation, verdict, and prediction scoring before conjectures update or the agenda reranks.
6. **Router-doc update phase**: after the post-result skeptic gate passes, update ledger, claim-evidence, results summary, results synthesis, conjectures, and agenda as applicable.
7. **Cross-document audit phase**: compare changed docs for duplicated detail, conjecture bleed, stale agenda entries, and terminology drift before stopping.

Prediction scoring belongs in the study note as part of the empirical record. It does not by itself update conjectures. Conjecture files, claim files, and agenda files are updated only after the post-result skeptic gate passes.

## Evidence And Verdict Rules

- Every new assay includes a positive control: a known-positive manipulation that demonstrates the metric can move under this harness.
- A flat result whose positive control also fails is `invalid` or `underpowered`, never `negative`.
- Every study design states the minimal effect of interest and whether the stated sample sizes can detect it. If they cannot, say so before the run.
- A `negative` verdict requires both a passing positive control and sample sizes that can detect the stated minimal effect of interest; otherwise label the result `ambiguous`, `underpowered`, or `invalid`.
- Every completed study scores the standing conjecture-file predictions it bears on as `confirmed`, `refuted`, or `untouched`. A run that scores no predictions says so explicitly.
- A streak of refuted or untouched predictions is an early signal that the conceptual frame is stale. Record the streak in the relevant conjecture file only after the post-result skeptic gate passes.
- Every freeze decision classifies itself as `question answered`, `question failure`, or `instrument failure`.
- `question answered` means the study line has answered the intended question well enough under its stated scope.
- `question failure` means the question was ill-posed or unanswerable as stated.
- `instrument failure` means the assay class, unit of intervention, metric, data, or harness was the limiting factor. It triggers the outside-view sweep defined in the agent contract.

## Skeptic Gates

Role: independent adversarial review at two pre-registered points in every non-trivial experiment.

Rules:

- The skeptic runs in a separate thread that rehydrates only from the version-controlled docs it needs: the study note, conjecture files, document rules, glossary, relevant evidence docs, and relevant artifacts.
- The skeptic must not inherit the working thread's live or compacted conversation context.
- This keeps review independent of the author's framing and continuously tests whether docs are self-sufficient for rehydration.
- A different model or vendor is encouraged but not required; a separate thread is required.
- Gate 1, pre-launch: before an experiment launches, the skeptic audits the study-note pre-run sections. Is the positive control real? Is the minimal-effect and power statement honest? Does the success condition discriminate live conjectures rather than admit a gameable pass? Are controls adequate?
- Gate 2, post-result: after results land and before conjecture updates or agenda reranking, the skeptic audits interpretation, verdict, and prediction scoring. Is the verdict over-claimed? Is `invalid` or `ambiguous` dodging a supported verdict? Is prediction scoring generous?
- Blocking concerns must be resolved before passing the gate, or explicitly overruled with a one-line written justification in the study note.
- The working thread owns the final call; unresolved disagreement is recorded, not hidden.
- Record each skeptic pass and its resolution in the study note's Skeptic Review sections. Do not scatter skeptic records across other docs.

## Agent Contract

Role: topic-specific operating contract for autonomous agents.

Rules:

- Keep project configuration, rehydration steps, infrastructure workflow, autonomy limits, document mapping, thread names, and cleanup rules here.
- Link to this document for portable document structure and terminology rules.
- Do not use the agent contract as the results summary, results ledger, or claim map.
- Update when process, guardrails, infrastructure, autonomy limits, or operating workflow changes.

## Results Summary

Role: compact current-state synthesis.

Use it to orient a human or agent quickly to:

- the current best empirical story
- the most important and insightful claims
- the strongest live caveats
- weak, stale, or mostly exhausted lines
- open questions that should shape near-term experiment choice

Rules:

- Include an executive summary.
- Cover only the current top claims, not all claims and not all experiments.
- Prefer current-state language over historical process language.
- Remove stale rejected frames instead of preserving them as findings.
- Link outward to the claim-evidence map, results synthesis, results ledger, and agenda for detail.
- Update only when a result changes the current story.
- Use at most a few anchor numbers, only when they change the present-state read.
- Prefer claim labels, role labels, and links over experiment, bundle, or artifact lists.
- If a paragraph names several runs, bundles, cells, metrics, or controls, move the detail to results synthesis, claim-evidence, a study note, or the ledger.

## Results Synthesis

Role: mid-level empirical synthesis for a primary results theme.

Use it to answer:

- which findings in the primary empirical line currently matter
- which figures are decision-relevant rather than merely recorded
- which caveats should constrain the next experiment
- where the story is strong, weak, or still ambiguous

Rules:

- Organize by empirical question, not by experiment chronology.
- Keep only figures that change interpretation or priority.
- Include qualitative reads for table rows, such as `strong`, `modest`, `negative`, `control-sensitive`, `method caveat`, or `decision-critical`.
- Do not duplicate the ranked experiment agenda or the claim-evidence map.
- Update when results change the synthesis, even if they do not yet justify a durable claim.
- If a repo accumulates multiple large empirical themes, keep the results synthesis doc as the integrative synthesis or index and move detailed theme-specific material into theme-specific synthesis docs.
- This is the normal home for detailed mechanism interpretation that would overload the results summary.

## Results Ledger

Role: append-only chronological record of completed result bundles.

Rules:

- Add new completed bundles to the end.
- Preserve experiment order.
- Record bundle coverage, artifacts, upload links, and caveats, not the current best story.
- Do not turn this file into a claim-evidence map.
- Do not rewrite old entries merely because later terminology improved unless the old entry is misleading or blocks navigation.

## Claim-Evidence Map

Role: durable empirical claims and evidence.

Rules:

- Add a new claim only when an experiment produces a genuinely new useful insight.
- If a result strengthens, weakens, narrows, or broadens an existing claim, update that claim instead of adding a near-duplicate.
- Keep fewer claims than experiments.
- Do not organize this file by experiment chronology.
- Link all relevant bundles and artifacts under the claim without emphasizing discovery sequence unless the sequence itself matters.
- Keep conjecture arguments in the conjecture files.

Suggested confidence labels:

- **High**: repeated across models or methods, with a direct artifact trail.
- **Medium**: supported, but dependent on interpretation, a limited assay, or partial replication.
- **Low**: suggestive, but not stable enough to lean on.

## Experiment Agenda

Role: ranked planned next experiments agenda.

Use it to answer:

- what experiment should run next
- which belief that experiment would update
- what result would change priorities
- which tempting lines are deliberately paused

Rules:

- Keep it ranked and current-facing.
- Do not preserve completed experiment plans here as history.
- The post-run doc update deletes completed entries from the agenda instead of marking them `completed` in place; completed trails belong in study notes and the results ledger.
- Keep the agenda within the entry cap defined in the agent contract. Exceeding the cap is itself a doc bug to fix before the next run.
- Every freeze decision must classify itself as `question answered`, `question failure`, or `instrument failure`.
- Move completed artifact trails to the results ledger.
- Keep durable claim updates in the claim-evidence map.
- Keep high-level synthesis in the results summary and detailed empirical synthesis in results synthesis.
- Prefer decisive discriminators over incremental cleanup.
- After each meaningful result, rerank this file rather than appending a historical note.

## Study Notes

Role: individual study plans, methods, artifacts, results, skeptic gates, and interpretation.

A study note should exist before any non-trivial experiment runs. It is both a pre-registration-lite plan and the final experiment record.

Required pre-run sections:

- **Question**: the specific empirical question.
- **Motivation**: why this question matters now.
- **Hypothesis / Competing Reads**: what live beliefs the result can distinguish, written neutrally rather than as advocacy.
- **Design**: model, data, intervention, controls, metrics, sample sizes, and scope. State the minimal effect of interest and whether the stated sample sizes can detect it; if they cannot, say so before the run.
- **Positive Control**: a known-positive manipulation included in the assay that demonstrates the metric can move under this harness.
- **Success / Failure / Ambiguous Conditions**: what result would count as positive, negative, inconclusive, or invalid before seeing the data.
- **Skeptic Review (pre-launch)**: the pre-launch skeptic-gate findings and their resolution. Written before launch.
- **Decision Impact**: what each outcome would change in results summary, results synthesis, claim-evidence map, conjectures, or agenda.
- **Risks And Confounds**: likely failure modes, missing controls, implementation risks, and interpretation traps.
- **Expected Artifacts**: planned output files, plots, tables, logs, and where they should land.

Required post-run sections:

- **Executive Summary**: short summary at the top of the study note.
- **Run Record**: command, environment, local/VM status, dates, artifact paths, upload links, commit or hash if relevant.
- **Results**: key quantitative and qualitative results.
- **Interpretation**: actual read against the pre-stated success, failure, ambiguous, and invalid conditions.
- **Prediction Scoring**: score each standing conjecture-file prediction this result bears on as `confirmed`, `refuted`, or `untouched`. A run that scores no predictions should say so explicitly.
- **Skeptic Review (post-result)**: the post-result skeptic-gate findings and their resolution. Written before conjectures update and the agenda reranks.
- **Limitations**: what the result does not show.
- **Doc Updates**: which docs were updated or intentionally left unchanged.
- **Next Read**: what should happen next, including whether the line is frozen, reopened, needs replication, or needs a different assay.

Study-note rules:

- Write the pre-run sections before launching the experiment.
- Create new study notes in the topic-specific study-note path from the agent contract.
- Do not move the goalposts after seeing results; if criteria change, add an explicit amendment with timing.
- Prefer `positive`, `negative`, `ambiguous`, `underpowered`, and `invalid` over vague `success`.
- A negative result is useful when the study note states what it rules out.
- Keep the executive summary short and refresh it after results are in.
- Preserve detailed methods, artifacts, raw numerical tables, run records, and implementation notes in the body.
- Keep raw logs and exhaustive artifact detail out of summary docs; link to them from the study note.
- Move durable claim changes to the claim-evidence map, bundle coverage to the results ledger, and priority changes to the agenda.

## Conjectures

Role: explicitly speculative beliefs, priors, and predictions.

Rules:

- Keep conjecture docs separate from empirical docs.
- Update conjectures only after empirical findings are recorded and the post-result skeptic gate has passed.
- State uncertainty plainly.
- Use conjectures to generate experiment ideas, not to backfill claims into evidence docs.
- Human-owned and agent-owned conjectures can disagree; that tension is useful and can influence agenda rankings.
- Do not list metrics, artifact filenames, bundle histories, or long experiment chains unless a specific cell anchors a belief, prediction, or alternative.
- Use evidence documents as backlinks, not copied summaries.
- Phrase content as beliefs, predictions, falsifiers, and alternatives. If the prose mainly says “we observed”, move it to an empirical doc.
- If a conjecture needs detailed support, link to results synthesis or claim-evidence rather than restating the result.

## Glossary

Role: canonical project vocabulary.

Rules:

- Define terms that recur, affect interpretation, appear in artifacts, or could drift across docs.
- Prefer stable definitions over experiment chronology, next-step planning, or bundle summaries.
- Do not create near-synonyms for existing terms.
- If a synonym is unavoidable for readability, link it to the canonical term or add it as an alias.
- Multi-topic projects still use one project-wide glossary to prevent rival meanings.

## Naming Convention

- Framework docs use `<prefix>-<role>.md`.
- Single-topic project and templates: prefix is `thor`.
- Multi-topic project: copy per-topic docs once per topic and replace `thor` with the topic name, e.g. `diversity-agent.md`, `ensemble-next-steps.md`.
- Project-wide docs keep the `thor-` prefix in every repo: `thor-document-rules.md` and `thor-glossary.md`.

## When To Update What

| Change | Update |
| --- | --- |
| New completed run or artifact bundle | Results ledger |
| New or changed durable empirical claim | Claim-evidence map |
| Changed current best story | Results summary |
| Changed primary empirical interpretation | Results synthesis |
| Changed next experiment priority | Experiment agenda |
| New recurring term | Glossary |
| Speculative belief update | Conjectures |
| Skeptic pass at a gate | Study note, Skeptic Review section |
| Experiment plan, methods, artifacts, or detailed result | Study note |
| Agent operating workflow or infrastructure contract | Agent contract |
| Portable doc structure or doc roles | Document rules |

## Post-Update Cross-Document Audit

After changing project docs, compare the changed documents against each other before stopping.

Ask:

- Is this paragraph mainly speculative? It belongs in conjectures.
- Is this paragraph empirical detail from one experiment? It belongs in a study note.
- Is this paragraph bundle chronology or artifact coverage? It belongs in the results ledger.
- Is this paragraph assigning evidence support to a durable claim? It belongs in the claim-evidence map.
- Is this paragraph detailed mechanism interpretation or comparison? It belongs in results synthesis.
- Is this paragraph the compact current-state answer? It belongs in results summary.
- Is this paragraph about future priority? It belongs in the agenda.

Then remove repetition:

- If the same result appears in multiple reader-facing docs, keep detail only in the most specific useful document.
- Higher-level docs should keep conclusions and links downward.
- Lower-level docs should keep evidence and avoid repeating project-level framing.
- If two documents contain near-identical paragraphs, delete one and replace it with a pointer.

Drift smells:

- A conjecture section reads like “we observed...” instead of “we expect...”
- A synthesis section repeats the claim index rather than interpreting patterns.
- A study section tries to settle the whole project instead of its own question.
- An agenda section preserves completed plans as history.
- A glossary term appears in artifacts before being defined.

## Anti-Patterns

- Do not turn results summary into a result ledger.
- Do not turn results synthesis into a chronological bundle list.
- Do not put experiment chronology in the claim-evidence map.
- Do not put speculative conjecture in empirical docs.
- Do not put results-summary or results-synthesis paragraphs in conjecture docs.
- Do not promote the latest mechanism chain into the results summary unless it changes the compact current-state story.
- Do not let the agenda preserve completed plans as history.
- Do not promote a flat result to `negative` without a passing positive control and adequate detectable effect size.
- Do not run the skeptic in the working thread's own context.
- Do not score a prediction in a way that updates conjectures before the post-result skeptic gate passes.
- Do not duplicate long document-purpose sections across docs; link to this file.
- Do not create method-specific top-level synthesis docs unless results synthesis has become too large to serve as the primary synthesis.
- Do not create a glossary per topic; there is exactly one `thor-glossary.md` per project.
