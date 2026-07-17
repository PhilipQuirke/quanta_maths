# How to (re)generate the per-model mechanism maps and diagrams

This is the operating guide for the two-stage pipeline that turns a model's
verified Hugging Face map into (1) a **maximal per-model map JSON** and (2) an
**auto-generated mechanism doc** (two Mermaid diagrams + token layout + node
inventory + polysemantic-sharing summary). It is model-general — the same code
produces artifacts for any of the ~40-model zoo (add / sub / mixed), each with
its own token positions and node inventory.

The generated artifacts live in **`results/maps/`, not `docs/`** — they are
regenerable outputs, not hand-maintained docs. This file is the only thing to
edit by hand.

Interpretation and causal verdicts (which route delivers the carry, whether the
combiner is a STEP, whether selection is distributed, etc.) are **deliberately
not** in the generated doc: they live in the study notes and
[maths-claim-evidence.md](maths-claim-evidence.md) (CE20–CE25 for the mixed
model) and the human-owned algorithm notes in [mixed_model.md](mixed_model.md).
Per-task algorithmic meanings (`SA`/`MD`/`ND`, `ST`/`MT`/`NT`/`GT`,
`SC`/`MB`/`NB`, `OPR`/`SGN`/`SLT`, `STC`/`MTC`/`NTC`, `SV`/`MV`/`NV`) are defined
once in the [glossary](thor-glossary.md#project-terms).

## Pipeline

| Stage | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1. Capture the map | `scripts/gen_model_maps.py` | `PhilipQuirke/QuantaMaths_<model>`: `features.json` (Algo roles) + `behaviors.json` (Fail%/Impact/Attn/Probe) | `results/maps/<model>.json` |
| 2. Render the doc | `scripts/gen_mechanism_docs.py` | `results/maps/<model>.json` | `results/maps/<model>_mechanism.md` |

```bash
# Stage 1 — write the maximal map JSON(s). Needs HuggingFace access; map-only
# (no model forward pass), so it is cheap to batch across the zoo.
PYTHONPATH=. python scripts/gen_model_maps.py [model_name ...]

# Stage 2 — generate the Mermaid mechanism doc from the JSON (offline once the
# JSON exists; captures the JSON on the fly if it is missing).
PYTHONPATH=. python scripts/gen_mechanism_docs.py [model_name ...]
```

Both default to the full analysable set
(`quanta_maths.maths_hf_update.ordered_analysis_models()` — the ~33 models whose
repo has both `behaviors.json` and `features.json`, ordered add → sub → mix,
small → large) if no model names are given. View the Mermaid diagrams with
GitHub's renderer or the VS Code Mermaid preview.

## Library entry points (`quanta_maths/maths_diagram.py`)

The JSON is the single source of truth; the doc is a pure function of it.

- `build_model_map(model_name, cfg, maths_nodes, behav_nodes) -> dict` — pure /
  offline builder of the maximal map dict from already-loaded node lists
  (unit-tested with fixtures; no HF, no forward pass).
- `capture_model_map(model_name, hf_repo=..., cfg=None) -> dict` — download the
  verified HF map and return the maximal map dict (map-only). This is the shared
  builder that `scripts/mixed_map.py` reuses (adding the per-class positive
  control it computes from a model load).
- `build_mechanism_markdown(model_map) -> str` — render the doc from a map dict
  (offline, deterministic). `build_mechanism_markdown_for_model(name)` is the
  capture-then-render convenience.
- `algo_task(body)` — parse the task out of an `Algo:` tag body
  (`A5.ND.A5 -> ND`).

## Maximal map JSON schema (`results/maps/<model>.json`)

```
{
  "model": "ins1_mix_d6_l3_h4_t40K_s372001",
  "config": {n_digits, n_layers, n_heads, n_ctx, perc_add, perc_sub},
  "positive_control_accuracy": {"ADD": .., "SUB": .., "NEG": ..},  # study scripts only
  "n_map_nodes": 98,
  "roles": {"<TASK>": [{loc, position, layer, is_head, num, algo,
                        impact[], fail[], attn[]}, ...]},
  "combiners": [{loc, position, layer, produces_A, fail[], impact[]}],
  "positions": {OPR, eq, SGN, answer_digits[], first_layer, last_layer}
}
```

## Worked example

`ins1_mix_d6_l3_h4_t40K_s372001` (6-digit add/sub, 3 layers, 4 heads,
initialised from a 6-digit addition model):
[`results/maps/ins1_mix_d6_l3_h4_t40K_s372001.json`](../results/maps/ins1_mix_d6_l3_h4_t40K_s372001.json)
and its rendered doc
[`results/maps/ins1_mix_d6_l3_h4_t40K_s372001_mechanism.md`](../results/maps/ins1_mix_d6_l3_h4_t40K_s372001_mechanism.md).

## Known gap (map completeness)

The zoo-wide maps (33 models on HF as of 2026-07-16) are richer than the first
cut — the combiner `STC`/`MTC`/`NTC` `Algo` tags and the `Probe:*` behaviour tags
are now present — but not yet "maximal" in every field. Tracked as the
map-completeness item in [maths-next-steps.md](maths-next-steps.md):

- the map-absent cascade-resolver `SV`/`MV`/`NV` role tags (so the logical diagram
  still draws a generic "cascade resolver" box);
- model provenance `init_from` (e.g. `ins1_*` initialised from a d6 addition model);
- per-class positive-control accuracy for non-mixed models;
- **sparse base-digit tagging on some large models** — e.g. `add_d14` has no `SA`
  tag, so its base-writer box shows "none" and the op class is inferred from the
  model name. Faithful to the map; fix upstream in the HF `features.json`.
