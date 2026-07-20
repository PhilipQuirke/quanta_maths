"""Exploration deliverable: render the 10 candidate visualizations
(``quanta_maths.maths_viz``) for one addition, one subtraction, and one mixed
model, and assemble a scored report at ``results/study-visualizations/report.md``.

Each visualization combines several fact types from the full analysis JSONs
(behaviors.json + features.json) into one view. The per-visualization usefulness
score and pros/cons below are the author's judgement (kept here as data so the
report regenerates verbatim); the diagrams themselves are auto-rendered from HF.

Run:  PYTHONPATH=. python scripts/gen_visualizations.py [add_model sub_model mix_model]
"""
from __future__ import annotations

import os
import sys

from quanta_maths.maths_viz import VISUALIZATIONS, load_viz_model

OUT_DIR = "results/study-visualizations"

DEFAULT_MODELS = [
    ("addition", "add_d6_l2_h3_t20K_s572091"),
    ("subtraction", "sub_d6_l2_h3_t30K_s372001"),
    ("mixed", "ins1_mix_d6_l3_h4_t40K_s372001"),
]

# Author scoring (1-5 usefulness) + pros/cons per visualization.
SCORING = {
    "V1": {"score": 5.0, "pros": [
        "Single densest view: every useful node with its role(s) and ablation "
        "importance on the real position x layer grid -- a strict upgrade of the "
        "paper's per-node subtask table (Tab. MathsPurposePerNode) with Fail% "
        "shading fused in.",
        "Token semantics (D/OP/=/A) in the header make the geography readable.",
        "Model-agnostic; scales to any size/op."],
        "cons": [
        "Wide for large-digit models (many columns).",
        "Fail% shade is 5-bucket, not exact; polysemantic cells get long."],
        "verdict": "Adopt as the headline map in the per-model HF doc."},
    "V2": {"score": 4.5, "pros": [
        "Links role -> which answer digits it supports -> importance -> "
        "redundancy (node count) in one compact matrix.",
        "Immediately shows the 'each digit served by ~2 base nodes' redundancy and "
        "the question-tail ST/MT fan-out across all digits."],
        "cons": [
        "Aggregates away node identity (pair with V1).",
        "Sign column can be sparse."],
        "verdict": "Adopt; strong companion to V1."},
    "V3": {"score": 4.0, "pros": [
        "Makes the 'each answer digit fetches its own two operands' routing "
        "explicit -- the attention story the paper describes in prose.",
        "Operand-matched fetches are starred, so the diagonal fetch pattern pops."],
        "cons": [
        "Busy for wide models even at top-2 edges/answer.",
        "Aggregates multiple heads per answer position."],
        "verdict": "Adopt for small/mid models; cap edges for large ones."},
    "V4": {"score": 4.0, "pros": [
        "Concrete causal chain for one answer digit with the actual node locs at "
        "each stage AND the delivery route -- bridges the map to the algorithm.",
        "Directly mirrors the paper's Hypothesis-3 pseudo-code as a picture."],
        "cons": [
        "One digit at a time; picking the 'best-covered' digit hides per-digit "
        "variation.",
        "The resolver/SV stage is schematic (SV rarely tagged in the map)."],
        "verdict": "Adopt as an illustrative inset, not the main map."},
    "V5": {"score": 4.5, "pros": [
        "Smallest high-signal view: shows the compute pipeline DEPTH (L0 fetch/"
        "compute -> mid resolve/select -> last combine) with role groups and "
        "importance per layer.",
        "Great orientation header before the detailed grids."],
        "cons": ["Coarse; hides positional structure."],
        "verdict": "Adopt as the doc's opening summary."},
    "V6": {"score": 4.0, "pros": [
        "Directly visualizes polysemantic reuse / the shared engine (SA&MD&ND, "
        "ST&MT, OPR&SGN co-location) -- the paper's central mixed-model claim.",
        "Quantifies sharing, not just asserts it."],
        "cons": [
        "Near-empty and low-value for addition-only models (little sharing).",
        "Symmetric matrix wastes half the grid."],
        "verdict": "Adopt for mixed/sub; suppress for pure addition."},
    "V7": {"score": 3.5, "pros": [
        "One glance shows carry/borrow delivery route differing by class "
        "(ADD residual vs SUB/NEG residual+attention) -- a headline mixed finding.",
        "Tiny and unambiguous."],
        "cons": [
        "Only 3 facts; not much beyond a sentence.",
        "Absent unless DELIVERY probes were run."],
        "verdict": "Include for mixed models; skip when no DELIVERY tags."},
    "V8": {"score": 4.0, "pros": [
        "Turns the paper's 'no single node necessary; necessity is class-level' "
        "claim into a per-role redundancy+importance readout.",
        "The Fail% spread bar exposes a few high-Fail base nodes vs many low-Fail "
        "carry/control nodes."],
        "cons": ["Distribution as min/med/max loses shape."],
        "verdict": "Adopt; cheap and evidentially on-message."},
    "V9": {"score": 2.5, "pros": [
        "Ties the LINXFER probe (operand linear-decodability) to the concrete "
        "fetch node -- a representation<->location link.",
        "Honest N/A signals coverage gaps (e.g. subtraction maps)."],
        "cons": [
        "Sparse or absent in most maps; add/mix only.",
        "Narrow: one probe family."],
        "verdict": "Keep as an optional appendix row; low priority."},
    "V10": {"score": 4.0, "pros": [
        "Shows where each operation's capacity sits across output digits and the "
        "cross-class overlap (polysemanty) using Math.* tags -- complements V6 on "
        "a digit axis.",
        "Reveals the 'more nodes on middle/high digits' capacity gradient."],
        "cons": [
        "Degenerates to one row for addition-only models.",
        "Counts, not causal weights."],
        "verdict": "Adopt for mixed; informative but secondary for single-op."},
}


def main(models):
    os.makedirs(OUT_DIR, exist_ok=True)
    loaded = []
    for kind, name in models:
        print(f"  loading {kind}: {name}")
        loaded.append((kind, name, load_viz_model(name)))

    out = []
    out.append("# Candidate node-fact visualizations - scored report\n")
    out.append(
        "Ten candidate visualizations for the auto-generated per-model HF doc, "
        "each **combining several fact types in one view** to expose the linkage "
        "between a node's importance, role, output digits, routing and delivery. "
        "Built by `quanta_maths.maths_viz` from the full `behaviors.json` + "
        "`features.json` (all node facts: `Fail%`, `Impact`, `Attn`, "
        "`Math.Add/Sub/Neg`, `Algo` roles, `Probe:{LINXFER,CARRYLAYER,CARRYDEFER,"
        "DELIVERY}`). Regenerate: `PYTHONPATH=. python scripts/gen_visualizations.py`.\n")
    out.append("**Models** (all 6-digit, accurate, identical token layout for "
               "comparison):\n")
    for kind, name, _ in loaded:
        out.append(f"- {kind}: `{name}`")
    out.append("")
    out.append("**Usefulness scale**: 5 = adopt as a default panel; 4 = adopt; "
               "3 = situational; 2 = optional/appendix; 1 = drop. Scores are the "
               "author's judgement; diagrams are auto-rendered.\n")

    # scoring summary
    out.append("## Scoring summary\n")
    out.append("| ID | Visualization | Facts linked | Score | Verdict |")
    out.append("|---|---|---|---|---|")
    for vid, name, _fn, links in VISUALIZATIONS:
        s = SCORING[vid]
        out.append(f"| {vid} | {name} | {links} | **{s['score']}** | {s['verdict']} |")
    out.append("")
    avg = sum(s["score"] for s in SCORING.values()) / len(SCORING)
    out.append(f"Mean usefulness across the ten: **{avg:.1f}/5**.\n")

    # per-visualization sections
    for vid, name, fn, links in VISUALIZATIONS:
        s = SCORING[vid]
        out.append(f"\n---\n\n## {vid}. {name}  (score {s['score']}/5)\n")
        out.append(f"*Facts linked: {links}.*\n")
        for kind, mname, model in loaded:
            out.append(f"\n### {vid} - {kind} (`{mname}`)\n")
            out.append(fn(model))
            out.append("")
        out.append("\n**Pros**")
        for p in s["pros"]:
            out.append(f"- {p}")
        out.append("\n**Cons**")
        for c in s["cons"]:
            out.append(f"- {c}")
        out.append(f"\n**Verdict:** {s['verdict']}\n")

    # recommendations
    out.append("\n---\n\n## Recommendation for the per-model HF doc\n")
    out.append(
        "Default panel (always): **V5** (layer-phase orientation) -> **V1** "
        "(headline node grid) -> **V2** (role x digit) -> **V8** (redundancy). "
        "For mixed/subtraction models add **V6** (shared engine), **V7** "
        "(delivery), **V10** (class capacity). **V3**/**V4** as illustrative "
        "insets on small/mid models. **V9** only when LINXFER probes exist. "
        "Addition-only models suppress V6/V7 and collapse V10 to one row.")
    out.append("")

    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}  ({len(loaded)} models x {len(VISUALIZATIONS)} views)")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        models = list(zip(["addition", "subtraction", "mixed"], sys.argv[1:]))
    else:
        models = DEFAULT_MODELS
    main(models)
