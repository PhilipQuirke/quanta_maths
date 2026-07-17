"""Generate per-model mechanism docs (Mermaid diagrams + text chunks) from a
model's verified HF map. Model-general so the same code produces a doc for any of
the ~40 maths models (add / sub / mixed), each with its own token positions and
node inventory.

Entry point: ``build_mechanism_markdown(model_name)`` -> a full Markdown doc with
(1) a logical mechanism diagram (no positions; adapts to the operation set),
(2) an actual-implementation diagram with real token positions, plus the token
layout, exact node inventory, and auto-detected node-sharing.

Everything is map-derived (no model forward passes), so it is cheap to batch.
Mermaid labels are sanitised (no ()[]{}|<> inside labels) so they render across
Mermaid versions (GitHub / VS Code).
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_model_loader import DEFAULT_HF_REPO, analysis_repo_id

# Task families (the algorithm's logical roles).
BASE_TASKS = ["SA", "MD", "ND"]        # base digit: add-sum / pos-diff / neg-diff
CARRY_TASKS = ["SC", "MB", "NB"]       # carry / borrow one
TRI_TASKS = ["ST", "MT", "NT", "GT"]   # tri-state + greater-than compare
CTRL_TASKS = ["OPR", "SGN", "SLT"]     # operator / sign / selector
COMBINER_TASKS = ["STC", "MTC", "NTC"]  # per-class combiner Algo tags (answer MLPs)

_FORBIDDEN = re.compile(r"[\[\]{}|<>()]")


def _san(text) -> str:
    """Strip characters that would break a Mermaid label (kept out of quotes)."""
    return _FORBIDDEN.sub("/", str(text))


def _san_label(text) -> str:
    """Sanitise a label but preserve the literal ``<br/>`` line-break tag."""
    return _san(str(text).replace("<br/>", "\x00")).replace("\x00", "<br/>")


_POS_LABEL = re.compile(r"^[AD]\d+$")


def algo_task(body: str) -> str:
    """Extract the task code from an ``Algo:`` tag body.

    Grammar: ``<posLabel>.<TASK>[.<extra>]`` where ``posLabel`` is an answer /
    digit label (``A5``, ``D4``) and ``TASK`` is the algorithmic role
    (``SA``, ``GT``, ``ND``, ``STC`` ...). A bare control task has no dot
    (``OPR``, ``SGN``). Examples::

        A5.SA    -> SA
        D4.GT    -> GT
        A5.ND.A5 -> ND     (trailing parameter ignored)
        A0.STC   -> STC
        OPR      -> OPR

    The task is the component after an optional leading position label; the old
    ``split('.')[-1]`` mis-read three-part tags (``A5.ND.A5`` -> ``A5``).
    """
    parts = body.split(".")
    if len(parts) > 1 and _POS_LABEL.match(parts[0]):
        return parts[1]
    return parts[0]


# ===========================================================================
# Map capture (map-only; no model load)
# ===========================================================================

def capture_role_registry(model_name: str, hf_repo: str = DEFAULT_HF_REPO,
                          cfg: MathsConfig = None) -> Tuple[MathsConfig, dict]:
    """Download a model's ``*_maths.json`` + ``*_behavior.json`` and return
    ``(cfg, registry)`` where registry has ``roles`` (task -> sorted node locs),
    ``combiners`` (last-layer answer-position MLP nodes), and key ``positions``.
    """
    from QuantaMechInterp.model_train_json import download_huggingface_json
    if cfg is None:
        cfg = MathsConfig()
        cfg.set_model_names(model_name)
    maths = download_huggingface_json(hf_repo, f"{model_name}_maths.json")
    behav = download_huggingface_json(hf_repo, f"{model_name}_behavior.json")

    def loc(n):
        return f"P{n['position']}L{n['layer']}{'H' if n['is_head'] else 'M'}{n['num']}"

    roles: Dict[str, list] = {}
    for node in maths:
        for t in node.get("tags", []):
            if t.startswith("Algo:"):
                task = algo_task(t.split(":", 1)[1])
                roles.setdefault(task, []).append(loc(node))
    roles = {t: sorted(set(v), key=_loc_key) for t, v in roles.items()}

    ll = cfg.n_layers - 1
    produce_pos = {int(cfg.an_to_position_name(k)[1:]) - 1: k
                   for k in range(cfg.n_digits + 2)}
    combiners = sorted({loc(n) for n in behav
                        if (not n["is_head"]) and n["layer"] == ll
                        and n["position"] in produce_pos}, key=_loc_key)

    positions = {
        "operand1": f"{cfg.dn_to_position_name(cfg.n_digits - 1)}-{cfg.dn_to_position_name(0)}",
        "OPR": cfg.op_position_name(),
        "operand2": f"{cfg.ddn_to_position_name(cfg.n_digits - 1)}-{cfg.ddn_to_position_name(0)}",
        "eq": f"P{2 * cfg.n_digits + 1}",
        "SGN": cfg.an_to_position_name(cfg.n_digits + 1),
        "answer": f"{cfg.an_to_position_name(cfg.n_digits)}-{cfg.an_to_position_name(0)}",
        "last_layer": ll,
    }
    return cfg, {"model": model_name, "roles": roles, "combiners": combiners,
                 "positions": positions}


# ===========================================================================
# Maximal per-model map (the results/maps/<model>.json artifact)
# ===========================================================================

def build_model_map(model_name: str, cfg: MathsConfig,
                    maths_nodes: list, behav_nodes: list) -> dict:
    """Build the maximal per-model map dict (the ``results/maps/<model>.json``
    schema) from already-loaded verified-map node lists.

    Pure and offline: no HuggingFace download and no model forward pass, so it is
    unit-testable with fixture node lists and is the single source of truth the
    diagram doc is generated from. Schema::

        {model, config, n_map_nodes, roles{task -> [rich node]},
         combiners[last-layer answer-position MLPs], positions}

    ``roles`` node entries carry ``loc/position/layer/is_head/num/algo`` plus the
    node's ``impact``/``fail``/``attn`` behaviour tags. Positive-control accuracy
    and provenance (``init_from``) are added by callers that load the model (see
    ``scripts/mixed_map.py``); their absence from the zoo-wide maps is the
    map-completeness backlog item in ``maths-next-steps.md``.
    """
    def loc(n):
        return f"P{n['position']}L{n['layer']}{'H' if n['is_head'] else 'M'}{n['num']}"

    btags = {}
    for n in behav_nodes:
        btags[(n["position"], n["layer"], n["is_head"], n["num"])] = n.get("tags", [])

    def behav_of(n, prefix):
        key = (n["position"], n["layer"], n["is_head"], n["num"])
        return [t for t in btags.get(key, []) if t.startswith(prefix)]

    roles: Dict[str, list] = {}
    for n in maths_nodes:
        for t in n.get("tags", []):
            if not t.startswith("Algo:"):
                continue
            body = t.split(":", 1)[1]
            roles.setdefault(algo_task(body), []).append({
                "loc": loc(n), "position": n["position"], "layer": n["layer"],
                "is_head": n["is_head"], "num": n["num"], "algo": body,
                "impact": behav_of(n, "Impact"),
                "fail": behav_of(n, "Fail%"),
                "attn": behav_of(n, "Attn"),
            })

    ll = cfg.n_layers - 1
    answer_produce_pos = {int(cfg.an_to_position_name(k)[1:]) - 1: k
                          for k in range(cfg.n_digits + 2)}
    combiners = []
    for n in behav_nodes:
        if n["is_head"] or n["layer"] != ll:
            continue
        if n["position"] in answer_produce_pos:
            combiners.append({
                "loc": loc(n), "position": n["position"], "layer": n["layer"],
                "produces_A": answer_produce_pos[n["position"]],
                "fail": [t for t in n.get("tags", []) if t.startswith("Fail%")],
                "impact": [t for t in n.get("tags", []) if t.startswith("Impact")],
            })

    positions = {
        "OPR": cfg.op_position_name(),
        "eq": f"P{2 * cfg.n_digits + 1}",
        "SGN": cfg.an_to_position_name(cfg.n_digits + 1),
        "answer_digits": [cfg.an_to_position_name(k)
                          for k in range(cfg.n_digits, -1, -1)],
        "first_layer": 0,
        "last_layer": ll,
    }
    return {
        "model": model_name,
        "config": {"n_digits": cfg.n_digits, "n_layers": cfg.n_layers,
                   "n_heads": cfg.n_heads, "n_ctx": cfg.n_ctx,
                   "perc_add": getattr(cfg, "perc_add", None),
                   "perc_sub": getattr(cfg, "perc_sub", None)},
        "n_map_nodes": len(maths_nodes),
        "roles": roles,
        "combiners": combiners,
        "positions": positions,
    }


def capture_model_map(model_name: str, repo_id: str = None,
                      cfg: MathsConfig = None) -> dict:
    """Download a model's canonical analysis JSONs and return the maximal
    per-model map dict (see :func:`build_model_map`).

    Source: the per-model ``PhilipQuirke/QuantaMaths_<name>`` repo, which holds
    ``features.json`` (Algo role tags -> role nodes) and ``behaviors.json``
    (``Fail%``/``Impact``/``Attn``/``Probe`` tags -> behaviour nodes). Both are
    plain ``[{position, layer, is_head, num, tags}]`` lists. Map-only (no model
    forward pass), so cheap to batch across the ~33-model analysable zoo.
    """
    from huggingface_hub import hf_hub_download
    from quanta_maths.maths_hf_update import BEHAVIORS_FILE, FEATURES_FILE
    if repo_id is None:
        repo_id = analysis_repo_id(model_name)
    if cfg is None:
        cfg = MathsConfig()
        cfg.set_model_names(model_name)
    with open(hf_hub_download(repo_id=repo_id, filename=FEATURES_FILE)) as f:
        feats = json.load(f)          # Algo role tags
    with open(hf_hub_download(repo_id=repo_id, filename=BEHAVIORS_FILE)) as f:
        behav = json.load(f)          # Fail% / Impact / Attn / Probe tags
    return build_model_map(model_name, cfg, feats, behav)


# ===========================================================================
# loc parsing / summarising
# ===========================================================================

def _parse_loc(loc: str) -> Tuple[int, int, str, int]:
    m = re.match(r"P(\d+)L(\d+)([HM])(\d+)", loc)
    return int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))


def _loc_key(loc: str):
    p, l, k, n = _parse_loc(loc)
    return (l, k, n, p)


def _pos_ranges(positions: List[int]) -> str:
    ps = sorted(set(positions))
    out, i = [], 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[j] + 1:
            j += 1
        out.append(f"P{ps[i]}" if i == j else f"P{ps[i]}-P{ps[j]}")
        i = j + 1
    return ",".join(out)


def _summarize_locs(locs: List[str]) -> str:
    """['P15L0H1','P16L0H1',...] -> 'L0H1@P15-P20; L0H2@P15-P20' (Mermaid-safe)."""
    if not locs:
        return "none"
    groups: Dict[tuple, list] = {}
    for loc in locs:
        p, l, k, n = _parse_loc(loc)
        groups.setdefault((l, k, n), []).append(p)
    parts = [f"L{l}{k}{n}@{_pos_ranges(ps)}" for (l, k, n), ps in sorted(groups.items())]
    return "; ".join(parts)


def _present(registry, tasks):
    return [t for t in tasks if registry["roles"].get(t)]


def _shared_overlaps(registry) -> List[Tuple[str, str, int]]:
    """Task pairs that occupy the same node(s) (polysemantic sharing), by count."""
    node_tasks: Dict[str, set] = {}
    for task, locs in registry["roles"].items():
        for loc in locs:
            node_tasks.setdefault(loc, set()).add(task)
    pair_counts: Dict[tuple, int] = {}
    for tasks in node_tasks.values():
        ts = sorted(tasks)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                pair_counts[(ts[i], ts[j])] = pair_counts.get((ts[i], ts[j]), 0) + 1
    return sorted(((a, b, c) for (a, b), c in pair_counts.items()), key=lambda x: -x[2])


# ===========================================================================
# Text chunks
# ===========================================================================

def token_layout_md(cfg: MathsConfig) -> str:
    nd = cfg.n_digits
    p = capture_positions(cfg)
    rows = [
        (f"D{nd-1}..D0 (operand 1)", p["operand1"]),
        ("OPR (operator + or -)", p["OPR"]),
        (f"D'{nd-1}..D'0 (operand 2)", p["operand2"]),
        ("= (equals)", p["eq"]),
        ("SGN (answer sign)", p["SGN"]),
        (f"A{nd}..A0 (answer digits)", p["answer"]),
    ]
    out = ["| Tokens | Positions |", "| --- | --- |"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(out)


def capture_positions(cfg: MathsConfig) -> dict:
    nd = cfg.n_digits
    return {
        "operand1": f"{cfg.dn_to_position_name(nd - 1)}-{cfg.dn_to_position_name(0)}",
        "OPR": cfg.op_position_name(),
        "operand2": f"{cfg.ddn_to_position_name(nd - 1)}-{cfg.ddn_to_position_name(0)}",
        "eq": f"P{2 * nd + 1}",
        "SGN": cfg.an_to_position_name(nd + 1),
        "answer": f"{cfg.an_to_position_name(nd)}-{cfg.an_to_position_name(0)}",
    }


def node_inventory_md(registry) -> str:
    """Every tagged role in the map, so the inventory is a faithful, lossless
    view of the JSON: the known task families in canonical order, then the
    per-class combiner Algo tags, then any other tagged role (e.g. `SS`,
    `SV`/`MV`/`NV`), then the behaviour-derived combiner MLPs."""
    known = TRI_TASKS + BASE_TASKS + CARRY_TASKS + CTRL_TASKS + COMBINER_TASKS
    roles = registry["roles"]
    rows = ["| Task | Nodes |", "| --- | --- |"]
    for t in TRI_TASKS + BASE_TASKS + CARRY_TASKS + CTRL_TASKS:
        if roles.get(t):
            rows.append(f"| `{t}` | {', '.join(roles[t])} |")
    for t in COMBINER_TASKS:
        if roles.get(t):
            rows.append(f"| `{t}` (combiner tag) | {', '.join(roles[t])} |")
    for t in sorted(k for k in roles if k not in known):
        rows.append(f"| `{t}` | {', '.join(roles[t])} |")
    if registry["combiners"]:
        rows.append(f"| combiner (last-layer answer MLP) | {', '.join(registry['combiners'])} |")
    return "\n".join(rows)


_CLASSDEFS = (
    "  classDef shared fill:#d5f5e3,stroke:#27ae60,color:#145a32;\n"
    "  classDef ctrl fill:#fdebd0,stroke:#e67e22,color:#7e5109;\n"
    "  classDef comb fill:#d6eaf8,stroke:#2e86c1,color:#1b4f72;\n"
    "  classDef out fill:#eaecee,stroke:#566573,color:#212f3d;\n"
    "  classDef inp fill:#ffffff,stroke:#333333,color:#111111;"
)


# ===========================================================================
# Diagram 1 — logical mechanism (no positions; adapts to present tasks)
# ===========================================================================

def logical_mechanism_mermaid(cfg: MathsConfig, registry) -> str:
    base = _present(registry, BASE_TASKS)
    carry = _present(registry, CARRY_TASKS)
    tri = _present(registry, TRI_TASKS)
    has_opr = bool(registry["roles"].get("OPR"))
    has_sgn = bool(registry["roles"].get("SGN"))
    has_slt = bool(registry["roles"].get("SLT"))

    base_lbl = _san(" / ".join(base) or "base digit")
    carry_lbl = _san(" / ".join(carry) or "carry")
    tri_lbl = _san(" / ".join(tri) or "tri-state")
    shared_title = "SHARED by " + _san(", ".join(_op_classes(registry))) + " - one node set"

    L = ["flowchart TD",
         '  IN["Operand digit pair Dn and D\'n"]:::inp']
    if has_opr:
        L.append('  OPR(["OPR: operator plus or minus"]):::ctrl')
    L += [f'  subgraph ENG["Per-digit engine - {shared_title}"]',
          "    direction TB",
          f'    BASE["Base digit&#160;&#160;{base_lbl}"]:::shared',
          f'    CB["Carry or Borrow-one&#160;&#160;{carry_lbl}"]:::shared',
          f'    TRI["TriCase and Compare&#160;&#160;{tri_lbl}"]:::shared',
          "  end",
          "  IN --> BASE", "  IN --> CB", "  IN --> TRI",
          '  RES["Cascade resolver&#160;&#160;binary carry or borrow<br/>STEP-thresholded combiner input"]:::shared',
          "  TRI --> RES", "  CB --> RES"]
    if has_sgn:
        L += ['  SGN(["SGN: answer sign<br/>plus if add, else plus if D ge D\' else minus"]):::ctrl',
              "  TRI -->|compare cascade to the top| SGN"]
        if has_opr:
            L.append("  OPR --> SGN")
    if has_slt:
        L.append('  SEL(["SLT selector: choose the operation readout<br/>distributed, not a low-rank switch"]):::ctrl')
        if has_opr:
            L.append("  OPR --> SEL")
        if has_sgn:
            L.append("  SGN --> SEL")
    L += ['  COMB["Combiner per answer digit - SHARED<br/>STEP: base plus carry, or base minus borrow, mod 10"]:::comb',
          "  BASE --> COMB", "  RES --> COMB"]
    if has_slt:
        L.append("  SEL --> COMB")
    L += ['  COMB --> AN(["Answer digit An"]):::out']
    if has_sgn:
        L.append('  SGN --> ASGN(["Answer sign - leading answer token"]):::out')
    L.append(_CLASSDEFS)
    return "```mermaid\n" + "\n".join(L) + "\n```"


# ===========================================================================
# Diagram 2 — actual implementation (with token positions)
# ===========================================================================

def implementation_mermaid(cfg: MathsConfig, registry) -> str:
    p = capture_positions(cfg)
    r = registry["roles"]
    tri_locs = _dedup([loc for t in TRI_TASKS for loc in r.get(t, [])])
    base_locs = _dedup([loc for t in BASE_TASKS for loc in r.get(t, [])])
    carry_locs = _dedup([loc for t in CARRY_TASKS for loc in r.get(t, [])])
    opr_locs = r.get("OPR", [])
    sgn_locs = r.get("SGN", [])
    slt_locs = r.get("SLT", [])
    ll = registry["positions"]["last_layer"]

    def node(idx, label):
        return f'  {idx}["{_san_label(label)}"]'

    L = ["flowchart LR",
         '  subgraph TOK["Tokens"]', "    direction TB",
         node("TD", f"operand1 = {p['operand1']}") + ":::out",
         node("TOP", f"OPR = {p['OPR']}") + ":::ctrl",
         node("TDp", f"operand2 = {p['operand2']}") + ":::out",
         node("TEQ", f"equals = {p['eq']}") + ":::out",
         node("TSG", f"SGN = {p['SGN']}") + ":::out",
         node("TAN", f"answer = {p['answer']}") + ":::out",
         "  end"]
    if tri_locs:
        L += ['  subgraph L0W["Early writers - tricase and compare"]',
              node("STMT", "tri/compare " + " ".join(_present(registry, TRI_TASKS)) + "<br/>" + _summarize_locs(tri_locs)) + ":::shared",
              "  end"]
    L += ['  subgraph L0A["Answer-position writers"]',
          node("BASE", "base " + " ".join(_present(registry, BASE_TASKS)) + "<br/>" + _summarize_locs(base_locs)) + ":::shared"]
    if carry_locs:
        L.append(node("CB", "carry/borrow " + " ".join(_present(registry, CARRY_TASKS)) + "<br/>" + _summarize_locs(carry_locs)) + ":::shared")
    if opr_locs:
        L.append(node("OPRh", "OPR heads<br/>" + _summarize_locs(opr_locs)) + ":::ctrl")
    L.append("  end")
    if slt_locs or sgn_locs:
        L.append('  subgraph LSEL["Selector and control"]')
        if slt_locs:
            L.append(node("SLT", "SLT selector<br/>" + _summarize_locs(slt_locs)) + ":::ctrl")
        if sgn_locs:
            L.append(node("SGNh", "SGN heads<br/>" + _summarize_locs(sgn_locs)) + ":::ctrl")
        L.append("  end")
    L += [f'  subgraph L2["Combiners - last layer L{ll}"]',
          node("COMB", "combiner MLPs<br/>" + _summarize_locs(registry["combiners"])) + ":::comb",
          "  end"]

    # edges
    if tri_locs:
        L += ["  TD --> STMT", "  TDp --> STMT",
              "  STMT -->|carry or borrow cascade| COMB"]
    L += ["  TD --> BASE", "  TDp --> BASE", "  BASE --> COMB"]
    if carry_locs:
        L.append("  CB --> COMB")
    if opr_locs:
        L.append("  TOP --> OPRh")
        if slt_locs:
            L.append("  OPRh --> SLT")
    if slt_locs:
        L.append("  SLT --> COMB")
    L.append("  COMB --> TAN")
    if sgn_locs:
        if tri_locs:
            L.append("  STMT -->|compare to top, resolved at equals| SGNh")
        if opr_locs:
            L.append("  TOP --> SGNh")
        L.append("  SGNh --> TSG")
    L.append(_CLASSDEFS)
    return "```mermaid\n" + "\n".join(L) + "\n```"


def _dedup(locs):
    seen, out = set(), []
    for loc in locs:
        if loc not in seen:
            seen.add(loc)
            out.append(loc)
    return sorted(out, key=_loc_key)


def _op_classes(registry) -> List[str]:
    cls = []
    if registry["roles"].get("SA"):
        cls.append("ADD")
    if registry["roles"].get("MD") or registry["roles"].get("MT"):
        cls.append("SUB")
    if registry["roles"].get("ND") or registry["roles"].get("NB"):
        cls.append("NEG")
    if cls:
        return cls
    # Fall back to the model-name operation when the map's role tags are too
    # sparse to infer the class set (e.g. a large addition model whose map has
    # no SA base-digit tag).
    name = registry.get("model", "")
    if name.startswith("add_"):
        return ["ADD"]
    if name.startswith("sub_"):
        return ["SUB", "NEG"]
    if "mix" in name:
        return ["ADD", "SUB", "NEG"]
    return ["the task"]


# ===========================================================================
# Full doc assembly
# ===========================================================================

def _registry_from_map(model_map: dict) -> dict:
    """Adapt a maximal model-map dict (:func:`build_model_map`) to the thin
    registry the Mermaid builders consume: ``roles`` task -> sorted ``[loc]``,
    ``combiners`` -> sorted ``[loc]``, and ``positions`` carrying ``last_layer``.
    """
    roles = {t: sorted({e["loc"] for e in entries}, key=_loc_key)
             for t, entries in model_map.get("roles", {}).items()}
    combiners = sorted({c["loc"] for c in model_map.get("combiners", [])},
                       key=_loc_key)
    last_layer = model_map.get("positions", {}).get("last_layer")
    if last_layer is None:
        last_layer = model_map.get("config", {}).get("n_layers", 1) - 1
    return {"model": model_map["model"], "roles": roles,
            "combiners": combiners, "positions": {"last_layer": last_layer}}


def build_mechanism_markdown(model_map: dict, cfg: MathsConfig = None) -> str:
    """Build the mechanism doc from a maximal model-map dict (the
    ``results/maps/<model>.json`` schema; see :func:`build_model_map`).

    Deterministic and offline: no model load and no HuggingFace download. Pass
    the dict from :func:`capture_model_map` or a loaded ``results/maps/*.json``.
    Interpretation and causal verdicts are intentionally excluded (they live in
    the study notes and claim-evidence); per-task algorithmic meanings live in
    the glossary, linked from the generated doc.
    """
    if isinstance(model_map, str):
        raise TypeError(
            "build_mechanism_markdown takes a model-map dict, not a model name. "
            "Use build_mechanism_markdown_for_model(name) to capture from "
            "HuggingFace, or load results/maps/<model>.json first.")
    model_name = model_map["model"]
    if cfg is None:
        cfg = MathsConfig()
        cfg.set_model_names(model_name)
    registry = _registry_from_map(model_map)
    classes = ", ".join(_op_classes(registry))
    overlaps = _shared_overlaps(registry)
    overlap_lines = [f"- `{a}` and `{b}` share **{c}** node(s)"
                     for a, b, c in overlaps if c > 0][:12]
    glossary = "../../docs/thor-glossary.md#project-terms"

    doc = f"""# Auto-generated mechanism doc — `{model_name}`

**Generated by `quanta_maths.maths_diagram.build_mechanism_markdown` from the
model's maximal map JSON** (`results/maps/{model_name}.json`). Model type:
{classes}. Config: n_digits={cfg.n_digits}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads},
n_ctx={cfg.n_ctx}.

This doc is **structural** (map-derived). Interpretation and causal verdicts live
in the study notes and [claim-evidence](../../docs/maths-claim-evidence.md).
Per-task algorithmic meanings (`SA`/`MD`/`ND`, `ST`/`MT`/`NT`/`GT`,
`SC`/`MB`/`NB`, `OPR`/`SGN`/`SLT`, `STC`/`MTC`/`NTC`) are defined in the
[glossary]({glossary}). To regenerate, see
[docs/mixed_model_mechanism.md](../../docs/mixed_model_mechanism.md).

Legend: **green = shared** across operations (one node set), **orange =
control/selection** (OPR/SGN/SLT), **blue = combiner** (last-layer answer MLP),
**grey = tokens/outputs**.

## 1. Logical mechanism (no token positions)

{logical_mechanism_mermaid(cfg, registry)}

## 2. Actual implementation (with token positions)

Token layout:

{token_layout_md(cfg)}

{implementation_mermaid(cfg, registry)}

### Exact node inventory (from the verified map)

Task-code meanings: see the [glossary]({glossary}).

{node_inventory_md(registry)}

### Node sharing detected in the map (polysemantic nodes)

{chr(10).join(overlap_lines) if overlap_lines else "- (no shared nodes detected)"}
"""
    return doc


def build_mechanism_markdown_for_model(model_name: str, repo_id: str = None,
                                       cfg: MathsConfig = None) -> str:
    """Convenience: capture the maximal map from HuggingFace, then build the doc.
    (:func:`build_mechanism_markdown` itself is offline and takes the map dict.)
    """
    model_map = capture_model_map(model_name, repo_id=repo_id, cfg=cfg)
    return build_mechanism_markdown(model_map, cfg=cfg)
