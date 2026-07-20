"""Exploratory per-model *visualizations* built from the full analysis JSONs.

Unlike ``maths_diagram`` (which renders the structural mechanism doc from the thin
role map), this module reads the **complete** ``behaviors.json`` + ``features.json``
node facts -- ``Fail%``, ``Impact``, ``Attn``, ``Math.Add/Sub/Neg`` (complexity +
PCA ``.SP/.MP``), ``Algo`` roles, and ``Probe:{LINXFER,CARRYLAYER,CARRYDEFER,
DELIVERY}`` -- and renders a battery of visualizations that each *combine several
fact types in one view* to expose the linkage between them (importance <-> role
<-> output digit <-> routing <-> delivery).

Design goal: candidate views for the auto-generated per-model HF doc. Each renderer
returns GitHub/HF-renderable Markdown (unicode-shaded tables or Mermaid), degrades
gracefully when a fact type is absent, and is pure given parsed nodes (so it is
offline-testable with fixtures). ``build_viz_model`` / ``load_viz_model`` assemble
the parsed model; ``VISUALIZATIONS`` lists the ten renderers with metadata.
"""
from __future__ import annotations

import json
import re
import statistics
from typing import Dict, List, Optional

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_model_loader import analysis_repo_id
from quanta_maths.maths_diagram import algo_task, _san, _san_label

# --- role taxonomy ---------------------------------------------------------
ROLE_ORDER = ["SA", "MD", "ND", "SC", "SS", "MB", "NB",
              "ST", "MT", "NT", "GT", "OPR", "SGN", "SLT", "STC", "MTC", "NTC"]
ROLE_GROUP = {
    "SA": "base", "MD": "base", "ND": "base",
    "SC": "carry", "SS": "carry", "MB": "carry", "NB": "carry",
    "ST": "tri", "MT": "tri", "NT": "tri", "GT": "tri",
    "OPR": "ctrl", "SGN": "ctrl", "SLT": "ctrl",
    "STC": "comb", "MTC": "comb", "NTC": "comb",
}
GROUP_ORDER = ["base", "carry", "tri", "ctrl", "comb"]


# ===========================================================================
# Parsing: behaviors.json + features.json -> structured per-node facts
# ===========================================================================

def _digits_after_A(minor: str) -> List[int]:
    """'A765' -> [7,6,5]; single-char digit indices (safe for n_digits<=9)."""
    m = re.match(r"A(\d+)", minor)
    if not m:
        return []
    return [int(c) for c in m.group(1)]


def parse_nodes(behav: list, feats: list, cfg: MathsConfig) -> List[dict]:
    """Merge the two node lists by (position,layer,is_head,num) and parse every
    tag family into a structured record. Pure/offline (no HF)."""
    def key(n):
        return (n["position"], n["layer"], n["is_head"], n["num"])

    def loc(k):
        p, l, h, nn = k
        return f"P{p}L{l}{'H' if h else 'M'}{nn}"

    rec: Dict[tuple, dict] = {}

    def get(k):
        if k not in rec:
            p, l, h, nn = k
            rec[k] = {"loc": loc(k), "position": p, "layer": l, "is_head": h,
                      "num": nn, "kind": "H" if h else "M", "fail": None,
                      "impact": [], "attn": {}, "roles": set(),
                      "math": {"Add": set(), "Sub": set(), "Neg": set()},
                      "pca": set(), "linxfer": {}, "carrylayer": None,
                      "carrydefer": None, "delivery": {}}
        return rec[k]

    for n in feats:
        node = get(key(n))
        for t in n.get("tags", []):
            if t.startswith("Algo:"):
                node["roles"].add(algo_task(t.split(":", 1)[1]))

    for n in behav:
        node = get(key(n))
        for t in n.get("tags", []):
            major, _, minor = t.partition(":")
            if major == "Fail%":
                try:
                    node["fail"] = int(minor)
                except ValueError:
                    pass
            elif major == "Impact":
                node["impact"] = _digits_after_A(minor)
            elif major == "Attn":
                mm = re.match(r"P(\d+)=(\d+)", minor)
                if mm:
                    node["attn"][int(mm.group(1))] = int(mm.group(2))
            elif major in ("Math.Add", "Math.Sub", "Math.Neg"):
                op = major.split(".")[1]
                if ".SP" in minor or ".MP" in minor:
                    node["pca"].add(minor)
                else:
                    node["math"][op].add(minor)
            elif major == "Probe":
                if ".LINXFER=" in minor:
                    d, v = minor.split(".LINXFER=")
                    node["linxfer"][d] = int(v)
                elif ".CARRYLAYER=" in minor:
                    node["carrylayer"] = int(minor.split("=")[1])
                elif ".CARRYDEFER=" in minor:
                    node["carrydefer"] = int(minor.split("=")[1])
                elif minor.startswith("DELIVERY."):
                    cls, _, route = minor[len("DELIVERY."):].partition("=")
                    node["delivery"][cls] = route
    return sorted(rec.values(),
                  key=lambda r: (r["layer"], r["kind"], r["num"], r["position"]))


def position_labels(cfg: MathsConfig) -> Dict[int, str]:
    """position -> token label (D5..D0, OP, D'5..D'0, =, A7(sign)..A0)."""
    nd = cfg.n_digits
    lab: Dict[int, str] = {}
    for k in range(nd):
        lab[int(cfg.dn_to_position_name(k)[1:])] = f"D{k}"
        lab[int(cfg.ddn_to_position_name(k)[1:])] = f"D'{k}"
    lab[int(cfg.op_position_name()[1:])] = "OP"
    lab[2 * nd + 1] = "="
    for k in range(nd + 2):
        lab[int(cfg.an_to_position_name(k)[1:])] = f"A{k}"
    return lab


def build_viz_model(model_name: str, behav: list, feats: list,
                    cfg: MathsConfig) -> dict:
    """Assemble a parsed model context used by all renderers (pure/offline)."""
    nd = cfg.n_digits
    nodes = parse_nodes(behav, feats, cfg)
    lab = position_labels(cfg)
    return {
        "name": model_name, "cfg": cfg, "n_digits": nd, "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads, "n_ctx": cfg.n_ctx, "nodes": nodes,
        "pos_label": lab, "label_pos": {v: k for k, v in lab.items()},
        "answer_digits": list(range(nd, -1, -1)),      # A_nd..A0 (high->low)
        "sign_digit": nd + 1,
        "op_classes": _op_classes_present(nodes, model_name),
    }


def load_viz_model(model_name: str, repo_id: str = None,
                   cfg: MathsConfig = None) -> dict:
    """Download behaviors.json + features.json from the analysis repo and build
    the parsed viz model. Map-only (no model forward pass)."""
    from huggingface_hub import hf_hub_download
    from quanta_maths.maths_hf_update import BEHAVIORS_FILE, FEATURES_FILE
    if repo_id is None:
        repo_id = analysis_repo_id(model_name)
    if cfg is None:
        cfg = MathsConfig()
        cfg.set_model_names(model_name)
    with open(hf_hub_download(repo_id=repo_id, filename=BEHAVIORS_FILE)) as f:
        behav = json.load(f)
    with open(hf_hub_download(repo_id=repo_id, filename=FEATURES_FILE)) as f:
        feats = json.load(f)
    return build_viz_model(model_name, behav, feats, cfg)


def _op_classes_present(nodes, model_name) -> List[str]:
    roles = {r for n in nodes for r in n["roles"]}
    cls = []
    if roles & {"SA", "SC", "SS", "ST", "STC"}:
        cls.append("ADD")
    if roles & {"MD", "MB", "MT", "MTC"}:
        cls.append("SUB")
    if roles & {"ND", "NB", "NT", "NTC"}:
        cls.append("NEG")
    if cls:
        return cls
    if model_name.startswith("add_"):
        return ["ADD"]
    if model_name.startswith("sub_"):
        return ["SUB", "NEG"]
    return ["ADD", "SUB", "NEG"]


# ===========================================================================
# small shared helpers
# ===========================================================================

def _heat(pct: Optional[int]) -> str:
    if pct is None:
        return " "
    if pct >= 80:
        return "█"
    if pct >= 50:
        return "▓"
    if pct >= 20:
        return "▒"
    if pct >= 5:
        return "░"
    return "·"


def _bar(frac: float, width: int = 8) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "·" * (width - n)


def _roles_str(node) -> str:
    rs = sorted(node["roles"], key=lambda r: (ROLE_ORDER.index(r)
                if r in ROLE_ORDER else 99, r))
    return "/".join(rs)


def _present_roles(nodes) -> List[str]:
    have = {r for n in nodes for r in n["roles"]}
    return [r for r in ROLE_ORDER if r in have] + \
           sorted(r for r in have if r not in ROLE_ORDER)


def _adig_label(model, d: int) -> str:
    return "sign" if d == model["sign_digit"] else f"A{d}"


# ===========================================================================
# VIZ 1 -- Enriched node map grid  (position x layer  x  role + Fail% + type)
# ===========================================================================

def viz_node_grid(model) -> str:
    nodes = model["nodes"]
    lab = model["pos_label"]
    positions = sorted({n["position"] for n in nodes})
    lanes = sorted({(n["layer"], n["kind"], n["num"]) for n in nodes})
    by = {(n["position"], n["layer"], n["kind"], n["num"]): n for n in nodes}

    head = ["Lane"] + [f"P{p}<br>{lab.get(p, '?')}" for p in positions]
    rows = ["| " + " | ".join(head) + " |",
            "|" + "|".join(["---"] * len(head)) + "|"]
    for (l, k, nn) in lanes:
        cells = [f"**L{l}{k}{nn}**"]
        for p in positions:
            node = by.get((p, l, k, nn))
            if not node:
                cells.append("")
                continue
            r = _roles_str(node) or "·"
            cells.append(f"{_heat(node['fail'])} {r}")
        rows.append("| " + " | ".join(cells) + " |")
    legend = ("Cell = `Fail% shade` + role(s). Shade: `·`<5 `░`<20 `▒`<50 "
              "`▓`<80 `█`>=80. Lanes are `L{layer}{H head|M mlp}{num}`.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 2 -- Role x answer-digit impact matrix  (role + Impact + Fail%)
# ===========================================================================

def viz_role_digit_matrix(model) -> str:
    nodes = model["nodes"]
    digits = model["answer_digits"] + [model["sign_digit"]]
    roles = _present_roles(nodes)
    cols = ["Role"] + [_adig_label(model, d) for d in digits]
    rows = ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in roles:
        rnodes = [n for n in nodes if r in n["roles"]]
        cells = [f"`{r}`"]
        for d in digits:
            hit = [n for n in rnodes if d in n["impact"]]
            if not hit:
                cells.append("")
                continue
            mx = max((n["fail"] or 0) for n in hit)
            cells.append(f"{_heat(mx)}{len(hit)}")
        rows.append("| " + " | ".join(cells) + " |")
    legend = ("Cell = `Fail% shade`+`# nodes` with that role whose ablation "
              "breaks that answer digit. Links **role -> output digit -> "
              "importance -> redundancy** in one grid.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 3 -- Attention fetch/routing flow  (Attn + position semantics + role)
# ===========================================================================

def _answer_positions(model):
    an = model["cfg"].an_to_position_name
    return {int(an(k)[1:]): k for k in range(model["n_digits"] + 1)}


def viz_attention_routing(model) -> str:
    nodes = model["nodes"]
    lab = model["pos_label"]
    ans_pos = _answer_positions(model)          # pos -> answer digit k
    edges = {}                                  # (src_pos, ap) -> summed pct
    for n in nodes:
        if not n["is_head"] or n["position"] not in ans_pos:
            continue
        for src, pct in n["attn"].items():
            edges[(src, n["position"])] = edges.get((src, n["position"]), 0) + pct
    # keep top-2 sources per answer position
    top = {}
    for (src, ap), pct in edges.items():
        top.setdefault(ap, []).append((pct, src))
    keep = set()
    for ap, lst in top.items():
        for pct, src in sorted(lst, reverse=True)[:2]:
            keep.add((src, ap))
    if not keep:
        return "_(no answer-position attention heads in this map)_"

    L = ["```mermaid", "flowchart LR"]
    for p in sorted({s for s, _ in keep}):
        L.append(f'  I{p}["{_san(lab.get(p, "P"+str(p)))}"]:::inp')
    for p in sorted({a for _, a in keep}):
        L.append(f'  O{p}["{_san(lab.get(p, "P"+str(p)))}"]:::out')
    for (src, ap) in sorted(keep):
        k = ans_pos[ap]
        # star marks an operand-matched fetch (answer Ak pulling Dk or D'k)
        star = "*" if lab.get(src, "") in (f"D{k}", f"D'{k}") else ""
        L.append(f'  I{src} -->|"{edges[(src, ap)]}{star}"| O{ap}')
    L += ["  classDef inp fill:#eaf2f8,stroke:#2e86c1,color:#1b4f72;",
          "  classDef out fill:#eaecee,stroke:#566573,color:#212f3d;",
          "```"]
    legend = ("Edges = summed attention % from an input token to an "
              "answer-position node (top-2 per answer digit); `*` marks an "
              "operand-matched fetch. Shows each answer digit **fetching its two "
              "operands** `Dk`,`D'k` (+ carry tail).")
    return legend + "\n\n" + "\n".join(L)


# ===========================================================================
# VIZ 4 -- Per-digit compute pipeline  (role chain + delivery + digit)
# ===========================================================================

def _locs_for(nodes, roles, digit=None) -> List[str]:
    out = []
    for n in nodes:
        if n["roles"] & set(roles) and (digit is None or digit in n["impact"]):
            out.append(n["loc"])
    return sorted(set(out))


def viz_digit_pipeline(model) -> str:
    nodes = model["nodes"]
    # choose the answer digit with the richest role coverage
    best, best_score = None, -1
    for d in model["answer_digits"]:
        base = _locs_for(nodes, ["SA", "MD", "ND"], d)
        carry = _locs_for(nodes, ["SC", "SS", "MB", "NB"], d)
        comb = _locs_for(nodes, ["STC", "MTC", "NTC"], d)
        score = (len(base) > 0) + (len(carry) > 0) + (len(comb) > 0) + \
            0.01 * (len(base) + len(carry) + len(comb))
        if score > best_score:
            best, best_score = d, score
    d = best if best is not None else 0
    base = _locs_for(nodes, ["SA", "MD", "ND"], d)
    carry = _locs_for(nodes, ["SC", "SS", "MB", "NB"], d)
    tri = _locs_for(nodes, ["ST", "MT", "NT", "GT"])
    comb = _locs_for(nodes, ["STC", "MTC", "NTC"], d) or \
        [n["loc"] for n in nodes if not n["is_head"]
         and n["layer"] == model["n_layers"] - 1 and d in n["impact"]]
    delivery = {}
    for n in nodes:
        delivery.update(n["delivery"])
    droute = ", ".join(f"{c}:{r}" for c, r in sorted(delivery.items())) or "n/a"

    def box(idx, title, locs, cls):
        return f'  {idx}["{_san_label(title + "<br/>" + (", ".join(locs) or "none"))}"]:::{cls}'

    L = ["```mermaid", "flowchart LR",
         f'  DK["operands D{d}, D\'{d}"]:::inp',
         box("BASE", f"base digit A{d}", base, "shared"),
         box("TRI", "tri-state and compare, question tail", tri, "shared"),
         box("CB", f"carry or borrow into A{d}", carry, "shared"),
         '  RES["resolve cascade to binary carry or borrow"]:::shared',
         box("COMB", f"combiner for A{d}", comb, "comb"),
         f'  AN(["answer digit A{d}"]):::out',
         "  DK --> BASE", "  DK --> CB", "  TRI --> RES", "  CB --> RES",
         "  BASE --> COMB",
         f'  RES -->|"delivery {_san(droute)}"| COMB',
         "  COMB --> AN",
         "  classDef shared fill:#d5f5e3,stroke:#27ae60,color:#145a32;",
         "  classDef comb fill:#d6eaf8,stroke:#2e86c1,color:#1b4f72;",
         "  classDef inp fill:#ffffff,stroke:#333;color:#111;",
         "  classDef out fill:#eaecee,stroke:#566573,color:#212f3d;", "```"]
    legend = (f"Causal chain for the best-covered answer digit **A{d}**, with the "
              "actual node locs at each stage and the per-class **delivery route** "
              "on the resolve->combiner edge.")
    return legend + "\n\n" + "\n".join(L)


# ===========================================================================
# VIZ 5 -- Layer-phase profile  (layer + role groups + Fail% + phase)
# ===========================================================================

def viz_layer_phase(model) -> str:
    nodes = model["nodes"]
    nl = model["n_layers"]
    rows = ["| Layer | Heads | MLPs | Role groups (n) | mean/max Fail% | Phase |",
            "|---|---|---|---|---|---|"]
    for l in range(nl):
        ln = [n for n in nodes if n["layer"] == l]
        if not ln:
            continue
        heads = sum(1 for n in ln if n["is_head"])
        mlps = sum(1 for n in ln if not n["is_head"])
        grp = {}
        for n in ln:
            for r in n["roles"]:
                g = ROLE_GROUP.get(r, "other")
                grp[g] = grp.get(g, 0) + 1
        grp_s = ", ".join(f"{g}:{grp[g]}" for g in GROUP_ORDER if g in grp) or "-"
        fails = [n["fail"] for n in ln if n["fail"] is not None]
        fs = f"{statistics.mean(fails):.0f}/{max(fails)}" if fails else "-"
        phase = ("fetch + per-digit compute" if l == 0 else
                 "combine + emit" if l == nl - 1 else "resolve + select")
        rows.append(f"| L{l} | {heads} | {mlps} | {grp_s} | {fs} | {phase} |")
    legend = ("Per layer: node counts, which **role groups** live there, the "
              "Fail% importance, and the algorithm **phase** -- the pipeline depth "
              "at a glance.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 6 -- Polysemantic role-sharing matrix  (role x role co-location)
# ===========================================================================

def viz_sharing_matrix(model) -> str:
    nodes = model["nodes"]
    roles = _present_roles(nodes)
    node_roles = [n["roles"] for n in nodes if n["roles"]]
    def co(a, b):
        return sum(1 for rs in node_roles if a in rs and b in rs)
    cols = ["role"] + [f"`{r}`" for r in roles]
    rows = ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, a in enumerate(roles):
        cells = [f"`{a}`"]
        for j, b in enumerate(roles):
            if j < i:
                cells.append("")
            elif j == i:
                cells.append(f"**{co(a, a)}**")
            else:
                c = co(a, b)
                cells.append(str(c) if c else "·")
        rows.append("| " + " | ".join(cells) + " |")
    legend = ("Diagonal = # nodes with that role; off-diagonal = # nodes carrying "
              "**both** roles (polysemantic sharing / the shared engine). "
              "e.g. `SA`&`MD`&`ND` co-located = one head computes all three.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 7 -- Delivery-route-by-class  (Probe:DELIVERY + class + combiner)
# ===========================================================================

_ROUTE_TXT = {"res": "residual only", "resatt": "residual + last-layer attn",
              "att": "attention only", "none": "not delivered"}


def viz_delivery_routes(model) -> str:
    delivery = {}
    for n in model["nodes"]:
        for c, r in n["delivery"].items():
            delivery[c] = r
    if not delivery:
        return "_(no `Probe:DELIVERY` tags in this model's map)_"
    L = ["```mermaid", "flowchart LR",
         '  SRC["resolved carry or borrow<br/>from question-tail cascade"]:::shared',
         '  COMB["answer-position combiner MLP"]:::comb']
    cmap = {"ADD": "add", "SUB": "sub", "NEG": "neg"}
    for c in ["ADD", "SUB", "NEG"]:
        if c in delivery:
            route = _ROUTE_TXT.get(delivery[c], delivery[c])
            L.append(f'  SRC -->|"{c}: {_san(route)}"| COMB')
    L += ["  COMB --> ANS([answer digit]):::out",
          "  classDef shared fill:#d5f5e3,stroke:#27ae60,color:#145a32;",
          "  classDef comb fill:#d6eaf8,stroke:#2e86c1,color:#1b4f72;",
          "  classDef out fill:#eaecee,stroke:#566573,color:#212f3d;", "```"]
    tbl = ["", "| Class | Route |", "|---|---|"]
    for c in ["ADD", "SUB", "NEG"]:
        if c in delivery:
            tbl.append(f"| {c} | {delivery[c]} = {_ROUTE_TXT.get(delivery[c], '?')} |")
    legend = ("How the resolved carry/borrow reaches the shared combiner, **per "
              "question class** -- exposes that ADD and SUB/NEG can use different "
              "routes in the same network.")
    return legend + "\n\n" + "\n".join(L) + "\n".join(tbl)


# ===========================================================================
# VIZ 8 -- Redundancy & importance spectrum per role  (count + Fail% spread)
# ===========================================================================

def viz_redundancy_spectrum(model) -> str:
    nodes = model["nodes"]
    roles = _present_roles(nodes)
    rows = ["| Role | # nodes | Fail% min/med/max | importance spread |",
            "|---|---|---|---|"]
    for r in roles:
        fs = sorted(n["fail"] for n in nodes if r in n["roles"]
                    and n["fail"] is not None)
        if not fs:
            rows.append(f"| `{r}` | {sum(1 for n in nodes if r in n['roles'])} | - | - |")
            continue
        med = statistics.median(fs)
        rows.append(f"| `{r}` | {len(fs)} | {min(fs)}/{med:.0f}/{max(fs)} | "
                    f"`{_bar(max(fs)/100)}` |")
    legend = ("Per role: **how many nodes** carry it (redundancy) and the spread "
              "of their ablation Fail% (importance). Many low-Fail% nodes for a "
              "role = redundant/class-level necessity, not a single critical node.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 9 -- Operand linear-decodability (Probe:LINXFER) fetch map
# ===========================================================================

def viz_linxfer_map(model) -> str:
    rows_data = []
    for n in model["nodes"]:
        for d, v in n["linxfer"].items():
            rows_data.append((d, n["loc"], v))
    if not rows_data:
        return ("_(no `Probe:LINXFER` tags -- operand linear-decodability was not "
                "probed for this model; typical for subtraction-only maps)_")
    rows_data.sort()
    rows = ["| Operand digit | fetch node | linear decode acc% | ",
            "|---|---|---|"]
    for d, loc, v in rows_data:
        rows.append(f"| {d} | `{loc}` | {_heat(v)} {v} |")
    legend = ("Where each operand digit becomes **linearly decodable** at its "
              "first-layer fetch site (balanced accuracy). Links the `Probe` "
              "geometry fact to the concrete fetch node.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# VIZ 10 -- Operation-class capacity across output digits (Math.* + Impact)
# ===========================================================================

def viz_class_capacity(model) -> str:
    nodes = model["nodes"]
    digits = model["answer_digits"] + [model["sign_digit"]]
    ops = [("Add", "ADD"), ("Sub", "SUB"), ("Neg", "NEG")]
    present = [(mk, lbl) for mk, lbl in ops
               if any(n["math"][mk] for n in nodes)]
    if not present:
        return "_(no `Math.Add/Sub/Neg` complexity tags in this map)_"
    cols = ["Op \\ digit"] + [_adig_label(model, d) for d in digits]
    rows = ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for mk, lbl in present:
        cells = [f"**{lbl}**"]
        for d in digits:
            hit = [n for n in nodes if n["math"][mk] and d in n["impact"]]
            if not hit:
                cells.append("")
                continue
            mx = max((n["fail"] or 0) for n in hit)
            cells.append(f"{_heat(mx)}{len(hit)}")
        rows.append("| " + " | ".join(cells) + " |")
    legend = ("For each operation class (from `Math.Add/Sub/Neg` tags), how many "
              "nodes (shaded by max Fail%) contribute to each output digit -- "
              "shows where each operation's **capacity** concentrates and the "
              "per-node polysemantic overlap between classes.")
    return legend + "\n\n" + "\n".join(rows)


# ===========================================================================
# Registry
# ===========================================================================

VISUALIZATIONS = [
    ("V1", "Enriched node-map grid", viz_node_grid,
     "location + role + Fail% + node-type"),
    ("V2", "Role x answer-digit impact matrix", viz_role_digit_matrix,
     "role + Impact-digit + Fail% + redundancy"),
    ("V3", "Attention fetch/routing flow", viz_attention_routing,
     "attention + position-semantics + role"),
    ("V4", "Per-digit compute pipeline", viz_digit_pipeline,
     "role-chain + Impact-digit + delivery route + location"),
    ("V5", "Layer-phase profile", viz_layer_phase,
     "layer + role-group + Fail% + phase"),
    ("V6", "Polysemantic role-sharing matrix", viz_sharing_matrix,
     "role co-location (shared engine)"),
    ("V7", "Delivery-route-by-class", viz_delivery_routes,
     "delivery route + question class + combiner"),
    ("V8", "Redundancy & importance spectrum", viz_redundancy_spectrum,
     "role + node-count + Fail% distribution"),
    ("V9", "Operand decodability (LINXFER) map", viz_linxfer_map,
     "Probe geometry + fetch location + operand digit"),
    ("V10", "Operation-class capacity map", viz_class_capacity,
     "Math.* op-class + Impact-digit + Fail%"),
]


def render_all(model) -> Dict[str, str]:
    return {vid: fn(model) for vid, _, fn, _ in VISUALIZATIONS}
