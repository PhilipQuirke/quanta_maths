"""Tests for the per-model mechanism-doc generator (maths_diagram).

Offline: loc summarising, shared-overlap detection, token layout, and Mermaid
well-formedness of both generated diagrams on a synthetic registry (no HF).
HF-gated: build the full doc for the mixed model and check it is well-formed and
contains the expected roles.
"""
import os
import re
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_diagram import (
    _summarize_locs, _pos_ranges, _shared_overlaps, _san, _op_classes,
    token_layout_md, node_inventory_md, algo_task, build_model_map,
    _registry_from_map,
    logical_mechanism_mermaid, implementation_mermaid, build_mechanism_markdown,
    build_mechanism_markdown_for_model,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"


def mermaid_wellformed(block: str):
    """Return a list of problems (empty == well-formed) for a mermaid block body."""
    problems = []
    for i, l in enumerate(block.splitlines()):
        if l.count('"') % 2 != 0:
            problems.append(f"uneven quotes line {i+1}")
    if block.count("(") != block.count(")"):
        problems.append("unbalanced parens")
    if block.count("[") != block.count("]"):
        problems.append("unbalanced brackets")
    if len(re.findall(r"\bsubgraph\b", block)) != len(re.findall(r"^\s*end\s*$", block, re.M)):
        problems.append("subgraph/end mismatch")
    for lab in re.findall(r'"([^"]*)"', block):
        if re.search(r"[\[\]{}|<>()]", lab.replace("<br/>", "")):
            problems.append(f"forbidden char in label: {lab}")
    return problems


def _mermaid_body(s: str) -> str:
    return re.findall(r"```mermaid\n(.*?)```", s, re.S)[0]


SYNTH_REGISTRY = {
    "model": MIX6,
    "roles": {
        "ST": ["P9L0H1", "P10L0H1"], "MT": ["P10L0H1", "P12L0H0"], "GT": ["P10L0H1"],
        "SA": ["P15L0H1", "P16L0H1"], "MD": ["P15L0H1", "P16L0H1"], "ND": ["P16L0H1"],
        "SC": ["P15L0H0"], "OPR": ["P6L0H0"], "SGN": ["P14L0H2"], "SLT": ["P16L1H1"],
    },
    "combiners": ["P15L2M0", "P16L2M0"],
    "positions": {"last_layer": 2},
}


class TestHelpers(unittest.TestCase):
    def test_pos_ranges(self):
        self.assertEqual(_pos_ranges([15, 16, 17, 18]), "P15-P18")
        self.assertEqual(_pos_ranges([9, 10, 13]), "P9-P10,P13")

    def test_summarize_locs(self):
        s = _summarize_locs(["P15L0H1", "P16L0H1", "P17L0H1"])
        self.assertEqual(s, "L0H1@P15-P17")

    def test_san_strips_forbidden(self):
        self.assertNotRegex(_san("a(b)[c]{d}|e<f>"), r"[\[\]{}|<>()]")

    def test_shared_overlaps(self):
        ov = dict(((a, b), c) for a, b, c in _shared_overlaps(SYNTH_REGISTRY))
        self.assertEqual(ov[("MD", "SA")], 2)   # both at P15L0H1, P16L0H1
        self.assertEqual(ov[("GT", "MT")], 1)   # both at P10L0H1

    def test_op_classes(self):
        self.assertEqual(_op_classes(SYNTH_REGISTRY), ["ADD", "SUB", "NEG"])

    def test_op_classes_name_fallback(self):
        # role tags too sparse to infer op -> fall back to the model-name operation
        self.assertEqual(_op_classes({"model": "add_d14_l2_h3_t60K_s572091",
                                      "roles": {"ST": ["P26L0H0"]}}), ["ADD"])
        self.assertEqual(_op_classes({"model": "sub_d6_l2_h3_t30K_s372001",
                                      "roles": {}}), ["SUB", "NEG"])

    def test_algo_task(self):
        self.assertEqual(algo_task("A5.SA"), "SA")
        self.assertEqual(algo_task("D4.GT"), "GT")
        self.assertEqual(algo_task("A0.STC"), "STC")
        self.assertEqual(algo_task("OPR"), "OPR")
        self.assertEqual(algo_task("SGN"), "SGN")
        # regression: three-part tag must resolve to the task, not the trailing
        # position parameter (old split('.')[-1] returned 'A5').
        self.assertEqual(algo_task("A5.ND.A5"), "ND")


# maths.json / behavior.json fixtures (no HF) exercising build_model_map,
# including the three-part-tag regression (A5.ND.A5 -> ND, not a bogus 'A5').
FIX_MATHS = [
    {"position": 6, "layer": 0, "is_head": True, "num": 0, "tags": ["Algo:OPR"]},
    {"position": 15, "layer": 0, "is_head": True, "num": 2,
     "tags": ["Algo:A5.SA", "Algo:A5.MD", "Algo:A5.ND.A5"]},
    {"position": 16, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Algo:A4.SA", "Algo:A4.ND"]},
    {"position": 9, "layer": 0, "is_head": True, "num": 1, "tags": ["Algo:A4.ST"]},
]
FIX_BEHAV = [
    {"position": 15, "layer": 0, "is_head": True, "num": 2,
     "tags": ["Fail%:57", "Impact:A5", "Attn:P0=49"]},
    {"position": 16, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Fail%:50", "Impact:A4"]},
    {"position": 15, "layer": 2, "is_head": False, "num": 0,
     "tags": ["Fail%:0", "Impact:A5"]},   # combiner (last layer, answer pos)
    {"position": 20, "layer": 2, "is_head": False, "num": 0,
     "tags": ["Fail%:1", "Impact:A0"]},   # combiner
]


class TestBuildModelMap(unittest.TestCase):
    def setUp(self):
        self.cfg = MathsConfig()
        self.cfg.set_model_names(MIX6)
        self.m = build_model_map(MIX6, self.cfg, FIX_MATHS, FIX_BEHAV)

    def test_three_part_tag_regresses_to_task(self):
        self.assertNotIn("A5", self.m["roles"])            # no bogus role
        nd = {e["loc"] for e in self.m["roles"]["ND"]}
        self.assertIn("P15L0H2", nd)                        # A5.ND.A5 -> ND@P15
        self.assertIn("P16L0H1", nd)                        # A4.ND

    def test_roles_and_behaviour_attached(self):
        sa = {e["loc"] for e in self.m["roles"]["SA"]}
        self.assertEqual(sa, {"P15L0H2", "P16L0H1"})
        p15 = next(e for e in self.m["roles"]["SA"] if e["loc"] == "P15L0H2")
        self.assertEqual(p15["fail"], ["Fail%:57"])
        self.assertEqual(p15["impact"], ["Impact:A5"])
        self.assertEqual(p15["attn"], ["Attn:P0=49"])

    def test_combiners_detected(self):
        self.assertEqual({c["loc"] for c in self.m["combiners"]},
                         {"P15L2M0", "P20L2M0"})
        self.assertTrue(all(isinstance(c["produces_A"], int)
                            for c in self.m["combiners"]))

    def test_schema_keys(self):
        for k in ("model", "config", "n_map_nodes", "roles", "combiners",
                  "positions"):
            self.assertIn(k, self.m)
        self.assertEqual(self.m["n_map_nodes"], len(FIX_MATHS))
        self.assertEqual(self.m["positions"]["OPR"], "P6")
        self.assertEqual(self.m["positions"]["last_layer"], 2)


class TestBuildDocFromMap(unittest.TestCase):
    """The doc generator consumes the map dict offline (no HF)."""
    def setUp(self):
        cfg = MathsConfig(); cfg.set_model_names(MIX6)
        self.doc = build_mechanism_markdown(
            build_model_map(MIX6, cfg, FIX_MATHS, FIX_BEHAV))

    def test_two_wellformed_diagrams(self):
        blocks = re.findall(r"```mermaid\n(.*?)```", self.doc, re.S)
        self.assertEqual(len(blocks), 2)
        for b in blocks:
            self.assertEqual(mermaid_wellformed(b), [])

    def test_inventory_uses_fixed_task(self):
        self.assertIn("`ND`", self.doc)
        self.assertNotIn("`A5`", self.doc)      # no bogus role leaks into doc
        self.assertIn("glossary", self.doc)     # task-code meanings linked out

    def test_requires_dict_not_name(self):
        with self.assertRaises(TypeError):
            build_mechanism_markdown(MIX6)


class TestGeneratorsOffline(unittest.TestCase):
    def setUp(self):
        self.cfg = MathsConfig()
        self.cfg.set_model_names(MIX6)  # offline parse

    def test_token_layout(self):
        md = token_layout_md(self.cfg)
        self.assertIn("P6", md)    # OPR
        self.assertIn("P13", md)   # '=' at 2*nd+1
        self.assertIn("P14", md)   # SGN

    def test_logical_mermaid_wellformed(self):
        m = logical_mechanism_mermaid(self.cfg, SYNTH_REGISTRY)
        self.assertEqual(mermaid_wellformed(_mermaid_body(m)), [])

    def test_implementation_mermaid_wellformed(self):
        m = implementation_mermaid(self.cfg, SYNTH_REGISTRY)
        body = _mermaid_body(m)
        self.assertEqual(mermaid_wellformed(body), [])
        self.assertIn("SLT", body)
        self.assertIn("L2M0", body)
        self.assertIn("<br/>", body)          # line breaks preserved (not mangled)
        self.assertNotIn("/br//", body)       # the _san mangling regression

    def test_node_inventory(self):
        md = node_inventory_md(SYNTH_REGISTRY)
        self.assertIn("`ST`", md)
        self.assertIn("combiner", md)

    def test_node_inventory_surfaces_all_roles(self):
        # SS (unknown-to-taxonomy) and STC (combiner Algo tag) must not be
        # silently dropped -- the inventory is a lossless view of the map.
        reg = {"model": MIX6, "roles": {
                   "SA": ["P15L0H1"], "SS": ["P15L0H1"], "STC": ["P15L2M0"]},
               "combiners": ["P15L2M0"], "positions": {"last_layer": 2}}
        md = node_inventory_md(reg)
        self.assertIn("`SS`", md)                       # catch-all role surfaced
        self.assertIn("`STC` (combiner tag)", md)       # combiner Algo tag row

    def test_add_only_registry_omits_control(self):
        # a pure-addition registry has no OPR/SGN/SLT -> logical diagram omits them
        reg = {"model": "add", "roles": {"SA": ["P15L0H1"], "SC": ["P15L0H0"],
               "ST": ["P9L0H1"]}, "combiners": ["P15L1M0"], "positions": {"last_layer": 1}}
        cfg = MathsConfig(); cfg.set_model_names("add_d6_l2_h3_t15K_s372001")
        m = logical_mechanism_mermaid(cfg, reg)
        self.assertEqual(mermaid_wellformed(_mermaid_body(m)), [])
        self.assertNotIn("SLT selector", m)
        self.assertEqual(_op_classes(reg), ["ADD"])


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestBuildDocHF(unittest.TestCase):
    def test_build_mixed_doc(self):
        doc = build_mechanism_markdown_for_model(MIX6)
        blocks = re.findall(r"```mermaid\n(.*?)```", doc, re.S)
        self.assertEqual(len(blocks), 2)
        for b in blocks:
            self.assertEqual(mermaid_wellformed(b), [])
        for needle in ("SLT", "combiner MLPs", "OPR", "SGN", "Node sharing detected",
                       "n_layers=3"):
            self.assertIn(needle, doc)
        # auto-detected polysemantic sharing on the real map
        self.assertRegex(doc, r"`(SA|MD)` and `(MD|SA)` share \*\*12\*\*")


if __name__ == "__main__":
    unittest.main()
