"""Offline tests for the node-fact visualization toolkit (maths_viz).

Uses fixture behaviors/features node lists (no HuggingFace) to exercise the tag
parser and a representative set of renderers, including Mermaid well-formedness.
"""
import re
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_viz import (
    build_viz_model, parse_nodes, render_all, VISUALIZATIONS,
    viz_sharing_matrix, viz_delivery_routes, viz_attention_routing,
    viz_role_digit_matrix, viz_node_grid, viz_linxfer_map,
)

MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"   # d6: D2=P3, D'2=P10, A2=P19

# features.json (Algo) fixture
FEATS = [
    {"position": 19, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Algo:A2.SA", "Algo:A2.MD", "Algo:A2.ND"]},
    {"position": 15, "layer": 0, "is_head": True, "num": 0,
     "tags": ["Algo:A6.SC"]},
    {"position": 13, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Algo:A6.ST", "Algo:A6.MT", "Algo:A6.GT"]},
    {"position": 19, "layer": 2, "is_head": False, "num": 0,
     "tags": ["Algo:A2.STC", "Algo:A2.MTC"]},
    {"position": 20, "layer": 0, "is_head": True, "num": 3, "tags": ["Algo:OPR"]},
]
# behaviors.json fixture (Fail%/Impact/Attn/Math/Probe)
BEHAV = [
    {"position": 19, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Fail%:55", "Impact:A2", "Attn:P3=50", "Attn:P10=48",
              "Math.Add:S2", "Math.Sub:M2", "Math.Neg:N2", "Probe:A2.LINXFER=90"]},
    {"position": 15, "layer": 0, "is_head": True, "num": 0,
     "tags": ["Fail%:8", "Impact:A6", "Attn:P5=50"]},
    {"position": 13, "layer": 0, "is_head": True, "num": 1,
     "tags": ["Fail%:15", "Impact:A6543210", "Math.Add:A2.SP"]},
    {"position": 19, "layer": 2, "is_head": False, "num": 0,
     "tags": ["Fail%:5", "Impact:A2"]},
    {"position": 20, "layer": 0, "is_head": True, "num": 3,
     "tags": ["Fail%:3", "Impact:A1", "Probe:DELIVERY.ADD=res",
              "Probe:DELIVERY.SUB=resatt", "Probe:DELIVERY.NEG=resatt"]},
]

_FORBIDDEN = re.compile(r"[\[\]{}|<>()]")


def mermaid_labels_ok(md: str) -> bool:
    body = re.findall(r"```mermaid\n(.*?)```", md, re.S)
    if not body:
        return True
    for lab in re.findall(r'"([^"]*)"', body[0]):
        if _FORBIDDEN.search(lab.replace("<br/>", "")):
            return False
    return True


class TestParse(unittest.TestCase):
    def setUp(self):
        self.cfg = MathsConfig(); self.cfg.set_model_names(MIX6)
        self.nodes = parse_nodes(BEHAV, FEATS, self.cfg)
        self.by = {n["loc"]: n for n in self.nodes}

    def test_merge_and_roles(self):
        n = self.by["P19L0H1"]
        self.assertEqual(n["roles"], {"SA", "MD", "ND"})
        self.assertEqual(n["fail"], 55)
        self.assertIn(2, n["impact"])
        self.assertEqual(n["attn"], {3: 50, 10: 48})

    def test_math_and_pca_split(self):
        n = self.by["P19L0H1"]
        self.assertIn("S2", n["math"]["Add"])
        self.assertIn("M2", n["math"]["Sub"])
        tri = self.by["P13L0H1"]
        self.assertIn("A2.SP", tri["pca"])          # PCA tag routed to pca, not math
        self.assertNotIn("A2.SP", tri["math"]["Add"])

    def test_probe_parsing(self):
        self.assertEqual(self.by["P19L0H1"]["linxfer"], {"A2": 90})
        d = self.by["P20L0H3"]["delivery"]
        self.assertEqual(d, {"ADD": "res", "SUB": "resatt", "NEG": "resatt"})

    def test_impact_multichar(self):
        self.assertEqual(self.by["P13L0H1"]["impact"], [6, 5, 4, 3, 2, 1, 0])


class TestRenderers(unittest.TestCase):
    def setUp(self):
        cfg = MathsConfig(); cfg.set_model_names(MIX6)
        self.model = build_viz_model(MIX6, BEHAV, FEATS, cfg)

    def test_render_all_ten(self):
        outs = render_all(self.model)
        self.assertEqual(sorted(outs), sorted(v[0] for v in VISUALIZATIONS))
        self.assertEqual(len(outs), 10)

    def test_sharing_matrix_counts_colocation(self):
        md = viz_sharing_matrix(self.model)
        self.assertIn("`SA`", md)          # SA/MD/ND co-located on P19L0H1
        self.assertIn("`ND`", md)

    def test_node_grid_has_lane_and_role(self):
        md = viz_node_grid(self.model)
        self.assertIn("L0H1", md)
        self.assertIn("SA", md)

    def test_role_digit_matrix(self):
        self.assertIn("`SA`", viz_role_digit_matrix(self.model))

    def test_delivery_mermaid_wellformed(self):
        md = viz_delivery_routes(self.model)
        self.assertIn("residual", md)
        self.assertTrue(mermaid_labels_ok(md))

    def test_attention_mermaid_wellformed_and_operand_star(self):
        md = viz_attention_routing(self.model)
        self.assertTrue(mermaid_labels_ok(md))
        self.assertIn("*", md)             # A2 fetches D2(P3)/D'2(P10): operand match

    def test_linxfer_present(self):
        self.assertIn("90", viz_linxfer_map(self.model))

    def test_linxfer_na_when_absent(self):
        cfg = MathsConfig(); cfg.set_model_names(MIX6)
        model = build_viz_model(MIX6, [{"position": 19, "layer": 0,
            "is_head": True, "num": 1, "tags": ["Fail%:5"]}], [], cfg)
        self.assertIn("no `Probe:LINXFER`", viz_linxfer_map(model))


if __name__ == "__main__":
    unittest.main()
