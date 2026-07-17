"""Coverage tests: the HF-update technique registry must cover addition AND
subtraction/mixed models (combiners STC/MTC/NTC, delivery route), routed by cfg.
Offline (registry routing) + HF (NTC + delivery on the mixed model).
"""
import os
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_hf_update import TECHNIQUES, techniques_for
from quanta_maths.maths_cascade import model_classes

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"


def _cfg(name):
    c = MathsConfig()
    c.set_model_names(name)
    return c


class TestRegistryRouting(unittest.TestCase):
    def test_addition_model_gets_add_combiner_not_neg(self):
        names = {t.name for t in techniques_for(_cfg("add_d6_l2_h3_t15K_s372001"))}
        self.assertIn("add_combiner_STC", names)
        self.assertNotIn("sub_combiner_MTC", names)
        self.assertNotIn("neg_combiner_NTC", names)
        self.assertIn("combiner_delivery_route", names)

    def test_subtraction_model_gets_mtc_and_ntc(self):
        names = {t.name for t in techniques_for(_cfg("sub_d6_l2_h3_t30K_s372001"))}
        self.assertIn("sub_combiner_MTC", names)
        self.assertIn("neg_combiner_NTC", names)     # the gap this fixes
        self.assertNotIn("add_combiner_STC", names)

    def test_mixed_model_gets_all_three_combiners(self):
        names = {t.name for t in techniques_for(_cfg(MIX6))}
        for n in ("add_combiner_STC", "sub_combiner_MTC", "neg_combiner_NTC",
                  "combiner_delivery_route"):
            self.assertIn(n, names)

    def test_model_classes(self):
        self.assertEqual(model_classes(_cfg("add_d6_l2_h3_t15K_s372001")), ["ADD"])
        self.assertEqual(model_classes(_cfg("sub_d6_l2_h3_t30K_s372001")), ["SUB", "NEG"])
        self.assertEqual(model_classes(_cfg(MIX6)), ["ADD", "SUB", "NEG"])

    def test_ntc_technique_owns_only_ntc_tags(self):
        ntc = next(t for t in TECHNIQUES if t.name == "neg_combiner_NTC")
        self.assertTrue(ntc.owns_tag("Algo:A2.NTC"))
        self.assertFalse(ntc.owns_tag("Algo:A2.MTC"))
        self.assertFalse(ntc.owns_tag("Algo:A2.STC"))

    def test_delivery_technique_owns_delivery_tags(self):
        d = next(t for t in TECHNIQUES if t.name == "combiner_delivery_route")
        self.assertTrue(d.owns_tag("Probe:DELIVERY.SUB=resatt"))
        self.assertFalse(d.owns_tag("Probe:A2.LINXFER=90"))


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestCoverageHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quanta_maths import load_maths_model_from_hf
        cls.model, cls.cfg = load_maths_model_from_hf(MIX6)

    def test_neg_combiner_tagged_on_mixed(self):
        from quanta_maths.maths_batch import tag_stc_nodes
        from QuantaMechInterp import UsefulNodeList
        # build a node list of the last-layer answer MLPs
        from QuantaMechInterp import UsefulNode
        nodes = UsefulNodeList()
        ll = self.cfg.n_layers - 1
        for k in range(1, self.cfg.n_digits):
            pos = int(self.cfg.an_to_position_name(k)[1:]) - 1
            nodes.nodes.append(UsefulNode(pos, ll, False, 0, []))
        added = tag_stc_nodes(self.model, self.cfg, nodes, cls="NEG")
        self.assertGreater(added, 0, "no NEG (NTC) combiner tagged on the mixed model")
        self.assertTrue(any(any(t.endswith(".NTC") for t in n.tags) for n in nodes.nodes))

    def test_delivery_route_add_vs_sub(self):
        from quanta_maths.maths_cascade import delivery_route
        self.assertEqual(delivery_route(self.model, self.cfg, "ADD"), "res")
        self.assertEqual(delivery_route(self.model, self.cfg, "SUB"), "resatt")
        self.assertEqual(delivery_route(self.model, self.cfg, "NEG"), "resatt")


if __name__ == "__main__":
    unittest.main()
