"""Tests for the HF analysis-JSON refresh framework (quanta_maths.maths_hf_update).

Offline: technique registry structure, applicability filters, idempotent
clear_owned, save-major mapping. HF-gated: a dry-run over one model that must
produce tags, back up originals, keep behaviors.json Algo-free, and upload nothing.
"""
import json
import os
import tempfile
import types
import unittest

from quanta_maths.maths_hf_update import (
    Technique, TECHNIQUES, techniques_for, register_technique,
    ordered_analysis_models, _op_group, _n_digits, _verify_roundtrip,
    upload_mechanism_docs, BEHAVIORS_FILE, FEATURES_FILE, MECHANISM_FILE, _SAVE_MAJOR,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"


class _FakeNode:
    def __init__(self, tags):
        self.tags = list(tags)


class _FakeNodes:
    def __init__(self, tags_per_node):
        self.nodes = [_FakeNode(t) for t in tags_per_node]


def _cfg(perc_add, perc_sub):
    c = types.SimpleNamespace()
    c.perc_add = perc_add
    c.perc_sub = perc_sub
    return c


class TestRegistry(unittest.TestCase):
    def test_all_techniques_well_formed(self):
        self.assertGreaterEqual(len(TECHNIQUES), 3)
        for t in TECHNIQUES:
            self.assertIn(t.target, (BEHAVIORS_FILE, FEATURES_FILE))
            self.assertTrue(callable(t.applies_to))
            self.assertTrue(callable(t.owns_tag))
            self.assertTrue(callable(t.run))

    def test_save_major_mapping(self):
        self.assertEqual(_SAVE_MAJOR[FEATURES_FILE], "Algo")
        self.assertEqual(_SAVE_MAJOR[BEHAVIORS_FILE], "")

    def test_mechanism_file_is_markdown_not_a_nodelist(self):
        # mechanism.md is uploaded to HF but is NOT a node-list JSON, so it must
        # not be in the save-major (round-trip) map.
        self.assertEqual(MECHANISM_FILE, "mechanism.md")
        self.assertNotIn(MECHANISM_FILE, _SAVE_MAJOR)

    def test_applicability_addition_only(self):
        names = {t.name for t in techniques_for(_cfg(100, 0))}
        self.assertIn("add_combiner_STC", names)
        self.assertIn("operand_linear_transfer_LINXFER", names)
        self.assertNotIn("sub_combiner_MTC", names)

    def test_applicability_subtraction_only(self):
        names = {t.name for t in techniques_for(_cfg(0, 100))}
        # subtraction covers positive-answer (MTC) AND negative-answer (NTC)
        # combiners, plus the class-agnostic delivery-route technique.
        self.assertEqual(names, {"sub_combiner_MTC", "neg_combiner_NTC",
                                 "combiner_delivery_route"})
        self.assertNotIn("add_combiner_STC", names)

    def test_applicability_mixed(self):
        names = {t.name for t in techniques_for(_cfg(34, 66))}
        self.assertIn("add_combiner_STC", names)
        self.assertIn("sub_combiner_MTC", names)
        self.assertIn("operand_linear_transfer_LINXFER", names)


class TestIdempotency(unittest.TestCase):
    def test_clear_owned_removes_only_owned(self):
        stc = [t for t in TECHNIQUES if t.name == "add_combiner_STC"][0]
        nodes = _FakeNodes([
            ["Fail%:1", "Algo:A4.STC", "Impact:A4"],   # STC owned + others
            ["Algo:A3.ST", "Algo:A2.STC"],             # ST (not owned) + STC (owned)
            ["Attn:P2=51"],                            # none owned
        ])
        removed = stc.clear_owned(nodes)
        self.assertEqual(removed, 2)  # two *.STC tags removed
        remaining = [t for n in nodes.nodes for t in n.tags]
        self.assertNotIn("Algo:A4.STC", remaining)
        self.assertNotIn("Algo:A2.STC", remaining)
        self.assertIn("Algo:A3.ST", remaining)   # ST preserved
        self.assertIn("Fail%:1", remaining)       # behavior tag preserved

    def test_linxfer_owns_probe_tags(self):
        lx = [t for t in TECHNIQUES if t.name == "operand_linear_transfer_LINXFER"][0]
        self.assertTrue(lx.owns_tag("Probe:A1.LINXFER=89"))
        self.assertFalse(lx.owns_tag("Algo:A4.STC"))
        self.assertFalse(lx.owns_tag("Fail%:3"))

    def test_is_present_skip_logic(self):
        stc = [t for t in TECHNIQUES if t.name == "add_combiner_STC"][0]
        self.assertTrue(stc.is_present(_FakeNodes([["Algo:A4.STC"], ["Fail%:1"]])))
        self.assertFalse(stc.is_present(_FakeNodes([["Algo:A3.ST"], ["Fail%:1"]])))


class TestOrdering(unittest.TestCase):
    def test_op_group_and_digits(self):
        self.assertEqual(_op_group("add_d5_l2_h3_t15K_s372001"), "add")
        self.assertEqual(_op_group("sub_d6_l2_h3_t30K_s372001"), "sub")
        self.assertEqual(_op_group("mix_d6_l3_h4_t40K_s372001"), "mix")
        self.assertEqual(_op_group("ins1_mix_d6_l3_h4_t40K_s372001"), "mix")
        self.assertEqual(_n_digits("add_d10_l2_h3_t40K_s572091"), 10)

    def test_ordered_add_sub_mix_small_to_large(self):
        models = ["mix_d10_l3_h4_t75K_s173289", "add_d6_l2_h3_t20K_s173289",
                  "add_d5_l2_h3_t15K_s372001", "sub_d10_l2_h3_t75K_s173289",
                  "sub_d6_l2_h3_t30K_s372001", "ins1_mix_d6_l3_h4_t40K_s372001"]
        out = ordered_analysis_models(models)
        self.assertEqual(out, [
            "add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289",
            "sub_d6_l2_h3_t30K_s372001", "sub_d10_l2_h3_t75K_s173289",
            "ins1_mix_d6_l3_h4_t40K_s372001", "mix_d10_l3_h4_t75K_s173289"])


class TestRoundtrip(unittest.TestCase):
    def test_verify_roundtrip_ok_and_detects_corruption(self):
        from QuantaMechInterp import UsefulNodeList, NodeLocation
        with tempfile.TemporaryDirectory() as d:
            nl = UsefulNodeList()
            nl.add_node_tag(NodeLocation(14, 1, False, 0), "Algo", "A2.STC")
            nl.add_node_tag(NodeLocation(9, 0, True, 1), "Algo", "A1.ST")
            p = os.path.join(d, FEATURES_FILE)
            nl.save_nodes(p, "Algo")
            self.assertTrue(_verify_roundtrip(p, "Algo"))
            # corrupt the file -> roundtrip must fail (or raise, caught as failure)
            with open(p, "w") as f:
                f.write("not valid json")
            try:
                ok = _verify_roundtrip(p, "Algo")
            except Exception:
                ok = False
            self.assertFalse(ok)


class TestRegisterTechnique(unittest.TestCase):
    def test_register_and_restore(self):
        n0 = len(TECHNIQUES)
        t = Technique(name="_tmp_test", target=FEATURES_FILE,
                      applies_to=lambda cfg: True, owns_tag=lambda s: False,
                      run=lambda m, c, nodes: 0)
        register_technique(t)
        try:
            self.assertEqual(len(TECHNIQUES), n0 + 1)
            self.assertIn(t, techniques_for(_cfg(100, 0)))
        finally:
            TECHNIQUES.remove(t)
        self.assertEqual(len(TECHNIQUES), n0)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestDryRunHF(unittest.TestCase):
    def test_dry_run_add_model(self):
        from quanta_maths.maths_hf_update import update_model
        name = "add_d6_l2_h3_t20K_s173289"
        with tempfile.TemporaryDirectory() as d:
            r = update_model(name, dry_run=True, work_dir=d)
            self.assertIsNone(r["error"])
            self.assertFalse(r["uploaded"])                 # dry run never uploads
            self.assertIn("add_combiner_STC", r["applicable"])
            self.assertNotIn("sub_combiner_MTC", r["applicable"])
            # originals backed up, updated written
            self.assertTrue(os.path.exists(os.path.join(d, name, "original", FEATURES_FILE)))
            feat = json.load(open(r["local_updated"][FEATURES_FILE]))
            beh = json.load(open(r["local_updated"][BEHAVIORS_FILE]))
            # features has combiner tags; behaviors has NO Algo pollution
            self.assertTrue(any(t.endswith(".STC") for n in feat for t in n["tags"]))
            self.assertFalse(any(t.startswith("Algo:") for n in beh for t in n["tags"]))

    def test_batch_manifest_dry_run(self):
        from quanta_maths.maths_hf_update import update_models
        with tempfile.TemporaryDirectory() as d:
            man = update_models(models=["add_d5_l2_h3_t15K_s372001"],
                                dry_run=True, work_dir=d)
            self.assertTrue(man["dry_run"])
            self.assertFalse(man["uploaded_any"])
            self.assertEqual(man["n_models"], 1)
            self.assertEqual(man["n_errors"], 0)
            self.assertTrue(os.path.exists(man["manifest_path"]))

    def test_mechanism_docs_dry_run_generates_md(self):
        with tempfile.TemporaryDirectory() as d:
            man = upload_mechanism_docs(models=["add_d5_l2_h3_t15K_s372001"],
                                        dry_run=True, work_dir=d)
            self.assertTrue(man["dry_run"])
            self.assertEqual(man["n_uploaded"], 0)      # dry run uploads nothing
            self.assertEqual(man["n_errors"], 0)
            mdpath = os.path.join(d, "add_d5_l2_h3_t15K_s372001", "updated", MECHANISM_FILE)
            self.assertTrue(os.path.exists(mdpath))     # generated locally for inspection
            self.assertGreater(os.path.getsize(mdpath), 0)


if __name__ == "__main__":
    unittest.main()
