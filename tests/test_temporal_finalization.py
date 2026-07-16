"""Tests for the temporal-finalization (Battery TF) library module.

Offline: the deterministic 9-free carry-chain stimulus builder + position algebra
(no model / no HF needed). HF-gated: run_temporal_finalization smoke on the small
5-digit model (reproduces the LAZY-propagation / eager-local verdict shape).
"""
import os
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_temporal_finalization import (
    build_carry_chain,
    dprime_pos,
    equals_pos,
    full_input_pos,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
HF_MODEL = "add_d5_l2_h3_t15K_s372001"


def _cfg(name: str = "add_d5_l2_h3_t15K_s372001") -> MathsConfig:
    cfg = MathsConfig()
    cfg.set_model_names(name)          # parses n_digits/layers/... offline (no HF)
    return cfg


class TestStimulusBuilder(unittest.TestCase):
    """The carry chain is a deterministic function of its factors — verify exactly."""

    def test_carry_top_is_make_carry_and_run_intact(self):
        cfg = _cfg()
        for _ in range(300):
            a, b, ct, lmc = build_carry_chain(cfg, deciding_digit=1, make_carry=True, run_intact=True)
            self.assertEqual(ct, 1)      # make-carry + intact run -> carry ripples to top
            self.assertEqual(lmc, 1)

    def test_run_break_decorrelates_top_from_local(self):
        cfg = _cfg()
        for _ in range(300):
            a, b, ct, lmc = build_carry_chain(cfg, deciding_digit=1, make_carry=True, run_intact=False)
            self.assertEqual(ct, 0)      # broken run blocks propagation -> no top carry
            self.assertEqual(lmc, 1)     # ...but the LOCAL make-carry still happened

    def test_no_make_carry_gives_no_top_carry(self):
        cfg = _cfg()
        for _ in range(300):
            a, b, ct, lmc = build_carry_chain(cfg, deciding_digit=1, make_carry=False, run_intact=True)
            self.assertEqual(ct, 0)
            self.assertEqual(lmc, 0)

    def test_nine_free_stimuli_have_no_9_token(self):
        cfg = _cfg()
        for mk in (True, False):
            for ri in (True, False):
                for _ in range(100):
                    a, b, _ct, _lmc = build_carry_chain(cfg, 1, mk, ri, nine_free=True)
                    self.assertNotIn("9", str(a))
                    self.assertNotIn("9", str(b))

    def test_position_algebra(self):
        cfg = _cfg()
        # D'_0 (units of operand B) = full input availability; = is the next token.
        self.assertEqual(full_input_pos(cfg), dprime_pos(cfg, 0))
        self.assertEqual(equals_pos(cfg), 2 * cfg.n_digits + 1)
        # D'_n tokens are MSD-first: higher digit -> earlier token.
        self.assertLess(dprime_pos(cfg, cfg.n_digits - 1), dprime_pos(cfg, 0))


class TestTechniqueRegistration(unittest.TestCase):
    """The TF technique must be registered so the 40-model HF runner picks it up."""

    def test_carry_technique_registered_and_targets_behaviors(self):
        from quanta_maths.maths_hf_update import TECHNIQUES, BEHAVIORS_FILE
        techs = {t.name: t for t in TECHNIQUES}
        self.assertIn("carry_temporal_finalization_CARRY", techs)
        t = techs["carry_temporal_finalization_CARRY"]
        self.assertEqual(t.target, BEHAVIORS_FILE)          # Probe: -> behaviors, not features
        self.assertTrue(t.owns_tag("Probe:A5.CARRYLAYER=1"))
        self.assertTrue(t.owns_tag("Probe:A5.CARRYDEFER=2"))
        self.assertFalse(t.owns_tag("Probe:A1.LINXFER=89"))  # does not steal other tags
        self.assertFalse(t.owns_tag("Algo:A5.STC"))
        cfg_add = _cfg("add_d5_l2_h3_t15K_s372001")
        self.assertTrue(t.applies_to(cfg_add))               # runs on addition models


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run the HF-gated smoke test")
class TestTemporalFinalizationHF(unittest.TestCase):
    def test_smoke_lazy_verdict_shape(self):
        from quanta_maths.maths_temporal_finalization import run_temporal_finalization
        r = run_temporal_finalization(HF_MODEL, n_q=200)
        v = r["verdict"]
        self.assertIn("read", v)
        self.assertEqual(r["n_layers"], 2)
        # propagated carry should not be canonical at layer 0 (early output)...
        self.assertIsNone(r["propagated_onset"][0])
        # ...and should appear at the final layer at/after full-input availability.
        self.assertIsNotNone(v["propagated_onset_pos"])
        self.assertGreaterEqual(v["propagated_onset_pos"], v["full_input_pos"])

    def test_tagger_emits_carry_tags(self):
        import os
        from quanta_maths import load_maths_model_from_hf
        from quanta_maths.maths_batch import _download_behavior_nodes
        from quanta_maths.maths_temporal_finalization import tag_carry_finalization_nodes
        d = "results/hf-update-test"
        os.makedirs(d, exist_ok=True)
        model, cfg = load_maths_model_from_hf(HF_MODEL, device="cpu")
        nodes, _ = _download_behavior_nodes(HF_MODEL, "PhilipQuirke/VerifiedArithmetic", d)
        n = tag_carry_finalization_nodes(model, cfg, nodes, n_q=200)
        self.assertGreaterEqual(n, 1)
        tags = [t for node in nodes.nodes for t in node.tags if ".CARRY" in t]
        self.assertTrue(any("CARRYLAYER" in t for t in tags))


if __name__ == "__main__":
    unittest.main()
