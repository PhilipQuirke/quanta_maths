"""Deep-cascade stimulus + combiner-delivery sweep tests (maths_cascade).

Offline: the cascade builder produces a genuine depth-k cascade (the true answer
digit A_k differs with the deciding carry) and respects the class sign. HF-gated:
on the mixed model, last-layer attention delivers the resolved borrow for SUB/NEG
but not the carry for ADD, carry/borrow-specifically (deciding-matched null ~0),
and an untrained control does not deliver.
"""
import os
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_cascade import (
    make_cascade_operands, cascade_answer_digit, combiner_delivery_flip,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"


class TestCascadeOperands(unittest.TestCase):
    def setUp(self):
        self.cfg = MathsConfig()
        self.cfg.set_model_names(MIX6)  # offline: parses n_digits=6, n_ctx=22

    def test_add_cascade_reaches_depth(self):
        for k in (2, 3, 4):
            a1, b1 = make_cascade_operands(self.cfg, "ADD", k, 1)
            a0, b0 = make_cascade_operands(self.cfg, "ADD", k, 0)
            d1 = cascade_answer_digit("ADD", a1, b1, k, self.cfg.n_digits)
            d0 = cascade_answer_digit("ADD", a0, b0, k, self.cfg.n_digits)
            self.assertNotEqual(d1, d0, f"ADD depth {k}: carry did not reach A_{k}")

    def test_sub_is_positive_and_cascades(self):
        for k in (2, 3, 4):
            a1, b1 = make_cascade_operands(self.cfg, "SUB", k, 1)
            a0, b0 = make_cascade_operands(self.cfg, "SUB", k, 0)
            self.assertGreater(a1, b1)  # D > D' (positive-answer subtraction)
            self.assertGreater(a0, b0)
            self.assertNotEqual(cascade_answer_digit("SUB", a1, b1, k, self.cfg.n_digits),
                                cascade_answer_digit("SUB", a0, b0, k, self.cfg.n_digits))

    def test_neg_is_negative_and_cascades(self):
        for k in (2, 3, 4):
            a1, b1 = make_cascade_operands(self.cfg, "NEG", k, 1)
            a0, b0 = make_cascade_operands(self.cfg, "NEG", k, 0)
            self.assertLess(a1, b1)  # D < D' (negative-answer subtraction)
            self.assertLess(a0, b0)
            self.assertNotEqual(cascade_answer_digit("NEG", a1, b1, k, self.cfg.n_digits),
                                cascade_answer_digit("NEG", a0, b0, k, self.cfg.n_digits))

    def test_null_variant_is_noncascading_but_distinct(self):
        # both carry_in=0 variants have the SAME answer digit (no cascade) but
        # DIFFERENT operands (a proper deciding-matched null).
        for cls in ("ADD", "SUB", "NEG"):
            a0, b0 = make_cascade_operands(self.cfg, cls, 3, 0, variant=0)
            a1, b1 = make_cascade_operands(self.cfg, cls, 3, 0, variant=1)
            self.assertNotEqual((a0, b0), (a1, b1))
            self.assertEqual(cascade_answer_digit(cls, a0, b0, 3, self.cfg.n_digits),
                             cascade_answer_digit(cls, a1, b1, 3, self.cfg.n_digits))

    def test_bad_read_digit_raises(self):
        with self.assertRaises(ValueError):
            make_cascade_operands(self.cfg, "ADD", 0, 1)  # read_digit must exceed deciding
        with self.assertRaises(ValueError):
            make_cascade_operands(self.cfg, "XYZ", 2, 1)  # unknown class


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestDeliveryHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quanta_maths import load_maths_model_from_hf
        cls.model, cls.cfg = load_maths_model_from_hf(MIX6)

    def test_sub_neg_attn_delivers_add_does_not(self):
        for k in (2, 3):
            add = combiner_delivery_flip(self.model, self.cfg, "ADD", k, "lastlayer_attn")
            sub = combiner_delivery_flip(self.model, self.cfg, "SUB", k, "lastlayer_attn")
            neg = combiner_delivery_flip(self.model, self.cfg, "NEG", k, "lastlayer_attn")
            for r in (add, sub, neg):
                self.assertTrue(r["stimulus_valid"])
                self.assertLess(r["null"]["rate"], 0.2)          # deciding-matched null ~0
            self.assertLess(add["flip"]["rate"], 0.2)             # ADD: attention does NOT deliver
            self.assertGreater(sub["flip"]["rate"], 0.8)          # SUB: attention delivers
            self.assertGreater(neg["flip"]["rate"], 0.8)          # NEG: attention delivers

    def test_full_resid_is_causal_positive_control(self):
        r = combiner_delivery_flip(self.model, self.cfg, "SUB", 3, "full_resid")
        self.assertGreater(r["flip"]["rate"], 0.8)
        self.assertLess(r["null"]["rate"], 0.2)

    def test_untrained_control_does_not_deliver(self):
        from quanta_maths import make_untrained_control
        ctrl = make_untrained_control(self.cfg)
        r = combiner_delivery_flip(ctrl, self.cfg, "SUB", 2, "lastlayer_attn")
        self.assertLess(r["flip"]["rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
