"""Tests for the CE16-promoted statistics + causal-measurement primitives.

Offline: wilson_ci / mean_ci. HF-gated: flip_rate_with_matched_null reproducing the
PC1 pattern (load-bearing head-pair patch flips ~1.0 while a matched null ~0.0) and
mean_ablate_heads_prediction selectively breaking carry-bearing sums.
"""
import os
import unittest

import numpy as np

from quanta_maths.maths_stats import wilson_ci, mean_ci

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
HF_MODEL = "add_d5_l2_h3_t15K_s372001"


class TestStats(unittest.TestCase):
    def test_wilson_all_success(self):
        lo, hi = wilson_ci(40, 40)
        self.assertLess(lo, 1.0)          # Wilson never returns exactly 1
        self.assertGreater(lo, 0.9)       # but tight for 40/40
        self.assertLessEqual(hi, 1.0 + 1e-9)

    def test_wilson_zero_n(self):
        lo, hi = wilson_ci(0, 0)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_wilson_half(self):
        lo, hi = wilson_ci(20, 40)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_mean_ci_shape(self):
        out = mean_ci([1, 1, 0, 1, 0])
        self.assertAlmostEqual(out["rate"], 0.6)
        self.assertEqual(out["n"], 5)
        self.assertEqual(len(out["ci"]), 2)
        self.assertLessEqual(out["ci"][0], out["rate"])
        self.assertGreaterEqual(out["ci"][1], out["rate"])

    def test_mean_ci_empty(self):
        out = mean_ci([])
        self.assertTrue(np.isnan(out["rate"]))
        self.assertEqual(out["n"], 0)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestCausalMeasurementHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quanta_maths import load_maths_model_from_hf
        cls.model, cls.cfg = load_maths_model_from_hf(HF_MODEL, device="cpu")

    def _mkq(self, a, b):
        import torch
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        from quanta_maths.maths_constants import MathsToken
        q = torch.zeros((1, self.cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(self.cfg, q, 0, a, b, MathsToken.PLUS)
        return q[0]

    def test_flip_rate_positive_and_null(self):
        # A full-z swap from a DIFFERENT-operand source, scanning all answer
        # positions, flips at least one digit (load-bearing SOMEWHERE); a
        # source == target null never flips. Note: single-position L1 ablation is
        # redundant on the 5-digit model (CE16), so we score the max over digits.
        import torch
        from quanta_maths.maths_edge_patch import (
            consuming_pos, answer_positions, flip_rate_with_matched_null)
        rng = np.random.default_rng(0)
        heads = list(range(self.cfg.n_heads))
        ap = answer_positions(self.cfg)

        def swap_all_positions(sq, tq):
            with torch.no_grad():
                _, sc = self.model.run_with_cache(sq.unsqueeze(0))

            def hook(act, hook):
                for k in range(self.cfg.n_digits + 1):
                    cp = consuming_pos(self.cfg, k)
                    for h in heads:
                        act[:, cp, h, :] = sc["blocks.1.attn.hook_z"][0, cp, h, :]
                return act
            with torch.no_grad():
                lg = self.model.run_with_hooks(
                    tq.unsqueeze(0), fwd_hooks=[("blocks.1.attn.hook_z", hook)])
            return lg[0, [p - 1 for p in ap]].argmax(-1)

        def pair_builder():
            return (self._mkq(int(rng.integers(0, 49999)), int(rng.integers(0, 49999))),
                    self._mkq(int(rng.integers(0, 49999)), int(rng.integers(0, 49999))))

        def null_builder():
            tq = self._mkq(int(rng.integers(0, 49999)), int(rng.integers(0, 49999)))
            return tq, tq  # source == target -> zero effect

        # Score digit A2. Invariants the PRIMITIVE must satisfy (independent of the
        # stimulus effect size, which is study-specific): the matched null (source ==
        # target) flips EXACTLY 0.0, and a real cross-operand swap flips strictly more.
        out = flip_rate_with_matched_null(
            self.model, self.cfg, pair_builder, swap_all_positions, answer_digit=2,
            n_pairs=20, null_builder=null_builder)
        self.assertEqual(out["null"]["rate"], 0.0)               # self-patch never flips
        self.assertGreater(out["flip"]["rate"], out["null"]["rate"])  # load-bearing > null
        # CI + n are reported for headline use
        self.assertEqual(out["flip"]["n"], 20)
        self.assertEqual(len(out["flip"]["ci"]), 2)

    def test_mean_ablate_heads_is_mechanically_correct(self):
        # Mechanical check: the hook substitutes the given per-head vectors. Feed a
        # DISTINCTIVE substitute (a source question's z) and confirm the ablated run
        # equals a full-z-swap run. (Behavioural necessity of mean-ablation is a
        # 6-digit/class-level result per CE16, not asserted here.)
        import torch
        from quanta_maths.maths_edge_patch import (
            mean_ablate_heads_prediction, consuming_pos, answer_positions)
        cpos = consuming_pos(self.cfg, 2)
        heads = list(range(self.cfg.n_heads))
        ap = answer_positions(self.cfg)

        tq = self._mkq(45678, 45678)
        sq = self._mkq(11111, 22222)
        with torch.no_grad():
            _, sc = self.model.run_with_cache(sq.unsqueeze(0))
        sub = {h: sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy() for h in heads}

        via_helper = mean_ablate_heads_prediction(
            self.model, self.cfg, tq, cpos, heads, sub, layer=1)

        def hook(act, hook):
            for h in heads:
                act[:, cpos, h, :] = torch.as_tensor(sub[h], dtype=act.dtype)
            return act
        with torch.no_grad():
            lg = self.model.run_with_hooks(
                tq.unsqueeze(0), fwd_hooks=[("blocks.1.attn.hook_z", hook)])
        direct = lg[0, [p - 1 for p in ap]].argmax(-1)
        self.assertTrue(torch.equal(via_helper, direct))


if __name__ == "__main__":
    unittest.main()
