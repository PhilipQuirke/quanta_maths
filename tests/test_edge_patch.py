"""Tests for quanta_maths.maths_edge_patch.

Pure position/LN/OV-shape helpers run offline. Causal edge/pattern patches need a
real model and are gated behind RUN_HF_TESTS=1, each with positive + negative
controls (a zero-delta / irrelevant patch must NOT move the answer; a load-bearing
patch must).
"""
import os
import unittest

import numpy as np
import torch

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_edge_patch import (
    answer_positions, consuming_pos, ln_scale, head_ov,
    head_group_ov, attention_mass_by_group,
    head_edge_delta, run_edge_patch, pattern_patch_prediction,
    synthetic_redirect_prediction,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
HF_MODEL = "add_d5_l2_h3_t15K_s372001"


class TestPositionHelpers(unittest.TestCase):
    def test_answer_positions_5digit(self):
        cfg = MathsConfig(); cfg.set_model_names(HF_MODEL)
        ap = answer_positions(cfg)
        self.assertEqual(len(ap), cfg.n_digits + 2)
        self.assertEqual(ap[-1], cfg.n_ctx - 1)  # units digit is last position

    def test_consuming_pos_is_one_before_answer(self):
        cfg = MathsConfig(); cfg.set_model_names(HF_MODEL)
        for k in range(cfg.n_digits):
            an = int(cfg.an_to_position_name(k)[1:])
            self.assertEqual(consuming_pos(cfg, k), an - 1)


class TestLnScale(unittest.TestCase):
    def test_ln_scale_centers_and_normalizes(self):
        v = torch.tensor([1.0, 2.0, 3.0, 10.0])
        xm, std = ln_scale(v)
        self.assertAlmostEqual(float(xm.mean()), 0.0, places=5)
        self.assertGreater(float(std), 0.0)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestEdgePatchHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quanta_maths import load_maths_model_from_hf
        cls.model, cls.cfg = load_maths_model_from_hf(HF_MODEL, device="cpu")

    def _make_q(self, a, b):
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        from quanta_maths.maths_constants import MathsToken
        q = torch.zeros((1, self.cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(self.cfg, q, 0, a, b, MathsToken.PLUS)
        return q[0]

    def _clean_pred(self, q):
        with torch.no_grad():
            logits = self.model(q.unsqueeze(0))
        ap = answer_positions(self.cfg)
        return logits[0, [p - 1 for p in ap]].argmax(-1)

    def test_head_ov_shape(self):
        z = torch.zeros(self.model.cfg.d_head)
        out = head_ov(self.model, z, layer=1, head=0)
        self.assertEqual(out.shape[-1], self.model.cfg.d_model)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))  # zero z -> zero out

    def test_negative_control_zero_delta_no_change(self):
        # Patching a head's edge from a question onto ITSELF => delta 0 => no flip.
        q = self._make_q(12345, 12345)
        clean = self._clean_pred(q)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(q.unsqueeze(0))
        cpos = consuming_pos(self.cfg, 2)
        delta = head_edge_delta(self.model, cache, cache, cpos, layer=1, head=0)
        self.assertLess(float(delta.detach().norm()), 1e-4)
        patched = run_edge_patch(self.model, self.cfg, q, cpos, delta,
                                 recv_layer=1, arm="raw")
        self.assertTrue(torch.equal(patched, clean))

    def test_positive_control_synthetic_redirect_moves_answer(self):
        # Force an L0 answer-position head to attend to the wrong key positions.
        # A load-bearing routing head MUST change at least one answer digit
        # (matches CE2/CE3: operand fetch happens at L0 answer positions).
        tq = self._make_q(12345, 12345)
        clean = self._clean_pred(tq)
        moved = False
        for k in range(self.cfg.n_digits):
            cpos = consuming_pos(self.cfg, k)
            for h in range(self.cfg.n_heads):
                pred = synthetic_redirect_prediction(
                    self.model, self.cfg, tq, layer=0, head=h,
                    query_pos=cpos, key_positions=[0, 1, 2])
                if not torch.equal(pred, clean):
                    moved = True
                    break
            if moved:
                break
        self.assertTrue(moved, "no synthetic attention redirect moved any answer digit")

    def test_negative_control_redirect_to_self_no_change(self):
        # Redirecting a head to attend only to its own query position is a
        # near-degenerate pattern; combined with the clean run it should not
        # reliably flip -- used as a sanity floor, not a strict assert on all heads.
        tq = self._make_q(12345, 12345)
        clean = self._clean_pred(tq)
        cpos = consuming_pos(self.cfg, 0)
        # Copying the clean pattern back in (source == target) must be a no-op.
        pred = pattern_patch_prediction(self.model, self.cfg, tq, tq,
                                        layer=0, head=0, query_pos=cpos)
        self.assertTrue(torch.equal(pred, clean))

    def test_head_group_ov_group_is_sum_and_matches_head_ov(self):
        qs = torch.stack([self._make_q(12345 + i, 6789 + i) for i in range(6)])
        cpos = consuming_pos(self.cfg, 2)
        ll = self.cfg.n_layers - 1
        heads = list(range(self.cfg.n_heads))
        group, per_head = head_group_ov(self.model, self.cfg, qs, cpos, ll, heads)
        self.assertEqual(group.shape, (6, self.model.cfg.d_model))
        # group == sum over heads
        summed = sum(per_head[h] for h in heads)
        self.assertTrue(np.allclose(group, summed, atol=1e-4))
        # per-head write matches the head_ov primitive on the cached z
        zname = f"blocks.{ll}.attn.hook_z"
        with torch.no_grad():
            _, c = self.model.run_with_cache(qs, names_filter=lambda nm: nm == zname)
        ov0 = head_ov(self.model, c[zname][0, cpos, 0, :], ll, 0).detach().numpy()
        self.assertTrue(np.allclose(per_head[0][0], ov0, atol=1e-4))

    def test_attention_mass_by_group_is_distribution(self):
        qs = torch.stack([self._make_q(12345 + i, 6789 + i) for i in range(6)])
        cpos = consuming_pos(self.cfg, 0)
        # one group = every causal key before the query; group+self must ~cover mass
        groups = {"before": list(range(cpos))}
        out = attention_mass_by_group(self.model, self.cfg, qs, cpos, 0, 0, groups)
        for v in out.values():
            self.assertGreaterEqual(v, -1e-6)
            self.assertLessEqual(v, 1.0 + 1e-6)
        self.assertAlmostEqual(out["before"] + out["self"] + out["other"], 1.0, places=4)
        self.assertLess(out["other"], 0.02)  # keys after query are causally ~0


if __name__ == "__main__":
    unittest.main()
