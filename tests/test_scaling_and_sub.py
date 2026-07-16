"""Scaling (~10-digit) and subtraction robustness tests for the library.

Offline: sub_labels borrow cascade, multi-layer site resolution, layer helpers +
bounds guard. HF-gated: end-to-end load/probe/combiner on a 10-digit addition
model and a 6-digit subtraction model.
"""
import os
import unittest

import numpy as np

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_constants import MathsToken, MathsTask
from quanta_maths.maths_probe import (
    sub_labels, neg_labels, site_hook_and_pos, first_layer, last_layer,
)
from quanta_maths.maths_search_sub import sub_mtc_functions, neg_ntc_functions

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
ADD10 = "add_d10_l2_h3_t40K_s572091"
SUB6 = "sub_d6_l2_h3_t30K_s372001"
MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"


class TestSubLabels(unittest.TestCase):
    def test_borrow_cascade(self):
        # 52 - 47: units 2-7<0 -> ST=1 (borrow), SA=5; tens gets borrow-in SV=1.
        SA, ST, SV = sub_labels(52, 47, 2, operation=MathsToken.MINUS)
        self.assertEqual(SA[0], 5)
        self.assertEqual(ST[0], 1)
        self.assertEqual(SV[0], 0)
        self.assertEqual(SV[1], 1)

    def test_u_class_equal_digits(self):
        # 33 - 33: each digit equal -> ST == 2 (ambiguous 'U')
        SA, ST, SV = sub_labels(33, 33, 2, operation=MathsToken.MINUS)
        self.assertEqual(ST[0], 2)
        self.assertEqual(ST[1], 2)

    def test_no_borrow(self):
        SA, ST, SV = sub_labels(99, 11, 2, operation=MathsToken.MINUS)
        self.assertEqual(ST[0], 0)  # 9-1>0
        self.assertEqual(SV[0], 0)

    def test_addition_default_unchanged(self):
        SA, ST, SV = sub_labels(55, 55, 2)  # PLUS default
        self.assertEqual(ST[0], 1)  # carry


class TestMultiLayerSites(unittest.TestCase):
    def test_layer_helpers(self):
        cfg = MathsConfig(); cfg.set_model_names("mix_d10_l3_h4_t75K_s173289")
        self.assertEqual(cfg.n_layers, 3)
        self.assertEqual(first_layer(cfg), 0)
        self.assertEqual(last_layer(cfg), 2)

    def test_explicit_layer_reaches_last_block(self):
        # The bug this guards: legacy _L0/_L1 names never reach blocks.2 on a
        # 3-layer model. An explicit last_layer must.
        cfg = MathsConfig(); cfg.set_model_names("mix_d10_l3_h4_t75K_s173289")
        hook, _ = site_hook_and_pos(cfg, "ans", 3, layer=last_layer(cfg))
        self.assertEqual(hook, "blocks.2.hook_resid_post")

    def test_out_of_range_layer_raises(self):
        cfg = MathsConfig(); cfg.set_model_names("add_d5_l2_h3_t15K_s372001")
        with self.assertRaises(ValueError):
            site_hook_and_pos(cfg, "Dpn_L5", 3)  # only 2 layers
        with self.assertRaises(ValueError):
            site_hook_and_pos(cfg, "ans", 3, layer=9)

    def test_legacy_names_still_work_on_2layer(self):
        cfg = MathsConfig(); cfg.set_model_names("add_d5_l2_h3_t15K_s372001")
        hook, pos = site_hook_and_pos(cfg, "Dpn_L1", 2)
        self.assertEqual(hook, "blocks.1.hook_resid_post")


class TestSubMtcOffline(unittest.TestCase):
    def test_tag(self):
        self.assertEqual(sub_mtc_functions.tag(2), "A2.MTC")
        self.assertEqual(MathsTask.MTC_TAG.value, "MTC")

    def test_operation_is_minus(self):
        self.assertEqual(sub_mtc_functions.operation(), MathsToken.MINUS)


class TestNegLabels(unittest.TestCase):
    def test_neg_answer_digits_are_magnitude(self):
        # 100 - 201 = -101 ; emitted magnitude digits are 1,0,1 (units..up).
        SA, ST, SV = neg_labels(100, 201, 6)
        self.assertEqual([SA[n] for n in range(3)], [1, 0, 1])

    def test_neg_tricase_and_borrow(self):
        # 100 - 201: compute on D'-D = 201-100. units 1-0=1>0 -> NT=0 no borrow;
        # tens 0-0 -> NT=2 (U); hundreds 2-1 -> NT=0. No borrow into units.
        SA, ST, SV = neg_labels(100, 201, 6)
        self.assertEqual(ST[0], 0)
        self.assertEqual(ST[1], 2)
        self.assertEqual(SV[0], 0)

    def test_requires_a_less_than_b(self):
        with self.assertRaises(ValueError):
            neg_labels(500, 100, 6)  # a >= b is not a NEG question


class TestNegNtcOffline(unittest.TestCase):
    def test_tag(self):
        self.assertEqual(neg_ntc_functions.tag(2), "A2.NTC")
        self.assertEqual(MathsTask.NTC_TAG.value, "NTC")

    def test_operation_is_minus(self):
        self.assertEqual(neg_ntc_functions.operation(), MathsToken.MINUS)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestScalingHF(unittest.TestCase):
    def _acc(self, model, cfg, op, n=48, seed=0):
        import torch
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        from quanta_maths.maths_edge_patch import answer_positions
        g = np.random.default_rng(seed)
        lim = 10 ** cfg.n_digits
        hit = 0
        for _ in range(n):
            if op == MathsToken.MINUS:
                a, b = int(g.integers(0, lim)), int(g.integers(0, lim))
            else:
                a, b = int(g.integers(0, lim // 2)), int(g.integers(0, lim // 2))
            q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
            make_a_maths_question_and_answer(cfg, q, 0, a, b, op)
            q = q[0]
            with torch.no_grad():
                lg = model(q.unsqueeze(0))
            ap = answer_positions(cfg)
            pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            hit += int((pred == q[ap]).all())
        return hit / n

    def test_10digit_addition_loads_and_scales(self):
        from quanta_maths import load_maths_model_from_hf
        model, cfg = load_maths_model_from_hf(ADD10)
        self.assertEqual(cfg.n_digits, 10)
        self.assertEqual(cfg.n_ctx, 34)
        self.assertGreater(self._acc(model, cfg, MathsToken.PLUS), 0.95)

    def test_10digit_probe_at_last_layer(self):
        from quanta_maths import load_maths_model_from_hf
        from quanta_maths.maths_probe import collect_site_activations, probe_accuracy_with_null
        model, cfg = load_maths_model_from_hf(ADD10)
        rng = np.random.default_rng(0)
        acts, labs = collect_site_activations(model, cfg, 250, [5], ["ans"], rng,
                                              layer=last_layer(cfg))
        out = probe_accuracy_with_null(acts[("ans", 5)], labs["ST"][5], rng, n_perm=20)
        self.assertGreater(out["observed_acc"], 0.9)

    def test_6digit_subtraction_loads_and_combiner_causal(self):
        from quanta_maths import load_maths_model_from_hf
        from quanta_maths.maths_batch import _combiner_is_causal
        model, cfg = load_maths_model_from_hf(SUB6)
        self.assertEqual(cfg.perc_sub, 100)
        self.assertGreater(self._acc(model, cfg, MathsToken.MINUS), 0.9)
        ll = last_layer(cfg)
        causal = [k for k in range(1, cfg.n_digits)
                  if _combiner_is_causal(model, cfg,
                                         int(cfg.an_to_position_name(k)[1:]) - 1,
                                         k, ll, operation=MathsToken.MINUS)]
        self.assertGreater(len(causal), 0, "no subtraction borrow-combiner found")

    def test_6digit_subtraction_st_decodable(self):
        from quanta_maths import load_maths_model_from_hf
        from quanta_maths.maths_probe import collect_site_activations, probe_accuracy_with_null
        model, cfg = load_maths_model_from_hf(SUB6)
        rng = np.random.default_rng(0)
        acts, labs = collect_site_activations(model, cfg, 300, [3], ["ans"], rng,
                                              operation=MathsToken.MINUS,
                                              layer=last_layer(cfg))
        out = probe_accuracy_with_null(acts[("ans", 3)], labs["ST"][3], rng, n_perm=20)
        self.assertGreater(out["observed_acc"], 0.8)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestMixedModelHF(unittest.TestCase):
    """The mixed add/sub model is 3-layer with three question classes; the NEG
    library additions (neg_labels, neg_ntc) must hold on it."""

    def _emit_answer_str(self, model, cfg, a, b, op):
        import torch
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        from quanta_maths.maths_edge_patch import answer_positions
        q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(cfg, q, 0, a, b, op)
        q = q[0]
        ap = answer_positions(cfg)
        with torch.no_grad():
            pred = model(q.unsqueeze(0))[0, [p - 1 for p in ap]].argmax(-1)
        return pred.tolist()  # token ids at [SGN, A6..A0]

    def test_neg_labels_match_emitted_digits(self):
        from quanta_maths import load_maths_model_from_hf
        model, cfg = load_maths_model_from_hf(MIX6)
        # 123456 - 654321 = -530865 (NEG). The EMITTED digit is the combiner
        # output (base neg-diff SA minus the neg-borrow-in SV), mod 10 -- the
        # subtraction analog of addition's (SA + carry) % 10. SA alone is the
        # base ND sub-task, not the emitted digit.
        a, b = 123456, 654321
        toks = self._emit_answer_str(model, cfg, a, b, MathsToken.MINUS)
        digits = toks[1:]  # A6..A0 token ids == digit values
        SA, ST, SV = neg_labels(a, b, cfg.n_digits)
        emitted = [(SA[n] - SV[n]) % 10 for n in range(cfg.n_digits - 1, -1, -1)]  # A5..A0
        self.assertEqual(digits[1:], emitted)  # A6 is 0 (7-digit answer); compare A5..A0

    def test_neg_combiner_causal_on_mixed(self):
        from quanta_maths import load_maths_model_from_hf
        from quanta_maths.maths_batch import _combiner_is_causal
        model, cfg = load_maths_model_from_hf(MIX6)
        ll = last_layer(cfg)
        causal = [k for k in range(1, cfg.n_digits)
                  if _combiner_is_causal(model, cfg,
                                         int(cfg.an_to_position_name(k)[1:]) - 1,
                                         k, ll, cls="NEG")]
        self.assertGreater(len(causal), 0, "no NEG neg-borrow-combiner found")


if __name__ == "__main__":
    unittest.main()
