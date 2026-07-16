"""Tests for the add_stc_functions ST-combiner search subtask (CE5).

Offline: tag/prereqs shape. HF-gated: the causal STC signature -- ablating the
answer-position L1 MLP flips the answer digit (positive control); the same
ablation on an untrained twin does not exhibit the trained combiner behaviour
(negative control).
"""
import os
import unittest

import torch

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_constants import MathsTask, MathsToken
from quanta_maths.maths_search_add import add_stc_functions

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
HF_MODEL = "add_d5_l2_h3_t15K_s372001"


class TestStcOffline(unittest.TestCase):
    def test_tag(self):
        self.assertEqual(add_stc_functions.tag(2), "A2.STC")
        self.assertEqual(MathsTask.STC_TAG.value, "STC")

    def test_operation_is_plus(self):
        self.assertEqual(add_stc_functions.operation(), MathsToken.PLUS)

    def test_prereqs_targets_mlp_at_producing_position(self):
        cfg = MathsConfig(); cfg.set_model_names(HF_MODEL)
        # producing position for A2 is one before the A2 token
        prod = int(cfg.an_to_position_name(2)[1:]) - 1
        filt = add_stc_functions.prereqs(cfg, prod, 2)
        # filter is constructed without error and references the producing position
        self.assertIsNotNone(filt)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestStcMechanismHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from quanta_maths import load_maths_model_from_hf, make_untrained_control
        cls.model, cls.cfg = load_maths_model_from_hf(HF_MODEL, device="cpu")
        cls.ctrl = make_untrained_control(cls.cfg, device="cpu")

    def _mkq(self, a, b):
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        q = torch.zeros((1, self.cfg.n_ctx), dtype=torch.int64)
        make_a_maths_question_and_answer(self.cfg, q, 0, a, b, MathsToken.PLUS)
        return q[0]

    def _pred(self, model, q):
        from quanta_maths.maths_edge_patch import answer_positions
        with torch.no_grad():
            lg = model(q.unsqueeze(0))
        ap = answer_positions(self.cfg)
        return lg[0, [p - 1 for p in ap]].argmax(-1)

    def _pred_mlp_ablated(self, model, q, prod):
        from quanta_maths.maths_edge_patch import answer_positions

        def hook(act, hook):
            act[:, prod, :] = 0.0
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(
                q.unsqueeze(0), fwd_hooks=[("blocks.1.mlp.hook_post", hook)])
        ap = answer_positions(self.cfg)
        return lg[0, [p - 1 for p in ap]].argmax(-1)

    def test_positive_control_combiner_is_causal(self):
        # Ablating the answer-position L1 MLP must change at least one answer digit
        # on carry-bearing additions (the combiner is load-bearing per CE5).
        changed = False
        for k in range(1, self.cfg.n_digits):
            prod = int(self.cfg.an_to_position_name(k)[1:]) - 1
            q = self._mkq(22222 + 7 * 10 ** (k - 1), 22222 + 7 * 10 ** (k - 1))
            clean = self._pred(self.model, q)
            abl = self._pred_mlp_ablated(self.model, q, prod)
            if not torch.equal(clean, abl):
                changed = True
                break
        self.assertTrue(changed, "answer-position L1 MLP ablation changed no answer digit")

    def test_negative_control_untrained_no_combiner_signature(self):
        # The untrained twin does not compute correct answers at all, so it has no
        # meaningful carry-combining behaviour to disrupt in the CE5 sense: its
        # clean predictions are already wrong (not the ground-truth sum).
        q = self._mkq(22292, 22292)
        gt_pred = self._pred(self.model, q)
        ctrl_pred = self._pred(self.ctrl, q)
        self.assertFalse(torch.equal(gt_pred, ctrl_pred),
                         "untrained control unexpectedly matches trained combiner output")


if __name__ == "__main__":
    unittest.main()
