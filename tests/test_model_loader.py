"""Tests for quanta_maths.maths_model_loader.

Offline tests (config building, untrained control, state-dict stripping) run always.
The end-to-end HF load + positive/negative control test hits HuggingFace and is
gated behind the RUN_HF_TESTS=1 environment variable so the default `pytest tests`
stays fast and network-free.
"""
import os
import unittest

import torch

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_model_loader import (
    build_maths_config,
    load_maths_model_from_hf,
    make_untrained_control,
    _strip_state_dict,
    DEFAULT_HF_REPO,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
HF_MODEL = "add_d5_l2_h3_t15K_s372001"  # primary study model


class TestModelLoaderOffline(unittest.TestCase):
    """No network required."""

    def test_build_config_no_json_parses_name(self):
        cfg = build_maths_config(HF_MODEL, use_train_json=False)
        self.assertEqual(cfg.n_digits, 5)
        self.assertEqual(cfg.n_layers, 2)
        self.assertEqual(cfg.n_heads, 3)
        self.assertEqual(cfg.training_seed, 372001)
        self.assertEqual(cfg.perc_add, 100)
        # 5-digit addition context: 2*5+2 question + 5+2 answer = 19
        self.assertEqual(cfg.n_ctx, 19)

    def test_strip_state_dict_unwraps_model_key(self):
        wrapped = {"model": {"embed.W_E": torch.zeros(1)}}
        self.assertIn("embed.W_E", _strip_state_dict(wrapped))
        flat = {"embed.W_E": torch.zeros(1)}
        self.assertIs(_strip_state_dict(flat), flat)

    def test_make_untrained_control_matches_architecture(self):
        cfg = build_maths_config(HF_MODEL, use_train_json=False)
        ctrl = make_untrained_control(cfg, device="cpu")
        self.assertEqual(ctrl.cfg.n_layers, cfg.n_layers)
        self.assertEqual(ctrl.cfg.n_heads, cfg.n_heads)
        self.assertEqual(ctrl.cfg.n_ctx, cfg.n_ctx)
        self.assertEqual(str(ctrl.cfg.device), "cpu")

    def test_default_repo_constant(self):
        self.assertEqual(DEFAULT_HF_REPO, "PhilipQuirke/VerifiedArithmetic")


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestModelLoaderHF(unittest.TestCase):
    """End-to-end load with positive + negative controls (needs network)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.cfg = load_maths_model_from_hf(HF_MODEL, device="cpu")

    def _random_additions(self, cfg, n=32, seed=0):
        from quanta_maths.maths_utilities import make_a_maths_question_and_answer
        from quanta_maths.maths_constants import MathsToken
        g = torch.Generator().manual_seed(seed)
        lim = 10 ** cfg.n_digits // 2
        qs = torch.zeros((n, cfg.n_ctx), dtype=torch.int64)
        for i in range(n):
            a = int(torch.randint(0, lim, (1,), generator=g))
            b = int(torch.randint(0, lim, (1,), generator=g))
            make_a_maths_question_and_answer(cfg, qs, i, a, b, MathsToken.PLUS)
        return qs

    def _accuracy(self, model, cfg, qs):
        na = cfg.n_digits + 2
        ans_pos = list(range(cfg.n_ctx - na, cfg.n_ctx))
        with torch.no_grad():
            logits = model(qs)
        pred = logits[:, [p - 1 for p in ans_pos]].argmax(-1)
        tgt = qs[:, ans_pos]
        return (pred == tgt).all(dim=1).float().mean().item()

    def test_config_loaded_from_train_json(self):
        # _train.json supplies d_head/d_model and the final loss.
        self.assertEqual(self.cfg.d_model, 510)
        self.assertEqual(self.cfg.d_head, 170)
        self.assertGreater(self.cfg.final_loss, 0.0)
        self.assertLess(self.cfg.final_loss, 1e-6)  # accurate model

    def test_positive_control_trained_model_accurate(self):
        qs = self._random_additions(self.cfg)
        acc = self._accuracy(self.model, self.cfg, qs)
        self.assertGreater(acc, 0.99, f"trained model accuracy {acc}")

    def test_negative_control_untrained_fails(self):
        ctrl = make_untrained_control(self.cfg, device="cpu")
        qs = self._random_additions(self.cfg)
        acc = self._accuracy(ctrl, self.cfg, qs)
        self.assertLess(acc, 0.5, f"untrained control unexpectedly accurate {acc}")


if __name__ == "__main__":
    unittest.main()
