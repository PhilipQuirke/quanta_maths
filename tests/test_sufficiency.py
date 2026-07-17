"""Circuit-sufficiency harness tests (maths_sufficiency).

Offline: keep-set masks + random-keep-set sizing/composition. HF-gated: on the
best-understood addition model the keep-useful (mean-ablation) retention is high,
keep-random is ~0, and baseline is ~1 (the sufficiency + specificity signature).
"""
import os
import unittest

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_sufficiency import masks_from_set, random_keep_set

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
ADD5 = "add_d5_l2_h3_t15K_s372001"


class TestMasks(unittest.TestCase):
    def setUp(self):
        self.cfg = MathsConfig()
        self.cfg.set_model_names(ADD5)  # offline: n_layers=2, n_heads=3, n_ctx=19

    def test_masks_mark_only_kept_nodes(self):
        keep = {(5, 0, True, 1), (5, 0, False, 0)}  # a head + an MLP at P5,L0
        head, mlp = masks_from_set(self.cfg, keep)
        self.assertTrue(head[0][5, 1])
        self.assertFalse(head[0][5, 0])
        self.assertTrue(mlp[0][5])
        self.assertFalse(mlp[1][5])
        self.assertEqual(int(head[0].sum()), 1)
        self.assertEqual(int(mlp[0].sum()), 1)

    def test_random_keep_set_size_and_composition(self):
        import numpy as np
        rk = random_keep_set(self.cfg, 10, 4, np.random.default_rng(0))
        self.assertEqual(sum(1 for (_, _, ih, _) in rk if ih), 10)
        self.assertEqual(sum(1 for (_, _, ih, _) in rk if not ih), 4)
        head, mlp = masks_from_set(self.cfg, rk)   # must be valid mask indices
        self.assertEqual(int(sum(h.sum() for h in head)), 10)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestSufficiencyHF(unittest.TestCase):
    def test_addition_circuit_is_sufficient_and_specific(self):
        from quanta_maths import load_maths_model_from_hf, circuit_sufficiency
        model, cfg = load_maths_model_from_hf(ADD5)
        out = circuit_sufficiency(model, cfg, ADD5, n=120, n_random=2)
        r = out["classes"]["ADD"]
        self.assertGreater(r["baseline"], 0.99)             # harness/positive control
        self.assertLess(r["keep_random_mean"], 0.1)         # keeping random nodes fails
        self.assertGreater(r["keep_useful_mean"], 0.8)      # useful circuit largely sufficient
        self.assertGreater(r["keep_useful_mean"], r["keep_random_mean"])


if __name__ == "__main__":
    unittest.main()
