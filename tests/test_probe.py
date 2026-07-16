"""Tests for quanta_maths.maths_probe (offline, no network).

Each capability gets a positive control (structured signal -> detected) and a
negative control (noise / shuffled labels -> chance), matching the study-harness
discipline.
"""
import unittest

import numpy as np

from quanta_maths.maths_config import MathsConfig
from quanta_maths.maths_probe import (
    sub_labels, TASK_CHANCE, ALL_SITES, site_hook_and_pos,
    probe_accuracy_with_null, class_mean_subspace, principal_angles_deg,
    real_dft_basis, freq1_plane_share, unique_linear_share, angular_order_stat,
    pc_plane_coords, wraparound_ratio, participation_ratio, dft_permutation_pvalues,
)


class TestSubLabels(unittest.TestCase):
    def test_carry_cascade(self):
        # 55 + 55 = 110. digit0: 5+5=10 -> SA0=0, ST0=1(carry), SV0=0(no carry in)
        SA, ST, SV = sub_labels(55, 55, 2)
        self.assertEqual(SA[0], 0)
        self.assertEqual(ST[0], 1)
        self.assertEqual(SV[0], 0)
        # digit1: 5+5=10, plus carry-in 1 -> SA1=0, ST1=1, SV1=1(carry in)
        self.assertEqual(SV[1], 1)
        self.assertEqual(ST[1], 1)

    def test_u_class(self):
        # 45 + 54: digit0 5+4=9 -> ST==2 (U); digit1 4+5=9 -> ST==2
        SA, ST, SV = sub_labels(45, 54, 2)
        self.assertEqual(ST[0], 2)
        self.assertEqual(ST[1], 2)

    def test_no_carry(self):
        SA, ST, SV = sub_labels(11, 11, 2)
        self.assertEqual(ST[0], 0)
        self.assertEqual(SV[0], 0)


class TestSiteAlgebra(unittest.TestCase):
    def test_sites_match_config_helpers(self):
        cfg = MathsConfig(); cfg.set_model_names("add_d5_l2_h3_t15K_s372001")
        nd = cfg.n_digits
        for n in range(nd):
            self.assertEqual(site_hook_and_pos(cfg, "Dpn_L0", n)[1], 2 * nd - n)
            self.assertEqual(site_hook_and_pos(cfg, "Dn_L0", n)[1], nd - 1 - n)
            self.assertEqual(site_hook_and_pos(cfg, "ans_L0", n)[1], cfg.n_ctx - 2 - n)

    def test_all_sites_resolve(self):
        cfg = MathsConfig(); cfg.set_model_names("add_d6_l2_h3_t20K_s173289")
        for s in ALL_SITES:
            hook, pos = site_hook_and_pos(cfg, s, 2)
            self.assertTrue(hook.startswith("blocks."))
            self.assertGreaterEqual(pos, 0)


class TestProbeControls(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_positive_control_linearly_separable(self):
        # 3-class signal on a low-dim subspace + noise -> probe well above chance/null.
        n, d = 600, 20
        y = self.rng.integers(0, 3, size=n)
        centers = self.rng.standard_normal((3, d)) * 3
        X = centers[y] + self.rng.standard_normal((n, d))
        out = probe_accuracy_with_null(X, y, self.rng, n_perm=50)
        self.assertGreater(out["observed_acc"], TASK_CHANCE["ST"] + 0.15)
        self.assertLess(out["null_p"], 0.05)

    def test_negative_control_pure_noise(self):
        n, d = 600, 20
        y = self.rng.integers(0, 3, size=n)
        X = self.rng.standard_normal((n, d))  # no signal
        out = probe_accuracy_with_null(X, y, self.rng, n_perm=50)
        # observed acc near chance and NOT significant vs null
        self.assertLess(out["observed_acc"], 0.5)
        self.assertGreater(out["null_p"], 0.05)

    def test_cross_val_probe_accuracy(self):
        from quanta_maths.maths_probe import cross_val_probe_accuracy
        y = self.rng.integers(0, 2, size=200)
        X_sig = y[:, None] * 3 + self.rng.standard_normal((200, 6))
        X_noise = self.rng.standard_normal((200, 6))
        self.assertGreater(cross_val_probe_accuracy(X_sig, y), 0.9)
        self.assertLess(cross_val_probe_accuracy(X_noise, y), 0.65)


class TestSubspaceGeometry(unittest.TestCase):
    def test_identical_subspaces_zero_angle(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, size=300)
        centers = rng.standard_normal((3, 10)) * 3
        X = centers[y] + rng.standard_normal((300, 10)) * 0.1
        B = class_mean_subspace(X, y)
        ang = principal_angles_deg(B, B)
        self.assertTrue(np.all(ang < 1.0))


class TestDftGeometry(unittest.TestCase):
    def test_basis_orthonormal(self):
        Q, names = real_dft_basis()
        self.assertEqual(len(names), 9)
        g = Q.T @ Q
        self.assertTrue(np.allclose(g, np.eye(9), atol=1e-8))

    def test_positive_control_planted_circle(self):
        # digits on a circle in a 2-plane -> high freq1 share, significant.
        rng = np.random.default_rng(2)
        ang = 2 * np.pi * np.arange(10) / 10
        M = np.zeros((10, 8))
        M[:, 0] = np.cos(ang); M[:, 1] = np.sin(ang)
        M += rng.standard_normal((10, 8)) * 0.01
        self.assertGreater(freq1_plane_share(M), 0.8)
        stats = dft_permutation_pvalues(M, rng, n_perm=500)
        self.assertLess(stats["freq1_p"], 0.05)
        # 9 sits next to 0 on a circle
        self.assertLess(wraparound_ratio(pc_plane_coords(M)), 2.0)

    def test_negative_control_isotropic_noise(self):
        rng = np.random.default_rng(3)
        M = rng.standard_normal((10, 30))
        stats = dft_permutation_pvalues(M, rng, n_perm=500)
        self.assertGreater(stats["freq1_p"], 0.05)
        self.assertGreater(participation_ratio(M), 5.0)  # high-dim, not low-rank


if __name__ == "__main__":
    unittest.main()
