"""Tests for quanta_maths.maths_batch (Stage 5 local batch tagging).

Offline: the ACCURATE_MODELS scope constant. HF-gated: an end-to-end 1-model
batch that must produce >=1 STC tag and write valid JSON locally.
"""
import json
import os
import tempfile
import unittest

from quanta_maths.maths_batch import ACCURATE_MODELS, run_batch

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"


class TestBatchOffline(unittest.TestCase):
    def test_accurate_models_scope(self):
        # Stage-5 decision: 4 accurate add models + accurate ins1_mix.
        self.assertEqual(len(ACCURATE_MODELS), 5)
        self.assertIn("add_d5_l2_h3_t15K_s372001", ACCURATE_MODELS)
        self.assertIn("ins1_mix_d6_l3_h4_t40K_s372001", ACCURATE_MODELS)


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestBatchHF(unittest.TestCase):
    def test_single_model_batch_writes_tags(self):
        with tempfile.TemporaryDirectory() as d:
            summary = run_batch(
                models=["add_d5_l2_h3_t15K_s372001"],
                local_dir=d, do_linxfer=False)
            s = summary["add_d5_l2_h3_t15K_s372001"]
            self.assertGreater(s["stc_tags"], 0)
            self.assertGreater(s["n_nodes"], 0)

            maths_path = os.path.join(d, "add_d5_l2_h3_t15K_s372001_maths.json")
            self.assertTrue(os.path.exists(maths_path))
            nodes = json.load(open(maths_path))
            stc = [n for n in nodes if any("STC" in t for t in n["tags"])]
            self.assertGreater(len(stc), 0)
            # STC tags must sit on MLP nodes (is_head False)
            for n in stc:
                self.assertFalse(n["is_head"])


if __name__ == "__main__":
    unittest.main()
