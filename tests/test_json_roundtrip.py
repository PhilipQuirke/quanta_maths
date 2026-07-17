"""Full-cycle JSON persistence tests for the technique pipeline.

Verifies that ALL registered techniques (add / sub / mixed) update the per-model
analysis JSON in a CONSISTENT manner, and that a full node-list JSON survives the
store -> (HF transport) -> download -> rehydrate cycle:

  * consistency: every technique's tag major survives its TARGET file's save
    filter (Algo -> features.json; Probe/behavior -> behaviors.json), so a tag a
    technique writes is actually persisted where it belongs.
  * round-trip: save_nodes -> (bytes copied, as HF upload/download would) ->
    load_nodes rehydrates identical structure + tags, is idempotent, and keeps the
    two-file split (features = Algo only; behaviors = everything else).

HF-gated tests additionally rehydrate a REAL published JSON and run the true
download -> tag -> save -> reload cycle via ``update_model`` (dry-run, no upload).
"""
import json
import os
import shutil
import tempfile
import unittest

from QuantaMechInterp import UsefulNode, UsefulNodeList

from quanta_maths.maths_hf_update import (
    TECHNIQUES, _SAVE_MAJOR, BEHAVIORS_FILE, FEATURES_FILE,
)

RUN_HF = os.environ.get("RUN_HF_TESTS") == "1"
MIX6 = "ins1_mix_d6_l3_h4_t40K_s372001"
HF_REPO = "PhilipQuirke/VerifiedArithmetic"

# A representative tag each technique OWNS (major:minor), used to check the tag it
# writes lands + survives in its target file. Update if a technique's tag changes.
SAMPLE_OWNED = {
    "add_combiner_STC": "Algo:A2.STC",
    "sub_combiner_MTC": "Algo:A2.MTC",
    "neg_combiner_NTC": "Algo:A2.NTC",
    "operand_linear_transfer_LINXFER": "Probe:A1.LINXFER=90",
    "carry_temporal_finalization_CARRY": "Probe:A5.CARRYLAYER=2",
    "combiner_delivery_route": "Probe:DELIVERY.SUB=resatt",
}


def _roundtrip(nodes: UsefulNodeList, path: str, major: str) -> UsefulNodeList:
    """save_nodes(major) -> copy the bytes (HF upload/download) -> load_nodes."""
    nodes.save_nodes(path, major)
    transported = path + ".downloaded"
    shutil.copyfile(path, transported)          # == HF upload then download
    reloaded = UsefulNodeList()
    reloaded.load_nodes(transported)
    return reloaded


class TestTechniqueConsistency(unittest.TestCase):
    def test_every_technique_has_a_sample_tag(self):
        # keeps SAMPLE_OWNED in sync with the registry (new techniques must be added)
        self.assertEqual(sorted(SAMPLE_OWNED), sorted(t.name for t in TECHNIQUES))

    def test_tag_major_survives_target_save_filter(self):
        """The tag a technique writes must (a) be owned by it, (b) survive its
        target file's save filter, and (c) route Algo->features, Probe->behaviors."""
        for t in TECHNIQUES:
            tag = SAMPLE_OWNED[t.name]
            self.assertTrue(t.owns_tag(tag), f"{t.name} does not own {tag}")
            major = tag.split(":")[0]
            # (b) survives the save filter of its target file
            save_major = _SAVE_MAJOR[t.target]
            kept = UsefulNode(18, 2, False, 0, [tag]).to_dict(save_major)["tags"]
            self.assertIn(tag, kept, f"{t.name}: {tag} dropped by {t.target} filter")
            # (c) routing: Algo tags -> features.json; everything else -> behaviors.json
            if major == "Algo":
                self.assertEqual(t.target, FEATURES_FILE, f"{t.name} Algo not in features")
            else:
                self.assertEqual(t.target, BEHAVIORS_FILE, f"{t.name} non-Algo not in behaviors")

    def test_owned_tag_predicates_are_disjoint(self):
        """No two techniques claim the same tag (clear_owned must not clobber)."""
        for name, tag in SAMPLE_OWNED.items():
            owners = [t.name for t in TECHNIQUES if t.owns_tag(tag)]
            self.assertEqual(owners, [name], f"{tag} owned by {owners}, expected [{name}]")


class TestFullCycleRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # answer-position last-layer MLP nodes (combiners) + an operand head
        self.positions = [(18, 2, False, 0), (19, 2, False, 0), (9, 0, True, 1)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _behaviors_list(self):
        nodes = UsefulNodeList()
        for (p, l, h, n) in self.positions:
            node = UsefulNode(p, l, h, n, [])
            node.add_tag("Fail%", "3")
            node.add_tag("Impact", "A2")
            # mixed-thread Probe tags (all three Probe sub-namespaces)
            node.add_tag("Probe", "A1.LINXFER=90")
            node.add_tag("Probe", "A5.CARRYLAYER=2")
            node.add_tag("Probe", "DELIVERY.SUB=resatt")
            nodes.nodes.append(node)
        return nodes

    def _features_list(self):
        nodes = UsefulNodeList()
        for (p, l, h, n) in self.positions:
            node = UsefulNode(p, l, h, n, [])
            for tag in ("A2.STC", "A2.MTC", "A2.NTC"):  # add / sub / neg combiners
                node.add_tag("Algo", tag)
            node.add_tag("Attn", "P6=32")   # a non-Algo tag that MUST be filtered out
            nodes.nodes.append(node)
        return nodes

    def test_features_roundtrip_keeps_only_algo(self):
        feats = self._features_list()
        reloaded = _roundtrip(feats, os.path.join(self.tmp, FEATURES_FILE), _SAVE_MAJOR[FEATURES_FILE])
        self.assertEqual(len(reloaded.nodes), len(self.positions))
        for node in reloaded.nodes:
            self.assertTrue(node.tags, "node lost all tags")
            self.assertTrue(all(t.startswith("Algo:") for t in node.tags),
                            f"non-Algo tag leaked into features.json: {node.tags}")
            for want in ("Algo:A2.STC", "Algo:A2.MTC", "Algo:A2.NTC"):
                self.assertIn(want, node.tags)
            self.assertNotIn("Attn:P6=32", node.tags)  # filtered by Algo-save

    def test_behaviors_roundtrip_keeps_all_and_no_algo(self):
        beh = self._behaviors_list()
        reloaded = _roundtrip(beh, os.path.join(self.tmp, BEHAVIORS_FILE), _SAVE_MAJOR[BEHAVIORS_FILE])
        for node in reloaded.nodes:
            self.assertIn("Fail%:3", node.tags)
            self.assertIn("Impact:A2", node.tags)
            self.assertIn("Probe:A1.LINXFER=90", node.tags)   # numeric suffix preserved
            self.assertIn("Probe:A5.CARRYLAYER=2", node.tags)
            self.assertIn("Probe:DELIVERY.SUB=resatt", node.tags)
            self.assertFalse(any(t.startswith("Algo:") for t in node.tags))

    def test_structure_preserved(self):
        beh = self._behaviors_list()
        reloaded = _roundtrip(beh, os.path.join(self.tmp, BEHAVIORS_FILE), "")
        got = {(n.position, n.layer, n.is_head, n.num) for n in reloaded.nodes}
        self.assertEqual(got, set(self.positions))

    def test_rehydrate_is_idempotent(self):
        """Loading the downloaded JSON twice must not duplicate tags (add_tag dedup)."""
        beh = self._behaviors_list()
        path = os.path.join(self.tmp, BEHAVIORS_FILE)
        beh.save_nodes(path, "")
        reloaded = UsefulNodeList()
        reloaded.load_nodes(path)
        counts1 = {n.name(): len(n.tags) for n in reloaded.nodes}
        reloaded.load_nodes(path)                # load again onto the same list
        counts2 = {n.name(): len(n.tags) for n in reloaded.nodes}
        self.assertEqual(counts1, counts2)

    def test_json_is_plain_serializable(self):
        """The saved file is valid JSON of the documented node schema."""
        feats = self._features_list()
        path = os.path.join(self.tmp, FEATURES_FILE)
        feats.save_nodes(path, "Algo")
        data = json.load(open(path))
        self.assertIsInstance(data, list)
        for d in data:
            self.assertEqual(set(d), {"position", "layer", "is_head", "num", "tags"})


@unittest.skipUnless(RUN_HF, "set RUN_HF_TESTS=1 to run HuggingFace integration tests")
class TestRealHFRehydrate(unittest.TestCase):
    def test_download_published_json_rehydrates(self):
        """A published node-list JSON on HF loads with our loader (the 'downloaded by
        another user, rehydrated' leg)."""
        from QuantaMechInterp.model_train_json import download_huggingface_json
        tmp = tempfile.mkdtemp()
        try:
            for suffix, want_major in [("maths", "Algo:"), ("behavior", None)]:
                data = download_huggingface_json(HF_REPO, f"{MIX6}_{suffix}.json")
                path = os.path.join(tmp, f"{suffix}.json")
                json.dump(data, open(path, "w"))
                nodes = UsefulNodeList()
                nodes.load_nodes(path)
                self.assertGreater(len(nodes.nodes), 0)
                if want_major:
                    self.assertTrue(any(t.startswith(want_major)
                                        for node in nodes.nodes for t in node.tags))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_update_model_full_cycle_dryrun(self):
        """True cycle: download real JSON -> run combiner techniques -> save ->
        reload the saved files (rehydrate). No upload (dry-run)."""
        from quanta_maths.maths_hf_update import update_model
        combiners = [t for t in TECHNIQUES if t.name.endswith(("STC", "MTC", "NTC"))]
        tmp = tempfile.mkdtemp()
        try:
            r = update_model(MIX6, techniques=combiners, dry_run=True, upload=False,
                             work_dir=tmp)
            if r["error"]:
                self.skipTest(f"analysis repo unavailable: {r['error']}")
            self.assertFalse(r["uploaded"])
            feat = UsefulNodeList(); feat.load_nodes(r["local_updated"][FEATURES_FILE])
            beh = UsefulNodeList(); beh.load_nodes(r["local_updated"][BEHAVIORS_FILE])
            self.assertGreater(len(feat.nodes), 0)
            self.assertGreater(len(beh.nodes), 0)
            # combiner tags round-tripped into features.json (Algo-only)
            self.assertTrue(any(t.split(":")[1].split(".")[-1] in {"STC", "MTC", "NTC"}
                                for node in feat.nodes for t in node.tags
                                if t.startswith("Algo:")))
            self.assertTrue(all(t.startswith("Algo:") for node in feat.nodes for t in node.tags))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
