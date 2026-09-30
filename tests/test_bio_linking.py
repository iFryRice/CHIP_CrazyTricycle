"""BIO linking must retain known unmapped text and abstain on dense ambiguity."""

import copy
import unittest

from scripts.evaluate_bio_pilot import merge_entities


class BioLinkingTests(unittest.TestCase):
    def test_known_unmapped_is_retained_but_ambiguous_semantic_match_is_not(self):
        nil = {"entity": {"offset": 0, "length": 4, "identifier": "-1", "text": "text", "note": None}, "span_score": 0.95, "source": "known_unmapped", "similarity": 1.0, "margin": 1.0}
        uncertain = {"entity": {"offset": 10, "length": 4, "identifier": "HP:1"}, "span_score": 0.95, "source": "semantic", "similarity": 0.99, "margin": 0.001}
        self.assertEqual(merge_entities([], [nil, uncertain], 0.5, 0.9, 0.02), [nil["entity"]])

    def test_union_preserves_original_offsets_notes_and_input_objects(self):
        base = [{"offset": 10, "length": 4, "identifier": "HP:1", "note": "NO"}]
        incoming = [{"entity": {"offset": 8, "length": 10, "identifier": "HP:2", "note": None}, "span_score": 0.99, "source": "hpo_exact", "similarity": 1.0, "margin": 1.0}]
        original = copy.deepcopy((base, incoming))
        self.assertEqual(merge_entities(base, incoming, 0.5, 0.9, 0.02), base)
        self.assertEqual((base, incoming), original)


if __name__ == "__main__":
    unittest.main()
