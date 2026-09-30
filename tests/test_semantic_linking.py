"""Semantic additions must preserve original spans and abstain on ambiguity."""

import copy
import unittest

from patientphex.semantic_linking import merge_neural_additions
from scripts.evaluate_semantic_linking import verify_upstream_partition


def candidate(offset=20, length=4, score=0.98, similarity=0.92, margin=0.05):
    return {"entity": {"offset": offset, "length": length, "identifier": "HP:0001250", "note": None}, "span_score": score, "similarity": similarity, "margin": margin, "source": "semantic"}


class SemanticLinkingTests(unittest.TestCase):
    def test_rejects_ner_model_fitted_with_outer_validation_or_b_labels(self):
        verify_upstream_partition({"training_document_ids": ["a"], "validation_document_ids": ["b"]}, {"b"}, {"a", "b"}, {"B"})
        for train_ids in (["a", "b"], ["a", "B"], ["a", "a"]):
            with self.subTest(train_ids=train_ids), self.assertRaises(ValueError):
                verify_upstream_partition({"training_document_ids": train_ids, "validation_document_ids": ["b"]}, {"b"}, {"a", "b"}, {"B"})

    def test_ambiguous_semantic_match_is_rejected_even_with_high_cosine(self):
        self.assertEqual(merge_neural_additions([], [candidate(similarity=0.99, margin=0.001)], 0.9, 0.85), [])

    def test_preserves_original_entity_and_does_not_mutate_inputs(self):
        base = [{"offset": 10, "length": 5, "identifier": "HP:1", "note": "NO"}]
        additions = [candidate(offset=10, length=9), candidate()]
        original = copy.deepcopy((base, additions))
        self.assertEqual(merge_neural_additions(base, additions, 0.9, 0.85), [base[0], additions[1]["entity"]])
        self.assertEqual((base, additions), original)

    def test_new_overlap_uses_joint_evidence_instead_of_longest_span(self):
        precise = candidate(offset=10, length=5, similarity=0.98)
        extended = candidate(offset=8, length=10, similarity=0.88)
        self.assertEqual(merge_neural_additions([], [extended, precise], 0.9, 0.85), [precise["entity"]])


if __name__ == "__main__":
    unittest.main()
