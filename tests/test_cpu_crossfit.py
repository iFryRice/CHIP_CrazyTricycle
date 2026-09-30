"""Nested candidate isolation and experiment promotion behavior."""

import copy
import unittest
from unittest.mock import patch

from scripts.optimize_cpu_association import assert_partition, crossfit_candidates, from_scores, promotion_checks


class CrossfitTests(unittest.TestCase):
    def test_candidate_fitter_never_sees_own_or_outer_validation_labels(self):
        documents = [{"pmc_id": str(i), "patient": [{}], "entities": [i], "association": [i]} for i in range(8)]
        original = copy.deepcopy(documents)

        class SpyExtractor:
            def __init__(self, ontology, mode):
                self.sources = set()

            def fit(self, docs):
                self.sources = {d["pmc_id"] for d in docs}
                return self

            def predict(self, doc):
                if doc["pmc_id"] in self.sources or doc["entities"] or doc["association"]:
                    raise AssertionError("Candidate generation received held-out labels.")
                return []

        with patch("scripts.optimize_cpu_association.EntityExtractor", SpyExtractor):
            candidates, provenance = crossfit_candidates(documents, None, {"outer", "B"})
        self.assertEqual(set(candidates), {str(i) for i in range(8)})
        for identifier, record in provenance["by_document"].items():
            self.assertNotIn(identifier, record["label_source_document_ids"])
            self.assertFalse(set(record["label_source_document_ids"]) & {"outer", "B"})
            self.assertEqual(len(record["label_source_document_ids"]), 6)
        self.assertEqual(documents, original)

    def test_rejects_duplicate_missing_and_forbidden_fold_members(self):
        docs = [{"pmc_id": "a"}, {"pmc_id": "b"}]
        for folds, forbidden in [([[docs[0]], [docs[0]]], set()), ([[docs[0]], []], set()), ([[docs[0]], [docs[1]]], {"b"})]:
            with self.subTest(folds=folds, forbidden=forbidden), self.assertRaises(ValueError):
                assert_partition(docs, folds, forbidden)

    def test_scores_preserve_empty_patients_and_threshold_boundary(self):
        doc = {"pmc_id": "a", "pmid": "1", "patient": [{"patient_id": "p"}, {"patient_id": "q"}]}
        pred = from_scores(doc, [], [{"patient_id": "p", "concept": "HP:0001250", "score": 0.3}], 0.3)
        self.assertEqual(pred["association"], [{"patient_id": "p", "phenotype": ["HP:0001250"]}, {"patient_id": "q", "phenotype": []}])

    def test_total_gain_alone_cannot_promote_inconsistent_candidate(self):
        base = {"score": 0.6, "mention": {}, "document": {}, "association_micro": {"precision": 0.7, "f1": 0.5}, "association_macro": {"f1": 0.5}}
        candidate = copy.deepcopy(base)
        candidate["score"] += 0.02
        candidate["association_micro"]["f1"] += 0.05
        candidate["association_macro"]["f1"] += 0.03
        checks = promotion_checks(base, candidate, [0.1, 0.1, -0.01, -0.01, -0.01], {"minimum_gain": 0.002, "minimum_nonworse_folds": 4, "maximum_precision_loss": 0.05})
        self.assertTrue(checks["minimum_gain"])
        self.assertFalse(checks["fold_consistency"])


if __name__ == "__main__":
    unittest.main()
