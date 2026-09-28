"""Regression tests for split integrity and answer isolation."""

import unittest

from patientphex.__main__ import blind
from patientphex.data import document_folds


class DataTests(unittest.TestCase):
    def test_folds_are_disjoint_complete_and_reproducible(self):
        documents = [
            {"pmc_id": str(index), "patient": [None] * (1 + index % 3)}
            for index in range(80)
        ]
        folds = document_folds(documents)
        self.assertEqual(folds, document_folds(list(reversed(documents))))
        self.assertEqual([len(fold) for fold in folds], [16] * 5)
        identifiers = [d["pmc_id"] for fold in folds for d in fold]
        self.assertEqual(len(set(identifiers)), 80)
        singles = [sum(len(d["patient"]) == 1 for d in fold) for fold in folds]
        self.assertLessEqual(max(singles) - min(singles), 1)

    def test_blind_removes_answers_without_mutating_original(self):
        document = {"pmc_id": "test", "entities": [{"text": "gold"}], "association": [{"patient_id": "P1"}], "patient": [{"patient_id": "P1"}]}
        prediction_input = blind(document)
        self.assertEqual(prediction_input["entities"], [])
        self.assertEqual(prediction_input["association"], [])
        self.assertTrue(document["entities"])
        self.assertEqual(prediction_input["patient"], document["patient"])


if __name__ == "__main__":
    unittest.main()
