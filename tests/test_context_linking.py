"""Structural association features and label-source guard behavior."""

import copy
import unittest

from patientphex.context_linking import ContextTreeLinker, structural_samples


def fixture():
    return {"pmc_id": "a", "patient": [{"patient_id": "P1", "mention": [{"offset": 0, "length": 9, "text": "Patient 1"}]}],
            "full_text": [{"offset": 0, "text": "Patient 1 has weakness.", "section_type": "CASE", "type": "paragraph"}],
            "entities": [], "association": [{"patient_id": "P1", "phenotype": ["HP:1"]}]}


class ContextLinkingTests(unittest.TestCase):
    def test_identity_free_features_and_negation(self):
        document = fixture()
        entity = {"offset": 14, "length": 8, "text": "weakness", "identifier": "HP:1", "note": None}
        before = copy.deepcopy(document)
        rows = structural_samples(document, [entity])
        self.assertEqual(len(rows), 1)
        features = rows[0][2]
        self.assertEqual(features["joint:any:explicit_sentence"], 1)
        self.assertFalse(any("HP:" in key or "context=" in key or "heading=" in key for key in features))
        self.assertEqual(structural_samples(document, [{**entity, "note": "NO"}]), [])
        self.assertEqual(document, before)

    def test_upstream_selection_leakage_is_rejected(self):
        provenance = {"kind": "out_of_fold", "by_document": {"a": {"training_document_ids": [], "label_source_document_ids": ["heldout"], "selection_document_ids": ["heldout"]}}}
        with self.assertRaisesRegex(ValueError, "leakage"):
            ContextTreeLinker().fit([fixture()], {"a": []}, provenance=provenance, forbidden_document_ids={"heldout"})

    def test_empty_training_and_blind_prediction_contract(self):
        model = ContextTreeLinker().fit([fixture()], {"a": []}, provenance={"kind": "fixed_knowledge_base", "uses_training_labels": False}, forbidden_document_ids={"b"})
        with self.assertRaisesRegex(ValueError, "blind"):
            model.predict_scores(fixture(), [])
        self.assertEqual(model.predict_scores({**fixture(), "association": []}, []), [])


if __name__ == "__main__":
    unittest.main()
