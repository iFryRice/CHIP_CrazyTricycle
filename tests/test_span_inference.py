"""Verify that inference data roles cannot silently leak training answers."""

import unittest

from scripts.predict_span_ner import select_documents


class InferenceRoleTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"training_document_ids": ["train"], "validation_document_ids": ["val"]}
        self.documents = [
            {"pmc_id": "train", "entities": [{"text": "gold"}], "association": [{"patient_id": "P1"}]},
            {"pmc_id": "val", "entities": [{"text": "secret"}], "association": []},
        ]

    def test_validation_is_blind_without_changing_source(self):
        selected = select_documents(self.documents, self.manifest, "validation", "validation")
        self.assertEqual([document["pmc_id"] for document in selected], ["val"])
        self.assertEqual(selected[0]["entities"], [])
        self.assertEqual(self.documents[1]["entities"], [{"text": "secret"}])

    def test_held_out_role_rejects_training_documents(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_documents(self.documents, self.manifest, "all", "validation")

    def test_target_rejects_labels_and_training_overlap(self):
        with self.assertRaisesRegex(ValueError, "answer"):
            select_documents(self.documents[1:], self.manifest, "all", "target")
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_documents([{"pmc_id": "train", "entities": [], "association": []}], self.manifest, "all", "target")

    def test_subset_requires_complete_document_coverage(self):
        with self.assertRaisesRegex(ValueError, "every document"):
            select_documents(self.documents[:1], self.manifest, "validation", "validation")

    def test_final_target_excludes_model_selection_documents(self):
        with self.assertRaisesRegex(ValueError, "model-selection"):
            select_documents([{"pmc_id": "val", "entities": [], "association": []}], self.manifest, "all", "target")


if __name__ == "__main__":
    unittest.main()
