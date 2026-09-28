"""Association regressions for patient scope and the official output contract."""

import copy
import unittest

from patientphex.association import AssociationModel


def make_document():
    first = "Patient 1 has seizures. She has short stature."
    second = "Patient 2 has hearing loss. He has no seizures."
    offset = len(first) + 10
    return {
        "pmc_id": "synthetic",
        "patient": [
            {"patient_id": "O1", "mention": [{"text": "Patient 1", "offset": 0, "length": 9}]},
            {"patient_id": "O2", "mention": [{"text": "Patient 2", "offset": offset, "length": 9}]},
        ],
        "full_text": [
            {"section_type": "CASE", "type": "paragraph", "offset": 0, "text": first},
            {"section_type": "CASE", "type": "paragraph", "offset": offset, "text": second},
        ],
        "entities": [
            {"identifier": "HP:0001250", "offset": first.index("seizures"), "length": 8, "text": "seizures", "note": None},
            {"identifier": "HP:0004322", "offset": first.index("short stature"), "length": 13, "text": "short stature", "note": None},
            {"identifier": "HP:0000365", "offset": offset + second.index("hearing loss"), "length": 12, "text": "hearing loss", "note": None},
            {"identifier": "HP:0001250", "offset": offset + second.index("seizures"), "length": 8, "text": "seizures", "note": "NO"},
        ],
        "association": [
            {"patient_id": "O1", "phenotype": ["HP:0001250", "HP:0004322"]},
            {"patient_id": "O2", "phenotype": ["HP:0000365"]},
        ],
    }


class AssociationTests(unittest.TestCase):
    def test_nearest_respects_patient_sections_and_negation(self):
        document = make_document()
        actual = AssociationModel("nearest").predict(document, document["entities"])
        self.assertEqual(actual, document["association"])

    def test_compound_unmapped_and_duplicate_contract(self):
        document = make_document()
        document["patient"] = document["patient"][:1]
        entities = copy.deepcopy(document["entities"][:1])
        entities[0]["identifier"] = "HP:0001250;HP:0004322"
        entities.append({**entities[0], "identifier": "-1", "text": "original unmapped text"})
        entities.append(copy.deepcopy(entities[0]))
        output = AssociationModel("nearest").predict(document, entities)
        self.assertEqual(set(output[0]["phenotype"]), {"HP:0001250", "HP:0004322", "original unmapped text"})
        self.assertEqual(len(output[0]["phenotype"]), 3)

    def test_preserves_empty_target_patients(self):
        document = make_document()
        self.assertEqual(AssociationModel("nearest").predict(document, []), [
            {"patient_id": "O1", "phenotype": []}, {"patient_id": "O2", "phenotype": []},
        ])

    def test_learned_uses_only_training_labels_and_does_not_mutate(self):
        document = make_document()
        source = copy.deepcopy(document)
        model = AssociationModel().fit([document], [document["entities"]])
        first = model.predict(document, document["entities"])
        document["association"] = [{"patient_id": "O1", "phenotype": ["HP:9999999"]}]
        second = model.predict(document, document["entities"])
        self.assertEqual(first, second)
        self.assertEqual(first, source["association"])
        self.assertEqual(document["entities"], source["entities"])
        self.assertEqual(document["full_text"], source["full_text"])

    def test_training_predictions_must_align(self):
        with self.assertRaises(ValueError):
            AssociationModel().fit([make_document()], [])

    def test_requires_fit_and_handles_empty_training(self):
        with self.assertRaises(RuntimeError):
            AssociationModel().predict(make_document(), [])
        document = make_document()
        model = AssociationModel().fit([])
        self.assertTrue(all(not row["phenotype"] for row in model.predict(document, document["entities"])))


if __name__ == "__main__":
    unittest.main()
