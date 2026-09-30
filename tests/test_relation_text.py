import unittest

from patientphex.relation_text import MARKERS, examples, marked_text, supervise


class OntologyStub:
    terms = {"HP:0001250": {"name": "Seizure"}}


class RelationTextTests(unittest.TestCase):
    def setUp(self):
        text = "Patient 1 has seizures. Patient 2 is well."
        self.document = {"pmc_id": "1", "pmid": "2", "full_text": [{"offset": 100, "text": text}],
                         "patient": [{"patient_id": "p1", "mention": [{"offset": 100, "length": 9, "text": "Patient 1"}]},
                                     {"patient_id": "p2", "mention": [{"offset": 124, "length": 9, "text": "Patient 2"}]}],
                         "entities": [], "association": []}
        self.entity = {"offset": 114, "length": 8, "text": "seizures", "identifier": "HP:0001250", "note": None}

    def test_patient_and_entity_roles_preserve_source(self):
        rows = examples(self.document, [self.entity], OntologyStub())
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["task_id"], rows[1]["task_id"])
        text = rows[0]["fragments"][0]["model_text"]
        self.assertIn("[TARGET]Patient 1[/TARGET]", text)
        self.assertIn("[OTHER]Patient 2[/OTHER]", text)
        self.assertIn("[PHENOTYPE]seizures[/PHENOTYPE]", text)
        for marker in MARKERS:
            text = text.replace(marker, "")
        self.assertEqual(text, self.document["full_text"][0]["text"])

    def test_partition_and_blind_input(self):
        rows = examples(self.document, [self.entity], OntologyStub())
        labelled = {**self.document, "association": [{"patient_id": "p1", "phenotype": ["HP:0001250"]},
                                                     {"patient_id": "p2", "phenotype": []}]}
        trained = supervise(rows, [labelled], {"1"}, {"9"})
        self.assertEqual([r["label"] for r in trained], [1, 0])
        self.assertEqual([r["weight"] for r in trained], [1, 0.5])
        with self.assertRaises(ValueError):
            supervise(rows, [labelled], {"1"}, {"1"})
        with self.assertRaises(ValueError):
            examples(labelled, [self.entity], OntologyStub())

    def test_negation_has_no_positive_candidate(self):
        self.assertEqual(examples(self.document, [{**self.entity, "note": "NO"}], OntologyStub()), [])

    def test_bad_source_marker_rejected(self):
        patients = [{"patient_id": "p1", "mention": [{"offset": 100, "length": 5, "text": "wrong"}]}]
        with self.assertRaises(ValueError):
            marked_text(self.document["full_text"][0], patients, "p1")


if __name__ == "__main__":
    unittest.main()
