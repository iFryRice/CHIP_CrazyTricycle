import copy
import unittest

from patientphex.relation_veto import filter_existing


class RelationVetoTests(unittest.TestCase):
    def setUp(self):
        self.base = {"pmc_id": "1", "pmid": "2", "entities": [{"text": "Unmapped Case", "identifier": "-1"}],
                     "association": [{"patient_id": "p1", "phenotype": ["HP:0001250", "Unmapped Case", "HP:0001;HP:0002"]},
                                     {"patient_id": "p2", "phenotype": []}]}

    def test_never_adds_and_preserves_raw_text_and_compounds(self):
        candidate = copy.deepcopy(self.base)
        candidate["association"][0]["phenotype"] = ["New Text", "Unmapped Case", "HP:0001;HP:0002"]
        candidate["association"][1]["phenotype"] = ["HP:0001250"]
        actual = filter_existing(self.base, candidate)
        self.assertEqual(actual["association"][0]["phenotype"], ["Unmapped Case", "HP:0001;HP:0002"])
        self.assertEqual(actual["association"][1]["phenotype"], [])
        self.assertEqual(self.base["association"][0]["phenotype"][0], "HP:0001250")
        self.assertEqual(actual["entities"], self.base["entities"])

    def test_rejects_changed_entities_or_missing_patients(self):
        candidate = copy.deepcopy(self.base)
        candidate["entities"] = []
        with self.assertRaises(ValueError): filter_existing(self.base, candidate)
        candidate = copy.deepcopy(self.base)
        candidate["association"].pop()
        with self.assertRaises(ValueError): filter_existing(self.base, candidate)


if __name__ == "__main__":
    unittest.main()
