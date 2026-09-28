"""Metric regressions for clinically relevant edge cases in the published rules."""

import copy
import unittest

from patientphex.evaluation import evaluate


def entity(identifier="HP:0000001", offset=0, text="rash", note=None):
    return {"identifier": identifier, "offset": offset, "length": len(text), "text": text,
            "type": "Phenotype", "note": note}


def document():
    return {
        "pmc_id": "1", "pmid": None,
        "patient": [{"patient_id": "P1", "mention": []}],
        "full_text": [{"section_type": "CASE", "type": "paragraph", "offset": 0,
                       "text": "rash and pain; no fever"}],
        "entities": [entity()],
        "association": [{"patient_id": "P1", "phenotype": ["HP:0000001"]}],
    }


class EvaluationTests(unittest.TestCase):
    def test_identity_and_compound_partial_credit(self):
        gold = document()
        self.assertEqual(evaluate([gold], [copy.deepcopy(gold)])["score"], 1.0)
        gold["entities"][0]["identifier"] = "HP:0000001;HP:0000002"
        gold["association"][0]["phenotype"] = ["HP:0000001;HP:0000002"]
        result = evaluate([gold], [document()])
        for metric in ("mention", "document", "association_micro"):
            self.assertEqual((result[metric]["tp"], result[metric]["fp"], result[metric]["fn"]), (1, 0, 1))
            self.assertAlmostEqual(result[metric]["f1"], 2 / 3)
        self.assertAlmostEqual(result["association_macro"]["f1"], 2 / 3)

    def test_exact_boundary_and_identifier_required(self):
        gold = document()
        for offset, identifier in ((1, "HP:0000001"), (0, "HP:0000002")):
            predicted = document()
            predicted["entities"][0].update(offset=offset, identifier=identifier)
            result = evaluate([gold], [predicted])
            self.assertEqual((result["mention"]["tp"], result["mention"]["fp"], result["mention"]["fn"]), (0, 1, 1))

    def test_negation_false_positive_and_false_negative(self):
        gold = document()
        gold["entities"].append(entity("HP:0000002", 18, "fever", "NO"))
        predicted = copy.deepcopy(gold)
        self.assertEqual(evaluate([gold], [predicted])["mention"]["f1"], 1.0)
        predicted["entities"][-1]["note"] = None
        self.assertEqual(evaluate([gold], [predicted])["mention"]["fp"], 1)
        predicted["entities"][0]["note"] = "NO"
        self.assertEqual(evaluate([gold], [predicted])["mention"]["fn"], 1)

    def test_unmapped_gold_matches_one_unit_by_span(self):
        gold, predicted = document(), document()
        gold["entities"][0]["identifier"] = "-1"
        gold["association"][0]["phenotype"] = ["rash"]
        predicted["association"][0]["phenotype"] = ["rash"]
        predicted["entities"][0]["identifier"] = "HP:0000001;HP:0000002"
        result = evaluate([gold], [predicted])
        self.assertEqual((result["mention"]["tp"], result["mention"]["fp"], result["mention"]["fn"]), (1, 1, 0))
        self.assertEqual(result["association_micro"]["f1"], 1.0)
        predicted["association"][0]["phenotype"] = ["Rash"]
        self.assertEqual(evaluate([gold], [predicted])["association_micro"]["f1"], 0.0)

    def test_macro_patient_weight_and_empty_sets(self):
        gold, predicted = document(), document()
        for doc in (gold, predicted):
            doc["patient"].append({"patient_id": "P2", "mention": []})
            doc["association"].append({"patient_id": "P2", "phenotype": []})
        predicted["association"][0]["phenotype"] = []
        result = evaluate([gold], [predicted])
        self.assertEqual(result["association_micro"]["f1"], 0.0)
        self.assertEqual(result["association_macro"]["f1"], 0.5)
        for doc in (gold, predicted):
            doc["entities"] = []
            doc["association"][0]["phenotype"] = []
        result = evaluate([gold], [predicted])
        self.assertEqual(result["mention"]["f1"], 0.0)
        self.assertEqual(result["document"]["f1"], 0.0)
        self.assertEqual(result["association_micro"]["f1"], 0.0)
        self.assertEqual(result["association_macro"]["f1"], 1.0)

    def test_reference_exclusion_title_and_abstract_retention(self):
        gold = document()
        gold["full_text"].extend([
            {"section_type": "REFERENCES", "type": "ref", "offset": 100, "text": "pain"},
            {"section_type": "ABSTRACT", "type": "abstract", "offset": 200, "text": "fever"},
        ])
        gold["entities"].extend([entity("HP:0000002", 100, "pain"), entity("HP:0000003", 200, "fever")])
        predicted = copy.deepcopy(gold)
        predicted["entities"] = predicted["entities"][:1]
        result = evaluate([gold], [predicted])
        self.assertEqual(result["mention"]["fn"], 1)
        predicted["entities"].append(entity("HP:0000002", 900, "pain"))
        self.assertEqual(evaluate([gold], [predicted])["mention"]["fp"], 1)

    def test_document_and_patient_mismatches_raise(self):
        gold = document()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            evaluate([gold], [])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate([gold], [gold, gold])
        for patient_id in (None, "unknown"):
            predicted = document()
            predicted["association"] = [] if patient_id is None else [{"patient_id": patient_id, "phenotype": []}]
            with self.assertRaisesRegex(ValueError, "patient coverage mismatch"):
                evaluate([gold], [predicted])


if __name__ == "__main__":
    unittest.main()
