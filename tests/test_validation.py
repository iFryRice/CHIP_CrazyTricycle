"""Submission corruption checks independent of the prediction implementation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from patientphex.validation import validate_submission


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "submission.jsonl"
        self.ontology = SimpleNamespace(allowed_ids={"HP:0000001", "HP:0000002"})
        self.source = {
            "pmc_id": "1", "pmid": None,
            "patient": [{"patient_id": "P1", "mention": []}],
            "full_text": [{"section_type": "CASE", "type": "paragraph", "offset": 10,
                           "text": "rash and pain\u2028\U0001f600"}],
        }
        self.prediction = {
            "pmc_id": "1", "pmid": None,
            "entities": [{"identifier": "HP:0000001", "type": "Phenotype", "offset": 10,
                          "length": 4, "text": "rash", "note": None}],
            "association": [{"patient_id": "P1", "phenotype": ["HP:0000001"]}],
        }

    def write(self, records=None):
        records = [self.prediction] if records is None else records
        self.path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    def validate(self):
        return validate_submission(self.path, [self.source], self.ontology)

    def test_valid_and_exact_nonascii_source_span(self):
        self.prediction["entities"].append({"identifier": "-1", "type": "Phenotype", "offset": 23,
                                            "length": 2, "text": "\u2028\U0001f600", "note": None})
        self.write()
        report = self.validate()
        self.assertTrue(report["valid"])
        self.assertEqual((report["documents"], report["patients"], report["entities"]), (1, 1, 2))
        self.assertEqual(len(report["sha256"]), 64)

    def test_wrong_span_text_length_and_bool_offset_rejected(self):
        for field, value in (("offset", 11), ("offset", True), ("length", 5), ("text", "Rash"), ("note", "D")):
            with self.subTest(field=field, value=value):
                original = copy.deepcopy(self.prediction)
                self.prediction["entities"][0][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.validate()
                self.prediction = original

    def test_nonbranch_and_duplicate_compounds_rejected(self):
        for identifier in ("HP:9999999", "HP:0000001;invalid", "HP:0000001;HP:0000001"):
            self.prediction["entities"][0]["identifier"] = identifier
            self.write()
            with self.assertRaises(ValueError):
                self.validate()
        self.prediction["entities"][0]["identifier"] = "HP:0000001;HP:0000002"
        self.prediction["association"][0]["phenotype"] = ["HP:0000001;HP:0000002"]
        self.write()
        self.assertEqual(self.validate()["association_pairs"], 2)
        self.prediction["association"][0]["phenotype"].append("HP:0000001")
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate phenotype"):
            self.validate()

    def test_mixed_mapped_and_unmapped_entity_matches_training_contract(self):
        self.prediction["entities"][0]["identifier"] = "HP:0000001;-1"
        self.prediction["association"][0]["phenotype"] = ["HP:0000001", "rash"]
        self.write()
        result = self.validate()
        self.assertEqual(result["unmapped_entities"], 1)
        self.assertEqual(result["compound_entities"], 1)
        self.assertEqual(result["association_pairs"], 2)
        self.prediction["association"][0]["phenotype"] = ["HP:0000001;-1"]
        self.write()
        with self.assertRaisesRegex(ValueError, "invalid/nonbranch"):
            self.validate()

    def test_document_patient_coverage_and_schema(self):
        self.write([self.prediction, self.prediction])
        with self.assertRaisesRegex(ValueError, "duplicate document"):
            self.validate()
        for field, value in (("pmc_id", "2"), ("pmid", "wrong"), ("association", []), ("patient", [])):
            original = copy.deepcopy(self.prediction)
            self.prediction[field] = value
            self.write()
            with self.assertRaises(ValueError):
                self.validate()
            self.prediction = original
        self.prediction["association"].append({"patient_id": "P1", "phenotype": []})
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate patient"):
            self.validate()

    def test_raw_source_text_is_allowed_without_entity_but_no_fabrication(self):
        self.prediction["association"][0]["phenotype"] = ["pain"]
        self.write()
        self.assertTrue(self.validate()["valid"])
        self.prediction["association"][0]["phenotype"] = ["invented symptom"]
        self.write()
        with self.assertRaisesRegex(ValueError, "not source text"):
            self.validate()

    def test_encoding_blank_lines_duplicate_keys_and_placeholder(self):
        self.write()
        payload = self.path.read_bytes()
        for corrupted in (b"\xef\xbb\xbf" + payload, b"\xff", payload + b"\n", b'{"pmc_id":"1","pmc_id":"1"}\n'):
            self.path.write_bytes(corrupted)
            with self.assertRaises(ValueError):
                self.validate()
        self.prediction["entities"] = []
        self.prediction["association"][0]["phenotype"] = []
        self.write()
        with self.assertRaisesRegex(ValueError, "all-empty"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
