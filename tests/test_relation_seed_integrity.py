"""Regression checks for frozen-seed and held-out provenance enforcement."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from patientphex.data import digest_file, write_json
from scripts.run_relation_seed_ensemble import verified_scores


class SeedIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "member/fold0"
        self.work.mkdir(parents=True)
        self.member = {"seed": 7, "work_dir": "member", "fold_plan_sha256": {"0": "frozen-plan"}}
        self.parameters = {"seed": 1, "epochs": 3}
        self.manifest = {"plan_sha256": "frozen-plan", "smoke": False, "fold": 0,
                         "parameters": {"seed": 7, "epochs": 3}, "training_document_ids": ["training"],
                         "validation_document_ids": ["held-out"]}
        write_json(self.work / "manifest.json", self.manifest)
        (self.work / "last.pt").write_bytes(b"frozen checkpoint bytes")
        write_json(self.work / "validation_scores.json", [{"pmc_id": "held-out", "patient_id": "p1", "concept": "HP:0001250", "score": 0.6}])
        self.summary = {"status": "completed", "epochs_completed": 3, "plan_sha256": "frozen-plan",
                        "checkpoint_sha256": digest_file(self.work / "last.pt"),
                        "scores_sha256": digest_file(self.work / "validation_scores.json")}
        write_json(self.work / "summary.json", self.summary)

    def verify(self):
        with patch("scripts.run_relation_seed_ensemble.ROOT", self.root):
            return verified_scores(self.member, 0, {"training": {}, "held-out": {}}, ["held-out"], self.parameters)

    def test_accepts_the_fixed_member(self):
        scores, _ = self.verify()
        self.assertEqual(scores, {("held-out", "p1", "HP:0001250"): 0.6})

    def test_rejects_training_label_leakage(self):
        self.manifest["training_document_ids"].append("held-out")
        write_json(self.work / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "partition"):
            self.verify()

    def test_rejects_undeclared_seed(self):
        self.manifest["parameters"]["seed"] = 99
        write_json(self.work / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "seed"):
            self.verify()

    def test_rejects_modified_probabilities(self):
        rows = json.loads((self.work / "validation_scores.json").read_text())
        rows[0]["score"] = 0.99
        write_json(self.work / "validation_scores.json", rows)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
