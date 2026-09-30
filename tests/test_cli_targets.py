"""Protect explicit B targeting while retaining the historical A defaults."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from patientphex.__main__ import check_target, target_paths, write_report


class TargetSelectionTests(unittest.TestCase):
    def test_a_and_b_defaults_keep_supervision_separate(self):
        args = Namespace(data_dir="PatientPheX-V1-A")
        stage, target, report, output = target_paths(args)
        self.assertEqual(stage, "A")
        self.assertEqual(target, Path("PatientPheX-V1-A/PatientPheX-A.jsonl"))
        self.assertEqual(output, Path("submissions/patientphex_a.jsonl"))
        args.target_set = "B"
        stage, target, report, output = target_paths(args)
        self.assertEqual(stage, "B")
        self.assertEqual(target, Path("PatientPheX-V1-B/PatientPheX-V1-B.jsonl"))
        self.assertEqual(report, Path("reports/cpu_b_control"))
        self.assertEqual(output, Path("submissions/patientphex_b_cpu.jsonl"))
        args.target_file = "explicit/input.jsonl"
        self.assertEqual(target_paths(args)[1], Path("explicit/input.jsonl"))

    def test_b_rejects_a_cohort_labels_and_training_overlap(self):
        targets = [{"pmc_id": str(index), "patient": [{}] * (3 if index < 44 else 2),
                    "entities": [], "association": []} for index in range(100)]
        check_target([{"pmc_id": "train"}], targets, "B")
        with self.assertRaises(ValueError):
            check_target([], targets[:20], "B")
        with self.assertRaises(ValueError):
            check_target([{"pmc_id": "0"}], targets, "B")
        targets[0]["association"] = [{"patient_id": "P1", "phenotype": []}]
        with self.assertRaises(ValueError):
            check_target([], targets, "B")

    def test_b_report_does_not_label_development_score_as_a_or_b_leaderboard(self):
        score = {name: {"precision": 0.5, "recall": 0.5, "f1": 0.5}
                 for name in ("mention", "document", "association_micro", "association_macro")}
        score["score"] = 0.5
        metrics = {"selected_metrics": score, "target_set": "B", "selected_configuration": "dictionary+nearest",
                   "training_documents": 80, "target_documents": 100, "target_patients": 244,
                   "fold_count": 5, "comparisons": {}, "predicted_entities": 100,
                   "predicted_associations": 100, "submission_bytes": 10000,
                   "target_path": "PatientPheX-V1-B/PatientPheX-V1-B.jsonl",
                   "output_path": "submissions/patientphex_b_cpu.jsonl", "report_dir": "reports/cpu_b_control"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, metrics)
            text = path.read_text(encoding="utf-8")
        self.assertIn("B 榜文献 100 篇", text)
        self.assertIn("不能视为独立测试或 B 榜成绩", text)
        self.assertIn("--target-set B", text)
        self.assertNotIn("A 榜文献", text)


if __name__ == "__main__":
    unittest.main()
