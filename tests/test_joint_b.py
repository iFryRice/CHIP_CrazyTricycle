"""Check blind partition coverage and the B promotion boundary."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from patientphex.data import digest_file
from scripts.run_joint_b import balanced_shards, check_confirmation


class JointBTests(unittest.TestCase):
    def test_balancing_preserves_documents_including_empty_tasks(self):
        tasks = {"c": [0]*5, "a": [0]*7, "b": [0]*2, "e": [], "d": []}
        shards = balanced_shards(tasks)
        self.assertEqual(sorted(sum(shards, [])), sorted(tasks))
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(shards, balanced_shards(dict(reversed(list(tasks.items())))))
        self.assertEqual([sum(len(tasks[i]) for i in shard) for shard in shards], [7, 7])

    def test_confirmation_rejects_failed_missing_or_changed_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text("{}", encoding="utf-8")
            fixed = {"threshold": 0.5}
            plan = {"confirmation_plan": "plan.json", "confirmation_summary": "summary.json", "fixed": fixed}
            names = ["remaining_mean_gain", "full_score_gain", "nonworse_folds", "component_guard",
                     "mention_improved", "format_guard", "positive_bootstrap_lower_bound"]
            result = {"plan_sha256": digest_file(root / "plan.json"), "selected": "joint_all_reviewed",
                      "promoted": True, "checks": dict.fromkeys(names, True)}
            with patch("scripts.run_joint_b.ROOT", root), patch("scripts.run_joint_b.load_plan", return_value={"fixed": fixed}):
                (root / "summary.json").write_text(json.dumps(result))
                self.assertEqual(check_confirmation(plan), {"fixed": fixed})
                for checks in [{}, {**result["checks"], names[0]: False}, {**result["checks"], names[0]: 1}]:
                    (root / "summary.json").write_text(json.dumps({**result, "checks": checks}))
                    with self.assertRaises(ValueError):
                        check_confirmation(plan)
                (root / "summary.json").write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    check_confirmation({**plan, "fixed": {"threshold": 0.4}})


if __name__ == "__main__":
    unittest.main()
