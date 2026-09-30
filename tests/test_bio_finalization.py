"""B inference must reject partial, wrong-cohort or wrong-plan BIO weights."""

import copy
import unittest

from scripts.predict_bio_supported_b import require_final_bio


class BioFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [str(index) for index in range(80)]
        self.manifest = {"fit_all": True, "smoke": False, "validation_document_ids": [], "training_document_ids": self.ids, "plan_sha256": "plan"}
        self.summary = {"status": "completed", "epochs_completed": 5, "validation_documents": 0, "plan_sha256": "plan"}
        self.checkpoint = {"epoch": 5, "completed_epoch": True, "ner_family": "bio", "plan_sha256": "plan"}

    def test_complete_final_fit_is_accepted(self):
        require_final_bio(self.manifest, self.summary, self.checkpoint, self.ids, "plan")

    def test_wrong_cohort_smoke_and_partial_checkpoint_are_rejected(self):
        cases = [("manifest", "training_document_ids", self.ids[:-1]), ("manifest", "smoke", True),
                 ("summary", "epochs_completed", 4), ("summary", "validation_documents", 16),
                 ("checkpoint", "completed_epoch", False), ("checkpoint", "plan_sha256", "old")]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                values = {name: copy.deepcopy(getattr(self, name)) for name in ("manifest", "summary", "checkpoint")}
                values[section][key] = value
                with self.assertRaises(ValueError):
                    require_final_bio(values["manifest"], values["summary"], values["checkpoint"], self.ids, "plan")


if __name__ == "__main__":
    unittest.main()
