"""Guard final B predictions against incomplete training and mixed configurations."""

from copy import deepcopy
import unittest

from scripts.finalize_b_pipeline import validate_final_model


class FinalModelGuardTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"parameters": {"epochs": 5, "pos_weight_cap": 100},
                         "sha256": {"model_files": {"vocab.txt": "expected"}}}
        self.summary = {"status": "completed", "fit_all": True, "epochs_completed": 5,
                        "last_epoch_complete": True, "last_metrics": None, "best_micro_f1": None}
        self.inference = {"checkpoint_path": "/project/run/last.pt", "checkpoint_epoch": 5}
        self.selection = {"ner_training_parameters": {"epochs": 5, "pos_weight_cap": 100},
                          "model_files_sha256": {"vocab.txt": "expected"}}

    def test_completed_matching_fit_is_accepted(self):
        validate_final_model(self.manifest, self.summary, self.inference, self.selection)

    def test_interrupted_or_partial_fit_is_rejected(self):
        for change in ({"status": "failed"}, {"epochs_completed": 1}, {"last_epoch_complete": False}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "not completed"):
                validate_final_model(self.manifest, {**self.summary, **change}, self.inference, self.selection)

    def test_old_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "completed last.pt"):
            validate_final_model(self.manifest, self.summary, {**self.inference, "checkpoint_epoch": 1}, self.selection)

    def test_selected_weighting_and_encoder_are_bound(self):
        altered = deepcopy(self.manifest)
        altered["parameters"]["pos_weight_cap"] = 1000
        with self.assertRaisesRegex(ValueError, "pos_weight_cap"):
            validate_final_model(altered, self.summary, self.inference, self.selection)
        altered = deepcopy(self.manifest)
        altered["sha256"]["model_files"]["vocab.txt"] = "different"
        with self.assertRaisesRegex(ValueError, "encoder/tokenizer"):
            validate_final_model(altered, self.summary, self.inference, self.selection)


if __name__ == "__main__":
    unittest.main()
