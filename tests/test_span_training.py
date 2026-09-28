"""Training invariants without network access or model downloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train_span_ner import (
    SpanCollator, build_model, candidate_mask, class_statistics, decode_batch,
    exclude_unencoded_gold, load_training_documents, prepare_output, score_predictions,
    select_fit_all, select_fold, span_loss,
)


def sample_window():
    return {
        "pmc_id": "a", "input_ids": [1, 2, 3, 4, 0],
        "attention_mask": [1, 1, 1, 1, 0], "special_tokens_mask": [1, 0, 0, 1, 1],
        "offset_mapping": [None, [100, 103], [104, 107], None, None],
        "span_labels": [
            {"start_token": 1, "end_token": 1, "labels": [1, 1]},
            {"start_token": 1, "end_token": 2, "labels": [1, 0]},
        ],
    }


class SpanTrainingDataTests(unittest.TestCase):
    def test_fit_all_uses_all_official_labels_without_opening_a_split(self):
        root = Path(__file__).resolve().parents[1]
        training, validation, split_hash = load_training_documents(
            root / "PatientPheX-V1-A/PatientPheX-train.jsonl", root / "nonexistent-validation-split.json",
            fold=999, fit_all=True,
        )
        self.assertEqual(len(training), 80)
        self.assertTrue(all(document["entities"] for document in training))
        self.assertEqual(validation, [])
        self.assertIsNone(split_hash)

    def test_fit_all_rejects_empty_or_missing_labels_and_incomplete_cohort(self):
        labeled = [{"pmc_id": str(index), "entities": [{"text": "example"}]} for index in range(80)]
        self.assertEqual(len(select_fit_all(labeled)), 80)
        for unlabeled in ({"pmc_id": "0", "entities": []}, {"pmc_id": "0"}):
            with self.assertRaisesRegex(ValueError, "nonempty entity labels"):
                select_fit_all([unlabeled, *labeled[1:]])
        with self.assertRaisesRegex(ValueError, "exactly the 80"):
            select_fit_all(labeled[:79])
        with self.assertRaisesRegex(ValueError, "unique"):
            select_fit_all([labeled[0], *labeled[:-1]])

    def test_fit_all_rejects_labeled_data_with_a_different_fingerprint(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fabricated_train.jsonl"
            documents = [{"pmc_id": str(index), "entities": [{"text": "example"}]} for index in range(80)]
            path.write_text("\n".join(json.dumps(document) for document in documents), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                load_training_documents(path, Path(temporary) / "missing_split.json", fold=0, fit_all=True)

    def test_saved_folds_must_partition_input_without_overlap(self):
        documents = [{"pmc_id": item} for item in ["a", "b", "c"]]
        train, validation = select_fold(documents, {"folds": [["a"], ["b", "c"]]}, 0)
        self.assertEqual([doc["pmc_id"] for doc in train], ["b", "c"])
        self.assertEqual([doc["pmc_id"] for doc in validation], ["a"])
        for split in [{"folds": [["a"], ["a", "b", "c"]]}, {"folds": [["a"], ["b"]]}]:
            with self.assertRaises(ValueError):
                select_fold(documents, split, 0)

    def test_existing_output_is_never_silently_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "run"
            prepare_output(target, resume=False)
            with self.assertRaises(FileExistsError):
                prepare_output(target, resume=False)
            with self.assertRaises(ValueError):
                prepare_output(target, resume=True)

    def test_unencoded_gold_ignores_only_intersecting_negative_label(self):
        window = sample_window()
        audit = {"unencoded_entities": [{"pmc_id": "a", "offset": 105, "length": 1,
                                         "note": None, "reasons": ["exact_alignment_failed"]}]}
        exclude_unencoded_gold([window], audit, width=4)
        self.assertEqual(window["ignored_candidate_labels"], [(2, 0, 0)])
        stats = class_statistics([window], 4, cap=1000)
        self.assertEqual(stats["candidate_spans"], 3)
        self.assertEqual(stats["supervised_candidates_per_label"], [2, 3])
        self.assertEqual(stats["positive_counts"], [2, 1])

    def test_corrupt_annotations_require_repair(self):
        with self.assertRaisesRegex(ValueError, "requires repair"):
            exclude_unencoded_gold([sample_window()], {"unencoded_entities": [
                {"pmc_id": "a", "reasons": ["text_mismatch"]}]}, 4)

    def test_recall_includes_gold_not_representable_by_tokenizer(self):
        documents = [{"pmc_id": "a", "entities": [
            {"offset": 100, "length": 3, "note": None},
            {"offset": 105, "length": 1, "note": "NO"},
        ]}]
        metrics = score_predictions({("a", 100, 3, 0): 0.9}, documents)
        self.assertEqual(metrics["NO"]["fn"], 1)
        self.assertEqual(metrics["micro"]["recall"], 0.5)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch training dependency is not installed")
class SpanTrainingTensorTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Optional transformers training dependency is not installed")
    def test_fit_all_run_never_evaluates_or_selects_a_best_checkpoint(self):
        import json
        import torch
        from unittest.mock import patch

        from scripts.train_span_ner import parser, run

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 12)
                self.config = SimpleNamespace(
                    hidden_size=12, max_position_embeddings=512,
                    save_pretrained=lambda path: Path(path).mkdir(parents=True),
                )

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        tokenizer = SimpleNamespace(pad_token_id=0, save_pretrained=lambda path: Path(path).mkdir(parents=True))
        audit = {"counts": {"unencoded_entities": 0}, "unencoded_entities": []}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model_path, output = folder / "encoder", folder / "run"
            model_path.mkdir()
            args = parser().parse_args([
                "--model-path", str(model_path), "--output-dir", str(output),
                "--fit-all", "--epochs", "1", "--device", "cpu", "--no-fp16", "--threads", "1",
                "--split", str(folder / "nonexistent_split.json"),
            ])
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                    patch("transformers.AutoModel.from_pretrained", return_value=TinyEncoder()), \
                    patch("scripts.train_span_ner.build_span_windows", return_value=([sample_window()], audit)) as window_builder, \
                    patch("scripts.train_span_ner.evaluate", side_effect=AssertionError("fit-all must never evaluate")) as evaluator:
                summary = run(args)
            self.assertEqual(window_builder.call_count, 1)
            self.assertEqual(len(window_builder.call_args.args[0]), 80)
            evaluator.assert_not_called()
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["fit_all"])
            self.assertEqual(len(manifest["training_document_ids"]), 80)
            self.assertEqual(manifest["validation_document_ids"], [])
            self.assertIsNone(manifest["sha256"]["split"])
            self.assertIsNone(summary["metrics"])
            self.assertIsNone(summary["last_metrics"])
            self.assertNotIn("best_micro_f1", summary)
            self.assertIn("fit_all_scope", summary)
            self.assertEqual(sorted(path.name for path in output.glob("*.pt")), ["final.pt", "last.pt"])
            self.assertFalse((output / "validation_predictions.jsonl").exists())
            last = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertIn("model_state_dict", last)
            self.assertIsNone(last["metrics"])
            self.assertIsNone(last["best_score"])

    def test_masks_reject_interior_specials_and_out_of_bounds(self):
        import torch

        mask = candidate_mask(torch.tensor([[True, False, True, True]]), 6)
        self.assertFalse(mask[0, 0, 2])
        self.assertTrue(mask[0, 2, 1])
        self.assertFalse(mask[0, 3, 1])
        self.assertFalse(mask[:, :, 4:].any())

    def test_overlapping_dual_labels_and_ignored_loss_gradients(self):
        import torch

        window = sample_window()
        window["ignored_candidate_labels"] = [(2, 0, 0)]
        batch = SpanCollator(0, 4)([window])
        self.assertEqual(batch["targets"][0, 1, 0].tolist(), [1, 1])
        self.assertEqual(batch["targets"][0, 1, 1].tolist(), [1, 0])
        logits = torch.zeros_like(batch["targets"], requires_grad=True)
        loss = span_loss(logits, batch["targets"], batch["loss_mask"], torch.ones(2), torch.ones(2))
        loss.backward()
        self.assertEqual(logits.grad[0, 2, 0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 2, 0, 1].item(), 0)
        self.assertLess(logits.grad[0, 1, 1, 0].item(), 0)
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)

    def test_inference_requires_no_gold_and_deduplicates_windows(self):
        import torch

        window = sample_window()
        del window["span_labels"]
        batch = SpanCollator(0, 4, include_labels=False)([window, window])
        self.assertIsNone(batch["targets"])
        probabilities = torch.zeros(2, 5, 4, 2)
        probabilities[0, 1, 1, 0] = 0.6
        probabilities[1, 1, 1, 0] = 0.9
        probabilities[:, 0, :, :] = 1.0
        predictions = {}
        decode_batch(probabilities, batch["mask"], batch["windows"], predictions, 0.5)
        self.assertEqual(set(predictions), {("a", 100, 7, 0)})
        self.assertAlmostEqual(predictions[("a", 100, 7, 0)], 0.9)

    def test_low_rank_model_backpropagates_for_long_width_and_short_sequence(self):
        import torch

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=12)
                self.embedding = torch.nn.Embedding(8, 12)

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        model = build_model(TinyEncoder(), max_span_width=8, rank=4, dropout=0)
        batch = SpanCollator(0, 8)([sample_window()])
        logits = model(**batch["inputs"])
        self.assertEqual(tuple(logits.shape), (1, 5, 8, 2))
        loss = span_loss(logits, batch["targets"], batch["loss_mask"], torch.ones(2), torch.ones(2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.encoder.embedding.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
