"""BIO projection and window merging must preserve the original evaluation space."""

import copy
import unittest

from patientphex.bio_data import add_window_probabilities, bio_targets, decode_probabilities


class BioDataTests(unittest.TestCase):
    def test_overlapping_gold_keeps_longest_and_records_omission(self):
        document = {"entities": [{"offset": 10, "length": 8}, {"offset": 10, "length": 4}]}
        window = {"offset_mapping": [None, [10, 14], [15, 18], None], "span_labels": [
            {"start_token": 1, "end_token": 2, "offset": 10, "length": 8},
            {"start_token": 1, "end_token": 1, "offset": 10, "length": 4}]}
        original = copy.deepcopy((document, window))
        labels, audit = bio_targets(window, document)
        self.assertEqual(labels, [-100, 1, 2, -100])
        self.assertEqual(audit["omitted_overlaps"], [(10, 4)])
        self.assertEqual((document, window), original)

    def test_clipped_gold_is_ignored_instead_of_labeled_as_background(self):
        document = {"entities": [{"offset": 5, "length": 12}]}
        window = {"offset_mapping": [None, [10, 14], [15, 17], [18, 22], None], "span_labels": []}
        labels, _ = bio_targets(window, document)
        self.assertEqual(labels, [-100, -100, -100, 0, -100])

    def test_overflow_probabilities_are_averaged_before_global_decoding(self):
        document = {"pmc_id": "x", "full_text": [{"offset": 10, "text": "short stature."}]}
        window = {"pmc_id": "x", "paragraph_index": 0, "offset_mapping": [None, [10, 15], [16, 23], [23, 24], None]}
        probs = [[1, 0, 0], [0.05, 0.9, 0.05], [0.05, 0.1, 0.85], [0.9, 0.05, 0.05], [1, 0, 0]]
        accumulator = {}
        add_window_probabilities(accumulator, window, probs)
        add_window_probabilities(accumulator, window, probs)
        result = decode_probabilities([document], accumulator)
        self.assertEqual(len(result[0]["spans"]), 1)
        self.assertEqual(result[0]["spans"][0]["text"], "short stature")
        self.assertEqual(result[0]["spans"][0]["offset"], 10)

    def test_invalid_probabilities_fail_and_orphan_i_is_repaired(self):
        window = {"pmc_id": "x", "paragraph_index": 0, "offset_mapping": [[0, 7]]}
        with self.assertRaises(ValueError):
            add_window_probabilities({}, window, [[0, 0, 0]])
        accumulator = {}
        add_window_probabilities(accumulator, window, [[0.1, 0.1, 0.8]])
        result = decode_probabilities([{"pmc_id": "x", "full_text": [{"offset": 0, "text": "seizure"}]}], accumulator)
        self.assertEqual(result[0]["spans"][0]["text"], "seizure")


if __name__ == "__main__":
    unittest.main()
