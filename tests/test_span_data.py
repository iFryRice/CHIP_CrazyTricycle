"""Span-window contracts using deterministic tokenization with no downloads."""

import copy
import re
import unittest

from patientphex.span_data import build_span_windows


class FakeFastTokenizer:
    """Whitespace words with two special tokens and real overflow behavior."""

    is_fast = True

    def __init__(self):
        self.calls = []

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        capacity = kwargs["max_length"] - 2
        stride = kwargs["stride"]
        tokens = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        result = {key: [] for key in (
            "input_ids", "attention_mask", "offset_mapping", "special_tokens_mask", "token_type_ids",
        )}
        start = 0
        while True:
            chunk = tokens[start:start + capacity]
            size = len(chunk) + 2
            result["input_ids"].append([101] + list(range(start + 1000, start + 1000 + len(chunk))) + [102])
            result["attention_mask"].append([1] * size)
            result["offset_mapping"].append([(0, 0)] + chunk + [(0, 0)])
            result["special_tokens_mask"].append([1] + [0] * len(chunk) + [1])
            result["token_type_ids"].append([0] * size)
            if start + capacity >= len(tokens):
                break
            start += capacity - stride
        return result


def entity(text, offset, identifier="HP:0001250", note=None):
    return {"offset": offset, "length": len(text), "text": text, "identifier": identifier, "note": note}


def document(text, entities=None, offset=100, pmc_id="test"):
    return {
        "pmc_id": pmc_id,
        "full_text": [{"offset": offset, "text": text, "section_type": "CASE", "type": "paragraph"}],
        "entities": entities or [],
    }


class SpanDataTests(unittest.TestCase):
    def test_overflow_exact_global_offsets_and_no_source_mutation(self):
        text = "alpha beta gamma delta epsilon zeta"
        gold = [entity("gamma delta", 111)]
        source = [document(text, gold)]
        original = copy.deepcopy(source)
        tokenizer = FakeFastTokenizer()
        windows, audit = build_span_windows(source, tokenizer, max_length=6, stride=2)
        self.assertEqual(source, original)
        self.assertEqual(len(windows), 2)
        self.assertEqual([len(w["span_labels"]) for w in windows], [1, 1])
        self.assertEqual([(w["span_labels"][0]["start_token"], w["span_labels"][0]["end_token"])
                          for w in windows], [(3, 4), (1, 2)])
        for window in windows:
            self.assertIsNone(window["offset_mapping"][0])
            self.assertIsNone(window["offset_mapping"][-1])
            span = window["span_labels"][0]
            start = window["offset_mapping"][span["start_token"]][0]
            end = window["offset_mapping"][span["end_token"]][1]
            self.assertEqual(text[start - 100:end - 100], "gamma delta")
        self.assertEqual(audit["counts"]["encoded_entities"], 1)
        self.assertEqual(audit["counts"]["entity_window_assignments"], 2)
        self.assertEqual(audit["unencoded_entities"], [])
        self.assertTrue(tokenizer.calls[0][1]["return_overflowing_tokens"])
        self.assertTrue(tokenizer.calls[0][1]["return_offsets_mapping"])

    def test_crossing_nested_and_same_span_targets_preserve_raw_metadata(self):
        gold = [
            entity("red blue", 100, "HP:0001250;HP:0004322"),
            entity("blue green", 104, "-1", "NO"),
            entity("blue", 104, "N", "D"),
            entity("blue", 104, "HP:0001250", "NO"),
            entity("blue", 104, "N", "D"),
        ]
        windows, audit = build_span_windows([document("red blue green", gold)], FakeFastTokenizer())
        labels = windows[0]["span_labels"]
        self.assertEqual([(s["start_token"], s["end_token"]) for s in labels], [(1, 2), (2, 2), (2, 3)])
        self.assertEqual(labels[1]["labels"], [1, 1])
        self.assertEqual(labels[1]["identifiers"], ["N", "HP:0001250"])
        self.assertEqual([a["entity_index"] for a in labels[1]["annotations"]], [2, 3, 4])
        self.assertEqual(labels[0]["identifiers"], ["HP:0001250;HP:0004322"])
        self.assertEqual(labels[2]["identifiers"], ["-1"])
        self.assertEqual(labels[2]["labels"], [0, 1])
        self.assertEqual(audit["counts"]["encoded_entities"], 5)
        self.assertEqual(audit["counts"]["span_window_targets"], 3)
        self.assertEqual(audit["counts"]["positive_entities"], 3)

    def test_exact_alignment_failure_is_not_silently_expanded(self):
        windows, audit = build_span_windows(
            [document("headache", [entity("head", 100)])], FakeFastTokenizer(),
        )
        self.assertEqual(windows[0]["span_labels"], [])
        self.assertEqual(audit["issue_counts"]["exact_alignment_failed"], 1)
        issue = audit["issues"]["exact_alignment_failed"][0]
        self.assertTrue(issue["start_aligned"])
        self.assertFalse(issue["end_aligned"])
        self.assertEqual(audit["unencoded_entities"][0]["entity_index"], 0)

    def test_overlong_span_and_uncovered_window_are_audited_separately(self):
        text = "aa bb cc dd ee"
        gold = [entity("aa bb cc", 100), entity("bb cc", 103)]
        windows, audit = build_span_windows(
            [document(text, gold)], FakeFastTokenizer(), max_length=4, stride=0, max_span_width=2,
        )
        self.assertFalse(any(window["span_labels"] for window in windows))
        self.assertEqual(audit["issue_counts"]["span_too_long"], 1)
        self.assertEqual(audit["issue_counts"]["no_window_coverage"], 2)
        self.assertEqual(audit["issue_counts"]["exact_alignment_failed"], 0)
        self.assertEqual(audit["unencoded_entities"][1]["reasons"], ["no_window_coverage"])

    def test_overlong_span_with_full_window_coverage(self):
        windows, audit = build_span_windows(
            [document("aa bb cc", [entity("aa bb cc", 100)])], FakeFastTokenizer(), max_span_width=2,
        )
        self.assertEqual(audit["unencoded_entities"][0]["reasons"], ["span_too_long"])
        self.assertEqual(audit["issues"]["span_too_long"][0]["token_width"], 3)
        self.assertEqual(windows[0]["span_labels"], [])

    def test_unicode_and_multiple_paragraph_offsets_do_not_get_concatenated(self):
        source = document("🧬 café", [entity("café", 102)], offset=100)
        source["full_text"].append({"offset": 999, "text": "seizure", "section_type": "TITLE", "type": "front"})
        source["entities"].append(entity("seizure", 999))
        windows, audit = build_span_windows([source], FakeFastTokenizer())
        self.assertEqual(windows[0]["offset_mapping"][1:3], [[100, 101], [102, 106]])
        self.assertEqual(windows[1]["paragraph_index"], 1)
        self.assertEqual(windows[1]["offset_mapping"][1], [999, 1006])
        self.assertEqual(audit["counts"]["encoded_entities"], 2)

    def test_invalid_source_spans_are_audited_without_repair(self):
        invalid = entity("aa", 100)
        invalid["length"] = -1
        gold = [invalid, entity("aa", 999), entity("zz", 100)]
        _, audit = build_span_windows([document("aa bb", gold)], FakeFastTokenizer())
        self.assertEqual(audit["counts"]["unencoded_entities"], 3)
        self.assertEqual(audit["issue_counts"]["invalid_entity_span"], 1)
        self.assertEqual(audit["issue_counts"]["entity_not_in_passage"], 1)
        self.assertEqual(audit["issue_counts"]["text_mismatch"], 1)

    def test_unlabelled_documents_and_empty_paragraphs(self):
        source = [document("", pmc_id="empty"), document("plain words", pmc_id="target")]
        windows, audit = build_span_windows(source, FakeFastTokenizer())
        self.assertEqual(len(windows), 2)
        self.assertEqual(audit["counts"]["empty_windows"], 1)
        self.assertEqual(audit["counts"]["entities"], 0)
        self.assertTrue(all(not window["span_labels"] for window in windows))

    def test_invalid_tokenizer_configuration_and_document_duplicates_fail(self):
        tokenizer = FakeFastTokenizer()
        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            build_span_windows([], tokenizer)
        with self.assertRaisesRegex(ValueError, "stride"):
            build_span_windows([], FakeFastTokenizer(), max_length=4, stride=2)
        with self.assertRaisesRegex(ValueError, "Duplicate document"):
            build_span_windows([document("aa"), document("bb")], FakeFastTokenizer())


if __name__ == "__main__":
    unittest.main()
