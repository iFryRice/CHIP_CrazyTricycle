"""Prompt provenance, exact text and original global offsets are mandatory."""

import copy
import unittest

from patientphex.llm_extraction import DemonstrationRetriever, budget_prompt, gold_phrases, parse_quotes, source_chunks


def make_document(identifier, text, entities=None):
    return {"pmc_id": identifier, "full_text": [{"offset": 100, "text": text, "section_type": "CASE", "type": "paragraph"}], "entities": entities or []}


class LlmExtractionTests(unittest.TestCase):
    def test_budget_drops_examples_but_never_truncates_source(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "|".join(item["content"] for item in messages)

            def __call__(self, text, **kwargs):
                return {"input_ids": list(text)}

        chunk = {"text": "Exact source phrase."}
        examples = [{"text": "x" * 1000, "phrases": [], "chunk_id": "b:0:1000", "pmc_id": "b"}]
        prepared = budget_prompt(Tokenizer(), chunk, examples, 1200, 20)
        self.assertEqual(prepared["dropped_demonstrations"], 1)
        self.assertEqual(prepared["messages"][-1]["content"], "Paragraph:\nExact source phrase.")
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            budget_prompt(Tokenizer(), chunk, [], 20, 10)

    def test_overlapping_chunks_are_exact_source_slices(self):
        text = "Café " + "short stature and seizures. " * 20
        document = make_document("a", text)
        before = copy.deepcopy(document)
        chunks = source_chunks(document, 90, 25)
        coverage = set()
        for chunk in chunks:
            relative = chunk["offset"] - 100
            self.assertEqual(chunk["text"], text[relative:relative + len(chunk["text"])] )
            coverage.update(range(relative, relative + len(chunk["text"])))
        self.assertEqual(coverage, set(range(len(text))))
        self.assertEqual(document, before)

    def test_demonstrations_cannot_use_held_out_or_the_query_document(self):
        docs = [make_document("a", "short stature was present"), make_document("b", "seizures were observed"), make_document("c", "hearing loss was diagnosed")]
        with self.assertRaisesRegex(ValueError, "forbidden"):
            DemonstrationRetriever().fit(docs, {"b"})
        retriever = DemonstrationRetriever().fit(docs, {"heldout"})
        chosen = retriever.retrieve(source_chunks(docs[0])[0])
        self.assertEqual({example["pmc_id"] for example in chosen}, {"b", "c"})

    def test_parser_rejects_normalized_or_invented_text_and_internal_substrings(self):
        chunk = {"offset": 50, "text": "Seizures occurred, then Seizures recurred. No losses."}
        spans, rejected = parse_quotes('["Seizures", "seizures", "hearing loss", "loss"]', chunk)
        self.assertEqual([span["offset"] for span in spans], [50, 74])
        self.assertEqual(len(rejected), 3)
        with self.assertRaises(ValueError):
            parse_quotes('{"mentions": ["Seizures"]}', chunk)

    def test_original_training_annotations_are_checked_before_prompting(self):
        document = make_document("a", "Short stature.", [{"offset": 100, "length": 13, "text": "short stature"}])
        with self.assertRaisesRegex(ValueError, "source text"):
            gold_phrases(document, source_chunks(document)[0])


if __name__ == "__main__":
    unittest.main()
